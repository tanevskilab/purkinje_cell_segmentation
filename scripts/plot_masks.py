"""Plot segmentation masks produced by the pipeline (SOPA_SEGMENT) for visual QC.

For every <sample>_mask.tif it writes three PNGs to --outdir:

  <sample>_overview.png  whole slide: H&E next to H&E with every cell's centroid marked;
                         cells within --min-area/--max-area (default: the large-cell mode
                         of the area histogram, likely Purkinje cells) are highlighted
  <sample>_areas.png     histogram of cell areas (pixels), highlighted range shaded
  <sample>_crops.png     full-resolution crops around randomly chosen highlighted cells,
                         with mask boundaries drawn on the H&E

The H&E comes from the matching <sample>.zarr store written by SOPA_WRITE_ZARR (its
downsampled pyramid level is used for the overview, so the full image is never loaded).
If no store is found, only the mask is plotted.

The mask is read in 1024-row bands (the mask TIFF's tile size), so memory stays small
even for the largest slides.

Usage (inside the pipeline's container, which has matplotlib/zarr/tifffile/skimage):
    singularity exec ghcr.io-sebgoti-sopa-cellpose-gpu-v1.img \\
        python scripts/plot_masks.py --masks results/masks --zarr-dir results/zarr \\
        --outdir results/plots
"""

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from skimage.segmentation import find_boundaries

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BAND_HEIGHT = 1024


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--masks", nargs="+", default=["results/masks"],
        help="Mask TIFFs, or directories containing *_mask.tif files.")
    p.add_argument("--zarr-dir", default="results/zarr",
        help="Directory with the <sample>.zarr stores from SOPA_WRITE_ZARR.")
    p.add_argument("--outdir", default="results/plots", help="Where to write the PNGs.")
    p.add_argument("--overview-level", type=int, default=3,
        help="Pyramid level of the Zarr store used for the overview (level n = 1/2^n size).")
    p.add_argument("--min-area", type=float, default=1000,
        help="Lower area bound (pixels) of the highlighted cells, e.g. Purkinje cells.")
    p.add_argument("--max-area", type=float, default=6000,
        help="Upper area bound (pixels) of the highlighted cells. Larger objects were mostly "
             "artifacts (air bubbles, vessels) on the first slide checked.")
    p.add_argument("--n-crops", type=int, default=9, help="Number of zoomed-in crops.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for picking crop cells.")
    p.add_argument("--crop-size", type=int, default=600, help="Crop width/height in pixels.")
    return p.parse_args()


def find_masks(inputs: list[str]) -> list[Path]:
    masks = []
    for item in map(Path, inputs):
        masks.extend(sorted(item.glob("*_mask.tif")) if item.is_dir() else [item])
    return masks


def open_he_levels(zarr_dir: Path, sample: str) -> dict[int, zarr.Array] | None:
    """Pyramid levels {0: s0, 1: s1, ...} of the store's (c, y, x) image, or None."""
    images = zarr_dir / f"{sample}.zarr" / "images"
    if not images.is_dir():
        log.warning(f"No Zarr store found at {images.parent} -- plotting the mask only.")
        return None
    image_dir = next(d for d in images.iterdir() if d.is_dir())
    return {int(d.name[1:]): zarr.open_array(str(d), mode="r")
            for d in image_dir.iterdir() if d.is_dir() and d.name.startswith("s")}


def to_rgb(chw: np.ndarray) -> np.ndarray:
    return np.moveaxis(np.asarray(chw), 0, -1)


def scan_mask(mask: zarr.Array, factor: int):
    """One pass over the mask in row bands: a strided overview plus per-cell stats.

    Returns (overview, areas, centroid_y, centroid_x), with the per-cell arrays indexed by
    label (index 0 = background).
    """
    height, width = mask.shape
    overview = []
    areas = np.zeros(1, np.int64)
    sum_y = np.zeros(1, np.float64)
    sum_x = np.zeros(1, np.float64)

    for y0 in range(0, height, BAND_HEIGHT):
        band = mask[y0:y0 + BAND_HEIGHT]
        # keep the rows that land on the overview grid (multiples of `factor`)
        overview.append(band[(-y0) % factor::factor, ::factor])

        ys, xs = np.nonzero(band)
        labels = band[ys, xs]
        n = max(len(areas), int(labels.max(initial=0)) + 1)
        areas = np.pad(areas, (0, n - len(areas)))
        sum_y = np.pad(sum_y, (0, n - len(sum_y)))
        sum_x = np.pad(sum_x, (0, n - len(sum_x)))
        areas += np.bincount(labels, minlength=n)
        sum_y += np.bincount(labels, weights=ys + y0, minlength=n)
        sum_x += np.bincount(labels, weights=xs, minlength=n)

    with np.errstate(invalid="ignore", divide="ignore"):
        return np.concatenate(overview), areas, sum_y / areas, sum_x / areas


def pick_crop_centres(candidates, cy, cx, n_crops, crop_size, shape, seed):
    """Centres of randomly chosen candidate cells, skipping cells inside an earlier crop."""
    half = crop_size // 2
    centres = []
    for label in np.random.default_rng(seed).permutation(candidates):
        if len(centres) == n_crops:
            break
        y, x = int(cy[label]), int(cx[label])
        if all(abs(y - py) > half or abs(x - px) > half for py, px, _ in centres):
            centres.append((min(max(y, half), shape[0] - half),
                            min(max(x, half), shape[1] - half), label))
    return centres


def plot_sample(mask_path: Path, args) -> None:
    sample = mask_path.name.removesuffix("_mask.tif")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    log.info(f"[{sample}] reading {mask_path}")

    with tifffile.TiffFile(mask_path) as tif:
        mask = zarr.open_array(tif.aszarr(), mode="r")
        # merged masks (MERGE_MASKS) are named <sample>_merged_mask.tif
        he = open_he_levels(Path(args.zarr_dir), sample.removesuffix("_merged"))
        level = args.overview_level if he and args.overview_level in he else max(he or [0])
        factor = 2 ** level if he else 8

        overview, areas, cy, cx = scan_mask(mask, factor)
        labels = np.flatnonzero(areas[1:]) + 1
        cell_areas = areas[labels]
        n_cells = len(labels)
        selected = labels[(cell_areas >= args.min_area) & (cell_areas <= args.max_area)]
        others = np.setdiff1d(labels, selected)
        sel_desc = f"{len(selected)} cells of {args.min_area:g}-{args.max_area:g} px"
        log.info(f"[{sample}] {n_cells} cells, median area {np.median(cell_areas):.0f} px; "
                 f"{sel_desc}")

        # --- overview: centroids on the downsampled H&E ---------------------------
        # (at overview scale most cells are only a few pixels wide, so a filled mask
        # overlay is invisible -- mark centroids instead)
        aspect = overview.shape[0] / overview.shape[1]
        fig, axes = plt.subplots(1, 2, figsize=(20, 10 * aspect + 1))
        background = (to_rgb(he[level][:])[: overview.shape[0], : overview.shape[1]]
                      if he else overview > 0)
        for ax in axes:
            ax.imshow(background, cmap=None if he else "gray")
            ax.set_axis_off()
        axes[0].set_title("H&E" if he else "Mask")
        axes[1].scatter(cx[others] / factor, cy[others] / factor, s=0.5, c="black",
                        linewidths=0, label=f"other cells ({len(others)})")
        axes[1].scatter(cx[selected] / factor, cy[selected] / factor, s=6, c="lime",
                        edgecolors="black", linewidths=0.3, label=sel_desc)
        axes[1].legend(loc="lower left", markerscale=3, fontsize=10)
        axes[1].set_title(f"Cell centroids ({n_cells} cells)")
        fig.suptitle(f"{sample}  ({mask.shape[0]} x {mask.shape[1]} px, shown at 1/{factor})")
        fig.tight_layout()
        fig.savefig(outdir / f"{sample}_overview.png", dpi=150)
        plt.close(fig)

        # --- area histogram -------------------------------------------------------
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(cell_areas, bins=np.logspace(np.log10(max(cell_areas.min(), 1)),
                                             np.log10(cell_areas.max() + 1), 60))
        ax.axvspan(args.min_area, args.max_area, color="lime", alpha=0.2, label=sel_desc)
        ax.legend()
        ax.set_xscale("log")
        ax.set_xlabel("Cell area (pixels)")
        ax.set_ylabel("Number of cells")
        ax.set_title(f"{sample}: {n_cells} cells, median {np.median(cell_areas):.0f} px")
        fig.tight_layout()
        fig.savefig(outdir / f"{sample}_areas.png", dpi=150)
        plt.close(fig)

        # --- full-resolution crops around randomly chosen highlighted cells --------
        centres = pick_crop_centres(selected, cy, cx, args.n_crops, args.crop_size,
                                    mask.shape, args.seed)
        if centres:
            ncols = min(3, len(centres))
            nrows = int(np.ceil(len(centres) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
            half = args.crop_size // 2
            for ax, (y, x, label) in zip(axes.flat, centres):
                ys, xs = slice(y - half, y + half), slice(x - half, x + half)
                m = mask[ys, xs]
                bounds = find_boundaries(m, mode="inner")
                if he:
                    rgb = to_rgb(he[0][:, ys, xs]).copy()
                    rgb[bounds] = (0, 255, 0)
                    ax.imshow(rgb)
                else:
                    ax.imshow(bounds, cmap="gray")
                ax.set_title(f"y={y}, x={x}  (cell {label}: {areas[label]} px)", fontsize=10)
            for ax in axes.flat:
                ax.set_axis_off()
            fig.suptitle(f"{sample}: {len(centres)} random cells of {args.min_area:g}-"
                         f"{args.max_area:g} px ({args.crop_size} px crops, full resolution)")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(outdir / f"{sample}_crops.png", dpi=120)
            plt.close(fig)

    log.info(f"[{sample}] plots written to {outdir}")


def main() -> None:
    args = parse_args()
    masks = find_masks(args.masks)
    if not masks:
        raise FileNotFoundError(f"No *_mask.tif files found in {args.masks}")
    for mask_path in masks:
        plot_sample(mask_path, args)


if __name__ == "__main__":
    main()
