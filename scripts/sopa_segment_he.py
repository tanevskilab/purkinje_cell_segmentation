"""Tile-based CellposeSAM segmentation of an H&E image via SOPA.

Adapted from ~/Repositories/tspc/bin/sopa_tile_segment.py, which targets
multiplex fluorescence images (separate nuclear/membrane channels picked by
name, e.g. DAPI/CD45). This repo's images are plain 3-channel H&E, so this
version drops the nuclear/membrane channel arguments entirely and always
segments using all 3 channels together (SOPA's RGB path: `channels=None`),
with CellposeSAM's `cpsam` model as the default (matching
segment_image.py's model choice) instead of the fluorescence-oriented
`cyto3` default.

Why tile instead of segment_image.py's whole-image approach
-------------------------------------------------------------
segment_image.py loads the full image with tifffile.imread() and runs
Cellpose.eval() on it in one call. For a multi-gigapixel whole-slide image
(e.g. 49900x64335), that is fatal: the raw array alone is ~9.6 GB, and
Cellpose's internal flow-field/probability buffers on top of it exceed
available memory. This script instead reads the Dask-backed Zarr store
written by sopa_write_zarr.py and segments tile-by-tile via SOPA's
StainingSegmentation, so peak RAM stays proportional to one tile
(--tile-width), not the whole slide.

Usage
-----
    python scripts/sopa_write_zarr.py \\
        --image 260427_CO36_RCLB5_VisiumHD_20x_5_cropped.tif \\
        --output results/slide.zarr

    python scripts/sopa_segment_he.py \\
        --zarr   results/slide.zarr \\
        --output results/slide_masks.tif \\
        --tile-width 4096 --tile-overlap 256 \\
        --cellpose-model cpsam
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import spatialdata
import tifffile
from skimage.draw import polygon as sk_polygon
from spatialdata import SpatialData

import sopa
import sopa.segmentation
import sopa.utils as sopa_utils
from sopa.constants import SopaKeys
from sopa.segmentation.methods import cellpose_patch
from sopa.segmentation import StainingSegmentation, solve_conflicts

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SOPA tile-based CellposeSAM segmentation of an H&E image "
                     "(Zarr-backed SpatialData store -> labelled mask TIFF).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--zarr", required=True,
        help="Path to the Zarr-backed SpatialData store (from sopa_write_zarr.py).")
    p.add_argument("--output", required=True, help="Path for output mask TIFF (e.g. mask.tif)")
    p.add_argument("--image-key", default=None,
        help="Key of the image inside the SpatialData store. "
             "Leave unset to use it automatically if the store has exactly one image.")
    p.add_argument("--tile-width", type=int, default=4096, help="Tile width in pixels")
    p.add_argument("--tile-overlap", type=int, default=256, help="Tile overlap in pixels")
    p.add_argument("--cellpose-model", default="cpsam",
        help="Cellpose built-in model type. Ignored if --pretrained-model is given.")
    p.add_argument("--pretrained-model", default=None,
        help="Path to a custom/fine-tuned Cellpose model file (e.g. one produced by "
             "napari_training.ipynb). Takes precedence over --cellpose-model when set.")
    p.add_argument("--diameter", type=float, default=0.0,
        help="Expected cell diameter in pixels. Set 0 for Cellpose auto-estimation.")
    p.add_argument("--min-area", type=float, default=50.0,
        help="Minimum cell area in pixels^2 to retain")
    p.add_argument("--flow-threshold", type=float, default=0.4,
        help="Cellpose flow_threshold (default: Cellpose's own default, 0.4). "
             "Raise to find more cells (more false positives from poorly-formed "
             "regions); lower to reduce spurious detections.")
    p.add_argument("--cellprob-threshold", type=float, default=0.0,
        help="Cellpose cellprob_threshold (default: Cellpose's own default, 0.0). "
             "Lower to find more cells (more false positives from dim/low-confidence "
             "regions); raise to reduce spurious detections.")
    p.add_argument("--gpu", dest="gpu", action="store_true", default=True,
        help="Use GPU for Cellpose (default). See --no-gpu to disable.")
    p.add_argument("--no-gpu", dest="gpu", action="store_false",
        help="Run Cellpose on CPU instead of GPU.")
    p.add_argument("--cache-dir", default=None,
        help="Directory for SOPA's per-tile segmentation cache. "
             "Defaults to a .sopa_cache/ folder next to the output file.")
    p.add_argument("--recover", action="store_true", default=False,
        help="Resume a previously interrupted segmentation run using cached tiles.")
    p.add_argument("--skip-tissue-roi", action="store_true", default=False,
        help="Skip tissue-ROI detection (sopa.segmentation.tissue) and segment every tile, "
             "including background-only ones. By default the ROI is used to skip tiles that "
             "don't touch tissue, avoiding false-positive cells from noise/artifacts in "
             "empty (slide background) tiles.")
    p.add_argument("--tissue-blur-kernel-size", type=int, default=5,
        help="Median blur kernel size (pixels) for tissue-ROI detection.")
    p.add_argument("--tissue-open-kernel-size", type=int, default=5,
        help="Morphological opening kernel size (pixels) for tissue-ROI detection.")
    p.add_argument("--tissue-close-kernel-size", type=int, default=5,
        help="Morphological closing kernel size (pixels) for tissue-ROI detection.")
    p.add_argument("--tissue-drop-threshold", type=float, default=0.01,
        help="Tissue regions smaller than this fraction of the total image area are dropped.")
    p.add_argument("--gaussian-sigma", type=float, default=0.0,
        help="Gaussian blur sigma applied to each tile before Cellpose (SOPA's "
             "StainingSegmentation default is 1.0). Defaults to 0 (off) here to match "
             "how the model was trained/validated in napari_training.ipynb, which runs "
             "Cellpose directly on the raw crop with no blur or CLAHE.")
    p.add_argument("--clip-limit", type=float, default=0.0,
        help="CLAHE (adaptive histogram equalization) clip limit applied to each tile "
             "before Cellpose (SOPA's StainingSegmentation default is 0.2). Defaults to 0 "
             "(off) here for the same reason as --gaussian-sigma -- and because skimage's "
             "CLAHE kernel size scales with the tile's pixel dimensions when left at its "
             "default, so enabling it makes results depend on --tile-width.")
    p.add_argument("--clahe-kernel-size", type=int, default=None,
        help="CLAHE kernel size in pixels. Only used if --clip-limit > 0. Left unset, "
             "skimage defaults it to the tile's own shape / 8, i.e. it scales with "
             "--tile-width -- set this explicitly if you enable --clip-limit so results "
             "don't change when you change the tile size.")
    p.add_argument("--backend", default="dask", choices=["dask", "threadpool", "none"],
        help="SOPA parallelization backend for tile segmentation. "
             "'dask' (default) processes tiles in parallel using dask.distributed. "
             "'threadpool' uses Python ThreadPoolExecutor. "
             "'none' runs tiles sequentially (safest on GPU, slowest on CPU).")
    return p.parse_args()


def resolve_image_key(sdata: SpatialData, image_key: str | None) -> str:
    key, _ = sopa_utils.get_spatial_image(sdata, key=image_key, return_key=True)
    return key


def find_tissue_roi(
    sdata: SpatialData,
    image_key: str,
    blur_kernel_size: int = 5,
    open_kernel_size: int = 5,
    close_kernel_size: int = 5,
    drop_threshold: float = 0.01,
) -> bool:
    """
    Contour the tissue using SOPA's "saturation" mode (HSV saturation channel
    + Otsu threshold + morphological cleanup) -- the mode SOPA recommends for
    H&E, as opposed to "staining" mode which is for fluorescence images with
    a bright nuclear/marker channel on a dark background. Save the result as
    sdata.shapes['region_of_interest']; make_tiles/sopa.make_image_patches
    picks this up automatically (its default roi_key) and skips any tile
    that doesn't touch it.

    Returns False (and leaves sdata unmodified) if no tissue region could be
    found at all, so callers can fall back to segmenting every tile rather
    than crashing.
    """
    log.info(f"Finding tissue ROI (saturation mode, blur={blur_kernel_size}, "
             f"open={open_kernel_size}, close={close_kernel_size})...")
    try:
        sopa.segmentation.tissue(
            sdata, image_key=image_key, mode="saturation",
            blur_kernel_size=blur_kernel_size, open_kernel_size=open_kernel_size,
            close_kernel_size=close_kernel_size, drop_threshold=drop_threshold,
        )
    except Exception as e:
        log.warning(
            f"Tissue ROI detection failed ({e!r}) -- segmenting every tile instead."
        )
        return False

    n = len(sdata.shapes["region_of_interest"])
    if n == 0:
        log.warning("Tissue ROI detection found 0 regions -- segmenting every tile instead.")
        del sdata.shapes["region_of_interest"]
        return False

    log.info(f"Tissue ROI: {n} region(s).")
    return True


def make_tiles(
    sdata: SpatialData,
    image_key: str,
    tile_width: int,
    tile_overlap: int,
) -> None:
    log.info(f"Tiling: width={tile_width}px, overlap={tile_overlap}px")
    sopa.make_image_patches(
        sdata,
        patch_width=tile_width,
        patch_overlap=tile_overlap,
        image_key=image_key,
    )
    n = len(sdata.shapes[SopaKeys.PATCHES])
    log.info(f"Generated {n} tiles.")


def run_segmentation(
    sdata: SpatialData,
    image_key: str,
    channel_names: list[str],
    cellpose_model: str,
    pretrained_model: str | None,
    diameter: float,
    min_area: float,
    use_gpu: bool,
    cache_dir: Path,
    recover: bool,
    backend: str = "dask",
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    gaussian_sigma: float = 0.0,
    clip_limit: float = 0.0,
    clahe_kernel_size: int | None = None,
) -> None:
    """
    Run CellposeSAM on every tile via StainingSegmentation, then resolve overlaps.

    Unlike the multiplex case (nuclear-only or [cytoplasm, nucleus] channel
    pairs), H&E segmentation uses all 3 RGB channels together: `channels=None`
    tells StainingSegmentation to assume/require an RGB image and pass all 3
    channels through as-is. `cellpose_patch`'s own `channels` argument still
    needs the 3 channel names (it only uses `len(channels)` to route into its
    3-channel/cellpose>=4.0 branch -- CellposeSAM ignores the actual channel
    identities and figures out RGB vs grayscale itself).

    StainingSegmentation reads each tile lazily from the Dask-backed image
    using .isel() -- only the tile's pixels are brought into RAM at a time.
    Results are cached as per-tile Parquet files in cache_dir, so the job can
    be resumed with --recover if it is killed.
    """
    log.info(f"Cellpose mode: RGB (all 3 channels) | channels: {channel_names}")
    log.info(f"flow_threshold={flow_threshold}  cellprob_threshold={cellprob_threshold}")
    if pretrained_model:
        log.info(f"Using custom pretrained model: {pretrained_model} (--cellpose-model ignored)")
    method = cellpose_patch(
        diameter=diameter if diameter > 0 else None,
        channels=channel_names,
        model_type=cellpose_model,
        pretrained_model=pretrained_model,
        gpu=use_gpu,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )

    # Configure parallelization backend.
    # NOTE: when using GPU (use_gpu=True) the 'dask' backend may cause
    # CUDA context errors across workers. If that happens, switch to 'none'.
    if backend == "none":
        sopa.settings.parallelization_backend = None
        log.info("Parallelization: sequential (no backend)")
    else:
        sopa.settings.parallelization_backend = backend
        log.info(f"Parallelization backend: {backend}")

    log.info(
        f"Starting tile-by-tile segmentation "
        f"(model={cellpose_model}, diameter={diameter}, gpu={use_gpu})"
    )

    log.info(f"Tile preprocessing: gaussian_sigma={gaussian_sigma}  clip_limit={clip_limit}"
             f"{f'  clahe_kernel_size={clahe_kernel_size}' if clip_limit > 0 else ''}")
    segmentation = StainingSegmentation(
        sdata,
        method=method,
        channels=None,  # RGB: use all 3 channels, in the image's own order
        image_key=image_key,
        min_area=min_area,
        gaussian_sigma=gaussian_sigma,
        clip_limit=clip_limit,
        clahe_kernel_size=clahe_kernel_size,
    )

    segmentation.write_patches_cells(str(cache_dir), recover=recover)

    log.info("All tiles segmented. Resolving boundary overlaps...")
    cells = StainingSegmentation.read_patches_cells(str(cache_dir))
    cells = solve_conflicts(cells)

    key_added = "cell_boundaries"
    StainingSegmentation.add_shapes(
        sdata, cells, image_key=image_key, key_added=key_added
    )

    n_cells = len(sdata.shapes[key_added])
    log.info(f"Segmentation complete: {n_cells} cells detected.")


def polygons_to_mask_tiff(
    sdata: SpatialData,
    shapes_key: str,
    image_key: str,
    output_path: Path,
) -> None:
    """
    Burn cell boundary polygons into a uint32 labelled instance mask and save
    as a (possibly BigTIFF) TIFF file, using skimage.draw.polygon (exterior
    ring only -- SOPA/Cellpose cell boundaries don't have interior holes).

    Memory note: this allocates a single (H x W) uint32 array. For a
    100 000 x 100 000 image that is ~40 GB.
    """
    image = sopa_utils.get_spatial_image(sdata, key=image_key)
    height = int(image.sizes["y"])
    width = int(image.sizes["x"])

    gdf = sdata.shapes[shapes_key]
    log.info(f"Rasterising {len(gdf)} polygons onto {height}x{width} canvas")

    mask = np.zeros((height, width), dtype=np.uint32)
    n_drawn = 0
    for idx, geom in enumerate(gdf.geometry, start=1):
        if geom is None or geom.is_empty:
            continue
        xs, ys = geom.exterior.coords.xy
        rr, cc = sk_polygon(np.asarray(ys), np.asarray(xs), shape=(height, width))
        mask[rr, cc] = idx
        n_drawn += 1

    n_cells = int(mask.max())
    log.info(f"Mask: {n_drawn} polygons drawn, {n_cells} labelled cells, max ID={n_cells}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_big = mask.nbytes > 2 ** 32  # >4 GB -> BigTIFF required

    log.info(f"Writing {'BigTIFF' if is_big else 'TIFF'}: {output_path}")
    tifffile.imwrite(
        output_path,
        mask,
        photometric="minisblack",
        bigtiff=is_big,
        compression="deflate",
        metadata={"axes": "YX"},
    )
    log.info(f"Done. Mask size on disk: {output_path.stat().st_size / 1e9:.2f} GB")


def main() -> None:
    args = parse_args()

    output_path = Path(args.output)

    if args.cache_dir:
        cache_dir = Path(args.cache_dir)
    else:
        cache_dir = output_path.parent / f".sopa_cache_{output_path.stem}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Tile cache directory: {cache_dir}")

    # Step 1 -- read the already-converted Zarr store (image is file-backed
    # from the start, no write-then-reopen dance needed here).
    log.info(f"Reading Zarr store: {args.zarr}")
    sdata = spatialdata.read_zarr(args.zarr)
    image_key = resolve_image_key(sdata, args.image_key)
    channel_names = list(sopa_utils.get_channel_names(sdata, image_key))
    log.info(f"Image key: '{image_key}' | Channels: {channel_names}")
    if len(channel_names) != 3:
        raise ValueError(
            f"Expected an RGB (3-channel) H&E image, found {len(channel_names)} "
            f"channels: {channel_names}. Use sopa_tile_segment.py instead for "
            "multiplex/fluorescence images."
        )

    # Step 2 -- find the tissue ROI, so background-only tiles get skipped during
    # segmentation instead of producing false-positive cells from slide artifacts
    if not args.skip_tissue_roi:
        find_tissue_roi(
            sdata, image_key,
            blur_kernel_size=args.tissue_blur_kernel_size,
            open_kernel_size=args.tissue_open_kernel_size,
            close_kernel_size=args.tissue_close_kernel_size,
            drop_threshold=args.tissue_drop_threshold,
        )

    # Step 3 -- tile (stores patches in sdata.shapes, in-memory only -- the
    # image itself is already file-backed via the Zarr store we just read)
    make_tiles(sdata, image_key, args.tile_width, args.tile_overlap)

    # Step 4 -- segment tile by tile
    run_segmentation(
        sdata,
        image_key=image_key,
        channel_names=channel_names,
        cellpose_model=args.cellpose_model,
        pretrained_model=args.pretrained_model,
        diameter=args.diameter,
        min_area=args.min_area,
        use_gpu=args.gpu,
        cache_dir=cache_dir,
        recover=args.recover,
        backend=args.backend,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        gaussian_sigma=args.gaussian_sigma,
        clip_limit=args.clip_limit,
        clahe_kernel_size=args.clahe_kernel_size,
    )

    # Step 5 -- rasterise to TIFF
    polygons_to_mask_tiff(
        sdata,
        shapes_key="cell_boundaries",
        image_key=image_key,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
