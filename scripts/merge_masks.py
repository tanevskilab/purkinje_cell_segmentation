"""Merge the Purkinje cell mask with a separately produced nuclei mask.

Both inputs are labelled instance masks (0 = background) of the same H&E image. The
Purkinje mask always wins:

  * every Purkinje cell is kept unchanged, with its original label;
  * a nucleus is dropped entirely if more than --max-overlap of its area lies inside
    Purkinje cells (i.e. it is the same object, or the nucleus of a segmented cell);
  * a kept nucleus that only partially touches a Purkinje cell keeps just its pixels
    outside that cell (Purkinje pixels are never overwritten).

Kept nuclei are relabelled to <label offset> + <original nucleus label>, where the offset
is the highest Purkinje label, so in the merged mask:

    1 .. offset      -> Purkinje cells (same IDs as the input Purkinje mask)
    > offset         -> nuclei          (original nucleus ID = label - offset)

A JSON summary with the offset and counts is written next to the merged mask.

Memory: both masks are read in 1024-row bands and the output is written tile by tile, so
RAM stays proportional to one band (plus per-label counters), not the whole slide.

Usage
-----
    python scripts/merge_masks.py \\
        --cells  results/masks/sample_mask.tif \\
        --nuclei nuclei/sample_nuclei_mask.tif \\
        --output results/merged/sample_merged_mask.tif \\
        --max-overlap 0.5
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import tifffile
import zarr

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TILE = 1024  # band height for reading and tile size of the output TIFF (multiple of 16)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", required=True, help="Purkinje cell mask TIFF (SOPA_SEGMENT output).")
    p.add_argument("--nuclei", required=True, help="Nuclei mask TIFF for the same image.")
    p.add_argument("--output", required=True, help="Path for the merged mask TIFF.")
    p.add_argument("--max-overlap", type=float, default=0.5,
        help="Drop a nucleus if more than this fraction of its area overlaps Purkinje cells.")
    return p.parse_args()


def open_mask(tif: tifffile.TiffFile) -> zarr.Array:
    """Lazy 2D view of the first series' full-resolution level."""
    z = zarr.open(tif.aszarr(level=0), mode="r")
    if not isinstance(z, zarr.Array):  # multiscale group -> full resolution
        z = z[sorted(z.keys())[0]]
    if z.ndim != 2:
        raise ValueError(f"{tif.filename}: expected a 2D label image, got shape {z.shape}")
    return z


def bands(*masks: zarr.Array):
    """Yield (y0, band_1, band_2, ...) over matching 1024-row bands."""
    for y0 in range(0, masks[0].shape[0], TILE):
        yield (y0, *(np.asarray(m[y0:y0 + TILE]) for m in masks))


def grow(counts: np.ndarray, n: int) -> np.ndarray:
    return counts if len(counts) >= n else np.pad(counts, (0, n - len(counts)))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(args.cells) as cells_tif, tifffile.TiffFile(args.nuclei) as nuc_tif:
        cells, nuclei = open_mask(cells_tif), open_mask(nuc_tif)
        if cells.shape != nuclei.shape:
            raise ValueError(f"Mask shapes differ: cells {cells.shape} vs nuclei {nuclei.shape}. "
                             "Both masks must be at the same resolution as the H&E image.")
        height, width = cells.shape
        log.info(f"Masks: {height} x {width} px (cells {cells.dtype}, nuclei {nuclei.dtype})")

        # --- pass 1: nucleus areas and how much of each lies inside Purkinje cells ----
        nuc_area = np.zeros(1, np.int64)
        nuc_overlap = np.zeros(1, np.int64)
        max_cell = 0
        for _, c, n in bands(cells, nuclei):
            max_cell = max(max_cell, int(c.max()))
            size = int(n.max()) + 1
            nuc_area = grow(nuc_area, size)
            nuc_overlap = grow(nuc_overlap, size)
            nuc_area[:size] += np.bincount(n.ravel(), minlength=size)
            nuc_overlap[:size] += np.bincount(n[c > 0], minlength=size)

        nuc_area[0] = nuc_overlap[0] = 0
        present = nuc_area > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.where(present, nuc_overlap / np.maximum(nuc_area, 1), 0.0)
        drop = present & (frac > args.max_overlap)
        keep = present & ~drop
        clipped = keep & (nuc_overlap > 0)

        offset = max_cell
        if offset + len(nuc_area) - 1 > np.iinfo(np.uint32).max:
            raise ValueError("Too many labels for a uint32 merged mask.")
        log.info(f"Purkinje cells: max label {max_cell} (= nucleus label offset)")
        log.info(f"Nuclei: {int(present.sum())} total, {int(drop.sum())} dropped "
                 f"(> {args.max_overlap:.0%} inside a Purkinje cell), {int(keep.sum())} kept, "
                 f"of which {int(clipped.sum())} clipped where they touch a Purkinje cell")

        # --- pass 2: write the merged mask tile by tile -------------------------------
        def tiles():
            for _, c, n in bands(cells, nuclei):
                merged = c.astype(np.uint32)
                fill = (c == 0) & keep[n]
                merged[fill] = n[fill].astype(np.uint32) + offset
                band = np.zeros((TILE, -(-width // TILE) * TILE), np.uint32)
                band[: merged.shape[0], :width] = merged
                for x0 in range(0, width, TILE):
                    yield band[:, x0:x0 + TILE]

        is_big = height * width * 4 > 2 ** 32
        log.info(f"Writing {'BigTIFF' if is_big else 'TIFF'}: {output}")
        tifffile.imwrite(output, tiles(), shape=(height, width), dtype=np.uint32,
                         tile=(TILE, TILE), photometric="minisblack", bigtiff=is_big,
                         compression="deflate", metadata={"axes": "YX"})

    summary = {
        "cells_mask": str(args.cells),
        "nuclei_mask": str(args.nuclei),
        "max_overlap": args.max_overlap,
        "nucleus_label_offset": int(offset),
        "label_ranges": {"purkinje_cells": [1, int(offset)],
                         "nuclei": "label - nucleus_label_offset = original nucleus label"},
        "nuclei_total": int(present.sum()),
        "nuclei_dropped": int(drop.sum()),
        "nuclei_kept": int(keep.sum()),
        "nuclei_kept_clipped": int(clipped.sum()),
    }
    summary_path = output.with_name(output.name.removesuffix(".tif") + "_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    log.info(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
