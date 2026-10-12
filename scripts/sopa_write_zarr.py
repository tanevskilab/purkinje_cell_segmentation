"""Convert an H&E TIFF into a Zarr-backed SpatialData store, without loading
any pixel data into RAM.

Adapted from ~/Repositories/tspc/bin/sopa_write_zarr.py (written for
multiplex OME-TIFFs) for this repo's plain H&E TIFFs: those have no OME-XML
(so channel names always fall back to numeric) and tifffile commonly reports
their axes as "YXS" (Y, X, Samples-per-pixel) rather than "YXC" -- the
original script only recognized "YXC", so it would silently mis-tag a
"YXS" R/G/B image's axes as "CYX" and segment garbage. This version treats
"YXS" the same as "YXC".

Memory strategy
---------------
Uses tifffile's aszarr()/dask.array.from_zarr() to build a Dask-backed image:
only the chunks actually touched get read from disk. The subsequent
`sdata.write()` streams chunk-by-chunk too, so peak RAM stays proportional to
one chunk, not the whole image -- unlike tifffile.imread(), which loads the
entire image at once (this is what made segment_image.py OOM on the
whole-slide crop: it calls tifffile.imread() then Cellpose.eval() on the full
~9.6 GB array plus its internal flow/mask buffers in one shot).

Usage
-----
    python scripts/sopa_write_zarr.py \\
        --image  260427_CO36_RCLB5_VisiumHD_20x_5_cropped.tif \\
        --output results/260427_CO36_RCLB5_VisiumHD_20x_5_cropped.zarr
"""

import argparse
import logging
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import dask
import dask.array as da
import tifffile
import zarr
from spatialdata import SpatialData
from spatialdata.models import Image2DModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

CHUNK_SIZE = 4096  # target y/x chunk size for the zarr store, in pixels


def load_lazy(path: str) -> tuple[SpatialData, str, list[str]]:
    """
    Build a Dask-backed SpatialData object from an H&E TIFF without loading
    any pixel data into RAM.

    Returns:
        sdata        : SpatialData with a single lazy (C, Y, X) image
        image_key    : key under which the image is stored in sdata.images
        channel_names: list of channel name strings (numeric ch0/ch1/ch2 for
                        plain H&E TIFFs, since there's no OME-XML to name them)
    """
    path = str(path)
    image_key = Path(path).stem.split(".")[0]  # use filename stem as key
    log.info(f"Reading TIFF metadata: {path}")

    with tifffile.TiffFile(path) as tif:
        # --- channel names from OME-XML, if present (metadata only) ---
        channel_names = None
        try:
            ome_xml = tif.ome_metadata
            if ome_xml:
                ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
                root = ET.fromstring(ome_xml)
                names = [
                    ch.get("Name", f"ch{i}")
                    for i, ch in enumerate(root.findall(".//ome:Channel", ns))
                ]
                if names:
                    channel_names = names
                    log.info(f"Channel names from OME-XML: {channel_names}")
        except Exception as e:
            log.warning(f"Could not parse OME-XML channel names: {e}")

        # --- shape and dtype from the first series ---
        series = tif.series[0]
        shape = series.shape  # e.g. (C, Y, X), (Y, X, C), or (Y, X, S) for plain RGB
        axes = series.axes
        dtype = series.dtype
        log.info(f"Series shape: {shape}, axes: {axes}, dtype: {dtype}")

        # Uncompressed, contiguous TIFFs (e.g. the VisiumHD H&E scans, which are
        # untiled strips of 1 row each) can be memory-mapped directly. Going
        # through aszarr() instead would give one dask chunk per *row* (tens of
        # thousands of tasks) and make the write extremely slow.
        contiguous = series.dataoffset is not None

    if contiguous:
        log.info("TIFF data is contiguous/uncompressed: reading via memmap")
        arr = da.from_array(tifffile.memmap(path, mode="r"), chunks=CHUNK_SIZE)
    else:
        arr = _load_aszarr(path)

    log.info(f"Dask array shape: {arr.shape}, chunks: {arr.chunksize}")

    # --- normalise to (C, Y, X) ---
    axes = axes.upper()
    if axes == "CYX":
        pass  # already correct
    elif axes in ("YXC", "YXS"):
        # "S" = samples-per-pixel, tifffile's axis label for plain (non-OME)
        # RGB TIFFs -- same (Y, X, channel-last) layout as "YXC".
        arr = da.moveaxis(arr, -1, 0)
        axes = "CYX"
    elif axes == "ZCYX":
        arr = arr[0]  # take first Z slice
        axes = "CYX"
    elif axes == "CZYX":
        arr = arr[:, 0]
        axes = "CYX"
    else:
        log.warning(f"Unexpected axes order '{axes}'; assuming first dim is C.")

    # tifffile's aszarr() inherits the source TIFF's internal tile/strip size,
    # which can produce a huge number of tiny zarr chunks. On a parallel
    # filesystem the per-file overhead of that many chunks dominates write
    # time, so rechunk to fewer, larger blocks (no-op for the memmap path).
    arr = arr.rechunk({0: -1, 1: CHUNK_SIZE, 2: CHUNK_SIZE})

    n_channels, height, width = arr.shape

    if channel_names is None:
        channel_names = [f"ch{i}" for i in range(n_channels)]
        log.warning(f"Falling back to numeric channel names: {channel_names}")
    elif len(channel_names) != n_channels:
        log.warning(
            f"OME-XML has {len(channel_names)} channel names but image has "
            f"{n_channels} channels. Falling back to numeric names."
        )
        channel_names = [f"ch{i}" for i in range(n_channels)]

    log.info(
        f"Image key: '{image_key}' | "
        f"Shape: {n_channels}c x {height}y x {width}x | "
        f"Channels: {channel_names}"
    )

    # --- wrap in SpatialData ---
    image_model = Image2DModel.parse(
        arr,
        dims=("c", "y", "x"),
        c_coords=channel_names,
        scale_factors=[2, 2, 2],
        # Without this, parse() rechunks every pyramid level to 256x256 regardless of the
        # dask chunking above -- ~150k chunk files for the largest slide's full-res level.
        chunks={"c": -1, "y": CHUNK_SIZE, "x": CHUNK_SIZE},
    )
    sdata = SpatialData(images={image_key: image_model})
    return sdata, image_key, channel_names


def _load_aszarr(path: str) -> da.Array:
    """Lazy dask array of the full-resolution level via tifffile's ZarrStore."""
    with tifffile.TiffFile(path) as tif:
        store = tif.aszarr(level=0)
    z = zarr.open(store, mode="r")

    # zarr v2 keeps Group in zarr.hierarchy; zarr v3 exposes it at zarr.Group.
    # Try both to stay compatible with whichever version is installed.
    try:
        group_type = zarr.hierarchy.Group
    except AttributeError:
        group_type = zarr.Group

    # zarr root may be a group (multiscale) or array (single level)
    if isinstance(z, group_type):
        # multiscale: key "0" is always full resolution in OME-Zarr convention
        first_key = sorted(z.keys())[0]
        return da.from_zarr(z[first_key])
    return da.from_zarr(z)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an H&E TIFF into a Zarr-backed SpatialData store "
                     "(preprocessing step for sopa_segment_he.py).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="Path to input H&E TIFF.")
    parser.add_argument("--output", required=True, help="Path for the output Zarr store (directory).")
    parser.add_argument("--overwrite", action="store_true", default=False,
        help="Remove --output first if it already exists, instead of failing.")
    parser.add_argument("--n-workers", type=int, default=1,
        help="Threads for dask's scheduler, driving chunk compression/write "
             "and multiscale downsampling in parallel. Should match the "
             "number of CPUs allocated to this task.")
    args = parser.parse_args()

    dask.config.set(scheduler="threads", num_workers=args.n_workers)
    log.info(f"Using dask threaded scheduler with {args.n_workers} workers.")

    output_path = Path(args.output)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"{output_path} already exists. Pass --overwrite to replace it.")
        log.info(f"--overwrite set: removing existing {output_path}")
        shutil.rmtree(output_path)

    sdata, image_key, channel_names = load_lazy(args.image)

    log.info(f"Writing SpatialData to Zarr store: {output_path}")
    sdata.write(output_path)
    log.info(
        f"Done. image_key='{image_key}'  n_channels={len(channel_names)}  "
        f"channels={channel_names}"
    )


if __name__ == "__main__":
    main()
