# purkinje_cell_segmentation
Purkinje cell detection workflow focussing on CellposeSAM-based segmentation from H&amp;E data with 20x magnification.

# Installation

## Clone the repository

```bash
git clone https://github.com/tanevskilab/purkinje_cell_segmentation.git
cd purkinje_cell_segmentation
```

All scripts are meant to be run inside a CellposeSAM-capable Docker container with a
CUDA-enabled `torch` build that matches the host's GPU. On this cluster, two images are
available:

- `community.wave.seqera.io/library/python_pip_cellpose:fdf7a8c3a305a26e` — plain Cellpose
  (torch 2.10.0+cu128, works on NVIDIA Blackwell GPUs e.g. RTX PRO 6000). Missing
  `matplotlib`, so install it inline (see below). Use for `segment_image.py`.
- `sopa-cellpose-gpu:blackwell` — Cellpose + SOPA + spatialdata, built from
  `sopa_cellpose_image/Dockerfile.blackwell` in the sibling `sopa_cellpose_image` repo
  (same torch/cu128 build, plus the packages `sopa_write_zarr.py`/`sopa_segment_he.py`
  need). Use for the SOPA scripts. Rebuild it with:
  `docker build -f Dockerfile.blackwell -t sopa-cellpose-gpu:blackwell .`

Both images expose the GPU via `--device nvidia.com/gpu=all` (this host uses CDI, not the
older `--gpus all` flag).

To keep the commands below short, set these once per shell session:

```bash
export REPO=/path/to/purkinje_cell_segmentation   # this repo, absolute path
export CELLPOSE="docker run --rm --device nvidia.com/gpu=all -v $REPO:/workspace -v ~/.cellpose:/root/.cellpose -w /workspace"
```

`$CELLPOSE` is just a `docker run` prefix -- append an image name and command to it. The
`~/.cellpose` mount caches downloaded/fine-tuned model weights across runs so they aren't
re-fetched every time.

# Usage

## Segmenting a single, moderately-sized image (`segment_image.py`)

Loads the whole image into memory and runs Cellpose in one call -- fine for individual
crops, but **will run out of memory on a whole-slide image** (its internal flow/mask
buffers scale with image size; a ~50k x 64k slide needs far more RAM than the raw
~9.6 GB pixel array). Use the SOPA scripts below for whole-slide images instead.

```bash
$CELLPOSE community.wave.seqera.io/library/python_pip_cellpose:fdf7a8c3a305a26e \
  bash -c "pip install --quiet matplotlib && python scripts/segment_image.py cpsam your_image.tif --output-dir results"
```

`cpsam` above is CellposeSAM's generic pretrained model. To use a fine-tuned model instead
(see "Model management" below), pass its local path in place of `cpsam`.

## Segmenting a whole-slide image (SOPA, tile-based)

`sopa_write_zarr.py` and `sopa_segment_he.py` process the image tile-by-tile via SOPA
instead of loading it all at once, so memory stays proportional to one tile
(`--tile-width`), not the whole slide. **Run them in this order** --
`sopa_segment_he.py` reads the Zarr store that `sopa_write_zarr.py` writes, so the first
must finish before the second starts:

```bash
# 1. Convert the H&E TIFF to a Zarr-backed SpatialData store (streams, no full-image load)
$CELLPOSE sopa-cellpose-gpu:blackwell scripts/sopa_write_zarr.py \
  --image your_image.tif --output results/slide.zarr --overwrite

# 2. Tile-based CellposeSAM segmentation via SOPA (must run after step 1 completes)
$CELLPOSE sopa-cellpose-gpu:blackwell scripts/sopa_segment_he.py \
  --zarr results/slide.zarr --output results/slide_masks.tif \
  --tile-width 4096 --tile-overlap 256 --cellpose-model cpsam --backend none
```

(`sopa-cellpose-gpu:blackwell`'s `ENTRYPOINT` is already `python3`, so no `--entrypoint`
flag is needed -- the script path is just passed as the command.)

To use a fine-tuned model instead of the generic `cpsam` weights, replace
`--cellpose-model cpsam` with `--pretrained-model models/<model_name>` (paths are relative
to `$REPO`, mounted at `/workspace`; see "Model management" below). `--backend none` runs
tiles sequentially, which is the safest option on GPU; `--backend dask` or
`--backend threadpool` may be faster but can hit CUDA context errors across workers.

Output: a labelled instance mask TIFF (`--output`), plus a `.sopa_cache_<name>/` directory
of per-tile results next to it (safe to delete once segmentation finishes; also what
`--recover` resumes from if the run is interrupted).

## Model management

The repo doesn't track trained model weights directly -- they're stored on Hugging Face
Hub (see `scripts/hf_config.py` for the repo id) and fetched/pushed with:

```bash
python scripts/download_model.py --output-dir models/
python scripts/upload_model.py <local_model_path>
```

Both need `pip install huggingface_hub` and, for a private repo, `huggingface-cli login`
(or `export HF_TOKEN=...`). New models come from human-in-the-loop correction in the
Cellpose GUI -- see `notebooks/napari_training.ipynb`.

