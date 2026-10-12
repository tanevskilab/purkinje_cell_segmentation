# purkinje_cell_segmentation
Purkinje cell detection workflow focussing on CellposeSAM-based segmentation from H&amp;E data with 20x magnification. A human-in-the-loop approach was used in order to train the model. The pipeline uses Nextflow, a workflow manager tool that increases its reusability by making an abstract layer that communicates with the underlying operating system. Since images are quite big, the pipeline is optimized for the Helix cluster so that the segmentation step can use the GPU of the system.

# Installation

## Clone the repository

```bash
git clone https://github.com/tanevskilab/purkinje_cell_segmentation.git
cd purkinje_cell_segmentation
```

The recommendation is to clone the repository inside a previously created workspace in Helix since it needs enough storage for all intermediate and final output files. 

# Description

## Nextflow pipeline (whole slides on the cluster)

`main.nf` chains the steps for every image matching `--images`:

1. `DOWNLOAD_MODEL` -- fetches the fine-tuned CellposeSAM checkpoint from Hugging Face
   (`--hf_repo_id`, `--hf_filename`; skipped if `--model /path/to/checkpoint` is given)
2. `SOPA_WRITE_ZARR` -- `scripts/sopa_write_zarr.py`, TIFF -> SpatialData Zarr store
3. `SOPA_SEGMENT` -- `scripts/sopa_segment_he.py` on a GPU node, Zarr -> labelled mask TIFF
4. `MERGE_MASKS` -- take a mask of the SpaceRanger software and merge it with the purkinje cell mask. 

All steps run in the `ghcr.io/sebgoti/sopa-cellpose-gpu:v1` image. Running the pipeline should be as simple as:

```bash
# all slides in /gpfs/bwfor/work/ws/hd_qg295-VisiumHD_images
sbatch submit_pipeline_sbatch.sh --outdir results
```

Outputs go to `<outdir>/model/`, `<outdir>/zarr/` (`--publish_zarr false` to skip copying
them), `<outdir>/masks/<sample>_mask.tif` and `<outdir>/pipeline_info/`. Segmentation
parameters (`--tile_width`, `--diameter`, `--flow_threshold`, ...) are listed in
`nextflow.config`. Jobs are charged to the `bw22b001` SLURM account (`#SBATCH --account` in
`submit_pipeline_sbatch.sh` and `--slurm_account` in `nextflow.config`); change both if needed.



