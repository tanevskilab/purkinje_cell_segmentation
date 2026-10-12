#!/usr/bin/env nextflow
/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    purkinje_cell_segmentation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1. DOWNLOAD_MODEL  : fetch the fine-tuned CellposeSAM checkpoint from Hugging Face
    2. SOPA_WRITE_ZARR : H&E TIFF -> Zarr-backed SpatialData store
    3. SOPA_SEGMENT    : tile-based CellposeSAM segmentation via SOPA -> labelled mask TIFF
    4. MERGE_MASKS     : (only with --nuclei_masks) add externally produced nuclei; the
                         Purkinje mask wins where they overlap
----------------------------------------------------------------------------------------
*/

include { DOWNLOAD_MODEL  } from './modules/local/download_model'
include { SOPA_WRITE_ZARR } from './modules/local/sopa_write_zarr'
include { SOPA_SEGMENT    } from './modules/local/sopa_segment'
include { MERGE_MASKS     } from './modules/local/merge_masks'

workflow {

    ch_images = channel
        .fromPath(params.images, checkIfExists: true)
        .map { image -> [ [id: image.baseName], image ] }

    // --model points at an already-downloaded checkpoint and skips the Hugging Face step
    if (params.model) {
        ch_model = channel.value(file(params.model, checkIfExists: true))
    } else {
        DOWNLOAD_MODEL(params.hf_repo_id, params.hf_filename)
        ch_model = DOWNLOAD_MODEL.out.model
    }

    SOPA_WRITE_ZARR(ch_images)

    SOPA_SEGMENT(SOPA_WRITE_ZARR.out.zarr, ch_model)

    // Nuclei masks are matched to images by sample id: <id><nuclei_mask_suffix>.tif(f)
    if (params.nuclei_masks) {
        ch_nuclei = channel
            .fromPath(params.nuclei_masks, checkIfExists: true)
            .map { mask ->
                def id = mask.name.replaceFirst(/\.tiff?$/, '')
                if (!id.endsWith(params.nuclei_mask_suffix)) {
                    error "Nuclei mask '${mask.name}' doesn't end in '${params.nuclei_mask_suffix}.tif' -- set --nuclei_mask_suffix to match your file names"
                }
                [ [id: id.substring(0, id.length() - params.nuclei_mask_suffix.length())], mask ]
            }

        ch_to_merge = SOPA_SEGMENT.out.mask
            .join(ch_nuclei, remainder: true)
            .filter { meta, cell_mask, nuclei_mask ->
                if (cell_mask && nuclei_mask) {
                    return true
                }
                log.warn cell_mask
                    ? "No nuclei mask for '${meta.id}' -- skipping MERGE_MASKS for it"
                    : "Nuclei mask '${nuclei_mask.name}' matches no segmented image -- ignored"
                return false
            }

        MERGE_MASKS(ch_to_merge)
    }
}
