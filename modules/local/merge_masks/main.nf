process MERGE_MASKS {
    tag "$meta.id"
    label 'process_merge'
    container params.container_sopa_cellpose

    input:
    tuple val(meta), path(cell_mask), path(nuclei_mask)

    output:
    tuple val(meta), path("${meta.id}_merged_mask.tif"), emit: mask
    tuple val(meta), path("${meta.id}_merged_mask_summary.json"), emit: summary
    path "versions.yml", emit: versions

    script:
    """
    python ${projectDir}/scripts/merge_masks.py \\
        --cells ${cell_mask} \\
        --nuclei ${nuclei_mask} \\
        --output ${meta.id}_merged_mask.tif \\
        --max-overlap ${params.merge_max_overlap}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)")
    END_VERSIONS
    """

    stub:
    """
    touch ${meta.id}_merged_mask.tif ${meta.id}_merged_mask_summary.json versions.yml
    """
}
