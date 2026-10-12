process SOPA_WRITE_ZARR {
    tag "$meta.id"
    label 'process_zarr'
    container params.container_sopa_cellpose

    input:
    tuple val(meta), path(image)

    output:
    tuple val(meta), path("${meta.id}.zarr"), emit: zarr
    path "versions.yml", emit: versions

    script:
    """
    python ${projectDir}/scripts/sopa_write_zarr.py \\
        --image ${image} \\
        --output ${meta.id}.zarr \\
        --n-workers ${task.cpus}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        spatialdata: \$(python -c "import spatialdata; print(spatialdata.__version__)")
        tifffile: \$(python -c "import tifffile; print(tifffile.__version__)")
    END_VERSIONS
    """

    stub:
    """
    mkdir -p ${meta.id}.zarr
    touch versions.yml
    """
}
