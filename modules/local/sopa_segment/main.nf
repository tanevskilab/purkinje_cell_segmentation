process SOPA_SEGMENT {
    tag "$meta.id"
    label 'process_gpu'
    container params.container_sopa_cellpose

    input:
    tuple val(meta), path(zarr_store)
    path model

    output:
    tuple val(meta), path("${meta.id}_mask.tif"), emit: mask
    path "versions.yml", emit: versions

    script:
    def gpu        = params.gpu ? '--gpu' : '--no-gpu'
    def tissue_roi = params.skip_tissue_roi ? '--skip-tissue-roi' : ''
    """
    # `import sopa` pulls in scanpy, whose numba JIT tries to cache next to its installed .py
    # files (read-only inside the container) -- point it at the task dir instead. Same for
    # Cellpose's model cache, which defaults to the (unwritable) \$HOME/.cellpose.
    export NUMBA_CACHE_DIR=\$PWD/.numba_cache
    export CELLPOSE_LOCAL_MODELS_PATH=\$PWD/.cellpose_cache
    export MPLCONFIGDIR=\$PWD/.mpl_cache
    mkdir -p \$NUMBA_CACHE_DIR \$CELLPOSE_LOCAL_MODELS_PATH \$MPLCONFIGDIR

    python ${projectDir}/scripts/sopa_segment_he.py \\
        --zarr ${zarr_store} \\
        --output ${meta.id}_mask.tif \\
        --cache-dir .sopa_cache \\
        --pretrained-model ${model} \\
        --tile-width ${params.tile_width} \\
        --tile-overlap ${params.tile_overlap} \\
        --diameter ${params.diameter} \\
        --min-area ${params.min_area} \\
        --flow-threshold ${params.flow_threshold} \\
        --cellprob-threshold ${params.cellprob_threshold} \\
        --gaussian-sigma ${params.gaussian_sigma} \\
        --clip-limit ${params.clip_limit} \\
        --tissue-blur-kernel-size ${params.tissue_blur_kernel_size} \\
        --tissue-open-kernel-size ${params.tissue_open_kernel_size} \\
        --tissue-close-kernel-size ${params.tissue_close_kernel_size} \\
        --tissue-drop-threshold ${params.tissue_drop_threshold} \\
        --backend ${params.sopa_backend} \\
        ${gpu} \\
        ${tissue_roi}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        sopa: \$(python -c "import sopa; print(sopa.__version__)")
        cellpose: \$(python -c "import importlib.metadata as m; print(m.version('cellpose'))")
        torch: \$(python -c "import torch; print(torch.__version__)")
        model: ${model.name}
    END_VERSIONS
    """

    stub:
    """
    touch ${meta.id}_mask.tif
    touch versions.yml
    """
}
