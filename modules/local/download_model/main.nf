process DOWNLOAD_MODEL {
    tag "$repo_id"
    label 'process_single'
    container params.container_sopa_cellpose

    input:
    val repo_id
    val filename

    output:
    path "models/*", emit: model
    path "versions.yml", emit: versions

    script:
    def file_arg = filename ? "--filename ${filename}" : ''
    """
    # huggingface_hub isn't baked into container_sopa_cellpose -- install it into the task dir
    # rather than pulling a second image just for this step.
    pip install --quiet --no-cache-dir --target \$PWD/.pylib huggingface_hub
    export PYTHONPATH=\$PWD/.pylib\${PYTHONPATH:+:\$PYTHONPATH}
    export HF_HOME=\$PWD/.hf_home

    python ${projectDir}/scripts/download_model.py \\
        --repo-id ${repo_id} \\
        ${file_arg} \\
        --output-dir models

    # hf_hub_download(local_dir=...) leaves its own metadata cache next to the model file
    rm -rf models/.cache

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        huggingface_hub: \$(python -c "import huggingface_hub; print(huggingface_hub.__version__)")
        hf_repo: ${repo_id}
    END_VERSIONS
    """

    stub:
    """
    mkdir -p models
    touch models/${filename ?: 'model'}
    touch versions.yml
    """
}
