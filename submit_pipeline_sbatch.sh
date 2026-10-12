#!/bin/bash -l
#SBATCH --job-name="purkinje-seg"
#SBATCH --partition=cpu-single
#SBATCH --account=bw22b001
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --mem=8gb

# Submits the Nextflow head job; each process is then submitted as its own SLURM job
# (see the `helix` profile in nextflow.config). SLURM_SUBMIT_DIR is used instead of
# "$(dirname "$0")" because sbatch runs a spooled copy of this script.
#
# Expects `nextflow` in ~/bin (see README); Java and Singularity come from cluster modules.
# The SLURM account set above is passed on to every job Nextflow submits.
#
# Usage: sbatch submit_pipeline_sbatch.sh [extra nextflow args, e.g. --images '...']

# loaded before `set -u`: module scripts reference unset variables
module load devel/java_jdk/1.18 system/singularity
export PATH="$HOME/bin:$PATH"

set -eu

REPO_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
cd "$REPO_DIR"

nextflow run main.nf \
    -profile helix,singularity \
    --slurm_account "${SLURM_JOB_ACCOUNT:-bw22b001}" \
    -resume \
    "$@"
