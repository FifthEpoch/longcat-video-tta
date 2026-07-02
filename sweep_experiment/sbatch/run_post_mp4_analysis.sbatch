#!/bin/bash
#SBATCH --job-name=post_mp4_an
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/post_mp4_an_%j.out
#SBATCH --error=sweep_experiment/slurm_log/post_mp4_an_%j.err

# CPU-only: full analysis chain after mp4 + VBench backfill complete.
#
#   sbatch --export=ALL,DATE_TAG=2026-07-02 sweep_experiment/sbatch/run_post_mp4_analysis.sbatch

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
export REPO="${PROJECT_ROOT}"
export DATE_TAG
bash scripts/run_post_mp4_analysis_chain.sh
echo "Finished at $(date)"
