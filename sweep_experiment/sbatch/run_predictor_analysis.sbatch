#!/bin/bash
#SBATCH --job-name=pred_transfer
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/pred_transfer_%j.out
#SBATCH --error=sweep_experiment/slurm_log/pred_transfer_%j.err

# CPU-only predictor transfer analysis (Steps 1–3).
# Submit:
#   sbatch sweep_experiment/sbatch/run_predictor_analysis.sbatch
# Or with custom date tag:
#   sbatch --export=ALL,DATE_TAG=2026-06-30,RUN_ROUTER=1 \
#     sweep_experiment/sbatch/run_predictor_analysis.sbatch

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-06-30}"
FEATURE_DATE="${FEATURE_DATE:-2026-06-09}"
RUN_ROUTER="${RUN_ROUTER:-1}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "  DATE_TAG=${DATE_TAG} FEATURE_DATE=${FEATURE_DATE} RUN_ROUTER=${RUN_ROUTER}"

export REPO="${PROJECT_ROOT}"
export DATE_TAG FEATURE_DATE RUN_ROUTER
bash scripts/run_predictor_analysis_suite.sh

echo "Finished at $(date)"
