#!/bin/bash
#SBATCH --job-name=five_exp_agg
#SBATCH --output=sweep_experiment/slurm_log/five_exp_agg_%j.out
#SBATCH --error=sweep_experiment/slurm_log/five_exp_agg_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-07-05}"

cd "${PROJECT_ROOT}"
OUT="${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/recommended_five_experiments"

python3 scripts/run_recommended_five_experiments.py --aggregate-only --output-dir "${OUT}"
echo "Aggregate done: ${OUT}/recommended_five_experiments_summary.md"
