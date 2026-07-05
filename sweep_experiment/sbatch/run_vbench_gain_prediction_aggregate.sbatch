#!/bin/bash
#SBATCH --job-name=vb_gain_agg
#SBATCH --output=sweep_experiment/slurm_log/vb_gain_agg_%j.out
#SBATCH --error=sweep_experiment/slurm_log/vb_gain_agg_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-07-05}"
OUT="${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/vbench_gain_prediction_experiments"

cd "${PROJECT_ROOT}"
python3 scripts/run_vbench_gain_prediction_experiments.py --aggregate-only --output-dir "${OUT}"
echo "Aggregate done: ${OUT}/vbench_gain_prediction_summary.md"
