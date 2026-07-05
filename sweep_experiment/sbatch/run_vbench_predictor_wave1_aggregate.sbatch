#!/bin/bash
#SBATCH --job-name=wave1_agg
#SBATCH --output=sweep_experiment/slurm_log/wave1_agg_%j.out
#SBATCH --error=sweep_experiment/slurm_log/wave1_agg_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-07-06}"
OUT="${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/wave1_predictor_experiments"

cd "${PROJECT_ROOT}"
python3 scripts/run_vbench_predictor_wave1_experiments.py --aggregate-only --output-dir "${OUT}"
echo "Done: ${OUT}/wave1_predictor_summary.md"
