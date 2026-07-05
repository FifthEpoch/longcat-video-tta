#!/bin/bash
#SBATCH --job-name=wave1_pred
#SBATCH --output=sweep_experiment/slurm_log/wave1_pred_%A_%a.out
#SBATCH --error=sweep_experiment/slurm_log/wave1_pred_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-07-06}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"

cd "${PROJECT_ROOT}"
EXP_LIST="${PROJECT_ROOT}/sweep_experiment/sbatch/.wave1_predictor_experiments.lst"
mapfile -t EXPS < "${EXP_LIST}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
EXP="${EXPS[$IDX]}"

echo "Task ${IDX}: ${EXP}"
export EXPERIMENT="${EXP}"
export DATE_TAG FEATURE_DATE
bash scripts/run_vbench_predictor_wave1_experiments.sh
