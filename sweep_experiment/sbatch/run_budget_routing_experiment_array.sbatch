#!/bin/bash
#SBATCH --job-name=budget_route
#SBATCH --output=sweep_experiment/slurm_log/budget_route_%A_%a.out
#SBATCH --error=sweep_experiment/slurm_log/budget_route_%A_%a.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-07-05}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"

cd "${PROJECT_ROOT}"

IFS=',' read -ra EXPS <<< "${EXP_CSV}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
EXP="${EXPS[$IDX]}"

echo "Task ${IDX}: experiment=${EXP}"
echo "Host: $(hostname)  Date: $(date)"

export EXPERIMENT="${EXP}"
export DATE_TAG FEATURE_DATE
bash scripts/run_budget_routing_experiments.sh

echo "Done ${EXP} at $(date)"
