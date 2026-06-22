#!/bin/bash
#SBATCH --job-name=budget_oracle_fvd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/budget_oracle_fvd_%j.out
#SBATCH --error=sweep_experiment/slurm_log/budget_oracle_fvd_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SERIES_ROOT="${SERIES_ROOT:-${PROJECT_ROOT}/sweep_experiment/results/panda_ood_budget_pilot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/sweep_experiment/reports/budget_oracle_fvd}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_1000_longcat.npz}"
SKIP_BUILD="${SKIP_BUILD:-0}"

cd "${PROJECT_ROOT}"
source /scratch/wc3013/conda-envs/longcat/bin/activate 2>/dev/null || true

EXTRA=()
[ "${SKIP_BUILD}" = "1" ] && EXTRA+=(--skip-build)

python sweep_experiment/scripts/run_budget_oracle_fvd.py \
    --series-root "${SERIES_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --gt-cache "${GT_CACHE}" \
    "${EXTRA[@]}"
