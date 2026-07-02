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

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SERIES_ROOT="${SERIES_ROOT:-${PROJECT_ROOT}/sweep_experiment/results/panda_ood_budget_pilot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/sweep_experiment/reports/budget_oracle_fvd}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_1000_longcat.npz}"
SKIP_BUILD="${SKIP_BUILD:-0}"
FORCE="${FORCE:-1}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

EXTRA=()
[ "${SKIP_BUILD}" = "1" ] && EXTRA+=(--skip-build)
[ "${FORCE}" = "1" ] && EXTRA+=(--force)

python sweep_experiment/scripts/run_budget_oracle_fvd.py \
    --series-root "${SERIES_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --gt-cache "${GT_CACHE}" \
    "${EXTRA[@]}"
