#!/bin/bash
#SBATCH --job-name=pilot_fvd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/pilot_matched_fvd_%j.out
#SBATCH --error=sweep_experiment/slurm_log/pilot_matched_fvd_%j.err

# Matched FVD on the same ~200 pilot videos as budget PSNR oracle:
#   always_notta, fixed_S10_LR5e3, (re-use oracle_best_psnr if present)
#
# Submit:
#   sbatch --account=torch_pr_36_mren sweep_experiment/sbatch/run_pilot_matched_fvd.sbatch
#
# NOTTA + fixed only (skip oracle re-eval):
#   POLICIES="always_notta fixed_S10_LR5e3" sbatch --export=ALL,POLICIES \
#     --account=torch_pr_36_mren sweep_experiment/sbatch/run_pilot_matched_fvd.sbatch

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_1000_longcat.npz}"
POLICIES="${POLICIES:-always_notta fixed_S10_LR5e3 oracle_best_psnr}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

python sweep_experiment/scripts/run_pilot_matched_fvd_baselines.py \
  --gt-cache "${GT_CACHE}" \
  --policies ${POLICIES}

echo ""
echo "Done. Summary:"
echo "  cat sweep_experiment/reports/pilot_matched_fvd/pilot_matched_fvd_summary.md"
echo "  jq '.fvd' sweep_experiment/reports/pilot_matched_fvd/always_notta/fvd.json"
echo "  jq '.fvd' sweep_experiment/reports/pilot_matched_fvd/fixed_S10_LR5e3/fvd.json"
