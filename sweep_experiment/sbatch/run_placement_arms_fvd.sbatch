#!/bin/bash
#SBATCH --job-name=placement_fvd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/placement_fvd_%j.out
#SBATCH --error=sweep_experiment/slurm_log/placement_fvd_%j.err

# ============================================================================
# EXP2 close-out: matched-N FVD for the placement arms (ADA_ADALN, ADA_RESID)
# vs NO-TTA, on the COMMON video set, same eval_fvd protocol as the 157.05
# preview baseline. Answers whether residual placement moves the TARGET metric.
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   sbatch sweep_experiment/sbatch/run_placement_arms_fvd.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"

PLACEMENT_SERIES="${PLACEMENT_SERIES:-sweep_experiment/results/placement_ablation_panda}"
PREVIEW_SERIES="${PREVIEW_SERIES:-sweep_experiment/results/panda_ood_budget_1000v_preview}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_ood_budget_1000v_preview_longcat.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-sweep_experiment/reports/budget_oracle_fvd_1000v_preview/placement_arms}"
ARMS="${ARMS:-ADA_ADALN ADA_RESID}"
NOTTA_RUN="${NOTTA_RUN:-NOTTA}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "${CONDA_ENV}"
unset PYTHONHOME PYTHONPATH
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log "${OUTPUT_ROOT}"

echo "============================================================"
echo "EXP2 placement arms — matched-N FVD"
echo "  placement_series : ${PLACEMENT_SERIES}"
echo "  preview_series   : ${PREVIEW_SERIES}"
echo "  gt_cache         : ${GT_CACHE}"
echo "  arms             : ${ARMS}   notta=${NOTTA_RUN}"
echo "  output_root      : ${OUTPUT_ROOT}"
echo "============================================================"

if [ ! -f "${GT_CACHE}" ]; then
  echo "ERROR: GT cache missing: ${GT_CACHE}" >&2
  exit 2
fi

python3 scripts/score_placement_arms_fvd.py \
  --placement-series "${PLACEMENT_SERIES}" \
  --preview-series "${PREVIEW_SERIES}" \
  --gt-cache "${GT_CACHE}" \
  --arms ${ARMS} \
  --notta-run "${NOTTA_RUN}" \
  --output-root "${OUTPUT_ROOT}"

echo ""
echo "Done. Summary:"
echo "  cat ${OUTPUT_ROOT}/placement_arms_fvd_summary.md"
