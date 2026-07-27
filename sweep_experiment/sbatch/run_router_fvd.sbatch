#!/bin/bash
#SBATCH --job-name=router_fvd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/router_fvd_%j.out
#SBATCH --error=sweep_experiment/slurm_log/router_fvd_%j.err

# ============================================================================
# FVD of the TRAINED routers (PSNR-motivated + VBench-motivated) on the 1000v
# OOD preview pool, matched-N against always_notta / fixed / oracle.
#
# For each (metric, block, actions) the deployable 5-fold OOF ridge router picks
# a config (or NO-TTA for 13-action) per video; we compose the chosen clips and
# score FVD against the SAME preview GT cache + [48:62] window as the matched
# baselines.  Requires the FVD duplication-bug fix (commit 2b23b8b) so config
# clips resolve 1:1.
#
# VBENCH_SUBDIR=vbench_results_geneval -> the VBench router trains on the
# corrected generated-only VBench scores (same as run_geneval_analyses).
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   sbatch sweep_experiment/sbatch/run_router_fvd.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"

SERIES_ROOT="${SERIES_ROOT:-sweep_experiment/results/panda_ood_budget_1000v_preview}"
FEATURE_DATE="${FEATURE_DATE:-sweep_experiment/reports/per_video_analysis/2026-07-12}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_ood_budget_1000v_preview_longcat.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-sweep_experiment/reports/budget_oracle_fvd_1000v_preview/routers}"

METRICS="${METRICS:-psnr vbench}"
BLOCKS="${BLOCKS:-A+B+C}"
ACTIONS="${ACTIONS:-12 13}"
MIN_VIDEOS="${MIN_VIDEOS:-256}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

# VBench router uses the corrected generated-only per-video scores.
export VBENCH_SUBDIR="${VBENCH_SUBDIR:-vbench_results_geneval}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log "${OUTPUT_ROOT}"

echo "============================================================"
echo "Router FVD (matched-N)"
echo "  series_root  : ${SERIES_ROOT}"
echo "  feature_date : ${FEATURE_DATE}"
echo "  gt_cache     : ${GT_CACHE}"
echo "  output_root  : ${OUTPUT_ROOT}"
echo "  metrics      : ${METRICS}"
echo "  blocks       : ${BLOCKS}"
echo "  actions      : ${ACTIONS}"
echo "  VBENCH_SUBDIR: ${VBENCH_SUBDIR}"
echo "============================================================"

if [ ! -f "${GT_CACHE}" ]; then
  echo "ERROR: GT cache missing: ${GT_CACHE}" >&2
  echo "Run run_preview_1000v_matched_fvd.sbatch first (builds the cache)." >&2
  exit 2
fi

python3 scripts/build_router_fvd_dirs.py \
  --series-root "${SERIES_ROOT}" \
  --feature-date "${FEATURE_DATE}" \
  --gt-cache "${GT_CACHE}" \
  --output-root "${OUTPUT_ROOT}" \
  --metrics ${METRICS} \
  --blocks ${BLOCKS} \
  --actions ${ACTIONS} \
  --min-videos "${MIN_VIDEOS}"

echo ""
echo "Done. Summary:"
echo "  cat ${OUTPUT_ROOT}/router_fvd_summary.md"
