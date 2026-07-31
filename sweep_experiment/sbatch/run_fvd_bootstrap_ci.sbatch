#!/bin/bash
#SBATCH --job-name=fvd_boot_ci
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/fvd_boot_ci_%j.out
#SBATCH --error=sweep_experiment/slurm_log/fvd_boot_ci_%j.err

# ============================================================================
# Bootstrap CI for FVD + paired ΔFVD vs NO-TTA on the 1000v OOD preview pool.
#
# Answers "does FVD also show up as null with a confidence interval?" the same
# way the PSNR/VBench significance analysis did — but at the I3D-feature level,
# since FVD is a distribution metric. Extracts I3D features once per policy
# (gen-only [48:62] window), restricts to the matched id set, then bootstraps
# the paired difference FVD(policy) − FVD(always_notta).
#
# Uses the SAME preview GT cache + [14 cond | 14 gen] window as the matched FVD
# baselines, and the FVD-duplication-bug-fixed matched policy dirs.
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   sbatch sweep_experiment/sbatch/run_fvd_bootstrap_ci.sbatch
#
# Add the trained routers too (after run_router_fvd.sbatch has built them):
#   EXTRA_POLICIES="psnr_router12:sweep_experiment/reports/budget_oracle_fvd_1000v_preview/routers/psnr_A+B+C_act12/videos" \
#     sbatch sweep_experiment/sbatch/run_fvd_bootstrap_ci.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"

MATCHED_ROOT="${MATCHED_ROOT:-sweep_experiment/reports/budget_oracle_fvd_1000v_preview}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_ood_budget_1000v_preview_longcat.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-sweep_experiment/reports/per_video_analysis/2026-07-31/fvd_bootstrap}"
N_BOOT="${N_BOOT:-2000}"
BASELINE="${BASELINE:-always_notta}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log "${OUTPUT_DIR}"

if [ ! -f "${GT_CACHE}" ]; then
  echo "ERROR: GT cache missing: ${GT_CACHE}" >&2
  exit 2
fi

# Core policies (override POLICIES entirely, or append EXTRA_POLICIES).
DEFAULT_POLICIES=(
  "always_notta:${MATCHED_ROOT}/matched/always_notta/videos"
  "fixed:${MATCHED_ROOT}/matched/fixed_S10_LR5e3/videos"
  "oracle:${MATCHED_ROOT}/oracle_best_psnr/videos"
)
if [ -n "${POLICIES:-}" ]; then
  read -r -a POLICY_ARR <<< "${POLICIES}"
else
  POLICY_ARR=("${DEFAULT_POLICIES[@]}")
fi
if [ -n "${EXTRA_POLICIES:-}" ]; then
  read -r -a EXTRA_ARR <<< "${EXTRA_POLICIES}"
  POLICY_ARR+=("${EXTRA_ARR[@]}")
fi

POLICY_FLAGS=()
for p in "${POLICY_ARR[@]}"; do
  POLICY_FLAGS+=(--policy "${p}")
done

echo "============================================================"
echo "FVD bootstrap CI"
echo "  gt_cache : ${GT_CACHE}"
echo "  baseline : ${BASELINE}"
echo "  n_boot   : ${N_BOOT}"
echo "  output   : ${OUTPUT_DIR}"
echo "  policies :"
for p in "${POLICY_ARR[@]}"; do echo "      ${p}"; done
echo "============================================================"

python3 sweep_experiment/scripts/fvd_bootstrap_ci.py \
  --gt-cache "${GT_CACHE}" \
  --num-cond-frames 14 --num-gen-frames 14 \
  --baseline "${BASELINE}" \
  --n-boot "${N_BOOT}" \
  --output-dir "${OUTPUT_DIR}" \
  "${POLICY_FLAGS[@]}"

echo ""
echo "Done. Summary:"
echo "  cat ${OUTPUT_DIR}/fvd_bootstrap_summary.md"
