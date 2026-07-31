#!/bin/bash
#SBATCH --job-name=router_fvd_ci
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/router_fvd_ci_%j.out
#SBATCH --error=sweep_experiment/slurm_log/router_fvd_ci_%j.err

# ============================================================================
# Bootstrap CI for the FVD of the DEPLOYABLE trained routers, vs NO-TTA.
#
# We already have ΔFVD-vs-NO-TTA CIs for the two ORACLE/heuristic policies
# (fixed = null, PSNR-oracle = improves but non-deployable). This job adds the
# four deployable routers (PSNR/VBench x 12/13 actions, block A+B+C, leakage-free
# 5-fold OOF ridge) to the SAME paired bootstrap so we can say, for each router,
# whether its ΔFVD-vs-NO-TTA 95% CI excludes 0.
#
# Two stages, one GPU:
#   1) build_router_fvd_dirs.py  -> compose each router's per-video picks into a
#      matched-N videos/ dir (symlinks) under OUTPUT_ROOT/routers (idempotent;
#      also writes each router's point FVD). Requires the FVD duplication-bug fix.
#   2) fvd_bootstrap_ci.py       -> extract I3D features once per policy, restrict
#      to the shared matched id set, bootstrap FVD(policy) - FVD(always_notta).
#
# VBENCH_SUBDIR=vbench_results_geneval so the VBench router trains on the
# corrected generated-only VBench scores (same as run_router_fvd / geneval).
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   sbatch sweep_experiment/sbatch/run_router_fvd_bootstrap_ci.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"

SERIES_ROOT="${SERIES_ROOT:-sweep_experiment/results/panda_ood_budget_1000v_preview}"
FEATURE_DATE="${FEATURE_DATE:-sweep_experiment/reports/per_video_analysis/2026-07-12}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_ood_budget_1000v_preview_longcat.npz}"
MATCHED_ROOT="${MATCHED_ROOT:-sweep_experiment/reports/budget_oracle_fvd_1000v_preview}"
ROUTER_ROOT="${ROUTER_ROOT:-${MATCHED_ROOT}/routers}"
OUTPUT_DIR="${OUTPUT_DIR:-sweep_experiment/reports/per_video_analysis/2026-07-31/fvd_bootstrap_routers}"

METRICS="${METRICS:-psnr vbench}"
BLOCKS="${BLOCKS:-A+B+C}"
ACTIONS="${ACTIONS:-12 13}"
N_BOOT="${N_BOOT:-2000}"
BASELINE="${BASELINE:-always_notta}"
BUILD_ROUTERS="${BUILD_ROUTERS:-1}"   # set 0 to reuse existing router dirs

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
mkdir -p sweep_experiment/slurm_log "${OUTPUT_DIR}" "${ROUTER_ROOT}"

if [ ! -f "${GT_CACHE}" ]; then
  echo "ERROR: GT cache missing: ${GT_CACHE}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Stage 1: compose the deployable router policy dirs (matched-N vs NO-TTA).
# min-videos 256 is fine (N~898); no --force needed.
# ---------------------------------------------------------------------------
if [ "${BUILD_ROUTERS}" = "1" ]; then
  echo "============================================================"
  echo "Stage 1: build router policy dirs -> ${ROUTER_ROOT}"
  echo "  metrics=${METRICS} blocks=${BLOCKS} actions=${ACTIONS}"
  echo "  VBENCH_SUBDIR=${VBENCH_SUBDIR}"
  echo "============================================================"
  python3 scripts/build_router_fvd_dirs.py \
    --series-root "${SERIES_ROOT}" \
    --feature-date "${FEATURE_DATE}" \
    --gt-cache "${GT_CACHE}" \
    --output-root "${ROUTER_ROOT}" \
    --metrics ${METRICS} \
    --blocks ${BLOCKS} \
    --actions ${ACTIONS}
else
  echo "BUILD_ROUTERS=0 -> reusing existing router dirs under ${ROUTER_ROOT}"
fi

# ---------------------------------------------------------------------------
# Stage 2: paired bootstrap CI over NO-TTA + fixed + oracle + the 4 routers.
# Router dir names come from build_router_fvd_dirs.py:
#   router_{metric}_{block_without_+}_{actions}act  (e.g. router_psnr_ABC_12act)
# ---------------------------------------------------------------------------
POLICY_FLAGS=(
  --policy "always_notta:${MATCHED_ROOT}/matched/always_notta/videos"
  --policy "fixed:${MATCHED_ROOT}/matched/fixed_S10_LR5e3/videos"
  --policy "oracle_psnr:${MATCHED_ROOT}/oracle_best_psnr/videos"
)
# Discover every router dir that actually got built (has a videos/ subdir).
for d in "${ROUTER_ROOT}"/router_*; do
  [ -d "${d}/videos" ] || continue
  name="$(basename "${d}")"
  POLICY_FLAGS+=(--policy "${name}:${d}/videos")
done

echo "============================================================"
echo "Stage 2: FVD bootstrap CI (baseline=${BASELINE}, n_boot=${N_BOOT})"
echo "  output : ${OUTPUT_DIR}"
echo "  policies:"
i=0
while [ $i -lt ${#POLICY_FLAGS[@]} ]; do
  [ "${POLICY_FLAGS[$i]}" = "--policy" ] && echo "      ${POLICY_FLAGS[$((i+1))]}"
  i=$((i+1))
done
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
