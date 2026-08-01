#!/bin/bash
#SBATCH --job-name=fvd_decomp_ci
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/fvd_decomp_ci_%j.out
#SBATCH --error=sweep_experiment/slurm_log/fvd_decomp_ci_%j.err

# ============================================================================
# DECOMPOSE the real oracle FVD headroom (ΔFVD ~ -10 vs NO-TTA, CI excludes 0).
#
# We know the PSNR-oracle composition lowers pooled FVD, but per-video PSNR
# selection is max-over-noise and the routers were FVD-null, so the MECHANISM is
# unknown. This job tells (A) vs (B) vs (C):
#   (A) a single config is more in-distribution   -> some always_<RUN> ~ oracle
#   (B) mixture diversity                          -> random_mixture ~ oracle
#   (C) genuine per-video targeting                -> only oracle hits it
#
# Stage 1 (CPU-fast): build_fvd_decomposition_dirs.py composes one policy dir per
#   single config (always_<RUN>) + a seeded random_mixture, matched to the NOTTA
#   common set, metric-fingerprint clip resolution (bijective).
# Stage 2 (GPU): fvd_bootstrap_ci.py scores NO-TTA + oracle + every single config
#   + random_mixture under the SAME matched-GT-cache paired bootstrap.
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   sbatch sweep_experiment/sbatch/run_fvd_decomposition_ci.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"

SERIES_ROOT="${SERIES_ROOT:-sweep_experiment/results/panda_ood_budget_1000v_preview}"
FEATURE_DATE="${FEATURE_DATE:-sweep_experiment/reports/per_video_analysis/2026-07-12}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_ood_budget_1000v_preview_longcat.npz}"
MATCHED_ROOT="${MATCHED_ROOT:-sweep_experiment/reports/budget_oracle_fvd_1000v_preview}"
DECOMP_ROOT="${DECOMP_ROOT:-${MATCHED_ROOT}/decomp}"
OUTPUT_DIR="${OUTPUT_DIR:-sweep_experiment/reports/per_video_analysis/2026-08-01/fvd_decomposition}"

SEED="${SEED:-42}"
N_MIXTURE="${N_MIXTURE:-1}"
N_BOOT="${N_BOOT:-2000}"
BASELINE="${BASELINE:-always_notta}"
BUILD="${BUILD:-1}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log "${OUTPUT_DIR}" "${DECOMP_ROOT}"

if [ ! -f "${GT_CACHE}" ]; then
  echo "ERROR: GT cache missing: ${GT_CACHE}" >&2
  exit 2
fi

# --- Stage 1: build single-config + random-mixture policy dirs (CPU) ---
if [ "${BUILD}" = "1" ]; then
  echo "============================================================"
  echo "Stage 1: build decomposition policy dirs -> ${DECOMP_ROOT}"
  echo "============================================================"
  python3 scripts/build_fvd_decomposition_dirs.py \
    --series-root "${SERIES_ROOT}" \
    --feature-date "${FEATURE_DATE}" \
    --output-root "${DECOMP_ROOT}" \
    --seed "${SEED}" \
    --n-mixture "${N_MIXTURE}"
else
  echo "BUILD=0 -> reusing existing dirs under ${DECOMP_ROOT}"
fi

# --- Stage 2: paired FVD bootstrap over all decomposition policies ---
POLICY_FLAGS=(
  --policy "always_notta:${MATCHED_ROOT}/matched/always_notta/videos"
  --policy "oracle_psnr:${MATCHED_ROOT}/oracle_best_psnr/videos"
)
for d in "${DECOMP_ROOT}"/always_* "${DECOMP_ROOT}"/random_mixture*; do
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
