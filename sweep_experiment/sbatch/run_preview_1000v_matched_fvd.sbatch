#!/bin/bash
#SBATCH --job-name=prev1k_fvd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/prev1k_matched_fvd_%j.out
#SBATCH --error=sweep_experiment/slurm_log/prev1k_matched_fvd_%j.err

# ============================================================================
# Matched FVD on the 1000v OOD-preview pool (panda_ood_budget_1000v_preview).
#
# Three policies on the SAME ~998 video IDs, ONE eval_fvd protocol, ONE GT
# cache built from the preview pool's own ground-truth continuation frames:
#   * always_notta        — in-pool NOTTA run
#   * fixed_S10_LR5e3      — deployable fixed AdaSteer config
#   * oracle_best_psnr     — per-video best-PSNR config (upper bound)
#
# Why a dedicated sbatch (not run_pilot_matched_fvd.sbatch):
#   1. NOTTA lives IN this series (panda_ood_budget_1000v_preview/NOTTA), not
#      in panda_1000v_standard — so --baseline-series-root = the preview root.
#   2. The GT distribution must be the PREVIEW pool's GT, not panda_1000_480p.
#      panda_1000_longcat.npz is the WRONG reference for this pool.
#   3. Oracle output-root is nested 2 levels deep so the baseline script's
#      first oracle-dir fallback (reports/budget_oracle_fvd/) does NOT collide
#      with the 200v pilot's oracle dir; the 3rd fallback (manifest.parent/
#      videos) then correctly resolves the preview oracle.
#
# Geometry (must match the sweep + saved mp4 tail): gen_start=48, cond=14,
# gen=14 -> generated frames = video[48:62]; saved mp4 = cond(14)+gen(14)=28,
# eval_fvd scores tail [14:28]. N=998 >= 256 so covariance is well-sampled.
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   sbatch sweep_experiment/sbatch/run_preview_1000v_matched_fvd.sbatch
#
# Skip GT-cache rebuild (cache already exists):
#   SKIP_GT_CACHE=1 sbatch ... run_preview_1000v_matched_fvd.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
SERIES_ROOT="${SERIES_ROOT:-sweep_experiment/results/panda_ood_budget_1000v_preview}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_ood_budget_1000v_preview_longcat.npz}"

# Nested 2 levels under reports/ to dodge the pilot budget_oracle_fvd/ fallback.
ORACLE_ROOT="${ORACLE_ROOT:-sweep_experiment/reports/budget_oracle_fvd_1000v_preview}"
MATCHED_ROOT="${MATCHED_ROOT:-${ORACLE_ROOT}/matched}"

FIXED_RUN="${FIXED_RUN:-S10_LR5e3}"
POLICIES="${POLICIES:-always_notta fixed_S10_LR5e3 oracle_best_psnr}"
MIN_LINKED="${MIN_LINKED:-256}"

GEN_START_FRAME="${GEN_START_FRAME:-48}"
NUM_GEN_FRAMES="${NUM_GEN_FRAMES:-14}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"

SKIP_GT_CACHE="${SKIP_GT_CACHE:-0}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "${CONDA_ENV}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log "$(dirname "${GT_CACHE}")"

echo "============================================================"
echo "Preview 1000v matched FVD"
echo "  data_dir     : ${DATA_DIR}"
echo "  series_root  : ${SERIES_ROOT}"
echo "  gt_cache     : ${GT_CACHE}"
echo "  oracle_root  : ${ORACLE_ROOT}"
echo "  matched_root : ${MATCHED_ROOT}"
echo "  fixed_run    : ${FIXED_RUN}"
echo "  policies     : ${POLICIES}"
echo "  geometry     : gen_start=${GEN_START_FRAME} cond=${NUM_COND_FRAMES} gen=${NUM_GEN_FRAMES}"
echo "============================================================"

# ---- Step 1: preview-pool GT cache (aligned to generated tail) -------------
if [ "${SKIP_GT_CACHE}" = "1" ] && [ -f "${GT_CACHE}" ]; then
  echo "[1/3] SKIP GT cache build (exists: ${GT_CACHE})"
else
  echo "[1/3] Building preview GT cache..."
  python sweep_experiment/scripts/precompute_gt_features.py \
    --data-dir "${DATA_DIR}" \
    --output "${GT_CACHE}" \
    --mode longcat \
    --gen-start-frame "${GEN_START_FRAME}" \
    --num-gen-frames "${NUM_GEN_FRAMES}" \
    --num-cond-frames "${NUM_COND_FRAMES}" \
    --device cuda
fi

# ---- Step 2: per-video best-PSNR oracle symlink dir ------------------------
echo "[2/3] Building oracle_best_psnr policy dir..."
python sweep_experiment/scripts/build_budget_oracle_policy_dirs.py \
  --series-root "${SERIES_ROOT}" \
  --output-root "${ORACLE_ROOT}" \
  --clean

# ---- Step 3: matched FVD for notta / fixed / oracle vs the SAME cache -------
echo "[3/3] Running matched FVD baselines..."
python sweep_experiment/scripts/run_pilot_matched_fvd_baselines.py \
  --series-root "${SERIES_ROOT}" \
  --baseline-series-root "${SERIES_ROOT}" \
  --oracle-manifest "${ORACLE_ROOT}/oracle_best_psnr/manifest.json" \
  --output-root "${MATCHED_ROOT}" \
  --gt-cache "${GT_CACHE}" \
  --fixed-run "${FIXED_RUN}" \
  --min-linked "${MIN_LINKED}" \
  --policies ${POLICIES}

echo ""
echo "Done. Summary:"
echo "  cat ${MATCHED_ROOT}/pilot_matched_fvd_summary.md"
echo "  jq '.fvd,.num_valid_pairs' ${MATCHED_ROOT}/always_notta/fvd.json"
echo "  jq '.fvd,.num_valid_pairs' ${MATCHED_ROOT}/fixed_S10_LR5e3/fvd.json"
echo "  jq '.fvd,.num_valid_pairs' ${ORACLE_ROOT}/oracle_best_psnr/fvd.json"
