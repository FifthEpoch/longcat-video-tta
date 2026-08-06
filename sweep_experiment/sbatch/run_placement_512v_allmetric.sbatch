#!/bin/bash
#SBATCH --job-name=placement_allmetric
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=06:00:00
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/placement_allmetric_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/placement_allmetric_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# ============================================================================
# EXP2 consolidated matched-N all-metric table (NO-TTA vs adaln vs residual).
#
# Builds ONE table with PSNR/SSIM/LPIPS + FVD + all 7 VBench++ dims (+ overall),
# every per-video metric on the SAME common video set across all three methods.
#
# Method:
#   1) For NOTTA, ADA_ADALN, ADA_RESID: trim cond frames on the FVD POLICY dirs
#      (<FVD_ROOT>/<M>/videos = exactly the reliable-N=500 common clips, symlinked
#      as <id>.mp4) -> videos_geneval, then compute 7-dim VBench -> vbench_results_geneval.
#      This ALSO fills the previously-missing NO-TTA gen-only VBench.
#   2) build_placement_allmetric_table.py intersects to the common id set S'
#      (pixel + all 7 VBench in all 3 methods), reads pixel from the SOURCE run
#      dirs and FVD from fvd.json (recomputing on S' only if S' < the FVD N).
#
# Submit (repo root on cluster):
#   sbatch sweep_experiment/sbatch/run_placement_512v_allmetric.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"

# FVD policy root produced by run_placement_arms_fvd.sbatch (has <M>/{videos,fvd.json})
FVD_ROOT="${FVD_ROOT:-sweep_experiment/reports/budget_oracle_fvd_1000v_preview/placement_arms_placement_ablation_panda_512v}"
ARMS="${ARMS:-NOTTA ADA_ADALN ADA_RESID}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_GEN_FRAMES="${NUM_GEN_FRAMES:-14}"
ALL_DIMS="${ALL_DIMS:-subject_consistency background_consistency aesthetic_quality motion_smoothness dynamic_degree imaging_quality temporal_flickering}"
MODE="${MODE:-custom_input}"

# SOURCE run dirs for per-video pixel metrics
PIXEL_NOTTA="${PIXEL_NOTTA:-sweep_experiment/results/panda_ood_budget_1000v_preview/NOTTA}"
PIXEL_ADALN="${PIXEL_ADALN:-sweep_experiment/results/placement_ablation_panda_512v/ADA_ADALN}"
PIXEL_RESID="${PIXEL_RESID:-sweep_experiment/results/placement_ablation_panda_512v/ADA_RESID}"

GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_ood_budget_1000v_preview_longcat.npz}"
OUT_TABLE="${OUT_TABLE:-sweep_experiment/reports/paper_tables/2026-08-06_placement_allmetric_matchedN.md}"

LONGCAT_PY="${LONGCAT_PY:-${SCRATCH_BASE}/conda-envs/longcat/bin/python}"
VBENCH_ENV_NAME="${VBENCH_ENV_NAME:-vbench-backfill}"
VBENCH_PY="${VBENCH_PY:-${SCRATCH_BASE}/conda-envs/${VBENCH_ENV_NAME}/bin/python}"
CACHE_DIR="${VBENCH_CACHE_DIR:-${SCRATCH_BASE}/vbench-cache}"

export HF_HOME="${CACHE_DIR}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${CACHE_DIR}/torch"
export VBENCH_CACHE_DIR="${CACHE_DIR}"
export TMPDIR="${SCRATCH_BASE}/tmp"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${TMPDIR}" \
    "${PROJECT_ROOT}/sweep_experiment/slurm_log"

cd "${PROJECT_ROOT}"
unset PYTHONHOME PYTHONPATH

echo "=============================================================="
echo "EXP2 consolidated matched-N all-metric table"
echo "  fvd_root   : ${FVD_ROOT}"
echo "  arms       : ${ARMS}"
echo "  cond/gen   : ${NUM_COND_FRAMES}/${NUM_GEN_FRAMES}"
echo "  pixel src  : NOTTA=${PIXEL_NOTTA}"
echo "               ADALN=${PIXEL_ADALN}"
echo "               RESID=${PIXEL_RESID}"
echo "  gt_cache   : ${GT_CACHE}"
echo "  out        : ${OUT_TABLE}"
echo "=============================================================="

for py in "${LONGCAT_PY}" "${VBENCH_PY}"; do
  if [ ! -x "${py}" ]; then echo "ERROR: python not found: ${py}" >&2; exit 2; fi
done

# --- 1) gen-only 7-dim VBench on each policy dir (the exact N=500 common clips) ---
for arm in ${ARMS}; do
  METHOD_DIR="${PROJECT_ROOT}/${FVD_ROOT}/${arm}"
  if [ ! -d "${METHOD_DIR}/videos" ]; then
    echo "ERROR: missing ${METHOD_DIR}/videos (run the FVD scorer first)" >&2
    exit 3
  fi
  echo ""
  echo "=== arm ${arm} ==="
  echo "[1/2] trim cond frames -> videos_geneval"
  "${LONGCAT_PY}" scripts/make_geneval_clips.py \
      --method-dir "${METHOD_DIR}" --num-cond-frames "${NUM_COND_FRAMES}"
  echo "[2/2] 7-dim VBench on gen-only clips -> vbench_results_geneval"
  "${VBENCH_PY}" scripts/run_vbench_backfill.py \
      --method-dir "${METHOD_DIR}" --dimensions ${ALL_DIMS} \
      --mode "${MODE}" --videos-subdir videos_geneval --out-subdir vbench_results_geneval
done

# --- 2) assemble the matched-N table ---
echo ""
echo "=== building consolidated matched-N table ==="
"${LONGCAT_PY}" scripts/build_placement_allmetric_table.py \
    --fvd-root "${FVD_ROOT}" \
    --pixel-notta "${PIXEL_NOTTA}" \
    --pixel-adaln "${PIXEL_ADALN}" \
    --pixel-resid "${PIXEL_RESID}" \
    --vbench-subdir vbench_results_geneval \
    --eval-fvd --gt-cache "${GT_CACHE}" \
    --num-cond-frames "${NUM_COND_FRAMES}" --num-gen-frames "${NUM_GEN_FRAMES}" \
    --python "${LONGCAT_PY}" \
    --out "${OUT_TABLE}"

echo ""
echo "Done. Table: ${OUT_TABLE}"
echo "  cat ${OUT_TABLE}"
