#!/bin/bash
#SBATCH --job-name=placement_vbench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/placement_vbench_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/placement_vbench_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# ============================================================================
# EXP2 all-metrics close-out (VBench++): for each placement arm, trim the
# conditioning frames to gen-only clips, then compute ALL 7 VBench dims on the
# gen-only clips into vbench_results_geneval/ (uncontaminated — matches the
# 999v geneval fix). Pairs with run_placement_arms_fvd.sbatch (FVD) so EXP2 is
# evaluated on the FULL metric set, not just PSNR/SSIM/LPIPS.
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   sbatch sweep_experiment/sbatch/run_placement_arms_vbench_geneval.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SERIES="${SERIES:-sweep_experiment/results/placement_ablation_panda}"
ARMS="${ARMS:-ADA_ADALN ADA_RESID}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
ALL_DIMS="${ALL_DIMS:-subject_consistency background_consistency aesthetic_quality motion_smoothness dynamic_degree imaging_quality temporal_flickering}"
MODE="${MODE:-custom_input}"

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

echo "============================================================"
echo "EXP2 VBench++ (gen-only, 7-dim) for placement arms"
echo "  series      : ${SERIES}"
echo "  arms        : ${ARMS}"
echo "  cond frames : ${NUM_COND_FRAMES}"
echo "  dims        : ${ALL_DIMS}"
echo "  longcat py  : ${LONGCAT_PY}"
echo "  vbench  py  : ${VBENCH_PY}"
echo "============================================================"

for py in "${LONGCAT_PY}" "${VBENCH_PY}"; do
  if [ ! -x "${py}" ]; then
    echo "ERROR: python not found: ${py}" >&2
    exit 2
  fi
done

for arm in ${ARMS}; do
  METHOD_DIR="${PROJECT_ROOT}/${SERIES}/${arm}"
  if [ ! -d "${METHOD_DIR}" ]; then
    echo "WARN: missing ${METHOD_DIR} — skip" >&2
    continue
  fi
  echo ""
  echo "=== arm ${arm} ==="
  echo "[1/2] trim conditioning frames -> videos_geneval (longcat env)"
  "${LONGCAT_PY}" scripts/make_geneval_clips.py \
      --method-dir "${METHOD_DIR}" \
      --num-cond-frames "${NUM_COND_FRAMES}"

  echo "[2/2] VBench 7-dim on gen-only clips -> vbench_results_geneval (vbench env)"
  "${VBENCH_PY}" scripts/run_vbench_backfill.py \
      --method-dir "${METHOD_DIR}" \
      --dimensions ${ALL_DIMS} \
      --mode "${MODE}" \
      --videos-subdir videos_geneval \
      --out-subdir vbench_results_geneval
done

echo ""
echo "=============================================================================="
echo "Done. Re-run the all-metric per-video analysis with the gen-only VBench:"
echo "  VBENCH_SUBDIR=vbench_results_geneval python3 scripts/analyze_population_effect.py \\"
echo "    --series-root ${SERIES} --notta-run ADA_ADALN --tta-run ADA_RESID \\"
echo "    --out sweep_experiment/reports/per_video_analysis/popeffect_resid_vs_adaln_allmetrics.json"
echo "=============================================================================="
