#!/bin/bash
# ============================================================================
# EXP2 — AdaSteer vector-placement ablation (2026-08-04 literature memo)
#
# Question: does injecting the learned δ into the MID-LATE RESIDUAL STREAM
# (the controllable band per the steering literature) beat injecting it into
# the global timestep/AdaLN embedding (original AdaSteer)?
#
# Two arms, IDENTICAL config except --delta-placement:
#   ADA_ADALN  : placement=adaln    (original global AdaSteer, control)
#   ADA_RESID  : placement=residual (auto ~55-80% depth band)
# Optionally a third arm ADA_RESID_MID (a single ~60%-depth block) if you pass
# RESID_MID_BLOCKS.
#
# Same OOD-stratified preview pool + geometry as the budget grid / best-of-k,
# so per-video pixel + VBench deltas are directly comparable to existing NOTTA.
#
# Usage (from repo root on the cluster):
#   bash delta_experiment/sbatch/submit_placement_ablation.sh            # submit
#   DRY_RUN=1 bash delta_experiment/sbatch/submit_placement_ablation.sh  # print only
# ============================================================================
set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SBATCH="${PROJECT_ROOT}/sweep_experiment/sbatch/run_sweep.sbatch"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"   # Torch HPC project account (required)

# --- shared config (held fixed across arms) --------------------------------
SERIES_NAME="${SERIES_NAME:-placement_ablation_panda}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
MAX_VIDEOS="${MAX_VIDEOS:-80}"        # small OOD-stratified subset (first N = stratified)
SEED="${SEED:-42}"

# geometry — MUST match the budget preview series (cond=14, frames=28, gsf=48)
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_FRAMES="${NUM_FRAMES:-28}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"

# AdaSteer hyperparams (S10-style; identical across arms)
DELTA_STEPS="${DELTA_STEPS:-10}"
DELTA_LR="${DELTA_LR:-1e-3}"

# per-video VBench on generated-only clips; pixel metrics always computed.
COMPUTE_VBENCH="${COMPUTE_VBENCH:-1}"
COMPUTE_FVD="${COMPUTE_FVD:-0}"       # N too small for reliable online FVD; score later on composed dirs
NO_SAVE_VIDEOS="${NO_SAVE_VIDEOS:-0}" # KEEP clips so we can compose FVD dirs afterwards

RESID_MID_BLOCKS="${RESID_MID_BLOCKS:-}"   # set e.g. "20" to add a single-block arm

echo "============================================================"
echo "EXP2 placement ablation"
echo "  account     : ${ACCOUNT}"
echo "  series      : ${SERIES_NAME}"
echo "  data_dir    : ${DATA_DIR}"
echo "  max_videos  : ${MAX_VIDEOS}   seed=${SEED}"
echo "  geometry    : cond=${NUM_COND_FRAMES} frames=${NUM_FRAMES} gsf=${GEN_START_FRAME}"
echo "  delta       : steps=${DELTA_STEPS} lr=${DELTA_LR}"
echo "  vbench=${COMPUTE_VBENCH} fvd=${COMPUTE_FVD} save_videos=$([ "${NO_SAVE_VIDEOS}" = 1 ] && echo no || echo yes)"
echo "============================================================"

submit_arm () {
  local run_id="$1" placement="$2" residual_blocks="$3"
  local -a envs=(
    "METHOD=delta_a" "SERIES_NAME=${SERIES_NAME}" "RUN_ID=${run_id}"
    "DATA_DIR=${DATA_DIR}" "CHECKPOINT_DIR=${CHECKPOINT_DIR}"
    "MAX_VIDEOS=${MAX_VIDEOS}" "SEED=${SEED}"
    "NUM_COND_FRAMES=${NUM_COND_FRAMES}" "NUM_FRAMES=${NUM_FRAMES}"
    "GEN_START_FRAME=${GEN_START_FRAME}"
    "DELTA_STEPS=${DELTA_STEPS}" "DELTA_LR=${DELTA_LR}"
    "DELTA_PLACEMENT=${placement}" "RESIDUAL_BLOCKS=${residual_blocks}"
    "COMPUTE_VBENCH=${COMPUTE_VBENCH}" "COMPUTE_FVD=${COMPUTE_FVD}"
    "NO_SAVE_VIDEOS=${NO_SAVE_VIDEOS}"
  )
  echo "-> arm ${run_id}: placement=${placement} residual_blocks=${residual_blocks:-<auto>}"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "   DRY: ${envs[*]} sbatch --account=${ACCOUNT} --job-name=exp2_${run_id} ${SBATCH}"
  else
    sbatch --account="${ACCOUNT}" \
           --export=ALL,"$(IFS=,; echo "${envs[*]}")" \
           --job-name="exp2_${run_id}" "${SBATCH}"
  fi
}

submit_arm "ADA_ADALN" "adaln"    ""
submit_arm "ADA_RESID" "residual" ""
if [ -n "${RESID_MID_BLOCKS}" ]; then
  submit_arm "ADA_RESID_MID" "residual" "${RESID_MID_BLOCKS}"
fi

echo ""
echo "Submitted. After completion, compare per-video ΔTTA-NOTTA between arms:"
echo "  python3 scripts/analyze_population_effect.py \\"
echo "    --series-root sweep_experiment/results/${SERIES_NAME} \\"
echo "    --notta-run NOTTA --tta-run ADA_RESID \\"
echo "    --out sweep_experiment/reports/per_video_analysis/popeffect_placement_resid.json"
echo "  (repeat with --tta-run ADA_ADALN; NOTTA must be present/symlinked in the series)"
