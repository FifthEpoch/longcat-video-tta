#!/bin/bash
#SBATCH --job-name=adb_inc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=sweep_experiment/slurm_log/%x_%j.out
#SBATCH --error=sweep_experiment/slurm_log/%x_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# ============================================================================
# Incremental AdaSteer budget job: S10_LR1e2 then S20_LR1e2 (same chunk).
#
# Phase 1 — train 10 steps, save delta, generate → S10 output dir
# Phase 2 — load delta, train 10 more steps, generate → S20 output dir
#
# Required env (set by submit_adasteer_budget_1000v_best_incremental.sh):
#   S10_OUTPUT_DIR, S20_OUTPUT_DIR, DATA_DIR, START_VIDEO_IDX, CHUNK_SIZE
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SWEEP_SBATCH="${PROJECT_ROOT}/sweep_experiment/sbatch/run_sweep.sbatch"

S10_OUT="${S10_OUTPUT_DIR:?S10_OUTPUT_DIR required}"
S20_OUT="${S20_OUTPUT_DIR:?S20_OUTPUT_DIR required}"
DELTA_CKPT="${S10_OUT}/delta_ckpt"
RUN_S10="${RUN_ID_S10:-S10_LR1e2}"
RUN_S20="${RUN_ID_S20:-S20_LR1e2}"

mkdir -p "${S10_OUT}" "${S20_OUT}" "${DELTA_CKPT}"

_run_phase() {
    local run_id="$1"
    local steps="$2"
    local out_dir="$3"
    local init_dir="${4:-}"
    local save_dir="${5:-}"

    export METHOD=delta_a
    export RUN_ID="${run_id}"
    export SERIES_NAME="${SERIES_NAME:-panda_ood_budget_1000v}"
    export DATA_DIR="${DATA_DIR:?DATA_DIR required}"
    export OUTPUT_DIR="${out_dir}"
    export MAX_VIDEOS="${MAX_VIDEOS:-1000}"
    export START_VIDEO_IDX="${START_VIDEO_IDX:-0}"
    export CHUNK_SIZE="${CHUNK_SIZE:-100}"
    export DELTA_STEPS="${steps}"
    export DELTA_LR="${DELTA_LR:-1.0e-2}"
    export NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
    export NUM_FRAMES="${NUM_FRAMES:-28}"
    export GEN_START_FRAME="${GEN_START_FRAME:-48}"
    export TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
    export TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
    export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
    export GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
    export RESOLUTION=480p
    export SEED=42
    export ES_DISABLE=1
    export COMPUTE_FVD=1
    export COMPUTE_FID=1
    export COMPUTE_VBENCH=0
    export NO_SAVE_VIDEOS="${NO_SAVE_VIDEOS:-1}"
    export CAPTION_GUARD_MODE=warn
    export FEATURE_FRAME_GUARD_MODE=warn
    export INITIAL_DELTA_DIR="${init_dir}"
    export SAVE_DELTA_DIR="${save_dir}"

    bash "${SWEEP_SBATCH}"
}

echo "============================================================"
echo "Incremental budget chunk ${CHUNK_IDX:-?}: ${RUN_S10} → ${RUN_S20}"
echo "  S10 out : ${S10_OUT}"
echo "  S20 out : ${S20_OUT}"
echo "  delta   : ${DELTA_CKPT}"
echo "============================================================"

echo ""
echo "=== Phase 1: ${RUN_S10} (10 steps, save delta) ==="
_run_phase "${RUN_S10}" 10 "${S10_OUT}" "" "${DELTA_CKPT}"

echo ""
echo "=== Phase 2: ${RUN_S20} (+10 steps from checkpoint) ==="
_run_phase "${RUN_S20}" 10 "${S20_OUT}" "${DELTA_CKPT}" ""

echo ""
echo "Done incremental chunk ${CHUNK_IDX:-?}"
