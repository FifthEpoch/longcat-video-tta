#!/bin/bash
#SBATCH --job-name=tta_sweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=sweep_experiment/slurm_log/%x_%j.out
#SBATCH --error=sweep_experiment/slurm_log/%x_%j.err

# ============================================================================
# Unified TTA Sweep Template
#
# This sbatch script handles all 5 TTA methods via the METHOD env var:
#   METHOD = full | lora | delta_a | delta_b | delta_c
#
# All hyperparameters are passed via environment variables.
# The sweep runner (run_sweep.py) sets these before submission.
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:-${SCRATCH_BASE}/longcat-video-tta/datasets/panda_100_480p}"

# Method and run identification
METHOD="${METHOD:?ERROR: METHOD env var is required (full|lora|delta_a|delta_b|delta_c)}"
RUN_ID="${RUN_ID:?ERROR: RUN_ID env var is required}"
SERIES_NAME="${SERIES_NAME:-sweep}"

# Output directory: sweep_experiment/results/<series_name>/<run_id>
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/sweep_experiment/results/${SERIES_NAME}/${RUN_ID}}"

# ============================================================================
# Common hyperparameters (with sensible defaults)
# ============================================================================
NUM_COND_FRAMES="${NUM_COND_FRAMES:-2}"
NUM_FRAMES="${NUM_FRAMES:-16}"
GEN_START_FRAME="${GEN_START_FRAME:-32}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-${GEN_START_FRAME}}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-${NUM_COND_FRAMES}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
RESOLUTION="${RESOLUTION:-480p}"
SEED="${SEED:-42}"
MAX_VIDEOS="${MAX_VIDEOS:-100}"
BATCH_VIDEOS="${BATCH_VIDEOS:-1}"

# Early stopping (shared by all methods)
ES_DISABLE="${ES_DISABLE:-}"
ES_CHECK_EVERY="${ES_CHECK_EVERY:-5}"
ES_PATIENCE="${ES_PATIENCE:-3}"
ES_ANCHOR_SIGMAS="${ES_ANCHOR_SIGMAS:-0.25,0.5,0.75}"
ES_NOISE_DRAWS="${ES_NOISE_DRAWS:-2}"
ES_STRATEGY="${ES_STRATEGY:-patience}"
ES_HOLDOUT_FRACTION="${ES_HOLDOUT_FRACTION:-0.25}"

# ============================================================================
# Method-specific hyperparameters (only used by their respective methods)
# ============================================================================
# Full / LoRA shared
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
NUM_STEPS="${NUM_STEPS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"

# Full-model specific
OPTIMIZER="${OPTIMIZER:-sgd}"

# LoRA specific
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
TARGET_MODULES="${TARGET_MODULES:-qkv,proj}"
TARGET_FFN="${TARGET_FFN:-}"
SKIP_GENERATION="${SKIP_GENERATION:-}"

# Delta A/B/C shared
DELTA_STEPS="${DELTA_STEPS:-20}"
DELTA_LR="${DELTA_LR:-1e-3}"

# Delta-B specific
NUM_GROUPS="${NUM_GROUPS:-4}"

# Delta-C specific
DELTA_MODE="${DELTA_MODE:-per_channel}"

# ============================================================================
echo "=============================================================================="
echo "TTA Sweep: ${METHOD} / ${RUN_ID}"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Series       : ${SERIES_NAME}"
echo "Method       : ${METHOD}"
echo "Run ID       : ${RUN_ID}"
echo "Start time   : $(date)"
echo "=============================================================================="
echo "Common config:"
echo "  Cond frames    : ${NUM_COND_FRAMES}"
echo "  Num frames     : ${NUM_FRAMES}"
echo "  Gen start      : ${GEN_START_FRAME}"
echo "  Infer steps    : ${NUM_INFERENCE_STEPS}"
echo "  Guidance       : ${GUIDANCE_SCALE}"
echo "  Max videos     : ${MAX_VIDEOS}"
echo "  Data dir       : ${DATA_DIR}"
echo "  Output dir     : ${OUTPUT_DIR}"

# ============================================================================
# Environment Setup
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "Activated conda: $CONDA_ENV"
else
    echo "ERROR: Conda environment not found at $CONDA_ENV" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

cd "$PROJECT_ROOT"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "sweep_experiment/slurm_log"

# ============================================================================
# GPU Info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Build early stopping flags
# ============================================================================
ES_FLAGS=""
if [ -n "${ES_DISABLE}" ]; then
    ES_FLAGS="${ES_FLAGS} --es-disable"
fi
ES_FLAGS="${ES_FLAGS} --es-check-every ${ES_CHECK_EVERY}"
ES_FLAGS="${ES_FLAGS} --es-patience ${ES_PATIENCE}"
ES_FLAGS="${ES_FLAGS} --es-anchor-sigmas ${ES_ANCHOR_SIGMAS}"
ES_FLAGS="${ES_FLAGS} --es-noise-draws ${ES_NOISE_DRAWS}"
ES_FLAGS="${ES_FLAGS} --es-strategy ${ES_STRATEGY}"
ES_FLAGS="${ES_FLAGS} --es-holdout-fraction ${ES_HOLDOUT_FRACTION}"

# ============================================================================
# Dispatch to the appropriate method script
# ============================================================================
case "${METHOD}" in

  full)
    echo "Method-specific config:"
    echo "  Learning rate  : ${LEARNING_RATE}"
    echo "  Num steps      : ${NUM_STEPS}"
    echo "  Warmup steps   : ${WARMUP_STEPS}"
    echo "=============================================================================="

    EXTRA_FLAGS=""
    if [ -n "${SKIP_GENERATION}" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --skip-generation"
    fi

    python lora_experiment/scripts/run_full_tta.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --learning-rate "${LEARNING_RATE}" \
        --num-steps "${NUM_STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --optimizer "${OPTIMIZER}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        --batch-videos "${BATCH_VIDEOS}" \
        ${EXTRA_FLAGS} \
        ${ES_FLAGS}
    ;;

  lora)
    echo "Method-specific config:"
    echo "  LoRA rank      : ${LORA_RANK}"
    echo "  LoRA alpha     : ${LORA_ALPHA}"
    echo "  Target modules : ${TARGET_MODULES}"
    echo "  Target FFN     : ${TARGET_FFN:-no}"
    echo "  Learning rate  : ${LEARNING_RATE}"
    echo "  Num steps      : ${NUM_STEPS}"
    echo "  Warmup steps   : ${WARMUP_STEPS}"
    echo "=============================================================================="

    EXTRA_FLAGS=""
    if [ -n "${TARGET_FFN}" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --target-ffn"
    fi
    if [ -n "${SKIP_GENERATION}" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --skip-generation"
    fi

    python lora_experiment/scripts/run_lora_tta.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}" \
        --target-modules "${TARGET_MODULES}" \
        --learning-rate "${LEARNING_RATE}" \
        --num-steps "${NUM_STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        --batch-videos "${BATCH_VIDEOS}" \
        ${EXTRA_FLAGS} \
        ${ES_FLAGS}
    ;;

  delta_a)
    echo "Method-specific config:"
    echo "  Delta LR       : ${DELTA_LR}"
    echo "  Delta steps    : ${DELTA_STEPS}"
    echo "=============================================================================="

    python delta_experiment/scripts/run_delta_a.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --delta-steps "${DELTA_STEPS}" \
        --delta-lr "${DELTA_LR}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        --batch-videos "${BATCH_VIDEOS}" \
        ${ES_FLAGS}
    ;;

  delta_b)
    echo "Method-specific config:"
    echo "  Delta LR       : ${DELTA_LR}"
    echo "  Delta steps    : ${DELTA_STEPS}"
    echo "  Num groups     : ${NUM_GROUPS}"
    echo "=============================================================================="

    python delta_experiment/scripts/run_delta_b.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --delta-steps "${DELTA_STEPS}" \
        --delta-lr "${DELTA_LR}" \
        --num-groups "${NUM_GROUPS}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        ${ES_FLAGS}
    ;;

  delta_c)
    echo "Method-specific config:"
    echo "  Delta LR       : ${DELTA_LR}"
    echo "  Delta steps    : ${DELTA_STEPS}"
    echo "  Delta mode     : ${DELTA_MODE}"
    echo "=============================================================================="

    python delta_experiment/scripts/run_delta_c.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${MAX_VIDEOS}" \
        --delta-steps "${DELTA_STEPS}" \
        --delta-lr "${DELTA_LR}" \
        --delta-mode "${DELTA_MODE}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --tta-total-frames "${TTA_TOTAL_FRAMES}" \
        --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}" \
        ${ES_FLAGS}
    ;;

  *)
    echo "ERROR: Unknown METHOD '${METHOD}'. Use: full|lora|delta_a|delta_b|delta_c" >&2
    exit 1
    ;;
esac

echo ""
echo "=============================================================================="
echo "TTA Sweep Complete: ${METHOD} / ${RUN_ID}"
echo "=============================================================================="
echo "End time: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "=============================================================================="
