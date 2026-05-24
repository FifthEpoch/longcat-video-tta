#!/bin/bash
#SBATCH --job-name=opensora_tta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=backbone_experiment/opensora/slurm_log/%x_%j.out
#SBATCH --error=backbone_experiment/opensora/slurm_log/%x_%j.err

# ============================================================================
# Open-Sora v2.0 TTA Experiment
#
# Dispatches to the appropriate TTA script based on METHOD env var:
#   METHOD = full | lora | delta_a
#
# All hyperparameters are passed via environment variables.
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
OPENSORA_ROOT="${OPENSORA_ROOT:-${SCRATCH_BASE}/longcat-video-tta/../Open-Sora-2.0}"
export OPENSORA_ROOT

# Open-Sora inference config
CONFIG_PATH="${CONFIG_PATH:-${OPENSORA_ROOT}/configs/diffusion/inference/256px.py}"

# Data directory
DATA_DIR="${DATA_DIR:-${SCRATCH_BASE}/longcat-video-tta/datasets/panda_100_480p}"

# Method and run identification
METHOD="${METHOD:?ERROR: METHOD env var is required (full|delta_a)}"
RUN_ID="${RUN_ID:?ERROR: RUN_ID env var is required}"
SERIES_NAME="${SERIES_NAME:-opensora_tta}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/backbone_experiment/opensora/results/${SERIES_NAME}/${RUN_ID}}"

# ============================================================================
# Common hyperparameters
# ============================================================================
NUM_COND_FRAMES="${NUM_COND_FRAMES:-5}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-33}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-5}"
GEN_START_FRAME="${GEN_START_FRAME:-33}"
NUM_GEN_FRAMES="${NUM_GEN_FRAMES:-129}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.5}"
GUIDANCE_IMG="${GUIDANCE_IMG:-3.0}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-455}"
SEED="${SEED:-42}"
MAX_VIDEOS="${MAX_VIDEOS:-100}"
PATCH_SIZE="${PATCH_SIZE:-2}"
SKIP_GENERATION="${SKIP_GENERATION:-}"

# Full-model specific
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_STEPS="${NUM_STEPS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
OPTIMIZER="${OPTIMIZER:-sgd}"

# LoRA specific
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-4.0}"

# Delta-A specific
DELTA_STEPS="${DELTA_STEPS:-20}"
DELTA_LR="${DELTA_LR:-1e-3}"

# ============================================================================
echo "=============================================================================="
echo "Open-Sora v2.0 TTA: ${METHOD} / ${RUN_ID}"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Series       : ${SERIES_NAME}"
echo "Method       : ${METHOD}"
echo "Run ID       : ${RUN_ID}"
echo "Start time   : $(date)"
echo "=============================================================================="
echo "Common config:"
echo "  Config path    : ${CONFIG_PATH}"
echo "  Cond frames    : ${NUM_COND_FRAMES}"
echo "  TTA frames     : ${TTA_TOTAL_FRAMES}"
echo "  Gen start      : ${GEN_START_FRAME}"
echo "  Gen frames     : ${NUM_GEN_FRAMES}"
echo "  Infer steps    : ${NUM_INFERENCE_STEPS}"
echo "  Guidance       : ${GUIDANCE_SCALE}"
echo "  Guidance img   : ${GUIDANCE_IMG}"
echo "  Resolution     : ${HEIGHT}x${WIDTH}"
echo "  Max videos     : ${MAX_VIDEOS}"
echo "  Data dir       : ${DATA_DIR}"
echo "  Output dir     : ${OUTPUT_DIR}"

# ============================================================================
# Environment Setup
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/opensora20"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "Activated conda: $CONDA_ENV"
else
    echo "ERROR: Conda environment not found at $CONDA_ENV" >&2
    exit 1
fi
# CRITICAL: unset PYTHONHOME and PYTHONPATH set by "module load anaconda3/..."
unset PYTHONHOME
unset PYTHONPATH

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

cd "$PROJECT_ROOT"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "backbone_experiment/opensora/slurm_log"

# ============================================================================
# GPU Info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Dispatch to the appropriate method script
# ============================================================================
COMMON_FLAGS="
    --config-path ${CONFIG_PATH}
    --data-dir ${DATA_DIR}
    --output-dir ${OUTPUT_DIR}
    --max-videos ${MAX_VIDEOS}
    --num-cond-frames ${NUM_COND_FRAMES}
    --tta-total-frames ${TTA_TOTAL_FRAMES}
    --tta-context-frames ${TTA_CONTEXT_FRAMES}
    --gen-start-frame ${GEN_START_FRAME}
    --num-gen-frames ${NUM_GEN_FRAMES}
    --num-inference-steps ${NUM_INFERENCE_STEPS}
    --guidance-scale ${GUIDANCE_SCALE}
    --guidance-img ${GUIDANCE_IMG}
    --height ${HEIGHT}
    --width ${WIDTH}
    --seed ${SEED}
    --patch-size ${PATCH_SIZE}
"

if [ -n "${SKIP_GENERATION}" ]; then
    COMMON_FLAGS="${COMMON_FLAGS} --skip-generation"
fi

case "${METHOD}" in

  full)
    echo "Method-specific config:"
    echo "  Learning rate  : ${LEARNING_RATE}"
    echo "  Num steps      : ${NUM_STEPS}"
    echo "  Warmup steps   : ${WARMUP_STEPS}"
    echo "  Optimizer      : ${OPTIMIZER}"
    echo "=============================================================================="

    python backbone_experiment/opensora/run_full_tta_opensora.py \
        ${COMMON_FLAGS} \
        --learning-rate "${LEARNING_RATE}" \
        --num-steps "${NUM_STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --optimizer "${OPTIMIZER}"
    ;;

  lora)
    echo "Method-specific config:"
    echo "  Learning rate  : ${LEARNING_RATE}"
    echo "  Num steps      : ${NUM_STEPS}"
    echo "  LoRA rank      : ${LORA_RANK}"
    echo "  LoRA alpha     : ${LORA_ALPHA}"
    echo "=============================================================================="

    python backbone_experiment/opensora/run_lora_tta_opensora.py \
        ${COMMON_FLAGS} \
        --learning-rate "${LEARNING_RATE}" \
        --num-steps "${NUM_STEPS}" \
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}" \
        --max-grad-norm "${MAX_GRAD_NORM}"
    ;;

  delta_a)
    echo "Method-specific config:"
    echo "  Delta LR       : ${DELTA_LR}"
    echo "  Delta steps    : ${DELTA_STEPS}"
    echo "=============================================================================="

    python backbone_experiment/opensora/run_delta_a_opensora.py \
        ${COMMON_FLAGS} \
        --delta-steps "${DELTA_STEPS}" \
        --delta-lr "${DELTA_LR}"
    ;;

  *)
    echo "ERROR: Unknown METHOD '${METHOD}'. Use: full|lora|delta_a" >&2
    exit 1
    ;;
esac

echo ""
echo "=============================================================================="
echo "Open-Sora v2.0 TTA Complete: ${METHOD} / ${RUN_ID}"
echo "=============================================================================="
echo "End time: $(date)"
echo "Results: ${OUTPUT_DIR}"
echo "=============================================================================="
