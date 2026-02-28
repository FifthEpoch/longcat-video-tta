#!/bin/bash
#SBATCH --job-name=cogvideo_tta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=backbone_experiment/cogvideo/slurm_log/cogvideo_tta_%j.out
#SBATCH --error=backbone_experiment/cogvideo/slurm_log/cogvideo_tta_%j.err

# ============================================================================
# CogVideoX-5B-I2V TTA: Full-model or Delta-A
#
# Dispatches to the selected TTA method via the METHOD env variable.
#
# Usage:
#   METHOD=full sbatch run_cogvideo_tta.sbatch
#   METHOD=delta_a sbatch run_cogvideo_tta.sbatch
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
METHOD="${METHOD:-delta_a}"
NUM_VIDEOS="${NUM_VIDEOS:-100}"
SEED="${SEED:-42}"

# Full-model TTA settings
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_STEPS="${NUM_STEPS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
OPTIMIZER="${OPTIMIZER:-sgd}"

# LoRA settings
LORA_RANK="${LORA_RANK:-4}"
LORA_ALPHA="${LORA_ALPHA:-4.0}"

# Delta-A settings
DELTA_STEPS="${DELTA_STEPS:-20}"
DELTA_LR="${DELTA_LR:-1e-3}"

# Generation settings
NUM_COND_FRAMES="${NUM_COND_FRAMES:-1}"
NUM_TRAIN_FRAMES="${NUM_TRAIN_FRAMES:-49}"
NUM_GEN_FRAMES="${NUM_GEN_FRAMES:-49}"
GEN_START_FRAME="${GEN_START_FRAME:-49}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-6.0}"

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
MODEL_PATH="${MODEL_PATH:-${SCRATCH_BASE}/cogvideo-checkpoints/CogVideoX-5b-I2V}"
DATA_DIR="${DATA_DIR:-${SCRATCH_BASE}/longcat-video-tta/datasets/panda_100_480p}"
META_PATH="${META_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/backbone_experiment/cogvideo/results/${METHOD}}"

echo "=============================================================================="
echo "CogVideoX-5B-I2V TTA — Method: ${METHOD}"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "=============================================================================="
echo "Configuration:"
echo "  Method         : ${METHOD}"
echo "  Videos         : ${NUM_VIDEOS}"
echo "  Model path     : ${MODEL_PATH}"
echo "  Data dir       : ${DATA_DIR}"
echo "  Output dir     : ${OUTPUT_DIR}"
if [ "${METHOD}" = "full" ]; then
    echo "  Learning rate  : ${LEARNING_RATE}"
    echo "  Num steps      : ${NUM_STEPS}"
    echo "  Optimizer      : ${OPTIMIZER}"
elif [ "${METHOD}" = "lora" ]; then
    echo "  Learning rate  : ${LEARNING_RATE}"
    echo "  Num steps      : ${NUM_STEPS}"
    echo "  LoRA rank      : ${LORA_RANK}"
    echo "  LoRA alpha     : ${LORA_ALPHA}"
elif [ "${METHOD}" = "delta_a" ]; then
    echo "  Delta steps    : ${DELTA_STEPS}"
    echo "  Delta LR       : ${DELTA_LR}"
fi
echo "  Cond frames    : ${NUM_COND_FRAMES}"
echo "  Train frames   : ${NUM_TRAIN_FRAMES}"
echo "  Gen frames     : ${NUM_GEN_FRAMES}"
echo "  Gen start      : ${GEN_START_FRAME}"
echo "  Infer steps    : ${NUM_INFERENCE_STEPS}"
echo "  Guidance       : ${GUIDANCE_SCALE}"
echo "=============================================================================="

# ============================================================================
# Environment Setup
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/cogvideo"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "Activated conda environment: $CONDA_ENV"
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

# Create output and log directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "backbone_experiment/cogvideo/slurm_log"

# ============================================================================
# GPU Info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Dispatch
# ============================================================================
META_FLAG=""
if [ -n "${META_PATH}" ]; then
    META_FLAG="--meta-path ${META_PATH}"
fi

if [ "${METHOD}" = "full" ]; then
    echo "Starting Full-Model TTA for CogVideoX-5B-I2V..."
    python backbone_experiment/cogvideo/run_full_tta_cogvideo.py \
        --model-path "${MODEL_PATH}" \
        --data-dir "${DATA_DIR}" \
        ${META_FLAG} \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${NUM_VIDEOS}" \
        --learning-rate "${LEARNING_RATE}" \
        --num-steps "${NUM_STEPS}" \
        --warmup-steps "${WARMUP_STEPS}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --optimizer "${OPTIMIZER}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-train-frames "${NUM_TRAIN_FRAMES}" \
        --num-gen-frames "${NUM_GEN_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --seed "${SEED}"

elif [ "${METHOD}" = "lora" ]; then
    echo "Starting LoRA TTA for CogVideoX-5B-I2V..."
    python backbone_experiment/cogvideo/run_lora_tta_cogvideo.py \
        --model-path "${MODEL_PATH}" \
        --data-dir "${DATA_DIR}" \
        ${META_FLAG} \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${NUM_VIDEOS}" \
        --learning-rate "${LEARNING_RATE}" \
        --num-steps "${NUM_STEPS}" \
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}" \
        --max-grad-norm "${MAX_GRAD_NORM}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-train-frames "${NUM_TRAIN_FRAMES}" \
        --num-gen-frames "${NUM_GEN_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --seed "${SEED}"

elif [ "${METHOD}" = "delta_a" ]; then
    echo "Starting Delta-A TTA for CogVideoX-5B-I2V..."
    python backbone_experiment/cogvideo/run_delta_a_cogvideo.py \
        --model-path "${MODEL_PATH}" \
        --data-dir "${DATA_DIR}" \
        ${META_FLAG} \
        --output-dir "${OUTPUT_DIR}" \
        --max-videos "${NUM_VIDEOS}" \
        --delta-steps "${DELTA_STEPS}" \
        --delta-lr "${DELTA_LR}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-train-frames "${NUM_TRAIN_FRAMES}" \
        --num-gen-frames "${NUM_GEN_FRAMES}" \
        --gen-start-frame "${GEN_START_FRAME}" \
        --num-inference-steps "${NUM_INFERENCE_STEPS}" \
        --guidance-scale "${GUIDANCE_SCALE}" \
        --seed "${SEED}"

else
    echo "ERROR: Unknown METHOD '${METHOD}'. Use 'full', 'lora', or 'delta_a'." >&2
    exit 1
fi

echo ""
echo "=============================================================================="
echo "CogVideoX-5B-I2V TTA Complete (${METHOD})"
echo "=============================================================================="
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}/summary.json"
echo "=============================================================================="
