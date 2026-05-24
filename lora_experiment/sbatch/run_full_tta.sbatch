#!/bin/bash
#SBATCH --job-name=full_tta_longcat
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=lora_experiment/sbatch/slurm_log/full_tta_%j.out
#SBATCH --error=lora_experiment/sbatch/slurm_log/full_tta_%j.err

# ============================================================================
# Full-model TTA: Fine-tune ALL DiT parameters per-video on LongCat-Video
#
# Full-model tuning requires storing a complete copy of base weights on CPU
# (13.6B params ≈ 27GB), plus optimizer states on GPU.
# Requesting 256G RAM (generous buffer since full-model TTA is the most
# memory-hungry method).
#
# Estimated time: ~8-16 hours for 100 videos (training + generation)
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
NUM_VIDEOS="${NUM_VIDEOS:-100}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
NUM_STEPS="${NUM_STEPS:-10}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-2}"
NUM_FRAMES="${NUM_FRAMES:-16}"
GEN_START_FRAME="${GEN_START_FRAME:-32}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"
RESOLUTION="${RESOLUTION:-480p}"
SEED="${SEED:-42}"
SKIP_GENERATION="${SKIP_GENERATION:-}"

# Early stopping configuration
ES_DISABLE="${ES_DISABLE:-}"
ES_CHECK_EVERY="${ES_CHECK_EVERY:-3}"
ES_PATIENCE="${ES_PATIENCE:-2}"
ES_ANCHOR_TIMESTEPS="${ES_ANCHOR_TIMESTEPS:-0.25,0.5,0.75}"
ES_NOISE_DRAWS="${ES_NOISE_DRAWS:-2}"
ES_STRATEGY="${ES_STRATEGY:-patience}"
ES_HOLDOUT_FRACTION="${ES_HOLDOUT_FRACTION:-0.25}"

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:-${SCRATCH_BASE}/longcat-video-tta/datasets/panda_100_480p}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/lora_experiment/results/full_tta_lr${LEARNING_RATE}_${NUM_STEPS}steps}"

echo "=============================================================================="
echo "Full-Model TTA for LongCat-Video"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "=============================================================================="
echo "Configuration:"
echo "  Videos            : ${NUM_VIDEOS}"
echo "  Learning rate     : ${LEARNING_RATE}"
echo "  Num steps         : ${NUM_STEPS}"
echo "  Num cond frames   : ${NUM_COND_FRAMES}"
echo "  Num frames        : ${NUM_FRAMES}"
echo "  Gen start         : ${GEN_START_FRAME}"
echo "  Inference steps   : ${NUM_INFERENCE_STEPS}"
echo "  Guidance scale    : ${GUIDANCE_SCALE}"
echo "  Checkpoint dir    : ${CHECKPOINT_DIR}"
echo "  Data dir          : ${DATA_DIR}"
echo "  Output dir        : ${OUTPUT_DIR}"
echo "=============================================================================="

# ============================================================================
# Environment Setup
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "✓ Activated conda environment: $CONDA_ENV"
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
mkdir -p "lora_experiment/sbatch/slurm_log"

# ============================================================================
# GPU Info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Build flags
# ============================================================================
EXTRA_FLAGS=""
if [ -n "${SKIP_GENERATION}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --skip-generation"
fi

# Build early stopping flags
ES_FLAGS=""
if [ -n "${ES_DISABLE}" ]; then
    ES_FLAGS="${ES_FLAGS} --es-disable"
fi
ES_FLAGS="${ES_FLAGS} --es-check-every ${ES_CHECK_EVERY}"
ES_FLAGS="${ES_FLAGS} --es-patience ${ES_PATIENCE}"
ES_FLAGS="${ES_FLAGS} --es-anchor-sigmas ${ES_ANCHOR_TIMESTEPS}"
ES_FLAGS="${ES_FLAGS} --es-noise-draws ${ES_NOISE_DRAWS}"
ES_FLAGS="${ES_FLAGS} --es-strategy ${ES_STRATEGY}"
ES_FLAGS="${ES_FLAGS} --es-holdout-fraction ${ES_HOLDOUT_FRACTION}"

# ============================================================================
# Run Full-Model TTA
# ============================================================================
echo "Starting Full-Model TTA..."

python lora_experiment/scripts/run_full_tta.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-videos "${NUM_VIDEOS}" \
    --learning-rate "${LEARNING_RATE}" \
    --num-steps "${NUM_STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --num-cond-frames "${NUM_COND_FRAMES}" \
    --num-frames "${NUM_FRAMES}" \
    --gen-start-frame "${GEN_START_FRAME}" \
    --num-inference-steps "${NUM_INFERENCE_STEPS}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --resolution "${RESOLUTION}" \
    --seed "${SEED}" \
    ${EXTRA_FLAGS} \
    ${ES_FLAGS}

echo ""
echo "=============================================================================="
echo "Full-Model TTA Complete"
echo "=============================================================================="
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}/summary.json"
echo "=============================================================================="
