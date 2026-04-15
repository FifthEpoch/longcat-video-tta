#!/bin/bash
#SBATCH --job-name=tinylora_tta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=delta_experiment/sbatch/slurm_log/tinylora_%j.out
#SBATCH --error=delta_experiment/sbatch/slurm_log/tinylora_%j.err

# ============================================================================
# TinyLoRA TTA: Ultra-low-rank SVD-based adapters for LongCat-Video
#
# Adapts the TinyLoRA method (Morris et al., 2026) for per-video TTA.
# Typical trainable params: 192 (default) down to ~4 (with weight tying).
#
# Estimated time: ~2-4 hours for 100 videos (training only, no generation)
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
NUM_VIDEOS="${NUM_VIDEOS:-100}"
TTA_STEPS="${TTA_STEPS:-20}"
TTA_LR="${TTA_LR:-1e-3}"
SVD_RANK="${SVD_RANK:-2}"
ALPHA="${ALPHA:-1.0}"
N_TIE="${N_TIE:-1}"
TARGET_PRESET="${TARGET_PRESET:-qkv_proj}"
TARGET_MODULES="${TARGET_MODULES:-}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-2}"
NUM_FRAMES="${NUM_FRAMES:-16}"
RESOLUTION="${RESOLUTION:-480p}"
SEED="${SEED:-42}"
SKIP_GENERATION="${SKIP_GENERATION:-}"

# Early stopping configuration
ES_DISABLE="${ES_DISABLE:-}"
ES_CHECK_EVERY="${ES_CHECK_EVERY:-5}"
ES_PATIENCE="${ES_PATIENCE:-3}"
ES_ANCHOR_SIGMAS="${ES_ANCHOR_SIGMAS:-0.25,0.5,0.75}"
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
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/delta_experiment/results/tinylora_r${SVD_RANK}_ntie${N_TIE}_${TARGET_PRESET}}"

echo "=============================================================================="
echo "TinyLoRA TTA for LongCat-Video"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "=============================================================================="
echo "Configuration:"
echo "  Videos         : ${NUM_VIDEOS}"
echo "  TTA steps      : ${TTA_STEPS}"
echo "  TTA LR         : ${TTA_LR}"
echo "  SVD rank       : ${SVD_RANK}"
echo "  Alpha          : ${ALPHA}"
echo "  Weight tying   : n_tie=${N_TIE}"
echo "  Target preset  : ${TARGET_PRESET}"
echo "  Target modules : ${TARGET_MODULES:-<from preset>}"
echo "  Cond frames    : ${NUM_COND_FRAMES}"
echo "  Num frames     : ${NUM_FRAMES}"
echo "  Checkpoint dir : ${CHECKPOINT_DIR}"
echo "  Data dir       : ${DATA_DIR}"
echo "  Output dir     : ${OUTPUT_DIR}"
echo "=============================================================================="

# ============================================================================
# Environment Setup
#
# CRITICAL: module load sets PYTHONHOME / PYTHONPATH to the system anaconda,
# which breaks the conda env's sys.path (wrong Python version, missing
# packages). We MUST unset these after conda activate. See:
#   sweep_experiment/docs/SBATCH_CONDA_BEST_PRACTICES.md
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
# Without this, even explicit $PYTHON uses the system anaconda's sys.path,
# causing "No module named 'torch'" and similar import errors.
unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

# Verify the environment is correct (catches broken conda activation early)
"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed — conda env is broken or PYTHONHOME was not unset" >&2
    exit 1
}

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

cd "$PROJECT_ROOT"

# Create output and log directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "delta_experiment/sbatch/slurm_log"

# ============================================================================
# GPU Info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Build extra flags
# ============================================================================
EXTRA_FLAGS=""
if [ -n "${TARGET_MODULES}" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --target-modules ${TARGET_MODULES}"
fi
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
ES_FLAGS="${ES_FLAGS} --es-anchor-sigmas ${ES_ANCHOR_SIGMAS}"
ES_FLAGS="${ES_FLAGS} --es-noise-draws ${ES_NOISE_DRAWS}"
ES_FLAGS="${ES_FLAGS} --es-strategy ${ES_STRATEGY}"
ES_FLAGS="${ES_FLAGS} --es-holdout-fraction ${ES_HOLDOUT_FRACTION}"

# ============================================================================
# Run TinyLoRA TTA
# ============================================================================
echo "Starting TinyLoRA TTA..."

"$PYTHON" delta_experiment/scripts/run_tinylora.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-videos "${NUM_VIDEOS}" \
    --svd-rank "${SVD_RANK}" \
    --alpha "${ALPHA}" \
    --n-tie "${N_TIE}" \
    --target-preset "${TARGET_PRESET}" \
    --tta-steps "${TTA_STEPS}" \
    --tta-lr "${TTA_LR}" \
    --num-cond-frames "${NUM_COND_FRAMES}" \
    --num-frames "${NUM_FRAMES}" \
    --resolution "${RESOLUTION}" \
    --seed "${SEED}" \
    ${EXTRA_FLAGS} \
    ${ES_FLAGS}

echo ""
echo "=============================================================================="
echo "TinyLoRA TTA Complete"
echo "=============================================================================="
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}/summary.json"
echo "=============================================================================="
