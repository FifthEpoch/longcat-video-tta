#!/bin/bash
#SBATCH --job-name=delta_c_tta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=delta_experiment/sbatch/slurm_log/delta_c_%j.out
#SBATCH --error=delta_experiment/sbatch/slurm_log/delta_c_%j.err

# ============================================================================
# Delta-C TTA: Output correction residual
#
# Learns a per-channel δ_out added to the denoiser's output prediction.
# Estimated time: ~2-4 hours for 100 videos (training only)
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
NUM_VIDEOS="${NUM_VIDEOS:-100}"
DELTA_STEPS="${DELTA_STEPS:-20}"
DELTA_LR="${DELTA_LR:-1e-3}"
DELTA_MODE="${DELTA_MODE:-per_channel}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-13}"
NUM_FRAMES="${NUM_FRAMES:-93}"
RESOLUTION="${RESOLUTION:-480p}"
SEED="${SEED:-42}"

# Early stopping configuration
ES_DISABLE="${ES_DISABLE:-}"
ES_CHECK_EVERY="${ES_CHECK_EVERY:-5}"
ES_PATIENCE="${ES_PATIENCE:-3}"
ES_ANCHOR_TIMESTEPS="${ES_ANCHOR_TIMESTEPS:-0.25,0.5,0.75}"
ES_NOISE_DRAWS="${ES_NOISE_DRAWS:-2}"
ES_STRATEGY="${ES_STRATEGY:-patience}"
ES_HOLDOUT_FRACTION="${ES_HOLDOUT_FRACTION:-0.25}"

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/env_setup/download_ucf101}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/delta_experiment/results/delta_c}"

echo "=============================================================================="
echo "Delta-C TTA for LongCat-Video"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
echo "=============================================================================="
echo "Configuration:"
echo "  Videos         : ${NUM_VIDEOS}"
echo "  Delta steps    : ${DELTA_STEPS}"
echo "  Delta LR       : ${DELTA_LR}"
echo "  Delta mode     : ${DELTA_MODE}"
echo "  Cond frames    : ${NUM_COND_FRAMES}"
echo "  Num frames     : ${NUM_FRAMES}"
echo "  Checkpoint dir : ${CHECKPOINT_DIR}"
echo "  Data dir       : ${DATA_DIR}"
echo "  Output dir     : ${OUTPUT_DIR}"
echo "=============================================================================="

# ============================================================================
# Environment Setup
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

CONDA_ENV="${SCRATCH_BASE}/conda-envs/opensora13"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "✓ Activated conda environment: $CONDA_ENV"
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
mkdir -p "delta_experiment/sbatch/slurm_log"

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
ES_FLAGS="${ES_FLAGS} --es-anchor-timesteps ${ES_ANCHOR_TIMESTEPS}"
ES_FLAGS="${ES_FLAGS} --es-noise-draws ${ES_NOISE_DRAWS}"
ES_FLAGS="${ES_FLAGS} --es-strategy ${ES_STRATEGY}"
ES_FLAGS="${ES_FLAGS} --es-holdout-fraction ${ES_HOLDOUT_FRACTION}"

# ============================================================================
# Run Delta-C
# ============================================================================
echo "Starting Delta-C TTA..."

python delta_experiment/scripts/run_delta_c.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-videos "${NUM_VIDEOS}" \
    --delta-steps "${DELTA_STEPS}" \
    --delta-lr "${DELTA_LR}" \
    --delta-mode "${DELTA_MODE}" \
    --num-cond-frames "${NUM_COND_FRAMES}" \
    --num-frames "${NUM_FRAMES}" \
    --resolution "${RESOLUTION}" \
    --seed "${SEED}" \
    ${ES_FLAGS}

echo ""
echo "=============================================================================="
echo "Delta-C TTA Complete"
echo "=============================================================================="
echo "End time: $(date)"
echo "Results saved to: ${OUTPUT_DIR}/summary.json"
echo "=============================================================================="
