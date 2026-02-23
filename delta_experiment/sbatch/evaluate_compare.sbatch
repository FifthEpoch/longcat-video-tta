#!/bin/bash
#SBATCH --job-name=delta_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=delta_experiment/sbatch/slurm_log/eval_%j.out
#SBATCH --error=delta_experiment/sbatch/slurm_log/eval_%j.err

# ============================================================================
# Evaluate baseline and compare all TTA methods
#
# This job:
#   1. Runs baseline evaluation (generate_vc without any adaptation)
#   2. Runs comparison across all methods that have results
#
# Estimated time: ~6-12 hours (mostly baseline generation)
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Configuration
# ============================================================================
NUM_VIDEOS="${NUM_VIDEOS:-100}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-13}"
NUM_FRAMES="${NUM_FRAMES:-93}"
RESOLUTION="${RESOLUTION:-480p}"
SEED="${SEED:-42}"
SKIP_BASELINE="${SKIP_BASELINE:-}"

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/open-sora-v1.3-experiment"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/env_setup/download_ucf101}"

RESULTS_BASE="${PROJECT_ROOT}/delta_experiment/results"
BASELINE_EVAL_DIR="${RESULTS_BASE}/baseline_eval"
DELTA_A_DIR="${RESULTS_BASE}/delta_a"
DELTA_B_DIR="${RESULTS_BASE}/delta_b"
DELTA_C_DIR="${RESULTS_BASE}/delta_c"
LORA_DIR="${RESULTS_BASE}/lora_best"
COMPARISON_DIR="${RESULTS_BASE}/comparison"

echo "=============================================================================="
echo "Delta TTA Evaluation & Comparison"
echo "=============================================================================="
echo "Job ID       : ${SLURM_JOB_ID}"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
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

mkdir -p "${BASELINE_EVAL_DIR}"
mkdir -p "${COMPARISON_DIR}"
mkdir -p "delta_experiment/sbatch/slurm_log"

# ============================================================================
# GPU Info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Step 1: Baseline Evaluation
# ============================================================================
if [ -z "${SKIP_BASELINE}" ]; then
    echo "=============================================================================="
    echo "Step 1: Baseline Evaluation"
    echo "=============================================================================="

    python delta_experiment/scripts/evaluate_delta.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --data-dir "${DATA_DIR}" \
        --output-dir "${BASELINE_EVAL_DIR}" \
        --mode baseline \
        --max-videos "${NUM_VIDEOS}" \
        --num-cond-frames "${NUM_COND_FRAMES}" \
        --num-frames "${NUM_FRAMES}" \
        --resolution "${RESOLUTION}" \
        --seed "${SEED}"

    echo "Baseline evaluation complete."
else
    echo "Skipping baseline evaluation (SKIP_BASELINE set)"
fi

# ============================================================================
# Step 2: Comparison
# ============================================================================
echo ""
echo "=============================================================================="
echo "Step 2: Compare All Methods"
echo "=============================================================================="

COMPARE_FLAGS=""
if [ -d "${BASELINE_EVAL_DIR}" ] && [ -f "${BASELINE_EVAL_DIR}/eval_summary.json" ]; then
    COMPARE_FLAGS="${COMPARE_FLAGS} --baseline-dir ${BASELINE_EVAL_DIR}"
fi
if [ -d "${LORA_DIR}" ]; then
    COMPARE_FLAGS="${COMPARE_FLAGS} --lora-dir ${LORA_DIR}"
fi
if [ -d "${DELTA_A_DIR}" ]; then
    COMPARE_FLAGS="${COMPARE_FLAGS} --delta-a-dir ${DELTA_A_DIR}"
fi
if [ -d "${DELTA_B_DIR}" ]; then
    COMPARE_FLAGS="${COMPARE_FLAGS} --delta-b-dir ${DELTA_B_DIR}"
fi
if [ -d "${DELTA_C_DIR}" ]; then
    COMPARE_FLAGS="${COMPARE_FLAGS} --delta-c-dir ${DELTA_C_DIR}"
fi

python delta_experiment/scripts/compare_methods.py \
    ${COMPARE_FLAGS} \
    --output-dir "${COMPARISON_DIR}"

echo ""
echo "=============================================================================="
echo "Evaluation & Comparison Complete"
echo "=============================================================================="
echo "End time   : $(date)"
echo "Baseline   : ${BASELINE_EVAL_DIR}/eval_summary.json"
echo "Comparison : ${COMPARISON_DIR}/comparison.json"
echo "=============================================================================="
