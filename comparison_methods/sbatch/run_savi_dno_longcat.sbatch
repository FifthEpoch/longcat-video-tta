#!/bin/bash
#SBATCH --job-name=savi_dno_lc
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --output=comparison_methods/slurm_log/savi_dno_longcat_%j.out
#SBATCH --error=comparison_methods/slurm_log/savi_dno_longcat_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "=============================================================================="
echo "SAVi-DNO Evaluation (Noise Optimization on LongCat Backbone)"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Euler steps: ${SAVI_EULER_STEPS:-10}"
echo "Num GPUs: ${NUM_GPUS:-2}"
echo "LR: ${SAVI_LR:-0.01}"
echo "Lambda: ${SAVI_LAM:-0.0012}"
echo "p: ${SAVI_P:-0.9}"
echo "Guidance: ${SAVI_GUIDANCE:-4.0}"
echo "Loss mode: ${SAVI_LOSS_MODE:-pixel} (PVDM-style: pixel+feature, no CFG)"
echo "No-optimize: ${SAVI_NO_OPTIMIZE:-}"
echo "=============================================================================="

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${SAVI_LC_DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
OUTPUT_DIR="${SAVI_LC_OUTPUT_DIR:-${PROJECT_ROOT}/comparison_methods/results/savi_dno_longcat_s${SAVI_EULER_STEPS:-10}}"
MAX_VIDEOS="${SAVI_LC_MAX_VIDEOS:-1000}"

# ============================================================================
# Environment Setup (LongCat conda env)
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

unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed" >&2
    exit 1
}

cd "${PROJECT_ROOT}"

# ============================================================================
# Build flags
# ============================================================================
GT_CACHE_FLAG=""
if [ -n "${GT_FEATURES_CACHE:-}" ] && [ -f "${GT_FEATURES_CACHE}" ]; then
    GT_CACHE_FLAG="--gt-features-cache ${GT_FEATURES_CACHE}"
fi
SAVE_LIST_FLAG=""
if [ -n "${SAVE_ONLY_LIST:-}" ] && [ -f "${SAVE_ONLY_LIST}" ]; then
    SAVE_LIST_FLAG="--save-only-list ${SAVE_ONLY_LIST}"
fi
NO_OPT_FLAG=""
if [ -n "${SAVI_NO_OPTIMIZE:-}" ] && [ "${SAVI_NO_OPTIMIZE}" = "1" ]; then
    NO_OPT_FLAG="--no-optimize"
fi
LOSS_FLAG=""
if [ "${SAVI_LOSS_MODE:-pixel}" = "latent" ]; then
    LOSS_FLAG="--latent-loss"
else
    LOSS_FLAG="--pixel-loss"
fi

# ============================================================================
# Run SAVi-DNO with LongCat backbone
# ============================================================================
"$PYTHON" comparison_methods/scripts/savi_dno_longcat.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-videos "${MAX_VIDEOS}" \
    --num-inference-steps "${SAVI_EULER_STEPS:-10}" \
    --num-gpus "${NUM_GPUS:-2}" \
    --guidance-scale "${SAVI_GUIDANCE:-4.0}" \
    --lr "${SAVI_LR:-0.01}" \
    --lam "${SAVI_LAM:-0.0012}" \
    --p "${SAVI_P:-0.9}" \
    --seed 42 \
    ${NO_OPT_FLAG} \
    ${LOSS_FLAG} \
    ${GT_CACHE_FLAG} \
    ${SAVE_LIST_FLAG}

echo ""
echo "=============================================================================="
echo "SAVi-DNO LongCat Complete"
echo "End time: $(date)"
echo "=============================================================================="
