#!/bin/bash
#SBATCH --job-name=compute_diffusion_ood
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=192G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/logs/compute_diffusion_ood_%j.out
#SBATCH --error=sweep_experiment/logs/compute_diffusion_ood_%j.err

# ============================================================================
# Per-video diffusion-OOD scoring against the LongCat-Video base model.
#
# Runs scripts/compute_diffusion_ood_score.py over a directory of real
# videos and emits a per-video OOD-score CSV. For each video the script
# loads the TTA-visible window, VAE-encodes, splits into clean cond +
# noised target latents the SAME way the TTA runners do, runs the
# DiT forward at three representative timesteps in both caption-
# conditioned and unconditional modes, and reports per-video flow-
# matching MSE.
#
# Resource profile (must match the TTA runners since the model loading
# is identical): h200 GPU, 192 G memory, 8 CPU. No backprop, no
# optimiser step, ~6 forward passes per video — expect ~5-15 s/video on
# H200. For 1000 videos that's ~1.5-4 hours; --time=04:00:00 is the
# conservative upper bound.
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

trap 'echo "ERROR on line ${LINENO}: command \"${BASH_COMMAND}\" failed with exit code $?" >&2' ERR

# ============================================================================
# Path setup
# ============================================================================
SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"

# ============================================================================
# Inputs / outputs (override via --export=ALL,VAR=value when submitting)
# ============================================================================
VIDEOS_DIR="${VIDEOS_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
CAPTIONS_CSV="${CAPTIONS_CSV:-${VIDEOS_DIR}/metadata.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv}"
TTA_VISIBLE_FRAMES="${TTA_VISIBLE_FRAMES:-auto}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
TIMESTEPS="${TIMESTEPS:-100,500,900}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
MAX_VIDEOS="${MAX_VIDEOS:-0}"
RESUME="${RESUME:-0}"

# ============================================================================
# Banner
# ============================================================================
echo "=============================================================================="
echo "Per-video diffusion-OOD scoring (LongCat-Video base model)"
echo "=============================================================================="
echo "Job ID         : ${SLURM_JOB_ID:-<unset>}"
echo "Node           : $(hostname)"
echo "Start time     : $(date)"
echo "=============================================================================="
echo "Resolved configuration:"
echo "  USER               : ${USER}"
echo "  SCRATCH_BASE       : ${SCRATCH_BASE}"
echo "  PROJECT_ROOT       : ${PROJECT_ROOT}"
echo "  CONDA_ENV          : ${CONDA_ENV}"
echo "  CHECKPOINT_DIR     : ${CHECKPOINT_DIR}"
echo "  VIDEOS_DIR         : ${VIDEOS_DIR}"
echo "  CAPTIONS_CSV       : ${CAPTIONS_CSV}"
echo "  OUTPUT_CSV         : ${OUTPUT_CSV}"
echo "  TTA_VISIBLE_FRAMES : ${TTA_VISIBLE_FRAMES}"
echo "  TTA_CONTEXT_FRAMES : ${TTA_CONTEXT_FRAMES}"
echo "  TIMESTEPS          : ${TIMESTEPS}"
echo "  SEED               : ${SEED}"
echo "  DEVICE             : ${DEVICE}"
echo "  MAX_VIDEOS         : ${MAX_VIDEOS}  (0 = all)"
echo "  RESUME             : ${RESUME}"
echo "=============================================================================="

# ============================================================================
# Environment setup
#
# CRITICAL: module load anaconda3/... sets PYTHONHOME / PYTHONPATH to the
# system anaconda, which shadows the conda env's sys.path and reproduces the
# "No module named 'torch'" symptom. We MUST unset both AFTER `conda
# activate`. See sweep_experiment/docs/SBATCH_CONDA_BEST_PRACTICES.md.
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

if [ -d "${CONDA_ENV}" ]; then
    conda activate "${CONDA_ENV}"
    echo "Activated conda: ${CONDA_ENV}"
else
    echo "ERROR: Conda environment not found at ${CONDA_ENV}" >&2
    exit 1
fi

unset PYTHONHOME
unset PYTHONPATH

PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python not found at ${PYTHON}" >&2
    exit 1
fi
echo "Using: ${PYTHON} ($("${PYTHON}" --version 2>&1))"

"${PYTHON}" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed — conda env is broken or PYTHONHOME was not unset" >&2
    exit 1
}

export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

cd "${PROJECT_ROOT}"
mkdir -p "$(dirname "${OUTPUT_CSV}")"
mkdir -p "sweep_experiment/logs"

# ============================================================================
# Sanity-check inputs
# ============================================================================
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    echo "ERROR: CHECKPOINT_DIR not found: ${CHECKPOINT_DIR}" >&2
    exit 1
fi
if [ ! -d "${VIDEOS_DIR}" ]; then
    echo "ERROR: VIDEOS_DIR not found: ${VIDEOS_DIR}" >&2
    exit 1
fi
if [ ! -f "${CAPTIONS_CSV}" ]; then
    echo "WARN: CAPTIONS_CSV not found: ${CAPTIONS_CSV} — rows will use empty captions" >&2
fi

# ============================================================================
# GPU info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Build extra flags
# ============================================================================
EXTRA_FLAGS=""
if [ "${RESUME}" = "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --resume"
fi
if [ "${MAX_VIDEOS}" != "0" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --max-videos ${MAX_VIDEOS}"
fi

# ============================================================================
# Run OOD-score computation
# ============================================================================
echo "Starting diffusion-OOD scoring..."

"${PYTHON}" scripts/compute_diffusion_ood_score.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --videos-dir "${VIDEOS_DIR}" \
    --captions-csv "${CAPTIONS_CSV}" \
    --output "${OUTPUT_CSV}" \
    --tta-visible-frames "${TTA_VISIBLE_FRAMES}" \
    --tta-context-frames "${TTA_CONTEXT_FRAMES}" \
    --timesteps "${TIMESTEPS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    ${EXTRA_FLAGS}

echo ""
echo "=============================================================================="
echo "Diffusion-OOD scoring complete"
echo "=============================================================================="
echo "End time   : $(date)"
echo "Output CSV : ${OUTPUT_CSV}"
echo "=============================================================================="
