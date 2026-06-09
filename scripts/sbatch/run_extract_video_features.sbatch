#!/bin/bash
#SBATCH --job-name=extract_video_features
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/logs/extract_video_features_%j.out
#SBATCH --error=sweep_experiment/logs/extract_video_features_%j.err

# ============================================================================
# Per-video feature extraction for the TTA-gain correlation analysis.
#
# Runs scripts/extract_video_features_for_tta.py over a directory of real
# videos and emits a per-video features CSV (RAFT flow, brightness/contrast,
# FFT energy, edge density, scene-change count, temporal stability, CLIP/DINO
# similarities, etc.) covering the TTA-visible frame window.
#
# Estimated time: ~30-90 minutes for 1000 panda-480p videos (RAFT + CLIP +
# DINO dominate). The 02:00:00 wall here is conservative.
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

# ============================================================================
# Inputs / outputs (override via --export=ALL,VAR=value when submitting)
# ============================================================================
VIDEOS_DIR="${VIDEOS_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
CAPTIONS_CSV="${CAPTIONS_CSV:-${VIDEOS_DIR}/metadata.csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv}"
TTA_VISIBLE_FRAMES="${TTA_VISIBLE_FRAMES:-auto}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"

# ============================================================================
# Banner
# ============================================================================
echo "=============================================================================="
echo "Per-video feature extraction for TTA correlation"
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
echo "  VIDEOS_DIR         : ${VIDEOS_DIR}"
echo "  CAPTIONS_CSV       : ${CAPTIONS_CSV}"
echo "  OUTPUT_CSV         : ${OUTPUT_CSV}"
echo "  TTA_VISIBLE_FRAMES : ${TTA_VISIBLE_FRAMES}"
echo "  BATCH_SIZE         : ${BATCH_SIZE}"
echo "  DEVICE             : ${DEVICE}"
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
# GPU info
# ============================================================================
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

# ============================================================================
# Run extractor
# ============================================================================
echo "Starting feature extraction..."

"${PYTHON}" scripts/extract_video_features_for_tta.py \
    --videos-dir "${VIDEOS_DIR}" \
    --captions-csv "${CAPTIONS_CSV}" \
    --output "${OUTPUT_CSV}" \
    --tta-visible-frames "${TTA_VISIBLE_FRAMES}" \
    --device "${DEVICE}" \
    --batch-size "${BATCH_SIZE}"

echo ""
echo "=============================================================================="
echo "Feature extraction complete"
echo "=============================================================================="
echo "End time   : $(date)"
echo "Output CSV : ${OUTPUT_CSV}"
echo "=============================================================================="
