#!/bin/bash
#SBATCH --job-name=panda70m_dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=datasets/slurm_log/panda70m_dl_%j.out
#SBATCH --error=datasets/slurm_log/panda70m_dl_%j.err

# ==============================================================================
# Download Panda-70M 100-video stratified subset for LongCat-Video TTA
# ==============================================================================
# Downloads videos via yt-dlp from YouTube using Panda-70M metadata.
# Uses HuggingFace datasets library to get metadata, stratifies by
# caption category, and retries aggressively (~15x candidate pool).
#
# No GPU needed — this is a CPU + network job.
#
# Usage:
#   sbatch --account=torch_pr_36_mren datasets/download_panda70m.sbatch
#
# Resume after partial download:
#   RESUME=1 sbatch --account=torch_pr_36_mren datasets/download_panda70m.sbatch
# ==============================================================================

set -euo pipefail

echo "=============================================================================="
echo "Download Panda-70M 100-video stratified subset"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================================================="

# ==============================================================================
# Configuration
# ==============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"
DATASET_DIR="${PROJECT_ROOT}/datasets/panda_100"

# Configurable via environment variables
NUM_VIDEOS="${NUM_VIDEOS:-100}"
SEED="${SEED:-42}"
MIN_DURATION="${MIN_DURATION:-4}"
MAX_DURATION="${MAX_DURATION:-30}"
MIN_FRAMES="${MIN_FRAMES:-33}"
HF_MAX_ROWS="${HF_MAX_ROWS:-50000}"
CANDIDATE_MULT="${CANDIDATE_MULT:-15}"
DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-120}"
RESUME="${RESUME:-0}"

# Optional: use existing local metadata instead of HuggingFace
# If the old Open-Sora workspace has a JSONL, we can use it as fallback
OLD_META="${SCRATCH_BASE}/open-sora-v2.0-experiment/datasets/panda_100/panda70m_metadata.jsonl"

# ==============================================================================
# Setup environment
# ==============================================================================
echo ""
echo "Setting up environment..."

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"

echo "Python: $(python --version)"
echo "Python executable: $(which python)"

# ==============================================================================
# Install download tools
# ==============================================================================
echo ""
echo "Installing download tools..."

pip install yt-dlp --quiet --upgrade 2>/dev/null || true
pip install datasets --quiet 2>/dev/null || true

# Ensure ffmpeg/ffprobe are available
if ! command -v ffprobe &>/dev/null; then
    echo "Installing ffmpeg via conda..."
    conda install -y -c conda-forge ffmpeg --quiet 2>/dev/null || true
fi

# Verify tools
echo "yt-dlp:  $(yt-dlp --version 2>/dev/null || echo 'NOT FOUND')"
echo "ffprobe: $(ffprobe -version 2>/dev/null | head -1 || echo 'NOT FOUND')"

# ==============================================================================
# Create output directories
# ==============================================================================
mkdir -p "${DATASET_DIR}/videos"
mkdir -p "${PROJECT_ROOT}/datasets/slurm_log"

# ==============================================================================
# Run download script
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Starting download: ${NUM_VIDEOS} videos, seed=${SEED}"
echo "Output: ${DATASET_DIR}"
echo "=============================================================================="

cd "${PROJECT_ROOT}"

# Build command
CMD=(
    python datasets/download_panda70m_subset.py
    --out-dir "${DATASET_DIR}"
    --num-videos "${NUM_VIDEOS}"
    --seed "${SEED}"
    --min-duration "${MIN_DURATION}"
    --max-duration "${MAX_DURATION}"
    --min-frames "${MIN_FRAMES}"
    --hf-max-rows "${HF_MAX_ROWS}"
    --candidate-multiplier "${CANDIDATE_MULT}"
    --download-timeout "${DOWNLOAD_TIMEOUT}"
)

# NOTE: The old Open-Sora metadata JSONL is malformed (mixed types).
# Always use HuggingFace for a clean, large candidate pool.
# To force local metadata (not recommended), set: USE_LOCAL_META=1
if [ "${USE_LOCAL_META:-0}" = "1" ] && [ -f "${OLD_META}" ]; then
    echo "Using local metadata at: ${OLD_META}"
    CMD+=(--meta-path "${OLD_META}")
else
    echo "Using HuggingFace metadata (iejMac/Panda-70M, ${HF_MAX_ROWS} rows)"
fi

# Resume support
if [ "${RESUME}" = "1" ]; then
    echo "Resume mode enabled"
    CMD+=(--resume)
fi

echo ""
echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Download job complete"
echo "=============================================================================="
echo "Dataset: ${DATASET_DIR}"
echo "Videos:  $(ls ${DATASET_DIR}/videos/*.mp4 2>/dev/null | wc -l) files"
echo ""
echo "End time: $(date)"
echo "=============================================================================="
