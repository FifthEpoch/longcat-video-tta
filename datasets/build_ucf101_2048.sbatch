#!/bin/bash
#SBATCH --job-name=build_ucf101_2048
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --output=datasets/slurm_log/build_ucf101_2048_%j.out
#SBATCH --error=datasets/slurm_log/build_ucf101_2048_%j.err

# ============================================================================
# Build the UCF-101 2048-video dataset as an extension of the existing
# 1000-video set.
#
# CPU job — ffmpeg transcoding only, no network calls. Faster than the
# Panda build (no yt-dlp).
#
# Submit:
#   sbatch --account=torch_pr_36_mren datasets/build_ucf101_2048.sbatch
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

SRC_DIR="${SRC_DIR:-${SCRATCH_BASE}/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org}"
EXISTING="${EXISTING:-${PROJECT_ROOT}/datasets/ucf101_test_480p}"
NEW="${NEW:-${PROJECT_ROOT}/datasets/ucf101_2048_480p}"
TARGET_VALID="${TARGET_VALID:-2048}"
VIDEOS_PER_CATEGORY="${VIDEOS_PER_CATEGORY:-13}"

echo "=============================================================================="
echo "Build UCF-101 2048-video dataset"
echo "=============================================================================="
echo "  Src       : ${SRC_DIR}"
echo "  Existing  : ${EXISTING}"
echo "  New       : ${NEW}"
echo "  Target    : ${TARGET_VALID} valid"
echo "  Per-cat   : ${VIDEOS_PER_CATEGORY}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${ENV_PATH}/bin:${PATH}"
PYTHON="${ENV_PATH}/bin/python"

"${PYTHON}" -c "import av" 2>/dev/null || pip install av --quiet
if ! command -v ffmpeg &>/dev/null; then
    conda install -y -c conda-forge ffmpeg --quiet 2>/dev/null || true
fi

mkdir -p "${PROJECT_ROOT}/datasets/slurm_log"
cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/build_ucf101_2048_dataset.py \
    --src-dir "${SRC_DIR}" \
    --existing-dataset "${EXISTING}" \
    --new-dataset "${NEW}" \
    --target-valid "${TARGET_VALID}" \
    --videos-per-category "${VIDEOS_PER_CATEGORY}"

echo ""
echo "=============================================================================="
echo "Build job finished"
echo "  Dataset    : ${NEW}"
echo "  Videos     : $(ls ${NEW}/videos/*.mp4 2>/dev/null | wc -l)"
echo "  Report     : ${NEW}/validation_report.json"
echo "=============================================================================="
