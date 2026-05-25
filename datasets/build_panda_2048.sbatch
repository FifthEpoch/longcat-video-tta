#!/bin/bash
#SBATCH --job-name=build_panda_2048
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=datasets/slurm_log/build_panda_2048_%j.out
#SBATCH --error=datasets/slurm_log/build_panda_2048_%j.err

# ============================================================================
# Build the Panda-70M 2048-video dataset as an extension of the existing
# 1000-video set (panda_1000_480p).
#
# This is a CPU + network job — no GPU needed. yt-dlp downloads run
# concurrently inside the Python script; the script is single-process
# but yt-dlp itself uses multiple connections.
#
# Submit:
#   sbatch --account=torch_pr_36_mren datasets/build_panda_2048.sbatch
#
# Resume (idempotent — safe to re-run; skips already-downloaded clips
# via download_panda70m_subset.py's --resume mode):
#   sbatch --account=torch_pr_36_mren --export=ALL,SKIP_DOWNLOAD=0 \
#       datasets/build_panda_2048.sbatch
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

EXISTING="${EXISTING:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
NEW="${NEW:-${PROJECT_ROOT}/datasets/panda_2048_480p}"
TARGET_VALID="${TARGET_VALID:-2048}"
OVERSAMPLE_TO="${OVERSAMPLE_TO:-2300}"
HF_MAX_ROWS="${HF_MAX_ROWS:-80000}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_SEEDING="${SKIP_SEEDING:-0}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

echo "=============================================================================="
echo "Build Panda-70M 2048-video dataset"
echo "=============================================================================="
echo "  Existing  : ${EXISTING}"
echo "  New       : ${NEW}"
echo "  Target    : ${TARGET_VALID} valid"
echo "  Oversample: ${OVERSAMPLE_TO}"
echo "  HF rows   : ${HF_MAX_ROWS}"
echo "  Skip dl   : ${SKIP_DOWNLOAD}"
echo "  Skip seed : ${SKIP_SEEDING}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${ENV_PATH}/bin:${PATH}"
PYTHON="${ENV_PATH}/bin/python"

# Make sure required tooling is present.
"${PYTHON}" -c "import av" 2>/dev/null || pip install av --quiet
pip install yt-dlp --upgrade --quiet 2>/dev/null || true
pip install gdown --quiet 2>/dev/null || true
pip install datasets --quiet 2>/dev/null || true
if ! command -v node &>/dev/null; then
    conda install -y -c conda-forge nodejs --quiet 2>/dev/null || true
fi
if ! command -v ffprobe &>/dev/null; then
    conda install -y -c conda-forge ffmpeg --quiet 2>/dev/null || true
fi

mkdir -p "${PROJECT_ROOT}/datasets/slurm_log"
cd "${PROJECT_ROOT}"

EXTRA_FLAGS=""
[ "${SKIP_DOWNLOAD}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-download"
[ "${SKIP_SEEDING}"  = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-seeding"

# YouTube cookies passthrough.
COOKIES_FLAG=""
COOKIES_PATH="${COOKIES_FILE:-${PROJECT_ROOT}/datasets/cookies.txt}"
if [ -f "${COOKIES_PATH}" ]; then
    COOKIES_FLAG="--cookies-file ${COOKIES_PATH}"
    echo "Found cookies: ${COOKIES_PATH}"
fi

"${PYTHON}" scripts/build_panda_2048_dataset.py \
    --existing-dataset "${EXISTING}" \
    --new-dataset "${NEW}" \
    --target-valid "${TARGET_VALID}" \
    --oversample-to "${OVERSAMPLE_TO}" \
    --hf-max-rows "${HF_MAX_ROWS}" \
    ${COOKIES_FLAG} \
    ${EXTRA_FLAGS}

echo ""
echo "=============================================================================="
echo "Build job finished"
echo "  Dataset    : ${NEW}"
echo "  Videos     : $(ls ${NEW}/videos/*.mp4 2>/dev/null | wc -l)"
echo "  Report     : ${NEW}/validation_report.json"
echo "=============================================================================="
