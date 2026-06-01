#!/bin/bash
#SBATCH --job-name=download_panda70m_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=03:00:00
#SBATCH --output=datasets/slurm_log/download_panda70m_full_%j.out
#SBATCH --error=datasets/slurm_log/download_panda70m_full_%j.err

# ============================================================================
# Download Panda-70M full training metadata CSV from Google Drive.
# ============================================================================
#
# Why: panda70m_training_2m.csv (already on disk) caps each source video at
# 3 segments -> only 3,302 valid segments emerged for our 2048-video pool
# in Phase 2A. The FULL training metadata stores ~18.7 segments/source video
# (70.7M segments / 3.78M sources), which projects to ~38K segments before
# the desirable_filtering / matching_score / duration gates and ~25-30K after.
# That clears the user's 25K target.
#
# Google Drive (per snap-research/Panda-70M README):
#   full split = 2.73 GB CSV, file ID 1pbh8W3qgst9CD7nlPhsH9wmUSWjQlGdW
#
# We use gdown (same tool that successfully fetched panda70m_training_2m.csv
# during the panda_pool_10k build job). gdown handles Google's >100MB
# confirmation token automatically.
#
# Submit (default = full split):
#   sbatch --account=torch_pr_36_mren \
#       datasets/download_panda70m_full_metadata.sbatch
#
# Override the split (e.g., 10m for a smaller/faster test):
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,SPLIT=10m \
#       datasets/download_panda70m_full_metadata.sbatch
#
# Force re-download even if the file exists:
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,FORCE=1 \
#       datasets/download_panda70m_full_metadata.sbatch
#
# Expected wall time: ~30-60 min for the 2.73 GB Full split (depends on
# Google Drive throughput; gdown typically pulls at 30-100 MB/s from this
# cluster).
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/datasets/panda_metadata_full}"
SPLIT="${SPLIT:-full}"
FORCE="${FORCE:-0}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

echo "=============================================================================="
echo "Download Panda-70M training metadata"
echo "=============================================================================="
echo "  out_dir : ${OUT_DIR}"
echo "  split   : ${SPLIT}"
echo "  force   : ${FORCE}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${ENV_PATH}/bin:${PATH}"
PYTHON="${ENV_PATH}/bin/python"

mkdir -p "${PROJECT_ROOT}/datasets/slurm_log"
mkdir -p "${OUT_DIR}"
cd "${PROJECT_ROOT}"

if ! "${PYTHON}" -c "import gdown" 2>/dev/null; then
    echo "Installing gdown ..."
    "${PYTHON}" -m pip install gdown --quiet --no-warn-script-location
fi

GDOWN_VERSION=$("${PYTHON}" -c "import gdown; print(gdown.__version__)" 2>/dev/null || echo unknown)
echo "  gdown   : ${GDOWN_VERSION}"
echo ""

EXTRA_FLAGS=""
[ "${FORCE}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --force"

"${PYTHON}" scripts/download_panda70m_full_metadata.py \
    --out-dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    ${EXTRA_FLAGS}

echo ""
echo "=============================================================================="
echo "Files in ${OUT_DIR}:"
echo "=============================================================================="
ls -lh "${OUT_DIR}/" 2>/dev/null || true
echo "=============================================================================="
