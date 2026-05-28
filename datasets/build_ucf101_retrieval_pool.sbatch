#!/bin/bash
#SBATCH --job-name=build_ucf101_pool_max
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=datasets/slurm_log/build_ucf101_pool_max_%j.out
#SBATCH --error=datasets/slurm_log/build_ucf101_pool_max_%j.err

# ============================================================================
# Build a CHUNKED UCF-101 retrieval pool (~28-32K chunks, ~50 GB).
#
# Why chunked: UCF-101 has only 13,320 source clips, so a 1-clip-per-source
# pool would top out at ~12K entries. The TTA pipeline only reads the first
# `tta_total_frames` (=48) of each pool clip (~1.92 s at 25 fps), so slicing
# each source into multiple 3-second non-overlapping chunks creates extra
# valid training segments without inventing fake samples. With default
# settings (chunk=3.0s, gap=0.5s, max=10/source) the pool reaches Panda
# 25K-pool parity.
#
# CPU + network only -- no GPU. ffprobe + ffmpeg are parallelised across
# CPU workers (default 8). Wall-time is generous (12h) to cover the initial
# RAR download + per-source probe + ~30K chunk encodes.
#
# Submit (default = chunking ENABLED, target 30K valid):
#   sbatch --account=torch_pr_36_mren datasets/build_ucf101_retrieval_pool.sbatch
#
# Resume (idempotent -- already-encoded chunks and already-manifested sources
# are skipped):
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,SKIP_DOWNLOAD=1 datasets/build_ucf101_retrieval_pool.sbatch
#
# Legacy 1-clip-per-source mode (~12K pool, no chunking):
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,CHUNKING=0,TARGET_VALID=12000 \
#       datasets/build_ucf101_retrieval_pool.sbatch
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

NEW="${NEW:-${PROJECT_ROOT}/datasets/ucf101_pool_max}"
RAR_SOURCE="${RAR_SOURCE:-${PROJECT_ROOT}/datasets/ucf101_source}"
TARGET_VALID="${TARGET_VALID:-30000}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Chunking knobs.
CHUNKING="${CHUNKING:-1}"
CHUNK_SECONDS="${CHUNK_SECONDS:-3.0}"
CHUNK_GAP="${CHUNK_GAP:-0.5}"
MAX_CHUNKS_PER_SOURCE="${MAX_CHUNKS_PER_SOURCE:-10}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_RESIZE="${SKIP_RESIZE:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

echo "=============================================================================="
echo "Build UCF-101 retrieval pool (CHUNKED max-size)"
echo "=============================================================================="
echo "  New              : ${NEW}"
echo "  Rar source       : ${RAR_SOURCE}"
echo "  Target valid     : ${TARGET_VALID}"
echo "  Resize workers   : ${NUM_WORKERS}"
echo "  Chunking enabled : ${CHUNKING}"
echo "  Chunk seconds    : ${CHUNK_SECONDS}"
echo "  Chunk gap        : ${CHUNK_GAP}"
echo "  Max chunks/src   : ${MAX_CHUNKS_PER_SOURCE}"
echo "  Skip download    : ${SKIP_DOWNLOAD}"
echo "  Skip resize      : ${SKIP_RESIZE}"
echo "  Skip validate    : ${SKIP_VALIDATE}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${ENV_PATH}/bin:${PATH}"
PYTHON="${ENV_PATH}/bin/python"

# Tooling: ffmpeg + ffprobe + a rar extractor.
"${PYTHON}" -c "import av" 2>/dev/null || pip install av --quiet
if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
    conda install -y -c conda-forge ffmpeg --quiet 2>/dev/null || true
fi
if ! command -v unrar &>/dev/null && ! command -v 7z &>/dev/null \
        && ! command -v bsdtar &>/dev/null; then
    echo "Installing unrar via conda..."
    conda install -y -c conda-forge unrar --quiet 2>/dev/null || true
fi
if ! command -v unrar &>/dev/null && ! command -v 7z &>/dev/null \
        && ! command -v bsdtar &>/dev/null; then
    echo "ERROR: no RAR extractor available (tried unrar, 7z, bsdtar). "
    echo "       Try: conda install -c conda-forge unrar"
    exit 2
fi

mkdir -p "${PROJECT_ROOT}/datasets/slurm_log"
cd "${PROJECT_ROOT}"

EXTRA_FLAGS=""
[ "${SKIP_DOWNLOAD}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-download"
[ "${SKIP_RESIZE}"   = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-resize"
[ "${SKIP_VALIDATE}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-validate"
[ "${CHUNKING}"      = "0" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --no-chunking"

"${PYTHON}" scripts/build_ucf101_retrieval_pool.py \
    --new-dataset "${NEW}" \
    --rar-source "${RAR_SOURCE}" \
    --target-valid "${TARGET_VALID}" \
    --num-workers "${NUM_WORKERS}" \
    --chunk-seconds "${CHUNK_SECONDS}" \
    --chunk-gap "${CHUNK_GAP}" \
    --max-chunks-per-source "${MAX_CHUNKS_PER_SOURCE}" \
    ${EXTRA_FLAGS}

echo ""
echo "=============================================================================="
echo "UCF pool build job finished"
echo "  Dataset    : ${NEW}"
echo "  Videos     : $(ls ${NEW}/videos/*.mp4 2>/dev/null | wc -l)"
echo "  Du -sh     : $(du -sh ${NEW} 2>/dev/null)"
echo "  Report     : ${NEW}/validation_report.json"
echo "=============================================================================="
echo ""
echo "Next step: pre-compute caption embeddings"
echo "  sbatch --account=torch_pr_36_mren \\"
echo "      --export=ALL,POOL_DIR=${NEW} \\"
echo "      delta_experiment/sbatch/precompute_pool_embeddings.sbatch"
echo "=============================================================================="
