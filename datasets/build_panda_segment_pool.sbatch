#!/bin/bash
#SBATCH --job-name=build_panda_segment_pool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=datasets/slurm_log/build_panda_segment_pool_%j.out
#SBATCH --error=datasets/slurm_log/build_panda_segment_pool_%j.err

# ============================================================================
# Build a per-SEGMENT Panda-70M retrieval pool (~6K-50K segments) by
# ffmpeg-cutting the existing 480p source videos at the segment timestamps
# stored in panda70m_training_2m.csv. NO YouTube downloads.
#
# Why this exists:
#   - Previous Panda pool builder downloaded full source videos and emitted
#     one row per source with a bracketed multi-caption string. That gives
#     2048 unique caption embeddings -- noisy as similarity queries.
#   - The Panda-70M source metadata stores PER-SEGMENT timestamps and
#     captions for every video as parallel nested lists, so we can extract
#     each segment as its own pool entry with its own clean caption.
#   - YouTube blocked our previous download attempt (panda_pool_10k job
#     finished with 0 valid videos after 20h of HTTP 429/403 + cookie wall).
#     This builder does not download anything from YouTube.
#
# Output (matches the schema of ucf101_pool_max/metadata.csv exactly, so
# common.py treats both pools identically):
#   videoID         = "<source_videoID>_seg<idx>"
#   source_video_id = the original Panda/YouTube videoID  (same-source filter)
#   chunk_index     = segment index within the source video
#   chunk_start_sec = segment start time (parsed from Panda timestamp column)
#   caption         = clean per-segment caption (UCF-grade)
#
# CPU-only, ~16 workers in parallel, ~3-5K segments expected from the
# 800K-row training_2m subset (3 segments/video x 2048 matched videos =
# ~6.1K segments before filtering, ~5K after the desirable+score+duration
# filters). For ~25-50K segments, point --source-metadata at the FULL
# Panda-70M metadata CSV (not the 2m subset).
#
# Submit (default = filter to "desirable" segments, keep all valid):
#   sbatch --account=torch_pr_36_mren \
#       datasets/build_panda_segment_pool.sbatch
#
# Resume (idempotent: already-cut segments + already-manifested sources are
# skipped):
#   re-submit the same sbatch.
#
# Override the source metadata to use the FULL Panda-70M CSV (~10x more
# segments per video):
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,SOURCE_METADATA=/path/to/panda70m_full.csv \
#       datasets/build_panda_segment_pool.sbatch
#
# Smoke-test on first 200 segments:
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,SEGMENT_LIMIT=200 \
#       datasets/build_panda_segment_pool.sbatch
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

NEW="${NEW:-${PROJECT_ROOT}/datasets/panda_segment_pool}"
SOURCE_POOL="${SOURCE_POOL:-${PROJECT_ROOT}/datasets/panda_2048_480p}"
SOURCE_METADATA="${SOURCE_METADATA:-${PROJECT_ROOT}/datasets/panda_pool_10k/panda70m_training_2m.csv}"

NUM_WORKERS="${NUM_WORKERS:-16}"
TARGET_WIDTH="${TARGET_WIDTH:-832}"
TARGET_HEIGHT="${TARGET_HEIGHT:-480}"
MIN_SEG_SEC="${MIN_SEG_SEC:-2.0}"
MAX_SEG_SEC="${MAX_SEG_SEC:-20.0}"
MIN_SCORE="${MIN_SCORE:-0.0}"
REQUIRE_DESIRABLE="${REQUIRE_DESIRABLE:-1}"
MAX_SEGS_PER_SOURCE="${MAX_SEGS_PER_SOURCE:-0}"
SEGMENT_LIMIT="${SEGMENT_LIMIT:-0}"
MIN_FRAMES="${MIN_FRAMES:-48}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

echo "=============================================================================="
echo "Build Panda-70M PER-SEGMENT retrieval pool"
echo "=============================================================================="
echo "  New              : ${NEW}"
echo "  Source pool      : ${SOURCE_POOL}"
echo "  Source metadata  : ${SOURCE_METADATA}"
echo "  Workers          : ${NUM_WORKERS}"
echo "  Target res       : ${TARGET_WIDTH}x${TARGET_HEIGHT}"
echo "  Segment dur      : [${MIN_SEG_SEC}, ${MAX_SEG_SEC}] s"
echo "  Min score        : ${MIN_SCORE}"
echo "  Require desirable: ${REQUIRE_DESIRABLE}"
echo "  Max segs/source  : ${MAX_SEGS_PER_SOURCE} (0 = no cap)"
echo "  Segment limit    : ${SEGMENT_LIMIT} (0 = unlimited)"
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

# Tooling: ffmpeg + ffprobe must be on PATH. Try conda-forge as a fallback.
if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
    echo "ffmpeg / ffprobe not on PATH; installing via conda-forge..."
    conda install -y -c conda-forge ffmpeg --quiet 2>/dev/null || true
fi
if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
    echo "ERROR: no ffmpeg/ffprobe available; aborting." >&2
    exit 2
fi

mkdir -p "${PROJECT_ROOT}/datasets/slurm_log"
cd "${PROJECT_ROOT}"

EXTRA_FLAGS=""
[ "${REQUIRE_DESIRABLE}" = "0" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --no-require-desirable"
[ "${SKIP_VALIDATE}"     = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-validate"

"${PYTHON}" scripts/build_panda_segment_pool.py \
    --source-pool      "${SOURCE_POOL}" \
    --source-metadata  "${SOURCE_METADATA}" \
    --new-dataset      "${NEW}" \
    --workers          "${NUM_WORKERS}" \
    --target-width     "${TARGET_WIDTH}" \
    --target-height    "${TARGET_HEIGHT}" \
    --min-segment-seconds "${MIN_SEG_SEC}" \
    --max-segment-seconds "${MAX_SEG_SEC}" \
    --min-matching-score  "${MIN_SCORE}" \
    --max-segments-per-source "${MAX_SEGS_PER_SOURCE}" \
    --limit            "${SEGMENT_LIMIT}" \
    --min-frames       "${MIN_FRAMES}" \
    ${EXTRA_FLAGS}

echo ""
echo "=============================================================================="
echo "Panda segment-pool build job finished"
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
