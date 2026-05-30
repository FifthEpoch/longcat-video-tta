#!/bin/bash
#SBATCH --job-name=build_panda_pool_10k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --time=20:00:00
#SBATCH --output=datasets/slurm_log/build_panda_pool_10k_%j.out
#SBATCH --error=datasets/slurm_log/build_panda_pool_10k_%j.err

# ============================================================================
# Build a head-trimmed Panda-70M retrieval pool (~10K clips, ~30 GB).
#
# Why head-trimmed: the TTA pipeline only reads the first `tta_total_frames`
# (=48) of each pool clip (~1.92 s at 25 fps). Filtering Panda-70M HF
# metadata to clips with reported duration in [2.5, 6.0] s and letting
# yt-dlp's `--download-sections *<start>-<end>` pull only that range keeps
# disk per clip at ~3 MB instead of the ~73 MB / clip we get on the
# eval-set ``panda_2048_480p`` (which keeps the full HF clip range,
# typically minutes long).
#
# Why a 20 h SLURM job is RESUMABLE:
#   yt-dlp success rate on Panda-70M URLs is ~30-40%, so 10K successful
#   clips take ~50-80 h of serial download time. The wrapper script
#   passes `--resume` to the underlying download_panda70m_subset.py,
#   which reads the existing manifest.jsonl and skips already-downloaded
#   videoIDs. Re-submitting this sbatch up to ~4 times will reach the
#   target.
#
# Submit (first run):
#   sbatch --account=torch_pr_36_mren \
#       datasets/build_panda_retrieval_pool.sbatch
#
# Re-submit to continue from where the previous job timed out:
#   sbatch --account=torch_pr_36_mren \
#       datasets/build_panda_retrieval_pool.sbatch
#   (no extra flags; --resume is on by default in the wrapper)
#
# Tweakable knobs (pass via --export=ALL,KEY=VAL):
#   TARGET_VALID         : final pool size after validation+trim (default 10000)
#   OVERSAMPLE_TO        : --num-videos passed to the download script (default 12000)
#   CANDIDATE_MULTIPLIER : sample N * mult candidates (default 3)
#   HF_MAX_ROWS          : cap on rows pulled from HF Panda-70M stream (default 300000)
#   MIN_DURATION         : drop clips shorter than this (default 2.5)
#   MAX_DURATION         : drop clips longer than this -- this controls disk (default 6.0)
#   COOKIES_FILE         : optional YouTube cookies.txt for age-gated/region-blocked clips
#
# Notes:
#   - --gres=gpu is intentionally NOT requested. yt-dlp + ffmpeg are
#     CPU + network bound; a GPU would idle for the entire 20 h run.
#   - HF stream + yt-dlp + ffmpeg are stable enough to share one job;
#     the script handles per-clip yt-dlp errors internally.
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

NEW="${NEW:-${PROJECT_ROOT}/datasets/panda_pool_10k}"
TARGET_VALID="${TARGET_VALID:-10000}"
OVERSAMPLE_TO="${OVERSAMPLE_TO:-12000}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-3}"
HF_MAX_ROWS="${HF_MAX_ROWS:-300000}"
MIN_DURATION="${MIN_DURATION:-2.5}"
MAX_DURATION="${MAX_DURATION:-6.0}"
DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-120}"
SEED="${SEED:-42}"
COOKIES_FILE="${COOKIES_FILE:-}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/hf-cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" \
    "${PROJECT_ROOT}/datasets/slurm_log"

echo "=============================================================================="
echo "Build Panda-70M retrieval pool (head-trimmed)"
echo "=============================================================================="
echo "  New dataset        : ${NEW}"
echo "  Target valid       : ${TARGET_VALID}"
echo "  Oversample to      : ${OVERSAMPLE_TO}"
echo "  Candidate mult.    : ${CANDIDATE_MULTIPLIER}"
echo "  HF max rows        : ${HF_MAX_ROWS}"
echo "  Duration window    : [${MIN_DURATION}, ${MAX_DURATION}] s"
echo "  Download timeout   : ${DOWNLOAD_TIMEOUT} s"
echo "  Cookies file       : ${COOKIES_FILE:-(none)}"
echo "  Skip download      : ${SKIP_DOWNLOAD}"
echo "  Skip validate      : ${SKIP_VALIDATE}"
echo "  Job ID             : ${SLURM_JOB_ID:-(not slurm)}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${ENV_PATH}/bin:${PATH}"
PYTHON="${ENV_PATH}/bin/python"

# Tooling sanity-check.
if ! command -v yt-dlp &>/dev/null; then
    echo "Installing yt-dlp via pip..."
    pip install -U yt-dlp --quiet || true
fi
if ! command -v yt-dlp &>/dev/null; then
    echo "ERROR: yt-dlp not available after install attempt."
    exit 2
fi
if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
    conda install -y -c conda-forge ffmpeg --quiet 2>/dev/null || true
fi

cd "${PROJECT_ROOT}"

EXTRA_FLAGS=""
[ "${SKIP_DOWNLOAD}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-download"
[ "${SKIP_VALIDATE}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-validate"
[ -n "${COOKIES_FILE}" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --cookies-file ${COOKIES_FILE}"

"${PYTHON}" scripts/build_panda_retrieval_pool.py \
    --new-dataset "${NEW}" \
    --target-valid "${TARGET_VALID}" \
    --oversample-to "${OVERSAMPLE_TO}" \
    --candidate-multiplier "${CANDIDATE_MULTIPLIER}" \
    --hf-max-rows "${HF_MAX_ROWS}" \
    --min-duration "${MIN_DURATION}" \
    --max-duration "${MAX_DURATION}" \
    --download-timeout "${DOWNLOAD_TIMEOUT}" \
    --seed "${SEED}" \
    ${EXTRA_FLAGS}

echo ""
echo "=============================================================================="
echo "Panda pool build job finished"
echo "  Dataset    : ${NEW}"
echo "  Videos     : $(ls ${NEW}/videos/*.mp4 2>/dev/null | wc -l)"
echo "  Du -sh     : $(du -sh ${NEW} 2>/dev/null | cut -f1)"
echo "  Manifest   : $(wc -l < ${NEW}/manifest.jsonl 2>/dev/null) entries"
echo "=============================================================================="
echo ""
echo "If the manifest is short of TARGET_VALID after this run (yt-dlp success"
echo "rate is ~30-40 percent), simply re-submit this sbatch -- the wrapper"
echo "passes --resume by default and the underlying download script skips"
echo "already-downloaded video IDs."
echo ""
echo "Once the pool reaches TARGET_VALID, pre-compute caption embeddings:"
echo "  sbatch --account=torch_pr_36_mren \\"
echo "      --export=ALL,POOL_DIR=${NEW} \\"
echo "      delta_experiment/sbatch/precompute_pool_embeddings.sbatch"
echo "=============================================================================="
