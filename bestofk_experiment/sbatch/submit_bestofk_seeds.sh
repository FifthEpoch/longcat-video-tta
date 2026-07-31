#!/bin/bash
# ============================================================================
# Submit best-of-k seed generation over the 1000v OOD preview pool, chunked.
# Each chunk lands in <OUTPUT_ROOT>/chunk_<n>/summary.json; point the offline
# probe at OUTPUT_ROOT (it merges chunk summaries):
#
#   python3 scripts/analyze_bestofk_headroom.py \
#     --summary bestofk_experiment/results/panda_1000v_k8 \
#     --output-dir sweep_experiment/reports/per_video_analysis/2026-07-31/bestofk
#
# Prototype default: first 200 videos, K=8 seeds, 25 videos/chunk (8 chunks).
# Scale up with MAX_VIDEOS / NUM_SEEDS / CHUNK_SIZE.
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   bash bestofk_experiment/sbatch/submit_bestofk_seeds.sh
#   DRY_RUN=1 bash bestofk_experiment/sbatch/submit_bestofk_seeds.sh   # preview
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SBATCH="${SBATCH:-bestofk_experiment/sbatch/run_bestofk_seeds.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
OUTPUT_ROOT="${OUTPUT_ROOT:-bestofk_experiment/results/panda_1000v_k8}"

MAX_VIDEOS="${MAX_VIDEOS:-200}"
CHUNK_SIZE="${CHUNK_SIZE:-25}"
NUM_SEEDS="${NUM_SEEDS:-8}"
SEED_STRIDE="${SEED_STRIDE:-1000}"
NO_SAVE_VIDEOS="${NO_SAVE_VIDEOS:-0}"
TIME_BUDGET="${TIME_BUDGET:-14:00:00}"
DRY_RUN="${DRY_RUN:-0}"

NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
NUM_FRAMES="${NUM_FRAMES:-28}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"

_exec_or_dry() {
    if [ "${DRY_RUN}" = "1" ]; then echo "[DRY] $*"; return 0; fi
    "$@"
}

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: dataset not found: ${DATA_DIR}" >&2
    exit 1
fi

echo "============================================================"
echo "Best-of-k seed generation submission"
echo "  data_dir    : ${DATA_DIR}"
echo "  output_root : ${OUTPUT_ROOT}"
echo "  max_videos  : ${MAX_VIDEOS}  chunk_size=${CHUNK_SIZE}  K=${NUM_SEEDS}"
echo "  geometry    : cond=${NUM_COND_FRAMES} frames=${NUM_FRAMES} gsf=${GEN_START_FRAME}"
echo "  dry_run     : ${DRY_RUN}"
echo "============================================================"

count=0
chunk=0
start=0
while [ "${start}" -lt "${MAX_VIDEOS}" ]; do
    out_dir="${PROJECT_ROOT}/${OUTPUT_ROOT}/chunk_${chunk}"
    job_name="bestofk_c${chunk}"
    _exec_or_dry sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${TIME_BUDGET}" \
        --export="ALL,DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_SEEDS=${NUM_SEEDS},SEED_STRIDE=${SEED_STRIDE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},NO_SAVE_VIDEOS=${NO_SAVE_VIDEOS},SEED=42" \
        "${SBATCH}"
    count=$((count + 1))
    chunk=$((chunk + 1))
    start=$((start + CHUNK_SIZE))
done

echo ""
echo "Submitted ${count} best-of-k chunks -> ${OUTPUT_ROOT}/chunk_*"
echo "After completion, run the routability probe:"
echo "  python3 scripts/analyze_bestofk_headroom.py --summary ${OUTPUT_ROOT} \\"
echo "    --output-dir sweep_experiment/reports/per_video_analysis/2026-07-31/bestofk"
