#!/bin/bash
# ============================================================================
# Submit the NO-TTA baseline on the OOD-preview 1000v set — the router's 13th
# candidate ("skip TTA"). Runs on the IDENTICAL pool / geometry / chunking as
# the 12-config AdaSteer budget grid so per-video metrics line up 1:1.
#
# No-TTA == METHOD=full with num_steps=0, lr=0 (pure conditioned continuation,
# no adaptation) — same convention as sweep_experiment/configs/panda_notta.yaml.
#
# Lands as sibling arm:  <PREVIEW_SERIES_ROOT>/NOTTA/chunk_*
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_notta_1000v_preview.sh
# Dry-run:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_notta_1000v_preview.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Same dataset + results tree as the 12-config grid (submit_adasteer_budget_1000v_preview.sh).
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/panda_ood_budget_1000v_preview}"
SERIES_NAME="${SERIES_NAME:-panda_ood_budget_1000v_preview}"
RUN_ID="${RUN_ID:-NOTTA}"

NUM_CHUNKS="${NUM_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"
TIME_BUDGET="${TIME_BUDGET:-8:00:00}"

NUM_FRAMES="${NUM_FRAMES:-28}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

DRY_RUN="${DRY_RUN:-0}"
# Save frames by default (needed for VBench backfill + reusable videos, and so
# NOTTA matches the grid arms which were run with NO_SAVE_VIDEOS=0).
NO_SAVE_VIDEOS="${NO_SAVE_VIDEOS:-0}"

_exec_or_dry() {
    if [ "${DRY_RUN}" = "1" ]; then echo "[DRY] $*"; return 0; fi
    "$@"
}

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: preview dataset not found: ${DATA_DIR}" >&2
    echo "Run: bash scripts/sample_segment_pool_ood_preview_1000v.sh" >&2
    exit 1
fi

# Dataset guard — same as the grid submitter: never launch against an
# incomplete / shifting dataset (that produced misaligned per-video sets).
VIDEO_DIR="${VIDEO_DIR:-${DATA_DIR}/videos}"
SKIP_DATASET_GUARD="${SKIP_DATASET_GUARD:-0}"
if [ "${SKIP_DATASET_GUARD}" != "1" ]; then
    if [ ! -d "${VIDEO_DIR}" ]; then
        echo "ERROR: video dir missing: ${VIDEO_DIR}" >&2; exit 1
    fi
    n_total=0; n_broken=0
    while IFS= read -r -d '' f; do
        n_total=$((n_total + 1))
        [ -e "${f}" ] || n_broken=$((n_broken + 1))
    done < <(find "${VIDEO_DIR}" -maxdepth 1 \( -type f -o -type l \) -print0)
    echo "Dataset guard: ${n_total} videos, ${n_broken} broken under ${VIDEO_DIR}"
    if [ "${n_total}" -lt "${MAX_VIDEOS}" ]; then
        echo "ERROR: only ${n_total} videos (< MAX_VIDEOS=${MAX_VIDEOS}). Dataset not fully materialized." >&2
        echo "Re-run: bash scripts/sample_segment_pool_ood_preview_1000v.sh  (SKIP_DATASET_GUARD=1 to override)" >&2
        exit 1
    fi
    if [ "${n_broken}" -gt 0 ]; then
        echo "ERROR: ${n_broken} broken/dangling video links (SKIP_DATASET_GUARD=1 to override)." >&2
        exit 1
    fi
fi

echo "============================================================"
echo "NO-TTA baseline 1000v PREVIEW submission (router 13th candidate)"
echo "============================================================"
echo "  account      : ${ACCOUNT}"
echo "  data_dir     : ${DATA_DIR}"
echo "  results      : ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}"
echo "  chunking     : ${NUM_CHUNKS} × ${CHUNK_SIZE} (max ${MAX_VIDEOS} videos)"
echo "  geometry     : cond=${NUM_COND_FRAMES} total=${NUM_FRAMES} gsf=${GEN_START_FRAME}"
echo "  dry run      : ${DRY_RUN}"
if [ "${NO_SAVE_VIDEOS}" = "1" ]; then
    echo "  save mp4s    : NO  (metrics-only — NO VBench, NO reusable videos)"
else
    echo "  save mp4s    : YES (frames written for VBench + downstream reuse)"
fi
echo "============================================================"
echo ""

count=0
for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
    start=$((chunk * CHUNK_SIZE))
    if [ "${start}" -ge "${MAX_VIDEOS}" ]; then break; fi

    out_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}/chunk_${chunk}"
    job_name="notta1k_prev_c${chunk}"

    _exec_or_dry sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${TIME_BUDGET}" \
        --export="ALL,METHOD=full,RUN_ID=${RUN_ID},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},LEARNING_RATE=0.0,NUM_STEPS=0,WARMUP_STEPS=0,WEIGHT_DECAY=0.0,MAX_GRAD_NORM=1.0,OPTIMIZER=sgd,BATCH_VIDEOS=1,NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=0,NO_SAVE_VIDEOS=${NO_SAVE_VIDEOS},CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn" \
        "${SWEEP_SBATCH}"
    count=$((count + 1))
done

echo ""
echo "Submitted ${count} NOTTA jobs -> ${RESULTS_SUBDIR}/${RUN_ID}"
echo ""
echo "After completion:"
echo "  bash scripts/run_preview_1000v_pipeline.sh merge     # merges NOTTA + the 12 arms"
echo "  bash scripts/run_preview_1000v_pipeline.sh vbench     # VBench on saved mp4s (incl. NOTTA)"
echo "============================================================"
