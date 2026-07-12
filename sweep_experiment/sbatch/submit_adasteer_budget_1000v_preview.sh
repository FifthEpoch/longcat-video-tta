#!/bin/bash
# ============================================================================
# Submit 12-config AdaSteer budget grid on the OOD-preview 1000v set.
#
# Prerequisite:
#   bash scripts/sample_segment_pool_ood_preview_1000v.sh
#
# Uses separate paths from the stale partial panda_ood_budget_1000v series
# (S2/S10/S20 @ 1e-2 on old panda_1000_480p eval).
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_preview.sh
#
# Dry-run:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_preview.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/panda_ood_budget_1000v_preview}"
SERIES_NAME="${SERIES_NAME:-panda_ood_budget_1000v_preview}"

NUM_CHUNKS="${NUM_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"

TIME_BUDGET="${TIME_BUDGET:-14:00:00}"

NUM_FRAMES="${NUM_FRAMES:-28}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

DRY_RUN="${DRY_RUN:-0}"
ONLY_RUNS="${ONLY_RUNS:-}"
NO_SAVE_VIDEOS="${NO_SAVE_VIDEOS:-1}"

# Same 12-config grid as pilot (not the 20-config full grid).
PREVIEW_RUNS=(
    "S2_LR1e3:2:1.0e-3"
    "S2_LR5e3:2:5.0e-3"
    "S2_LR1e2:2:1.0e-2"
    "S5_LR1e3:5:1.0e-3"
    "S5_LR5e3:5:5.0e-3"
    "S5_LR1e2:5:1.0e-2"
    "S10_LR1e3:10:1.0e-3"
    "S10_LR5e3:10:5.0e-3"
    "S10_LR1e2:10:1.0e-2"
    "S20_LR1e3:20:1.0e-3"
    "S20_LR5e3:20:5.0e-3"
    "S20_LR1e2:20:1.0e-2"
)

count=0

_in_filter() {
    local needle="$1"
    [ -z "${ONLY_RUNS}" ] && return 0
    for m in ${ONLY_RUNS}; do
        if [ "${m}" = "${needle}" ]; then return 0; fi
    done
    return 1
}

_exec_or_dry() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY] $*"
        return 0
    fi
    "$@"
}

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: preview dataset not found: ${DATA_DIR}" >&2
    echo "Run: bash scripts/sample_segment_pool_ood_preview_1000v.sh" >&2
    exit 1
fi

echo "============================================================"
echo "AdaSteer budget-grid 1000v PREVIEW submission"
echo "============================================================"
echo "  account      : ${ACCOUNT}"
echo "  data_dir     : ${DATA_DIR}"
echo "  results      : ${PROJECT_ROOT}/${RESULTS_SUBDIR}"
echo "  configs      : ${#PREVIEW_RUNS[@]} runs"
echo "  chunking     : ${NUM_CHUNKS} × ${CHUNK_SIZE} (max ${MAX_VIDEOS} videos)"
echo "  dry run      : ${DRY_RUN}"
echo "  only runs    : ${ONLY_RUNS:-<all>}"
echo "============================================================"
echo ""

for spec in "${PREVIEW_RUNS[@]}"; do
    IFS=":" read -r run_id delta_steps delta_lr <<< "${spec}"
    if ! _in_filter "${run_id}"; then continue; fi

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        start=$((chunk * CHUNK_SIZE))
        if [ "${start}" -ge "${MAX_VIDEOS}" ]; then break; fi

        out_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${run_id}/chunk_${chunk}"
        job_name="adb1k_prev_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${TIME_BUDGET}" \
            --export="ALL,METHOD=delta_a,RUN_ID=${run_id},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},DELTA_STEPS=${delta_steps},DELTA_LR=${delta_lr},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=0,NO_SAVE_VIDEOS=${NO_SAVE_VIDEOS},CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
done

echo ""
echo "Submitted ${count} jobs."
echo ""
echo "After completion:"
echo "  bash scripts/run_preview_1000v_pipeline.sh merge"
echo "  bash scripts/run_preview_1000v_pipeline.sh features"
echo ""
echo "For VBench routers (needs mp4s):"
echo "  bash scripts/run_preview_1000v_pipeline.sh sweep-mp4"
echo "  bash scripts/run_preview_1000v_pipeline.sh vbench"
echo "  bash scripts/run_preview_1000v_pipeline.sh routers"
echo "============================================================"
