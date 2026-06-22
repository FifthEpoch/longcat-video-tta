#!/bin/bash
# ============================================================================
# Submit S2_LR1e2 on the full 1000v Panda set (companion to best incremental).
#
# Pilot winner for population PSNR (2 steps @ LR=1e-2). Standalone chunked
# runs — no incremental delta trick (only 2 steps).
#
# Results land alongside S10/S20 from submit_adasteer_budget_1000v_best_incremental.sh:
#   sweep_experiment/results/panda_ood_budget_1000v/S2_LR1e2/
#
# Prerequisites:
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#
# Submit:
#   bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_best_s2.sh
#
# Dry-run:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_best_s2.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/panda_ood_budget_1000v}"
SERIES_NAME="${SERIES_NAME:-panda_ood_budget_1000v}"

RUN_ID="${RUN_ID:-S2_LR1e2}"
DELTA_STEPS="${DELTA_STEPS:-2}"
DELTA_LR="${DELTA_LR:-1.0e-2}"

NUM_CHUNKS="${NUM_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"
TIME_BUDGET="${TIME_BUDGET:-12:00:00}"

NUM_FRAMES="${NUM_FRAMES:-28}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

DRY_RUN="${DRY_RUN:-0}"

_exec_or_dry() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY] $*"
        return 0
    fi
    "$@"
}

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: data_dir not found: ${DATA_DIR}" >&2
    exit 1
fi

echo "============================================================"
echo "1000v best-budget S2_LR1e2 submission (standalone chunked)"
echo "============================================================"
echo "  account      : ${ACCOUNT}"
echo "  data_dir     : ${DATA_DIR}"
echo "  results      : ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}"
echo "  config       : ${RUN_ID} (${DELTA_STEPS} steps, LR=${DELTA_LR})"
echo "  chunking     : ${NUM_CHUNKS} × ${CHUNK_SIZE} = ${MAX_VIDEOS} videos"
echo "  dry run      : ${DRY_RUN}"
echo "============================================================"
echo ""

count=0
for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
    start=$((chunk * CHUNK_SIZE))
    if [ "${start}" -ge "${MAX_VIDEOS}" ]; then break; fi

    out_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}/chunk_${chunk}"
    job_name="adb1k_${RUN_ID}_c${chunk}"

    _exec_or_dry sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${TIME_BUDGET}" \
        --export="ALL,METHOD=delta_a,RUN_ID=${RUN_ID},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},DELTA_STEPS=${DELTA_STEPS},DELTA_LR=${DELTA_LR},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=0,NO_SAVE_VIDEOS=${NO_SAVE_VIDEOS:-1},CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn" \
        "${SWEEP_SBATCH}"
    count=$((count + 1))
done

echo ""
echo "============================================================"
echo "Submitted ${count} jobs for ${RUN_ID}."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* adb1k_S2_LR1e2_'"
echo ""
echo "After completion:"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}"
echo ""
echo "  # Or merge all best-budget runs (S2 + S10 + S20):"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/${RESULTS_SUBDIR} --recursive"
echo ""
echo "  python scripts/analyze_adasteer_budget_oracle.py --bootstrap \\"
echo "      --series-root ${PROJECT_ROOT}/${RESULTS_SUBDIR} \\"
echo "      --baseline-series-root ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_standard \\"
echo "      --fixed-run-id ${RUN_ID} \\"
echo "      --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\"
echo "      --output sweep_experiment/reports/per_video_analysis/\$(date +%Y-%m-%d)/adasteer_budget_1000v_best.md"
echo "============================================================"
