#!/bin/bash
# ============================================================================
# Submit 1000v AdaSteer best-budget configs with incremental S10→S20 training.
#
# One job per chunk runs:
#   S10_LR1e2: 10 steps @ LR=1e-2, save delta, generate
#   S20_LR1e2: +10 steps from S10 delta (20 total), generate
#
# Saves ~33% TTA compute vs independent S10+S20 from scratch (10+10 vs 10+20).
#
# Prerequisites:
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#
# Submit:
#   bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_best_incremental.sh
#
# Dry-run:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_best_incremental.sh
#
# Oracle FVD (optional; requires NO_SAVE_VIDEOS=0 on a re-run):
#   NO_SAVE_VIDEOS=0 bash ...   # or separate eval job
#   sbatch sweep_experiment/sbatch/run_budget_oracle_fvd.sbatch
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
INCREMENTAL_SBATCH="${INCREMENTAL_SBATCH:-sweep_experiment/sbatch/run_adasteer_budget_incremental.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/panda_ood_budget_1000v}"
SERIES_NAME="${SERIES_NAME:-panda_ood_budget_1000v}"

RUN_S10="${RUN_S10:-S10_LR1e2}"
RUN_S20="${RUN_S20:-S20_LR1e2}"
DELTA_LR="${DELTA_LR:-1.0e-2}"

NUM_CHUNKS="${NUM_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"
TIME_BUDGET="${TIME_BUDGET:-16:00:00}"

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
echo "1000v best-budget INCREMENTAL submission"
echo "============================================================"
echo "  account      : ${ACCOUNT}"
echo "  data_dir     : ${DATA_DIR}"
echo "  results      : ${PROJECT_ROOT}/${RESULTS_SUBDIR}"
echo "  configs      : ${RUN_S10} → ${RUN_S20} (LR=${DELTA_LR})"
echo "  chunking     : ${NUM_CHUNKS} × ${CHUNK_SIZE} = ${MAX_VIDEOS} videos"
echo "  dry run      : ${DRY_RUN}"
echo "============================================================"
echo ""

count=0
for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
    start=$((chunk * CHUNK_SIZE))
    if [ "${start}" -ge "${MAX_VIDEOS}" ]; then break; fi

    s10_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_S10}/chunk_${chunk}"
    s20_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_S20}/chunk_${chunk}"
    job_name="adb1k_inc_c${chunk}"

    _exec_or_dry sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${TIME_BUDGET}" \
        --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},CHUNK_IDX=${chunk},RUN_ID_S10=${RUN_S10},RUN_ID_S20=${RUN_S20},DELTA_LR=${DELTA_LR},S10_OUTPUT_DIR=${s10_dir},S20_OUTPUT_DIR=${s20_dir},NO_SAVE_VIDEOS=${NO_SAVE_VIDEOS:-1}" \
        "${INCREMENTAL_SBATCH}"
    count=$((count + 1))
done

echo ""
echo "============================================================"
echo "Submitted ${count} incremental jobs (${RUN_S10}+${RUN_S20} per job)."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* adb1k_inc_'"
echo ""
echo "After completion:"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/${RESULTS_SUBDIR} --recursive"
echo ""
echo "  python scripts/analyze_adasteer_budget_oracle.py --bootstrap \\"
echo "      --series-root ${PROJECT_ROOT}/${RESULTS_SUBDIR} \\"
echo "      --baseline-series-root ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_standard \\"
echo "      --fixed-run-id ${RUN_S10} \\"
echo "      --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\"
echo "      --output sweep_experiment/reports/per_video_analysis/\$(date +%Y-%m-%d)/adasteer_budget_1000v_best.md"
echo "============================================================"
