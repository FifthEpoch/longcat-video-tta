#!/bin/bash
# ============================================================================
# 1000v budget configs with **inline VBench++** for PI review (4-day window).
#
# Submits pilot-validated configs at full 999v scale (NOT the 200v OOD pilot set):
#   S2_LR1e3  — modal VBench-oracle winner on budget pilot
#   S10_LR5e3 — deployable fixed AdaSteer (headline recipe)
#   S5_LR1e3  — secondary modal winner (optional; included by default)
#
# 3 configs × 10 chunks = 30 GPU jobs (~12h each if queue allows → ~1–2 days).
#
# Submit:
#   bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_vbench_review.sh
#
# Subset:
#   ONLY_RUNS="S2_LR1e3 S10_LR5e3" bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_vbench_review.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/panda_1000v_adasteer_budget_vbench}"
SERIES_NAME="${SERIES_NAME:-panda_1000v_adasteer_budget_vbench}"

NUM_CHUNKS="${NUM_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"
TIME_SHORT="${TIME_SHORT:-12:00:00}"
TIME_LONG="${TIME_LONG:-16:00:00}"

ONLY_RUNS="${ONLY_RUNS:-S2_LR1e3 S10_LR5e3 S5_LR1e3}"
DRY_RUN="${DRY_RUN:-0}"

GRID_RUNS=(
    "S2_LR1e3:2:1.0e-3"
    "S5_LR1e3:5:1.0e-3"
    "S10_LR5e3:10:5.0e-3"
)

NUM_FRAMES="${NUM_FRAMES:-28}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

_in_filter() {
    local needle="$1"
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

_pick_time() {
    local steps="$1"
    if [ "${steps}" -ge 20 ]; then echo "${TIME_LONG}"; else echo "${TIME_SHORT}"; fi
}

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: data_dir not found: ${DATA_DIR}" >&2
    exit 1
fi

echo "============================================================"
echo "1000v budget configs + inline VBench++ (review sweep)"
echo "============================================================"
echo "  account   : ${ACCOUNT}"
echo "  results   : ${PROJECT_ROOT}/${RESULTS_SUBDIR}"
echo "  configs   : ${ONLY_RUNS}"
echo "  dry run   : ${DRY_RUN}"
echo "============================================================"

count=0
for spec in "${GRID_RUNS[@]}"; do
    IFS=":" read -r run_id delta_steps delta_lr <<< "${spec}"
    if ! _in_filter "${run_id}"; then continue; fi
    wall_time="$(_pick_time "${delta_steps}")"

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        start=$((chunk * CHUNK_SIZE))
        if [ "${start}" -ge "${MAX_VIDEOS}" ]; then break; fi

        out_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${run_id}/chunk_${chunk}"
        job_name="vb1k_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${wall_time}" \
            --export="ALL,METHOD=delta_a,RUN_ID=${run_id},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},DELTA_STEPS=${delta_steps},DELTA_LR=${delta_lr},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
done

echo ""
echo "Submitted ${count} jobs."
echo "After merge + VBench refresh:"
echo "  python scripts/analyze_adasteer_budget_vbench_oracle.py --bootstrap \\"
echo "    --series-root ${PROJECT_ROOT}/${RESULTS_SUBDIR} \\"
echo "    --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\"
echo "    --output sweep_experiment/reports/per_video_analysis/\$(date +%Y-%m-%d)/adasteer_budget_vbench_1000v.md"
echo "============================================================"
