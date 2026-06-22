#!/bin/bash
# ============================================================================
# Submit the full 1000-video AdaSteer budget-grid sweep (H9), chunked.
#
# Full grid: 20 configs (5 LRs × 4 steps) documented in
# sweep_experiment/configs/panda_1000v_adasteer_budget_grid.yaml.
#
# Chunking: 10 chunks × 100 videos = 1000 total per config.
# Metrics-only (NO_SAVE_VIDEOS=1) to keep disk usage manageable.
#
# Submit (after `git pull`):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_chunked.sh
#
# Dry-run:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_chunked.sh
#
# Pilot first (recommended):
#   bash sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh
#
# Subset of configs:
#   ONLY_RUNS="S10_LR5e3 S10_LR1e3" bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_chunked.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/panda_1000v_adasteer_budget}"
SERIES_NAME="${SERIES_NAME:-panda_1000v_adasteer_budget}"

NUM_CHUNKS="${NUM_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"

# Wall-time: up to 20 steps × ~5 min/video ≈ 16 h for worst configs.
TIME_SHORT="${TIME_SHORT:-12:00:00}"
TIME_LONG="${TIME_LONG:-16:00:00}"

NUM_FRAMES="${NUM_FRAMES:-28}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

DRY_RUN="${DRY_RUN:-0}"
ONLY_RUNS="${ONLY_RUNS:-}"

# Full 20-config grid: run_id:steps:lr
GRID_RUNS=(
    "S2_LR1e3:2:1.0e-3"
    "S2_LR2p5e3:2:2.5e-3"
    "S2_LR5e3:2:5.0e-3"
    "S2_LR7p5e3:2:7.5e-3"
    "S2_LR1e2:2:1.0e-2"
    "S5_LR1e3:5:1.0e-3"
    "S5_LR2p5e3:5:2.5e-3"
    "S5_LR5e3:5:5.0e-3"
    "S5_LR7p5e3:5:7.5e-3"
    "S5_LR1e2:5:1.0e-2"
    "S10_LR1e3:10:1.0e-3"
    "S10_LR2p5e3:10:2.5e-3"
    "S10_LR5e3:10:5.0e-3"
    "S10_LR7p5e3:10:7.5e-3"
    "S10_LR1e2:10:1.0e-2"
    "S20_LR1e3:20:1.0e-3"
    "S20_LR2p5e3:20:2.5e-3"
    "S20_LR5e3:20:5.0e-3"
    "S20_LR7p5e3:20:7.5e-3"
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

_pick_time() {
    local steps="$1"
    if [ "${steps}" -ge 20 ]; then
        echo "${TIME_LONG}"
    else
        echo "${TIME_SHORT}"
    fi
}

if [ ! -d "${DATA_DIR}" ]; then
    echo "ERROR: data_dir not found: ${DATA_DIR}" >&2
    exit 1
fi

echo "============================================================"
echo "1000v AdaSteer budget-grid submission (H9, full 20-config)"
echo "============================================================"
echo "  account      : ${ACCOUNT}"
echo "  data_dir     : ${DATA_DIR}"
echo "  results      : ${PROJECT_ROOT}/${RESULTS_SUBDIR}"
echo "  configs      : ${#GRID_RUNS[@]} grid runs"
echo "  chunking     : ${NUM_CHUNKS} × ${CHUNK_SIZE} = ${MAX_VIDEOS} videos"
echo "  dry run      : ${DRY_RUN}"
echo "  only runs    : ${ONLY_RUNS:-<all>}"
echo "============================================================"
echo ""

for spec in "${GRID_RUNS[@]}"; do
    IFS=":" read -r run_id delta_steps delta_lr <<< "${spec}"
    if ! _in_filter "${run_id}"; then continue; fi
    wall_time="$(_pick_time "${delta_steps}")"

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        start=$((chunk * CHUNK_SIZE))
        if [ "${start}" -ge "${MAX_VIDEOS}" ]; then break; fi

        out_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${run_id}/chunk_${chunk}"
        job_name="adb1k_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${wall_time}" \
            --export="ALL,METHOD=delta_a,RUN_ID=${run_id},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},DELTA_STEPS=${delta_steps},DELTA_LR=${delta_lr},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=0,NO_SAVE_VIDEOS=1,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
done

echo ""
echo "============================================================"
echo "Submitted ${count} jobs."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* adb1k_'"
echo ""
echo "After completion, merge chunks:"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/${RESULTS_SUBDIR} --recursive"
echo ""
echo "Analyze oracle uplift:"
echo "  python scripts/analyze_adasteer_budget_oracle.py --bootstrap \\"
echo "      --series-root ${PROJECT_ROOT}/${RESULTS_SUBDIR} \\"
echo "      --baseline-series-root ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_standard \\"
echo "      --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\"
echo "      --output sweep_experiment/reports/per_video_analysis/\$(date +%Y-%m-%d)/adasteer_budget_oracle.md"
echo "============================================================"
