#!/bin/bash
# ============================================================================
# Submit the OOD-quintile-stratified AdaSteer budget-grid PILOT (H9).
#
# Scope: 12 configs (LR 1e-3, 5e-3, 1e-2 × steps 2, 5, 10, 20) on the
# ~200-video pilot set (40 videos × 5 OOD quintiles) built by
# scripts/sample_ood_quintile_videos.py.
#
# Prerequisites (once per cluster checkout):
#   python scripts/sample_ood_quintile_videos.py \\
#       --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
#       --source-dataset datasets/panda_1000_480p \\
#       --output-json sweep_experiment/lists/panda_ood_budget_pilot_videos.json \\
#       --create-dataset datasets/panda_ood_budget_pilot_480p
#
# Submit (after `git pull`):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh
#
# Dry-run:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh
#
# Subset of configs:
#   ONLY_RUNS="S10_LR5e3 S10_LR1e3" bash sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh
#
# Oracle FVD mp4 re-run (metrics already exist; save videos only):
#   python sweep_experiment/scripts/plan_budget_oracle_fvd_rerun.py
#   ONLY_RUNS="<from planner>" NO_SAVE_VIDEOS=0 bash sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_pilot_480p}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/panda_ood_budget_pilot}"
SERIES_NAME="${SERIES_NAME:-panda_ood_budget_pilot}"

# 200 videos → 2 chunks × 100 (matches standard 1000v chunk size).
NUM_CHUNKS="${NUM_CHUNKS:-2}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-200}"

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

# 12-config pilot subset (see panda_1000v_adasteer_budget_grid.yaml header).
PILOT_RUNS=(
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
    echo "ERROR: pilot dataset not found: ${DATA_DIR}" >&2
    echo "Run sample_ood_quintile_videos.py with --create-dataset first." >&2
    exit 1
fi

echo "============================================================"
echo "AdaSteer budget-grid PILOT submission (H9)"
echo "============================================================"
echo "  account      : ${ACCOUNT}"
echo "  data_dir     : ${DATA_DIR}"
echo "  results      : ${PROJECT_ROOT}/${RESULTS_SUBDIR}"
echo "  configs      : ${#PILOT_RUNS[@]} pilot runs"
echo "  chunking     : ${NUM_CHUNKS} × ${CHUNK_SIZE} (max ${MAX_VIDEOS} videos)"
echo "  dry run      : ${DRY_RUN}"
echo "  only runs    : ${ONLY_RUNS:-<all>}"
echo "============================================================"
echo ""

for spec in "${PILOT_RUNS[@]}"; do
    IFS=":" read -r run_id delta_steps delta_lr <<< "${spec}"
    if ! _in_filter "${run_id}"; then continue; fi

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        start=$((chunk * CHUNK_SIZE))
        if [ "${start}" -ge "${MAX_VIDEOS}" ]; then break; fi

        out_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${run_id}/chunk_${chunk}"
        job_name="adb_pilot_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${TIME_BUDGET}" \
            --export="ALL,METHOD=delta_a,RUN_ID=${run_id},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},DELTA_STEPS=${delta_steps},DELTA_LR=${delta_lr},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=0,NO_SAVE_VIDEOS=${NO_SAVE_VIDEOS:-1},CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
done

echo ""
echo "============================================================"
echo "Submitted ${count} jobs."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* adb_pilot_'"
echo ""
echo "After completion, merge chunks:"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/${RESULTS_SUBDIR} --recursive"
echo ""
echo "Analyze oracle uplift + full metrics table:"
echo "  python scripts/analyze_adasteer_budget_oracle.py --bootstrap \\"
echo "      --series-root ${PROJECT_ROOT}/${RESULTS_SUBDIR} \\"
echo "      --baseline-series-root ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_standard \\"
echo "      --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\"
echo "      --output sweep_experiment/reports/per_video_analysis/\$(date +%Y-%m-%d)/adasteer_budget_oracle_pilot.md"
echo ""
echo "Oracle FVD (requires saved mp4s; pilot used NO_SAVE_VIDEOS=1):"
echo "  sbatch --account=${ACCOUNT} sweep_experiment/sbatch/run_budget_oracle_fvd.sbatch"
echo "  # then re-run analyzer with --oracle-fvd-json sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/fvd.json"
echo "============================================================"
