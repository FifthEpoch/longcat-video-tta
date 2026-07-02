#!/bin/bash
# ============================================================================
# LoRA rank=1 @ 999v Panda — mirror validated R8 recipe (VBench++ for review).
#
# R8: rank=8, alpha=16, 10 steps, lr=5e-5, wd=0.01, warmup=3, max_grad_norm=10
# R1: rank=1, alpha=2  (same alpha/rank ratio), all else identical.
#
# Submit:
#   bash sweep_experiment/sbatch/submit_lora_r1_1000v_panda.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/panda_1000v_standard}"
SERIES_NAME="${SERIES_NAME:-panda_1000v}"
RUN_ID="${RUN_ID:-LORA_R1_TTA}"

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
echo "LoRA R1 @ 999v Panda (mirror R8 recipe, inline VBench++)"
echo "============================================================"
echo "  account      : ${ACCOUNT}"
echo "  run_id       : ${RUN_ID}"
echo "  results      : ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}"
echo "  chunking     : ${NUM_CHUNKS} × ${CHUNK_SIZE} = ${MAX_VIDEOS} videos"
echo "  dry run      : ${DRY_RUN}"
echo "============================================================"
echo ""

count=0
for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
    start=$((chunk * CHUNK_SIZE))
    if [ "${start}" -ge "${MAX_VIDEOS}" ]; then break; fi

    out_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}/chunk_${chunk}"
    job_name="t1k_panda_${RUN_ID}_c${chunk}"

    _exec_or_dry sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${TIME_BUDGET}" \
        --export="ALL,METHOD=lora,RUN_ID=${RUN_ID},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},LORA_RANK=1,LORA_ALPHA=2,LORA_TARGET_BLOCKS=all,NUM_STEPS=10,LEARNING_RATE=5.0e-5,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0,TARGET_FFN=0,NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn" \
        "${SWEEP_SBATCH}"
    count=$((count + 1))
done

echo ""
echo "Submitted ${count} jobs. Monitor: squeue -u \$USER | grep ${RUN_ID}"
echo ""
echo "After completion:"
echo "  python sweep_experiment/scripts/merge_chunks.py --results-dir ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}"
echo "  python scripts/update_merged_with_vbench.py --method-dir ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}"
echo "  # Compare ΔVBench vs NOTTA/ADA/R8 in per_video_vbench_gains.csv refresh"
echo "============================================================"
