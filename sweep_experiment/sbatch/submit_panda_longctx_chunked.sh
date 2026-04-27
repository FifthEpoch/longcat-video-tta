#!/bin/bash
# ============================================================================
# Submit Panda-70M long-context generation in 100-video chunks
#
# 4 methods x 10 chunks = 40 jobs
# Each chunk: 100 videos, 93 frames (14 cond + 79 gen), ~16-20h
#
# BEFORE RUNNING:
#   cd /scratch/wc3013/longcat-video-tta
#   git pull
#   bash sweep_experiment/sbatch/submit_panda_longctx_chunked.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/scratch/wc3013/longcat-video-tta"
SWEEP_SBATCH="sweep_experiment/sbatch/run_sweep.sbatch"
TL_SBATCH="delta_experiment/sbatch/run_tinylora.sbatch"
ACCOUNT="torch_pr_36_mren"

DATA_DIR="${PROJECT_ROOT}/datasets/panda_1000_480p"
RESULTS="${PROJECT_ROOT}/sweep_experiment/results/panda_longctx_1000v"
TL_RESULTS="${PROJECT_ROOT}/delta_experiment/results/tinylora_longctx_1000v"

NUM_CHUNKS=10
CHUNK_SIZE=100
TIME="20:00:00"
NUM_FRAMES=93

count=0

submit_sweep() {
    local method="$1" run_id="$2"
    shift 2
    local extra_env="$*"

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local start=$((chunk * CHUNK_SIZE))
        local out_dir="${RESULTS}/${run_id}/chunk_${chunk}"
        local job_name="lc_P_${run_id}_c${chunk}"

        sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${TIME}" \
            --export="ALL,METHOD=${method},RUN_ID=${run_id},SERIES_NAME=panda_longctx_1000v,DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=1000,START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=14,NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=14,TTA_TOTAL_FRAMES=14,TTA_CONTEXT_FRAMES=14,NUM_INFERENCE_STEPS=50,GUIDANCE_SCALE=4.0,RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn${extra_env}" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
}

echo "============================================================"
echo "Panda-70M Long-Context Chunked Submission"
echo "  93 frames, 1000 videos, 4 methods x ${NUM_CHUNKS} chunks = 40 jobs"
echo "============================================================"
echo ""

echo "No-TTA (10 chunks)..."
submit_sweep "delta_a" "NOTTA" ",DELTA_STEPS=0,DELTA_LR=0.005"

echo "AdaSteer S10 (10 chunks)..."
submit_sweep "delta_a" "ADA_S10" ",DELTA_STEPS=10,DELTA_LR=5.0e-3"

echo "LoRA R8 (10 chunks)..."
submit_sweep "lora" "LORA_R8" ",LORA_RANK=8,LORA_ALPHA=16,LORA_TARGET_BLOCKS=all,NUM_STEPS=10,LEARNING_RATE=5.0e-5,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0"

echo "TinyLoRA LAST24 (10 chunks)..."
for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
    start=$((chunk * CHUNK_SIZE))
    out_dir="${TL_RESULTS}/PANDA_TL_LAST24/chunk_${chunk}"
    job_name="lc_P_TL_c${chunk}"

    sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${TIME}" \
        --export="ALL,OUTPUT_DIR=${out_dir},DATA_DIR=${DATA_DIR},NUM_VIDEOS=1000,START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_FRAMES=${NUM_FRAMES},NUM_COND_FRAMES=14,GEN_START_FRAME=14,TTA_TOTAL_FRAMES=14,TTA_CONTEXT_FRAMES=14,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,ES_DISABLE=1,NO_SAVE_VIDEOS=0,SVD_RANK=2,N_TIE=1,TARGET_PRESET=qkv_proj,TARGET_BLOCKS=last_24,TTA_STEPS=20,TTA_LR=1e-3" \
        "${TL_SBATCH}"
    count=$((count + 1))
done

echo ""
echo "============================================================"
echo "Submitted ${count} Panda-70M jobs."
echo ""
echo "Results layout:"
echo "  ${RESULTS}/{NOTTA,ADA_S10,LORA_R8}/chunk_{0..9}/"
echo "  ${TL_RESULTS}/PANDA_TL_LAST24/chunk_{0..9}/"
echo ""
echo "After completion, merge with:"
echo "  python sweep_experiment/scripts/merge_chunks.py --results-dir ${RESULTS} --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py --results-dir ${TL_RESULTS}/PANDA_TL_LAST24"
echo ""
echo "Monitor: squeue -u \$USER | grep lc_P"
echo "============================================================"
