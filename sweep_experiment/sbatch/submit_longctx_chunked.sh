#!/bin/bash
# ============================================================================
# Submit long-context generation in 100-video chunks (parallel)
#
# 4 methods x 2 datasets x 10 chunks = 80 jobs
# Each chunk processes 100 videos and takes ~16h (Panda 93f) or ~8.5h (UCF 61f)
#
# BEFORE RUNNING:
#   cd /scratch/wc3013/longcat-video-tta
#   git pull
#   bash sweep_experiment/sbatch/submit_longctx_chunked.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/scratch/wc3013/longcat-video-tta"
SWEEP_SBATCH="sweep_experiment/sbatch/run_sweep.sbatch"
TL_SBATCH="delta_experiment/sbatch/run_tinylora.sbatch"
ACCOUNT="torch_pr_36_mren"

PANDA_DATA="${PROJECT_ROOT}/datasets/panda_1000_480p"
UCF_DATA="${PROJECT_ROOT}/datasets/ucf101_1000_480p"

PANDA_RESULTS="${PROJECT_ROOT}/sweep_experiment/results/panda_longctx_1000v"
UCF_RESULTS="${PROJECT_ROOT}/sweep_experiment/results/ucf_longctx_1000v"
TL_RESULTS="${PROJECT_ROOT}/delta_experiment/results/tinylora_longctx_1000v"

NUM_CHUNKS=10
CHUNK_SIZE=100

# Time estimates (with buffer):
#   Panda 93f: ~572s/video x 100 = ~16h → request 20h
#   UCF   61f: ~308s/video x 100 = ~8.5h → request 12h
PANDA_TIME="20:00:00"
UCF_TIME="12:00:00"

count=0

submit_sweep() {
    local method="$1" run_id="$2" data_dir="$3" results_base="$4"
    local time_limit="$5" dataset_tag="$6" num_frames="$7"
    shift 7
    local extra_env="$*"

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local start=$((chunk * CHUNK_SIZE))
        local out_dir="${results_base}/${run_id}/chunk_${chunk}"
        local job_name="lc_${dataset_tag}_${run_id}_c${chunk}"

        sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${time_limit}" \
            --export="ALL,METHOD=${method},RUN_ID=${run_id},SERIES_NAME=longctx_chunked,DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},MAX_VIDEOS=1000,START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=14,NUM_FRAMES=${num_frames},GEN_START_FRAME=14,TTA_TOTAL_FRAMES=14,TTA_CONTEXT_FRAMES=14,NUM_INFERENCE_STEPS=50,GUIDANCE_SCALE=4.0,RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,NO_SAVE_VIDEOS=1,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn${extra_env}" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
}

submit_tinylora() {
    local data_dir="$1" results_base="$2" time_limit="$3" dataset_tag="$4" num_frames="$5"

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local start=$((chunk * CHUNK_SIZE))
        local out_dir="${results_base}/${dataset_tag}_TL_LAST24/chunk_${chunk}"
        local job_name="lc_${dataset_tag}_TL_c${chunk}"

        sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${time_limit}" \
            --export="ALL,OUTPUT_DIR=${out_dir},DATA_DIR=${data_dir},NUM_VIDEOS=1000,START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_FRAMES=${num_frames},NUM_COND_FRAMES=14,GEN_START_FRAME=14,TTA_TOTAL_FRAMES=14,TTA_CONTEXT_FRAMES=14,COMPUTE_FVD=1,COMPUTE_FID=1,ES_DISABLE=1,NO_SAVE_VIDEOS=1,SVD_RANK=2,N_TIE=1,TARGET_PRESET=qkv_proj,TARGET_BLOCKS=last_24,TTA_STEPS=20,TTA_LR=1e-3" \
            "${TL_SBATCH}"
        count=$((count + 1))
    done
}

echo "============================================================"
echo "Long-Context Chunked Submission"
echo "  4 methods x 2 datasets x ${NUM_CHUNKS} chunks = $((4 * 2 * NUM_CHUNKS)) jobs"
echo "  Chunk size: ${CHUNK_SIZE} videos"
echo "============================================================"
echo ""

# ── Panda-70M (93 frames, 1000 videos) ──
echo "=== Panda-70M (93 frames) ==="

echo "  No-TTA (10 chunks)..."
submit_sweep "delta_a" "NOTTA" "${PANDA_DATA}" "${PANDA_RESULTS}" "${PANDA_TIME}" "P" 93 ",DELTA_STEPS=0,DELTA_LR=0.005"

echo "  AdaSteer S10 (10 chunks)..."
submit_sweep "delta_a" "ADA_S10" "${PANDA_DATA}" "${PANDA_RESULTS}" "${PANDA_TIME}" "P" 93 ",DELTA_STEPS=10,DELTA_LR=5.0e-3"

echo "  LoRA R8 (10 chunks)..."
submit_sweep "lora" "LORA_R8" "${PANDA_DATA}" "${PANDA_RESULTS}" "${PANDA_TIME}" "P" 93 ",LORA_RANK=8,LORA_ALPHA=16,LORA_TARGET_BLOCKS=all,NUM_STEPS=10,LEARNING_RATE=5.0e-5,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0"

echo "  TinyLoRA LAST24 (10 chunks)..."
submit_tinylora "${PANDA_DATA}" "${TL_RESULTS}" "${PANDA_TIME}" "PANDA" 93

echo ""

# ── UCF-101 (61 frames, 1000 videos) ──
echo "=== UCF-101 (61 frames) ==="

echo "  No-TTA (10 chunks)..."
submit_sweep "delta_a" "NOTTA" "${UCF_DATA}" "${UCF_RESULTS}" "${UCF_TIME}" "U" 61 ",DELTA_STEPS=0,DELTA_LR=0.005"

echo "  AdaSteer S10 (10 chunks)..."
submit_sweep "delta_a" "ADA_S10" "${UCF_DATA}" "${UCF_RESULTS}" "${UCF_TIME}" "U" 61 ",DELTA_STEPS=10,DELTA_LR=5.0e-3"

echo "  LoRA R8 (10 chunks)..."
submit_sweep "lora" "LORA_R8" "${UCF_DATA}" "${UCF_RESULTS}" "${UCF_TIME}" "U" 61 ",LORA_RANK=8,LORA_ALPHA=16,LORA_TARGET_BLOCKS=all,NUM_STEPS=10,LEARNING_RATE=5.0e-5,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0"

echo "  TinyLoRA LAST24 (10 chunks)..."
submit_tinylora "${UCF_DATA}" "${TL_RESULTS}" "${UCF_TIME}" "UCF" 61

echo ""
echo "============================================================"
echo "Submitted ${count} jobs total."
echo ""
echo "Results layout:"
echo "  ${PANDA_RESULTS}/{NOTTA,ADA_S10,LORA_R8}/chunk_{0..9}/"
echo "  ${UCF_RESULTS}/{NOTTA,ADA_S10,LORA_R8}/chunk_{0..9}/"
echo "  ${TL_RESULTS}/{PANDA,UCF}_TL_LAST24/chunk_{0..9}/"
echo ""
echo "After all chunks complete, merge with:"
echo "  python sweep_experiment/scripts/merge_chunks.py --results-dir <run_dir>"
echo ""
echo "Monitor: squeue -u \$USER | grep lc_"
echo "============================================================"
