#!/bin/bash
# ============================================================================
# Submit TinyLoRA long-context generation jobs
#
# Uses the best TinyLoRA config (TL_LAST24) at long generation horizons
# to enable comparison: No-TTA vs LoRA vs TinyLoRA vs AdaSteer
#
# Frame geometry matches existing long-context experiments:
#   Panda-70M: 93 frames (14 cond + 79 gen), 50 videos
#   UCF-101:   61 frames (14 cond + 47 gen), 50 videos
#
# BEFORE RUNNING:
#   cd /scratch/wc3013/longcat-video-tta
#   git pull
#   bash delta_experiment/sbatch/submit_tinylora_longctx.sh
# ============================================================================

set -euo pipefail

SBATCH_SCRIPT="delta_experiment/sbatch/run_tinylora.sbatch"
RESULTS_BASE="/scratch/wc3013/longcat-video-tta/delta_experiment/results/tinylora_longctx"

PANDA_DATA="/scratch/wc3013/longcat-video-tta/datasets/panda_1000_480p"
UCF_DATA="/scratch/wc3013/longcat-video-tta/datasets/ucf101_100_480p"

# Best TinyLoRA config from sweep: TL_LAST24
BEST_CONFIG="SVD_RANK=2,N_TIE=1,TARGET_PRESET=qkv_proj,TARGET_BLOCKS=last_24,TTA_STEPS=20,TTA_LR=1e-3"

echo "============================================"
echo "TinyLoRA Long-Context Generation"
echo "============================================"
echo "Results base: ${RESULTS_BASE}/"
echo ""

# --- Panda-70M: 93 frames (14 cond + 79 gen), 50 videos ---
echo "Submitting: Panda-70M TinyLoRA (93 frames, 50 videos)"
sbatch \
    --job-name="tl_panda_longctx" \
    --time=24:00:00 \
    --export="ALL,OUTPUT_DIR=${RESULTS_BASE}/PANDA_TL_LAST24,DATA_DIR=${PANDA_DATA},NUM_VIDEOS=50,NUM_FRAMES=93,NUM_COND_FRAMES=14,GEN_START_FRAME=14,TTA_TOTAL_FRAMES=14,TTA_CONTEXT_FRAMES=14,COMPUTE_FVD=1,COMPUTE_FID=1,ES_DISABLE=1,${BEST_CONFIG}" \
    "${SBATCH_SCRIPT}"

# --- UCF-101: 61 frames (14 cond + 47 gen), 50 videos ---
echo "Submitting: UCF-101 TinyLoRA (61 frames, 50 videos)"
sbatch \
    --job-name="tl_ucf_longctx" \
    --time=24:00:00 \
    --export="ALL,OUTPUT_DIR=${RESULTS_BASE}/UCF_TL_LAST24,DATA_DIR=${UCF_DATA},NUM_VIDEOS=50,NUM_FRAMES=61,NUM_COND_FRAMES=14,GEN_START_FRAME=14,TTA_TOTAL_FRAMES=14,TTA_CONTEXT_FRAMES=14,COMPUTE_FVD=1,COMPUTE_FID=1,ES_DISABLE=1,${BEST_CONFIG}" \
    "${SBATCH_SCRIPT}"

echo ""
echo "============================================"
echo "Submitted 2 TinyLoRA long-context jobs."
echo "Results will be in: ${RESULTS_BASE}/"
echo "Monitor with: squeue -u \$USER | grep tl_"
echo "============================================"
