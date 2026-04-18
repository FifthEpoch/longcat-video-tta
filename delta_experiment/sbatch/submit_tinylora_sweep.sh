#!/bin/bash
# ============================================================================
# Submit all TinyLoRA sweep jobs
#
# BEFORE RUNNING:
#   cd /scratch/wc3013/longcat-video-tta
#   git pull
#   bash delta_experiment/sbatch/submit_tinylora_sweep.sh
# ============================================================================

set -euo pipefail

SBATCH_SCRIPT="delta_experiment/sbatch/run_tinylora.sbatch"
DATA_DIR="/scratch/wc3013/longcat-video-tta/datasets/panda_1000_480p"
RESULTS_BASE="/scratch/wc3013/longcat-video-tta/delta_experiment/results/tinylora_sweep"

submit() {
    local run_id="$1"; shift
    local extra_exports=""
    for arg in "$@"; do
        extra_exports="${extra_exports},${arg}"
    done
    echo "Submitting: ${run_id}"
    sbatch \
        --job-name="tl_${run_id}" \
        --export="ALL,OUTPUT_DIR=${RESULTS_BASE}/${run_id},DATA_DIR=${DATA_DIR},NUM_VIDEOS=100,COMPUTE_FVD=1,COMPUTE_FID=1,ES_DISABLE=1${extra_exports}" \
        "${SBATCH_SCRIPT}"
}

echo "Results base: ${RESULTS_BASE}/"
echo ""

# --- 1) Rank sweep (bare, no aug) ---
submit TL_BARE_R1   SVD_RANK=1 N_TIE=1 TARGET_PRESET=qkv_proj TARGET_BLOCKS=all TTA_STEPS=20 TTA_LR=1e-3
submit TL_BARE_R2   SVD_RANK=2 N_TIE=1 TARGET_PRESET=qkv_proj TARGET_BLOCKS=all TTA_STEPS=20 TTA_LR=1e-3
submit TL_BARE_R4   SVD_RANK=4 N_TIE=1 TARGET_PRESET=qkv_proj TARGET_BLOCKS=all TTA_STEPS=20 TTA_LR=1e-3

# --- 2) Weight tying ---
submit TL_TIED_R2   SVD_RANK=2 N_TIE=48 TARGET_PRESET=qkv_proj TARGET_BLOCKS=all TTA_STEPS=20 TTA_LR=1e-3

# --- 3) Target module sweep ---
submit TL_ALLATTN_R2 SVD_RANK=2 N_TIE=1 TARGET_PRESET=all_attn TARGET_BLOCKS=all TTA_STEPS=20 TTA_LR=1e-3
submit TL_ALL_R2     SVD_RANK=2 N_TIE=1 TARGET_PRESET=all     TARGET_BLOCKS=all TTA_STEPS=20 TTA_LR=1e-3

# --- 4) Augmentation ---
submit TL_AUG_R2    SVD_RANK=2 N_TIE=1 TARGET_PRESET=qkv_proj TARGET_BLOCKS=all TTA_STEPS=20 TTA_LR=1e-3 AUG_ENABLED=1 AUG_FLIP=1

# --- 5) Step count ---
submit TL_STEPS10     SVD_RANK=2 N_TIE=1 TARGET_PRESET=qkv_proj TARGET_BLOCKS=all TTA_STEPS=10 TTA_LR=1e-3
submit TL_STEPS10_AUG SVD_RANK=2 N_TIE=1 TARGET_PRESET=qkv_proj TARGET_BLOCKS=all TTA_STEPS=10 TTA_LR=1e-3 AUG_ENABLED=1 AUG_FLIP=1

# --- 6) LR sweep ---
submit TL_LR5E3_R2  SVD_RANK=2 N_TIE=1 TARGET_PRESET=qkv_proj TARGET_BLOCKS=all TTA_STEPS=20 TTA_LR=5e-3

# --- 7) Block subset ---
submit TL_LAST5   SVD_RANK=2 N_TIE=1 TARGET_PRESET=qkv_proj TARGET_BLOCKS=last_5  TTA_STEPS=20 TTA_LR=1e-3
submit TL_LAST10  SVD_RANK=2 N_TIE=1 TARGET_PRESET=qkv_proj TARGET_BLOCKS=last_10 TTA_STEPS=20 TTA_LR=1e-3
submit TL_LAST24  SVD_RANK=2 N_TIE=1 TARGET_PRESET=qkv_proj TARGET_BLOCKS=last_24 TTA_STEPS=20 TTA_LR=1e-3

echo ""
echo "============================================"
echo "Submitted 13 TinyLoRA jobs."
echo "Results will be in: ${RESULTS_BASE}/"
echo "Monitor with: squeue -u \$USER | grep tl_"
echo "============================================"
