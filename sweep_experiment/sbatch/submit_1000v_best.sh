#!/bin/bash
# ============================================================================
# Submit 1000-video evaluation of best configs
#
# BEFORE RUNNING:
#   cd /scratch/wc3013/longcat-video-tta
#   git pull
#   bash sweep_experiment/sbatch/submit_1000v_best.sh
# ============================================================================

set -euo pipefail

ACCOUNT="torch_pr_36_mren"
DATA_DIR="/scratch/wc3013/longcat-video-tta/datasets/panda_1000_480p"

echo "=== 1000-video best-method evaluation ==="
echo ""

echo "[1/2] No-TTA baseline + Delta Vector (AS_BARE)..."
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_1000v_best_methods.yaml \
    --data-dir "${DATA_DIR}" \
    --account "${ACCOUNT}"

echo ""
echo "[2/2] LoRA (2 configs: R4/S20 + R8/S10)..."
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_1000v_lora.yaml \
    --data-dir "${DATA_DIR}" \
    --account "${ACCOUNT}"

echo ""
echo "============================================"
echo "Submitted 4 jobs total:"
echo "  - NOTTA (no TTA baseline)"
echo "  - DV_BARE (Delta Vector, steps=10, lr=0.005)"
echo "  - LORA_R4_S20 (rank=4, last_4, 20 steps, lr=1e-5)"
echo "  - LORA_R8_S10 (rank=8, all, 10 steps, lr=5e-5)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "============================================"
