#!/bin/bash
# ============================================================================
# Submit 1000-video evaluation of best configs
#
# Estimated wall times (per-video × 1000):
#   NOTTA:     ~80s/vid  → ~22 hours
#   DV_BARE:   ~134s/vid → ~37 hours
#   LORA_R8:   ~99s/vid  → ~28 hours
#
# All scripts checkpoint per-video, so preemption/requeue is safe.
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
    --account "${ACCOUNT}" \
    --time 48:00:00

echo ""
echo "[2/2] LoRA (R8, 10 steps)..."
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_1000v_lora.yaml \
    --data-dir "${DATA_DIR}" \
    --account "${ACCOUNT}" \
    --time 48:00:00

echo ""
echo "============================================"
echo "Submitted 3 jobs total:"
echo "  - NOTTA       (~22h)"
echo "  - DV_BARE     (~37h)"
echo "  - LORA_R8_S10 (~28h)"
echo ""
echo "Monitor: squeue -u \$USER"
echo "============================================"
