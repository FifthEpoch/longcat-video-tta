#!/bin/bash
# ============================================================================
# Phase 2: Submit all 12 micro-adaptation TTA runs
#
# Usage (on cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/scripts/submit_phase2.sh          # submit all
#   bash sweep_experiment/scripts/submit_phase2.sh --dry-run # preview only
# ============================================================================
set -euo pipefail

ACCOUNT="torch_pr_36_mren"
DATA_DIR="/scratch/wc3013/longcat-video-tta/datasets/ucf101_500_480p"
EXTRA_ARGS="${@}"  # pass --dry-run or other flags

echo "=============================================================================="
echo "Phase 2: Micro-Adaptation TTA Sweep (12 runs)"
echo "=============================================================================="
echo "Account  : ${ACCOUNT}"
echo "Dataset  : ${DATA_DIR}"
echo "Extra    : ${EXTRA_ARGS:-<none>}"
echo "=============================================================================="
echo ""

# --- Full-model TTA: 3 runs (LR = 5e-6, 1e-6, 5e-7) ---
echo ">>> Submitting Full-model LR sweep (3 runs)..."
python3 sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/phase2_full_lr.yaml \
    --account "${ACCOUNT}" \
    --data-dir "${DATA_DIR}" \
    ${EXTRA_ARGS}

echo ""

# --- LoRA rescue: 5 runs (rank x LR) ---
echo ">>> Submitting LoRA rescue sweep (5 runs)..."
python3 sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/phase2_lora_rescue.yaml \
    --account "${ACCOUNT}" \
    --data-dir "${DATA_DIR}" \
    ${EXTRA_ARGS}

echo ""

# --- AdaSteer rescue: 4 runs (LR = 1e-3 to 5e-5) ---
echo ">>> Submitting AdaSteer rescue sweep (4 runs)..."
python3 sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/phase2_adasteer_rescue.yaml \
    --account "${ACCOUNT}" \
    --data-dir "${DATA_DIR}" \
    ${EXTRA_ARGS}

echo ""
echo "=============================================================================="
echo "Phase 2 submission complete: 12 total runs"
echo "=============================================================================="
echo "Monitor: squeue -u \$USER"
echo "Results: sweep_experiment/results/phase2_*/"
echo "=============================================================================="
