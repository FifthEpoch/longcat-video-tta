#!/bin/bash
# Submit long-context frame generation experiments
#
# 6 jobs total:
#   Panda-70M (93 frames): NOTTA, AdaSteer, LoRA
#   UCF-101   (61 frames): NOTTA, AdaSteer, LoRA
#
# Usage:
#   bash sweep_experiment/sbatch/submit_longctx.sh          # submit all
#   bash sweep_experiment/sbatch/submit_longctx.sh --dry-run # preview only

set -euo pipefail

ACCOUNT="torch_pr_36_mren"
UCF_DATA="/scratch/wc3013/longcat-video-tta/datasets/ucf101_100_480p"
PANDA_DATA="/scratch/wc3013/longcat-video-tta/datasets/panda_1000_480p"

EXTRA_ARGS="${*}"

echo "=============================================="
echo "Long-Context Frame Generation Experiments"
echo "=============================================="
echo ""

echo "--- Panda-70M: 93-frame (Baseline + AdaSteer) ---"
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_longctx_delta_a.yaml \
    --data-dir "${PANDA_DATA}" \
    --account "${ACCOUNT}" \
    --time 24:00:00 \
    ${EXTRA_ARGS}

echo ""
echo "--- Panda-70M: 93-frame (LoRA) ---"
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_longctx_lora.yaml \
    --data-dir "${PANDA_DATA}" \
    --account "${ACCOUNT}" \
    --time 24:00:00 \
    ${EXTRA_ARGS}

echo ""
echo "--- UCF-101: 61-frame (Baseline + AdaSteer) ---"
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/ucf_longctx_delta_a.yaml \
    --data-dir "${UCF_DATA}" \
    --account "${ACCOUNT}" \
    --time 24:00:00 \
    ${EXTRA_ARGS}

echo ""
echo "--- UCF-101: 61-frame (LoRA) ---"
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/ucf_longctx_lora.yaml \
    --data-dir "${UCF_DATA}" \
    --account "${ACCOUNT}" \
    --time 24:00:00 \
    ${EXTRA_ARGS}

echo ""
echo "=============================================="
echo "All long-context jobs submitted."
echo "=============================================="
