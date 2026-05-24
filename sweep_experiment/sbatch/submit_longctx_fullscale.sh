#!/bin/bash
# ============================================================================
# Submit ALL long-context generation jobs at full scale
#
# 4 methods x 2 datasets = 8 jobs total
#
# Panda-70M: 93 frames (14 cond + 79 gen), 1000 videos
#   Estimated ~160h per method (video saving OFF)
#
# UCF-101:   61 frames (14 cond + 47 gen), 100 videos (full dataset)
#   Estimated ~9h per method (video saving ON)
#
# BEFORE RUNNING:
#   cd /scratch/wc3013/longcat-video-tta
#   git pull
#   bash sweep_experiment/sbatch/submit_longctx_fullscale.sh
# ============================================================================

set -euo pipefail

PROJECT_ROOT="/scratch/wc3013/longcat-video-tta"
PANDA_DATA="${PROJECT_ROOT}/datasets/panda_1000_480p"
UCF_DATA="${PROJECT_ROOT}/datasets/ucf101_100_480p"
ACCOUNT="torch_pr_36_mren"
PYTHON="${PROJECT_ROOT}/../conda-envs/longcat/bin/python"

echo "============================================================"
echo "Long-Context Full-Scale Submission"
echo "  4 methods x 2 datasets = 8 jobs"
echo "============================================================"
echo ""

# ============================================================================
# 1) Panda-70M: No-TTA + AdaSteer (via sweep framework)
# ============================================================================
echo "--- Panda-70M: No-TTA + AdaSteer ---"
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_longctx_1000v_delta_a.yaml \
    --data-dir "${PANDA_DATA}" \
    --account "${ACCOUNT}" \
    --time 168:00:00

echo ""

# ============================================================================
# 2) Panda-70M: LoRA R8 (via sweep framework)
# ============================================================================
echo "--- Panda-70M: LoRA R8 ---"
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_longctx_1000v_lora.yaml \
    --data-dir "${PANDA_DATA}" \
    --account "${ACCOUNT}" \
    --time 168:00:00

echo ""

# ============================================================================
# 3) UCF-101: No-TTA + AdaSteer (via sweep framework)
# ============================================================================
echo "--- UCF-101: No-TTA + AdaSteer ---"
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/ucf_longctx_100v_delta_a.yaml \
    --data-dir "${UCF_DATA}" \
    --account "${ACCOUNT}" \
    --time 12:00:00

echo ""

# ============================================================================
# 4) UCF-101: LoRA R8 (via sweep framework)
# ============================================================================
echo "--- UCF-101: LoRA R8 ---"
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/ucf_longctx_100v_lora.yaml \
    --data-dir "${UCF_DATA}" \
    --account "${ACCOUNT}" \
    --time 12:00:00

echo ""

# ============================================================================
# 5) TinyLoRA (both datasets, via direct sbatch)
# ============================================================================
echo "--- TinyLoRA (Panda + UCF) ---"
bash delta_experiment/sbatch/submit_tinylora_longctx.sh

echo ""
echo "============================================================"
echo "All 8 long-context full-scale jobs submitted."
echo ""
echo "Expected results directories:"
echo "  sweep_experiment/results/panda_longctx_1000v/{NOTTA,ADA_S10,LORA_R8}/"
echo "  sweep_experiment/results/ucf_longctx_100v/{NOTTA,ADA_S10,LORA_R8}/"
echo "  delta_experiment/results/tinylora_longctx/{PANDA_TL_LAST24,UCF_TL_LAST24}/"
echo ""
echo "Monitor: squeue -u \$USER"
echo "============================================================"
