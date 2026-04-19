#!/bin/bash
# ============================================================================
# Submit visual demo experiments: rollout sweeps + guidance sweep
#
# Total jobs: 18
#   UCF rollout:     6 jobs (NOTTA/ADA × R1/R2/R3)
#   Panda rollout:   6 jobs (NOTTA/ADA × R1/R2/R3)
#   Panda guidance:  6 jobs (NOTTA/ADA × G1/G2/G4)
#
# All runs save videos (no_save_videos: false) for visual comparison.
# 100 videos each, estimated ~3-6h per job depending on rollout depth.
#
# BEFORE RUNNING:
#   cd /scratch/wc3013/longcat-video-tta
#   git pull
#   bash sweep_experiment/sbatch/submit_visual_demo.sh
# ============================================================================

set -euo pipefail

ACCOUNT="torch_pr_36_mren"
UCF_DATA="/scratch/wc3013/longcat-video-tta/datasets/ucf101_100_480p"
PANDA_DATA="/scratch/wc3013/longcat-video-tta/datasets/panda_1000_480p"

echo "=== Visual Demo Experiments ==="
echo ""

echo "[1/3] UCF-101 rollout sweep (6 jobs)..."
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/ucf_rollout_sweep.yaml \
    --data-dir "${UCF_DATA}" \
    --account "${ACCOUNT}" \
    --time 12:00:00

echo ""
echo "[2/3] Panda-70M rollout sweep (6 jobs)..."
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_rollout_sweep.yaml \
    --data-dir "${PANDA_DATA}" \
    --account "${ACCOUNT}" \
    --time 12:00:00

echo ""
echo "[3/3] Panda-70M guidance sweep (6 jobs)..."
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_guidance_sweep.yaml \
    --data-dir "${PANDA_DATA}" \
    --account "${ACCOUNT}" \
    --time 12:00:00

echo ""
echo "============================================"
echo "Submitted 18 jobs total:"
echo ""
echo "UCF rollout:    NOTTA_R1, NOTTA_R2, NOTTA_R3, ADA_R1, ADA_R2, ADA_R3"
echo "Panda rollout:  NOTTA_R1, NOTTA_R2, NOTTA_R3, ADA_R1, ADA_R2, ADA_R3"
echo "Panda guidance: NOTTA_G1, NOTTA_G2, NOTTA_G4, ADA_G1, ADA_G2, ADA_G4"
echo ""
echo "Monitor: squeue -u \$USER"
echo "============================================"
