#!/bin/bash
# 16-image NOTTA I2V at 5 s and 30 s. Submit only after the 2×5 s smoke
# (15880611) mp4s look like the cond images, not TTC-v1 noise.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_i2v_notta16.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-i2v_notta_16v}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

J5=$(sbatch --parsable --account="${ACCOUNT}" \
    --export=ALL,HORIZON_S=5,N_VIDEOS=16,SEED=0,SERIES="${SERIES}" \
    "${SB}/run_i2v_notta.sbatch")
echo "i2v_notta_16v 5s  n=16  job ${J5}"

J30=$(sbatch --parsable --account="${ACCOUNT}" \
    --export=ALL,HORIZON_S=30,N_VIDEOS=16,SEED=0,SERIES="${SERIES}" \
    "${SB}/run_i2v_notta.sbatch")
echo "i2v_notta_16v 30s n=16  job ${J30}"

echo "When they finish:"
echo "  cat ${PROJECT_ROOT}/wan_experiment/results/${SERIES}/h5s_shard0/summary.json"
echo "  cat ${PROJECT_ROOT}/wan_experiment/results/${SERIES}/h30s_shard0/summary.json"
echo "Cancel:  scancel ${J5} ${J30}"
