#!/bin/bash
# 2-image 30 s chunked smoke: NOTTA vs always-BoN k=4.
# Shared prefix (chunk 0 = seed 0). Search starts at chunk 1.
# Do not add TTC until both write real mp4s.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_i2v_bon_smoke.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-i2v_chunked_smoke}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

J1=$(sbatch --parsable --account="${ACCOUNT}" \
    --export=ALL,METHOD=notta,HORIZON_S=30,N_VIDEOS=2,SEED=0,SEARCH_K=1,SERIES="${SERIES}" \
    "${SB}/run_i2v_chunked.sbatch")
echo "chunked NOTTA   30s n=2  job ${J1}"

J2=$(sbatch --parsable --account="${ACCOUNT}" \
    --export=ALL,METHOD=always_bon,HORIZON_S=30,N_VIDEOS=2,SEED=0,SEARCH_K=4,SEARCH_FROM=1,SERIES="${SERIES}" \
    "${SB}/run_i2v_chunked.sbatch")
echo "chunked always-BoN k=4  30s n=2  job ${J2}"

echo "When they finish:"
echo "  cat ${PROJECT_ROOT}/wan_experiment/results/${SERIES}/notta_h30s_shard0/summary.json"
echo "  cat ${PROJECT_ROOT}/wan_experiment/results/${SERIES}/always_bon_h30s_shard0/summary.json"
echo "Cancel:  scancel ${J1} ${J2}"
