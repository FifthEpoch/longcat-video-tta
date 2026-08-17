#!/bin/bash
# 16-image 30 s chunked three-way (seed-invariant cand0):
#   NOTTA | always-BoN k=4 | gated-BoN k=4 (gate=2.0)
# Pull the RNG fix before submitting. No TTC.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_i2v_bon16.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-i2v_bon_16v}"
COMMON="HORIZON_S=30,N_VIDEOS=16,SEED=0,SEARCH_FROM=1,CHUNK_LATENTS=24,SERIES=${SERIES}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

J1=$(sbatch --parsable --account="${ACCOUNT}" --time=04:00:00 \
    --export=ALL,METHOD=notta,SEARCH_K=1,${COMMON} \
    "${SB}/run_i2v_chunked.sbatch")
echo "16v chunked NOTTA          job ${J1}"

J2=$(sbatch --parsable --account="${ACCOUNT}" --time=04:00:00 \
    --export=ALL,METHOD=always_bon,SEARCH_K=4,${COMMON} \
    "${SB}/run_i2v_chunked.sbatch")
echo "16v chunked always-BoN k=4 job ${J2}"

J3=$(sbatch --parsable --account="${ACCOUNT}" --time=04:00:00 \
    --export=ALL,METHOD=gated_bon,SEARCH_K=4,GATE_THRESHOLD=2.0,${COMMON} \
    "${SB}/run_i2v_chunked.sbatch")
echo "16v chunked gated-BoN k=4  job ${J3}"

echo "When they finish:"
echo "  cat ${PROJECT_ROOT}/wan_experiment/results/${SERIES}/notta_h30s_shard0/summary.json | head"
echo "  cat ${PROJECT_ROOT}/wan_experiment/results/${SERIES}/always_bon_h30s_shard0/summary.json | head"
echo "  cat ${PROJECT_ROOT}/wan_experiment/results/${SERIES}/gated_bon_h30s_shard0/summary.json | head"
echo "Chunk-0 cand0 scores must match across the three methods."
echo "Cancel:  scancel ${J1} ${J2} ${J3}"
