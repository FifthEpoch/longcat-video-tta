#!/bin/bash
# 32-image 30 s three-way with the hybrid gate + per-chunk traces.
#   NOTTA | always-BoN k=4 | gated-BoN k=4
#   fire if (chunk==1 and incoming>0.8) or incoming>2.0
#           or (Δincoming>0.5 and incoming_prev>0.5)
# Re-runs all three so the logging schema matches. No TTC.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_i2v_bon32_hybrid.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-i2v_bon_32v_hybrid}"
N_VIDEOS="${N_VIDEOS:-32}"
GATE="GATE_THRESHOLD=2.0,GATE_CH1_THRESHOLD=0.8,GATE_DELTA=0.5,GATE_DELTA_PREV_MIN=0.5"
COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=1,CHUNK_LATENTS=24,SERIES=${SERIES},${GATE}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

# always-on 32 × ~267 s ≈ 2.4 h; 6 h wall leaves room for load jitter.
J1=$(sbatch --parsable --account="${ACCOUNT}" --time=04:00:00 \
    --export=ALL,METHOD=notta,SEARCH_K=1,${COMMON} \
    "${SB}/run_i2v_chunked.sbatch")
echo "32v hybrid NOTTA          job ${J1}"

J2=$(sbatch --parsable --account="${ACCOUNT}" --time=06:00:00 \
    --export=ALL,METHOD=always_bon,SEARCH_K=4,${COMMON} \
    "${SB}/run_i2v_chunked.sbatch")
echo "32v hybrid always-BoN k=4 job ${J2}"

J3=$(sbatch --parsable --account="${ACCOUNT}" --time=06:00:00 \
    --export=ALL,METHOD=gated_bon,SEARCH_K=4,${COMMON} \
    "${SB}/run_i2v_chunked.sbatch")
echo "32v hybrid gated-BoN k=4  job ${J3}"

echo "When they finish:"
echo "  python wan_experiment/scripts/analyze_i2v_bon.py \\"
echo "    --series-dir ${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
echo "Chunk-0 cand0 scores must match across the three methods."
echo "Cancel:  scancel ${J1} ${J2} ${J3}"
