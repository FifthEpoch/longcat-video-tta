#!/bin/bash
# Sticky gated-search only, same 32 images / seeds / alarms as
# i2v_bon_32v_hybrid. Once any alarm fires on a video, later pieces
# keep searching (reason=already_on if no fresh alarm).
# Do-nothing and always-search are reused from the hybrid series.
# No test-time training.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_i2v_bon32_sticky.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-i2v_bon_32v_sticky}"
BASELINE="${BASELINE:-i2v_bon_32v_hybrid}"
N_VIDEOS="${N_VIDEOS:-32}"
GATE="GATE_THRESHOLD=2.0,GATE_CH1_THRESHOLD=0.8,GATE_DELTA=0.5,GATE_DELTA_PREV_MIN=0.5,GATE_STICKY=1"
COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=1,CHUNK_LATENTS=24,SERIES=${SERIES},${GATE}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

# Hybrid gated was ~173 s/clip; sticky searches more later pieces,
# so budget toward always-search (~258 s) × 32 ≈ 2.3 h. 6 h wall.
J1=$(sbatch --parsable --account="${ACCOUNT}" --time=06:00:00 \
    --export=ALL,METHOD=gated_bon,SEARCH_K=4,${COMMON} \
    "${SB}/run_i2v_chunked.sbatch")
echo "32v sticky gated-search k=4  job ${J1}  series=${SERIES}"

echo "When it finishes, pair against the hybrid do-nothing / always-search:"
echo "  python wan_experiment/scripts/analyze_i2v_bon.py \\"
echo "    --series-dir ${PROJECT_ROOT}/wan_experiment/results/${SERIES} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/${BASELINE}"
echo "Cancel:  scancel ${J1}"
