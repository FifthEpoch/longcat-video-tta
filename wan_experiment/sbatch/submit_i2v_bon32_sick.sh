#!/bin/bash
# Search-while-sick gated-search only. Same 32 images / seeds / alarms
# as i2v_bon_32v_hybrid. Stay-on after an alarm, but turn memory off
# if the last second recovered by >0.5 or is now below 1.0.
# Do-nothing and always-search are reused from the hybrid series.
# No test-time training.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_i2v_bon32_sick.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-i2v_bon_32v_sick}"
BASELINE="${BASELINE:-i2v_bon_32v_hybrid}"
N_VIDEOS="${N_VIDEOS:-32}"
GATE="GATE_THRESHOLD=2.0,GATE_CH1_THRESHOLD=0.8,GATE_DELTA=0.5,GATE_DELTA_PREV_MIN=0.5,GATE_STICKY=1,GATE_SICK_MIN=1.0,GATE_RECOVERY=0.5"
COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=1,CHUNK_LATENTS=24,SERIES=${SERIES},${GATE}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

# Between hybrid (~173 s) and forever-sticky (~256 s). 6 h wall.
J1=$(sbatch --parsable --account="${ACCOUNT}" --time=06:00:00 \
    --export=ALL,METHOD=gated_bon,SEARCH_K=4,${COMMON} \
    "${SB}/run_i2v_chunked.sbatch")
echo "32v search-while-sick gated-search k=4  job ${J1}  series=${SERIES}"

echo "When it finishes, pair against the hybrid do-nothing / always-search:"
echo "  python wan_experiment/scripts/analyze_i2v_bon.py \\"
echo "    --series-dir ${PROJECT_ROOT}/wan_experiment/results/${SERIES} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/${BASELINE}"
echo "Cancel:  scancel ${J1}"
