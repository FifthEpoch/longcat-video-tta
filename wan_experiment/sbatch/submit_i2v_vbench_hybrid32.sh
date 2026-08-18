#!/bin/bash
# Official VBench on the already-finished hybrid 32v mp4s.
# One sequential GPU job (GRES-friendly; 15959146 may already be queued).
# last5 first, then full 30 s. No generation. No TTC.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_i2v_vbench_hybrid32.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-i2v_bon_32v_hybrid}"
ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
CLIPS="${CLIPS:-last5 full}"

VIDEO_DIRS="${ROOT}/notta_h30s_shard0,${ROOT}/always_bon_h30s_shard0,${ROOT}/gated_bon_h30s_shard0"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

J1=$(sbatch --parsable --account="${ACCOUNT}" --time=08:00:00 \
    --export=ALL,VIDEO_DIRS="${VIDEO_DIRS}",CLIPS="${CLIPS}" \
    "${SB}/run_i2v_vbench.sbatch")
echo "32v hybrid official VBench  job ${J1}"
echo "  series=${SERIES}"
echo "  clips=${CLIPS}"
echo "  dirs=${VIDEO_DIRS}"
echo "When it finishes:"
echo "  python wan_experiment/scripts/analyze_i2v_vbench.py \\"
echo "    --series-dir ${ROOT} --clip last5 \\"
echo "    --out ${PROJECT_ROOT}/sweep_experiment/reports/paper_tables/\$(date +%F)_wan_i2v_bon32_vbench_last5.md"
echo "  python wan_experiment/scripts/analyze_i2v_vbench.py \\"
echo "    --series-dir ${ROOT} --clip full \\"
echo "    --out ${PROJECT_ROOT}/sweep_experiment/reports/paper_tables/\$(date +%F)_wan_i2v_bon32_vbench_full.md"
echo "Cancel:  scancel ${J1}"
