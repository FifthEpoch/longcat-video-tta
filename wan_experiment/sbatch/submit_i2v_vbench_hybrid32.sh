#!/bin/bash
# Official VBench on the already-finished hybrid 32v mp4s.
# Space-separated METHODS only — SLURM --export splits on commas
# (job 15959601 scored notta only because of that).
# Existing notta last5/full results are skipped.
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
METHODS="${METHODS:-notta always_bon gated_bon}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

J1=$(sbatch --parsable --account="${ACCOUNT}" --time=08:00:00 \
    --export=ALL,SERIES_DIR="${ROOT}",METHODS="${METHODS}",CLIPS="${CLIPS}" \
    "${SB}/run_i2v_vbench.sbatch")
echo "32v hybrid official VBench  job ${J1}"
echo "  series=${SERIES}"
echo "  methods=${METHODS}"
echo "  clips=${CLIPS}"
echo "  notta last5/full already exist from 15959601 and will be skipped"
echo "When it finishes:"
echo "  python wan_experiment/scripts/analyze_i2v_vbench.py \\"
echo "    --series-dir ${ROOT} --clip last5 \\"
echo "    --out ${PROJECT_ROOT}/sweep_experiment/reports/paper_tables/\$(date +%F)_wan_i2v_bon32_vbench_last5.md"
echo "  python wan_experiment/scripts/analyze_i2v_vbench.py \\"
echo "    --series-dir ${ROOT} --clip full \\"
echo "    --out ${PROJECT_ROOT}/sweep_experiment/reports/paper_tables/\$(date +%F)_wan_i2v_bon32_vbench_full.md"
echo "Cancel:  scancel ${J1}"
