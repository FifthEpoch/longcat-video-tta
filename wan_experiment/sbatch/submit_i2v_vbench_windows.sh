#!/bin/bash
# VBench++ on every 5 s window of the hybrid 32v 30 s mp4s.
# full + last5 already exist and are not re-run. Official number stays
# the full clip; these windows are the trend plot.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_i2v_vbench_windows.sh
#
# Space-separated CLIPS only — SLURM --export splits on commas.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-i2v_bon_32v_hybrid}"
ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
METHODS="${METHODS:-notta always_bon gated_bon}"
CLIPS="${CLIPS:-w0_5 w5_10 w10_15 w15_20 w20_25 w25_30}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

# 3 methods × 6 windows × 7 dims × 32 videos. Existing full/last5 skipped.
J1=$(sbatch --parsable --account="${ACCOUNT}" --time=16:00:00 \
    --export=ALL,SERIES_DIR="${ROOT}",METHODS="${METHODS}",CLIPS="${CLIPS}" \
    "${SB}/run_i2v_vbench.sbatch")
echo "32v hybrid VBench 5 s windows  job ${J1}"
echo "  series=${SERIES}"
echo "  methods=${METHODS}"
echo "  clips=${CLIPS}"
echo "When it finishes:"
echo "  python wan_experiment/scripts/analyze_i2v_vbench_trend.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --out ${PROJECT_ROOT}/sweep_experiment/reports/paper_tables/\$(date +%F)_wan_i2v_bon32_vbench_trend.md"
echo "Cancel:  scancel ${J1}"
