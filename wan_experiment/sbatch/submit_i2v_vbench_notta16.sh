#!/bin/bash
# VBench++ on the existing 16v NOTTA 5 s and 30 s mp4s, including the
# same 16-frame head/tail windows as score_i2v_drift.py (skip cond
# frame 0). Official comparable number is still the full clip.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_i2v_vbench_notta16.sh
#
# Space-separated VIDEO_DIRS / CLIPS only — SLURM --export splits on commas.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-i2v_notta_16v}"
ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS="${VIDEO_DIRS:-${ROOT}/h5s_shard0 ${ROOT}/h30s_shard0}"
# full = official. first5 = same-duration 5 s vs 30 s opening.
# first1/last1 = 16 frames, skip f0 (handpicked drift match).
CLIPS="${CLIPS:-full first5 first1 last1}"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

# 2 dirs × 4 clips × 7 dims × 16 videos. Existing joined.json skipped.
J1=$(sbatch --parsable --account="${ACCOUNT}" --time=08:00:00 \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS="${CLIPS}" \
    "${SB}/run_i2v_vbench.sbatch")
echo "16v NOTTA 5s/30s VBench head-tail  job ${J1}"
echo "  series=${SERIES}"
echo "  video_dirs=${VIDEO_DIRS}"
echo "  clips=${CLIPS}"
echo "When it finishes:"
echo "  python wan_experiment/scripts/analyze_i2v_vbench_horizon.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --out ${PROJECT_ROOT}/sweep_experiment/reports/paper_tables/\$(date +%F)_wan_i2v_notta16_vbench_headtail.md"
echo "Cancel:  scancel ${J1}"
