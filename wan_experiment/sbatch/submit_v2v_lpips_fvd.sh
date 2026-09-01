#!/bin/bash
# Fill LPIPS + aligned-tail FVD on caption-128. No new videos.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_lpips_fvd.sh

set -euo pipefail
SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-v2v_panda_caption_128v}"
ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"

J=$(sbatch --parsable --account="${ACCOUNT}" \
    --export=ALL,SERIES_DIR="${ROOT}",METHODS="notta rolling_notta sf_pseudo sf_always_search" \
    "${SB}/run_v2v_lpips_fvd.sbatch")
echo "V2V LPIPS+FVD job ${J} series ${SERIES}"
echo "Cancel this job only:  scancel ${J}"
echo "Do not remake 128 videos."
