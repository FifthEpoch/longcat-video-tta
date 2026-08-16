#!/bin/bash
# 2-image 5 s NOTTA I2V smoke. Do not launch the 16×30 s suite until this
# writes real video (not noise) and summary.json has n_ok==n.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_i2v_smoke.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

J=$(sbatch --parsable --account="${ACCOUNT}" \
    --export=ALL,HORIZON_S=5,N_VIDEOS=2,SEED=0,SERIES=i2v_notta_smoke \
    "${SB}/run_i2v_notta.sbatch")
echo "i2v_notta_smoke 5s n=2  job ${J}"
echo "When it finishes:"
echo "  cat ${PROJECT_ROOT}/wan_experiment/results/i2v_notta_smoke/h5s_shard0/summary.json"
echo "  ls -la ${PROJECT_ROOT}/wan_experiment/results/i2v_notta_smoke/h5s_shard0/"
echo "Cancel:  scancel ${J}"
