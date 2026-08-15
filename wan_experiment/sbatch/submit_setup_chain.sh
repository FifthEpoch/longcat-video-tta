#!/bin/bash
# Submit the overnight Wan / Self-Forcing setup chain and exit.
# Jobs 1 (env, GPU) and 2 (download, CPU) run in parallel; job 3 (healthcheck,
# GPU) starts only if both succeed.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_setup_chain.sh
#
# Then disconnect. Poll later with:  squeue -u $USER
# When idle, read:  wan_experiment/results/setup_healthcheck/report.json

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

echo "Submitting Wan/Self-Forcing setup chain (account=${ACCOUNT})"

J1=$(sbatch --parsable --account="${ACCOUNT}" "${SB}/setup_env.sbatch")
echo "  [1/3] setup_env      job ${J1}  (GPU, ~2h)"

J2=$(sbatch --parsable --account="${ACCOUNT}" "${SB}/download_assets.sbatch")
echo "  [2/3] download       job ${J2}  (CPU, ~4h, parallel with 1)"

J3=$(sbatch --parsable --account="${ACCOUNT}" \
    --dependency=afterok:${J1}:${J2} \
    "${SB}/healthcheck.sbatch")
echo "  [3/3] healthcheck    job ${J3}  (GPU, afterok ${J1}+${J2})"

echo ""
echo "Disconnect is fine. When ${J3} finishes:"
echo "  cat ${PROJECT_ROOT}/wan_experiment/results/setup_healthcheck/report.json"
echo "  tail -n 80 ${PROJECT_ROOT}/wan_experiment/slurm_log/wan_healthcheck_${J3}.out"
echo ""
echo "Cancel the whole chain:  scancel ${J1} ${J2} ${J3}"
