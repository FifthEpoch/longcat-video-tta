#!/usr/bin/env bash
# Submit matched pilot FVD baselines (GPU, proper longcat conda in sbatch).
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
POLICIES="${POLICIES:-always_notta fixed_S10_LR5e3}"

cd "$REPO"

if [ ! -f sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/manifest.json ]; then
  echo "ERROR: run budget oracle FVD build first (need manifest.json)" >&2
  exit 1
fi

export POLICIES
JOB=$(sbatch --parsable --account="${ACCOUNT}" \
  --export=ALL,POLICIES \
  sweep_experiment/sbatch/run_pilot_matched_fvd.sbatch)
echo "Submitted pilot_matched_fvd job ${JOB}"
echo "  POLICIES=${POLICIES}"
echo "Monitor: sacct -j ${JOB} -X  &&  tail -f sweep_experiment/slurm_log/pilot_matched_fvd_${JOB}.out"
