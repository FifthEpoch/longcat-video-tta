#!/bin/bash
#SBATCH --job-name=vb_slide_an
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/vb_slide_an_%j.out
#SBATCH --error=sweep_experiment/slurm_log/vb_slide_an_%j.err

# CPU-only: budget-grid VBench++ sliding-config oracle after backfill completes.
#
# Submit via submit_budget_vbench_sliding_analysis.sh (chains GPU backfill → this job).
#
# Manual:
#   sbatch --export=ALL,DATE_TAG=2026-07-01 \
#     sweep_experiment/sbatch/run_budget_vbench_sliding_analysis.sbatch

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
WAIT_MINUTES="${WAIT_MINUTES:-0}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

echo "Job ${SLURM_JOB_ID:-local} on $(hostname) at $(date)"
echo "  DATE_TAG=${DATE_TAG} WAIT_MINUTES=${WAIT_MINUTES}"

if [ "${WAIT_MINUTES}" -gt 0 ]; then
  echo "Polling for VBench backfill (up to ${WAIT_MINUTES} min)..."
  deadline=$(( $(date +%s) + WAIT_MINUTES * 60 ))
  while [ "$(date +%s)" -lt "${deadline}" ]; do
    n_ok=0
    for run_id in S2_LR1e3 S2_LR5e3 S2_LR1e2 S5_LR1e3 S5_LR5e3 S5_LR1e2 \
                  S10_LR1e3 S10_LR5e3 S10_LR1e2 S20_LR1e3 S20_LR5e3 S20_LR1e2; do
      if find "sweep_experiment/results/panda_ood_budget_pilot/${run_id}" \
          -path '*/vbench_results/vbench_imaging_quality_eval_results.json' 2>/dev/null | grep -q .; then
        n_ok=$((n_ok + 1))
      fi
    done
    echo "  configs with IQ vbench: ${n_ok}/12"
    if [ "${n_ok}" -ge 12 ]; then
      break
    fi
    sleep 120
  done
fi

export REPO="${PROJECT_ROOT}"
export DATE_TAG
bash scripts/run_budget_vbench_sliding_analysis.sh

echo "Finished at $(date)"
