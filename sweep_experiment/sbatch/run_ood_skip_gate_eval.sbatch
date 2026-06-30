#!/bin/bash
#SBATCH --job-name=ood_skip_gate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=sweep_experiment/slurm_log/ood_skip_gate_%j.out
#SBATCH --error=sweep_experiment/slurm_log/ood_skip_gate_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-06-30}"
BASE="${BASE:-${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/${DATE_TAG}}"

cd "${PROJECT_ROOT}"

python3 scripts/eval_ood_skip_gate_policy.py \
  --gains-csv "${BASE}/vbench_agreement/per_video_vbench_gains.csv" \
  --ood-csv "${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv" \
  --output-dir "${BASE}/ood_skip_gate"

echo "Done: ${BASE}/ood_skip_gate/ood_skip_gate_summary.md"
