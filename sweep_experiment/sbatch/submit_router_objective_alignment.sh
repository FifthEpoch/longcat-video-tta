#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log
JID=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --cpus-per-task=4 \
  --mem=16G \
  --time=00:30:00 \
  --job-name=router_align \
  --output="sweep_experiment/slurm_log/router_objective_alignment_%j.out" \
  --error="sweep_experiment/slurm_log/router_objective_alignment_%j.err" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},FEATURE_DATE=${FEATURE_DATE}" \
  --wrap="cd ${PROJECT_ROOT} && bash scripts/run_router_objective_alignment.sh")
echo "Router objective alignment: ${JID}"
echo "  cat sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/router_objective_alignment/summary.md"
