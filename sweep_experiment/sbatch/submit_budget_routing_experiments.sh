#!/usr/bin/env bash
# Submit budget routing experiments as parallel CPU jobs (pilot N=200, no GPU).
#
# Usage (cluster):
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   bash sweep_experiment/sbatch/submit_budget_routing_experiments.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DATE_TAG="${DATE_TAG:-2026-07-05}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

EXPERIMENTS=(
  baseline_linear_total
  dim_dynamic_degree
  dim_aesthetic_quality
  dim_imaging_quality
  dim_subject_consistency
  coarse_steps_lr
  probe_simulated
  proxy_psnr_all
  proxy_bestof3_psnr
  pairwise_logistic_top4
  pairwise_gbm_top4
  composite_psnr_ridge
  mlp_shallow
)

N=${#EXPERIMENTS[@]}
# Slurm --export splits on commas; write one experiment name per line instead.
EXP_LIST="${PROJECT_ROOT}/sweep_experiment/sbatch/.budget_routing_experiments.lst"
printf '%s\n' "${EXPERIMENTS[@]}" > "${EXP_LIST}"

echo "Submitting ${N} parallel CPU jobs (one experiment each)..."
echo "Experiment list: ${EXP_LIST}"

JOB=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name=budget_route \
  --array=0-$((N - 1)) \
  --cpus-per-task=4 \
  --mem=16G \
  --time=02:00:00 \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG},FEATURE_DATE=${FEATURE_DATE}" \
  sweep_experiment/sbatch/run_budget_routing_experiment_array.sbatch)

echo "Array job: ${JOB}  (tasks 0-$((N - 1)))"

AGG=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name=budget_route_agg \
  --dependency=afterany:${JOB} \
  --cpus-per-task=2 \
  --mem=8G \
  --time=00:30:00 \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG},FEATURE_DATE=${FEATURE_DATE}" \
  sweep_experiment/sbatch/run_budget_routing_aggregate.sbatch)

echo "Aggregate job: ${AGG}  (after array completes)"

echo ""
echo "Monitor:"
echo "  squeue -u \$USER | grep budget_route"
echo "  tail -f sweep_experiment/slurm_log/budget_route_${JOB}_*.out"
echo "Results: sweep_experiment/reports/per_video_analysis/${DATE_TAG}/budget_routing_experiments/"
