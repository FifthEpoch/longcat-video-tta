#!/usr/bin/env bash
# Submit Wave-1 predictor screen (7 parallel CPU jobs + aggregate).
#
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   bash sweep_experiment/sbatch/submit_vbench_predictor_wave1.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DATE_TAG="${DATE_TAG:-2026-07-06}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

EXPERIMENTS=(
  exp14_multi_verifier_deploy
  exp14_multi_verifier_full
  exp15_tail_only_gate
  exp16_knn_probe_manifold
  exp17_per_dim_fuse_router
  exp18_logistic_3way_gate
  exp19_feature_dim_correlation
)

N=${#EXPERIMENTS[@]}
EXP_LIST="${PROJECT_ROOT}/sweep_experiment/sbatch/.wave1_predictor_experiments.lst"
printf '%s\n' "${EXPERIMENTS[@]}" > "${EXP_LIST}"

echo "Submitting ${N} Wave-1 CPU jobs (~5 min each)..."

JOB=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name=wave1_pred \
  --array=0-$((N - 1)) \
  --cpus-per-task=4 \
  --mem=16G \
  --time=00:30:00 \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG},FEATURE_DATE=${FEATURE_DATE}" \
  sweep_experiment/sbatch/run_vbench_predictor_wave1_array.sbatch)

AGG=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name=wave1_agg \
  --dependency=afterany:${JOB} \
  --cpus-per-task=1 \
  --mem=4G \
  --time=00:10:00 \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG}" \
  sweep_experiment/sbatch/run_vbench_predictor_wave1_aggregate.sbatch)

echo "Array: ${JOB}"
echo "Aggregate: ${AGG}"
echo ""
echo "When done (~10 min):"
echo "  cat sweep_experiment/reports/per_video_analysis/${DATE_TAG}/wave1_predictor_experiments/wave1_predictor_summary.md"
echo "  cat sweep_experiment/reports/per_video_analysis/${DATE_TAG}/wave1_predictor_experiments/wave1_decision.json"
