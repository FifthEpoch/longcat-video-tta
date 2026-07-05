#!/usr/bin/env bash
# Submit VBench gain prediction experiments (7 parallel CPU jobs + aggregate).
#
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   bash sweep_experiment/sbatch/submit_vbench_gain_prediction_experiments.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DATE_TAG="${DATE_TAG:-2026-07-05}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

EXPERIMENTS=(
  exp6_knn_oracle_transfer
  exp7_gain_predictor_probe
  exp8_abstain_route_3way
  exp9_multitask_aestech
  exp10_dover_aestech_proxy
  exp11_tier3_probe_ridge
  exp12_trajectory_ridge
)

N=${#EXPERIMENTS[@]}
EXP_LIST="${PROJECT_ROOT}/sweep_experiment/sbatch/.vbench_gain_prediction_experiments.lst"
printf '%s\n' "${EXPERIMENTS[@]}" > "${EXP_LIST}"

echo "Submitting ${N} parallel CPU jobs..."

JOB=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name=vb_gain \
  --array=0-$((N - 1)) \
  --cpus-per-task=4 \
  --mem=16G \
  --time=01:00:00 \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG},FEATURE_DATE=${FEATURE_DATE}" \
  sweep_experiment/sbatch/run_vbench_gain_prediction_array.sbatch)

echo "Array job: ${JOB}"

AGG=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name=vb_gain_agg \
  --dependency=afterany:${JOB} \
  --cpus-per-task=1 \
  --mem=4G \
  --time=00:10:00 \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG}" \
  sweep_experiment/sbatch/run_vbench_gain_prediction_aggregate.sbatch)

echo "Aggregate job: ${AGG}"
echo "Results: sweep_experiment/reports/per_video_analysis/${DATE_TAG}/vbench_gain_prediction_experiments/"
