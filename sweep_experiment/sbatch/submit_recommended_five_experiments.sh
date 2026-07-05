#!/usr/bin/env bash
# Submit the recommended 5-experiment program (Exp1-4 CPU; Exp5 stub only).
#
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   bash sweep_experiment/sbatch/submit_recommended_five_experiments.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DATE_TAG="${DATE_TAG:-2026-07-05}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

EXPERIMENTS=(
  exp1_probe_and_route
  exp2_dyn_delta_router
  exp3_pairwise_ranker
  exp4_bestof3_nr_proxy
  exp5_iq_constrained
)

N=${#EXPERIMENTS[@]}
EXP_LIST="${PROJECT_ROOT}/sweep_experiment/sbatch/.recommended_five_experiments.lst"
printf '%s\n' "${EXPERIMENTS[@]}" > "${EXP_LIST}"

echo "Submitting ${N} jobs (Exp1-4 CPU analysis; Exp5 writes skip stub)..."

JOB=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name=five_exp \
  --array=0-$((N - 1)) \
  --cpus-per-task=4 \
  --mem=16G \
  --time=01:00:00 \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG},FEATURE_DATE=${FEATURE_DATE}" \
  sweep_experiment/sbatch/run_recommended_five_array.sbatch)

echo "Array job: ${JOB}"

AGG=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --job-name=five_exp_agg \
  --dependency=afterany:${JOB} \
  --cpus-per-task=1 \
  --mem=4G \
  --time=00:10:00 \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG}" \
  sweep_experiment/sbatch/run_recommended_five_aggregate.sbatch)

echo "Aggregate job: ${AGG}"
echo "Results: sweep_experiment/reports/per_video_analysis/${DATE_TAG}/recommended_five_experiments/"
echo ""
echo "Note: Exp5 (IQ-constrained TTA) needs a separate GPU implementation — see exp5_iq_constrained.json"
