#!/usr/bin/env bash
# Extract rich VAE latent profiles for budget pilot (200 videos) + optional router eval.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   bash sweep_experiment/sbatch/submit_vae_latent_profile_pilot.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
DATE_TAG="${DATE_TAG:-2026-07-06}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

EXTRACT_JID=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},OUTPUT_CSV=${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}/vae_latent_profile_features.csv,VIDEOS_DIR=${PROJECT_ROOT}/datasets/panda_ood_budget_pilot_480p" \
  scripts/sbatch/run_extract_vae_latent_profile.sbatch)

EVAL_JID=$(sbatch --parsable \
  --account="${ACCOUNT}" \
  --dependency=afterok:${EXTRACT_JID} \
  --cpus-per-task=4 \
  --mem=16G \
  --time=00:30:00 \
  --job-name=vae_latent_eval \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},FEATURE_DATE=${FEATURE_DATE},DATE_TAG=${DATE_TAG}" \
  --wrap="cd ${PROJECT_ROOT} && bash scripts/run_vbench_latent_profile_router.sh")

echo "Extract: ${EXTRACT_JID}"
echo "Eval (after extract): ${EVAL_JID}"
echo ""
echo "When done:"
echo "  cat sweep_experiment/reports/per_video_analysis/${DATE_TAG}/vae_latent_profile_router/summary.md"
