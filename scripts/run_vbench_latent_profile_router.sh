#!/usr/bin/env bash
# Evaluate VAE latent profile routers vs exp7 baseline (CPU, N=200).
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
DATE_TAG="${DATE_TAG:-2026-07-06}"
OUT="${OUT:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/vae_latent_profile_router}"

cd "$REPO"

python3 scripts/run_vbench_latent_profile_router.py \
  --series-root "$REPO/sweep_experiment/results/panda_ood_budget_pilot" \
  --feature-date "$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}" \
  --output-dir "$OUT"

echo "Done: $OUT/summary.md"
