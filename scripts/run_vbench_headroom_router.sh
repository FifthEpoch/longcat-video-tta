#!/usr/bin/env bash
# Train VBench++ headroom routers (method apply + budget sliding oracle).
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
FEATURE_DATE="${FEATURE_DATE:-2026-06-09}"
TASK="${TASK:-all}"

BASE="$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}"
FEAT="$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}"
OUT="$BASE/vbench_headroom_router"
VBENCH_CSV="${VBENCH_CSV:-$BASE/vbench_agreement/per_video_vbench_gains.csv}"
BUDGET_SERIES="${BUDGET_SERIES:-$REPO/sweep_experiment/results/panda_ood_budget_pilot}"

cd "$REPO"
mkdir -p "$OUT"

AUX=(
  --features-csv "$FEAT/video_features.csv"
  --ood-csv "$FEAT/diffusion_ood_scores.csv"
  --tier3-csv "$FEAT/tier3_probe_features.csv"
  --flow-csv "$FEAT/flow_shape_features.csv"
  --bpp-csv "$FEAT/bpp_features.csv"
  --fft-csv "$FEAT/fft_features.csv"
  --vae-recerr-csv "$FEAT/vae_recerr_features.csv"
  --motion-csv "$FEAT/latent_motion_features.csv"
  --loss-var-csv "$FEAT/loss_variance_features.csv"
  --budget-series-root "$BUDGET_SERIES"
  --task "$TASK"
  --output-dir "$OUT"
)

if [ -f "$VBENCH_CSV" ]; then
  AUX+=(--gains-csv "$VBENCH_CSV")
else
  echo "WARN: missing $VBENCH_CSV — method_gain task may fail" >&2
  AUX+=(--gains-csv "$FEAT/../2026-06-28/vbench_agreement/per_video_vbench_gains.csv")
fi

python3 scripts/train_vbench_headroom_router.py "${AUX[@]}"
