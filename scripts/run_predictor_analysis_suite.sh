#!/usr/bin/env bash
# Cheap CPU predictor-transfer analysis (Steps 1–3).
#
# Outputs under sweep_experiment/reports/per_video_analysis/${DATE_TAG}/predictor_transfer/:
#   baseline_outcome_predictors.md
#   feature_outcome_battery.md
#   router_auc_summary.md
#   *.csv
#
# Prerequisites:
#   - per_video_vbench_gains.csv (run_panda_vbench_agreement.sh)
#   - Phase-0 feature CSVs under FEATURE_DATE (default 2026-06-09)
#
# Usage (cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash scripts/run_predictor_analysis_suite.sh
#
# Skip router AUC (Step 3):
#   RUN_ROUTER=0 bash scripts/run_predictor_analysis_suite.sh
set -euo pipefail

REPO="${REPO:-/scratch/wc3013/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
FEATURE_DATE="${FEATURE_DATE:-2026-06-09}"
RUN_ROUTER="${RUN_ROUTER:-1}"

BASE="${BASE:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}}"
FEAT_BASE="$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}"
OUT="$BASE/predictor_transfer"

VBENCH_CSV="${VBENCH_CSV:-$BASE/vbench_agreement/per_video_vbench_gains.csv}"
FEATURES_CSV="${FEATURES_CSV:-$FEAT_BASE/video_features.csv}"
OOD_CSV="${OOD_CSV:-$FEAT_BASE/diffusion_ood_scores.csv}"
TIER3_CSV="${TIER3_CSV:-$FEAT_BASE/tier3_probe_features.csv}"
FLOW_CSV="${FLOW_CSV:-$FEAT_BASE/flow_shape_features.csv}"
BPP_CSV="${BPP_CSV:-$FEAT_BASE/bpp_features.csv}"
FFT_CSV="${FFT_CSV:-$FEAT_BASE/fft_features.csv}"
VAE_RECERR_CSV="${VAE_RECERR_CSV:-$FEAT_BASE/vae_recerr_features.csv}"
MOTION_CSV="${MOTION_CSV:-$FEAT_BASE/latent_motion_features.csv}"
LOSS_VAR_CSV="${LOSS_VAR_CSV:-$FEAT_BASE/loss_variance_features.csv}"

cd "$REPO"
mkdir -p "$OUT"

if [[ ! -f "$VBENCH_CSV" ]]; then
  echo "Missing $VBENCH_CSV" >&2
  echo "Run: bash scripts/run_panda_vbench_agreement.sh" >&2
  exit 1
fi
if [[ ! -f "$FEATURES_CSV" ]]; then
  echo "Missing $FEATURES_CSV" >&2
  exit 1
fi

AUX_FLAGS=()
[[ -f "$OOD_CSV" ]] && AUX_FLAGS+=(--ood-csv "$OOD_CSV")
[[ -f "$TIER3_CSV" ]] && AUX_FLAGS+=(--tier3-csv "$TIER3_CSV")
[[ -f "$FLOW_CSV" ]] && AUX_FLAGS+=(--flow-csv "$FLOW_CSV")
[[ -f "$BPP_CSV" ]] && AUX_FLAGS+=(--bpp-csv "$BPP_CSV")
[[ -f "$FFT_CSV" ]] && AUX_FLAGS+=(--fft-csv "$FFT_CSV")
[[ -f "$VAE_RECERR_CSV" ]] && AUX_FLAGS+=(--vae-recerr-csv "$VAE_RECERR_CSV")
[[ -f "$MOTION_CSV" ]] && AUX_FLAGS+=(--motion-csv "$MOTION_CSV")
[[ -f "$LOSS_VAR_CSV" ]] && AUX_FLAGS+=(--loss-var-csv "$LOSS_VAR_CSV")

echo "=== Step 1/3: NOTTA baseline → Δ outcomes ==="
python3 scripts/analyze_baseline_outcome_predictors.py \
  --gains-csv "$VBENCH_CSV" \
  --output-dir "$OUT"

echo ""
echo "=== Step 2/3: Phase-0 feature battery → all Δ outcomes ==="
python3 scripts/analyze_feature_outcome_battery.py \
  --gains-csv "$VBENCH_CSV" \
  --features-csv "$FEATURES_CSV" \
  --output-dir "$OUT" \
  "${AUX_FLAGS[@]}"

if [[ "$RUN_ROUTER" == "1" ]]; then
  echo ""
  echo "=== Step 3/3: Router AUC screen ==="
  python3 scripts/analyze_router_auc.py \
    --gains-csv "$VBENCH_CSV" \
    --features-csv "$FEATURES_CSV" \
    --output-dir "$OUT" \
    "${AUX_FLAGS[@]:-}"
else
  echo ""
  echo "=== Step 3/3: skipped (RUN_ROUTER=0) ==="
fi

echo ""
echo "Done: $OUT/"
echo "  baseline_outcome_predictors.md"
echo "  feature_outcome_battery.md"
if [[ "$RUN_ROUTER" == "1" ]]; then
  echo "  router_auc_summary.md"
fi
