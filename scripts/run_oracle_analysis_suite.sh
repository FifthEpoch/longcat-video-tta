#!/usr/bin/env bash
# Oracle + cross-metric analysis suite (run on cluster after git pull).
#
# Produces under sweep_experiment/reports/per_video_analysis/${DATE_TAG}/:
#   oracle_vbench/oracle_vbench_summary.md
#   cross_metric_corr/heatmap_*.png, scatter_*.png, correlation_summary.md
#   metric_cache/wide_metrics.csv  (reused if sources unchanged)
#
# Prerequisites:
#   - per_video_vbench_gains.csv from run_panda_vbench_agreement.sh
#   - diffusion_ood_scores.csv + video_features from Phase-0 pipeline
#   - Optional: phase1_oracle_fvd for method-oracle FVD row
#   - Budget FVD: run_budget_oracle_fvd.py with NO_SAVE_VIDEOS=0 (see below)
set -euo pipefail

REPO="${REPO:-/scratch/wc3013/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
BASE="${BASE:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}}"
VBENCH_CSV="${VBENCH_CSV:-$BASE/vbench_agreement/per_video_vbench_gains.csv}"
OOD_CSV="${OOD_CSV:-$REPO/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv}"
CACHE_DIR="${CACHE_DIR:-$BASE/metric_cache}"
METHOD_FVD="${METHOD_FVD:-$REPO/sweep_experiment/reports/phase1_oracle_fvd/oracle_best_psnr/fvd.json}"
BUDGET_FVD="${BUDGET_FVD:-$REPO/sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/fvd.json}"
BUDGET_SERIES="${BUDGET_SERIES:-$REPO/sweep_experiment/results/panda_ood_budget_pilot}"
BASELINE_SERIES="${BASELINE_SERIES:-$REPO/sweep_experiment/results/panda_1000v_standard}"

cd "$REPO"

if [[ ! -f "$VBENCH_CSV" ]]; then
  echo "Missing $VBENCH_CSV — run: bash scripts/run_panda_vbench_agreement.sh" >&2
  exit 1
fi

echo "=== 1. Oracle VBench++ analysis ==="
python3 scripts/analyze_oracle_vbench.py \
  --vbench-gains-csv "$VBENCH_CSV" \
  --ood-csv "$OOD_CSV" \
  --cache-dir "$CACHE_DIR" \
  --budget-series-root "$BUDGET_SERIES" \
  --baseline-series-root "$BASELINE_SERIES" \
  --method-fvd-json "$METHOD_FVD" \
  --budget-fvd-json "$BUDGET_FVD" \
  --output-dir "$BASE/oracle_vbench"

echo ""
echo "=== 2. Cross-metric correlation plots ==="
python3 scripts/plot_cross_metric_correlations.py \
  --vbench-gains-csv "$VBENCH_CSV" \
  --ood-csv "$OOD_CSV" \
  --cache-dir "$CACHE_DIR" \
  --output-dir "$BASE/cross_metric_corr" \
  --method-dirs "NOTTA:$BASELINE_SERIES/NOTTA" \
  --method-dirs "ADA:$BASELINE_SERIES/ADA" \
  --method-dirs "LORA_R8_TTA:$BASELINE_SERIES/LORA_R8_TTA" \
  --method-dirs "K5_SIM:$REPO/sweep_experiment/results/panda_1000v_retrieval/K5_SIM" \
  --method-dirs "K10_SIM:$REPO/sweep_experiment/results/panda_1000v_retrieval/K10_SIM"

echo ""
echo "=== 3. VBench predictor table (reuse cache) ==="
python3 scripts/correlate_vbench_gain_with_features.py \
  --gains-csv "$VBENCH_CSV" \
  --features-csv "$REPO/sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv" \
  --ood-csv "$OOD_CSV" \
  --output-dir "$BASE/vbench_predictors"

echo ""
echo "=== 4. Win/loss magnitudes (reuse CSV) ==="
python3 scripts/analyze_vbench_magnitude_from_csv.py \
  "$VBENCH_CSV" \
  --output "$BASE/vbench_agreement/vbench_magnitude_summary.md"

echo ""
echo "Done."
echo "  Oracle VBench:     $BASE/oracle_vbench/oracle_vbench_summary.md"
echo "  Correlations:      $BASE/cross_metric_corr/"
echo ""
echo "Budget-oracle FVD ceiling (NOT yet computed for pilot — NO_SAVE_VIDEOS=1):"
echo "  # Re-run pilot/1000v best configs with NO_SAVE_VIDEOS=0, then:"
echo "  python3 sweep_experiment/scripts/run_budget_oracle_fvd.py \\"
echo "    --series-root $BUDGET_SERIES \\"
echo "    --gt-cache gt_caches/panda_1000_longcat.npz"
echo "  # For 1000v best configs:"
echo "  #   --series-root sweep_experiment/results/panda_ood_budget_1000v"
