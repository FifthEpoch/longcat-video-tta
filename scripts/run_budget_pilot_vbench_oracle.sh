#!/usr/bin/env bash
# Re-run budget config oracle VBench++ after pilot VBench backfill completes.
set -euo pipefail

REPO="${REPO:-/scratch/wc3013/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
BASE="${BASE:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}}"
VBENCH_CSV="${VBENCH_CSV:-$BASE/vbench_agreement/per_video_vbench_gains.csv}"
OOD_CSV="${OOD_CSV:-$REPO/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv}"
BUDGET_SERIES="${BUDGET_SERIES:-$REPO/sweep_experiment/results/panda_ood_budget_pilot}"

cd "$REPO"

# Quick sanity: at least one grid run has vbench json
sample=$(find "${BUDGET_SERIES}" -path '*/vbench_results/vbench_aesthetic_quality_eval_results.json' 2>/dev/null | head -1)
if [ -z "${sample}" ]; then
    echo "ERROR: no VBench results under ${BUDGET_SERIES} — run submit_budget_pilot_vbench_backfill.sh first" >&2
    exit 1
fi

python3 scripts/analyze_oracle_vbench.py \
  --vbench-gains-csv "$VBENCH_CSV" \
  --ood-csv "$OOD_CSV" \
  --budget-series-root "$BUDGET_SERIES" \
  --baseline-series-root "$REPO/sweep_experiment/results/panda_1000v_standard" \
  --method-fvd-json "$REPO/sweep_experiment/reports/phase1_oracle_fvd/oracle_best_psnr/fvd.json" \
  --budget-fvd-json "$REPO/sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/fvd.json" \
  --output-dir "$BASE/oracle_vbench"

echo "Updated $BASE/oracle_vbench/oracle_vbench_summary.md (check §3 budget VBench oracle)"
