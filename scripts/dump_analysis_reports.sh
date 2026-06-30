#!/usr/bin/env bash
# Dump all analysis markdown/JSON for pasting to agent chat.
# Usage: bash scripts/dump_analysis_reports.sh [DATE_TAG]
# Example: bash scripts/dump_analysis_reports.sh 2026-06-30
set -euo pipefail

REPO="${REPO:-/scratch/wc3013/longcat-video-tta}"
DATE_TAG="${1:-$(date +%Y-%m-%d)}"
BASE="${REPO}/sweep_experiment/reports/per_video_analysis/${DATE_TAG}"

cd "$REPO"

print_file() {
  local f="$1"
  echo ""
  echo "######################################################################"
  echo "### FILE: $f"
  echo "######################################################################"
  if [[ -f "$f" ]]; then
    cat "$f"
  else
    echo "MISSING: $f"
  fi
}

echo "========== FILE INVENTORY =========="
find "$BASE" -type f \( -name '*.md' -o -name '*.json' -o -name '*.png' \) 2>/dev/null | sort || true
echo ""
echo "Budget FVD (if computed):"
ls -la "${REPO}/sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/fvd.json" 2>/dev/null || echo "  (not yet)"
ls -la "${REPO}/sweep_experiment/reports/phase1_oracle_fvd/oracle_best_psnr/fvd.json" 2>/dev/null || echo "  (missing method oracle FVD)"

print_file "${BASE}/SNAPSHOT.md"
print_file "${BASE}/oracle_vbench/oracle_vbench_summary.md"
print_file "${BASE}/cross_metric_corr/correlation_summary.md"
print_file "${BASE}/vbench_predictors/vbench_correlation_summary.md"
print_file "${BASE}/vbench_agreement/vbench_magnitude_summary.md"
print_file "${BASE}/vbench_agreement/vbench_agreement_summary.md"
print_file "${BASE}/vbench_population_and_per_video_breakdown.md"
print_file "${REPO}/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/adasteer_budget_oracle_pilot.md"
print_file "${REPO}/sweep_experiment/reports/phase1_oracle_fvd/oracle_best_psnr/fvd.json"
print_file "${REPO}/sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/fvd.json"
print_file "${REPO}/sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/manifest.json"

echo ""
echo "######################################################################"
echo "### CSV HEAD: ${BASE}/vbench_agreement/per_video_vbench_gains.csv"
echo "######################################################################"
head -3 "${BASE}/vbench_agreement/per_video_vbench_gains.csv" 2>/dev/null || echo "MISSING"
echo "..."
wc -l "${BASE}/vbench_agreement/per_video_vbench_gains.csv" 2>/dev/null || true

echo ""
echo "######################################################################"
echo "### PNG LIST: ${BASE}/cross_metric_corr/"
echo "######################################################################"
ls -la "${BASE}/cross_metric_corr/"*.png 2>/dev/null || echo "No PNGs"

echo ""
echo "========== DONE — paste full output above =========="
