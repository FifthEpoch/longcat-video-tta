#!/usr/bin/env bash
# Full CPU analysis chain after mp4 + VBench backfill (no GPU).
#
# Usage:
#   DATE_TAG=2026-07-02 bash scripts/run_post_mp4_analysis_chain.sh
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
FEATURE_DATE="${FEATURE_DATE:-2026-06-09}"
BASE="$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}"
FEAT="$REPO/sweep_experiment/reports/per_video_analysis/${FEATURE_DATE}"

cd "$REPO"
mkdir -p "$BASE"

echo "=== 1/5 VBench agreement CSV (if missing) ==="
if [ ! -f "$BASE/vbench_agreement/per_video_vbench_gains.csv" ]; then
  bash scripts/run_panda_vbench_agreement.sh || true
else
  echo "  skip — CSV exists"
fi

echo ""
echo "=== 2/5 Budget VBench sliding-config oracle ==="
DATE_TAG="$DATE_TAG" bash scripts/run_budget_vbench_sliding_analysis.sh

echo ""
echo "=== 3/5 Oracle + cross-metric suite ==="
DATE_TAG="$DATE_TAG" bash scripts/run_oracle_analysis_suite.sh

echo ""
echo "=== 4/5 Predictor transfer (Steps 1–3) ==="
DATE_TAG="$DATE_TAG" FEATURE_DATE="$FEATURE_DATE" bash scripts/run_predictor_analysis_suite.sh

echo ""
echo "=== 5/5 VBench headroom router (learned, objective=VBench++) ==="
bash scripts/run_vbench_headroom_router.sh

echo ""
echo "Done. Key outputs:"
echo "  $BASE/adasteer_budget_vbench_oracle_pilot.md"
echo "  $BASE/vbench_headroom_router/vbench_headroom_router_summary.md"
echo "  $BASE/oracle_vbench/oracle_vbench_summary.md"
