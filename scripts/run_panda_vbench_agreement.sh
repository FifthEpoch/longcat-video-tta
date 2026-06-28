#!/usr/bin/env bash
# Run per-video VBench++ cross-metric agreement on Panda 1000v (standard + retrieval).
# Execute on the cluster login node from repo root:
#
#   bash scripts/run_panda_vbench_agreement.sh
#
set -euo pipefail

REPO="${REPO:-/scratch/wc3013/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
OUT="${OUT:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/vbench_agreement}"

BASELINE="$REPO/sweep_experiment/results/panda_1000v_standard/NOTTA"
METHODS=(
  "$REPO/sweep_experiment/results/panda_1000v_standard/ADA"
  "$REPO/sweep_experiment/results/panda_1000v_standard/LORA_R8_TTA"
  "$REPO/sweep_experiment/results/panda_1000v_retrieval/K5_SIM"
  "$REPO/sweep_experiment/results/panda_1000v_retrieval/K5_RAND"
  "$REPO/sweep_experiment/results/panda_1000v_retrieval/K10_SIM"
  "$REPO/sweep_experiment/results/panda_1000v_retrieval/K10_RAND"
)

cd "$REPO"
python3 scripts/analyze_per_video_vbench_agreement.py \
  --baseline-dir "$BASELINE" \
  --method-dirs "${METHODS[@]}" \
  --output-dir "$OUT"

echo ""
echo "Done. Summary: $OUT/vbench_agreement_summary.md"
echo "CSV:        $OUT/per_video_vbench_gains.csv"
