#!/usr/bin/env bash
# VBench++ sliding-config oracle on the H9 budget pilot (CPU-only analysis).
#
# Prerequisites:
#   - Per-config VBench under panda_ood_budget_pilot/*/chunk_*/vbench_results/
#   - Run submit_budget_pilot_vbench_backfill.sh first if dims are missing
#
# Usage (cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash scripts/run_budget_vbench_sliding_analysis.sh
#
# Optional:
#   DATE_TAG=2026-07-01 REQUIRE_ALL_DIMS=1 bash scripts/run_budget_vbench_sliding_analysis.sh
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
BASE="${BASE:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}}"
SERIES="${SERIES:-$REPO/sweep_experiment/results/panda_ood_budget_pilot}"
OOD_CSV="${OOD_CSV:-$REPO/sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv}"
REQUIRE_ALL_DIMS="${REQUIRE_ALL_DIMS:-0}"

cd "$REPO"
mkdir -p "$BASE"

ALL_DIMS=(
  subject_consistency background_consistency aesthetic_quality
  motion_smoothness dynamic_degree imaging_quality temporal_flickering
)
PILOT_RUNS=(
  S2_LR1e3 S2_LR5e3 S2_LR1e2
  S5_LR1e3 S5_LR5e3 S5_LR1e2
  S10_LR1e3 S10_LR5e3 S10_LR1e2
  S20_LR1e3 S20_LR5e3 S20_LR1e2
)

echo "=== VBench coverage check ==="
missing=0
for run_id in "${PILOT_RUNS[@]}"; do
  dir="${SERIES}/${run_id}"
  if [ ! -d "${dir}" ]; then
    echo "  MISSING dir: ${run_id}"
    missing=$((missing + 1))
    continue
  fi
  n_mp4=$(find "${dir}" -path '*/videos/*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  sample=$(find "${dir}" -path '*/vbench_results/vbench_aesthetic_quality_eval_results.json' 2>/dev/null | head -1)
  if [ -z "${sample}" ]; then
    echo "  NO VBENCH: ${run_id} (${n_mp4} mp4s) — needs backfill"
    missing=$((missing + 1))
  else
    echo "  OK: ${run_id} (${n_mp4} mp4s, vbench present)"
  fi
done

if [ "${missing}" -gt 0 ] && [ "${REQUIRE_ALL_DIMS}" = "1" ]; then
  echo ""
  echo "ERROR: ${missing} configs lack VBench — run backfill first:" >&2
  echo "  bash sweep_experiment/sbatch/submit_budget_pilot_vbench_backfill.sh" >&2
  exit 1
fi

if [ "${missing}" -gt 0 ]; then
  echo ""
  echo "WARN: ${missing} configs incomplete — analysis uses available dims only."
fi

OUT="${BASE}/adasteer_budget_vbench_oracle_pilot.md"
python3 scripts/analyze_adasteer_budget_vbench_oracle.py \
  --series-root "${SERIES}" \
  --baseline-series-root "${REPO}/sweep_experiment/results/panda_1000v_standard" \
  --ood-csv "${OOD_CSV}" \
  --output "${OUT}" \
  --bootstrap

echo ""
echo "=== Also refresh method+budget PSNR-oracle VBench summary (§3) ==="
VBENCH_CSV="${VBENCH_CSV:-$BASE/vbench_agreement/per_video_vbench_gains.csv}"
if [ -f "${VBENCH_CSV}" ]; then
  python3 scripts/analyze_oracle_vbench.py \
    --vbench-gains-csv "${VBENCH_CSV}" \
    --ood-csv "${OOD_CSV}" \
    --budget-series-root "${SERIES}" \
    --baseline-series-root "${REPO}/sweep_experiment/results/panda_1000v_standard" \
    --output-dir "${BASE}/oracle_vbench" || true
else
  echo "Skip analyze_oracle_vbench.py — no ${VBENCH_CSV}"
fi

echo ""
echo "Done. Report: ${OUT}"
