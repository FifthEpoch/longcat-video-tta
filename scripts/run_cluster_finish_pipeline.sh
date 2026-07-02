#!/usr/bin/env bash
# Status check + submit/run remaining pipeline after adb_pilot / VBench jobs.
#
# Usage (cluster login):
#   cd /scratch/wc3013/longcat-video-tta
#   git pull --ff-only origin main
#   bash scripts/run_cluster_finish_pipeline.sh          # status only
#   SUBMIT_MISSING=1 bash scripts/run_cluster_finish_pipeline.sh
#   RUN_ANALYSIS=1 DATE_TAG=2026-07-02 bash scripts/run_cluster_finish_pipeline.sh
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
SUBMIT_MISSING="${SUBMIT_MISSING:-0}"
RUN_ANALYSIS="${RUN_ANALYSIS:-0}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SERIES="${SERIES:-$REPO/sweep_experiment/results/panda_ood_budget_pilot}"
TARGET_MP4="${TARGET_MP4:-2400}"

cd "$REPO"
mkdir -p sweep_experiment/slurm_log

PILOT_RUNS=(
  S2_LR1e3 S2_LR5e3 S2_LR1e2 S5_LR1e3 S5_LR5e3 S5_LR1e2
  S10_LR1e3 S10_LR5e3 S10_LR1e2 S20_LR1e3 S20_LR5e3 S20_LR1e2
)

echo "========== Cluster job snapshot =========="
squeue -u "$USER" 2>/dev/null | head -20 || true
echo ""

echo "========== adb_pilot mp4 progress (target ${TARGET_MP4} grid mp4s) =========="
total_mp4=0
need_mp4=()
for r in "${PILOT_RUNS[@]}"; do
  n=$(find "${SERIES}/${r}" -path '*/videos/*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  total_mp4=$((total_mp4 + n))
  st="OK"
  [ "$n" -lt 200 ] && st="PARTIAL" && need_mp4+=("$r")
  printf "  %-12s %3s/200 mp4s  [%s]\n" "$r" "$n" "$st"
done
echo "  TOTAL: ${total_mp4} / ${TARGET_MP4}"
echo ""

echo "========== VBench backfill coverage (IQ eval json per chunk) =========="
need_vb=()
for r in "${PILOT_RUNS[@]}"; do
  n_chunks=$(find "${SERIES}/${r}" -mindepth 1 -maxdepth 1 -type d -name 'chunk_*' 2>/dev/null | wc -l | tr -d ' ')
  n_iq=$(find "${SERIES}/${r}" -path '*/vbench_results/vbench_imaging_quality_eval_results.json' 2>/dev/null | wc -l | tr -d ' ')
  st="OK"
  if [ "$n_iq" -lt "$n_chunks" ] && [ "$(find "${SERIES}/${r}" -path '*/videos/*.mp4' 2>/dev/null | wc -l)" -gt 0 ]; then
    st="NEEDS_BACKFILL"
    need_vb+=("$r")
  fi
  printf "  %-12s vbench_chunks=%s/%s  [%s]\n" "$r" "$n_iq" "$n_chunks" "$st"
done
echo ""

echo "========== Analysis artifacts (${DATE_TAG}) =========="
BASE="$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}"
for f in \
  "$BASE/adasteer_budget_vbench_oracle_pilot.md" \
  "$BASE/vbench_headroom_router/vbench_headroom_router_summary.md" \
  "$REPO/sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/fvd.json"; do
  if [ -f "$f" ]; then
    echo "  OK $(basename "$f")"
  else
    echo "  MISSING $f"
  fi
done
echo ""

if [ "${SUBMIT_MISSING}" = "1" ]; then
  echo "========== Submit missing VBench backfill =========="
  if [ "${#need_vb[@]}" -gt 0 ]; then
    ONLY_RUNS="${need_vb[*]}" bash sweep_experiment/sbatch/submit_budget_pilot_vbench_backfill.sh
  else
    echo "  (none needed)"
  fi
  echo ""
  echo "========== Submit budget oracle FVD (if mp4s present) =========="
  if [ "$total_mp4" -ge 2000 ]; then
    sbatch --account="${ACCOUNT}" sweep_experiment/sbatch/run_budget_oracle_fvd.sbatch || true
  else
    echo "  SKIP FVD — only ${total_mp4} mp4s (wait for adb_pilot)"
  fi
fi

if [ "${RUN_ANALYSIS}" = "1" ]; then
  echo "========== CPU analysis chain =========="
  bash scripts/run_post_mp4_analysis_chain.sh
fi

echo ""
echo "Next steps:"
echo "  1. Wait for adb_pilot + vb_budget to finish"
echo "  2. SUBMIT_MISSING=1 bash scripts/run_cluster_finish_pipeline.sh"
echo "  3. RUN_ANALYSIS=1 DATE_TAG=${DATE_TAG} bash scripts/run_cluster_finish_pipeline.sh"
