#!/usr/bin/env bash
# Generate population + per-video VBench++ breakdown (Panda 1000v).
set -euo pipefail

REPO="${REPO:-/scratch/wc3013/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-$(date +%Y-%m-%d)}"
OUT="${OUT:-$REPO/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/vbench_breakdown.md}"

cd "$REPO"
python3 scripts/summarize_vbench_population_per_video.py \
  --baseline-dir sweep_experiment/results/panda_1000v_standard/NOTTA \
  --output "$OUT" \
  --method "NOTTA:sweep_experiment/results/panda_1000v_standard/NOTTA:No-TTA" \
  --method "ADA:sweep_experiment/results/panda_1000v_standard/ADA:AdaSteer" \
  --method "LORA:sweep_experiment/results/panda_1000v_standard/LORA_R8_TTA:LoRA-R8" \
  --method "K5_SIM:sweep_experiment/results/panda_1000v_retrieval/K5_SIM:AdaSteer+K5 SIM" \
  --method "K5_RAND:sweep_experiment/results/panda_1000v_retrieval/K5_RAND:AdaSteer+K5 RAND" \
  --method "K10_SIM:sweep_experiment/results/panda_1000v_retrieval/K10_SIM:AdaSteer+K10 SIM" \
  --method "K10_RAND:sweep_experiment/results/panda_1000v_retrieval/K10_RAND:AdaSteer+K10 RAND"

echo "Report: $OUT"
