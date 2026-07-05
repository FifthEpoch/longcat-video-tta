#!/usr/bin/env bash
# Wave-2 GPU follow-up placeholder — submit ONLY if wave1_decision.json has gpu_go=true.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   DEC=$(python3 -c "import json;print(json.load(open('sweep_experiment/reports/per_video_analysis/2026-07-06/wave1_predictor_experiments/wave1_decision.json'))['gpu_go'])")
#   [ "$DEC" = "True" ] && bash sweep_experiment/sbatch/submit_vbench_predictor_wave2.sh
#
# TODO (Wave-2): VideoAlign probe scoring, CFG-gap GPU extract, 999v 3-way gate retrain.
set -euo pipefail

echo "Wave-2 not yet implemented. Check wave1_decision.json first."
echo "If GO: implement VideoAlign on probe mp4s + CFG-gap feature job."
exit 1
