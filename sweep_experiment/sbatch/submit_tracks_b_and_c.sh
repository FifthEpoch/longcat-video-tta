#!/usr/bin/env bash
# Fire Track B (Panda retrieval) and Track C (DOVER probe routing) in parallel.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   bash sweep_experiment/sbatch/submit_tracks_b_and_c.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
cd "${PROJECT_ROOT}"

echo "============================================================"
echo "Launching Track B + Track C in parallel"
echo "============================================================"

bash sweep_experiment/sbatch/submit_panda_1000v_retrieval.sh &
PID_B=$!

bash sweep_experiment/sbatch/submit_dover_probe_routing.sh &
PID_C=$!

wait "${PID_B}" && echo "Track B submit OK" || echo "Track B submit FAILED"
wait "${PID_C}" && echo "Track C submit OK" || echo "Track C submit FAILED"

echo ""
echo "Monitor:"
echo "  squeue -u \$USER | grep -E 't1kr_panda|dover_'"
