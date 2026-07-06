#!/usr/bin/env bash
# Wave-2: learned verifier routing (Options 1–4). Replaces placeholder.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull
#   bash scripts/setup_verifier_models.sh   # first time only
#   bash sweep_experiment/sbatch/submit_vbench_predictor_wave2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec bash "${SCRIPT_DIR}/submit_verifier_options_pilot.sh" "$@"
