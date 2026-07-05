#!/bin/bash
#SBATCH --job-name=dover_eval
#SBATCH --output=sweep_experiment/slurm_log/dover_eval_%j.out
#SBATCH --error=sweep_experiment/slurm_log/dover_eval_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
OUT_SCORES="${OUT_SCORES:?}"
OUT_EVAL="${OUT_EVAL:?}"

cd "${PROJECT_ROOT}"
python3 scripts/run_dover_probe_routing_eval.py \
  --dover-csv-dir "${OUT_SCORES}" \
  --output-dir "${OUT_EVAL}"

echo "Done: ${OUT_EVAL}/exp13_dover_probe_route.md"
