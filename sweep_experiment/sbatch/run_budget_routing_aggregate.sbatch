#!/bin/bash
#SBATCH --job-name=budget_route_agg
#SBATCH --output=sweep_experiment/slurm_log/budget_route_agg_%j.out
#SBATCH --error=sweep_experiment/slurm_log/budget_route_agg_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
DATE_TAG="${DATE_TAG:-2026-07-05}"
FEATURE_DATE="${FEATURE_DATE:-2026-07-06}"
OUT="${PROJECT_ROOT}/sweep_experiment/reports/per_video_analysis/${DATE_TAG}/budget_routing_experiments"

cd "${PROJECT_ROOT}"

python3 scripts/aggregate_budget_routing_results.py --input-dir "${OUT}"
python3 scripts/eval_budget_routing_on_999v.py --output "${OUT}/999v_proxy_bestof3.md" || true

echo "Aggregate done: ${OUT}"
