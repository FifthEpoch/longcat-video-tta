#!/usr/bin/env bash
# Submit tonight's analysis jobs (copy-paste on cluster after git pull / scp new scripts).
#
# 1. OOD skip-gate eval (CPU, ~2 min) — uses existing 999v CSV
# 2. Budget pilot VBench backfill (12× GPU, ~30–90 min each) — needs mp4s
# 3. Budget oracle FVD (1× GPU) — needs mp4s; optional DEP_ON_MP4=0 to skip
#
# Usage:
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_tonight_analysis_jobs.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DATE_TAG="${DATE_TAG:-2026-06-30}"
DEP_ON_MP4="${DEP_ON_MP4:-1}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

echo "========== 1/3 OOD skip-gate policy eval (CPU) =========="
SKIP_JOB=$(sbatch --parsable --account="${ACCOUNT}" \
  --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG}" \
  sweep_experiment/sbatch/run_ood_skip_gate_eval.sbatch)
echo "Submitted ood_skip_gate: ${SKIP_JOB}"

echo ""
echo "========== 2/3 Budget pilot VBench backfill (12× GPU) =========="
echo "(Skips configs with no mp4s — re-run script after mp4 jobs finish if needed)"
bash sweep_experiment/sbatch/submit_budget_pilot_vbench_backfill.sh || true

echo ""
echo "========== 3/3 Budget oracle FVD (1× GPU) =========="
N_MP4=$(find sweep_experiment/results/panda_ood_budget_pilot -path '*/videos/*.mp4' 2>/dev/null | wc -l | tr -d ' ')
if [ "${DEP_ON_MP4}" = "1" ] && [ "${N_MP4}" = "0" ]; then
    echo "SKIP budget_oracle_fvd — no mp4s yet (mp4 re-run jobs still going?)"
    echo "  When ready: sbatch sweep_experiment/sbatch/run_budget_oracle_fvd.sbatch"
else
    FVD_JOB=$(sbatch --parsable --account="${ACCOUNT}" \
      sweep_experiment/sbatch/run_budget_oracle_fvd.sbatch)
    echo "Submitted budget_oracle_fvd: ${FVD_JOB}"
fi

echo ""
echo "========== After VBench backfill jobs finish =========="
echo "  bash scripts/run_budget_pilot_vbench_oracle.sh"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f sweep_experiment/slurm_log/ood_skip_gate_${SKIP_JOB}.out"
