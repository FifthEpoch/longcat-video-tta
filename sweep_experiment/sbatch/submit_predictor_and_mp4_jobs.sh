#!/usr/bin/env bash
# Submit parallel work: cheap CPU predictor analysis + optional GPU mp4 re-run.
#
# Phase A (always): 1× CPU job — Steps 1–3 predictor transfer (~5 min)
# Phase B (optional): 24× GPU jobs — budget pilot mp4 re-run for oracle FVD/VBench
#
# Usage:
#   cd /scratch/wc3013/longcat-video-tta
#   git pull
#   bash sweep_experiment/sbatch/submit_predictor_and_mp4_jobs.sh
#
# CPU only (skip mp4 re-run):
#   SUBMIT_MP4=0 bash sweep_experiment/sbatch/submit_predictor_and_mp4_jobs.sh
#
# Mp4 only (predictor analysis already done):
#   SUBMIT_PREDICTOR=0 SUBMIT_MP4=1 bash sweep_experiment/sbatch/submit_predictor_and_mp4_jobs.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DATE_TAG="${DATE_TAG:-2026-06-30}"
FEATURE_DATE="${FEATURE_DATE:-2026-06-09}"
SUBMIT_PREDICTOR="${SUBMIT_PREDICTOR:-1}"
SUBMIT_MP4="${SUBMIT_MP4:-1}"
RUN_ROUTER="${RUN_ROUTER:-1}"

ONLY_RUNS="${ONLY_RUNS:-S20_LR1e2 S10_LR1e2 S2_LR5e3 S20_LR1e3 S20_LR5e3 S2_LR1e3 S5_LR1e2 S5_LR1e3 S10_LR5e3 S5_LR5e3 S10_LR1e3 S2_LR1e2}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

echo "============================================================"
echo "Predictor transfer + optional mp4 re-run"
echo "============================================================"
echo "  SUBMIT_PREDICTOR : ${SUBMIT_PREDICTOR}"
echo "  SUBMIT_MP4       : ${SUBMIT_MP4}"
echo "  DATE_TAG         : ${DATE_TAG}"
echo "  RUN_ROUTER       : ${RUN_ROUTER}"
echo "============================================================"
echo ""

if [[ "${SUBMIT_PREDICTOR}" == "1" ]]; then
  echo "========== Phase A: CPU predictor analysis =========="
  PRED_JOB=$(sbatch --parsable --account="${ACCOUNT}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},DATE_TAG=${DATE_TAG},FEATURE_DATE=${FEATURE_DATE},RUN_ROUTER=${RUN_ROUTER}" \
    sweep_experiment/sbatch/run_predictor_analysis.sbatch)
  echo "Submitted pred_transfer: ${PRED_JOB}"
  echo "  tail -f sweep_experiment/slurm_log/pred_transfer_${PRED_JOB}.out"
  echo ""
fi

if [[ "${SUBMIT_MP4}" == "1" ]]; then
  echo "========== Phase B: GPU mp4 re-run (24 jobs) =========="
  echo "Clears checkpoints + NO_SAVE_VIDEOS=0 for oracle FVD/VBench backfill."
  N_MP4=$(find sweep_experiment/results/panda_ood_budget_pilot -path '*/videos/*.mp4' 2>/dev/null | wc -l | tr -d ' ')
  if [[ "${N_MP4}" -gt 0 ]]; then
    echo "WARNING: ${N_MP4} mp4s already exist — re-run will overwrite/regenerate chunks."
  fi
  ONLY_RUNS="${ONLY_RUNS}" NO_SAVE_VIDEOS=0 FORCE_MP4_RERUN=1 \
    bash sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh
  echo ""
  echo "Monitor mp4 jobs:"
  echo "  squeue -u \$USER | grep adb_pilot"
  echo "  find sweep_experiment/results/panda_ood_budget_pilot -name '*.mp4' | wc -l"
  echo ""
  echo "After mp4 jobs finish (~14h/chunk):"
  echo "  bash sweep_experiment/sbatch/submit_budget_pilot_vbench_backfill.sh"
  echo "  sbatch --account=${ACCOUNT} sweep_experiment/sbatch/run_budget_oracle_fvd.sbatch"
  echo "  bash scripts/run_budget_pilot_vbench_oracle.sh"
fi

echo ""
echo "Predictor outputs (after Phase A):"
echo "  sweep_experiment/reports/per_video_analysis/${DATE_TAG}/predictor_transfer/"
