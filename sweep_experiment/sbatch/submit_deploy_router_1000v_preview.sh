#!/usr/bin/env bash
# CPU router eval suite for 1000v OOD-preview (after sweep merge + features + VBench).
#
# Runs (chained):
#   1) deploy_strict_router (Block A/B/C experiments)
#   2) deploy_router_aux_metrics (PSNR/SSIM/LPIPS on VB picks)
#   3) deploy_psnr_router
#   4) router_objective_alignment
#
# Prereqs:
#   - merge_chunks on panda_ood_budget_1000v_preview
#   - bash scripts/sbatch/submit_preview_1000v_features.sh  (done)
#   - VBench backfill on saved mp4s (for VB-target routers)
#
# Submit:
#   bash sweep_experiment/sbatch/submit_deploy_router_1000v_preview.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/preview_1000v_env.sh
source "${SCRIPT_DIR}/../../scripts/preview_1000v_env.sh"

PROJECT_ROOT="${REPO}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

J1=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --cpus-per-task=8 \
    --mem=32G \
    --time=02:00:00 \
    --job-name=router_prev1k_strict \
    --output="sweep_experiment/slurm_log/deploy_strict_router_preview_%j.out" \
    --export="ALL,REPO=${PROJECT_ROOT},PREVIEW_FEATURE_DATE=${PREVIEW_FEATURE_DATE},PREVIEW_DATE_TAG=${PREVIEW_DATE_TAG}" \
    --wrap="cd ${PROJECT_ROOT} && bash scripts/run_deploy_strict_router_preview.sh")

J2=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=01:00:00 \
    --dependency=afterok:${J1} \
    --job-name=router_prev1k_aux \
    --output="sweep_experiment/slurm_log/deploy_router_aux_preview_%j.out" \
    --export="ALL,REPO=${PROJECT_ROOT},PREVIEW_FEATURE_DATE=${PREVIEW_FEATURE_DATE},RUN_FVD=0" \
    --wrap="cd ${PROJECT_ROOT} && bash scripts/run_deploy_router_aux_metrics_preview.sh")

J3=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=01:00:00 \
    --dependency=afterok:${J1} \
    --job-name=router_prev1k_psnr \
    --output="sweep_experiment/slurm_log/deploy_psnr_router_preview_%j.out" \
    --export="ALL,REPO=${PROJECT_ROOT},PREVIEW_FEATURE_DATE=${PREVIEW_FEATURE_DATE}" \
    --wrap="cd ${PROJECT_ROOT} && bash scripts/run_deploy_psnr_router_preview.sh")

J4=$(sbatch --parsable \
    --account="${ACCOUNT}" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:45:00 \
    --dependency=afterok:${J2}:${J3} \
    --job-name=router_prev1k_align \
    --output="sweep_experiment/slurm_log/router_align_preview_%j.out" \
    --export="ALL,REPO=${PROJECT_ROOT},PREVIEW_FEATURE_DATE=${PREVIEW_FEATURE_DATE}" \
    --wrap="cd ${PROJECT_ROOT} && bash scripts/run_router_objective_alignment_preview.sh")

echo "Deploy strict router : ${J1}"
echo "Aux metrics          : ${J2} (after ${J1})"
echo "PSNR router          : ${J3} (after ${J1})"
echo "Objective alignment  : ${J4} (after ${J2} ${J3})"
echo ""
echo "When done:"
echo "  cat ${PREVIEW_FEATURE_DIR}/deploy_strict_router/summary.md"
echo "  cat ${PREVIEW_FEATURE_DIR}/deploy_router_aux_metrics/summary.md"
echo "  cat ${PREVIEW_FEATURE_DIR}/deploy_psnr_router/summary.md"
echo "  cat ${PREVIEW_FEATURE_DIR}/router_objective_alignment/summary.md"
