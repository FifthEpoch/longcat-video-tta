#!/bin/bash
# ==============================================================================
# External Method Comparison: Master Submission Script
#
# Submits all jobs in the correct dependency order:
#   Phase 1: Environment setup (PVDM + DFoT conda envs)
#   Phase 2: Data preparation (reformat UCF-101 for each method)
#   Phase 3: Evaluation (PVDM baseline, SAVi-DNO, DFoT)
#
# Usage (on cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash comparison_methods/scripts/submit_all.sh
#
# To run only specific phases:
#   bash comparison_methods/scripts/submit_all.sh --phase 1    # env setup only
#   bash comparison_methods/scripts/submit_all.sh --phase 2    # data prep only
#   bash comparison_methods/scripts/submit_all.sh --phase 3    # eval only
# ==============================================================================
set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
PROJECT_ROOT="/scratch/wc3013/longcat-video-tta"
PHASE="${1:---all}"

cd "${PROJECT_ROOT}"
mkdir -p comparison_methods/slurm_log

echo "=============================================================================="
echo "External Method Comparison: Job Submission"
echo "=============================================================================="
echo "Account: ${ACCOUNT}"
echo "Phase: ${PHASE}"
echo "=============================================================================="
echo ""

# ==============================================================================
# Phase 1: Environment Setup
# ==============================================================================
if [ "${PHASE}" = "--all" ] || [ "${PHASE}" = "--phase" -a "${2:-}" = "1" ] || [ "${PHASE}" = "1" ]; then
    echo ">>> Phase 1: Submitting environment setup jobs..."

    PVDM_ENV_JOB=$(sbatch --account="${ACCOUNT}" \
        --parsable \
        comparison_methods/env_setup/setup_pvdm_env.sbatch)
    echo "  PVDM env setup: Job ${PVDM_ENV_JOB}"

    DFOT_ENV_JOB=$(sbatch --account="${ACCOUNT}" \
        --parsable \
        comparison_methods/env_setup/setup_dfot_env.sbatch)
    echo "  DFoT env setup: Job ${DFOT_ENV_JOB}"

    echo ""
fi

# ==============================================================================
# Phase 2: Data Preparation
# ==============================================================================
if [ "${PHASE}" = "--all" ] || [ "${PHASE}" = "--phase" -a "${2:-}" = "2" ] || [ "${PHASE}" = "2" ]; then
    echo ">>> Phase 2: Submitting data preparation jobs..."

    # These run on CPU partition since they only need ffmpeg
    PVDM_DEP=""
    DFOT_DEP=""
    if [ -n "${PVDM_ENV_JOB:-}" ]; then
        PVDM_DEP="--dependency=afterok:${PVDM_ENV_JOB}"
    fi
    if [ -n "${DFOT_ENV_JOB:-}" ]; then
        DFOT_DEP="--dependency=afterok:${DFOT_ENV_JOB}"
    fi

    PVDM_DATA_JOB=$(sbatch --account="${ACCOUNT}" \
        --parsable \
        --partition=cpu_short \
        --job-name=prep_pvdm \
        --time=01:00:00 \
        --cpus-per-task=8 \
        --mem=16GB \
        --output=comparison_methods/slurm_log/prep_pvdm_%j.out \
        --error=comparison_methods/slurm_log/prep_pvdm_%j.err \
        ${PVDM_DEP} \
        --wrap="cd ${PROJECT_ROOT} && \
                python comparison_methods/data/prepare_ucf101_pvdm.py \
                    --src-dir datasets/ucf101_500_480p \
                    --dst-dir comparison_methods/data/ucf101_pvdm")
    echo "  PVDM data prep: Job ${PVDM_DATA_JOB}"

    DFOT_DATA_JOB=$(sbatch --account="${ACCOUNT}" \
        --parsable \
        --partition=cpu_short \
        --job-name=prep_dfot \
        --time=01:00:00 \
        --cpus-per-task=8 \
        --mem=16GB \
        --output=comparison_methods/slurm_log/prep_dfot_%j.out \
        --error=comparison_methods/slurm_log/prep_dfot_%j.err \
        ${DFOT_DEP} \
        --wrap="cd ${PROJECT_ROOT} && \
                python comparison_methods/data/prepare_ucf101_dfot.py \
                    --src-dir datasets/ucf101_500_480p \
                    --dst-dir comparison_methods/data/ucf101_dfot")
    echo "  DFoT data prep: Job ${DFOT_DATA_JOB}"

    echo ""
fi

# ==============================================================================
# Phase 3: Evaluation
# ==============================================================================
if [ "${PHASE}" = "--all" ] || [ "${PHASE}" = "--phase" -a "${2:-}" = "3" ] || [ "${PHASE}" = "3" ]; then
    echo ">>> Phase 3: Submitting evaluation jobs..."

    PVDM_EVAL_DEP=""
    DFOT_EVAL_DEP=""
    if [ -n "${PVDM_DATA_JOB:-}" ]; then
        PVDM_EVAL_DEP="--dependency=afterok:${PVDM_DATA_JOB}"
    fi
    if [ -n "${DFOT_DATA_JOB:-}" ]; then
        DFOT_EVAL_DEP="--dependency=afterok:${DFOT_DATA_JOB}"
    fi

    # PVDM baseline
    PVDM_EVAL_JOB=$(sbatch --account="${ACCOUNT}" \
        --parsable \
        ${PVDM_EVAL_DEP} \
        comparison_methods/sbatch/run_pvdm.sbatch)
    echo "  PVDM baseline eval: Job ${PVDM_EVAL_JOB}"

    # SAVi-DNO (10 DDIM steps)
    SAVI_10_JOB=$(sbatch --account="${ACCOUNT}" \
        --parsable \
        ${PVDM_EVAL_DEP} \
        --export=ALL,DDIM_STEPS=10,SAVI_LR=0.01,SAVI_LAM=0.0012,SAVI_P=0.9 \
        comparison_methods/sbatch/run_savi_dno.sbatch)
    echo "  SAVi-DNO (10 steps): Job ${SAVI_10_JOB}"

    # SAVi-DNO (50 DDIM steps) - optional longer run
    SAVI_50_JOB=$(sbatch --account="${ACCOUNT}" \
        --parsable \
        ${PVDM_EVAL_DEP} \
        --export=ALL,DDIM_STEPS=50,SAVI_LR=0.01,SAVI_LAM=0.0012,SAVI_P=0.9 \
        comparison_methods/sbatch/run_savi_dno.sbatch)
    echo "  SAVi-DNO (50 steps): Job ${SAVI_50_JOB}"

    # DFoT
    DFOT_EVAL_JOB=$(sbatch --account="${ACCOUNT}" \
        --parsable \
        ${DFOT_EVAL_DEP} \
        comparison_methods/sbatch/run_dfot.sbatch)
    echo "  DFoT eval: Job ${DFOT_EVAL_JOB}"

    echo ""
fi

echo "=============================================================================="
echo "Submission Complete"
echo "=============================================================================="
echo "Monitor: squeue -u \$USER"
echo "Results: comparison_methods/results/"
echo ""
echo "After all jobs finish, run:"
echo "  python comparison_methods/scripts/compare_all.py \\"
echo "      --results-dir comparison_methods/results \\"
echo "      --longcat-dir sweep_experiment/results \\"
echo "      --latex"
echo "=============================================================================="
