#!/bin/bash
# ==============================================================================
# External Method Comparison on Panda-70M: Submission Script
#
# Assumes PVDM and DFoT conda envs are already set up (from UCF-101 runs).
# Submits data preparation + evaluation jobs for Panda-70M (1000 videos).
#
# Usage (on cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash comparison_methods/scripts/submit_panda.sh
# ==============================================================================
set -euo pipefail

ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
PROJECT_ROOT="/scratch/wc3013/longcat-video-tta"
PANDA_SRC="datasets/panda_1000_480p"
MAX_VIDEOS=1000
GT_CACHE_PVDM="${PROJECT_ROOT}/gt_caches/panda_1000_pvdm.npz"
GT_CACHE_DFOT="${PROJECT_ROOT}/gt_caches/panda_1000_dfot.npz"
SAVE_LIST="${PROJECT_ROOT}/sweep_experiment/reports/panda_retain_videos.json"

cd "${PROJECT_ROOT}"
mkdir -p comparison_methods/slurm_log

echo "=============================================================================="
echo "External Method Comparison: Panda-70M (${MAX_VIDEOS} videos)"
echo "=============================================================================="
echo "Account : ${ACCOUNT}"
echo "Source  : ${PANDA_SRC}"
echo "=============================================================================="
echo ""

# Verify dataset exists
if [ ! -f "${PANDA_SRC}/metadata.csv" ]; then
    echo "ERROR: Panda-70M dataset not found at ${PANDA_SRC}"
    echo "Download and resize first:"
    echo "  NUM_VIDEOS=1000 MIN_FRAMES=62 sbatch --account=${ACCOUNT} datasets/download_panda70m.sbatch"
    echo "  SRC_DIR=datasets/panda_1000 DST_DIR=datasets/panda_1000_480p sbatch --account=${ACCOUNT} --partition=cpu_short datasets/resize_videos.sbatch"
    exit 1
fi

# ==============================================================================
# Phase 1: Data Preparation (reformat Panda-70M for each method)
# ==============================================================================
echo ">>> Phase 1: Submitting data preparation jobs..."

CONDA_INIT="module purge && module load anaconda3/2025.06 && source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh && conda activate /scratch/wc3013/conda-envs/longcat"

PVDM_DATA_JOB=$(sbatch --account="${ACCOUNT}" \
    --parsable \
    --partition=cpu_short \
    --job-name=prep_pvdm_panda \
    --time=04:00:00 \
    --cpus-per-task=8 \
    --mem=32GB \
    --output=comparison_methods/slurm_log/prep_pvdm_panda_%j.out \
    --error=comparison_methods/slurm_log/prep_pvdm_panda_%j.err \
    --wrap="${CONDA_INIT} && cd ${PROJECT_ROOT} && \
            python comparison_methods/data/prepare_ucf101_pvdm.py \
                --src-dir ${PANDA_SRC} \
                --dst-dir comparison_methods/data/panda_pvdm")
echo "  PVDM data prep: Job ${PVDM_DATA_JOB}"

DFOT_DATA_JOB=$(sbatch --account="${ACCOUNT}" \
    --parsable \
    --partition=cpu_short \
    --job-name=prep_dfot_panda \
    --time=04:00:00 \
    --cpus-per-task=8 \
    --mem=32GB \
    --output=comparison_methods/slurm_log/prep_dfot_panda_%j.out \
    --error=comparison_methods/slurm_log/prep_dfot_panda_%j.err \
    --wrap="${CONDA_INIT} && cd ${PROJECT_ROOT} && \
            python comparison_methods/data/prepare_ucf101_dfot.py \
                --src-dir ${PANDA_SRC} \
                --dst-dir comparison_methods/data/panda_dfot")
echo "  DFoT data prep: Job ${DFOT_DATA_JOB}"

echo ""

# ==============================================================================
# Phase 2: Evaluation (depends on data prep completing)
# ==============================================================================
echo ">>> Phase 2: Submitting evaluation jobs (with data-prep dependencies)..."

# PVDM baseline
PVDM_EVAL_JOB=$(sbatch --account="${ACCOUNT}" \
    --parsable \
    --dependency=afterok:${PVDM_DATA_JOB} \
    --export=ALL,PVDM_DATA_DIR=${PROJECT_ROOT}/comparison_methods/data/panda_pvdm,PVDM_OUTPUT_DIR=${PROJECT_ROOT}/comparison_methods/results/panda_pvdm_baseline,PVDM_MAX_VIDEOS=${MAX_VIDEOS},GT_FEATURES_CACHE=${GT_CACHE_PVDM},SAVE_ONLY_LIST=${SAVE_LIST} \
    comparison_methods/sbatch/run_pvdm.sbatch)
echo "  PVDM baseline eval: Job ${PVDM_EVAL_JOB}"

# SAVi-DNO (10 DDIM steps)
SAVI_10_JOB=$(sbatch --account="${ACCOUNT}" \
    --parsable \
    --dependency=afterok:${PVDM_DATA_JOB} \
    --export=ALL,DDIM_STEPS=10,SAVI_LR=0.01,SAVI_LAM=0.0012,SAVI_P=0.9,PVDM_DATA_DIR=${PROJECT_ROOT}/comparison_methods/data/panda_pvdm,SAVI_OUTPUT_DIR=${PROJECT_ROOT}/comparison_methods/results/panda_savi_dno_s10,PVDM_MAX_VIDEOS=${MAX_VIDEOS},GT_FEATURES_CACHE=${GT_CACHE_PVDM},SAVE_ONLY_LIST=${SAVE_LIST} \
    comparison_methods/sbatch/run_savi_dno.sbatch)
echo "  SAVi-DNO (10 steps): Job ${SAVI_10_JOB}"

# SAVi-DNO (50 DDIM steps)
SAVI_50_JOB=$(sbatch --account="${ACCOUNT}" \
    --parsable \
    --dependency=afterok:${PVDM_DATA_JOB} \
    --export=ALL,DDIM_STEPS=50,SAVI_LR=0.01,SAVI_LAM=0.0012,SAVI_P=0.9,PVDM_DATA_DIR=${PROJECT_ROOT}/comparison_methods/data/panda_pvdm,SAVI_OUTPUT_DIR=${PROJECT_ROOT}/comparison_methods/results/panda_savi_dno_s50,PVDM_MAX_VIDEOS=${MAX_VIDEOS},GT_FEATURES_CACHE=${GT_CACHE_PVDM},SAVE_ONLY_LIST=${SAVE_LIST} \
    comparison_methods/sbatch/run_savi_dno.sbatch)
echo "  SAVi-DNO (50 steps): Job ${SAVI_50_JOB}"

# DFoT
DFOT_EVAL_JOB=$(sbatch --account="${ACCOUNT}" \
    --parsable \
    --dependency=afterok:${DFOT_DATA_JOB} \
    --export=ALL,DFOT_DATA_DIR=${PROJECT_ROOT}/comparison_methods/data/panda_dfot,DFOT_OUTPUT_DIR=${PROJECT_ROOT}/comparison_methods/results/panda_dfot_k600,DFOT_MAX_VIDEOS=${MAX_VIDEOS},GT_FEATURES_CACHE=${GT_CACHE_DFOT},SAVE_ONLY_LIST=${SAVE_LIST} \
    comparison_methods/sbatch/run_dfot.sbatch)
echo "  DFoT eval: Job ${DFOT_EVAL_JOB}"

echo ""
echo "=============================================================================="
echo "Panda-70M Comparison Methods: Submission Complete"
echo "=============================================================================="
echo "Monitor: squeue -u \$USER"
echo "Results: comparison_methods/results/panda_*/"
echo "=============================================================================="
