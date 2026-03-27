#!/bin/bash
#SBATCH --job-name=savi_dno
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --output=comparison_methods/slurm_log/savi_dno_%j.out
#SBATCH --error=comparison_methods/slurm_log/savi_dno_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "=============================================================================="
echo "SAVi-DNO Evaluation (Noise Optimization on PVDM)"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "DDIM steps: ${DDIM_STEPS:-10}"
echo "LR: ${SAVI_LR:-0.01}"
echo "Lambda: ${SAVI_LAM:-0.0012}"
echo "p: ${SAVI_P:-0.9}"
echo "=============================================================================="

SCRATCH_BASE="/scratch/wc3013"
ENV_PATH="${SCRATCH_BASE}/conda-envs/pvdm"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
PVDM_DIR="${PROJECT_ROOT}/comparison_methods/PVDM"
DATA_DIR="${PROJECT_ROOT}/comparison_methods/data/ucf101_pvdm"
MAPPING="${DATA_DIR}/video_mapping.csv"
OUTPUT_DIR="${PROJECT_ROOT}/comparison_methods/results/savi_dno_s${DDIM_STEPS:-10}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"
unset PYTHONHOME
unset PYTHONPATH

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PVDM_DIR}:${PYTHONPATH:-}"

python comparison_methods/scripts/savi_dno.py \
    --pvdm-dir "${PVDM_DIR}" \
    --pretrained-dir "${PVDM_DIR}/pretrained" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --mapping-csv "${MAPPING}" \
    --max-videos 500 \
    --ddim-steps "${DDIM_STEPS:-10}" \
    --lr "${SAVI_LR:-0.01}" \
    --lam "${SAVI_LAM:-0.0012}" \
    --p "${SAVI_P:-0.9}" \
    --seed 42

echo ""
echo "=============================================================================="
echo "SAVi-DNO Complete"
echo "End time: $(date)"
echo "=============================================================================="
