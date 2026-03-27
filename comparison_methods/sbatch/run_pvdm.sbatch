#!/bin/bash
#SBATCH --job-name=pvdm_eval
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=comparison_methods/slurm_log/pvdm_eval_%j.out
#SBATCH --error=comparison_methods/slurm_log/pvdm_eval_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "=============================================================================="
echo "PVDM Baseline Evaluation"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================================================="

SCRATCH_BASE="/scratch/wc3013"
ENV_PATH="${SCRATCH_BASE}/conda-envs/pvdm"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
PVDM_DIR="${PROJECT_ROOT}/comparison_methods/PVDM"
DATA_DIR="${PROJECT_ROOT}/comparison_methods/data/ucf101_pvdm"
MAPPING="${DATA_DIR}/video_mapping.csv"
OUTPUT_DIR="${PROJECT_ROOT}/comparison_methods/results/pvdm_baseline"

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
    --ddim-steps 10 \
    --no-optimize \
    --seed 42

echo ""
echo "=============================================================================="
echo "PVDM Baseline Complete"
echo "End time: $(date)"
echo "=============================================================================="
