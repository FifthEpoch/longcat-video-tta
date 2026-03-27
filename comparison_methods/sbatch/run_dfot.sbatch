#!/bin/bash
#SBATCH --job-name=dfot_eval
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=comparison_methods/slurm_log/dfot_eval_%j.out
#SBATCH --error=comparison_methods/slurm_log/dfot_eval_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

echo "=============================================================================="
echo "DFoT Evaluation (K600 pretrained on UCF-101)"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================================================="

SCRATCH_BASE="/scratch/wc3013"
ENV_PATH="${SCRATCH_BASE}/conda-envs/dfot"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
DFOT_DIR="${PROJECT_ROOT}/comparison_methods/DFoT"
DATA_DIR="${PROJECT_ROOT}/comparison_methods/data/ucf101_dfot"
MAPPING="${DATA_DIR}/video_mapping.csv"
CKPT="${DFOT_DIR}/pretrained/pretrained_models/DFoT_K600.ckpt"
OUTPUT_DIR="${PROJECT_ROOT}/comparison_methods/results/dfot_k600"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"
unset PYTHONHOME
unset PYTHONPATH

cd "${PROJECT_ROOT}"

python comparison_methods/scripts/run_dfot_eval.py \
    --dfot-dir "${DFOT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --checkpoint "${CKPT}" \
    --mapping-csv "${MAPPING}" \
    --context-length 5 \
    --pred-length 12 \
    --max-videos 500 \
    --seed 42

echo ""
echo "=============================================================================="
echo "DFoT Evaluation Complete"
echo "End time: $(date)"
echo "=============================================================================="
