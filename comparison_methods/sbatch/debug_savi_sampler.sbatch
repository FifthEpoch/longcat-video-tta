#!/bin/bash
#SBATCH --job-name=savi_dbg
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=01:00:00
#SBATCH --output=comparison_methods/slurm_log/savi_dbg_%j.out
#SBATCH --error=comparison_methods/slurm_log/savi_dbg_%j.err

set -uo pipefail
export PYTHONNOUSERSITE=1
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${SAVI_LC_DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "$CONDA_ENV"
unset PYTHONHOME; unset PYTHONPATH
PYTHON="${CONDA_ENV}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
cd "${PROJECT_ROOT}"
mkdir -p comparison_methods/slurm_log

"$PYTHON" comparison_methods/scripts/debug_savi_sampler.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --data-dir "${DATA_DIR}" \
    --max-videos "${DBG_MAX_VIDEOS:-2}" \
    --num-cond-frames "${DBG_NUM_COND:-14}" \
    --num-frames "${DBG_NUM_FRAMES:-28}" \
    --gen-start-frame "${DBG_GEN_START:-48}" \
    --num-inference-steps "${DBG_STEPS:-10}" \
    --guidance-scale "${DBG_GUIDANCE:-4.0}"
