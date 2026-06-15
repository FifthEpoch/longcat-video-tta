#!/bin/bash
#SBATCH --job-name=gt_cache_panda
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=sweep_experiment/slurm_log/precompute_gt_panda_%j.out
#SBATCH --error=sweep_experiment/slurm_log/precompute_gt_panda_%j.err

# Precompute GT I3D/FID feature cache for Panda 1000v standard sweeps.
#
# Output matches headline panda_1000v_standard geometry:
#   gen_start_frame=48, num_gen_frames=14 (GT continuation window)
#
# Submit:
#   cd /scratch/wc3013/longcat-video-tta
#   sbatch --account=torch_pr_36_mren \
#     sweep_experiment/sbatch/precompute_panda_gt_cache.sbatch
#
# Override paths:
#   DATA_DIR=datasets/panda_1000_480p \
#   GT_CACHE=gt_caches/panda_1000_longcat.npz \
#   sbatch --account=torch_pr_36_mren \
#     sweep_experiment/sbatch/precompute_panda_gt_cache.sbatch

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
GT_CACHE="${GT_CACHE:-${PROJECT_ROOT}/gt_caches/panda_1000_longcat.npz}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
NUM_GEN_FRAMES="${NUM_GEN_FRAMES:-14}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"

cd "$PROJECT_ROOT"
mkdir -p sweep_experiment/slurm_log "$(dirname "$GT_CACHE")"

if [[ -f "$GT_CACHE" && "${FORCE_RECOMPUTE:-0}" != "1" ]]; then
  echo "GT cache already exists: $GT_CACHE"
  ls -lh "$GT_CACHE"
  exit 0
fi

echo "Precomputing GT cache"
echo "  data_dir         : $DATA_DIR"
echo "  output           : $GT_CACHE"
echo "  gen_start_frame  : $GEN_START_FRAME"
echo "  num_gen_frames   : $NUM_GEN_FRAMES"
echo "  num_cond_frames  : $NUM_COND_FRAMES"

python sweep_experiment/scripts/precompute_gt_features.py \
  --data-dir "$DATA_DIR" \
  --output "$GT_CACHE" \
  --mode longcat \
  --gen-start-frame "$GEN_START_FRAME" \
  --num-gen-frames "$NUM_GEN_FRAMES" \
  --num-cond-frames "$NUM_COND_FRAMES" \
  --device cuda

echo ""
echo "Done. GT cache: $GT_CACHE"
ls -lh "$GT_CACHE"
