#!/bin/bash
#SBATCH --job-name=wan_v2v_lpips_fvd
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_v2v_lpips_fvd_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_v2v_lpips_fvd_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# Fill LPIPS on existing pixel jsons, then I3D FVD on aligned 30 s tails.
# No new generate. L40S. Skip-existing per-video caches.

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
CONDA_ENV="${CONDA_ENV:-${SCRATCH_BASE}/conda-envs/self_forcing}"
SERIES_DIR="${SERIES_DIR:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_128v}"
METHODS="${METHODS:-notta rolling_notta sf_pseudo sf_always_search}"
# Isolated extras. Do NOT pip-resolve torch (16737041 pulled CUDA wheels).
LPIPS_TARGET="${LPIPS_TARGET:-${SCRATCH_BASE}/pip-extras/lpips-nodeps}"

cd "${PROJECT_ROOT}"
mkdir -p wan_experiment/slurm_log "${LPIPS_TARGET}"
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
export TMPDIR="${SCRATCH_BASE}/tmp"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$TMPDIR"

PYTHON="${CONDA_ENV}/bin/python"
echo "install lpips --no-deps → ${LPIPS_TARGET} (conda keeps torch)"
"${PYTHON}" -m pip install --no-cache-dir --no-deps --target "${LPIPS_TARGET}" \
    lpips huggingface_hub
export PYTHONPATH="${LPIPS_TARGET}${PYTHONPATH:+:$PYTHONPATH}"

"${PYTHON}" -c "import lpips, torch; print('lpips ok', torch.__file__, 'cuda', torch.cuda.is_available())"

echo "===== fill LPIPS (keep PSNR/SSIM) ====="
"${PYTHON}" -u wan_experiment/scripts/score_v2v_pixel_metrics.py \
    --series-dir "${SERIES_DIR}" \
    --methods ${METHODS} \
    --device cuda \
    --lpips-step "${LPIPS_STEP:-8}" \
    --fill-lpips \
    --require-lpips

echo "===== aligned-tail FVD ====="
"${PYTHON}" -u wan_experiment/scripts/score_v2v_aligned_fvd.py \
    --series-dir "${SERIES_DIR}" \
    --methods ${METHODS} \
    --device cuda \
    --batch-size "${FVD_BATCH:-4}"

echo "done lpips+fvd series=${SERIES_DIR}"
