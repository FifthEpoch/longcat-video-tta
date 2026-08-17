#!/bin/bash
#SBATCH --job-name=wan_vram_probe
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_vram_probe_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_vram_probe_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CONDA_ENV="${SCRATCH_BASE}/conda-envs/self_forcing"

export TMPDIR="${SCRATCH_BASE}/tmp"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TORCH_HOME="${SCRATCH_BASE}/.cache/torch"
mkdir -p "$TMPDIR" "$HF_HOME" "$TORCH_HOME" \
    "${PROJECT_ROOT}/wan_experiment/slurm_log"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${CONDA_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${CONDA_ENV}/bin/python"

cd "${PROJECT_ROOT}"
"${PYTHON}" wan_experiment/scripts/probe_vram.py

echo "==== probe json ===="
cat "${PROJECT_ROOT}/wan_experiment/results/setup_healthcheck/vram_probe.json"
