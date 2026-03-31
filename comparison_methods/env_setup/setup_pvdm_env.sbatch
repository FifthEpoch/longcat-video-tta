#!/bin/bash
#SBATCH --job-name=setup_pvdm
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=comparison_methods/slurm_log/setup_pvdm_%j.out
#SBATCH --error=comparison_methods/slurm_log/setup_pvdm_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
export TMPDIR="/scratch/wc3013/tmp"
export PIP_CACHE_DIR="/scratch/wc3013/pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

echo "=============================================================================="
echo "PVDM Environment Setup"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================================================="

SCRATCH_BASE="/scratch/wc3013"
ENV_PATH="${SCRATCH_BASE}/conda-envs/pvdm"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
PVDM_DIR="${PROJECT_ROOT}/comparison_methods/PVDM"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# --- Create conda environment ---
echo "Creating conda environment (Python 3.10)..."
if [ -d "${ENV_PATH}" ]; then
    echo "Removing old env (PyTorch 1.12 incompatible with H200)..."
    conda env remove -p "${ENV_PATH}" -y 2>/dev/null || rm -rf "${ENV_PATH}"
fi
conda create -p "${ENV_PATH}" python=3.10 -y
conda activate "${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"
python --version
python -m pip install -U pip setuptools wheel

# --- PyTorch 2.6 + CUDA 12.4 (H200/Hopper-compatible) ---
echo "Installing PyTorch 2.6.0+cu124..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# --- PVDM dependencies ---
echo "Installing PVDM dependencies..."
pip install numpy natsort tqdm gdown omegaconf einops lpips \
    tensorboard imageio av moviepy pillow scipy

# --- Clone PVDM ---
echo "Cloning PVDM repository..."
if [ -d "${PVDM_DIR}/.git" ]; then
    echo "PVDM repo already cloned."
    cd "${PVDM_DIR}" && git pull && cd "${PROJECT_ROOT}"
else
    rm -rf "${PVDM_DIR}"
    git clone https://github.com/sihyun-yu/PVDM.git "${PVDM_DIR}"
fi

# --- Download pretrained UCF-101 weights ---
echo "Downloading pretrained UCF-101 weights..."
PRETRAINED_DIR="${PVDM_DIR}/pretrained"
mkdir -p "${PRETRAINED_DIR}"

AE_PATH="${PRETRAINED_DIR}/pvdm_ucf101_autoencoder.pth"
DIFF_PATH="${PRETRAINED_DIR}/pvdm_ucf101_diffusion.pth"

if [ ! -f "${AE_PATH}" ]; then
    python -c "import gdown; gdown.download(id='1L8l8h_SKN1BPm1P9cUhUmTsVZN0SCmqe', output='${AE_PATH}', quiet=False)"
else
    echo "Autoencoder weights already present."
fi

if [ ! -f "${DIFF_PATH}" ]; then
    python -c "import gdown; gdown.download(id='11EDh97-lq07xgvAx3P5KURQfrPtrcX0G', output='${DIFF_PATH}', quiet=False)"
else
    echo "Diffusion weights already present."
fi

# --- Eval dependencies ---
echo "Installing evaluation dependencies..."
pip install scikit-image torchmetrics

# --- Verification ---
echo ""
echo "=============================================================================="
echo "Verification"
echo "=============================================================================="
python -c "import sys, torch, numpy, einops, omegaconf, imageio; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('All PVDM deps OK')"

echo ""
echo "=============================================================================="
echo "PVDM Setup Complete!"
echo "=============================================================================="
echo "Environment: ${ENV_PATH}"
echo "PVDM repo: ${PVDM_DIR}"
echo "End time: $(date)"
echo "=============================================================================="
