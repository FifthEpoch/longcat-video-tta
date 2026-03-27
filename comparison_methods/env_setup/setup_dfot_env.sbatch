#!/bin/bash
#SBATCH --job-name=setup_dfot
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --output=comparison_methods/slurm_log/setup_dfot_%j.out
#SBATCH --error=comparison_methods/slurm_log/setup_dfot_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1
export TMPDIR="/scratch/wc3013/tmp"
export PIP_CACHE_DIR="/scratch/wc3013/pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

echo "=============================================================================="
echo "DFoT Environment Setup"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=============================================================================="

SCRATCH_BASE="/scratch/wc3013"
ENV_PATH="${SCRATCH_BASE}/conda-envs/dfot"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
DFOT_DIR="${PROJECT_ROOT}/comparison_methods/DFoT"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# --- Create conda environment ---
echo "Creating conda environment (Python 3.10)..."
if [ -d "${ENV_PATH}" ]; then
    echo "Environment already exists at ${ENV_PATH}, reusing."
else
    conda create -p "${ENV_PATH}" python=3.10 -y
fi
conda activate "${ENV_PATH}"
export PATH="${ENV_PATH}/bin:${PATH}"
python --version
python -m pip install -U pip setuptools wheel

# --- Clone DFoT ---
echo "Cloning DFoT repository..."
if [ -d "${DFOT_DIR}/.git" ]; then
    echo "DFoT repo already cloned."
    cd "${DFOT_DIR}" && git pull && cd "${PROJECT_ROOT}"
else
    rm -rf "${DFOT_DIR}"
    git clone https://github.com/kwsong0113/diffusion-forcing-transformer.git "${DFOT_DIR}"
fi

# --- Install PyTorch (compatible with their requirements) ---
echo "Installing PyTorch (latest stable for cu124)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# --- Install DFoT requirements ---
echo "Installing DFoT requirements..."
cd "${DFOT_DIR}"
pip install -r requirements.txt
cd "${PROJECT_ROOT}"

# --- Download pretrained K600 checkpoint ---
echo "Downloading DFoT K600 pretrained checkpoint..."
CKPT_DIR="${DFOT_DIR}/pretrained"
mkdir -p "${CKPT_DIR}"

if [ ! -f "${CKPT_DIR}/DFoT_K600.ckpt" ]; then
    python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='kiwhansong/DFoT',
                       filename='pretrained_models/DFoT_K600.ckpt',
                       local_dir='${CKPT_DIR}')
print('Downloaded to:', path)
"
else
    echo "K600 checkpoint already present."
fi

# --- Verification ---
echo ""
echo "=============================================================================="
echo "Verification"
echo "=============================================================================="
python -c "
import sys, torch
print('Python:', sys.version.split()[0])
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
import hydra, omegaconf, einops, lightning, scipy, transformers
print('hydra:', hydra.__version__ if hasattr(hydra, '__version__') else 'OK')
print('lightning:', lightning.__version__)
print('All DFoT deps OK')
"

echo ""
echo "=============================================================================="
echo "DFoT Setup Complete!"
echo "=============================================================================="
echo "Environment: ${ENV_PATH}"
echo "DFoT repo: ${DFOT_DIR}"
echo "End time: $(date)"
echo "=============================================================================="
