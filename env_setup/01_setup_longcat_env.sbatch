#!/bin/bash
#SBATCH --job-name=setup_longcat_env
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=03:00:00
#SBATCH --output=env_setup/slurm_log/setup_longcat_env_%j.out
#SBATCH --error=env_setup/slurm_log/setup_longcat_env_%j.err

# ==============================================================================
# LongCat-Video Environment Setup
# ==============================================================================
# Installs PyTorch 2.6.0 + flash-attn 2.7.4 + LongCat-Video dependencies
# into the "longcat" conda environment (already created with Python 3.10).
#
# Key differences from opensora20 env:
#   - PyTorch 2.6.0 (vs 2.4.0)
#   - flash-attn 2.7.4.post1 (built for torch 2.6)
#   - transformers 4.41.0, diffusers 0.35.1
#   - NO colossalai, mmengine, xformers, liger-kernel
#
# Requires H200 GPU node to validate installation on target hardware.
# ==============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

# ---------------------------------------------------------------------------
# Fix cross-device link errors: pip cache and TMPDIR must be on the same
# filesystem as the conda env (/scratch), NOT on /home.
# ---------------------------------------------------------------------------
export TMPDIR="/scratch/wc3013/tmp"
export PIP_CACHE_DIR="/scratch/wc3013/pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

echo "=============================================================================="
echo "LongCat-Video Environment Setup"
echo "=============================================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Start time: $(date)"
echo "=============================================================================="
echo ""

# ==============================================================================
# Configuration
# ==============================================================================
SCRATCH_BASE="/scratch/wc3013"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
LONGCAT_REPO="${PROJECT_ROOT}/LongCat-Video"

# ==============================================================================
# Load Conda & Activate Environment
# ==============================================================================
echo "Loading conda module..."
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

if [ ! -d "${ENV_PATH}" ]; then
    echo "ERROR: Conda environment not found at ${ENV_PATH}"
    echo "Please run: conda create -n longcat python=3.10 -y"
    exit 1
fi

conda activate "${ENV_PATH}"

# Verify activation
if [ "$CONDA_PREFIX" != "${ENV_PATH}" ]; then
    echo "ERROR: Failed to activate environment!"
    echo "CONDA_PREFIX: $CONDA_PREFIX"
    echo "Expected: ${ENV_PATH}"
    exit 1
fi

export PATH="${ENV_PATH}/bin:${PATH}"
echo "Environment activated: ${CONDA_PREFIX}"
echo "Python: $(python --version)"
echo "Python executable: $(which python)"
echo ""

# Upgrade pip
python -m pip install -U pip setuptools wheel

# ==============================================================================
# Step 1: Install PyTorch 2.6.0 with CUDA 12.4
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 1: PyTorch 2.6.0 + CUDA 12.4"
echo "=============================================================================="

pip install numpy==1.26.4

echo "Installing PyTorch 2.6.0..."
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 \
    torchvision

echo ""
echo "Verifying PyTorch..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
assert '2.6.0' in torch.__version__, f'Expected torch 2.6.0, got {torch.__version__}'
print('PyTorch OK!')
"

# ==============================================================================
# Step 2: Install flash-attn (requires GPU compilation)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 2: Flash Attention 2.7.4.post1"
echo "=============================================================================="
echo "This will compile from source — expect 15-20 minutes..."
echo ""

# Build dependencies
pip install psutil packaging ninja

# flash-attn needs to be compiled against the installed torch+CUDA
# --no-build-isolation ensures it uses the installed torch
if pip install flash-attn==2.7.4.post1 --no-build-isolation; then
    echo ""
    python -c "
import flash_attn
print(f'flash-attn: {flash_attn.__version__}')
print('flash-attn installed successfully!')
"
else
    echo ""
    echo "WARNING: flash-attn 2.7.4.post1 failed. Trying without version pin..."
    if pip install flash-attn --no-build-isolation; then
        python -c "
import flash_attn
print(f'flash-attn: {flash_attn.__version__}')
print('flash-attn installed (version may differ from pinned).')
"
    else
        echo "ERROR: flash-attn installation failed completely."
        echo "LongCat-Video requires flash-attn. Please debug manually."
        echo "Common fixes:"
        echo "  - Ensure CUDA toolkit headers are available"
        echo "  - Try: MAX_JOBS=4 pip install flash-attn --no-build-isolation"
        exit 1
    fi
fi

# ==============================================================================
# Step 3: Install LongCat-Video requirements
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 3: LongCat-Video Requirements"
echo "=============================================================================="

if [ ! -f "${LONGCAT_REPO}/requirements.txt" ]; then
    echo "ERROR: LongCat-Video not found at ${LONGCAT_REPO}"
    echo "Please clone it: git clone https://github.com/meituan-longcat/LongCat-Video.git"
    exit 1
fi

echo "Installing from ${LONGCAT_REPO}/requirements.txt..."
echo "(Skipping torch and flash-attn since already installed)"

# Install remaining requirements, skipping torch/flash-attn (already handled)
pip install \
    transformers==4.41.0 \
    diffusers==0.35.1 \
    einops==0.8.0 \
    ftfy==6.2.0 \
    loguru==0.7.2 \
    psutil==6.0.0 \
    opencv-python==4.9.0.80 \
    pyarrow==20.0.0 \
    imageio==2.37.0 \
    imageio-ffmpeg==0.6.0 \
    safetensors \
    accelerate \
    sentencepiece \
    protobuf

echo "LongCat-Video requirements installed."

# ==============================================================================
# Step 4: Install PyAV via conda (better ffmpeg handling)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 4: PyAV Installation"
echo "=============================================================================="

# LongCat requires av==12.0.0
conda install -y av=12.0.0 -c conda-forge || pip install av==12.0.0

python -c "
import av
print(f'PyAV: {av.__version__}')
"

# ==============================================================================
# Step 5: Install TTA experiment dependencies
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 5: TTA Experiment Dependencies"
echo "=============================================================================="

pip install \
    lpips \
    scikit-image \
    scikit-learn \
    pandas \
    tqdm \
    matplotlib \
    torchmetrics

echo "TTA experiment dependencies installed."

# ==============================================================================
# Step 6: Streamlit (optional, for interactive demos)
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 6: Streamlit (optional)"
echo "=============================================================================="

pip install streamlit==1.50.0 || echo "WARNING: streamlit install failed (optional)"

# ==============================================================================
# Step 7: Final Verification
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Step 7: Final Verification"
echo "=============================================================================="

python -c "
import sys
print('=' * 70)
print('LONGCAT-VIDEO ENVIRONMENT VERIFICATION')
print('=' * 70)
print()
print(f'Python: {sys.version}')
print(f'Location: {sys.executable}')
print()

# Core ML
import numpy as np
import torch
print('Core ML Stack:')
print(f'  NumPy:      {np.__version__}')
print(f'  PyTorch:    {torch.__version__}')
print(f'  CUDA:       {torch.version.cuda if torch.cuda.is_available() else \"NOT AVAILABLE\"}')
if torch.cuda.is_available():
    print(f'  GPU:        {torch.cuda.get_device_name(0)}')
    print(f'  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print()

# Flash attention
import flash_attn
print(f'  flash-attn: {flash_attn.__version__}')

# LongCat-Video dependencies
import transformers
import diffusers
import einops
import cv2
import av
import imageio
print()
print('LongCat-Video Dependencies:')
print(f'  transformers: {transformers.__version__}')
print(f'  diffusers:    {diffusers.__version__}')
print(f'  einops:       {einops.__version__}')
print(f'  opencv:       {cv2.__version__}')
print(f'  PyAV:         {av.__version__}')
print(f'  imageio:      {imageio.__version__}')

# Evaluation deps
import lpips
import pandas
print()
print('Evaluation Dependencies:')
print(f'  lpips:  installed')
print(f'  pandas: {pandas.__version__}')

# Test LongCat-Video imports
sys.path.insert(0, '${LONGCAT_REPO}')
print()
print('LongCat-Video Module Imports:')
try:
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    print('  LongCatVideoTransformer3DModel: OK')
except Exception as e:
    print(f'  LongCatVideoTransformer3DModel: FAILED ({e})')

try:
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
    print('  LongCatVideoPipeline: OK')
except Exception as e:
    print(f'  LongCatVideoPipeline: FAILED ({e})')

try:
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    print('  AutoencoderKLWan: OK')
except Exception as e:
    print(f'  AutoencoderKLWan: FAILED ({e})')

try:
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
    print('  FlowMatchEulerDiscreteScheduler: OK')
except Exception as e:
    print(f'  FlowMatchEulerDiscreteScheduler: FAILED ({e})')

print()

# Version assertions
errors = []
if '2.6.0' not in torch.__version__:
    errors.append(f'PyTorch must be 2.6.0, got {torch.__version__}')
if transformers.__version__ != '4.41.0':
    errors.append(f'transformers must be 4.41.0, got {transformers.__version__}')

if errors:
    print('ERRORS:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    print('All version checks passed!')

print()
print('=' * 70)
print('ENVIRONMENT SETUP COMPLETE AND VERIFIED!')
print('=' * 70)
"

VERIFICATION_EXIT=$?

if [ $VERIFICATION_EXIT -ne 0 ]; then
    echo ""
    echo "Verification failed! Check errors above."
    exit 1
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================================================="
echo "Setup Complete!"
echo "=============================================================================="
echo "Environment: ${ENV_PATH}"
echo "Project root: ${PROJECT_ROOT}"
echo ""
echo "To use in future jobs:"
echo "  module load anaconda3/2025.06"
echo "  source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh"
echo "  conda activate ${ENV_PATH}"
echo ""
echo "End time: $(date)"
echo "=============================================================================="
