#!/bin/bash
#SBATCH --job-name=setup_cogvideo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=backbone_experiment/cogvideo/slurm_log/setup_cogvideo_%j.out
#SBATCH --error=backbone_experiment/cogvideo/slurm_log/setup_cogvideo_%j.err

# ============================================================================
# CogVideoX-5B-I2V Environment Setup
#
# Installs dependencies into the existing 'cogvideo' conda env and downloads
# model weights. Must run on a GPU node so CUDA is detected correctly.
#
# Prerequisites: conda create -n cogvideo python=3.10 -y
# ============================================================================

set -euo pipefail

SCRATCH_BASE="/scratch/wc3013"
MODEL_DIR="${SCRATCH_BASE}/cogvideo-checkpoints/CogVideoX-5b-I2V"

echo "=============================================================================="
echo "CogVideoX-5B-I2V Environment Setup"
echo "=============================================================================="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "Start     : $(date)"
echo "=============================================================================="

# ============================================================================
# Activate conda
# ============================================================================
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

conda activate cogvideo
echo "Python: $(which python)"
echo "Python version: $(python --version)"

# ============================================================================
# Step 1: Install PyTorch with CUDA support
# ============================================================================
echo ""
echo ">>> Step 1/4: Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ============================================================================
# Step 2: Install diffusers, transformers, peft
# ============================================================================
echo ""
echo ">>> Step 2/4: Installing diffusers + transformers + peft..."
pip install "diffusers[torch]" transformers accelerate
pip install peft

# ============================================================================
# Step 3: Install evaluation dependencies
# ============================================================================
echo ""
echo ">>> Step 3/4: Installing evaluation deps..."
pip install lpips av torchmetrics scikit-image numpy

# ============================================================================
# Step 4: Download CogVideoX-5B-I2V model weights
# ============================================================================
echo ""
echo ">>> Step 4/4: Downloading CogVideoX-5B-I2V weights to ${MODEL_DIR}..."

if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/model_index.json" ]; then
    echo "Model already exists at ${MODEL_DIR}, skipping download."
else
    mkdir -p "${MODEL_DIR}"
    python -c "
import torch
from diffusers import CogVideoXImageToVideoPipeline

print('Downloading CogVideoX-5B-I2V from HuggingFace...')
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    'THUDM/CogVideoX-5b-I2V',
    torch_dtype=torch.bfloat16,
)
print('Saving to local path...')
pipe.save_pretrained('${MODEL_DIR}')
print('Done.')
"
fi

# ============================================================================
# Verification
# ============================================================================
echo ""
echo ">>> Verification..."
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()} ({torch.cuda.get_device_name(0)})')

import diffusers
print(f'Diffusers {diffusers.__version__}')

import transformers
print(f'Transformers {transformers.__version__}')

import peft
print(f'PEFT {peft.__version__}')

import lpips
print('LPIPS: OK')

import av
print('PyAV: OK')

# Quick model load test
from diffusers import CogVideoXImageToVideoPipeline
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    '${MODEL_DIR}',
    torch_dtype=torch.bfloat16,
)
print(f'Transformer params: {sum(p.numel() for p in pipe.transformer.parameters()):,}')
print(f'VAE params: {sum(p.numel() for p in pipe.vae.parameters()):,}')
print('All checks passed!')
"

echo ""
echo "=============================================================================="
echo "Setup complete!"
echo "=============================================================================="
echo "End time: $(date)"
echo "Model path: ${MODEL_DIR}"
echo ""
echo "To run TTA experiments:"
echo "  METHOD=delta_a NUM_VIDEOS=10 sbatch backbone_experiment/cogvideo/run_cogvideo_tta.sbatch"
echo "=============================================================================="
