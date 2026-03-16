# Research Partner Cluster Setup Guide

This guide walks through everything needed to set up the LongCat-Video TTA experiment environment on the NYU Greene HPC cluster from scratch, on a **separate login node** with your own scratch space.

**SLURM Account:** `torch_pr_36_mren`
**Cluster:** NYU Greene HPC
**GPU Partition:** `h200_coursework` (H200 GPUs, 141 GB VRAM each)

---

## Overview

You will need to:
1. Clone the experiment repo + the LongCat-Video dependency
2. Create a conda environment and install all dependencies
3. Download the LongCat-Video model checkpoints (~50 GB)
4. Prepare the evaluation dataset (Panda-70M 100-video subset)
5. (Optional) Prepare the 1000-video retrieval pool for batch-size experiments
6. Update path constants to use YOUR scratch directory
7. Submit experiment jobs

**Estimated setup time:** 4-6 hours (mostly waiting for SLURM jobs)

---

## Step 0: Understand the Directory Layout

After setup, your scratch space should look like this:

```
/scratch/<YOUR_NETID>/
├── conda-envs/
│   └── longcat/                  # Conda environment (Python 3.10)
├── longcat-video-checkpoints/    # Model weights (~50 GB)
├── longcat-video-tta/            # Project root
│   ├── LongCat-Video/            # Cloned dependency (git repo)
│   ├── datasets/
│   │   ├── panda_100_480p/       # 100-video eval set
│   │   └── panda_1000_480p/      # 1000-video retrieval pool (optional)
│   ├── delta_experiment/
│   │   └── scripts/              # TTA method implementations
│   ├── lora_experiment/
│   │   └── scripts/              # LoRA / Full-model TTA implementations
│   ├── sweep_experiment/
│   │   ├── configs/              # Experiment YAML configs
│   │   ├── sbatch/               # SLURM job templates
│   │   ├── scripts/              # Sweep runner, progress checker
│   │   ├── results/              # Output (one dir per series/run)
│   │   └── slurm_log/            # SLURM stdout/stderr logs
│   └── env_setup/
│       └── 01_setup_longcat_env.sbatch
├── tmp/                          # Temp dir (for pip/builds)
└── pip-cache/                    # Pip cache dir
```

---

## Step 1: Clone the Repositories

```bash
# Set your scratch base
export SCRATCH="/scratch/<YOUR_NETID>"
export PROJECT_ROOT="${SCRATCH}/longcat-video-tta"

# Create directories
mkdir -p "${SCRATCH}/conda-envs"
mkdir -p "${SCRATCH}/tmp"
mkdir -p "${SCRATCH}/pip-cache"
mkdir -p "${PROJECT_ROOT}"

# Clone the experiment repo
cd "${PROJECT_ROOT}"
git clone https://github.com/<REPO_OWNER>/longcat-video-tta.git .
# Or if it's a private repo, use SSH:
# git clone git@github.com:<REPO_OWNER>/longcat-video-tta.git .

# Clone the LongCat-Video dependency
git clone https://github.com/meituan-longcat/LongCat-Video.git
```

---

## Step 2: Update Path Constants

The sbatch scripts and Python code reference `/scratch/wc3013/` in several places. You need to update these to your own scratch path.

### Files that need path updates:

**1. `sweep_experiment/sbatch/run_sweep.sbatch`** (most critical):
```bash
# Change these lines near the top:
SCRATCH_BASE="/scratch/<YOUR_NETID>"          # was /scratch/wc3013
```

**2. `env_setup/01_setup_longcat_env.sbatch`**:
```bash
SCRATCH_BASE="/scratch/<YOUR_NETID>"          # was /scratch/wc3013
```

**3. `baseline_experiment/sbatch/download_model.sbatch`**:
```bash
SCRATCH="/scratch/<YOUR_NETID>"               # was /scratch/wc3013
```

**4. `datasets/download_panda70m.sbatch`**:
```bash
SCRATCH_BASE="/scratch/<YOUR_NETID>"          # was /scratch/wc3013
```

**5. `datasets/resize_videos.sbatch`**:
```bash
# Update any hardcoded /scratch/wc3013 references
```

**6. `sweep_experiment/scripts/check_all_progress.sh`**:
```bash
PROJECT_ROOT="/scratch/<YOUR_NETID>/longcat-video-tta"
```

**Tip:** Do a global search-and-replace:
```bash
cd "${PROJECT_ROOT}"
grep -r "/scratch/wc3013" --include="*.sbatch" --include="*.sh" --include="*.yaml" -l
# Then update each file
```

---

## Step 3: Create Conda Environment

```bash
# Load conda
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Create the environment
conda create -p "${SCRATCH}/conda-envs/longcat" python=3.10 -y
```

---

## Step 4: Install Dependencies (Requires GPU Node)

The flash-attention library must be compiled on a GPU node. Submit the setup script:

```bash
cd "${PROJECT_ROOT}"
mkdir -p env_setup/slurm_log

sbatch --account=torch_pr_36_mren env_setup/01_setup_longcat_env.sbatch
```

**This takes ~1-2 hours.** Monitor with:
```bash
squeue -u $USER
# When done, check the log:
cat env_setup/slurm_log/setup_longcat_env_*.out
```

The script installs:
- PyTorch 2.6.0 (CUDA 12.4)
- flash-attn 2.7.4.post1 (compiled from source)
- transformers 4.41.0, diffusers 0.35.1
- LongCat-Video requirements (einops, imageio, av, etc.)
- TTA experiment deps (lpips, scikit-image, scikit-learn, pandas, torchmetrics)

---

## Step 5: Download Model Checkpoints

The LongCat-Video model is ~50 GB. Download it via HuggingFace:

```bash
mkdir -p baseline_experiment/sbatch/slurm_log

sbatch --account=torch_pr_36_mren baseline_experiment/sbatch/download_model.sbatch
```

**This takes ~1-2 hours.** The model will be saved to `${SCRATCH}/longcat-video-checkpoints/`.

Verify the download:
```bash
ls -lh ${SCRATCH}/longcat-video-checkpoints/
# Should contain: config.json, model files, tokenizer, vae, etc.
```

---

## Step 6: Prepare the Evaluation Dataset

### Option A: Copy from original researcher (fastest)

If you have access to the same filesystem, just copy:
```bash
cp -r /scratch/wc3013/longcat-video-tta/datasets/panda_100_480p \
      ${PROJECT_ROOT}/datasets/panda_100_480p
```

### Option B: Download from scratch

This downloads 100 videos from Panda-70M via YouTube and resizes them to 480p.

**Important:** YouTube may block downloads from data center IPs. You may need to provide browser cookies:

1. Install browser extension "Get cookies.txt LOCALLY" in Chrome
2. Go to youtube.com, sign in
3. Export cookies, save as `cookies.txt`
4. Upload to cluster: `scp cookies.txt <NETID>@greene.hpc.nyu.edu:${PROJECT_ROOT}/datasets/cookies.txt`

Then submit:
```bash
mkdir -p datasets/slurm_log

# Download 100 videos
sbatch --account=torch_pr_36_mren datasets/download_panda70m.sbatch

# After download completes, resize to 480p
sbatch --account=torch_pr_36_mren datasets/resize_videos.sbatch
```

**This takes ~4-8 hours total.**

### Option C: Prepare the 1000-video retrieval pool (for batch-size ablation only)

```bash
# Download 1000 videos
NUM_VIDEOS=1000 sbatch --account=torch_pr_36_mren datasets/download_panda70m.sbatch

# After download, resize
SRC_DIR=${PROJECT_ROOT}/datasets/panda_1000 \
DST_DIR=${PROJECT_ROOT}/datasets/panda_1000_480p \
sbatch --account=torch_pr_36_mren datasets/resize_videos.sbatch
```

---

## Step 7: Verify Everything Works

Quick sanity check on a login node (no GPU needed for this):

```bash
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate ${SCRATCH}/conda-envs/longcat

cd ${PROJECT_ROOT}

python -c "
import torch
import flash_attn
import sys
sys.path.insert(0, 'LongCat-Video')
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
print('All imports OK')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'flash-attn: {flash_attn.__version__}')
"
```

---

## Step 8: Run Experiments

### How the sweep system works

Each experiment is defined by a **YAML config** in `sweep_experiment/configs/`. The config specifies:
- The TTA method (full, lora, delta_a, delta_b, delta_c, norm_tune, film)
- Fixed hyperparameters
- A list of sweep rows (each row = one SLURM job)

The runner script reads the config and submits one sbatch job per sweep row:

```bash
# Dry-run (print commands without submitting):
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/<CONFIG>.yaml \
    --account torch_pr_36_mren \
    --dry-run

# Actually submit:
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/<CONFIG>.yaml \
    --account torch_pr_36_mren
```

### Checking progress

```bash
bash sweep_experiment/scripts/check_all_progress.sh
```

### Checking a specific series

```bash
# See if summary.json exists (= complete):
ls sweep_experiment/results/<SERIES_NAME>/*/summary.json

# Check SLURM job status:
squeue -u $USER
```

---

## Experiments That Need to Be Run

See `EXPERIMENT_STATUS.md` for the full status. The key experiments awaiting execution:

### Priority 1: Queued experiments (already submitted, just need GPU time)

These are already in the SLURM queue under `wc3013`. If you are running them under YOUR account, re-submit them:

| Config | Description | Jobs |
|--------|-------------|------|
| `series_adasteer_extended_data.yaml` | 5s/8s input windows | 10 |
| `series_adasteer_param_sweep.yaml` | Parameter dim sweep | 7 |
| `series_adasteer_long_train_es.yaml` | Long-train with ES | 5 |
| `series_full_long_train_100v.yaml` | Full-model long-train | 2 |
| `series_lora_ultra_long_train.yaml` | LoRA long-train | 3 |

### Priority 2: Batch-size ablation (not yet submitted)

**Prerequisite:** 1000-video Panda-70M dataset at `datasets/panda_1000_480p/`

All three methods now have retrieval-augmented batch TTA fully implemented. Submit all once the 1000-video pool is ready:

```bash
python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/exp5_batch_size_delta_a.yaml \
    --account torch_pr_36_mren

python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/exp5_batch_size_lora.yaml \
    --account torch_pr_36_mren

python sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/exp5_batch_size_full.yaml \
    --account torch_pr_36_mren
```

### Priority 3: Open-Sora 2.0 backbone experiments

These require a **separate conda environment** and the Open-Sora 2.0 codebase. See `backbone_experiment/opensora/` for setup instructions (if they exist).

---

## Troubleshooting

### "QOSGrpMemLimit" in squeue

This means the total memory across your running + pending jobs exceeds the QOS limit. Jobs will start as others finish. Nothing to fix — just wait.

### SLURM job fails immediately

Check the error log:
```bash
cat sweep_experiment/slurm_log/tta_<SERIES>_<RUNID>_<JOBID>.err
```

Common issues:
- **"Conda environment not found"**: Path mismatch. Check `SCRATCH_BASE` in the sbatch.
- **"No video files found"**: Dataset not prepared or wrong `DATA_DIR`.
- **"CUDA out of memory"**: Some methods (full-model) need 192-256 GB system RAM. Check `--mem` in sbatch.

### Jobs stuck on "Priority"

The `cs` partition jobs (dataset prep) show `Priority` — they'll start when resources are available. H200 jobs show `QOSGrpMemLimit` — same thing, just waiting for your quota.

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `sweep_experiment/scripts/run_sweep.py` | Submit sweep jobs from YAML config |
| `sweep_experiment/sbatch/run_sweep.sbatch` | Unified SLURM job template |
| `sweep_experiment/scripts/check_all_progress.sh` | Check results of all experiments |
| `sweep_experiment/scripts/extract_summary.py` | Extract metrics from `summary.json` |
| `delta_experiment/scripts/common.py` | Shared utilities (model loading, metrics, etc.) |
| `delta_experiment/scripts/run_delta_a.py` | AdaSteer-1 TTA runner |
| `delta_experiment/scripts/run_delta_b.py` | AdaSteer G>1 TTA runner |
| `delta_experiment/scripts/run_delta_c.py` | Output residual TTA runner |
| `lora_experiment/scripts/run_lora_tta.py` | LoRA TTA runner |
| `lora_experiment/scripts/run_full_tta.py` | Full-model TTA runner |
| `env_setup/01_setup_longcat_env.sbatch` | Environment setup script |
| `baseline_experiment/sbatch/download_model.sbatch` | Model download script |
| `EXPERIMENT_STATUS.md` | Full experiment status tracking |
