# SBATCH + Conda Environment Best Practices (NYU Greene HPC)

**Purpose:** Reference document for AI agents and developers writing or modifying SLURM sbatch scripts that use a conda environment on NYU Greene (or similar HPC clusters). Follow these rules exactly to avoid `ModuleNotFoundError` for `torch`, `sentence_transformers`, and other packages.

---

## The Problem

On HPC clusters, `module load anaconda3/2025.06` adds the **system anaconda** to your shell. This sets environment variables (`PYTHONHOME`, `PYTHONPATH`, etc.) that tell Python where to find packages. When you then `conda activate` your **own** conda env (e.g. `/scratch/wc3013/conda-envs/longcat`), the activation updates `PATH` and `CONDA_PREFIX` but **does NOT unset** `PYTHONHOME` or `PYTHONPATH`.

This causes a critical mismatch:
- `PYTHONHOME` still points to the system anaconda (e.g. Python 3.12)
- Your conda env uses a different Python (e.g. Python 3.10)
- Python resolves `sys.path` using `PYTHONHOME`, so it looks for packages in the **wrong** site-packages directory
- Result: `ModuleNotFoundError: No module named 'torch'` (or any other package)

This failure is **intermittent** because:
- It depends on the module version installed on each compute node
- The system anaconda gets upgraded periodically (changing its Python version)
- Some nodes may cache older module versions

---

## The Correct Pattern

Every sbatch script that uses the longcat conda environment **MUST** follow this exact sequence:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH ... (other SBATCH directives)

set -euo pipefail
export PYTHONNOUSERSITE=1

# ============================================================================
# Path Setup
# ============================================================================
SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"

# ============================================================================
# Environment Setup
# ============================================================================

# Step 1: Load system anaconda (provides the `conda` command)
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Step 2: Activate your conda environment
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
if [ -d "$CONDA_ENV" ]; then
    conda activate "$CONDA_ENV"
    echo "Activated conda: $CONDA_ENV"
else
    echo "ERROR: Conda environment not found at $CONDA_ENV" >&2
    exit 1
fi

# Step 3: CRITICAL — Unset PYTHONHOME and PYTHONPATH
# Without this, Python uses the system anaconda's sys.path, not the conda env's.
# This causes "No module named 'torch'" and similar errors.
unset PYTHONHOME
unset PYTHONPATH

# Step 4: Define an explicit Python path (belt-and-suspenders with Step 3)
PYTHON="${CONDA_ENV}/bin/python"
if [ ! -x "$PYTHON" ]; then
    echo "ERROR: Python not found at $PYTHON" >&2
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version 2>&1))"

# Step 5: Verify the environment works (catches problems early)
"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || {
    echo "ERROR: torch import failed — environment is broken" >&2
    exit 1
}

# Step 6: Set remaining env vars
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"

cd "$PROJECT_ROOT"

# ============================================================================
# Run your Python script — ALWAYS use "$PYTHON", never bare "python"
# ============================================================================
"$PYTHON" my_script.py --arg1 val1 --arg2 val2
```

---

## Rules (MUST follow)

### Rule 1: Always `unset PYTHONHOME` and `unset PYTHONPATH` after conda activate

```bash
conda activate "$CONDA_ENV"
unset PYTHONHOME
unset PYTHONPATH
```

**Why:** `module load anaconda3/...` sets these to the system anaconda. Even after `conda activate`, they remain set and override the conda env's package resolution.

### Rule 2: Always use `"$PYTHON"` (explicit path), never bare `python`

```bash
PYTHON="${CONDA_ENV}/bin/python"
"$PYTHON" my_script.py   # CORRECT
python my_script.py       # WRONG — may resolve to system python
```

**Why:** Even after `conda activate`, `PATH` ordering can be unpredictable on compute nodes. The explicit path guarantees we use the right binary.

### Rule 3: Always set `PYTHONNOUSERSITE=1`

```bash
export PYTHONNOUSERSITE=1
```

**Why:** Prevents Python from importing packages from `~/.local/lib/python3.X/site-packages`, which may contain incompatible versions installed by `pip install --user`.

### Rule 4: Always verify with a smoke test

```bash
"$PYTHON" -c "import torch; print(f'torch {torch.__version__}')" || exit 1
```

**Why:** Catches environment problems immediately (within seconds), instead of failing 30 minutes into a multi-hour job after model loading.

### Rule 5: Always quote `"$PYTHON"` and `"$CONDA_ENV"`

```bash
"$PYTHON" script.py       # CORRECT
$PYTHON script.py         # WRONG — breaks if path has spaces
```

### Rule 6: Keep `set -euo pipefail` at the top

```bash
set -euo pipefail
```

**Why:** Ensures the script exits immediately if any command fails, rather than silently continuing with a broken environment.

---

## Common Errors and Their Causes

| Error | Cause | Fix |
|-------|-------|-----|
| `No module named 'torch'` | `PYTHONHOME` set by `module load` points to system Python | Add `unset PYTHONHOME` after `conda activate` |
| `No module named 'sentence_transformers'` | Same as above, or package not installed in conda env | `unset PYTHONHOME` + `pip install sentence-transformers` in the env |
| `python: command not found` | `module purge` removed system Python from PATH, conda activate failed | Check that `source .../conda.sh` succeeded before `conda activate` |
| Script runs but wrong Python version | `PYTHONPATH` includes system anaconda packages | Add `unset PYTHONPATH` after `conda activate` |
| Works on login node but fails on compute node | Login node has different module versions or cached PATH | Always test with `srun --pty bash` before submitting |

---

## Anti-patterns (DO NOT do these)

### Anti-pattern 1: Missing `unset PYTHONHOME`

```bash
# BAD — will fail intermittently
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/wc3013/conda-envs/longcat
python my_script.py  # torch not found!
```

### Anti-pattern 2: Using bare `python` without explicit path

```bash
# BAD — may resolve to system python
PYTHON="${CONDA_ENV}/bin/python"
"$PYTHON" script_a.py
python script_b.py      # Oops, forgot to use $PYTHON — may use wrong python
```

### Anti-pattern 3: Mixing `$PYTHON` and bare `python` in the same script

```bash
# BAD — inconsistent, hard to spot
case "${METHOD}" in
  full)  "$PYTHON" run_full.py ;;
  lora)  python run_lora.py ;;    # BUG: uses bare python
esac
```

### Anti-pattern 4: Skipping the smoke test

```bash
# BAD — won't know env is broken until 30min into the job
"$PYTHON" my_long_running_script.py  # fails after loading 27GB model
```

---

## Template: Minimal sbatch with conda

Copy-paste this as a starting point for any new sbatch script:

```bash
#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:h200:1
#SBATCH --output=sweep_experiment/slurm_log/%x_%j.out
#SBATCH --error=sweep_experiment/slurm_log/%x_%j.err

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"

# --- Conda setup (DO NOT MODIFY this block) ---
module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "$CONDA_ENV"
unset PYTHONHOME
unset PYTHONPATH
PYTHON="${CONDA_ENV}/bin/python"
"$PYTHON" -c "import torch; print(f'torch {torch.__version__}, CUDA {torch.cuda.is_available()}')" || exit 1
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "$HF_HOME"
# --- End conda setup ---

cd "$PROJECT_ROOT"

# Your code here:
"$PYTHON" your_script.py --your-args
```

---

## Checklist for reviewing sbatch scripts

Before submitting any sbatch script, verify:

- [ ] `set -euo pipefail` is at the top
- [ ] `export PYTHONNOUSERSITE=1` is set
- [ ] `module purge` + `module load anaconda3/2025.06` is present
- [ ] `source .../conda.sh` is called
- [ ] `conda activate "$CONDA_ENV"` is called
- [ ] **`unset PYTHONHOME`** appears AFTER `conda activate`
- [ ] **`unset PYTHONPATH`** appears AFTER `conda activate`
- [ ] `PYTHON="${CONDA_ENV}/bin/python"` is defined
- [ ] ALL Python calls use `"$PYTHON"`, never bare `python`
- [ ] A smoke test (`"$PYTHON" -c "import torch"`) runs before the main script
- [ ] `LD_LIBRARY_PATH` includes `${CONDA_ENV}/lib`
