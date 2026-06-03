#!/bin/bash
# =============================================================================
# One-time setup for the VBench backfill pipeline.
#
# Creates a separate ``vbench-backfill`` conda env pinned to versions that
# work with VBench 0.1.5 (NumPy 1.x, PyTorch 2.5.x). The longcat env stays
# untouched so TTA continues to work with NumPy 2.x / PyTorch 2.6.
#
# Pre-downloads VBench's pretrained checkpoints to a known cache dir so that
# GPU compute jobs don't need outbound network at SLURM run time.
#
# Run on the LOGIN NODE (not a compute node) — needs network access.
#
#   bash scripts/setup_vbench_backfill_env.sh
#
# Idempotent: safe to re-run.
# =============================================================================

set -euo pipefail

ENV_NAME="${VBENCH_ENV_NAME:-vbench-backfill}"
SCRATCH_BASE="${SCRATCH_BASE:-/scratch/$USER}"
CACHE_DIR="${VBENCH_CACHE_DIR:-${SCRATCH_BASE}/vbench-cache}"

echo "=============================================================================="
echo "VBench backfill env setup"
echo "=============================================================================="
echo "  ENV_NAME       : ${ENV_NAME}"
echo "  CACHE_DIR      : ${CACHE_DIR}"
echo "=============================================================================="

mkdir -p "${CACHE_DIR}"

# ---- Create or reuse env ----------------------------------------------------
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[1/3] env '${ENV_NAME}' already exists; skipping creation."
else
    echo "[1/3] Creating conda env '${ENV_NAME}' (python=3.10) ..."
    conda create -n "${ENV_NAME}" python=3.10 -y
fi

# Activate env (works under bash with conda init)
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

# ---- Install pinned deps ----------------------------------------------------
echo "[2/3] Installing pinned dependencies for VBench 0.1.5 ..."
pip install --no-cache-dir \
    'numpy==1.26.4' \
    'torch==2.5.1' \
    'torchvision==0.20.1' \
    'vbench==0.1.5' \
    'decord' \
    'opencv-python' \
    'einops' \
    'timm' \
    'pyiqa' \
    'open-clip-torch' \
    'ftfy' \
    'regex' \
    'tqdm'

# ---- Verify all 7 dimensions import cleanly --------------------------------
echo "[3/3] Verifying dimension imports + pre-downloading checkpoints ..."

export HF_HOME="${CACHE_DIR}/hf"
export TORCH_HOME="${CACHE_DIR}/torch"
export VBENCH_CACHE_DIR="${CACHE_DIR}"
mkdir -p "${HF_HOME}" "${TORCH_HOME}"

python3 - <<'PY'
import os, sys, importlib, traceback

print("\n===== Importable dimension modules =====")
DIMS = ["subject_consistency","background_consistency","motion_smoothness",
        "dynamic_degree","aesthetic_quality","imaging_quality","temporal_flickering"]
ok = []
for dim in DIMS:
    try:
        importlib.import_module(f"vbench.{dim}")
        print(f"  vbench.{dim} -> OK")
        ok.append(dim)
    except Exception as e:
        print(f"  vbench.{dim} -> FAIL: {type(e).__name__}: {e}")

if len(ok) < 7:
    print(f"\n[warn] {len(ok)}/7 dimensions importable.")
    sys.exit(0)

print("\n===== Instantiating VBench (triggers checkpoint downloads) =====")
import torch, vbench as _v
pkg_dir = os.path.dirname(_v.__file__)
full_info = os.path.join(pkg_dir, "VBench_full_info.json")
if not os.path.exists(full_info):
    full_info = os.path.join(os.path.dirname(pkg_dir), "vbench", "VBench_full_info.json")

cache = os.environ.get("VBENCH_CACHE_DIR", os.path.expanduser("~/vbench-cache"))
output_path = os.path.join(cache, "_pretrain_check")
os.makedirs(output_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  device         : {device}")
print(f"  full_info_json : {full_info}")
print(f"  output_path    : {output_path}")

from vbench import VBench
try:
    vb = VBench(device, full_info, output_path)
    print("  VBench instance creation: OK")
except Exception as e:
    print(f"  VBench instance creation: FAIL: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
PY

echo ""
echo "=============================================================================="
echo "Setup complete."
echo ""
echo "Cache dir : ${CACHE_DIR}"
echo "Env name  : ${ENV_NAME}  (NOT longcat — keep them separate!)"
echo ""
echo "Next: run discovery to see how many method dirs need backfill:"
echo "  python3 scripts/discover_vbench_backfill_targets.py"
echo ""
echo "Then submit the sweep:"
echo "  bash scripts/submit_vbench_backfill_all.sh"
echo "=============================================================================="
