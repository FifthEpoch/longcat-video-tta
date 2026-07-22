#!/bin/bash
#SBATCH --job-name=vbench_backfill
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=12:00:00
# Logs use absolute path via --chdir in submit script; fallback set below after PROJECT_ROOT.
#SBATCH --output=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/vbench_backfill_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/vbench_backfill_%j.err
# Allow preemptible H200 GPUs; cancelled jobs will be requeued (per cluster IT guidance)
#SBATCH --comment="preemption=yes;requeue=true"

# ============================================================================
# Backfill VBench++ dimensions on saved generated videos.
#
# Activates the dedicated 'vbench-backfill' conda env (NOT longcat) so that
# numpy 1.x / torch 2.5.1 are available — VBench 0.1.5 doesn't work with
# numpy 2.x / torch 2.6 used by the rest of the pipeline.
#
# Required env (pass via --export=ALL):
#   METHOD_DIR : path to a method dir, e.g.
#                sweep_experiment/results/panda_1000v_standard/NOTTA
#
# Optional env:
#   DIMS       : space-separated dim list. Default:
#                "motion_smoothness dynamic_degree imaging_quality temporal_flickering"
#   MODE       : VBench mode (default: custom_input)
#   FORCE      : 1 to re-run even if result files exist (default: 0)
#
# Submit:
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,METHOD_DIR=/scratch/$USER/longcat-video-tta/sweep_experiment/results/panda_1000v_standard/NOTTA \
#       sweep_experiment/sbatch/run_vbench_backfill.sbatch
#
# Wall: ~30-90 min per method dir (10 chunks × 4 dims × ~1-3 min/dim) on H200.
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/$USER"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_NAME="${VBENCH_ENV_NAME:-vbench-backfill}"
# Prefer explicit env python — conda activate is unreliable on some compute nodes.
DEFAULT_VBENCH_PYTHON="${SCRATCH_BASE}/conda-envs/${ENV_NAME}/bin/python"
CACHE_DIR="${VBENCH_CACHE_DIR:-${SCRATCH_BASE}/vbench-cache}"

cd "${PROJECT_ROOT}"

METHOD_DIR="${METHOD_DIR:?ERROR: METHOD_DIR must be set (--export=ALL,METHOD_DIR=...)}"
# Discovery TSV uses repo-relative paths; normalize to absolute under PROJECT_ROOT.
if [[ "${METHOD_DIR}" != /* ]]; then
    METHOD_DIR="${PROJECT_ROOT}/${METHOD_DIR}"
fi
DIMS_DEFAULT="motion_smoothness dynamic_degree imaging_quality temporal_flickering"
DIMS="${DIMS:-$DIMS_DEFAULT}"
MODE="${MODE:-custom_input}"
# Which per-chunk clip dir to evaluate and where to write results. For the
# GENERATED-ONLY VBench recompute (conditioning frames trimmed) pass
# VIDEOS_SUBDIR=videos_geneval OUT_SUBDIR=vbench_results_geneval.
VIDEOS_SUBDIR="${VIDEOS_SUBDIR:-videos}"
OUT_SUBDIR="${OUT_SUBDIR:-vbench_results}"
FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then
    FORCE_FLAG="--force"
fi

# Make HF / torch caches predictable
export HF_HOME="${CACHE_DIR}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${CACHE_DIR}/torch"
export VBENCH_CACHE_DIR="${CACHE_DIR}"
export TMPDIR="${SCRATCH_BASE}/tmp"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${TMPDIR}" \
    "${PROJECT_ROOT}/sweep_experiment/slurm_log"

echo "=============================================================================="
echo "VBench backfill"
echo "=============================================================================="
echo "  Method dir : ${METHOD_DIR}"
echo "  Dims       : ${DIMS}"
echo "  Mode       : ${MODE}"
echo "  Videos dir : ${VIDEOS_SUBDIR}"
echo "  Out dir    : ${OUT_SUBDIR}"
echo "  Force      : ${FORCE:-0}"
echo "  Env        : ${ENV_NAME}"
echo "  Cache      : ${CACHE_DIR}"
echo "  Job ID     : ${SLURM_JOB_ID:-(not slurm)}"
echo "=============================================================================="

if [ ! -d "${METHOD_DIR}" ]; then
    echo "ERROR: ${METHOD_DIR} does not exist."
    exit 2
fi

# Activate the vbench-backfill conda env (not longcat!)
if [ -n "${VBENCH_PYTHON:-}" ] && [ -x "${VBENCH_PYTHON}" ]; then
    PYTHON="${VBENCH_PYTHON}"
elif [ -x "${DEFAULT_VBENCH_PYTHON}" ]; then
    PYTHON="${DEFAULT_VBENCH_PYTHON}"
else
    module purge
    module load anaconda3/2025.06
    source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
    conda activate "${ENV_NAME}"
    PYTHON="$(which python)"
fi
unset PYTHONHOME
unset PYTHONPATH

if [ ! -x "${PYTHON}" ]; then
    echo "ERROR: Python not found. Set VBENCH_PYTHON or create ${DEFAULT_VBENCH_PYTHON}" >&2
    exit 2
fi

# Sanity-print so we can diagnose any mix-up later
echo ""
echo "Active env       : ${CONDA_DEFAULT_ENV:-?}"
echo "Python           : ${PYTHON}"
"${PYTHON}" -c "
import importlib
import numpy, torch, vbench, cv2
print('  numpy =', numpy.__version__)
print('  torch =', torch.__version__)
print('  cv2   =', cv2.__version__)
print('  cuda available =', torch.cuda.is_available())
import importlib.metadata
try:
    print('  vbench =', importlib.metadata.version('vbench'))
except Exception:
    print('  vbench = (unknown)')
for d in ['motion_smoothness','dynamic_degree','imaging_quality','temporal_flickering']:
    importlib.import_module(f'vbench.{d}')
    print(f'  vbench.{d} -> OK')
"
echo ""

cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/run_vbench_backfill.py \
    --method-dir "${METHOD_DIR}" \
    --dimensions ${DIMS} \
    --mode "${MODE}" \
    --videos-subdir "${VIDEOS_SUBDIR}" \
    --out-subdir "${OUT_SUBDIR}" \
    ${FORCE_FLAG}

EXIT=$?

echo ""
echo "=============================================================================="
echo "Backfill exit: ${EXIT}"
echo ""
echo "Next step: fold new dims into merged_summary.json"
echo "  python3 scripts/update_merged_with_vbench.py --method-dir '${METHOD_DIR}'"
echo "=============================================================================="

exit ${EXIT}
