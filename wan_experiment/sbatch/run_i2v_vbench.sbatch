#!/bin/bash
#SBATCH --job-name=wan_i2v_vbench
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:h200:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_i2v_vbench_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_i2v_vbench_%j.err
#SBATCH --comment="preemption=yes;requeue=true"

# Official VBench quality scoring on already-generated Wan I2V mp4s.
# Uses vbench-backfill (NOT self_forcing, NOT longcat).
#
# Required:
#   VIDEO_DIRS  comma-separated shard dirs (mp4 + summary.json)
# Optional:
#   CLIPS       space-separated: last5 and/or full and/or first5 (default: last5 full)
#   DIMS        VBench quality dims
#   FORCE       1 to re-run existing dims
#
# last5 is scored first: it is the outcome window and is much cheaper
# than the full 481-frame clip.

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_NAME="${VBENCH_ENV_NAME:-vbench-backfill}"
DEFAULT_VBENCH_PYTHON="${SCRATCH_BASE}/conda-envs/${ENV_NAME}/bin/python"
CACHE_DIR="${VBENCH_CACHE_DIR:-${SCRATCH_BASE}/vbench-cache}"

cd "${PROJECT_ROOT}"

VIDEO_DIRS="${VIDEO_DIRS:?ERROR: VIDEO_DIRS must be set}"
CLIPS="${CLIPS:-last5 full}"
DIMS_DEFAULT="subject_consistency background_consistency aesthetic_quality imaging_quality motion_smoothness dynamic_degree temporal_flickering"
DIMS="${DIMS:-$DIMS_DEFAULT}"
MODE="${MODE:-custom_input}"
FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then
    FORCE_FLAG="--force"
fi

export HF_HOME="${CACHE_DIR}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${CACHE_DIR}/torch"
export VBENCH_CACHE_DIR="${CACHE_DIR}"
export TMPDIR="${SCRATCH_BASE}/tmp"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TORCH_HOME}" "${TMPDIR}" \
    "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [ -n "${VBENCH_PYTHON:-}" ] && [ -x "${VBENCH_PYTHON}" ]; then
    PYTHON="${VBENCH_PYTHON}"
elif [ -x "${DEFAULT_VBENCH_PYTHON}" ]; then
    PYTHON="${DEFAULT_VBENCH_PYTHON}"
else
    echo "ERROR: vbench-backfill python not found at ${DEFAULT_VBENCH_PYTHON}" >&2
    exit 2
fi
unset PYTHONHOME
unset PYTHONPATH

echo "=============================================================================="
echo "Wan I2V official VBench"
echo "  VIDEO_DIRS : ${VIDEO_DIRS}"
echo "  CLIPS      : ${CLIPS}"
echo "  DIMS       : ${DIMS}"
echo "  MODE       : ${MODE}"
echo "  Python     : ${PYTHON}"
echo "  Job        : ${SLURM_JOB_ID:-(not slurm)}"
echo "=============================================================================="
"${PYTHON}" -c "
import numpy, torch, vbench, cv2
print('  numpy', numpy.__version__)
print('  torch', torch.__version__)
print('  cv2', cv2.__version__)
print('  cuda', torch.cuda.is_available())
"

STATUS=0
IFS=',' read -ra DIRS <<< "${VIDEO_DIRS}"
for raw in "${DIRS[@]}"; do
    d="$(echo "${raw}" | xargs)"
    if [[ "${d}" != /* ]]; then
        d="${PROJECT_ROOT}/${d}"
    fi
    if [ ! -d "${d}" ]; then
        echo "ERROR: missing ${d}" >&2
        STATUS=2
        continue
    fi
    for clip in ${CLIPS}; do
        echo ""
        echo "----- ${d}  clip=${clip} -----"
        "${PYTHON}" "${PROJECT_ROOT}/wan_experiment/scripts/score_i2v_vbench.py" \
            --video-dir "${d}" \
            --clip "${clip}" \
            --mode "${MODE}" \
            --dimensions ${DIMS} \
            ${FORCE_FLAG} || STATUS=$?
    done
done

echo ""
echo "=============================================================================="
echo "VBench job exit status so far: ${STATUS}"
echo "Analyze (login node, any python):"
echo "  python wan_experiment/scripts/analyze_i2v_vbench.py \\"
echo "    --series-dir wan_experiment/results/i2v_bon_32v_hybrid --clip last5"
echo "=============================================================================="
exit ${STATUS}
