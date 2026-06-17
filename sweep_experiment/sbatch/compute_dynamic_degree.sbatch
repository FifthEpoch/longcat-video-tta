#!/bin/bash
#SBATCH --job-name=compute_dynamic_degree
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=sweep_experiment/slurm_log/compute_dynamic_degree_%j.out
#SBATCH --error=sweep_experiment/slurm_log/compute_dynamic_degree_%j.err

# ============================================================================
# Compute per-video dynamicness (VBench-style RAFT mean optical flow) for an
# evaluation video directory. Output JSON is consumed by
# scripts/analyze_per_video_tta_gain.py and plot_dynamicness_correlation.py.
#
# Required env (pass via --export=ALL):
#   VIDEOS_DIR  : eval video directory (e.g. datasets/panda_1000_480p)
#   OUTPUT_JSON : destination JSON path (typically VIDEOS_DIR/dynamic_degree.json)
#
# Optional env:
#   TTA_VISIBLE_FRAMES : default auto (0:48 for panda_1000v_standard TTA window)
#   MAX_FRAMES         : default 0 (decode full visible span). Set to 28 to
#                        reproduce the pre-frame_window-fix 28-frame slice.
#   TARGET_SIZE        : HxW (default 256x320, both divisible by 8 as RAFT requires)
#   METHOD             : raft (default) or farneback
#
# Submit (Panda standard horizon):
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,\
# VIDEOS_DIR=/scratch/${USER}/longcat-video-tta/datasets/panda_1000_480p,\
# OUTPUT_JSON=/scratch/${USER}/longcat-video-tta/datasets/panda_1000_480p/dynamic_degree.json \
#       sweep_experiment/sbatch/compute_dynamic_degree.sbatch
#
# Submit (UCF standard horizon):
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,\
# VIDEOS_DIR=/scratch/${USER}/longcat-video-tta/datasets/ucf101_1000_480p,\
# OUTPUT_JSON=/scratch/${USER}/longcat-video-tta/datasets/ucf101_1000_480p/dynamic_degree.json \
#       sweep_experiment/sbatch/compute_dynamic_degree.sbatch
#
# Expected wall time: ~10-20 min on an H200 for 1000 videos x 48 frames at
# 256x320 (47 RAFT frame-pairs per video).
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

VIDEOS_DIR="${VIDEOS_DIR:?ERROR: VIDEOS_DIR must be set (--export=ALL,VIDEOS_DIR=...)}"
OUTPUT_JSON="${OUTPUT_JSON:?ERROR: OUTPUT_JSON must be set}"
TTA_VISIBLE_FRAMES="${TTA_VISIBLE_FRAMES:-auto}"
MAX_FRAMES="${MAX_FRAMES:-0}"
TARGET_SIZE="${TARGET_SIZE:-256x320}"
METHOD="${METHOD:-raft}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
export HF_HOME="${HF_HOME:-${SCRATCH_BASE}/hf-cache}"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${TORCH_HOME:-${SCRATCH_BASE}/torch-hub-cache}"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${HF_HUB_CACHE}" \
    "${TORCH_HOME}" "${PROJECT_ROOT}/sweep_experiment/slurm_log"

echo "=============================================================================="
echo "Compute per-video dynamicness (RAFT mean optical flow)"
echo "=============================================================================="
echo "  Videos dir         : ${VIDEOS_DIR}"
echo "  Output             : ${OUTPUT_JSON}"
echo "  TTA visible frames : ${TTA_VISIBLE_FRAMES}"
echo "  Max frames cap     : ${MAX_FRAMES} (0 = full visible span)"
echo "  Target size        : ${TARGET_SIZE}"
echo "  Method             : ${METHOD}"
echo "  TORCH_HOME         : ${TORCH_HOME}"
echo "  Job ID             : ${SLURM_JOB_ID:-(not slurm)}"
echo "=============================================================================="

if [ ! -d "${VIDEOS_DIR}" ]; then
    echo "ERROR: ${VIDEOS_DIR} does not exist."
    exit 2
fi

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

if [ ! -d "${ENV_PATH}" ]; then
    echo "ERROR: Conda environment not found at ${ENV_PATH}" >&2
    exit 1
fi

conda activate "${ENV_PATH}"
unset PYTHONHOME
unset PYTHONPATH
export LD_LIBRARY_PATH="${ENV_PATH}/lib:${LD_LIBRARY_PATH:-}"
export PATH="${ENV_PATH}/bin:${PATH}"
PYTHON="${ENV_PATH}/bin/python"

if ! "${PYTHON}" -c "import cv2; import torch" 2>/dev/null; then
    echo "ERROR: cv2/torch import failed in ${ENV_PATH}." >&2
    echo "  conda activate ${ENV_PATH} && python -c \"import cv2, torch\"" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"

EXTRA=()
if [ "${MAX_FRAMES}" != "0" ]; then
    EXTRA+=(--max-frames "${MAX_FRAMES}")
fi

"${PYTHON}" scripts/compute_dynamic_degree.py \
    --videos-dir "${VIDEOS_DIR}" \
    --output-json "${OUTPUT_JSON}" \
    --tta-visible-frames "${TTA_VISIBLE_FRAMES}" \
    --target-size "${TARGET_SIZE}" \
    --method "${METHOD}" \
    "${EXTRA[@]}"

echo ""
echo "=============================================================================="
echo "Done. Dynamicness JSON: ${OUTPUT_JSON}"
echo ""
echo "Verify metadata (post frame_window fix, expect tta_visible_range=0:48):"
echo "  python3 -c \"import json; d=json.load(open('${OUTPUT_JSON}')); print(d.get('tta_visible_range'), d.get('n_frames_used'))\""
echo ""
echo "Next: regenerate per_video_gains.csv, e.g."
echo ""
echo "  python3 scripts/analyze_per_video_tta_gain.py \\"
echo "      --series-path sweep_experiment/results/panda_1000v_standard \\"
echo "      --tinylora-series-path delta_experiment/results/tinylora_panda_1000v_standard \\"
echo "      --dynamicness-json ${OUTPUT_JSON} \\"
echo "      --captions-csv ${VIDEOS_DIR}/metadata.csv \\"
echo "      --output-dir sweep_experiment/reports/per_video_analysis/2026-06-09"
echo "=============================================================================="
