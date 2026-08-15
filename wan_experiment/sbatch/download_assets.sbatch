#!/bin/bash
#SBATCH --job-name=wan_sf_dl
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_sf_dl_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_sf_dl_%j.err

# CPU download of Wan2.1-T2V-1.3B + Self-Forcing DMD ckpt + VBench-I2V images.
# Uses the EXISTING longcat env only for huggingface-cli / gdown (no new deps
# installed into longcat). Idempotent; huggingface-cli resumes.
#
#   sbatch wan_experiment/sbatch/download_assets.sbatch

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
LONGCAT_ENV="${SCRATCH_BASE}/conda-envs/longcat"
CKPT_ROOT="${CKPT_ROOT:-${SCRATCH_BASE}/wan-checkpoints}"
WAN_DIR="${CKPT_ROOT}/Wan2.1-T2V-1.3B"
SF_CKPT="${CKPT_ROOT}/self_forcing_dmd.pt"
I2V_DIR="${I2V_DIR:-${PROJECT_ROOT}/datasets/vbench_i2v}"
FORCE="${FORCE:-0}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$CKPT_ROOT" "$I2V_DIR" \
    "${PROJECT_ROOT}/wan_experiment/slurm_log"

echo "=============================================================================="
echo "Wan / Self-Forcing asset download"
echo "  job     : ${SLURM_JOB_ID:-local}  node=$(hostname)  $(date)"
echo "  wan     : ${WAN_DIR}"
echo "  sf ckpt : ${SF_CKPT}"
echo "  i2v     : ${I2V_DIR}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${LONGCAT_ENV}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${LONGCAT_ENV}/bin:${PATH}"
PYTHON="${LONGCAT_ENV}/bin/python"

# huggingface-cli + gdown from longcat (install to USER site only if missing;
# PYTHONNOUSERSITE is on, so install into the env if needed — both are tiny).
"${PYTHON}" -c "import huggingface_hub" 2>/dev/null || \
    "${LONGCAT_ENV}/bin/pip" install -q "huggingface_hub[cli]"
"${PYTHON}" -c "import gdown" 2>/dev/null || \
    "${LONGCAT_ENV}/bin/pip" install -q gdown

HF="${LONGCAT_ENV}/bin/huggingface-cli"
if [ ! -x "${HF}" ]; then
    HF="${PYTHON} -m huggingface_hub.commands.huggingface_cli"
fi

# ---- Wan2.1-T2V-1.3B (~15 GB; T5 encoder is ~11.4 GB) --------------------
if [ "${FORCE}" = "1" ] || [ ! -f "${WAN_DIR}/config.json" ]; then
    echo "Downloading Wan-AI/Wan2.1-T2V-1.3B -> ${WAN_DIR}"
    ${HF} download Wan-AI/Wan2.1-T2V-1.3B \
        --local-dir "${WAN_DIR}" --local-dir-use-symlinks False
else
    echo "Wan dir already has config.json; skip (FORCE=1 to redo)"
fi

# ---- Self-Forcing DMD checkpoint -----------------------------------------
if [ "${FORCE}" = "1" ] || [ ! -f "${SF_CKPT}" ]; then
    echo "Downloading gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt"
    ${HF} download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt \
        --local-dir "${CKPT_ROOT}/_sf_dl"
    # Hub layout is local-dir/checkpoints/self_forcing_dmd.pt
    if [ -f "${CKPT_ROOT}/_sf_dl/checkpoints/self_forcing_dmd.pt" ]; then
        mv -f "${CKPT_ROOT}/_sf_dl/checkpoints/self_forcing_dmd.pt" "${SF_CKPT}"
    elif [ -f "${CKPT_ROOT}/_sf_dl/self_forcing_dmd.pt" ]; then
        mv -f "${CKPT_ROOT}/_sf_dl/self_forcing_dmd.pt" "${SF_CKPT}"
    else
        echo "ERROR: downloaded Self-Forcing ckpt not found under ${CKPT_ROOT}/_sf_dl" >&2
        find "${CKPT_ROOT}/_sf_dl" -type f | head -50 >&2
        exit 1
    fi
else
    echo "Self-Forcing DMD already at ${SF_CKPT}; skip"
fi

# ---- VBench-I2V image suite (Google Drive; official VBench++ script) -----
# Folder: https://drive.google.com/drive/folders/1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx
# gdown --folder is the documented path. If Drive rate-limits, the healthcheck
# will flag images as missing and the rest of the stack is still usable.
N_IMG=$(find "${I2V_DIR}" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')
if [ "${FORCE}" = "1" ] || [ "${N_IMG}" -lt 8 ]; then
    echo "Downloading VBench-I2V image suite -> ${I2V_DIR} (have ${N_IMG} images)"
    set +e
    "${PYTHON}" -m gdown --folder \
        "https://drive.google.com/drive/folders/1fdOZKQ7HWZtgutCKKA7CMzOhMFUGv4Zx" \
        -O "${I2V_DIR}" --remaining-ok
    GDOWN_RC=$?
    set -e
    if [ "${GDOWN_RC}" -ne 0 ]; then
        echo "WARNING: gdown VBench-I2V failed (rc=${GDOWN_RC}). Healthcheck will mark images missing."
    fi
else
    echo "VBench-I2V already has ${N_IMG} images; skip"
fi

echo "---- sizes ----"
du -sh "${WAN_DIR}" "${SF_CKPT}" "${I2V_DIR}" 2>/dev/null || true
echo "=============================================================================="
echo "Download complete: $(date)"
echo "=============================================================================="
