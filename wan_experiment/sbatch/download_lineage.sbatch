#!/bin/bash
#SBATCH --job-name=wan_lineage_dl
#SBATCH --account=torch_pr_36_mren
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_lineage_dl_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/wan_experiment/slurm_log/wan_lineage_dl_%j.err

# Clone LongLive v1.0 + Rolling Forcing and download their 1.3B checkpoints.
# CPU-only. Idempotent. Also installs peft into self_forcing if missing.
#
#   sbatch wan_experiment/sbatch/download_lineage.sbatch

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
LONGCAT_ENV="${SCRATCH_BASE}/conda-envs/longcat"
SF_ENV="${SCRATCH_BASE}/conda-envs/self_forcing"
CKPT_ROOT="${CKPT_ROOT:-${SCRATCH_BASE}/wan-checkpoints}"
TP="${SCRATCH_BASE}/third_party"
LL_ROOT="${LL_ROOT:-${TP}/LongLive}"
RF_ROOT="${RF_ROOT:-${TP}/RollingForcing}"
LL_DIR="${CKPT_ROOT}/longlive"
LL_BASE="${LL_DIR}/models/longlive_base.pt"
LL_LORA="${LL_DIR}/models/lora.pt"
RF_CKPT="${CKPT_ROOT}/rolling_forcing_dmd.pt"
FORCE="${FORCE:-0}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
export HF_HOME="${SCRATCH_BASE}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_HUB_ENABLE_HF_TRANSFER=0
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$HF_HOME" "$CKPT_ROOT" "$LL_DIR/models" \
    "$TP" "${PROJECT_ROOT}/wan_experiment/slurm_log"

echo "=============================================================================="
echo "Self-Forcing lineage asset download"
echo "  job     : ${SLURM_JOB_ID:-local}  node=$(hostname)  $(date)"
echo "  LongLive: ${LL_ROOT}"
echo "  Rolling : ${RF_ROOT}"
echo "  LL ckpt : ${LL_BASE}"
echo "  RF ckpt : ${RF_CKPT}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${LONGCAT_ENV}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${LONGCAT_ENV}/bin:${PATH}"
PYTHON="${LONGCAT_ENV}/bin/python"

"${PYTHON}" -c "import huggingface_hub" 2>/dev/null || \
    "${LONGCAT_ENV}/bin/pip" install -q "huggingface_hub[cli]"

HF="${LONGCAT_ENV}/bin/huggingface-cli"
if [ ! -x "${HF}" ]; then
    HF="${PYTHON} -m huggingface_hub.commands.huggingface_cli"
fi

if [ "${FORCE}" = "1" ] || [ ! -f "${LL_ROOT}/pipeline/causal_inference.py" ]; then
    echo "Cloning NVlabs/LongLive v1.0 -> ${LL_ROOT}"
    rm -rf "${LL_ROOT}"
    git clone --single-branch --branch v1.0 --depth 1 \
        https://github.com/NVlabs/LongLive.git "${LL_ROOT}"
else
    echo "LongLive clone already present; skip"
fi

if [ "${FORCE}" = "1" ] || [ ! -f "${RF_ROOT}/pipeline/rolling_forcing_inference.py" ]; then
    echo "Cloning TencentARC/RollingForcing -> ${RF_ROOT}"
    rm -rf "${RF_ROOT}"
    git clone --depth 1 https://github.com/TencentARC/RollingForcing.git "${RF_ROOT}"
else
    echo "RollingForcing clone already present; skip"
fi

if [ "${FORCE}" = "1" ] || [ ! -f "${LL_BASE}" ] || [ ! -f "${LL_LORA}" ]; then
    echo "Downloading Efficient-Large-Model/LongLive-1.3B models/"
    ${HF} download Efficient-Large-Model/LongLive-1.3B \
        models/longlive_base.pt models/lora.pt \
        --local-dir "${LL_DIR}" --local-dir-use-symlinks False
else
    echo "LongLive ckpts already present; skip"
fi

if [ "${FORCE}" = "1" ] || [ ! -f "${RF_CKPT}" ]; then
    echo "Downloading TencentARC/RollingForcing checkpoints/rolling_forcing_dmd.pt"
    ${HF} download TencentARC/RollingForcing checkpoints/rolling_forcing_dmd.pt \
        --local-dir "${CKPT_ROOT}/_rf_dl"
    if [ -f "${CKPT_ROOT}/_rf_dl/checkpoints/rolling_forcing_dmd.pt" ]; then
        mv -f "${CKPT_ROOT}/_rf_dl/checkpoints/rolling_forcing_dmd.pt" "${RF_CKPT}"
    elif [ -f "${CKPT_ROOT}/_rf_dl/rolling_forcing_dmd.pt" ]; then
        mv -f "${CKPT_ROOT}/_rf_dl/rolling_forcing_dmd.pt" "${RF_CKPT}"
    else
        echo "ERROR: RF ckpt not found under ${CKPT_ROOT}/_rf_dl" >&2
        find "${CKPT_ROOT}/_rf_dl" -type f | head -50 >&2
        exit 1
    fi
else
    echo "Rolling Forcing ckpt already at ${RF_CKPT}; skip"
fi

if [ -x "${SF_ENV}/bin/python" ]; then
    "${SF_ENV}/bin/python" -c "import peft" 2>/dev/null || \
        "${SF_ENV}/bin/pip" install -q peft
    echo "peft: ok in self_forcing"
else
    echo "WARNING: self_forcing env missing; LongLive jobs will fail to import peft"
fi

echo "---- sizes ----"
du -sh "${LL_ROOT}" "${RF_ROOT}" "${LL_BASE}" "${LL_LORA}" "${RF_CKPT}" 2>/dev/null || true
ls -lh "${LL_BASE}" "${LL_LORA}" "${RF_CKPT}"
echo "=============================================================================="
echo "Lineage download complete: $(date)"
echo "=============================================================================="
