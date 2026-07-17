#!/bin/bash
# =============================================================================
# SMOKE TEST for the three comparison baselines (NOT for paper numbers).
#
# Runs SAVi-DNO (fair), SlowFast-VGen/Temp-LoRA, and TTC end-to-end on a tiny
# handful of videos with reduced steps, just to prove each pipeline runs on the
# cluster, saves an mp4, and produces a PSNR. FVD/VBench are intentionally OFF
# (FVD needs >=256 videos; VBench is slow) -- turn them on for the real runs.
#
# Requests 2 GPUs because SAVi-DNO splits the DiT across cuda:0/1 for the
# memory-heavy differentiable noise optimization. TTC and Temp-LoRA use cuda:0.
#
# Usage:
#   sbatch [--account=...] comparison_methods/sbatch/smoke_test_baselines.sbatch
# Optional env overrides:
#   CHECKPOINT_DIR, SMOKE_DATA_DIR, SMOKE_N (default 2), SMOKE_ONLY=savi|temp|ttc
# =============================================================================
#SBATCH --job-name=smoke_baselines
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256GB
#SBATCH --time=03:00:00
#SBATCH --output=comparison_methods/slurm_log/smoke_baselines_%j.out
#SBATCH --error=comparison_methods/slurm_log/smoke_baselines_%j.err

set -uo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
DATA_DIR="${SMOKE_DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
N="${SMOKE_N:-2}"
ONLY="${SMOKE_ONLY:-all}"
OUT_ROOT="${PROJECT_ROOT}/comparison_methods/results/_smoke"

echo "=============================================================================="
echo "SMOKE TEST -- comparison baselines (not paper numbers)"
echo "  Job $SLURM_JOB_ID on $SLURM_NODELIST  @ $(date)"
echo "  checkpoint : ${CHECKPOINT_DIR}"
echo "  data       : ${DATA_DIR}"
echo "  videos     : ${N}   only=${ONLY}"
echo "  out root   : ${OUT_ROOT}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
CONDA_ENV="${SCRATCH_BASE}/conda-envs/longcat"
conda activate "$CONDA_ENV"
unset PYTHONHOME
unset PYTHONPATH
PYTHON="${CONDA_ENV}/bin/python"
export LD_LIBRARY_PATH="${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
cd "${PROJECT_ROOT}"
mkdir -p comparison_methods/slurm_log "${OUT_ROOT}"

"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())" || {
    echo "FATAL: torch import failed"; exit 1; }

RC_SAVI=0; RC_TEMP=0; RC_TTC=0

run_one () { echo; echo "########## $1 ##########"; }

# --- 1) SAVi-DNO (fair, leakage-free) : 14+14@48, tiny steps, 2-GPU split ---
if [ "$ONLY" = "all" ] || [ "$ONLY" = "savi" ]; then
run_one "SAVi-DNO (fair)"
"$PYTHON" comparison_methods/scripts/savi_dno_longcat.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" --data-dir "${DATA_DIR}" \
    --output-dir "${OUT_ROOT}/savi_dno" \
    --max-videos "${N}" --num-gpus 2 \
    --num-cond-frames 14 --num-frames 28 --gen-start-frame 48 \
    --num-inference-steps 6 --rollout-steps 2
RC_SAVI=$?
echo "[smoke] SAVi-DNO exit=${RC_SAVI}"
fi

# --- 2) SlowFast-VGen / Temp-LoRA : short horizon, 2 fast-update steps ---
if [ "$ONLY" = "all" ] || [ "$ONLY" = "temp" ]; then
run_one "SlowFast-VGen (Temp-LoRA)"
"$PYTHON" lora_experiment/scripts/run_temp_lora_tta.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" --data-dir "${DATA_DIR}" \
    --output-dir "${OUT_ROOT}/temp_lora" \
    --max-videos "${N}" --restart \
    --num-cond-frames 14 --num-frames 28 --gen-start-frame 48 \
    --num-steps 2 --num-inference-steps 8 \
    --temp-lora-step-latents 1 \
    --caption-guard-mode warn
RC_TEMP=$?
echo "[smoke] Temp-LoRA exit=${RC_TEMP}"
fi

# --- 3) TTC : LONG horizon 14+79@14, reduced inference steps ---
if [ "$ONLY" = "all" ] || [ "$ONLY" = "ttc" ]; then
run_one "TTC (long horizon)"
"$PYTHON" comparison_methods/scripts/ttc_longcat.py \
    --checkpoint-dir "${CHECKPOINT_DIR}" --data-dir "${DATA_DIR}" \
    --output-dir "${OUT_ROOT}/ttc" \
    --max-videos "${N}" \
    --num-cond-frames 14 --num-frames 93 --gen-start-frame 14 \
    --num-inference-steps 8 \
    --ttc-sigma-threshold 0.3 --ttc-cadence 1 --ttc-weight 0.1
RC_TTC=$?
echo "[smoke] TTC exit=${RC_TTC}"
fi

echo
echo "=============================================================================="
echo "SMOKE SUMMARY  @ $(date)"
for pair in "SAVi-DNO:${RC_SAVI}" "Temp-LoRA:${RC_TEMP}" "TTC:${RC_TTC}"; do
    name="${pair%%:*}"; rc="${pair##*:}"
    dir="${OUT_ROOT}/$( [ "$name" = "SAVi-DNO" ] && echo savi_dno || { [ "$name" = "Temp-LoRA" ] && echo temp_lora || echo ttc; } )"
    n_mp4=$(find "$dir" -name '*.mp4' 2>/dev/null | wc -l | tr -d ' ')
    ok="[?]"; [ "$rc" = "0" ] && ok="[PASS]" || ok="[FAIL rc=$rc]"
    echo "  ${ok} ${name}: mp4s=${n_mp4}  summary=$( [ -f "$dir/summary.json" ] && echo yes || echo NO )  ($dir)"
done
echo "=============================================================================="
