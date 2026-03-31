#!/bin/bash
# ============================================================================
# Panda-70M: Submit all 9 LongCat TTA runs (1 no-TTA + 4 Full + 4 LoRA)
#
# Usage (on cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/scripts/submit_panda.sh          # submit all
#   bash sweep_experiment/scripts/submit_panda.sh --dry-run # preview only
# ============================================================================
set -euo pipefail

ACCOUNT="torch_pr_36_mren"
DATA_DIR="/scratch/wc3013/longcat-video-tta/datasets/panda_1000_480p"
TIME_LIMIT="48:00:00"
EXTRA_ARGS="${@}"

echo "=============================================================================="
echo "Panda-70M: Full Experiment Suite (9 LongCat TTA runs)"
echo "=============================================================================="
echo "Account  : ${ACCOUNT}"
echo "Dataset  : ${DATA_DIR}"
echo "Time     : ${TIME_LIMIT}"
echo "Extra    : ${EXTRA_ARGS:-<none>}"
echo "=============================================================================="
echo ""

# Verify dataset exists
if [ ! -f "${DATA_DIR}/metadata.csv" ]; then
    echo "ERROR: Dataset not found at ${DATA_DIR}"
    echo "Run the dataset preparation first:"
    echo "  NUM_VIDEOS=1000 MIN_FRAMES=62 sbatch --account=${ACCOUNT} datasets/download_panda70m.sbatch"
    echo "  SRC_DIR=datasets/panda_1000 DST_DIR=datasets/panda_1000_480p sbatch --account=${ACCOUNT} --partition=cpu_short datasets/resize_videos.sbatch"
    exit 1
fi

echo "Dataset check: $(wc -l < "${DATA_DIR}/metadata.csv") lines in metadata.csv"
echo "Videos: $(ls "${DATA_DIR}/videos/"*.mp4 2>/dev/null | wc -l) files"
echo ""

# --- No-TTA baseline (1 run) ---
echo ">>> Submitting No-TTA baseline (1 run)..."
python3 sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_notta.yaml \
    --account "${ACCOUNT}" \
    --data-dir "${DATA_DIR}" \
    --time "${TIME_LIMIT}" \
    ${EXTRA_ARGS}

echo ""

# --- Full-model TTA ES+CLIP ablation (4 runs) ---
echo ">>> Submitting Full-model TTA ablation (4 runs)..."
python3 sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_full_ablation.yaml \
    --account "${ACCOUNT}" \
    --data-dir "${DATA_DIR}" \
    --time "${TIME_LIMIT}" \
    ${EXTRA_ARGS}

echo ""

# --- LoRA TTA ES+CLIP ablation (4 runs) ---
echo ">>> Submitting LoRA TTA ablation (4 runs)..."
python3 sweep_experiment/scripts/run_sweep.py \
    --config sweep_experiment/configs/panda_lora_ablation.yaml \
    --account "${ACCOUNT}" \
    --data-dir "${DATA_DIR}" \
    --time "${TIME_LIMIT}" \
    ${EXTRA_ARGS}

echo ""
echo "=============================================================================="
echo "Panda-70M submission complete: 9 total LongCat TTA runs"
echo "=============================================================================="
echo "Monitor: squeue -u \$USER"
echo "Results: sweep_experiment/results/panda_*/"
echo ""
echo "Runs without ES may exceed 48h for 1000 videos."
echo "Checkpoint-based resume: resubmit the same command to continue."
echo "=============================================================================="
