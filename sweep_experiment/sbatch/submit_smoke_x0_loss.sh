#!/bin/bash
# ============================================================================
# Smoke-test: anchor-frame x0 consistency loss (Modification 1).
#
# Fires a SINGLE chunk of LORA_R8_TTA on Panda 1000v with the new
# `--anchor-x0-weight 1.0` flag active. Output lands in
#   sweep_experiment/results/panda_1000v_standard/LORA_R8_TTA_X0_W1.0/chunk_0/
#
# The point is to verify the patch *before* scaling: confirm the loss values
# are sane (non-NaN), the gradients flow, the per-video PSNR/SSIM/LPIPS
# numbers are non-degenerate, and the run-id banner records the resolved
# anchor_x0_weight so future log archaeology is unambiguous.
#
# This is a sibling of submit_standard_1000v_noprompt.sh (same shell style,
# same sbatch wrapper, same series dir as headline LORA_R8_TTA). It differs
# only in:
#   1. Submits a SINGLE chunk (NUM_CHUNKS=1, CHUNK_SIZE=100, MAX_VIDEOS=100).
#   2. Run ID is `LORA_R8_TTA_X0_W1.0` (suffix `_X0_W1.0` makes it
#      distinguishable from headline LORA_R8_TTA in the merged-summary
#      readout).
#   3. Exports `ANCHOR_X0_WEIGHT=1.0`, which run_sweep.sbatch translates
#      into the runner's `--anchor-x0-weight 1.0` flag.
#   4. Uses the EXACT LORA_R8_TTA hyperparameters from
#      submit_standard_1000v_chunked.sh:
#         LORA_RANK=8, LORA_ALPHA=16, LORA_TARGET_BLOCKS=all,
#         NUM_STEPS=10, LEARNING_RATE=5.0e-5, WARMUP_STEPS=3,
#         WEIGHT_DECAY=0.01, MAX_GRAD_NORM=10.0, TARGET_FFN=0.
#      (NB: the runner default lr is 2e-4 — the headline cell uses 5e-5;
#       the smoke must match the headline so the only changing variable
#       is the x0 loss.)
#
# Expected wallclock: ~2 GPU h (one LORA_R8_TTA chunk × 100 videos on H200,
# matching the headline LORA_R8_TTA chunk timing; the x0 term adds zero
# extra forward passes so wall is unchanged).
#
# Submit (after `git pull` on the cluster, once it returns from maintenance):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_smoke_x0_loss.sh
#
# Dry-run (prints sbatch lines without firing):
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_smoke_x0_loss.sh
#
# After the smoke completes, compare against the headline LORA_R8_TTA chunk_0
# on Panda 1000v standard:
#   diff <(jq -r '.results[] | "\(.video_name) \(.psnr)"' \
#            sweep_experiment/results/panda_1000v_standard/LORA_R8_TTA/chunk_0/summary.json \
#            | sort) \
#        <(jq -r '.results[] | "\(.video_name) \(.psnr)"' \
#            sweep_experiment/results/panda_1000v_standard/LORA_R8_TTA_X0_W1.0/chunk_0/summary.json \
#            | sort)
# Decision rule (per LITERATURE_tta_recipe_modifications_2026-06-12.md §3,
# Modification 1):
#   - median |ΔPSNR| > 0.5 dB in EITHER direction on the chunk → signal worth
#     scaling to the full 10-chunk sweep at λ ∈ {0.01, 0.1, 1.0, 10.0}.
#   - NaN gradients OR median |ΔPSNR| < 0.05 dB → loss formulation is not the
#     binding constraint; proceed to Modification 2 (VAE-decoder-only TTA).
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Match LORA_R8_TTA headline wall.
TIME_SHORT="${TIME_SHORT:-12:00:00}"

# SMOKE: single chunk, 100 videos, default seed.
NUM_CHUNKS="${NUM_CHUNKS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-100}"

# Frame geometry — standard 28-frame horizon (matches LORA_R8_TTA headline).
NUM_FRAMES="${NUM_FRAMES:-28}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

# The recipe modification under test.
ANCHOR_X0_WEIGHT="${ANCHOR_X0_WEIGHT:-1.0}"

# Run-id suffix encodes the lambda value so future archaeology is unambiguous.
# Default suffix tracks ANCHOR_X0_WEIGHT — override only if you know what you
# are doing.
RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-_X0_W${ANCHOR_X0_WEIGHT}}"
RUN_ID="${RUN_ID:-LORA_R8_TTA${RUN_ID_SUFFIX}}"

DATASET_TAG="${DATASET_TAG:-panda}"
SERIES_NAME="${SERIES_NAME:-panda_1000v_standard}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
RESULTS_SUBDIR="${RESULTS_SUBDIR:-sweep_experiment/results/${SERIES_NAME}}"

DRY_RUN="${DRY_RUN:-0}"

_exec_or_dry() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY] $*"
        return 0
    fi
    "$@"
}

echo "============================================================"
echo "Smoke-test: anchor-frame x0 consistency loss (Modification 1)"
echo "============================================================"
echo "  series       : ${SERIES_NAME}"
echo "  run id       : ${RUN_ID}"
echo "  account      : ${ACCOUNT}"
echo "  chunks       : ${NUM_CHUNKS} x ${CHUNK_SIZE} videos = ${MAX_VIDEOS}"
echo "  data dir     : ${DATA_DIR}"
echo "  results dir  : ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}"
echo "  anchor x0 wt : ${ANCHOR_X0_WEIGHT}"
echo "  dry run      : ${DRY_RUN}"
echo "============================================================"
echo ""

count=0
for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
    start=$((chunk * CHUNK_SIZE))
    out_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}/chunk_${chunk}"
    job_name="smokex0_${DATASET_TAG}_${RUN_ID}_c${chunk}"

    # LORA_R8_TTA headline hyperparameters from submit_standard_1000v_chunked.sh.
    # Keep these identical to the headline cell so the ONLY changing variable
    # vs LORA_R8_TTA is ANCHOR_X0_WEIGHT.
    LORA_KV="LORA_RANK=8,LORA_ALPHA=16,LORA_TARGET_BLOCKS=all,NUM_STEPS=10,LEARNING_RATE=5.0e-5,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0,TARGET_FFN=0"

    _exec_or_dry sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${TIME_SHORT}" \
        --export="ALL,METHOD=lora,RUN_ID=${RUN_ID},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn,ANCHOR_X0_WEIGHT=${ANCHOR_X0_WEIGHT},${LORA_KV}" \
        "${SWEEP_SBATCH}"
    count=$((count + 1))
done

echo ""
echo "============================================================"
echo "Submitted ${count} job(s)."
echo ""
echo "Monitor:"
echo "  squeue -u \$USER | grep '^[0-9]* smokex0_'"
echo ""
echo "After completion:"
echo "  ls -lh ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}/chunk_0/"
echo "  jq '.results | length, (map(.psnr) | add/length), (map(.success) | all)' \\"
echo "      ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}/chunk_0/summary.json"
echo ""
echo "Compare against headline LORA_R8_TTA chunk_0 ΔPSNR; per"
echo "LITERATURE_tta_recipe_modifications_2026-06-12.md §3, Modification 1:"
echo "  - median |ΔPSNR| > 0.5 dB (either direction) → scale to 10-chunk λ-sweep"
echo "  - NaN grads OR |ΔPSNR| < 0.05 dB             → move on to Modification 2"
echo "============================================================"
