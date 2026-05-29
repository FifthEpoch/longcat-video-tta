#!/bin/bash
# ============================================================================
# Submit the 1000-video LONG-HORIZON sweep, chunked.
#
# Sibling of submit_standard_1000v_chunked.sh, but at NUM_FRAMES=76 (14 cond
# + 62 gen frames) instead of NUM_FRAMES=28. This is the regime where the
# pretrained model degrades enough for TTA to show measurable gains:
#   - Prior panda_longctx_1000v (NUM_FRAMES=76, single-job, timed out):
#       NoTTA  FVD 278.7   (vs 154.7 at standard horizon)
#       ADA    FVD 282.4
#     The single-job runs timed out so we never got clean numbers. This
#     chunked sweep is the proper redo.
#
# Methods (trimmed vs standard sweep):
#   1. NOTTA          (delta_a, delta_steps=0)         -- no-TTA baseline
#   2. ADA            (delta_a, S10/LR=5e-3 on Panda,  -- AdaSteer (our method)
#                      S5/LR=2.5e-3 on UCF)
#   3. LORA_R8_TTA    (lora,    rank=8, alpha=16,      -- LoRA TTA baseline
#                      all blocks, 10 steps, lr=5e-5)     (industry-default
#                                                          LoRA recipe)
#
# Why TinyLoRA is dropped at long horizon:
#   TinyLoRA's 20-step recipe takes ~16h/chunk at standard horizon. At
#   NUM_FRAMES=76 the per-video TTA+gen cost roughly doubles, putting every
#   chunk at the 24h preemption wall. The TinyLoRA story is already
#   established at standard horizon; long horizon is the "does AdaSteer beat
#   NoTTA and the LoRA TTA baseline when the model is degrading?" experiment.
#   Three methods cleanly answer that.
#
# Datasets:
#   Panda: datasets/panda_1000_480p   (1000-video Panda subset)
#   UCF  : datasets/ucf101_1000_480p  (1000-video UCF subset)
#
# Default scope is UCF only (the OOD-recovery experiment). Pass
# ONLY_DATASET=both to also run Panda long-horizon, or ONLY_DATASET=panda
# for Panda-only.
#
# Chunking: 10 chunks x 100 videos = 1000 total per (dataset, method).
# Wall-time bumped to 20h to absorb the longer rollout while staying under
# the 24h preemption wall.
#
# Submit (after `git pull` on the cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_longhorizon_1000v_chunked.sh
#
# Dry-run that prints sbatch lines without firing them:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_longhorizon_1000v_chunked.sh
#
# Both datasets:
#   ONLY_DATASET=both bash sweep_experiment/sbatch/submit_longhorizon_1000v_chunked.sh
#
# Subset of methods:
#   ONLY_METHODS="NOTTA ADA" bash sweep_experiment/sbatch/submit_longhorizon_1000v_chunked.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Per-chunk wall-time. 100 videos/chunk x ~12-min/video TTA+gen at long
# horizon = ~20h for the delta_a path. Set to 20h to leave buffer under
# the 24h preemption wall.
TIME_LONG="${TIME_LONG:-20:00:00}"

NUM_CHUNKS="${NUM_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"

# Frame geometry -- LONG horizon, 76-frame total (14 cond + 62 gen).
# Matches the prior panda_longctx_1000v setup (which timed out as a
# single-job run). TTA training context unchanged; only the rollout
# generation horizon is extended.
NUM_FRAMES="${NUM_FRAMES:-76}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

DRY_RUN="${DRY_RUN:-0}"
ONLY_DATASET="${ONLY_DATASET:-ucf}"
ONLY_METHODS="${ONLY_METHODS:-}"

count=0

_in_filter() {
    local needle="$1"
    [ -z "${ONLY_METHODS}" ] && return 0
    for m in ${ONLY_METHODS}; do
        if [ "${m}" = "${needle}" ]; then return 0; fi
    done
    return 1
}

_exec_or_dry() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY] $*"
        return 0
    fi
    "$@"
}

submit_sweep_chunks() {
    # Args: $1=dataset_tag (panda|ucf101)  $2=method (lora|delta_a)
    #       $3=run_id  $4=results_subdir  $5=data_dir  $6+=extra,KV pairs
    local dataset_tag="$1"; shift
    local method="$1"; shift
    local run_id="$1"; shift
    local results_subdir="$1"; shift
    local data_dir="$1"; shift
    local extra="$*"

    if ! _in_filter "${run_id}"; then return 0; fi

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local start=$((chunk * CHUNK_SIZE))
        local out_dir="${PROJECT_ROOT}/${results_subdir}/${run_id}/chunk_${chunk}"
        local job_name="t1kLH_${dataset_tag}_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${TIME_LONG}" \
            --export="ALL,METHOD=${method},RUN_ID=${run_id},SERIES_NAME=${dataset_tag}_1000v_longhorizon,DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn${extra}" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
}

submit_dataset() {
    # Args: $1=dataset_tag  $2=data_dir  $3=sweep_results_subdir
    #       $4=adasteer_steps  $5=adasteer_lr
    local dataset_tag="$1"
    local data_dir="$2"
    local sweep_subdir="$3"
    local ada_steps="$4"
    local ada_lr="$5"

    echo "############################################################"
    echo "Submitting dataset: ${dataset_tag}  (long horizon, ${NUM_FRAMES} frames)"
    echo "  data_dir     : ${data_dir}"
    echo "  sweep results: ${sweep_subdir}"
    echo "  AdaSteer     : steps=${ada_steps} lr=${ada_lr}"
    echo "############################################################"

    # 1) NoTTA
    submit_sweep_chunks "${dataset_tag}" "delta_a" "NOTTA" \
        "${sweep_subdir}" "${data_dir}" \
        ",DELTA_STEPS=0,DELTA_LR=${ada_lr}"

    # 2) AdaSteer
    submit_sweep_chunks "${dataset_tag}" "delta_a" "ADA" \
        "${sweep_subdir}" "${data_dir}" \
        ",DELTA_STEPS=${ada_steps},DELTA_LR=${ada_lr}"

    # 3) LoRA TTA baseline (rank=8, industry-default LoRA recipe).
    submit_sweep_chunks "${dataset_tag}" "lora" "LORA_R8_TTA" \
        "${sweep_subdir}" "${data_dir}" \
        ",LORA_RANK=8,LORA_ALPHA=16,LORA_TARGET_BLOCKS=all,NUM_STEPS=10,LEARNING_RATE=5.0e-5,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0,TARGET_FFN=0"

    echo ""
}

echo "============================================================"
echo "1000-video LONG-HORIZON sweep submission (${NUM_FRAMES} frames)"
echo "============================================================"
echo "  account     : ${ACCOUNT}"
echo "  num chunks  : ${NUM_CHUNKS}  x ${CHUNK_SIZE} videos = ${MAX_VIDEOS}"
echo "  num frames  : ${NUM_FRAMES}  (${NUM_COND_FRAMES} cond + $((NUM_FRAMES - NUM_COND_FRAMES)) gen)"
echo "  wall-time   : ${TIME_LONG}"
echo "  dry run     : ${DRY_RUN}"
echo "  only dataset: ${ONLY_DATASET}"
echo "  only methods: ${ONLY_METHODS:-<NOTTA ADA LORA_R8_TTA>}"
echo "============================================================"
echo ""

if [ "${ONLY_DATASET}" = "panda" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "panda" \
        "${PROJECT_ROOT}/datasets/panda_1000_480p" \
        "sweep_experiment/results/panda_1000v_longhorizon" \
        "10" "5.0e-3"
fi

if [ "${ONLY_DATASET}" = "ucf" ] || [ "${ONLY_DATASET}" = "ucf101" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "ucf101" \
        "${PROJECT_ROOT}/datasets/ucf101_1000_480p" \
        "sweep_experiment/results/ucf101_1000v_longhorizon" \
        "5" "2.5e-3"
fi

echo "============================================================"
echo "Submitted ${count} jobs."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* t1kLH_'"
echo ""
echo "After completion, merge chunks:"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/ucf101_1000v_longhorizon --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_longhorizon --recursive"
echo "============================================================"
