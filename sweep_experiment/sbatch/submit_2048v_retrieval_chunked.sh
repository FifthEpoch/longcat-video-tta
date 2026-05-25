#!/bin/bash
# ============================================================================
# Submit the 2048-video retrieval-augmented AdaSteer sweep, chunked.
#
# Methods per dataset (all AdaSteer with batch-level neighbours):
#   1. K5_RAND   batch_videos=5,  batch_method=random
#   2. K10_RAND  batch_videos=10, batch_method=random
#   3. K5_SIM    batch_videos=5,  batch_method=similarity
#   4. K10_SIM   batch_videos=10, batch_method=similarity
#
# Step-budget semantics (post-fix in _optimize_delta_a_batch):
#   each video in the training batch receives `delta_steps` gradient updates.
#   total optimiser steps per eval video = delta_steps * K.
#
# Wall-time per chunk (K=10 chunks roughly double K=5 cost):
#   K=5  : ~10h at 128 vids/chunk -> 14h with buffer
#   K=10 : ~20h at 128 vids/chunk -> 22h with buffer (resumable via checkpoint)
#
# Each chunk writes:
#   sweep_experiment/results/<dataset>_2048v_retrieval/<RUN_ID>/chunk_<i>/
# After completion merge with sweep_experiment/scripts/merge_chunks.py.
#
# Prerequisites (one-time per cluster login env):
#   1) Fix sentence-transformers ImportError (only required for *_SIM runs):
#        pip install --no-deps --force-reinstall "sentence-transformers==2.7.0"
#      Smoke-test:
#        python -c "from sentence_transformers import SentenceTransformer; \
#          m=SentenceTransformer('all-MiniLM-L6-v2'); \
#          print('OK', m.get_sentence_embedding_dimension())"
#   2) Ensure datasets are built:
#        sbatch datasets/build_panda_2048.sbatch
#        sbatch datasets/build_ucf101_2048.sbatch
#
# Submit (after `git pull` on the cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_2048v_retrieval_chunked.sh
#
# Dry-run (prints sbatch lines without firing them):
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_2048v_retrieval_chunked.sh
#
# Submit only one dataset:
#   ONLY_DATASET=panda bash sweep_experiment/sbatch/submit_2048v_retrieval_chunked.sh
#   ONLY_DATASET=ucf   bash sweep_experiment/sbatch/submit_2048v_retrieval_chunked.sh
#
# Submit only specific run IDs (subset of K5_RAND K10_RAND K5_SIM K10_SIM):
#   ONLY_METHODS="K5_RAND K5_SIM" bash sweep_experiment/sbatch/submit_2048v_retrieval_chunked.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Per-K wall-time. K=10 needs ~2x K=5; we still chunk at 128 videos but allow
# more wall, and rely on the run_delta_a.py checkpoint to resume on preempt.
TIME_K5="${TIME_K5:-14:00:00}"
TIME_K10="${TIME_K10:-22:00:00}"

NUM_CHUNKS="${NUM_CHUNKS:-16}"
CHUNK_SIZE="${CHUNK_SIZE:-128}"
MAX_VIDEOS="${MAX_VIDEOS:-2048}"

# Frame geometry — standard 28-frame horizon (matches headline sweep).
NUM_FRAMES="${NUM_FRAMES:-28}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

DRY_RUN="${DRY_RUN:-0}"
ONLY_DATASET="${ONLY_DATASET:-both}"
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

submit_retrieval_chunks() {
    # Args:
    #   $1 dataset_tag       (panda|ucf101)
    #   $2 run_id            (K5_RAND|K10_RAND|K5_SIM|K10_SIM)
    #   $3 results_subdir
    #   $4 data_dir
    #   $5 pool_dir
    #   $6 wall_time
    #   $7 delta_steps       AdaSteer steps per video
    #   $8 delta_lr
    #   $9 batch_videos      K
    #  $10 batch_method      random|similarity
    local dataset_tag="$1"
    local run_id="$2"
    local results_subdir="$3"
    local data_dir="$4"
    local pool_dir="$5"
    local wall_time="$6"
    local delta_steps="$7"
    local delta_lr="$8"
    local batch_videos="$9"
    local batch_method="${10}"

    if ! _in_filter "${run_id}"; then return 0; fi

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local start=$((chunk * CHUNK_SIZE))
        local out_dir="${PROJECT_ROOT}/${results_subdir}/${run_id}/chunk_${chunk}"
        local job_name="t2kr_${dataset_tag}_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${wall_time}" \
            --export="ALL,METHOD=delta_a,RUN_ID=${run_id},SERIES_NAME=${dataset_tag}_2048v_retrieval,DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},RETRIEVAL_POOL_DIR=${pool_dir},BATCH_VIDEOS=${batch_videos},BATCH_METHOD=${batch_method},DELTA_STEPS=${delta_steps},DELTA_LR=${delta_lr},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
}

submit_dataset() {
    # Args:
    #   $1 dataset_tag
    #   $2 data_dir (also used as retrieval pool)
    #   $3 sweep_results_subdir
    #   $4 delta_steps
    #   $5 delta_lr
    local dataset_tag="$1"
    local data_dir="$2"
    local sweep_subdir="$3"
    local delta_steps="$4"
    local delta_lr="$5"
    local pool_dir="${data_dir}"

    echo "############################################################"
    echo "Submitting retrieval sweep for dataset: ${dataset_tag}"
    echo "  data_dir      : ${data_dir}"
    echo "  retrieval pool: ${pool_dir}"
    echo "  results subdir: ${sweep_subdir}"
    echo "  AdaSteer base : steps=${delta_steps} lr=${delta_lr}"
    echo "############################################################"

    submit_retrieval_chunks "${dataset_tag}" "K5_RAND"  "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K5}"  "${delta_steps}" "${delta_lr}" 5  "random"

    submit_retrieval_chunks "${dataset_tag}" "K10_RAND" "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K10}" "${delta_steps}" "${delta_lr}" 10 "random"

    submit_retrieval_chunks "${dataset_tag}" "K5_SIM"   "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K5}"  "${delta_steps}" "${delta_lr}" 5  "similarity"

    submit_retrieval_chunks "${dataset_tag}" "K10_SIM"  "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K10}" "${delta_steps}" "${delta_lr}" 10 "similarity"

    echo ""
}

echo "============================================================"
echo "2048-video retrieval-augmented AdaSteer submission"
echo "============================================================"
echo "  account     : ${ACCOUNT}"
echo "  num chunks  : ${NUM_CHUNKS}  x ${CHUNK_SIZE} videos = ${MAX_VIDEOS}"
echo "  K=5 wall    : ${TIME_K5}"
echo "  K=10 wall   : ${TIME_K10}"
echo "  dry run     : ${DRY_RUN}"
echo "  only dataset: ${ONLY_DATASET}"
echo "  only methods: ${ONLY_METHODS:-<all>}"
echo "============================================================"
echo ""

if [ "${ONLY_DATASET}" = "panda" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "panda" \
        "${PROJECT_ROOT}/datasets/panda_2048_480p" \
        "sweep_experiment/results/panda_2048v_retrieval" \
        10 "5.0e-3"
fi

if [ "${ONLY_DATASET}" = "ucf" ] || [ "${ONLY_DATASET}" = "ucf101" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "ucf101" \
        "${PROJECT_ROOT}/datasets/ucf101_2048_480p" \
        "sweep_experiment/results/ucf101_2048v_retrieval" \
        5 "2.5e-3"
fi

echo "============================================================"
echo "Submitted ${count} jobs."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* t2kr_'"
echo ""
echo "After completion, merge chunks:"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/panda_2048v_retrieval --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/ucf101_2048v_retrieval --recursive"
echo "============================================================"
