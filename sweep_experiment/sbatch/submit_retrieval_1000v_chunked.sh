#!/bin/bash
# ============================================================================
# Submit the 1000-video retrieval-augmented AdaSteer sweep, chunked
# (paper-grade headline retrieval evaluation).
#
# This is the focused 1000v counterpart to submit_2048v_retrieval_chunked.sh.
# Three differences vs the 2048v template:
#   1. NUM_CHUNKS=10  CHUNK_SIZE=100/94  MAX_VIDEOS=1000/932
#      (matches the chunk count of panda_1000v_standard /
#       ucf101_932v_standard so the retrieval variants line up
#       1-for-1 against the non-retrieval headline numbers).
#   2. eval-set != retrieval-pool. The 2048v template used the same dir
#      for both ("self-retrieval"); here:
#        Panda  : eval = panda_1000_480p   pool = panda_2048_480p
#        UCF    : eval = ucf101_std_480p   pool = ucf101_pool_max (26K)
#      Same-source exclusion (already patched into common.py via the
#      v2/v3 deploys) prevents the eval clip from matching itself in
#      the pool.
#   3. Job tag prefix `t1kr_*` (vs `t2kr_*`) so squeue/sacct can
#      distinguish the two retrieval sweeps cleanly.
#
# Methods per dataset (all AdaSteer with batch-level neighbours):
#   1. K5_RAND   batch_videos=5,  batch_method=random
#   2. K10_RAND  batch_videos=10, batch_method=random
#   3. K5_SIM    batch_videos=5,  batch_method=similarity
#   4. K10_SIM   batch_videos=10, batch_method=similarity
#
# Step-budget semantics (post-fix in _optimize_delta_a_batch):
#   each video in the training batch receives `delta_steps` gradient updates.
#   Total optimiser steps per eval video = delta_steps * K.
#
# Wall-time per chunk (similarity/random use the same K-cost model):
#   K=5  : ~10 h at 100 vids/chunk  -> 14 h with buffer
#   K=10 : ~20 h at 100 vids/chunk  -> 22 h with buffer
#
# Each chunk writes:
#   sweep_experiment/results/<dataset>_<size>_retrieval/<RUN_ID>/chunk_<i>/
# After completion, merge with sweep_experiment/scripts/merge_chunks.py.
#
# Prerequisites (one-time per cluster login env):
#   1) Pre-compute pool caption embeddings (one job per pool, ~30 min):
#        sbatch --account=torch_pr_36_mren \
#            --export=ALL,POOL_DIR=/scratch/wc3013/longcat-video-tta/datasets/panda_2048_480p \
#            delta_experiment/sbatch/precompute_pool_embeddings.sbatch
#        sbatch --account=torch_pr_36_mren \
#            --export=ALL,POOL_DIR=/scratch/wc3013/longcat-video-tta/datasets/ucf101_pool_max \
#            delta_experiment/sbatch/precompute_pool_embeddings.sbatch
#      Verify each pool now has caption_embeddings.npy + .json.
#
#   2) Fix sentence-transformers ImportError if it returns
#      ('cannot import name is_nltk_available'):
#        pip install --no-deps --force-reinstall "sentence-transformers==2.7.0"
#
#   3) Confirm the v2 + v3 common.py patches are in place (these are what
#      enable same-source exclusion + cached embedding loading):
#        grep -nE '_entry_source_id|caption_embeddings\.npy' \
#            delta_experiment/scripts/common.py | head
#
# Submit (after `git pull` on the cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_retrieval_1000v_chunked.sh
#
# Dry-run (prints sbatch lines without firing them):
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_retrieval_1000v_chunked.sh
#
# Submit only one dataset:
#   ONLY_DATASET=panda bash sweep_experiment/sbatch/submit_retrieval_1000v_chunked.sh
#   ONLY_DATASET=ucf   bash sweep_experiment/sbatch/submit_retrieval_1000v_chunked.sh
#
# Submit only specific run IDs (subset of K5_RAND K10_RAND K5_SIM K10_SIM):
#   ONLY_METHODS="K5_RAND K5_SIM" bash sweep_experiment/sbatch/submit_retrieval_1000v_chunked.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Per-K wall-time. K=10 needs ~2x K=5; chunks are smaller than 2048v
# (100 vs 128 videos) so absolute wall-times are slightly lower, but we
# keep a buffer for the H200 sharing.
TIME_K5="${TIME_K5:-14:00:00}"
TIME_K10="${TIME_K10:-22:00:00}"

# Chunking. The defaults below give exactly 10 chunks at 100 vids each
# for Panda (1000) and slightly oversize for UCF (94 vids/chunk * 10 =
# 940 attempts capping at 932 -- the runner caps at MAX_VIDEOS).
NUM_CHUNKS="${NUM_CHUNKS:-10}"

# Frame geometry -- standard 28-frame horizon (matches the headline
# panda_1000v_standard and ucf101_932v_standard sweeps so retrieval
# results line up 1-for-1 with non-retrieval baselines).
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
    #   $1  dataset_tag       (panda|ucf101)
    #   $2  run_id            (K5_RAND|K10_RAND|K5_SIM|K10_SIM)
    #   $3  results_subdir
    #   $4  data_dir          (eval set)
    #   $5  pool_dir          (retrieval pool, may differ from data_dir)
    #   $6  wall_time
    #   $7  delta_steps
    #   $8  delta_lr
    #   $9  batch_videos      K
    #   $10 batch_method      random|similarity
    #   $11 max_videos
    #   $12 chunk_size
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
    local max_videos="${11}"
    local chunk_size="${12}"

    if ! _in_filter "${run_id}"; then return 0; fi

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local start=$((chunk * chunk_size))
        # Don't fire chunks whose start is already past the eval set.
        if [ "${start}" -ge "${max_videos}" ]; then break; fi

        local out_dir="${PROJECT_ROOT}/${results_subdir}/${run_id}/chunk_${chunk}"
        local job_name="t1kr_${dataset_tag}_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${wall_time}" \
            --export="ALL,METHOD=delta_a,RUN_ID=${run_id},SERIES_NAME=${dataset_tag}_$([ "${dataset_tag}" = "panda" ] && echo "1000v_retrieval" || echo "932v_retrieval"),DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},RETRIEVAL_POOL_DIR=${pool_dir},BATCH_VIDEOS=${batch_videos},BATCH_METHOD=${batch_method},DELTA_STEPS=${delta_steps},DELTA_LR=${delta_lr},MAX_VIDEOS=${max_videos},START_VIDEO_IDX=${start},CHUNK_SIZE=${chunk_size},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
}

submit_dataset() {
    # Args:
    #   $1 dataset_tag
    #   $2 data_dir          (eval set, e.g. panda_1000_480p)
    #   $3 pool_dir          (retrieval pool, e.g. panda_2048_480p; may differ)
    #   $4 sweep_results_subdir
    #   $5 delta_steps
    #   $6 delta_lr
    #   $7 max_videos        (1000 panda, 932 ucf)
    #   $8 chunk_size        (100 panda, 94 ucf -- 10 chunks total)
    local dataset_tag="$1"
    local data_dir="$2"
    local pool_dir="$3"
    local sweep_subdir="$4"
    local delta_steps="$5"
    local delta_lr="$6"
    local max_videos="$7"
    local chunk_size="$8"

    echo "############################################################"
    echo "Submitting retrieval sweep for dataset: ${dataset_tag}"
    echo "  data_dir       : ${data_dir}    (eval set, ${max_videos} videos)"
    echo "  retrieval pool : ${pool_dir}"
    echo "  results subdir : ${sweep_subdir}"
    echo "  AdaSteer base  : steps=${delta_steps} lr=${delta_lr}"
    echo "  chunking       : ${NUM_CHUNKS} chunks x ${chunk_size} videos"
    echo "############################################################"

    if [ ! -d "${data_dir}" ]; then
        echo "  WARN: eval data_dir does not exist: ${data_dir}"
        echo "        skipping submission for ${dataset_tag}"
        return 0
    fi
    if [ ! -d "${pool_dir}" ]; then
        echo "  WARN: retrieval pool does not exist: ${pool_dir}"
        echo "        skipping submission for ${dataset_tag}"
        return 0
    fi
    if [ ! -f "${pool_dir}/caption_embeddings.npy" ]; then
        echo "  WARN: ${pool_dir}/caption_embeddings.npy not present."
        echo "        Similarity-method retrieval will fall back to encoding"
        echo "        captions on-the-fly per TTA job (~30-60 s overhead per"
        echo "        job). Recommended: pre-compute embeddings first via"
        echo "        delta_experiment/sbatch/precompute_pool_embeddings.sbatch."
    fi

    submit_retrieval_chunks "${dataset_tag}" "K5_RAND"  "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K5}"  "${delta_steps}" "${delta_lr}" \
        5  "sequential" "${max_videos}" "${chunk_size}"

    submit_retrieval_chunks "${dataset_tag}" "K10_RAND" "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K10}" "${delta_steps}" "${delta_lr}" \
        10 "sequential" "${max_videos}" "${chunk_size}"

    submit_retrieval_chunks "${dataset_tag}" "K5_SIM"   "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K5}"  "${delta_steps}" "${delta_lr}" \
        5  "similarity" "${max_videos}" "${chunk_size}"

    submit_retrieval_chunks "${dataset_tag}" "K10_SIM"  "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K10}" "${delta_steps}" "${delta_lr}" \
        10 "similarity" "${max_videos}" "${chunk_size}"

    echo ""
}

echo "============================================================"
echo "1000-video retrieval-augmented AdaSteer submission"
echo "============================================================"
echo "  account     : ${ACCOUNT}"
echo "  num chunks  : ${NUM_CHUNKS}"
echo "  K=5 wall    : ${TIME_K5}"
echo "  K=10 wall   : ${TIME_K10}"
echo "  dry run     : ${DRY_RUN}"
echo "  only dataset: ${ONLY_DATASET}"
echo "  only methods: ${ONLY_METHODS:-<all>}"
echo "============================================================"
echo ""

if [ "${ONLY_DATASET}" = "panda" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "panda" \
        "${PROJECT_ROOT}/datasets/panda_1000_480p" \
        "${PROJECT_ROOT}/datasets/panda_2048_480p" \
        "sweep_experiment/results/panda_1000v_retrieval" \
        10 "5.0e-3" \
        1000 100
fi

if [ "${ONLY_DATASET}" = "ucf" ] || [ "${ONLY_DATASET}" = "ucf101" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "ucf101" \
        "${PROJECT_ROOT}/datasets/ucf101_std_480p" \
        "${PROJECT_ROOT}/datasets/ucf101_pool_max" \
        "sweep_experiment/results/ucf101_932v_retrieval" \
        5 "2.5e-3" \
        932 94
fi

echo "============================================================"
echo "Submitted ${count} jobs."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* t1kr_'"
echo ""
echo "After completion, merge chunks:"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_retrieval --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/ucf101_932v_retrieval --recursive"
echo "============================================================"
