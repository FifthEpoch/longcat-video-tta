#!/bin/bash
# ============================================================================
# Submit the retrieval-augmented AdaSteer sweep WITHOUT TTA-time captions
# (Panda 1000v ablation against the 25K segment pool).
#
# This script combines two existing knobs:
#   1. Batch-level retrieval — eval video + K-1 nearest (SIM) or sequential
#      (RAND) neighbours sampled from a retrieval pool into the AdaSteer
#      training batch (existing in submit_retrieval_1000v_chunked.sh).
#   2. --tta-disable-caption — drops captions during the TTA training step
#      only; inference / generation still receives the real caption from
#      the eval set (added in commit 16c1532, propagated via
#      TTA_DISABLE_CAPTION=1 in run_sweep.sbatch -> run_delta_a.py).
#
# Question being asked: does retrieval-augmented batch TTA still help (or
# hurt) when the prompt is dropped at TTA time? The headline retrieval
# sweep (submit_retrieval_1000v_chunked.sh, run IDs K{5,10}_{SIM,RAND})
# keeps captions during TTA; this sibling sweep keeps everything else
# identical but blanks all TTA-time captions (eval video AND every
# retrieved neighbour — both flow through the same per-entry encode_prompt
# call in run_delta_a.py:872 inside the for-loop at line 848). Compare
# the two side-by-side to attribute any gain/loss to the neighbour-caption
# signal vs the neighbour-video signal alone.
#
# Why NOTTA is NOT in this sweep:
#   NOTTA never runs a TTA step, so dropping the TTA-time caption is a
#   no-op (NOTTA_NOPROMPT would be byte-identical to NOTTA). Reuse the
#   existing NOTTA row from the headline `panda_1000v_standard` table
#   when constructing the paper table for this ablation.
#
# Methods (4):
#   1. K5_RAND_NOPROMPT   batch_videos=5,  batch_method=sequential, no TTA caption
#   2. K10_RAND_NOPROMPT  batch_videos=10, batch_method=sequential, no TTA caption
#   3. K5_SIM_NOPROMPT    batch_videos=5,  batch_method=similarity,  no TTA caption
#   4. K10_SIM_NOPROMPT   batch_videos=10, batch_method=similarity,  no TTA caption
#
# Default scope: Panda 1000v ONLY (4 methods × 1 dataset × 10 chunks = 40 jobs).
# UCF dispatch is wired but opt-in via ONLY_DATASET=ucf or ONLY_DATASET=both
# (UCF was already shown to be a poor retrieval testbed — class-block
# layout, see INDEX.md headline table — so we don't run it by default).
#
# Series dirs (SAME as the headline retrieval submitter — the _NOPROMPT
# methods land alongside the existing K{5,10}_{SIM,RAND} headline rows;
# for Panda the dir is currently empty because the headline retrieval
# sweep against the 25K pool is "step 4" of the in-flight pipeline):
#   sweep_experiment/results/panda_1000v_retrieval/<RUN_ID>/chunk_<i>/
#   sweep_experiment/results/ucf101_932v_retrieval/<RUN_ID>/chunk_<i>/   (UCF only)
#
# Prerequisites for the default Panda submission:
#   1) Step 2 — Panda 25K segment-pool build complete:
#        ls /scratch/$USER/longcat-video-tta/datasets/panda_segment_pool/videos/*.mp4 | wc -l
#      should be ≈ 22-25K (currently 3,302; relaunch pending against the
#      patched scripts/build_panda_segment_pool.py from commit 5d565d4).
#   2) Step 3 — caption embeddings precomputed for the pool:
#        sbatch --account=torch_pr_36_mren \
#            --export=ALL,POOL_DIR=/scratch/$USER/longcat-video-tta/datasets/panda_segment_pool \
#            delta_experiment/sbatch/precompute_pool_embeddings.sbatch
#      Verify `caption_embeddings.npy` + `.json` are present in the pool.
#   3) Fix sentence-transformers ImportError if it returns
#      ('cannot import name is_nltk_available'):
#        pip install --no-deps --force-reinstall "sentence-transformers==2.7.0"
#
# Submit (after `git pull` on the cluster, when steps 2+3 above are done):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh
#
# Dry-run that prints sbatch lines without firing them:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh
#
# Smoke-test single chunk × single method (recommended before firing the
# full 40-job sweep — verifies the 25K pool + embeddings are wired and
# the no-prompt flag reaches the runner):
#   DRY_RUN=0 NUM_CHUNKS=1 ONLY_DATASET=panda ONLY_METHODS="K5_SIM_NOPROMPT" \
#       bash sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh
#
# Test against the current 3.3K pool first (instead of waiting for the
# 25K relaunch):
#   PANDA_POOL=/scratch/$USER/longcat-video-tta/datasets/panda_segment_pool \
#       NUM_CHUNKS=1 ONLY_METHODS="K5_SIM_NOPROMPT" \
#       bash sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh
#   (PANDA_POOL default already points at panda_segment_pool — this is
#   just an explicit override for clarity.)
#
# Submit both datasets (Panda + UCF):
#   ONLY_DATASET=both bash sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh
#
# Subset of methods:
#   ONLY_METHODS="K5_RAND_NOPROMPT K5_SIM_NOPROMPT" \
#       bash sweep_experiment/sbatch/submit_retrieval_1000v_noprompt.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Per-K wall-time. Match the headline retrieval submitter exactly
# (no-prompt does not change the per-step cost — same delta_steps, same
# K-cost model — so wall budget is identical).
TIME_K5="${TIME_K5:-14:00:00}"
TIME_K10="${TIME_K10:-22:00:00}"

# Chunking. Matches submit_retrieval_1000v_chunked.sh (10 chunks × 100
# videos = 1000 for Panda; 10 chunks × 94 = 940 -> capped at 932 for UCF).
NUM_CHUNKS="${NUM_CHUNKS:-10}"

# Frame geometry — standard 28-frame horizon, identical to the headline
# retrieval submit script.
NUM_FRAMES="${NUM_FRAMES:-28}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

DRY_RUN="${DRY_RUN:-0}"
# Default scope is Panda only — see header. UCF dispatch is wired but
# opt-in (ONLY_DATASET=ucf or ONLY_DATASET=both).
ONLY_DATASET="${ONLY_DATASET:-panda}"
ONLY_METHODS="${ONLY_METHODS:-}"

# Pool overrides. The Panda default is `panda_segment_pool` — i.e. the
# 25K-target pool destination (currently 3,302 segments; will be ~22-25K
# after the in-flight step 2 relaunch + step 3 embedding precompute). The
# user can override with PANDA_POOL=... to point at any other pool dir
# (e.g. `panda_2048_480p` for a back-compat A/B against the prior 2K pool).
PANDA_POOL="${PANDA_POOL:-${PROJECT_ROOT}/datasets/panda_segment_pool}"
UCF_POOL="${UCF_POOL:-${PROJECT_ROOT}/datasets/ucf101_pool_max}"

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
    #   $2  run_id            (K5_RAND_NOPROMPT|K10_RAND_NOPROMPT|K5_SIM_NOPROMPT|K10_SIM_NOPROMPT)
    #   $3  results_subdir
    #   $4  data_dir          (eval set)
    #   $5  pool_dir          (retrieval pool, may differ from data_dir)
    #   $6  wall_time
    #   $7  delta_steps
    #   $8  delta_lr
    #   $9  batch_videos      K
    #   $10 batch_method      sequential|similarity
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
        local job_name="t1krnp_${dataset_tag}_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${wall_time}" \
            --export="ALL,METHOD=delta_a,RUN_ID=${run_id},SERIES_NAME=${dataset_tag}_$([ "${dataset_tag}" = "panda" ] && echo "1000v_retrieval" || echo "932v_retrieval"),DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},RETRIEVAL_POOL_DIR=${pool_dir},BATCH_VIDEOS=${batch_videos},BATCH_METHOD=${batch_method},DELTA_STEPS=${delta_steps},DELTA_LR=${delta_lr},MAX_VIDEOS=${max_videos},START_VIDEO_IDX=${start},CHUNK_SIZE=${chunk_size},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn,TTA_DISABLE_CAPTION=1" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
}

submit_dataset() {
    # Args:
    #   $1 dataset_tag
    #   $2 data_dir          (eval set)
    #   $3 pool_dir          (retrieval pool)
    #   $4 sweep_results_subdir
    #   $5 delta_steps
    #   $6 delta_lr
    #   $7 max_videos
    #   $8 chunk_size
    local dataset_tag="$1"
    local data_dir="$2"
    local pool_dir="$3"
    local sweep_subdir="$4"
    local delta_steps="$5"
    local delta_lr="$6"
    local max_videos="$7"
    local chunk_size="$8"

    echo "############################################################"
    echo "Submitting retrieval+NOPROMPT sweep for dataset: ${dataset_tag}"
    echo "  data_dir       : ${data_dir}    (eval set, ${max_videos} videos)"
    echo "  retrieval pool : ${pool_dir}"
    echo "  results subdir : ${sweep_subdir}"
    echo "  AdaSteer base  : steps=${delta_steps} lr=${delta_lr}"
    echo "  chunking       : ${NUM_CHUNKS} chunks x ${chunk_size} videos"
    echo "  TTA caption    : DISABLED (TTA_DISABLE_CAPTION=1)"
    echo "############################################################"

    # Skip path-existence checks during DRY_RUN so the script can be
    # smoke-validated from a laptop. Cluster runs still get the WARN +
    # skip behaviour from the headline retrieval submitter.
    if [ "${DRY_RUN}" != "1" ]; then
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
    fi

    submit_retrieval_chunks "${dataset_tag}" "K5_RAND_NOPROMPT"  "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K5}"  "${delta_steps}" "${delta_lr}" \
        5  "sequential" "${max_videos}" "${chunk_size}"

    submit_retrieval_chunks "${dataset_tag}" "K10_RAND_NOPROMPT" "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K10}" "${delta_steps}" "${delta_lr}" \
        10 "sequential" "${max_videos}" "${chunk_size}"

    submit_retrieval_chunks "${dataset_tag}" "K5_SIM_NOPROMPT"   "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K5}"  "${delta_steps}" "${delta_lr}" \
        5  "similarity" "${max_videos}" "${chunk_size}"

    submit_retrieval_chunks "${dataset_tag}" "K10_SIM_NOPROMPT"  "${sweep_subdir}" \
        "${data_dir}" "${pool_dir}" "${TIME_K10}" "${delta_steps}" "${delta_lr}" \
        10 "similarity" "${max_videos}" "${chunk_size}"

    echo ""
}

echo "============================================================"
echo "Retrieval+NOPROMPT AdaSteer submission (Panda 1000v default)"
echo "============================================================"
echo "  account     : ${ACCOUNT}"
echo "  num chunks  : ${NUM_CHUNKS}"
echo "  K=5 wall    : ${TIME_K5}"
echo "  K=10 wall   : ${TIME_K10}"
echo "  dry run     : ${DRY_RUN}"
echo "  only dataset: ${ONLY_DATASET}"
echo "  only methods: ${ONLY_METHODS:-<all>}"
echo "  panda pool  : ${PANDA_POOL}"
echo "  ucf pool    : ${UCF_POOL}"
echo "  TTA caption : DISABLED (TTA_DISABLE_CAPTION=1)"
echo "============================================================"
echo ""

if [ "${ONLY_DATASET}" = "panda" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "panda" \
        "${PROJECT_ROOT}/datasets/panda_1000_480p" \
        "${PANDA_POOL}" \
        "sweep_experiment/results/panda_1000v_retrieval" \
        10 "5.0e-3" \
        1000 100
fi

if [ "${ONLY_DATASET}" = "ucf" ] || [ "${ONLY_DATASET}" = "ucf101" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "ucf101" \
        "${PROJECT_ROOT}/datasets/ucf101_std_480p" \
        "${UCF_POOL}" \
        "sweep_experiment/results/ucf101_932v_retrieval" \
        5 "2.5e-3" \
        932 94
fi

echo "============================================================"
echo "Submitted ${count} jobs."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* t1krnp_'"
echo ""
echo "After completion, merge chunks (the _NOPROMPT methods land in the"
echo "same series dir as the headline retrieval runs — the merge command"
echo "is the same one used for the headline retrieval sweep):"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_retrieval --recursive"
if [ "${ONLY_DATASET}" = "both" ] || [ "${ONLY_DATASET}" = "ucf" ] || [ "${ONLY_DATASET}" = "ucf101" ]; then
    echo "  python sweep_experiment/scripts/merge_chunks.py \\"
    echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/ucf101_932v_retrieval --recursive"
fi
echo ""
echo "Then build the paper table for the retrieval×NOPROMPT ablation"
echo "(reuse the existing NOTTA row from panda_1000v_standard — NOTTA"
echo "does not run TTA so dropping the TTA caption is a no-op):"
echo "  python scripts/build_paper_tables.py --regime panda_std \\"
echo "      --output sweep_experiment/reports/paper_tables/\$(date +%Y-%m-%d)_panda_retrieval_noprompt.md"
echo "============================================================"
