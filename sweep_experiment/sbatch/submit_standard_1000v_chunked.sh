#!/bin/bash
# ============================================================================
# Submit the 1000-video standard-horizon sweep, chunked.
#
# Sibling of submit_standard_2048v_chunked.sh, scoped to the pre-existing
# 1000-video datasets so we can get standard-horizon numbers faster while
# the 2048v build/sweep continues in parallel.
#
# Methods per dataset:
#   1. NOTTA          (delta_a, delta_steps=0)         -- no-TTA baseline
#   2. ADA            (delta_a, S10/LR=5e-3 on Panda,  -- AdaSteer (our method)
#                      S5/LR=2.5e-3 on UCF)
#   3. LORA_R8_TTA    (lora,    rank=8, alpha=16,      -- LoRA TTA baseline
#                      all blocks, 10 steps, lr=5e-5)     (best-validated LoRA
#                                                          recipe; see notes)
#   4. TL_BARE_R2     (tinylora, rank=2, n_tie=1,      -- TinyLoRA bare
#                      qkv_proj, all blocks, 20 steps,
#                      lr=1e-3)
#   5. TL_TIED_R2     (tinylora, rank=2, n_tie=48,     -- TinyLoRA tied
#                      qkv_proj, all blocks, 20 steps,
#                      lr=1e-3)
#
# LoRA TTA-baseline config selection (LORA_R8_TTA):
#   - Industry-default LoRA recipe (R=8, alpha=16, all qkv+proj blocks).
#   - Best PSNR among all LoRA variants tested on panda_1000_480p 100v
#     (PSNR=18.616 vs NoTTA 18.612; FVD=644.6 vs NoTTA 641.1 -- neutral).
#   - The only LoRA validated at full 1000v scale (long-context Panda):
#     FVD 282.4 vs NoTTA 278.7, VBench aesthetic 0.485 vs NoTTA 0.440.
#   - The previous LORA_R1 (lr=2e-4) variant was DROPPED: lr=2e-4 caused
#     catastrophic collapse at 20 steps (PSNR -1.17 dB, FVD +31 worse) in
#     the Apr 7 `panda_lora_lr_sanity` sweep, and the 5-step incarnation
#     was never validated. Not worth 20 chunks of paper-grade compute.
#
# Datasets:
#   Panda: datasets/panda_1000_480p   (the existing 1000-video dataset)
#   UCF  : datasets/ucf101_test_480p  (the existing 1000-video UCF subset)
#
# Chunking: 10 chunks x 100 videos = 1000 total per (dataset, method).
# Smaller chunk size (vs 2048v's 128) keeps each chunk under the prior
# 24h preemption wall even for TinyLoRA's 20-step recipe.
#
# COMPUTE_VBENCH=1 -> VBench++ runs inline per chunk. NO_SAVE_VIDEOS=0 ->
# generated mp4s retained for post-hoc analysis.
#
# Why a separate 1000v sweep when 2048v is already submitted/queued:
#   - The previous panda_1000v_s10_lr005_validation / ucf101_1000v_s5_lr0025_validation
#     runs TIMED OUT (single-job, no chunking).
#   - We want clean 1000v standard-horizon FVD/FID/VBench at N matched
#     across NoTTA / AdaSteer for the paper-grade comparison.
#   - 1000v finishes meaningfully sooner than 2048v and is what the
#     previous experimental record (200v -> 1000v -> 2048v) promotes to.
#
# Submit (after `git pull` on the cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_standard_1000v_chunked.sh
#
# Dry-run that prints sbatch lines without firing them:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_standard_1000v_chunked.sh
#
# Single dataset:
#   ONLY_DATASET=panda bash sweep_experiment/sbatch/submit_standard_1000v_chunked.sh
#   ONLY_DATASET=ucf   bash sweep_experiment/sbatch/submit_standard_1000v_chunked.sh
#
# Subset of methods:
#   ONLY_METHODS="NOTTA ADA" bash sweep_experiment/sbatch/submit_standard_1000v_chunked.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
TL_SBATCH="${TL_SBATCH:-delta_experiment/sbatch/run_tinylora.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Per-chunk wall-time. 100 videos/chunk x ~5-min/video TTA+gen = ~8h for the
# delta_a path, ~12h for tinylora's 20-step recipe.
TIME_SHORT="${TIME_SHORT:-12:00:00}"
TIME_TL="${TIME_TL:-16:00:00}"

NUM_CHUNKS="${NUM_CHUNKS:-10}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-1000}"

# Frame geometry — standard 28-frame horizon (matches 2048v sweep).
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
        local job_name="t1k_${dataset_tag}_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${TIME_SHORT}" \
            --export="ALL,METHOD=${method},RUN_ID=${run_id},SERIES_NAME=${dataset_tag}_1000v,DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn${extra}" \
            "${SWEEP_SBATCH}"
        count=$((count + 1))
    done
}

submit_tinylora_chunks() {
    local dataset_tag="$1"; shift
    local run_id="$1"; shift
    local data_dir="$1"; shift
    local results_subdir="$1"; shift
    local extra="$*"

    if ! _in_filter "${run_id}"; then return 0; fi

    for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
        local start=$((chunk * CHUNK_SIZE))
        local out_dir="${PROJECT_ROOT}/${results_subdir}/${run_id}/chunk_${chunk}"
        local job_name="t1k_${dataset_tag}_${run_id}_c${chunk}"

        _exec_or_dry sbatch \
            --account="${ACCOUNT}" \
            --job-name="${job_name}" \
            --time="${TIME_TL}" \
            --export="ALL,DATA_DIR=${data_dir},OUTPUT_DIR=${out_dir},NUM_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn${extra}" \
            "${TL_SBATCH}"
        count=$((count + 1))
    done
}

submit_dataset() {
    # Args: $1=dataset_tag  $2=data_dir  $3=sweep_results_subdir
    #       $4=tinylora_results_subdir  $5=adasteer_steps  $6=adasteer_lr
    local dataset_tag="$1"
    local data_dir="$2"
    local sweep_subdir="$3"
    local tl_subdir="$4"
    local ada_steps="$5"
    local ada_lr="$6"

    echo "############################################################"
    echo "Submitting dataset: ${dataset_tag}"
    echo "  data_dir     : ${data_dir}"
    echo "  sweep results: ${sweep_subdir}"
    echo "  TL results   : ${tl_subdir}"
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
    #    See header docstring for the config selection rationale.
    submit_sweep_chunks "${dataset_tag}" "lora" "LORA_R8_TTA" \
        "${sweep_subdir}" "${data_dir}" \
        ",LORA_RANK=8,LORA_ALPHA=16,LORA_TARGET_BLOCKS=all,NUM_STEPS=10,LEARNING_RATE=5.0e-5,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0,TARGET_FFN=0"

    # 4) TinyLoRA BARE
    submit_tinylora_chunks "${dataset_tag}" "TL_BARE_R2" "${data_dir}" "${tl_subdir}" \
        ",SVD_RANK=2,N_TIE=1,TARGET_PRESET=qkv_proj,TARGET_BLOCKS=all,TTA_STEPS=20,TTA_LR=1e-3"

    # 5) TinyLoRA TIED
    submit_tinylora_chunks "${dataset_tag}" "TL_TIED_R2" "${data_dir}" "${tl_subdir}" \
        ",SVD_RANK=2,N_TIE=48,TARGET_PRESET=qkv_proj,TARGET_BLOCKS=all,TTA_STEPS=20,TTA_LR=1e-3"

    echo ""
}

echo "============================================================"
echo "1000-video standard-horizon sweep submission"
echo "============================================================"
echo "  account     : ${ACCOUNT}"
echo "  num chunks  : ${NUM_CHUNKS}  x ${CHUNK_SIZE} videos = ${MAX_VIDEOS}"
echo "  dry run     : ${DRY_RUN}"
echo "  only dataset: ${ONLY_DATASET}"
echo "  only methods: ${ONLY_METHODS:-<all>}"
echo "============================================================"
echo ""

if [ "${ONLY_DATASET}" = "panda" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "panda" \
        "${PROJECT_ROOT}/datasets/panda_1000_480p" \
        "sweep_experiment/results/panda_1000v_standard" \
        "delta_experiment/results/tinylora_panda_1000v_standard" \
        "10" "5.0e-3"
fi

if [ "${ONLY_DATASET}" = "ucf" ] || [ "${ONLY_DATASET}" = "ucf101" ] || [ "${ONLY_DATASET}" = "both" ]; then
    submit_dataset "ucf101" \
        "${PROJECT_ROOT}/datasets/ucf101_1000_480p" \
        "sweep_experiment/results/ucf101_1000v_standard" \
        "delta_experiment/results/tinylora_ucf101_1000v_standard" \
        "5" "2.5e-3"
fi

echo "============================================================"
echo "Submitted ${count} jobs."
echo ""
echo "Monitor:  squeue -u \$USER | grep '^[0-9]* t1k_'"
echo ""
echo "After completion, merge chunks:"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/panda_1000v_standard --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/sweep_experiment/results/ucf101_1000v_standard --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/delta_experiment/results/tinylora_panda_1000v_standard --recursive"
echo "  python sweep_experiment/scripts/merge_chunks.py \\"
echo "      --results-dir ${PROJECT_ROOT}/delta_experiment/results/tinylora_ucf101_1000v_standard --recursive"
echo "============================================================"
