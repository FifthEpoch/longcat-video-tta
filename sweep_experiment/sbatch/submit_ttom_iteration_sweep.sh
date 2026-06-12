#!/usr/bin/env bash
# ============================================================================
# TTOM iteration-saturation sweep (Track D Wave D2)
#
# Sweeps the TTA-step count across {10, 20, 40, 80, 160} for three methods
# (ADA / LORA_R8_TTA / TL_BARE_R2) on Panda 1000v chunk_0 (videos 0-99).
# Plots ΔPSNR / ΔLPIPS / ΔFVD vs iteration count to test whether our setting
# reproduces TTOM's (Qu et al., ICLR 2026) saturate-then-degrade curve or
# shows a distinct noise-floor-limited pattern.
#
# Decision rule baked into PAPER_FRAGMENT_ttom_positioning_2026-06-12.md:
#   - Crossover at high-iter end (ΔPSNR/ΔLPIPS revert toward baseline as
#     N grows) -> shared mechanism with TTOM (over-optimization saturation).
#   - Monotonic-flat curve at the per-video noise floor -> distinct mechanism
#     from TTOM (per-video reconstructive TTA is rate-limited differently).
# Either outcome is paper-defensible and pre-empts the obvious reviewer
# challenge "did you just not run enough iterations?"
#
# Subset choice: chunk_0 (Panda 1000v, videos 0-99) matches the D1 smoke-test
# (`submit_smoke_x0_loss.sh`) so the two waves are directly comparable; the
# pre-existing per-video analysis at
# sweep_experiment/reports/per_video_analysis/2026-06-09/ also covers this
# exact set, so winner/loser overlap is a free secondary read.
#
# Hyperparameters: every knob besides the TTA-step count is frozen at the
# headline Panda recipes from `submit_standard_1000v_chunked.sh` so the ONLY
# changing variable is the TTA-step count. Specifically:
#   ADA          : DELTA_LR=5.0e-3                        (DELTA_STEPS varies)
#   LORA_R8_TTA  : LORA_RANK=8, LORA_ALPHA=16, LORA_TARGET_BLOCKS=all,
#                  LEARNING_RATE=5.0e-5, WARMUP_STEPS=3, WEIGHT_DECAY=0.01,
#                  MAX_GRAD_NORM=10.0, TARGET_FFN=0       (NUM_STEPS varies)
#   TL_BARE_R2   : SVD_RANK=2, N_TIE=1, TARGET_PRESET=qkv_proj,
#                  TARGET_BLOCKS=all, TTA_LR=1e-3         (TTA_STEPS varies)
#
# Output dirs (alongside the headline cells so paper-table builders find
# them naturally — same series dirs as ADA / LORA_R8_TTA / TL_BARE_R2):
#   sweep_experiment/results/panda_1000v_standard/{ADA,LORA_R8_TTA}_TTA<N>/chunk_0/
#   delta_experiment/results/tinylora_panda_1000v_standard/TL_BARE_R2_TTA<N>/chunk_0/
#
# Total: 3 methods × 5 tta-steps × 1 chunk = 15 jobs.
# Wallclock per job: roughly ~1-3 GPU h × tta-step multiplier (10 → ~baseline
# 2 GPU h on H200 matching the `submit_smoke_x0_loss.sh` estimate; 160 → up
# to ~16x the TTA-time component, capped by the 24h cluster preemption wall).
# Total GPU-h: ~125 GPU-h serial; with the 2-way h200 cap ~24-48 wallclock h.
#
# Refs:
#   sweep_experiment/reports/PAPER_FRAGMENT_ttom_positioning_2026-06-12.md
#       (§"Suggested control" — locked spec for this wave)
#   sweep_experiment/reports/RUNBOOK_friday_morning_2026-06-12.md §4 D2
#       (submission authorisation; wave D2 of Track D)
#   TTOM (Qu et al., ICLR 2026 — https://openreview.net/pdf?id=wqCwcTZsrv;
#       arXiv 2510.07940 — Table 8 motivates the high-iter end of the grid)
#
# ----------------------------------------------------------------------------
# Submit (after `git pull` on the cluster):
#   cd /scratch/wc3013/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh
#
# Dry-run that prints the 15 sbatch lines without firing them:
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh
#
# Subset of methods (e.g. fire just LORA + TL_BARE):
#   ONLY_METHODS="LORA_R8_TTA TL_BARE_R2" \
#       bash sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh
#
# Subset of tta-steps (e.g. only the high-iter tail for a re-fire):
#   ONLY_TTA_STEPS="80 160" \
#       bash sweep_experiment/sbatch/submit_ttom_iteration_sweep.sh
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
TL_SBATCH="${TL_SBATCH:-delta_experiment/sbatch/run_tinylora.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Hard-coded paper-defense control scope. Override at your own risk.
#   - DATASET="panda" only (UCF not in spec for this wave).
#   - chunk_0 only (matches D1 smoke + the 2026-06-09 per-video analysis).
#   - 3 methods only (ADA / LORA_R8_TTA / TL_BARE_R2). Spec excludes the
#     4th headline method (TL_TIED_R2) — we sweep iteration count on the
#     three method *families* (delta-bias / LoRA / TinyLoRA-bare) rather
#     than on every TinyLoRA tying variant.
DATASET_TAG="panda"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SWEEP_RESULTS_SUBDIR="${SWEEP_RESULTS_SUBDIR:-sweep_experiment/results/panda_1000v_standard}"
TL_RESULTS_SUBDIR="${TL_RESULTS_SUBDIR:-delta_experiment/results/tinylora_panda_1000v_standard}"
SERIES_NAME="${SERIES_NAME:-panda_1000v_standard}"

# Single chunk × 100 videos = chunk_0 of the headline 1000v split.
NUM_CHUNKS="${NUM_CHUNKS:-1}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
MAX_VIDEOS="${MAX_VIDEOS:-100}"

# Frame geometry — standard 28-frame horizon (matches headline standard-1000v).
NUM_FRAMES="${NUM_FRAMES:-28}"
NUM_COND_FRAMES="${NUM_COND_FRAMES:-14}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
TTA_TOTAL_FRAMES="${TTA_TOTAL_FRAMES:-48}"
TTA_CONTEXT_FRAMES="${TTA_CONTEXT_FRAMES:-14}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.0}"

# Sweep grid. Outer loop over TTA-step counts; inner loop over methods.
METHODS=("ADA" "LORA_R8_TTA" "TL_BARE_R2")
TTA_STEP_GRID=(10 20 40 80 160)

DRY_RUN="${DRY_RUN:-0}"
ONLY_METHODS="${ONLY_METHODS:-}"
ONLY_TTA_STEPS="${ONLY_TTA_STEPS:-}"

count=0

_in_method_filter() {
    local needle="$1"
    [ -z "${ONLY_METHODS}" ] && return 0
    for m in ${ONLY_METHODS}; do
        if [ "${m}" = "${needle}" ]; then return 0; fi
    done
    return 1
}

_in_tta_filter() {
    local needle="$1"
    [ -z "${ONLY_TTA_STEPS}" ] && return 0
    for n in ${ONLY_TTA_STEPS}; do
        if [ "${n}" = "${needle}" ]; then return 0; fi
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

# Wallclock per job, scaled by TTA-step count. Anchored to the
# `submit_smoke_x0_loss.sh` ~2 GPU h baseline for LORA_R8_TTA × 100 videos at
# 10 steps; the TTA-time component scales linearly in N while generation /
# VBench costs are fixed, so the wall does NOT scale 1:1 with N. The cluster
# preemption wall caps at 24h regardless. Override per-step via env var
# WALL_FOR_<N> (e.g. `WALL_FOR_160=18:00:00 bash ...`) if needed.
_wall_for_tta_steps() {
    local n="$1"
    local override_var="WALL_FOR_${n}"
    if [ -n "${!override_var:-}" ]; then
        echo "${!override_var}"
        return 0
    fi
    case "${n}" in
        10)  echo "06:00:00" ;;
        20)  echo "08:00:00" ;;
        40)  echo "12:00:00" ;;
        80)  echo "18:00:00" ;;
        160) echo "24:00:00" ;;
        *)   echo "12:00:00" ;;
    esac
}

# ----------------------------------------------------------------------------
# Per-method submission helpers. Each builds the run_id, output dir, and the
# correct method-specific TTA-step env-var, then sbatches into the existing
# run_sweep.sbatch / run_tinylora.sbatch wrappers. The wrapper passes ALL
# other knobs at headline values so the only changing variable in this sweep
# is the TTA-step count.
#
# CRITICAL — env-var per method (cross-checked vs the case statements in
# `sweep_experiment/sbatch/run_sweep.sbatch` and
# `delta_experiment/sbatch/run_tinylora.sbatch`):
#   ADA          (delta_a) -> DELTA_STEPS  -> --delta-steps
#   LORA_R8_TTA  (lora)    -> NUM_STEPS    -> --num-steps
#   TL_BARE_R2   (tinylora)-> TTA_STEPS    -> --tta-steps
# ----------------------------------------------------------------------------
submit_ada_job() {
    # Args: $1 = tta_steps (N)
    local tta_steps="$1"
    local run_id="ADA_TTA${tta_steps}"
    local out_dir="${PROJECT_ROOT}/${SWEEP_RESULTS_SUBDIR}/${run_id}/chunk_0"
    local job_name="t1k_ttom_${run_id}_c0"
    local wall
    wall="$(_wall_for_tta_steps "${tta_steps}")"

    _exec_or_dry sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${wall}" \
        --export="ALL,METHOD=delta_a,RUN_ID=${run_id},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=0,CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn,DELTA_STEPS=${tta_steps},DELTA_LR=5.0e-3" \
        "${SWEEP_SBATCH}"
    count=$((count + 1))
}

submit_lora_r8_tta_job() {
    local tta_steps="$1"
    local run_id="LORA_R8_TTA_TTA${tta_steps}"
    local out_dir="${PROJECT_ROOT}/${SWEEP_RESULTS_SUBDIR}/${run_id}/chunk_0"
    local job_name="t1k_ttom_${run_id}_c0"
    local wall
    wall="$(_wall_for_tta_steps "${tta_steps}")"

    _exec_or_dry sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${wall}" \
        --export="ALL,METHOD=lora,RUN_ID=${run_id},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=0,CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn,LORA_RANK=8,LORA_ALPHA=16,LORA_TARGET_BLOCKS=all,NUM_STEPS=${tta_steps},LEARNING_RATE=5.0e-5,WARMUP_STEPS=3,WEIGHT_DECAY=0.01,MAX_GRAD_NORM=10.0,TARGET_FFN=0" \
        "${SWEEP_SBATCH}"
    count=$((count + 1))
}

submit_tl_bare_r2_job() {
    local tta_steps="$1"
    local run_id="TL_BARE_R2_TTA${tta_steps}"
    local out_dir="${PROJECT_ROOT}/${TL_RESULTS_SUBDIR}/${run_id}/chunk_0"
    local job_name="t1k_ttom_${run_id}_c0"
    local wall
    wall="$(_wall_for_tta_steps "${tta_steps}")"

    _exec_or_dry sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${wall}" \
        --export="ALL,DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},NUM_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=0,CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn,SVD_RANK=2,N_TIE=1,TARGET_PRESET=qkv_proj,TARGET_BLOCKS=all,TTA_STEPS=${tta_steps},TTA_LR=1e-3" \
        "${TL_SBATCH}"
    count=$((count + 1))
}

submit_method_at_tta_steps() {
    # Args: $1 = method run-id base (ADA|LORA_R8_TTA|TL_BARE_R2)  $2 = tta_steps
    local method="$1"
    local tta_steps="$2"

    if ! _in_method_filter "${method}"; then return 0; fi
    if ! _in_tta_filter "${tta_steps}"; then return 0; fi

    case "${method}" in
        ADA)         submit_ada_job          "${tta_steps}" ;;
        LORA_R8_TTA) submit_lora_r8_tta_job  "${tta_steps}" ;;
        TL_BARE_R2)  submit_tl_bare_r2_job   "${tta_steps}" ;;
        *)
            echo "ERROR: unknown method '${method}' (expected one of: ${METHODS[*]})" >&2
            exit 1
            ;;
    esac
}

# ----------------------------------------------------------------------------
# Submission summary banner
# ----------------------------------------------------------------------------
echo "============================================================"
echo "TTOM iteration-saturation sweep (Track D Wave D2)"
echo "============================================================"
echo "  account        : ${ACCOUNT}"
echo "  dataset        : ${DATASET_TAG}  (chunk_0 only; videos 0-$((CHUNK_SIZE - 1)))"
echo "  series         : ${SERIES_NAME}"
echo "  data dir       : ${DATA_DIR}"
echo "  sweep results  : ${SWEEP_RESULTS_SUBDIR}"
echo "  TL results     : ${TL_RESULTS_SUBDIR}"
echo "  methods        : ${METHODS[*]}"
echo "  tta-step grid  : ${TTA_STEP_GRID[*]}"
echo "  num chunks     : ${NUM_CHUNKS} x ${CHUNK_SIZE} videos = ${MAX_VIDEOS}"
echo "  dry run        : ${DRY_RUN}"
echo "  only methods   : ${ONLY_METHODS:-<all 3>}"
echo "  only tta-steps : ${ONLY_TTA_STEPS:-<all 5>}"
echo "============================================================"
echo ""

# Outer loop: TTA-step count. Inner loop: method. This emission order
# (ADA_TTA10, LORA_R8_TTA_TTA10, TL_BARE_R2_TTA10, ADA_TTA20, ...) groups
# jobs that will fight for the same h200 queue priority into adjacent
# submissions, so the squeue snapshot reads as a clean grid.
for tta_steps in "${TTA_STEP_GRID[@]}"; do
    for method in "${METHODS[@]}"; do
        submit_method_at_tta_steps "${method}" "${tta_steps}"
    done
done

# ----------------------------------------------------------------------------
# Post-submission summary
# ----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "Submitted ${count} job(s)."
echo ""
echo "Expected total GPU-h: ~125 GPU-h serial (3 methods x 5 tta-steps;"
echo "per-job wall scales with N, anchored at ~2 GPU h for N=10)."
echo "Per-job wall by TTA-step:"
for n in "${TTA_STEP_GRID[@]}"; do
    printf "  N=%-3d -> wall=%s\n" "${n}" "$(_wall_for_tta_steps "${n}")"
done
echo ""
echo "Output directories (15 leaves under three series dirs):"
for n in "${TTA_STEP_GRID[@]}"; do
    echo "  ${PROJECT_ROOT}/${SWEEP_RESULTS_SUBDIR}/ADA_TTA${n}/chunk_0/"
    echo "  ${PROJECT_ROOT}/${SWEEP_RESULTS_SUBDIR}/LORA_R8_TTA_TTA${n}/chunk_0/"
    echo "  ${PROJECT_ROOT}/${TL_RESULTS_SUBDIR}/TL_BARE_R2_TTA${n}/chunk_0/"
done
echo ""
echo "Monitor:"
echo "  squeue -u \"\${USER}\" | grep t1k_ttom_"
echo ""
echo "After completion, compare against the headline chunk_0 results:"
echo "  - sweep_experiment/results/panda_1000v_standard/{ADA,LORA_R8_TTA}/chunk_0/summary.json"
echo "  - delta_experiment/results/tinylora_panda_1000v_standard/TL_BARE_R2/chunk_0/summary.json"
echo "Plot ΔPSNR / ΔLPIPS / ΔFVD vs TTA-step count per method per spec in"
echo "PAPER_FRAGMENT_ttom_positioning_2026-06-12.md \"Suggested control\"."
echo "============================================================"
