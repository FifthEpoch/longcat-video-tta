#!/bin/bash
# ============================================================================
# Smoke-test: VAE-Decoder-Only TTA (Modification 2).
#
# Fires a SINGLE chunk of VAE_DEC_TTA on Panda 1000v (videos 0-99 of the
# 1000-video standard horizon split). The recipe freezes the DiT entirely
# and adapts only vae.decoder on the round-trip reconstruction loss
#   L = MSE(VAE.decode(VAE.encode(pixel_frames_train)), pixel_frames_train)
# per video. Decoder is restored to its pristine snapshot between videos.
#
# Output lands in
#   sweep_experiment/results/panda_1000v_standard/VAE_DEC_TTA_LR1e-5/chunk_0/
# alongside the headline cells, so paper-table builders find it naturally.
#
# This is a sibling of submit_smoke_x0_loss.sh (Modification 1). It differs
# only in:
#   1. METHOD=vae_decoder (new dispatch case in run_sweep.sbatch).
#   2. Hyperparameters are recipe-specific (VAE_TTA_STEPS, VAE_TTA_LR,
#      VAE_TTA_LPIPS_WEIGHT) — there is no headline cell to "match" since
#      this is a wholly new recipe.
#   3. Frame geometry matches the headline LORA_R8_TTA cell (28 frames,
#      gen_start_frame=48, num_cond_frames=14) so the resulting per-video
#      ΔPSNR / ΔLPIPS / ΔFVD against NOTTA_LR0 and LORA_R8_TTA on the same
#      chunk are directly comparable.
#
# Expected wallclock: ~3-5 GPU h (one chunk × 100 videos on H200). The TTA
# step is dominated by the VAE encode + decode + pixel-MSE backward and is
# 5-10× cheaper than a DiT-forward TTA step, so the per-video TTA cost is
# small; the inference cost dominates. SBATCH cap is 12h to match the
# headline LORA_R8_TTA wall.
#
# Submit (after `git pull` on the cluster):
#   cd /scratch/${USER}/longcat-video-tta
#   bash sweep_experiment/sbatch/submit_smoke_vae_decoder_tta.sh
#
# Dry-run (prints sbatch lines without firing):
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_smoke_vae_decoder_tta.sh
#
# After the smoke completes, compare against NOTTA and LORA_R8_TTA on the
# same chunk_0 split (per video — order is preserved by the 1000v split):
#   diff <(jq -r '.results[] | "\(.video_name) \(.psnr)"' \
#            sweep_experiment/results/panda_1000v_standard/NOTTA/chunk_0/summary.json \
#            | sort) \
#        <(jq -r '.results[] | "\(.video_name) \(.psnr)"' \
#            sweep_experiment/results/panda_1000v_standard/VAE_DEC_TTA_LR1e-5/chunk_0/summary.json \
#            | sort)
#
# Decision rule (per LITERATURE_tta_recipe_modifications_2026-06-12.md §3,
# Modification 2):
#   - PRIMARY: held-out ΔPSNR on the four §2.3 beneficiary videos
#     (panda_0461, panda_0555, panda_0862, panda_0431) under VAE-decoder TTA
#     exceeds +1.0 dB on ≥3 of 4 → bottleneck IS the VAE → scale up to
#     full 10-chunk × {1e-6, 1e-5, 1e-4} LR sweep (~30 GPU-h).
#   - SECONDARY: aggregate median |ΔPSNR| > 0.5 dB across the 100-video
#     chunk → also scale up.
#   - NULL OUTCOME: neither cohort triggers → VAE is not the binding
#     constraint either; the bottleneck is most likely the supervisory
#     signal (Mod 3 augmentation-consistency) or a fundamentally weak
#     conditioning signal (Mod 5 continual streaming or Mod 8 amortised
#     hypernetwork — both expensive). Document as a negative result.
# ============================================================================
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/wc3013/longcat-video-tta}"
SWEEP_SBATCH="${SWEEP_SBATCH:-sweep_experiment/sbatch/run_sweep.sbatch}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Match LORA_R8_TTA headline wall — VAE-decoder TTA per chunk should be
# faster but the inference cost is identical, and the SBATCH cap is only
# a ceiling.
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

# The recipe under test.
VAE_TTA_STEPS="${VAE_TTA_STEPS:-10}"
VAE_TTA_LR="${VAE_TTA_LR:-1e-5}"
VAE_TTA_LPIPS_WEIGHT="${VAE_TTA_LPIPS_WEIGHT:-0.0}"
VAE_TTA_GRAD_CLIP="${VAE_TTA_GRAD_CLIP:-1.0}"
VAE_TTA_WEIGHT_DECAY="${VAE_TTA_WEIGHT_DECAY:-0.0}"

# Run-id suffix encodes the LR so future archaeology is unambiguous when
# the eventual λ-sweep produces VAE_DEC_TTA_LR1e-6 / LR1e-4 / etc.
RUN_ID_SUFFIX="${RUN_ID_SUFFIX:-_LR${VAE_TTA_LR}}"
RUN_ID="${RUN_ID:-VAE_DEC_TTA${RUN_ID_SUFFIX}}"

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
echo "Smoke-test: VAE-Decoder-Only TTA (Modification 2)"
echo "============================================================"
echo "  series       : ${SERIES_NAME}"
echo "  run id       : ${RUN_ID}"
echo "  account      : ${ACCOUNT}"
echo "  chunks       : ${NUM_CHUNKS} x ${CHUNK_SIZE} videos = ${MAX_VIDEOS}"
echo "  data dir     : ${DATA_DIR}"
echo "  results dir  : ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}"
echo "  vae steps    : ${VAE_TTA_STEPS}"
echo "  vae lr       : ${VAE_TTA_LR}"
echo "  vae lpips w  : ${VAE_TTA_LPIPS_WEIGHT}"
echo "  vae grad clp : ${VAE_TTA_GRAD_CLIP}"
echo "  vae wd       : ${VAE_TTA_WEIGHT_DECAY}"
echo "  dry run      : ${DRY_RUN}"
echo "============================================================"
echo ""

count=0
for chunk in $(seq 0 $((NUM_CHUNKS - 1))); do
    start=$((chunk * CHUNK_SIZE))
    out_dir="${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}/chunk_${chunk}"
    job_name="smokev2_${DATASET_TAG}_${RUN_ID}_c${chunk}"

    _exec_or_dry sbatch \
        --account="${ACCOUNT}" \
        --job-name="${job_name}" \
        --time="${TIME_SHORT}" \
        --export="ALL,METHOD=vae_decoder,RUN_ID=${RUN_ID},SERIES_NAME=${SERIES_NAME},DATA_DIR=${DATA_DIR},OUTPUT_DIR=${out_dir},MAX_VIDEOS=${MAX_VIDEOS},START_VIDEO_IDX=${start},CHUNK_SIZE=${CHUNK_SIZE},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},TTA_TOTAL_FRAMES=${TTA_TOTAL_FRAMES},TTA_CONTEXT_FRAMES=${TTA_CONTEXT_FRAMES},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},GUIDANCE_SCALE=${GUIDANCE_SCALE},RESOLUTION=480p,SEED=42,ES_DISABLE=1,COMPUTE_FVD=1,COMPUTE_FID=1,COMPUTE_VBENCH=1,NO_SAVE_VIDEOS=0,CAPTION_GUARD_MODE=warn,FEATURE_FRAME_GUARD_MODE=warn,VAE_TTA_STEPS=${VAE_TTA_STEPS},VAE_TTA_LR=${VAE_TTA_LR},VAE_TTA_LPIPS_WEIGHT=${VAE_TTA_LPIPS_WEIGHT},VAE_TTA_GRAD_CLIP=${VAE_TTA_GRAD_CLIP},VAE_TTA_WEIGHT_DECAY=${VAE_TTA_WEIGHT_DECAY}" \
        "${SWEEP_SBATCH}"
    count=$((count + 1))
done

echo ""
echo "============================================================"
echo "Submitted ${count} job(s)."
echo ""
echo "Monitor:"
echo "  squeue -u \$USER | grep '^[0-9]* smokev2_'"
echo ""
echo "After completion:"
echo "  ls -lh ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}/chunk_0/"
echo "  jq '.results | length, (map(.psnr) | add/length), (map(.success) | all)' \\"
echo "      ${PROJECT_ROOT}/${RESULTS_SUBDIR}/${RUN_ID}/chunk_0/summary.json"
echo ""
echo "Compare against headline NOTTA + LORA_R8_TTA chunk_0 ΔPSNR; per"
echo "LITERATURE_tta_recipe_modifications_2026-06-12.md §3, Modification 2:"
echo "  - ΔPSNR > +1.0 dB on ≥3 of {panda_0461, _0555, _0862, _0431} → SCALE UP"
echo "  - aggregate median |ΔPSNR| > 0.5 dB                            → also scale up"
echo "  - neither triggers                                              → null result"
echo "============================================================"
