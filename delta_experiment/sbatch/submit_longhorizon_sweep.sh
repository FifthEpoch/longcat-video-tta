#!/bin/bash
# ============================================================================
# Sharded LONG-horizon sweep (native geometry, ~1 minute of video).
#
# The 2026-08-08 controls showed that at LongCat's native 13-cond/80-gen window
# drift is real but mild over 6 chunks (=480 gen frames, ~30s). Reviewers of
# "long-horizon video continuation" expect 30s..2min+ (StreamingT2V ~2min/1200f,
# Rolling Forcing multi-minute, LongCat's own design point ~1min). 6 chunks sits
# at the LOW end. This sweep pushes to NUM_CHUNKS=12 => 12 x 80 = 960 generated
# frames ~= 60s @16fps -- ~50% of the 2-min field ceiling and ~= LongCat's 1-min
# design point, i.e. a genuinely long horizon rather than the lower bound.
#
# One native 12-chunk video is ~110 min at 50 steps, so we SHARD across jobs
# (SHARD_SIZE videos/job, own OUTPUT_DIR + checkpoint each -> no race) to stay in
# the 12h wall. After all shards finish, merge into one verdict:
#   python scripts/merge_drift_shards.py --shards-root <base>
#
# Usage:
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash delta_experiment/sbatch/submit_longhorizon_sweep.sh
#   # NOTTA gating sweep is the default. To also run the streaming delta (EXP4):
#   METHOD=delta_stream bash delta_experiment/sbatch/submit_longhorizon_sweep.sh
#   # knobs: POOL_N, SHARD_SIZE, NUM_CHUNKS, NUM_INFERENCE_STEPS, DRY_RUN=1
# ============================================================================
set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SBATCH="${PROJECT_ROOT}/delta_experiment/sbatch/run_longhorizon_drift.sbatch"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

METHOD="${METHOD:-notta}"                    # notta | delta | delta_stream
ROLLOUT_MODE="${ROLLOUT_MODE:-native}"       # native geometry for a real horizon
POOL_N="${POOL_N:-8}"                        # total videos across the sweep
SHARD_SIZE="${SHARD_SIZE:-2}"                # videos per job (fits 12h wall)
NUM_CHUNKS="${NUM_CHUNKS:-12}"               # 12 x 80 = 960 gen frames ~= 60s
NUM_COND_FRAMES="${NUM_COND_FRAMES:-13}"
NUM_FRAMES="${NUM_FRAMES:-93}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-50}"
SEED="${SEED:-42}"

# delta / streaming-delta knobs (only used when METHOD != notta)
DELTA_STEPS="${DELTA_STEPS:-10}"
DELTA_LR="${DELTA_LR:-1e-3}"
DELTA_PLACEMENT="${DELTA_PLACEMENT:-adaln}"
STREAM_REFIT_STEPS="${STREAM_REFIT_STEPS:-5}"
STREAM_REFIT_LR="${STREAM_REFIT_LR:-0}"
STREAM_BLEND="${STREAM_BLEND:-0.5}"
# 'clean' (default) = clean-anchored re-fit: condition on the drifted context but
# flow-match toward the CLEAN chunk-0 real latents (removes the train-on-own-drift
# flaw that made STREAM_TARGET=generated null on 2026-08-09). 'generated' = old.
STREAM_TARGET="${STREAM_TARGET:-clean}"

DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"
# encode the stream target in the series name so clean vs generated never collide.
if [ "${METHOD}" = "delta_stream" ]; then
  SERIES="${SERIES:-longhorizon_sweep_${METHOD}_${STREAM_TARGET}_${ROLLOUT_MODE}_${NUM_CHUNKS}ch}"
else
  SERIES="${SERIES:-longhorizon_sweep_${METHOD}_${ROLLOUT_MODE}_${NUM_CHUNKS}ch}"
fi
BASE="${PROJECT_ROOT}/sweep_experiment/results/${SERIES}"

NSHARDS=$(( (POOL_N + SHARD_SIZE - 1) / SHARD_SIZE ))
GEN_FRAMES=$(( (NUM_FRAMES - NUM_COND_FRAMES) * NUM_CHUNKS ))

echo "============================================================"
echo "Long-horizon SWEEP (sharded)"
echo "  account : ${ACCOUNT}"
echo "  method  : ${METHOD}   mode=${ROLLOUT_MODE}"
echo "  horizon : ${NUM_CHUNKS} chunks x $((NUM_FRAMES-NUM_COND_FRAMES)) gen = ${GEN_FRAMES} frames (~$((GEN_FRAMES/16))s @16fps)"
echo "  geometry: cond=${NUM_COND_FRAMES} frames=${NUM_FRAMES} steps=${NUM_INFERENCE_STEPS}"
echo "  pool    : N=${POOL_N}  shard_size=${SHARD_SIZE}  -> ${NSHARDS} jobs"
echo "  series  : ${SERIES}"
if [ "${METHOD}" = "delta_stream" ]; then
  echo "  stream  : target=${STREAM_TARGET} refit_steps=${STREAM_REFIT_STEPS} refit_lr=${STREAM_REFIT_LR} blend(anchor)=${STREAM_BLEND}"
fi
echo "============================================================"

jids=()
for (( s=0; s<NSHARDS; s++ )); do
  start=$(( s * SHARD_SIZE ))
  outdir="${BASE}/shard_$(printf '%04d' "${s}")"
  echo "-> shard ${s}: videos [${start}, $((start+SHARD_SIZE))) -> ${outdir}"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "   DRY_RUN=1 -> not submitting"
    continue
  fi
  jid=$(sbatch --parsable --account="${ACCOUNT}" \
    --export=ALL,"METHOD=${METHOD},ROLLOUT_MODE=${ROLLOUT_MODE},NUM_VIDEOS=${POOL_N},START_VIDEO_IDX=${start},CHUNK_SIZE=${SHARD_SIZE},NUM_CHUNKS=${NUM_CHUNKS},NUM_COND_FRAMES=${NUM_COND_FRAMES},NUM_FRAMES=${NUM_FRAMES},GEN_START_FRAME=${GEN_START_FRAME},NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS},SEED=${SEED},DELTA_STEPS=${DELTA_STEPS},DELTA_LR=${DELTA_LR},DELTA_PLACEMENT=${DELTA_PLACEMENT},STREAM_REFIT_STEPS=${STREAM_REFIT_STEPS},STREAM_REFIT_LR=${STREAM_REFIT_LR},STREAM_BLEND=${STREAM_BLEND},STREAM_TARGET=${STREAM_TARGET},DATA_DIR=${DATA_DIR},CHECKPOINT_DIR=${CHECKPOINT_DIR},OUTPUT_DIR=${outdir}" \
    "${SBATCH}")
  echo "   job ${jid}"
  jids+=("${jid}")
done

echo ""
echo "Submitted ${#jids[@]} shard jobs: ${jids[*]:-<none>}"
echo ""
echo "After ALL shards finish, merge + verdict + plots:"
echo "  python scripts/merge_drift_shards.py --shards-root ${BASE}"
echo "  python scripts/plot_drift_curves.py --summary ${BASE}/merged_summary.json --out-dir ${BASE}/plots"
echo ""
echo "Then paired test vs the NOTTA run at the same geometry:"
echo "  python scripts/compare_drift_paired.py \\"
echo "    --notta ${PROJECT_ROOT}/sweep_experiment/results/longhorizon_sweep_notta_${ROLLOUT_MODE}_${NUM_CHUNKS}ch/merged_summary.json \\"
echo "    --delta ${BASE}/merged_summary.json --out-dir ${BASE}/paired --label-b ${METHOD}-${STREAM_TARGET}"
echo ""
echo "Note: all shards share NUM_VIDEOS=${POOL_N} + SEED=${SEED} so the video"
echo "list ordering is identical; each shard slices [start, start+shard_size)."
echo ""
echo "FALLBACK (extend horizon even more): re-launch NOTTA + this arm with a"
echo "longer rollout, e.g. NUM_CHUNKS=18 (~72s) or 24 (~96s). Cost scales ~linearly"
echo "(~9.3 min/native chunk); drop SHARD_SIZE to 1 to stay in the 12h wall, e.g.:"
echo "  NUM_CHUNKS=24 SHARD_SIZE=1 bash delta_experiment/sbatch/submit_longhorizon_sweep.sh   # NOTTA"
echo "  NUM_CHUNKS=24 SHARD_SIZE=1 METHOD=delta_stream bash delta_experiment/sbatch/submit_longhorizon_sweep.sh"
