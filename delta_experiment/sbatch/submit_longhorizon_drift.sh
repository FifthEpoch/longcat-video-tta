#!/bin/bash
# ============================================================================
# Submitter for the long-horizon drift diagnostic + its two controls.
#
# The 2026-08-07 drift run (NOTTA, reencode 14/28 geometry) showed LongCat
# degrades monotonically over an 8-chunk autoregressive rollout (over-saturation
# +58%, contrast +13%, HF-artifact sharpness +258%). Two confounds remained, and
# this submitter runs BOTH controls the user asked for:
#
#   EXP-A (native-protocol control): is the drift real or a short-window
#     re-conditioning artifact? Re-run NOTTA at LongCat's idiomatic
#     13-cond / 93-frame (80-gen) window. generate_vc has NO KV-cache carryover
#     across windows, so native long-horizon IS this external rollout -- only the
#     GEOMETRY differs. Compare against the existing reencode NOTTA result.
#
#   EXP-B (intervention-in-rollout): does an AdaSteer delta flatten the drift?
#     Train delta ONCE on the observed frames (exact run_delta_a recipe) and hold
#     it FIXED across the reencode-geometry rollout. Paired seeds with the
#     existing NOTTA reencode run, so it is a clean per-video comparison. If a
#     fixed context-0 delta decays as the rollout leaves the trained distribution
#     -> motivates a streaming / per-chunk re-fit delta (EXP4).
#
# Usage:
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash delta_experiment/sbatch/submit_longhorizon_drift.sh
#   # add arms:  ARMS="notta_native delta_reencode delta_native" bash <this>
#   # after each finishes:
#   python scripts/plot_drift_curves.py \
#       --summary sweep_experiment/results/diag_longhorizon_drift_<arm>/summary.json \
#       --out-dir sweep_experiment/results/diag_longhorizon_drift_<arm>/plots
# ============================================================================
set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SBATCH="${PROJECT_ROOT}/delta_experiment/sbatch/run_longhorizon_drift.sbatch"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"

# Arms to submit: space-separated <method>_<mode> tokens.
#   method = notta | delta      mode = reencode | native
# Default = the two decisive new controls (the reencode NOTTA reference already
# exists from the 2026-08-07 run and is NOT re-submitted here).
ARMS="${ARMS:-notta_native delta_reencode}"

SEED="${SEED:-42}"
GEN_START_FRAME="${GEN_START_FRAME:-48}"
DELTA_STEPS="${DELTA_STEPS:-10}"
DELTA_LR="${DELTA_LR:-1e-3}"
DELTA_PLACEMENT="${DELTA_PLACEMENT:-adaln}"
SERIES_BASE="${SERIES_BASE:-diag_longhorizon_drift}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/datasets/panda_ood_budget_1000v_preview_480p}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${SCRATCH_BASE}/longcat-video-checkpoints}"

# reencode geometry (matches EXP2/EXP3 + the existing drift run)
RE_NUM_VIDEOS="${RE_NUM_VIDEOS:-24}"
RE_NUM_CHUNKS="${RE_NUM_CHUNKS:-8}"
RE_NUM_COND_FRAMES="${RE_NUM_COND_FRAMES:-14}"
RE_NUM_FRAMES="${RE_NUM_FRAMES:-28}"
# native geometry (LongCat idiomatic 13-cond/93-frame -> 80 gen/chunk). Heavier
# per chunk, so fewer videos/chunks by default to stay in the 12h budget; still
# enough to reveal a per-chunk trend. All env-overridable.
NAT_NUM_VIDEOS="${NAT_NUM_VIDEOS:-16}"
NAT_NUM_CHUNKS="${NAT_NUM_CHUNKS:-6}"
NAT_NUM_COND_FRAMES="${NAT_NUM_COND_FRAMES:-13}"
NAT_NUM_FRAMES="${NAT_NUM_FRAMES:-93}"

echo "============================================================"
echo "Long-horizon drift diagnostic + controls"
echo "  account : ${ACCOUNT}"
echo "  arms    : ${ARMS}"
echo "  seed=${SEED}  gsf=${GEN_START_FRAME}  delta(steps=${DELTA_STEPS} lr=${DELTA_LR} ${DELTA_PLACEMENT})"
echo "  reencode: N=${RE_NUM_VIDEOS} chunks=${RE_NUM_CHUNKS} cond=${RE_NUM_COND_FRAMES} frames=${RE_NUM_FRAMES}"
echo "  native  : N=${NAT_NUM_VIDEOS} chunks=${NAT_NUM_CHUNKS} cond=${NAT_NUM_COND_FRAMES} frames=${NAT_NUM_FRAMES}"
echo "============================================================"

submit_arm() {
  local method="$1" mode="$2"
  local nvid nchunks ncond nframes
  if [ "${mode}" = "native" ]; then
    nvid="${NAT_NUM_VIDEOS}"; nchunks="${NAT_NUM_CHUNKS}"
    ncond="${NAT_NUM_COND_FRAMES}"; nframes="${NAT_NUM_FRAMES}"
  else
    nvid="${RE_NUM_VIDEOS}"; nchunks="${RE_NUM_CHUNKS}"
    ncond="${RE_NUM_COND_FRAMES}"; nframes="${RE_NUM_FRAMES}"
  fi
  local series="${SERIES_BASE}_${method}_${mode}"
  local outdir="${PROJECT_ROOT}/sweep_experiment/results/${series}"

  echo "-> arm ${method}_${mode}: N=${nvid} chunks=${nchunks} cond=${ncond} frames=${nframes}"
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "   DRY_RUN=1 -> not submitting (${outdir})"
    return 0
  fi
  local jid
  jid=$(sbatch --parsable --account="${ACCOUNT}" \
    --export=ALL,"METHOD=${method},ROLLOUT_MODE=${mode},NUM_VIDEOS=${nvid},NUM_CHUNKS=${nchunks},NUM_COND_FRAMES=${ncond},NUM_FRAMES=${nframes},GEN_START_FRAME=${GEN_START_FRAME},SEED=${SEED},DELTA_STEPS=${DELTA_STEPS},DELTA_LR=${DELTA_LR},DELTA_PLACEMENT=${DELTA_PLACEMENT},DATA_DIR=${DATA_DIR},CHECKPOINT_DIR=${CHECKPOINT_DIR},OUTPUT_DIR=${outdir}" \
    "${SBATCH}")
  echo "   job ${jid}  -> ${outdir}/summary.json"
}

for arm in ${ARMS}; do
  method="${arm%%_*}"
  mode="${arm#*_}"
  case "${method}" in notta|delta) ;; *) echo "  !! bad method in arm '${arm}' (want notta|delta)"; exit 2;; esac
  case "${mode}"   in reencode|native) ;; *) echo "  !! bad mode in arm '${arm}' (want reencode|native)"; exit 2;; esac
  submit_arm "${method}" "${mode}"
done

echo ""
echo "After each arm finishes, build drift-curve PNGs, e.g.:"
for arm in ${ARMS}; do
  echo "  python scripts/plot_drift_curves.py --summary ${PROJECT_ROOT}/sweep_experiment/results/${SERIES_BASE}_${arm}/summary.json --out-dir ${PROJECT_ROOT}/sweep_experiment/results/${SERIES_BASE}_${arm}/plots"
done
echo ""
echo "Interpretation:"
echo "  EXP-A native control : compare notta_native drift verdict vs the existing"
echo "    reencode NOTTA run. Drift persists at native geometry => inherent, not"
echo "    a short-window re-conditioning artifact."
echo "  EXP-B intervention   : compare delta_reencode drift verdict vs the existing"
echo "    reencode NOTTA run (paired seeds). Flatter curves => a fixed delta helps;"
echo "    decaying benefit => motivates a streaming per-chunk delta (EXP4)."
