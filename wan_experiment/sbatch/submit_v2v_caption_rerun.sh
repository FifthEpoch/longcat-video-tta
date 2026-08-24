#!/bin/bash
# Caption-conditioned replay of prior V2V generates.
# Stem-prompt dirs stay untouched. New series names only.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_caption_rerun.sh          # WAVE=1
#   WAVE=2 bash wan_experiment/sbatch/submit_v2v_caption_rerun.sh
#   WAVE=3 bash wan_experiment/sbatch/submit_v2v_caption_rerun.sh
#   WAVE=4 bash wan_experiment/sbatch/submit_v2v_caption_rerun.sh
#
# WAVE=1 is the paper N=32 set (submit this now).
# Do not scancel stem always-search 16288113–115.
# No TTC. No I2V. VIDEO_WORKERS=1.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
WAVE="${WAVE:-1}"
GEN_WALL="${GEN_WALL:-04:00:00}"
SEARCH_WALL="${SEARCH_WALL:-08:00:00}"
LONG_WALL="${LONG_WALL:-12:00:00}"
VBENCH_WALL="${VBENCH_WALL:-12:00:00}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: ${VIDEO_DIR} missing." >&2
    exit 1
fi
if [[ ! -f "${VIDEO_DIR}/metadata.csv" ]]; then
    echo "ERROR: ${VIDEO_DIR}/metadata.csv missing." >&2
    exit 1
fi
if [[ ! -f /scratch/${USER}/wan-checkpoints/self_forcing_dmd.pt ]]; then
    echo "ERROR: Self-Forcing ckpt missing." >&2
    exit 1
fi

echo "---- caption preflight (first-segment, not stem) ----"
python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, ".")
from scripts.caption_utils import load_resolved_captions_csv, canonical_video_id
root = Path("datasets/panda_1000_480p")
caps = load_resolved_captions_csv(root / "metadata.csv", warn_missing=False)
vids = sorted(p for p in root.rglob("*.mp4"))[:4]
bad = 0
for p in vids:
    cid = canonical_video_id(p.name)
    cap = caps.get(cid) or caps.get(p.stem) or ""
    print(f"{p.name}  source=metadata_csv")
    print(f"  {cap[:160]}")
    if not cap or cap.replace("_", " ") == p.stem.replace("_", " "):
        bad += 1
        print("  ERROR: looks like a stem prompt")
print(f"mapped={len(caps)} videos_checked={len(vids)} bad={bad}")
if bad or len(caps) < 32:
    raise SystemExit("preflight failed")
PY

JOBS=()
submit_method() {
    local series="$1"
    local method="$2"
    local k="$3"
    local n="$4"
    local wall="$5"
    local extra="${6:-}"
    local J
    J=$(sbatch --parsable --account="${ACCOUNT}" --time="${wall}" \
        --export=ALL,METHOD="${method}",SEARCH_K="${k}",HORIZON_S=30,N_VIDEOS="${n}",SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES="${series}",NUM_SHARDS=1,VIDEO_DIR="${VIDEO_DIR}",VIDEO_WORKERS=1,LIVE_MIN=0.012,PSEUDO_GAMMA=0.0,NOISE_TAU=0.04${extra} \
        "${SB}/run_v2v_chunked.sbatch")
    echo "V2V ${series} ${method} k=${k} n=${n} job ${J}"
    JOBS+=("${J}")
}

submit_vbench() {
    local series="$1"
    shift
    local dirs=("$@")
    local root="${PROJECT_ROOT}/wan_experiment/results/${series}"
    local VIDEO_DIRS
    VIDEO_DIRS=$(printf '%s ' "${dirs[@]}")
    local DEPS
    DEPS=$(IFS=:; echo "${JOBS[*]}")
    local VB
    VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
        --dependency="afterok:${DEPS}" \
        --export=ALL,SERIES_DIR="${root}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
        "${SB}/run_i2v_vbench.sbatch")
    echo "VBench ${series} job ${VB} afterok ${DEPS}"
    JOBS+=("${VB}")
}

run_wave1() {
    local S="v2v_panda_caption_32v"
    local R="${PROJECT_ROOT}/wan_experiment/results/${S}"
    # Paper hosts + SF/RF family + always-search twins.
    submit_method "${S}" notta 1 32 "${GEN_WALL}"
    submit_method "${S}" rolling_notta 1 32 "${GEN_WALL}"
    submit_method "${S}" sf_rewind 1 32 "${GEN_WALL}"
    submit_method "${S}" sf_sick_search 4 32 "${SEARCH_WALL}"
    submit_method "${S}" sf_pseudo 4 32 "${SEARCH_WALL}"
    submit_method "${S}" sf_sink 1 32 "${GEN_WALL}"
    submit_method "${S}" sf_always_search 4 32 "${SEARCH_WALL}"
    submit_method "${S}" rf_always_search 4 32 "${SEARCH_WALL}"
    submit_method "${S}" rf_rewind 1 32 "${GEN_WALL}"
    submit_method "${S}" rf_sick_search 4 32 "${SEARCH_WALL}"
    submit_method "${S}" rf_pseudo 4 32 "${SEARCH_WALL}"
    submit_method "${S}" rf_sink 1 32 "${GEN_WALL}"
    submit_vbench "${S}" \
        "${R}/notta_h30s_shard0" \
        "${R}/rolling_notta_h30s_shard0" \
        "${R}/sf_rewind_h30s_shard0" \
        "${R}/sf_sick_search_h30s_shard0" \
        "${R}/sf_pseudo_h30s_shard0" \
        "${R}/sf_sink_h30s_shard0" \
        "${R}/sf_always_search_h30s_shard0" \
        "${R}/rf_always_search_h30s_shard0" \
        "${R}/rf_rewind_h30s_shard0" \
        "${R}/rf_sick_search_h30s_shard0" \
        "${R}/rf_pseudo_h30s_shard0" \
        "${R}/rf_sink_h30s_shard0"
    echo "WAVE=1 paper N=32. Cite vs caption notta. Stem tables stay audit."
}

run_wave2() {
    local S="v2v_panda_caption_closed_32v"
    local R="${PROJECT_ROOT}/wan_experiment/results/${S}"
    submit_method "${S}" seed_bon 4 32 "${SEARCH_WALL}"
    submit_method "${S}" quiet_bon 4 32 "${SEARCH_WALL}"
    submit_method "${S}" live_bon 4 32 "${SEARCH_WALL}"
    submit_method "${S}" appear_bon 4 32 "${SEARCH_WALL}"
    submit_method "${S}" sf_roll 1 32 "${GEN_WALL}"
    submit_method "${S}" rf_chunk 1 32 "${GEN_WALL}"
    submit_method "${S}" sf_recache 1 32 "${GEN_WALL}"
    submit_method "${S}" rf_recache 1 32 "${GEN_WALL}"
    submit_vbench "${S}" \
        "${R}/seed_bon_h30s_shard0" \
        "${R}/quiet_bon_h30s_shard0" \
        "${R}/live_bon_h30s_shard0" \
        "${R}/appear_bon_h30s_shard0" \
        "${R}/sf_roll_h30s_shard0" \
        "${R}/rf_chunk_h30s_shard0" \
        "${R}/sf_recache_h30s_shard0" \
        "${R}/rf_recache_h30s_shard0"
    echo "WAVE=2 closed N=32. Compare to v2v_panda_caption_32v/notta."
}

run_wave3() {
    local S="v2v_panda_caption_8v"
    local R="${PROJECT_ROOT}/wan_experiment/results/${S}"
    submit_method "${S}" motion_bon 4 8 "${GEN_WALL}"
    submit_method "${S}" backtrack 1 8 "${GEN_WALL}"
    submit_method "${S}" hinge_bon 4 8 "${GEN_WALL}"
    submit_method "${S}" late_bon 4 8 "${GEN_WALL}"
    submit_method "${S}" hist_drop 4 8 "${GEN_WALL}"
    submit_method "${S}" good_backtrack 1 8 "${GEN_WALL}"
    submit_method "${S}" cached_bon 4 8 "${GEN_WALL}"
    submit_method "${S}" sink 1 8 "${GEN_WALL}"
    submit_method "${S}" tail_hist 1 8 "${GEN_WALL}"
    submit_method "${S}" live_hist 4 8 "${GEN_WALL}"
    submit_method "${S}" longlive_notta 1 8 "${GEN_WALL}"
    submit_method "${S}" longlive_sink 1 8 "${GEN_WALL}"
    submit_method "${S}" longlive_prefix_sink 1 8 "${GEN_WALL}"
    submit_method "${S}" longlive_live_bon 4 8 "${GEN_WALL}"
    submit_method "${S}" live_appear 4 8 "${GEN_WALL}"
    submit_method "${S}" pseudo_gate 4 8 "${GEN_WALL}"
    submit_method "${S}" pseudo_appear 4 8 "${GEN_WALL}"
    submit_method "${S}" noise_probe 1 8 "${GEN_WALL}"
    submit_method "${S}" noise_bon 4 8 "${GEN_WALL}"
    submit_method "${S}" rolling_rho_lo 1 8 "${GEN_WALL}"
    submit_method "${S}" rolling_rho_hi 1 8 "${GEN_WALL}"
    submit_method "${S}" rolling_adapt 1 8 "${GEN_WALL}"
    submit_method "${S}" rolling_look 4 8 "${SEARCH_WALL}"
    submit_vbench "${S}" \
        "${R}/motion_bon_h30s_shard0" \
        "${R}/backtrack_h30s_shard0" \
        "${R}/hinge_bon_h30s_shard0" \
        "${R}/late_bon_h30s_shard0" \
        "${R}/hist_drop_h30s_shard0" \
        "${R}/good_backtrack_h30s_shard0" \
        "${R}/cached_bon_h30s_shard0" \
        "${R}/sink_h30s_shard0" \
        "${R}/tail_hist_h30s_shard0" \
        "${R}/live_hist_h30s_shard0" \
        "${R}/longlive_notta_h30s_shard0" \
        "${R}/longlive_sink_h30s_shard0" \
        "${R}/longlive_prefix_sink_h30s_shard0" \
        "${R}/longlive_live_bon_h30s_shard0" \
        "${R}/live_appear_h30s_shard0" \
        "${R}/pseudo_gate_h30s_shard0" \
        "${R}/pseudo_appear_h30s_shard0" \
        "${R}/noise_probe_h30s_shard0" \
        "${R}/noise_bon_h30s_shard0" \
        "${R}/rolling_rho_lo_h30s_shard0" \
        "${R}/rolling_rho_hi_h30s_shard0" \
        "${R}/rolling_adapt_h30s_shard0" \
        "${R}/rolling_look_h30s_shard0"
    echo "WAVE=3 N=8 discovery. Skip shift_search/knob_probe. N=8 notta = first 8 of caption_32v."
}

run_wave4() {
    local S="v2v_panda_caption_128v"
    local R="${PROJECT_ROOT}/wan_experiment/results/${S}"
    if [[ ! -f /scratch/${USER}/wan-checkpoints/rolling_forcing_dmd.pt ]]; then
        echo "ERROR: Rolling Forcing ckpt missing." >&2
        exit 1
    fi
    submit_method "${S}" notta 1 128 "${LONG_WALL}"
    submit_method "${S}" rolling_notta 1 128 "${LONG_WALL}"
    submit_vbench "${S}" \
        "${R}/notta_h30s_shard0" \
        "${R}/rolling_notta_h30s_shard0"
    echo "WAVE=4 caption N=128 hosts only. Do not mix with stem rolling-128."
}

case "${WAVE}" in
    1) run_wave1 ;;
    2) run_wave2 ;;
    3) run_wave3 ;;
    4) run_wave4 ;;
    all)
        echo "WAVE=all queues 1+2+3+4. H200 extras will sit. Prefer WAVE=1 first."
        run_wave1
        JOBS=()
        run_wave2
        JOBS=()
        run_wave3
        JOBS=()
        run_wave4
        ;;
    *)
        echo "ERROR: WAVE must be 1, 2, 3, 4, or all (got ${WAVE})" >&2
        exit 2
        ;;
esac

echo "Caption rerun WAVE=${WAVE}. Sidecar must be prompt_source=metadata_csv."
echo "If the first sidecar is still stem, scancel this wave only:  scancel ${JOBS[*]}"
echo "Do not scancel 16288113 16288114 16288115."
echo "No TTC. No I2V. Do not write caption numbers into stem tables."
