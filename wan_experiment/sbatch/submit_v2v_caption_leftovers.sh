#!/bin/bash
# Caption-conditioned leftover ρ / look. Stem leftovers stay untouched.
#
# The Aug-22 series `v2v_panda_rolling_leftovers_8v` used filename
# prompts (`panda 0013`). Tails morphed into pandas. This wave is the
# first caption-CSV replay of those four RF knobs.
#
# Host is existing caption Rolling (do not remake):
#   v2v_panda_caption_32v/rolling_notta_h30s_shard0  (first 8)
# Do not remake cite-128. Do not launch WAVE=3 (19 extra methods).
# Do not remake keep / intra / denoise / AdaSteer / Pseudo-next
# (those already used metadata.csv).
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_caption_leftovers.sh
#
# No TTC. No I2V. VIDEO_WORKERS=1. k=4 only on rolling_look.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_caption_leftovers_8v}"
N_VIDEOS="${N_VIDEOS:-8}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    N_VIDEOS=2
    SERIES="${SERIES}_smoke"
fi
ROLL_WALL="${ROLL_WALL:-04:00:00}"
LOOK_WALL="${LOOK_WALL:-08:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"
RF_CAP="${RF_CAP:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v/rolling_notta_h30s_shard0}"
SF_CAP="${SF_CAP:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v/notta_h30s_shard0}"

cd "${PROJECT_ROOT}"
mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"

if [[ "${SERIES}" == "v2v_panda_rolling_leftovers_8v" ]]; then
    echo "ERROR: refuse overwrite of stem leftover dir. Use v2v_panda_caption_leftovers_8v." >&2
    exit 2
fi
if [[ ! -d "${VIDEO_DIR}" ]]; then
    echo "ERROR: ${VIDEO_DIR} missing." >&2
    exit 1
fi
if [[ ! -f "${VIDEO_DIR}/metadata.csv" ]]; then
    echo "ERROR: ${VIDEO_DIR}/metadata.csv missing." >&2
    exit 1
fi
if [[ ! -f /scratch/${USER}/wan-checkpoints/rolling_forcing_dmd.pt ]]; then
    echo "ERROR: Rolling Forcing ckpt missing." >&2
    exit 1
fi
if [[ ! -d "${RF_CAP}" ]]; then
    echo "ERROR: caption Rolling host missing: ${RF_CAP}" >&2
    exit 1
fi

echo "---- caption leftover preflight (first-segment, not stem) ----"
python3 - <<'PY'
from pathlib import Path
import sys
sys.path.insert(0, ".")
from scripts.caption_utils import load_resolved_captions_csv, canonical_video_id
root = Path("datasets/panda_1000_480p")
caps = load_resolved_captions_csv(root / "metadata.csv", warn_missing=False)
vids = sorted(p for p in root.rglob("*.mp4"))[:8]
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
if bad or len(caps) < 8:
    raise SystemExit("preflight failed")
PY

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},VIDEO_WORKERS=1,LIVE_MIN=0.012,PSEUDO_GAMMA=0.0,NOISE_TAU=0.04"

JOBS=()
METHODS_RUN=()
submit_method() {
    local method="$1"
    local k="$2"
    local wall="$3"
    local J
    J=$(sbatch --parsable --account="${ACCOUNT}" --time="${wall}" \
        --export=ALL,METHOD="${method}",SEARCH_K="${k}",${COMMON} \
        "${SB}/run_v2v_chunked.sbatch")
    echo "V2V ${SERIES} ${method} k=${k} n=${N_VIDEOS} job ${J}"
    JOBS+=("${J}")
    METHODS_RUN+=("${method}")
}

# RF-host knobs only. ρ is a Rolling window schedule, not an SF widget.
submit_method rolling_rho_lo 1 "${ROLL_WALL}"
submit_method rolling_rho_hi 1 "${ROLL_WALL}"
submit_method rolling_adapt 1 "${ROLL_WALL}"
submit_method rolling_look 4 "${LOOK_WALL}"

ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
VIDEO_DIRS=""
for m in "${METHODS_RUN[@]}"; do
    VIDEO_DIRS="${VIDEO_DIRS} ${ROOT}/${m}_h30s_shard0"
done
VIDEO_DIRS="${VIDEO_DIRS} ${RF_CAP} ${SF_CAP}"
DEPS=$(IFS=:; echo "${JOBS[*]}")
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --dependency="afterok:${DEPS}" \
    --export=ALL,SERIES_DIR="${ROOT}",VIDEO_DIRS="${VIDEO_DIRS}",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench full-clip job ${VB} afterok ${DEPS}"
JOBS+=("${VB}")

echo "Caption leftovers N=${N_VIDEOS}. Cite vs caption Rolling, not stem leftovers."
echo "Sidecar must be prompt_source=metadata_csv. If the first is stem, scancel this wave only:"
echo "  scancel ${JOBS[*]}"
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v \\"
echo "    --allow-partial"
echo "  python3 -u wan_experiment/scripts/pair_v2v_tails.py \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v \\"
echo "    --series-dir ${ROOT}"
echo "No TTC. No I2V. Do not remake cite-128. Do not write caption numbers into stem leftover tables."
