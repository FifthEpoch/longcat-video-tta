#!/bin/bash
# Caption N=8 FIFO lookahead + lock-score.
# Diagnostic. Separate from mixctx. Do not remake cite-128.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   bash wan_experiment/sbatch/submit_v2v_caption_fifo_tscore.sh
#
# No TTC. No I2V. VIDEO_WORKERS=1. k=1.

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
VIDEO_DIR="${VIDEO_DIR:-${PROJECT_ROOT}/datasets/panda_1000_480p}"
SERIES="${SERIES:-v2v_panda_caption_fifo_tscore_8v}"
N_VIDEOS="${N_VIDEOS:-8}"
if [[ "${SMOKE:-0}" == "1" ]]; then
    N_VIDEOS=2
    SERIES="${SERIES}_smoke"
fi
GEN_WALL="${GEN_WALL:-04:00:00}"
VBENCH_WALL="${VBENCH_WALL:-08:00:00}"
RF_CAP="${RF_CAP:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v/rolling_notta_h30s_shard0}"
SF_CAP="${SF_CAP:-${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v/notta_h30s_shard0}"

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
if [[ ! -f /scratch/${USER}/wan-checkpoints/rolling_forcing_dmd.pt ]]; then
    echo "ERROR: Rolling Forcing ckpt missing." >&2
    exit 1
fi
if [[ ! -f /scratch/${USER}/wan-checkpoints/self_forcing_dmd.pt ]]; then
    echo "ERROR: Self Forcing ckpt missing." >&2
    exit 1
fi
if [[ ! -d "${RF_CAP}" || ! -d "${SF_CAP}" ]]; then
    echo "ERROR: caption-32 hosts missing." >&2
    exit 1
fi

echo "---- caption fifo/tscore preflight (first-segment, not stem) ----"
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

COMMON="HORIZON_S=30,N_VIDEOS=${N_VIDEOS},SEED=0,SEARCH_FROM=0,PREFIX_LATENTS=9,CHUNK_LATENTS=21,SERIES=${SERIES},NUM_SHARDS=1,VIDEO_DIR=${VIDEO_DIR},VIDEO_WORKERS=1"

JOBS=()
METHODS_RUN=()
submit_method() {
    local method="$1"
    local J
    J=$(sbatch --parsable --account="${ACCOUNT}" --time="${GEN_WALL}" \
        --export=ALL,METHOD="${method}",SEARCH_K=1,${COMMON} \
        "${SB}/run_v2v_chunked.sbatch")
    echo "V2V ${SERIES} ${method} k=1 n=${N_VIDEOS} job ${J}"
    JOBS+=("${J}")
    METHODS_RUN+=("${method}")
}

submit_method rolling_fifo
submit_method rolling_fifo_sick
submit_method rf_tscore
submit_method rf_tscore_always
submit_method sf_tscore
submit_method sf_tscore_always

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

echo "Caption fifo/tscore N=${N_VIDEOS}. Cite RF vs Rolling first-8; SF vs SF first-8."
echo "fifo sidecar fifo_n>0. tscore sidecar rewind_logs score0."
echo "If stem, scancel this wave only:"
echo "  scancel ${JOBS[*]}"
echo "When generate finishes:"
echo "  python3 -u wan_experiment/scripts/analyze_v2v_bakeoff.py \\"
echo "    --series-dir ${ROOT} \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v \\"
echo "    --allow-partial"
echo "  python3 -u wan_experiment/scripts/pair_v2v_tails.py \\"
echo "    --baseline-dir ${PROJECT_ROOT}/wan_experiment/results/v2v_panda_caption_32v \\"
echo "    --series-dir ${ROOT}"
echo "No TTC. No I2V. Do not remake cite-128."
