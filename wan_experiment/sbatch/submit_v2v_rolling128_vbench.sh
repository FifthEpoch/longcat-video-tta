#!/bin/bash
# Resume 128 rolling_notta VBench after the ~2 h preemption.
# Existing per-dim eval_results are skipped (no FORCE).
# 16209128 died mid-subject; 16228045 wrote through dynamic_degree
# and died at temporal_flickering. joined.json is still missing.
#
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   # login CPU: join whatever dims exist (subject/IQ/Dyn enough to decide)
#   JOIN_ONLY=1 bash wan_experiment/sbatch/submit_v2v_rolling128_vbench.sh
#   # GPU: remaining dims only (usually flickering; + subject if n<128)
#   bash wan_experiment/sbatch/submit_v2v_rolling128_vbench.sh

set -euo pipefail

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
SB="${PROJECT_ROOT}/wan_experiment/sbatch"
SERIES="${SERIES:-v2v_panda_rolling_128v}"
ROOT="${PROJECT_ROOT}/wan_experiment/results/${SERIES}"
OUT="${ROOT}/rolling_notta_h30s_shard0"
VBDIR="${OUT}/vbench_full"
N_EXPECT="${N_EXPECT:-128}"
VBENCH_WALL="${VBENCH_WALL:-04:00:00}"
JOIN_ONLY="${JOIN_ONLY:-0}"

if [ ! -d "${OUT}" ]; then
    echo "ERROR: missing ${OUT}" >&2
    exit 1
fi

echo "=============================================================================="
echo "128 rolling VBench resume"
echo "  OUT    : ${OUT}"
echo "  VBDIR  : ${VBDIR}"
echo "=============================================================================="
ls -1 "${VBDIR}" 2>/dev/null | sed 's/^/  /' || echo "  (no vbench_full yet)"

python3 - "${VBDIR}" "${N_EXPECT}" <<'PY'
import json, sys
from pathlib import Path
vb = Path(sys.argv[1])
n_expect = int(sys.argv[2])
dims = [
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "imaging_quality", "motion_smoothness", "dynamic_degree",
    "temporal_flickering",
]
print("per-dim n:")
for dim in dims:
    p = vb / f"vbench_{dim}_eval_results.json"
    if not p.is_file():
        print(f"  {dim:24s}  MISSING")
        continue
    try:
        data = json.loads(p.read_text())
    except Exception as exc:
        print(f"  {dim:24s}  PARSE FAIL {exc}")
        continue
    body = data.get(dim) or (next(iter(data.values())) if data else None)
    items = body
    if isinstance(body, list) and body and isinstance(body[0], (int, float)) and len(body) >= 2:
        items = body[1]
    n = 0
    if isinstance(items, list):
        n = sum(1 for it in items if isinstance(it, dict))
    elif isinstance(items, dict):
        n = len(items)
    flag = "  INCOMPLETE" if n < n_expect else ""
    print(f"  {dim:24s}  n={n}{flag}")
print(f"  joined.json           {'OK' if (vb / 'joined.json').is_file() else 'MISSING'}")
PY

if [ "${JOIN_ONLY}" = "1" ]; then
    python3 "${PROJECT_ROOT}/wan_experiment/scripts/score_i2v_vbench.py" \
        --video-dir "${OUT}" \
        --clip full \
        --join-only || true
    python3 -u "${PROJECT_ROOT}/wan_experiment/scripts/analyze_v2v_bakeoff.py" \
        --series-dir "${ROOT}" --allow-partial
    echo "Join-only done. If flickering is MISSING, submit the GPU resume (no JOIN_ONLY)."
    exit 0
fi

# Skip-existing would keep a killed 91/128 subject file. Drop incomplete
# eval_results so the resume re-runs only those dims.
python3 - "${VBDIR}" "${N_EXPECT}" <<'PY'
import json, sys
from pathlib import Path
vb = Path(sys.argv[1])
n_expect = int(sys.argv[2])
dims = [
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "imaging_quality", "motion_smoothness", "dynamic_degree",
    "temporal_flickering",
]
for dim in dims:
    p = vb / f"vbench_{dim}_eval_results.json"
    if not p.is_file():
        continue
    try:
        data = json.loads(p.read_text())
    except Exception:
        print(f"drop unreadable {p.name}")
        p.unlink()
        continue
    body = data.get(dim) or (next(iter(data.values())) if data else None)
    items = body
    if isinstance(body, list) and body and isinstance(body[0], (int, float)) and len(body) >= 2:
        items = body[1]
    n = 0
    if isinstance(items, list):
        n = sum(1 for it in items if isinstance(it, dict))
    elif isinstance(items, dict):
        n = len(items)
    if n < n_expect:
        print(f"drop incomplete {p.name} n={n} want {n_expect}")
        p.unlink()
PY

mkdir -p "${PROJECT_ROOT}/wan_experiment/slurm_log"
VB=$(sbatch --parsable --account="${ACCOUNT}" --time="${VBENCH_WALL}" \
    --export=ALL,SERIES_DIR="${ROOT}",METHODS="rolling_notta",CLIPS=full \
    "${SB}/run_i2v_vbench.sbatch")
echo "VBench rolling-128 remaining dims job ${VB}  wall ${VBENCH_WALL}"
echo "Skip-existing is on. Cancel this job only:  scancel ${VB}"
