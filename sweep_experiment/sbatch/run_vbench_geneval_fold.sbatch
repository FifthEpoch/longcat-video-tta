#!/bin/bash
#SBATCH --job-name=vb_geneval_fold
#SBATCH --account=torch_pr_36_mren
#SBATCH --output=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/vb_geneval_fold_%j.out
#SBATCH --error=/scratch/%u/longcat-video-tta/sweep_experiment/slurm_log/vb_geneval_fold_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

# ============================================================================
# Step 3 (sbatch): fold GENERATED-ONLY VBench into merged_summary for all arms.
# Reads vbench_results_geneval/, stashes the old full-clip means under
# merged_summary["vbench_fullclip_deprecated"], rebuilds "vbench" from gen-only.
# CPU-only. Depends on the gen-only VBench backfill jobs (afterok).
#
# Submit:
#   sbatch sweep_experiment/sbatch/run_vbench_geneval_fold.sbatch
# ============================================================================
set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/${USER}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_BASE}/longcat-video-tta}"
SERIES="${SERIES:-sweep_experiment/results/panda_ood_budget_1000v_preview}"

ARMS=(
    NOTTA
    S2_LR1e3 S2_LR5e3 S2_LR1e2
    S5_LR1e3 S5_LR5e3 S5_LR1e2
    S10_LR1e3 S10_LR5e3 S10_LR1e2
    S20_LR1e3 S20_LR5e3 S20_LR1e2
)

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${SCRATCH_BASE}/conda-envs/longcat"

cd "${PROJECT_ROOT}"
mkdir -p sweep_experiment/slurm_log

fail=0
for a in "${ARMS[@]}"; do
    echo "=== fold ${a} ==="
    if python3 scripts/update_merged_with_vbench.py \
        --method-dir "${SERIES}/${a}" --force --deprecate-existing \
        --vbench-subdir vbench_results_geneval --videos-subdir videos_geneval; then
        :
    else
        echo "  [error] fold failed for ${a}" >&2; fail=$((fail + 1))
    fi
done

echo ""
echo "=== verify gen-only vbench folded (7 dims present) ==="
# Python check (no jq dependency). Passes when all 7 gen-only dims are present.
# vbench_fullclip_deprecated is INFORMATIONAL only: it exists only if the
# merged_summary previously held full-clip VBench means (the preview series did
# not, so its absence is expected and not a failure).
python3 - "${SERIES}" "${ARMS[@]}" <<'PY'
import json, sys
from pathlib import Path
series = sys.argv[1]; arms = sys.argv[2:]
DIMS = ["subject_consistency", "background_consistency", "aesthetic_quality",
        "motion_smoothness", "dynamic_degree", "imaging_quality",
        "temporal_flickering"]
bad = 0
for a in arms:
    p = Path(series) / a / "merged_summary.json"
    try:
        s = json.loads(p.read_text())
    except Exception as e:
        print(f"  {a}: cannot read merged_summary.json ({e})"); bad += 1; continue
    vb = s.get("vbench") or {}
    missing = [d for d in DIMS if d not in vb]
    dep = "yes" if "vbench_fullclip_deprecated" in s else "no (none pre-existed)"
    if missing:
        print(f"  {a}: gen-only vbench MISSING dims={missing}"); bad += 1
    else:
        print(f"  {a}: gen-only vbench ok (7 dims)  [deprecated_stash={dep}]")
sys.exit(1 if bad else 0)
PY
pyrc=$?

[ "${fail}" = "0" ] && [ "${pyrc}" = "0" ] \
    || { echo "ERROR: fold/verify failed (fold_fail=${fail} verify_rc=${pyrc})" >&2; exit 1; }
echo "fold: all arms OK"
