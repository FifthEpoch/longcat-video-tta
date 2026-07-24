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
echo "=== verify gen-only vbench present + old means preserved ==="
for a in "${ARMS[@]}"; do
    if jq -e '.vbench_fullclip_deprecated and .vbench.aesthetic_quality' \
        "${SERIES}/${a}/merged_summary.json" >/dev/null 2>&1; then
        echo "  ${a}: vbench ok"
    else
        echo "  ${a}: vbench MISSING/incomplete"; fail=$((fail + 1))
    fi
done

[ "${fail}" = "0" ] || { echo "ERROR: ${fail} arm(s) failed fold/verify" >&2; exit 1; }
echo "fold: all arms OK"
