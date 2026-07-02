#!/usr/bin/env bash
# ============================================================================
# Submit all GPU/CPU jobs for ~4-day PI review (VBench++ narrative).
#
# Phases (set SKIP_*=1 to skip):
#   A  LoRA R1 @ 999v (10 GPU jobs, inline VBench++)
#   B  Budget best configs @ 999v with VBench (30 GPU jobs default)
#   C  Pilot feature backfill for router N=200 (GPU fan-out)
#
# Usage:
#   cd /scratch/wc3013/longcat-video-tta && git pull --ff-only origin main
#   DRY_RUN=1 bash sweep_experiment/sbatch/submit_review_experiments.sh
#   bash sweep_experiment/sbatch/submit_review_experiments.sh
#
# Monitor:
#   squeue -u $USER
#   cat sweep_experiment/reports/PROJECT_STATUS.md
# ============================================================================
set -euo pipefail

REPO="${REPO:-/scratch/${USER}/longcat-video-tta}"
ACCOUNT="${ACCOUNT:-torch_pr_36_mren}"
DRY_RUN="${DRY_RUN:-0}"

SKIP_LORA_R1="${SKIP_LORA_R1:-0}"
SKIP_BUDGET_VBENCH="${SKIP_BUDGET_VBENCH:-0}"
SKIP_PILOT_FEATURES="${SKIP_PILOT_FEATURES:-0}"

cd "${REPO}"

_run() {
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY] $*"
        DRY_RUN=1 "$@"
    else
        "$@"
    fi
}

echo "============================================================"
echo "Review experiment batch — VBench++ focus"
echo "  account: ${ACCOUNT}  dry_run: ${DRY_RUN}"
echo "============================================================"

if [ "${SKIP_LORA_R1}" != "1" ]; then
    echo ""
    echo "=== Phase A: LoRA R1 @ 999v (mirror R8, COMPUTE_VBENCH=1) ==="
    _run bash sweep_experiment/sbatch/submit_lora_r1_1000v_panda.sh
else
    echo "SKIP Phase A (LoRA R1)"
fi

if [ "${SKIP_BUDGET_VBENCH}" != "1" ]; then
    echo ""
    echo "=== Phase B: Budget S2/S5/S10 @ 999v + inline VBench++ ==="
    _run bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_vbench_review.sh
else
    echo "SKIP Phase B (budget 1000v VBench)"
fi

if [ "${SKIP_PILOT_FEATURES}" != "1" ]; then
    echo ""
    echo "=== Phase C: Pilot 200v Phase-0 features (router N=200) ==="
    if [ "${DRY_RUN}" = "1" ]; then
        echo "[DRY] bash sweep_experiment/sbatch/submit_pilot_router_features.sh"
    else
        bash sweep_experiment/sbatch/submit_pilot_router_features.sh
    fi
else
    echo "SKIP Phase C (pilot features)"
fi

echo ""
echo "============================================================"
echo "Submitted review batch. Timeline (~4 days if queue cooperates):"
echo "  Day 0–2: GPU sweeps finish → merge_chunks + update_merged_with_vbench"
echo "  Day 2–3: Refresh VBench gains + router + budget VBench oracle @ 999v"
echo "  Day 4:   PI narrative from PROJECT_STATUS + new 999v VBench table"
echo ""
echo "Post-merge commands (run when jobs complete):"
echo "  bash scripts/run_review_analysis_when_ready.sh"
echo "============================================================"
