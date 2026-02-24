#!/bin/bash
# ============================================================================
# Comprehensive TTA Experiment Progress Checker
#
# Usage: bash sweep_experiment/scripts/check_all_progress.sh
# ============================================================================

set -euo pipefail
PROJECT_ROOT="/scratch/wc3013/longcat-video-tta"
RESULTS_ROOT="${PROJECT_ROOT}/sweep_experiment/results"

echo "================================================================================"
echo "       COMPREHENSIVE STATUS REPORT ($(date))"
echo "================================================================================"

# ── Helper: extract metrics from summary.json ──
extract() {
    local dir="$1"
    local summary="${dir}/summary.json"
    if [ -f "$summary" ]; then
        python3 -c "
import json, statistics
with open('$summary') as f:
    d = json.load(f)
results = d.get('results', [])
ok = [r for r in results if r.get('success', False)]
psnrs = [r['psnr'] for r in ok if 'psnr' in r]
n_total = d.get('num_videos', len(results))
n_ok = len(ok)
if psnrs:
    m = statistics.mean(psnrs)
    s = statistics.stdev(psnrs) if len(psnrs) > 1 else 0
    print(f'{n_ok}/{n_total} ok, PSNR={m:.2f}+/-{s:.2f}')
else:
    print(f'{n_ok}/{n_total} ok, no PSNR data')
" 2>/dev/null || echo "parse error"
    elif [ -f "${dir}/checkpoint.json" ]; then
        python3 -c "
import json
with open('${dir}/checkpoint.json') as f:
    d = json.load(f)
idx = d.get('next_idx', 0)
n = len(d.get('results', []))
print(f'in-progress: {n} videos (checkpoint next_idx={idx})')
" 2>/dev/null || echo "checkpoint exists, parse error"
    elif [ -d "$dir" ]; then
        echo "dir exists, no data"
    else
        echo "NOT STARTED"
    fi
}

# ── Currently running SLURM jobs ──
echo ""
echo "=== CURRENTLY RUNNING JOBS ==="
squeue -u wc3013 2>/dev/null || echo "(squeue not available)"

# ============================================================================
# EXPERIMENT 0: BASELINES
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 0: BASELINES (no TTA)"
echo "================================================================================"
echo "  Panda-70M no-TTA:  $(extract ${RESULTS_ROOT}/panda_no_tta_continuation/NOTTA)"
echo "  UCF-101 no-TTA:    $(extract ${RESULTS_ROOT}/ucf101_no_tta/UCF_NOTTA)"

# baseline_experiment results
BASE_RESULTS="${PROJECT_ROOT}/baseline_experiment/results"
for d in "${BASE_RESULTS}"/cond*_gen* "${BASE_RESULTS}"/baseline_480p; do
    if [ -d "$d" ] && [ -f "$d/summary.json" ]; then
        name=$(basename "$d")
        echo "  baseline/$name: $(extract $d)"
    fi
done

# ============================================================================
# EXPERIMENT 1: CONDITIONING + GENERATION LENGTH
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 1: CONDITIONING + GENERATION LENGTH"
echo "================================================================================"
echo ""
echo "--- Exp 3: Training Frames Ablation ---"
for method in full delta_a delta_b delta_c lora; do
    echo "  [$method]"
    for d in "${RESULTS_ROOT}"/exp3_train_frames_${method}/TF_*; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        echo "    $name: $(extract $d)"
    done
done

echo ""
echo "--- Exp 4: Generation Horizon Ablation ---"
for method in full delta_a delta_b delta_c lora; do
    echo "  [$method]"
    for d in "${RESULTS_ROOT}"/exp4_gen_horizon_${method}/GH_*; do
        [ -d "$d" ] || continue
        name=$(basename "$d")
        echo "    $name: $(extract $d)"
    done
done

# ============================================================================
# EXPERIMENT 2: FULL-MODEL TTA
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 2: FULL-MODEL TTA"
echo "================================================================================"
echo ""
echo "--- LR Sweep (Series 1) ---"
for d in "${RESULTS_ROOT}"/full_lr_sweep/F*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Iteration Sweep (Series 2) ---"
for d in "${RESULTS_ROOT}"/full_iter_sweep/F*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Long Train (Series 17) ---"
for d in "${RESULTS_ROOT}"/full_long_train/FLT*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done

# ============================================================================
# EXPERIMENT 3: SINGLE VECTOR TTA (AdaSteer-1 / Delta-A)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 3: SINGLE VECTOR TTA (AdaSteer-1)"
echo "================================================================================"
echo ""
echo "--- LR Sweep (Series 7) ---"
for d in "${RESULTS_ROOT}"/delta_a_lr_sweep/DA*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Iteration Sweep (Series 8) ---"
for d in "${RESULTS_ROOT}"/delta_a_iter_sweep/DA*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Long Train (Series 18) ---"
for d in "${RESULTS_ROOT}"/delta_a_long_train/DALT*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Optimized (Series 23/23b) ---"
for d in "${RESULTS_ROOT}"/adasteer_optimized/DAO* "${RESULTS_ROOT}"/adasteer_norm_combined/DAO*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done

# ============================================================================
# EXPERIMENT 4: MULTIPLE VECTOR TTA (AdaSteer G>1 / Delta-B)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 4: MULTIPLE VECTOR TTA (AdaSteer G>1)"
echo "================================================================================"
echo ""
echo "--- Groups Sweep (Series 9) ---"
for d in "${RESULTS_ROOT}"/delta_b_groups_sweep/DB*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- LR Sweep (Series 10) ---"
for d in "${RESULTS_ROOT}"/delta_b_lr_sweep/DB*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Iteration Sweep (Series 9b) ---"
for d in "${RESULTS_ROOT}"/delta_b_iter_sweep/DB*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Low-LR Sweep (Series 24) ---"
for d in "${RESULTS_ROOT}"/adasteer_low_lr/DBL*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Hidden-State Residual (Series 19) ---"
for d in "${RESULTS_ROOT}"/delta_b_hidden/DBH*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Equivalence Verification (Series 25) ---"
for d in "${RESULTS_ROOT}"/adasteer_equiv_verify/DAV*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Groups at 5 Steps (Series 26) ---"
for d in "${RESULTS_ROOT}"/adasteer_groups_5step/AS*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Ratio Sweep (Series 27) ---"
for d in "${RESULTS_ROOT}"/adasteer_ratio_sweep/AR*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done

# ============================================================================
# EXPERIMENT 5: LoRA TTA
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 5: LoRA TTA"
echo "================================================================================"
echo ""
echo "--- Rank Sweep (Series 3) ---"
for d in "${RESULTS_ROOT}"/lora_rank_sweep/L*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Iteration Sweep (Series 6) ---"
for d in "${RESULTS_ROOT}"/lora_iter_sweep/L*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Constrained Sweep (Series 14) ---"
for d in "${RESULTS_ROOT}"/lora_constrained_sweep/LA*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Ultra-Constrained (Series 16) ---"
for d in "${RESULTS_ROOT}"/lora_ultra_constrained/LB*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Built-in Comparison (Series 15) ---"
for d in "${RESULTS_ROOT}"/lora_builtin_comparison/LN*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done

# ============================================================================
# EXPERIMENT 6: OUTPUT RESIDUAL (Delta-C)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 6: OUTPUT RESIDUAL (Delta-C)"
echo "================================================================================"
echo ""
echo "--- LR Sweep (Series 11) ---"
for d in "${RESULTS_ROOT}"/delta_c_lr_sweep/DC*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Iteration Sweep (Series 12) ---"
for d in "${RESULTS_ROOT}"/delta_c_iter_sweep/DC*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done

# ============================================================================
# EARLY STOPPING ABLATION
# ============================================================================
echo ""
echo "================================================================================"
echo "  EARLY STOPPING ABLATION"
echo "================================================================================"
for prefix in es_ablation_check_freq es_ablation_disable es_ablation_holdout \
              es_ablation_noise_draws es_ablation_patience es_ablation_sigmas; do
    for d in "${RESULTS_ROOT}/${prefix}"/ES_* "${RESULTS_ROOT}/${prefix}"/ES*; do
        [ -d "$d" ] || continue
        echo "  $(basename $(dirname $d))/$(basename $d): $(extract $d)"
    done
done

# ============================================================================
# EXTENDED EXPERIMENTS (Series 21-22)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXTENDED EXPERIMENTS"
echo "================================================================================"
echo ""
echo "--- Norm Tuning / TENT-style (Series 21) ---"
for d in "${RESULTS_ROOT}"/norm_tune_sweep/NT*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- FiLM Adapter (Series 22) ---"
for d in "${RESULTS_ROOT}"/film_adapter_sweep/FM*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done

# ============================================================================
# NEW SERIES (28-29) if submitted
# ============================================================================
echo ""
echo "================================================================================"
echo "  NEW SERIES (28-29)"
echo "================================================================================"
echo ""
echo "--- Extended Data (Series 28) ---"
for d in "${RESULTS_ROOT}"/adasteer_extended_data/EX_*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done
echo ""
echo "--- Parameter Dimension Sweep (Series 29) ---"
for d in "${RESULTS_ROOT}"/adasteer_param_sweep/PD* "${RESULTS_ROOT}"/adasteer_param_sweep/PG*; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
done

# ============================================================================
# UCF-101 CROSS-DATASET
# ============================================================================
echo ""
echo "================================================================================"
echo "  UCF-101 CROSS-DATASET"
echo "================================================================================"
for d in "${RESULTS_ROOT}"/ucf101_*/UCF_*; do
    [ -d "$d" ] || continue
    echo "  $(dirname $d | xargs basename)/$(basename $d): $(extract $d)"
done

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "                              SUMMARY COUNTS"
echo "================================================================================"
total_dirs=$(find "${RESULTS_ROOT}" -maxdepth 2 -name "summary.json" 2>/dev/null | wc -l)
in_progress=$(find "${RESULTS_ROOT}" -maxdepth 2 -name "checkpoint.json" ! -path "*/summary.json" 2>/dev/null | while read ckpt; do
    dir=$(dirname "$ckpt")
    [ -f "$dir/summary.json" ] || echo "$dir"
done | wc -l)
echo "  Complete (summary.json): ${total_dirs}"
echo "  In-progress (checkpoint only): ${in_progress}"
echo "================================================================================"
