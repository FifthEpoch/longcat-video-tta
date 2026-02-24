#!/bin/bash
# ============================================================================
# Comprehensive TTA Experiment Progress Checker
#
# Usage: bash sweep_experiment/scripts/check_all_progress.sh
# ============================================================================

set -euo pipefail
PROJECT_ROOT="/scratch/wc3013/longcat-video-tta"
RESULTS_ROOT="${PROJECT_ROOT}/sweep_experiment/results"
BASE_RESULTS="${PROJECT_ROOT}/baseline_experiment/results"

echo "================================================================================"
echo "       COMPREHENSIVE STATUS REPORT ($(date))"
echo "================================================================================"

# ── Helper: extract metrics from TTA sweep summary.json ──
extract() {
    local dir="$1"
    local summary="${dir}/summary.json"
    if [ -f "$summary" ]; then
        python3 -c "
import json, statistics
with open('$summary') as f:
    d = json.load(f)

# TTA sweep format: results array with per-video dicts
results = d.get('results', [])
if results:
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
# Baseline format: metrics.psnr.mean at top level
elif 'metrics' in d and 'psnr' in d['metrics']:
    pm = d['metrics']['psnr'].get('mean', 0)
    ps = d['metrics']['psnr'].get('std', 0)
    n_ok = d.get('num_successful', '?')
    n_total = d.get('num_videos', '?')
    print(f'{n_ok}/{n_total} ok, PSNR={pm:.2f}+/-{ps:.2f}')
else:
    n_total = d.get('num_videos', '?')
    print(f'0/{n_total} ok, unrecognized format')
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

# ── Helper: list subdirs and extract ──
scan_series() {
    local base_dir="$1"
    local pattern="${2:-*}"
    if [ -d "$base_dir" ]; then
        for d in "${base_dir}"/${pattern}; do
            [ -d "$d" ] || continue
            echo "  $(basename $d): $(extract $d)"
        done
    else
        echo "  (directory not found: $base_dir)"
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
echo ""
echo "--- Baseline Sweep (baseline_experiment) ---"
for d in "${BASE_RESULTS}"/cond*_gen* "${BASE_RESULTS}"/baseline_480p; do
    [ -d "$d" ] || continue
    echo "  $(basename $d): $(extract $d)"
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
    scan_series "${RESULTS_ROOT}/exp3_train_frames_${method}" "TF_*"
done

echo ""
echo "--- Exp 4: Generation Horizon Ablation ---"
for method in full delta_a delta_b delta_c lora; do
    echo "  [$method]"
    scan_series "${RESULTS_ROOT}/exp4_gen_horizon_${method}" "GH_*"
done

# ============================================================================
# EXPERIMENT 2: FULL-MODEL TTA
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 2: FULL-MODEL TTA"
echo "================================================================================"
echo "--- LR Sweep (Series 1) ---"
scan_series "${RESULTS_ROOT}/full_lr_sweep" "F*"
echo "--- Iteration Sweep (Series 2) ---"
scan_series "${RESULTS_ROOT}/full_iter_sweep" "F*"
echo "--- Long Train with ES (Series 17) ---"
scan_series "${RESULTS_ROOT}/full_long_train" "FLT*"

# ============================================================================
# EXPERIMENT 3: SINGLE VECTOR TTA (AdaSteer-1 / Delta-A)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 3: SINGLE VECTOR TTA (AdaSteer-1)"
echo "================================================================================"
echo "--- LR Sweep (Series 7) ---"
scan_series "${RESULTS_ROOT}/delta_a_lr_sweep" "DA*"
echo "--- Iteration Sweep (Series 8) ---"
scan_series "${RESULTS_ROOT}/delta_a_iter_sweep" "DA*"
echo "--- Long Train with ES (Series 18) ---"
scan_series "${RESULTS_ROOT}/delta_a_long_train" "DALT*"
echo "--- Optimized: More Frames + Extended Training (Series 23) ---"
scan_series "${RESULTS_ROOT}/delta_a_optimized" "DAO*"
echo "--- Combined: AdaSteer + Norm Tuning (Series 23b) ---"
scan_series "${RESULTS_ROOT}/delta_a_norm_combined" "DAO*"

# ============================================================================
# EXPERIMENT 4: MULTIPLE VECTOR TTA (AdaSteer G>1 / Delta-B)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 4: MULTIPLE VECTOR TTA (AdaSteer G>1)"
echo "================================================================================"
echo "--- Groups Sweep (Series 9, 20 steps, lr=1e-2) ---"
scan_series "${RESULTS_ROOT}/delta_b_groups_sweep" "DB*"
echo "--- LR Sweep (Series 10, G=4, 20 steps) ---"
scan_series "${RESULTS_ROOT}/delta_b_lr_sweep" "DB*"
echo "--- Iteration Sweep (Series 9b, G=1, lr=1e-2) ---"
scan_series "${RESULTS_ROOT}/delta_b_iter_sweep" "DB*"
echo "--- Low-LR Sweep (Series 24, G=1, 20 steps) ---"
scan_series "${RESULTS_ROOT}/delta_b_low_lr" "DBL*"
echo "--- Hidden-State Residual (Series 19) ---"
scan_series "${RESULTS_ROOT}/delta_b_hidden_sweep" "DBH*"
echo "--- Equivalence Verification (Series 25) ---"
scan_series "${RESULTS_ROOT}/delta_a_equiv_verify" "DAV*"
echo "--- Groups at 5 Steps (Series 26) ---"
scan_series "${RESULTS_ROOT}/adasteer_groups_5step" "AS*"
echo "--- Ratio Sweep: Frames x Groups (Series 27) ---"
scan_series "${RESULTS_ROOT}/adasteer_ratio_sweep" "AR*"

# ============================================================================
# EXPERIMENT 5: LoRA TTA
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 5: LoRA TTA"
echo "================================================================================"
echo "--- Rank Sweep (Series 3) ---"
scan_series "${RESULTS_ROOT}/lora_rank_sweep" "L*"
echo "--- Iteration Sweep (Series 6) ---"
scan_series "${RESULTS_ROOT}/lora_iter_sweep" "L*"
echo "--- Constrained Sweep (Series 14) ---"
scan_series "${RESULTS_ROOT}/lora_constrained_sweep" "LA*"
echo "--- Ultra-Constrained (Series 16) ---"
scan_series "${RESULTS_ROOT}/lora_ultra_constrained" "LB*"
echo "--- Built-in Comparison (Series 15) ---"
scan_series "${RESULTS_ROOT}/lora_builtin_comparison" "LN*"

# ============================================================================
# EXPERIMENT 6: OUTPUT RESIDUAL (Delta-C)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 6: OUTPUT RESIDUAL (Delta-C)"
echo "================================================================================"
echo "--- LR Sweep (Series 11) ---"
scan_series "${RESULTS_ROOT}/delta_c_lr_sweep" "DC*"
echo "--- Iteration Sweep (Series 12) ---"
scan_series "${RESULTS_ROOT}/delta_c_iter_sweep" "DC*"

# ============================================================================
# EARLY STOPPING ABLATION
# ============================================================================
echo ""
echo "================================================================================"
echo "  EARLY STOPPING ABLATION"
echo "================================================================================"
for prefix in es_ablation_check_freq es_ablation_disable es_ablation_holdout \
              es_ablation_noise_draws es_ablation_patience es_ablation_sigmas; do
    echo "--- ${prefix} ---"
    scan_series "${RESULTS_ROOT}/${prefix}" "ES_*"
done

# ============================================================================
# EXTENDED EXPERIMENTS (Series 21-22)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXTENDED EXPERIMENTS"
echo "================================================================================"
echo "--- Norm Tuning / TENT-style (Series 21) ---"
scan_series "${RESULTS_ROOT}/norm_tune_sweep" "NT*"
echo "--- FiLM Adapter (Series 22) ---"
scan_series "${RESULTS_ROOT}/film_adapter_sweep" "FM*"
echo "--- Best Methods with Proper ES ---"
scan_series "${RESULTS_ROOT}/best_methods_proper_es"

# ============================================================================
# NEW SERIES (28-29)
# ============================================================================
echo ""
echo "================================================================================"
echo "  NEW SERIES (28-29)"
echo "================================================================================"
echo "--- Extended Data (Series 28) ---"
scan_series "${RESULTS_ROOT}/adasteer_extended_data" "EX_*"
echo "--- Parameter Dimension Sweep (Series 29) ---"
scan_series "${RESULTS_ROOT}/adasteer_param_sweep"

# ============================================================================
# UCF-101 CROSS-DATASET
# ============================================================================
echo ""
echo "================================================================================"
echo "  UCF-101 CROSS-DATASET"
echo "================================================================================"
scan_series "${RESULTS_ROOT}/ucf101_no_tta" "UCF_*"
scan_series "${RESULTS_ROOT}/ucf101_full" "UCF_*"
scan_series "${RESULTS_ROOT}/ucf101_delta_a" "UCF_*"
scan_series "${RESULTS_ROOT}/ucf101_lora" "UCF_*"

# ============================================================================
# BACKBONE EXPERIMENTS
# ============================================================================
echo ""
echo "================================================================================"
echo "  BACKBONE EXPERIMENTS"
echo "================================================================================"
echo "--- Open-Sora v2.0 ---"
if [ -d "${PROJECT_ROOT}/backbone_experiment/opensora/results" ]; then
    for d in "${PROJECT_ROOT}/backbone_experiment/opensora/results"/*/; do
        [ -d "$d" ] || continue
        echo "  $(basename $d): $(extract $d)"
    done
else
    echo "  (not started)"
fi
echo "--- CogVideoX-5B-I2V ---"
if [ -d "${PROJECT_ROOT}/backbone_experiment/cogvideo/results" ]; then
    for d in "${PROJECT_ROOT}/backbone_experiment/cogvideo/results"/*/; do
        [ -d "$d" ] || continue
        echo "  $(basename $d): $(extract $d)"
    done
else
    echo "  (not started)"
fi

# ============================================================================
# SUMMARY
# ============================================================================
echo ""
echo "================================================================================"
echo "                              SUMMARY COUNTS"
echo "================================================================================"
complete=0
in_prog=0
for series_dir in "${RESULTS_ROOT}"/*/; do
    [ -d "$series_dir" ] || continue
    for run_dir in "${series_dir}"*/; do
        [ -d "$run_dir" ] || continue
        if [ -f "${run_dir}summary.json" ]; then
            complete=$((complete + 1))
        elif [ -f "${run_dir}checkpoint.json" ]; then
            in_prog=$((in_prog + 1))
        fi
    done
done
echo "  Complete (summary.json): ${complete}"
echo "  In-progress (checkpoint only): ${in_prog}"
echo "================================================================================"
