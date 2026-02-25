#!/bin/bash
# ============================================================================
# Comprehensive TTA Experiment Progress Checker
#
# Displays: PSNR, SSIM, LPIPS, train/gen time, config, early stopping stats
#
# Usage: bash sweep_experiment/scripts/check_all_progress.sh
# ============================================================================

set -euo pipefail
PROJECT_ROOT="/scratch/wc3013/longcat-video-tta"
RESULTS_ROOT="${PROJECT_ROOT}/sweep_experiment/results"
BASE_RESULTS="${PROJECT_ROOT}/baseline_experiment/results"
EXTRACTOR="${PROJECT_ROOT}/sweep_experiment/scripts/extract_summary.py"

scan() {
    python3 "$EXTRACTOR" "$@"
}

echo "================================================================================"
echo "       COMPREHENSIVE STATUS REPORT ($(date))"
echo "================================================================================"

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
echo "  Goal: Establish video continuation quality without any adaptation"
echo "================================================================================"
echo ""
echo "--- TTA Baseline (sweep format, 14 cond, 14 gen, anchor=32) ---"
scan "${RESULTS_ROOT}/panda_no_tta_continuation" "NOTTA"
scan "${RESULTS_ROOT}/ucf101_no_tta" "UCF_"
echo ""
echo "--- Baseline Sweep (various cond/gen combos, no TTA) ---"
for d in "${BASE_RESULTS}"/cond*_gen* "${BASE_RESULTS}"/baseline_480p; do
    [ -d "$d" ] || continue
    scan "$d"
done

# ============================================================================
# EXPERIMENT 1: CONDITIONING + GENERATION LENGTH
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 1: CONDITIONING + GENERATION LENGTH"
echo "  Goal: How does # of cond frames and gen horizon affect quality?"
echo "  Config: anchor=32, total=28 frames, best LR per method, 20 steps"
echo "================================================================================"
echo ""
echo "--- Exp 3: Training Frames Ablation (2/7/14/24 cond frames) ---"
for method in full delta_a delta_b delta_c lora; do
    echo "  [$method]"
    scan "${RESULTS_ROOT}/exp3_train_frames_${method}" "TF_"
done

echo ""
echo "--- Exp 4: Generation Horizon (16/28/44/72 total frames, 14 cond) ---"
for method in full delta_a delta_b delta_c lora; do
    echo "  [$method]"
    scan "${RESULTS_ROOT}/exp4_gen_horizon_${method}" "GH_"
done

# ============================================================================
# EXPERIMENT 2: FULL-MODEL TTA
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 2: FULL-MODEL TTA"
echo "  Goal: Verify TTA works, find best LR, establish TTA upper bound"
echo "  Config: 13.6B params, SGD, 14 cond, 14 gen, anchor=32"
echo "================================================================================"
echo ""
echo "--- LR Sweep (Series 1): 20 steps, LR={1e-6, 5e-6, 1e-5, 5e-5, 1e-4} ---"
scan "${RESULTS_ROOT}/full_lr_sweep" "F"
echo ""
echo "--- Iteration Sweep (Series 2): lr=1e-5, steps={5, 10, 20, 40, 80} ---"
scan "${RESULTS_ROOT}/full_iter_sweep" "F"
echo ""
echo "--- Long Train + ES (Series 17): lr=1e-5, 500 max steps, ES p=10, 30 vids ---"
scan "${RESULTS_ROOT}/full_long_train" "FLT"

# ============================================================================
# EXPERIMENT 3: SINGLE VECTOR TTA (AdaSteer-1 / Delta-A)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 3: SINGLE VECTOR TTA (AdaSteer-1)"
echo "  Goal: Can 512 params match full-model TTA?"
echo "  Config: 512 params, AdamW, shared timestep embedding offset"
echo "================================================================================"
echo ""
echo "--- LR Sweep (Series 7): 20 steps, LR={1e-3, 5e-3, 1e-2, 5e-2, 1e-1} ---"
scan "${RESULTS_ROOT}/delta_a_lr_sweep" "DA"
echo ""
echo "--- Iteration Sweep (Series 8): lr=5e-3, steps={5, 10, 20, 40, 80} ---"
scan "${RESULTS_ROOT}/delta_a_iter_sweep" "DA"
echo ""
echo "--- Long Train + ES (Series 18): lr=5e-3, 200 max steps, ES, 30 vids ---"
scan "${RESULTS_ROOT}/delta_a_long_train" "DALT"
echo ""
echo "--- Optimized (Series 23): 24 cond frames / 100-step extended ---"
scan "${RESULTS_ROOT}/delta_a_optimized" "DAO"
echo ""
echo "--- Combined: AdaSteer + Norm Tuning (Series 23b) ---"
scan "${RESULTS_ROOT}/delta_a_norm_combined" "DAO"

# ============================================================================
# EXPERIMENT 4: MULTIPLE VECTOR TTA (AdaSteer G>1 / Delta-B)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 4: MULTIPLE VECTOR TTA (AdaSteer G>1)"
echo "  Goal: Does per-block specialization help? Optimal group count?"
echo "  Config: G×512 params, AdamW, per-group timestep offset"
echo "================================================================================"
echo ""
echo "--- Groups Sweep (Series 9): G={1,2,4,8,16,48}, 20 steps, lr=1e-2 ---"
scan "${RESULTS_ROOT}/delta_b_groups_sweep" "DB"
echo ""
echo "--- LR Sweep (Series 10): G=4, 20 steps, LR={1e-3..1e-1} ---"
scan "${RESULTS_ROOT}/delta_b_lr_sweep" "DB"
echo ""
echo "--- Iter Sweep (Series 9b): G=1, lr=1e-2, steps={5,10,40,80} ---"
scan "${RESULTS_ROOT}/delta_b_iter_sweep" "DB"
echo ""
echo "--- Low-LR Sweep (Series 24): G=1, 20 steps, LR={5e-4..5e-3} ---"
scan "${RESULTS_ROOT}/delta_b_low_lr" "DBL"
echo ""
echo "--- Hidden-State Residual (Series 19): hidden target, G={1,4,12}, LR sweep ---"
scan "${RESULTS_ROOT}/delta_b_hidden_sweep" "DBH"
echo ""
echo "--- Equivalence Verification (Series 25): Delta-A vs Delta-B(G=1) ---"
scan "${RESULTS_ROOT}/delta_a_equiv_verify" "DAV"
echo ""
echo "--- Groups at 5 Steps (Series 26): G={1,4,12,48}, lr=1e-2, 5 steps ---"
scan "${RESULTS_ROOT}/adasteer_groups_5step" "AS"
echo ""
echo "--- Ratio Sweep (Series 27): cond={2,14,24} × G={1,4,12}, 5 steps ---"
scan "${RESULTS_ROOT}/adasteer_ratio_sweep" "AR"

# ============================================================================
# EXPERIMENT 5: LoRA TTA
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 5: LoRA TTA"
echo "  Goal: Standard PEFT baseline for TTA comparison"
echo "  Config: AdamW, qkv+proj targets across 48 blocks"
echo "================================================================================"
echo ""
echo "--- Rank Sweep (Series 3): r={1,4,8,16,32}, α=2r, lr=2e-4, 20 steps ---"
scan "${RESULTS_ROOT}/lora_rank_sweep" "L"
echo ""
echo "--- Iter Sweep (Series 6): r=1, lr=2e-4, steps={5,10,20,40} ---"
scan "${RESULTS_ROOT}/lora_iter_sweep" "L"
echo ""
echo "--- Constrained (Series 14): r=1, last_{4,8,16} blocks, α={0.1,0.5,1.0} ---"
scan "${RESULTS_ROOT}/lora_constrained_sweep" "LA"
echo ""
echo "--- Ultra-Constrained (Series 16): r=1, last_{1,2} blocks, α={0.01,0.05} ---"
scan "${RESULTS_ROOT}/lora_ultra_constrained" "LB"
echo ""
echo "--- Built-in LoRA (Series 15): diffusers built-in LoRA for sanity check ---"
scan "${RESULTS_ROOT}/lora_builtin_comparison" "LN"

# ============================================================================
# EXPERIMENT 6: OUTPUT RESIDUAL (Delta-C)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 6: OUTPUT RESIDUAL (Delta-C)"
echo "  Goal: Naive output-space TTA baseline"
echo "  Config: 16 params (per-channel), AdamW"
echo "================================================================================"
echo ""
echo "--- LR Sweep (Series 11): 20 steps, LR={1e-2, 5e-2, 1e-1, 5e-1, 1.0} ---"
scan "${RESULTS_ROOT}/delta_c_lr_sweep" "DC"
echo ""
echo "--- Iter Sweep (Series 12): lr=1e-2, steps={5, 10, 20, 40} ---"
scan "${RESULTS_ROOT}/delta_c_iter_sweep" "DC"

# ============================================================================
# EARLY STOPPING ABLATION
# ============================================================================
echo ""
echo "================================================================================"
echo "  EARLY STOPPING ABLATION"
echo "  Config: full-model TTA (lr=1e-5, 20 steps) — all null results (inert method)"
echo "================================================================================"
for prefix in es_ablation_check_freq es_ablation_disable es_ablation_holdout \
              es_ablation_noise_draws es_ablation_patience es_ablation_sigmas; do
    echo ""
    echo "--- ${prefix} ---"
    scan "${RESULTS_ROOT}/${prefix}" "ES_"
done

# ============================================================================
# EXTENDED EXPERIMENTS
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXTENDED EXPERIMENTS"
echo "================================================================================"
echo ""
echo "--- Norm Tuning / TENT-style (Series 21) ---"
echo "  Tunes affine params of normalization layers"
scan "${RESULTS_ROOT}/norm_tune_sweep" "NT"
echo ""
echo "--- FiLM Adapter (Series 22) ---"
echo "  Learns corrections to adaLN modulation output (6×4096-dim)"
scan "${RESULTS_ROOT}/film_adapter_sweep" "FM"
echo ""
echo "--- Best Methods with Proper ES ---"
scan "${RESULTS_ROOT}/best_methods_proper_es"

# ============================================================================
# NEW SERIES (28-32)
# ============================================================================
echo ""
echo "================================================================================"
echo "  NEW SERIES (28-32)"
echo "================================================================================"
echo ""
echo "--- Extended Data (Series 28): 5s/8s input windows ---"
scan "${RESULTS_ROOT}/adasteer_extended_data"
echo ""
echo "--- Parameter Dimension Sweep (Series 29): delta_dim={32..512}, G={1,2,3} ---"
scan "${RESULTS_ROOT}/adasteer_param_sweep"
echo ""
echo "--- AdaSteer Long-Train + ES (Series 30): lower LR, 100-200 steps ---"
scan "${RESULTS_ROOT}/adasteer_long_train_es" "ALES"
echo ""
echo "--- Full-Model Long-Train 100v (Series 31): lr={1e-5,5e-5}, 200-500 steps ---"
scan "${RESULTS_ROOT}/full_long_train_100v" "FLT"
echo ""
echo "--- Ultra-LoRA Long-Train + ES (Series 32): last_{1,2}, 50 steps ---"
scan "${RESULTS_ROOT}/lora_ultra_long_train" "LBES"

# ============================================================================
# BATCH-SIZE ABLATION (Experiment 5)
# ============================================================================
echo ""
echo "================================================================================"
echo "  EXPERIMENT 5: VIDEO BATCH-SIZE ABLATION"
echo "  Goal: Instance-level (batch=1) vs batch-level vs dataset-level TTA"
echo "  Config: Shared delta/LoRA/full across K similar videos grouped by prompt"
echo "================================================================================"
echo ""
echo "--- Delta-A Batch-Size (Series 37): batch_videos={1,5,10,25,100} ---"
scan "${RESULTS_ROOT}/exp5_batch_size_delta_a" "BS_DA"
echo ""
echo "--- LoRA Batch-Size (Series 38): batch_videos={1,5,10,25,100} ---"
scan "${RESULTS_ROOT}/exp5_batch_size_lora" "BS_L"
echo ""
echo "--- Full-Model Batch-Size (Series 39): batch_videos={1,5,10,25,100} ---"
scan "${RESULTS_ROOT}/exp5_batch_size_full" "BS_F"

# ============================================================================
# UCF-101 CROSS-DATASET
# ============================================================================
echo ""
echo "================================================================================"
echo "  UCF-101 CROSS-DATASET"
echo "  Config: Same as best Panda-70M configs, applied to UCF-101"
echo "================================================================================"
scan "${RESULTS_ROOT}/ucf101_no_tta" "UCF_"
scan "${RESULTS_ROOT}/ucf101_full" "UCF_"
scan "${RESULTS_ROOT}/ucf101_delta_a" "UCF_"
scan "${RESULTS_ROOT}/ucf101_lora" "UCF_"

# ============================================================================
# BACKBONE EXPERIMENTS
# ============================================================================
echo ""
echo "================================================================================"
echo "  BACKBONE EXPERIMENTS"
echo "================================================================================"
echo ""
echo "--- Open-Sora v2.0 ---"
if [ -d "${PROJECT_ROOT}/backbone_experiment/opensora/results" ]; then
    scan "${PROJECT_ROOT}/backbone_experiment/opensora/results"
else
    echo "  (not started)"
fi
echo ""
echo "--- CogVideoX-5B-I2V ---"
if [ -d "${PROJECT_ROOT}/backbone_experiment/cogvideo/results" ]; then
    scan "${PROJECT_ROOT}/backbone_experiment/cogvideo/results"
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
python3 -c "
import os, json

results_root = '${RESULTS_ROOT}'
complete = 0
in_prog = 0
not_started = 0

for series in sorted(os.listdir(results_root)):
    sp = os.path.join(results_root, series)
    if not os.path.isdir(sp):
        continue
    for run in sorted(os.listdir(sp)):
        rp = os.path.join(sp, run)
        if not os.path.isdir(rp):
            continue
        if os.path.isfile(os.path.join(rp, 'summary.json')):
            complete += 1
        elif os.path.isfile(os.path.join(rp, 'checkpoint.json')):
            in_prog += 1
        else:
            not_started += 1

print(f'  Complete (summary.json):    {complete}')
print(f'  In-progress (checkpoint):   {in_prog}')
print(f'  Empty dirs (no data):       {not_started}')
print(f'  Total experiment dirs:      {complete + in_prog + not_started}')
" 2>/dev/null || echo "  (count unavailable)"
echo "================================================================================"
