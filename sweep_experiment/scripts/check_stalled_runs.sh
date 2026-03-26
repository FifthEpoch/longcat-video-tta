#!/usr/bin/env bash
# Check status of runs that appear "in-progress" in the summary but are not in squeue.
# Run from project root on the cluster: /scratch/wc3013/longcat-video-tta
#
# Stalled runs to investigate (exclude currently running: tta_exp3_*, longcat_):
#   adasteer_ratio_sweep/AR4, AR5
#   delta_a_optimized/DAO2
#   exp3_train_frames_delta_a/TF_DA3
#   exp5_batch_size_lora_ucf101/BS_L2
#   full_iter_sweep/F10
#   full_long_train_100v/FLT3

set -euo pipefail
ROOT="${1:-.}"
cd "$ROOT"

# Result dirs: sweep_experiment/results/<series>/<run_id>
# Also check: sweep_experiment/results_no_preempt/<series>/<run_id>
# Logs: sweep_experiment/slurm_log/<JobName>_<JobID>.out and .err

echo "=== 1. Result dirs: summary.json (complete) vs checkpoint.json (incomplete) + mtime ==="
for series_run in \
  "adasteer_ratio_sweep:AR4" \
  "adasteer_ratio_sweep:AR5" \
  "delta_a_optimized:DAO2" \
  "exp3_train_frames_delta_a:TF_DA3" \
  "exp5_batch_size_lora_ucf101:BS_L2" \
  "full_iter_sweep:F10" \
  "full_long_train_100v:FLT3"; do
  series="${series_run%%:*}"
  run_id="${series_run##*:}"
  echo "--- $series / $run_id ---"
  for base in sweep_experiment/results sweep_experiment/results_no_preempt; do
    dir="$base/$series/$run_id"
    if [[ -d "$dir" ]]; then
      echo "  $dir"
      if [[ -f "$dir/summary.json" ]]; then
        echo "    summary.json: exists (run completed)"
      else
        echo "    summary.json: missing"
      fi
      if [[ -f "$dir/checkpoint.json" ]]; then
        mtime=$(stat -c '%y' "$dir/checkpoint.json" 2>/dev/null || stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$dir/checkpoint.json" 2>/dev/null || echo "?")
        echo "    checkpoint.json: exists; last modified: $mtime"
        next_idx=$(python3 -c "import json; print(json.load(open('$dir/checkpoint.json')).get('next_idx','?'))" 2>/dev/null || echo "?")
        echo "    next_idx (videos done): $next_idx"
      else
        echo "    checkpoint.json: missing"
      fi
    fi
  done
done

echo ""
echo "=== 2. sacct: recent jobs for these series (State, ExitCode, Start, End) ==="
# Job names from run_sweep.py: tta_<series_name>_<run_id>
sacct -u wc3013 -S 2026-02-01 --format=JobID,JobName%50,State,ExitCode,Start,End -n \
  | grep -E "tta_adasteer_ratio_sweep_AR[45]|tta_delta_a_optimized_DAO2|tta_exp3_train_frames_delta_a_TF_DA3|tta_exp5_batch_size_lora_ucf101_BS_L2|tta_full_iter_sweep_F10|tta_full_long_train_100v_FLT3" \
  || true

echo ""
echo "=== 3. List slurm log files matching these runs ==="
for pattern in tta_adasteer_ratio_sweep_AR4 tta_adasteer_ratio_sweep_AR5 tta_delta_a_optimized_DAO2 tta_exp3_train_frames_delta_a_TF_DA3 tta_exp5_batch_size_lora_ucf101_BS_L2 tta_full_iter_sweep_F10 tta_full_long_train_100v_FLT3; do
  ls sweep_experiment/slurm_log/${pattern}_*.out 2>/dev/null || echo "(no log: ${pattern}_*.out)"
done

echo ""
echo "=== 4. For each run: show last 30 lines of .out and last 20 of .err (set JOBID manually if needed) ==="
echo "Run these by hand after you have JobIDs from step 2 or 3:"
echo "  tail -30 sweep_experiment/slurm_log/tta_<series>_<runid>_<JOBID>.out"
echo "  tail -20 sweep_experiment/slurm_log/tta_<series>_<runid>_<JOBID>.err"
