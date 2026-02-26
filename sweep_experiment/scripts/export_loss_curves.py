#!/usr/bin/env python3
"""
Extract per-step anchor validation loss curves from summary.json files.

For runs with early stopping enabled, each video's result contains
early_stopping_info.loss_history: a list of [step, loss] pairs.

This script aggregates those across videos for selected runs and outputs
a compact JSON suitable for plotting loss-over-steps curves.

Usage (on cluster):
    cd /scratch/wc3013/longcat-video-tta
    python sweep_experiment/scripts/export_loss_curves.py > loss_curves.json

Then scp loss_curves.json to your local machine.
"""

import json
import math
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/scratch/wc3013/longcat-video-tta",
))
RESULTS_DIR = PROJECT_ROOT / "sweep_experiment" / "results"

RUNS_TO_EXTRACT = [
    # Long training runs (most interesting for loss curves)
    "delta_a_long_train/DALT1",
    "full_long_train/FLT1",

    # Iter sweep runs (compare 5/10/40/80 step curves)
    "delta_a_iter_sweep/DA5",    # 5 steps
    "delta_a_iter_sweep/DA6",    # 10 steps
    "delta_a_iter_sweep/DA7",    # 40 steps
    "delta_a_iter_sweep/DA8",    # 80 steps
    "delta_a_iter_sweep/DA100",  # 100 steps

    "delta_b_iter_sweep/DB7",    # 5 steps
    "delta_b_iter_sweep/DB8",    # 10 steps
    "delta_b_iter_sweep/DB9",    # 40 steps
    "delta_b_iter_sweep/DB10",   # 80 steps

    "full_iter_sweep/F6",        # 5 steps
    "full_iter_sweep/F7",        # 10 steps
    "full_iter_sweep/F8",        # 40 steps
    "full_iter_sweep/F9",        # 80 steps

    "lora_iter_sweep/L13",       # 5 steps
    "lora_iter_sweep/L14",       # 10 steps
    "lora_iter_sweep/L15",       # 40 steps
    "lora_iter_sweep/L16",       # 80 steps

    # ES ablation runs (all have loss curves by design)
    "es_ablation_patience/ES_P1",
    "es_ablation_patience/ES_P2",
    "es_ablation_patience/ES_P3",
    "es_ablation_patience/ES_P4",
    "es_ablation_patience/ES_P5",

    "es_ablation_check_freq/ES_CF1",
    "es_ablation_check_freq/ES_CF2",
    "es_ablation_check_freq/ES_CF3",
    "es_ablation_check_freq/ES_CF4",

    # Standard best runs (20 steps)
    "delta_a_lr_sweep/DA2",      # best delta-A
    "delta_b_low_lr/DBL4",       # best delta-B
    "full_lr_sweep/F2",          # best full
]


def extract_loss_curves(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None

    with open(summary_path) as f:
        s = json.load(f)

    results = s.get("results", [])
    ok = [r for r in results if r.get("success", False)]

    per_video_curves = []
    for r in ok:
        es_info = r.get("early_stopping_info")
        if not es_info:
            continue
        lh = es_info.get("loss_history")
        if not lh:
            continue
        per_video_curves.append({
            "video": r.get("video_name", ""),
            "loss_history": lh,
            "stopped_early": es_info.get("stopped_early", False),
            "best_step": es_info.get("best_step", 0),
            "best_loss": es_info.get("best_loss"),
        })

    if not per_video_curves:
        return None

    # Aggregate: mean/std loss at each step across videos
    step_losses = defaultdict(list)
    for vc in per_video_curves:
        for step, loss in vc["loss_history"]:
            if loss is not None and not math.isnan(loss) and not math.isinf(loss):
                step_losses[step].append(loss)

    agg_curve = []
    for step in sorted(step_losses.keys()):
        vals = step_losses[step]
        agg_curve.append({
            "step": step,
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "count": len(vals),
        })

    # Also extract per-video training losses if available
    train_losses = []
    for r in ok:
        fl = r.get("final_loss")
        if fl is not None and not math.isnan(fl):
            train_losses.append(fl)

    return {
        "series": run_dir.parent.name,
        "run_id": run_dir.name,
        "method": s.get("method", ""),
        "n_videos_with_curves": len(per_video_curves),
        "n_ok": len(ok),
        "total_steps": s.get("delta_steps") or s.get("num_steps") or 0,
        "psnr_mean": statistics.mean(
            [r["psnr"] for r in ok if r.get("psnr") and not math.isnan(r["psnr"])]
        ) if ok else None,
        "aggregate_curve": agg_curve,
        "stopped_early_count": sum(1 for vc in per_video_curves if vc["stopped_early"]),
        "best_step_mean": statistics.mean(
            [vc["best_step"] for vc in per_video_curves]
        ) if per_video_curves else None,
        "final_train_loss_mean": statistics.mean(train_losses) if train_losses else None,
    }


def main():
    all_curves = []

    for run_path in RUNS_TO_EXTRACT:
        run_dir = RESULTS_DIR / run_path
        if not run_dir.exists():
            print(f"  SKIP (not found): {run_path}", file=sys.stderr)
            continue

        rec = extract_loss_curves(run_dir)
        if rec:
            all_curves.append(rec)
            n = rec["n_videos_with_curves"]
            steps = len(rec["aggregate_curve"])
            print(f"  OK: {run_path} ({n} videos, {steps} step points)", file=sys.stderr)
        else:
            print(f"  SKIP (no loss curves): {run_path}", file=sys.stderr)

    json.dump(all_curves, sys.stdout, indent=2, default=str)
    print(file=sys.stderr)
    print(f"Exported {len(all_curves)} runs with loss curves", file=sys.stderr)


if __name__ == "__main__":
    main()
