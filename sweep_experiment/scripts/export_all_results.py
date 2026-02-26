#!/usr/bin/env python3
"""
Export ALL experiment results into a single JSON file for graphing.

Walks every results directory, reads summary.json and config.json,
and produces a flat list of records with standardised keys.

Usage (on cluster):
    cd /scratch/wc3013/longcat-video-tta
    python sweep_experiment/scripts/export_all_results.py > all_results.json

The output is a JSON array of objects, one per completed run.
"""

import json
import math
import os
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    "/scratch/wc3013/longcat-video-tta",
))
RESULTS_DIRS = [
    PROJECT_ROOT / "sweep_experiment" / "results",
    PROJECT_ROOT / "baseline_experiment" / "results",
    PROJECT_ROOT / "backbone_experiment" / "opensora" / "results",
    PROJECT_ROOT / "backbone_experiment" / "cogvideo" / "results",
]


def safe_mean(xs):
    return statistics.mean(xs) if xs else None

def safe_std(xs):
    return statistics.stdev(xs) if len(xs) > 1 else 0.0 if xs else None

def safe_median(xs):
    return statistics.median(xs) if xs else None


def extract_run(run_dir: Path) -> dict | None:
    """Extract a single run directory into a flat record."""
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoint.json"

    if not summary_path.exists():
        if checkpoint_path.exists():
            with open(checkpoint_path) as f:
                ck = json.load(f)
            return {
                "status": "in_progress",
                "series": run_dir.parent.name,
                "run_id": run_dir.name,
                "videos_done": ck.get("next_idx", 0),
            }
        return None

    with open(summary_path) as f:
        s = json.load(f)

    rec = {
        "status": "complete",
        "series": run_dir.parent.name,
        "run_id": run_dir.name,
        "path": str(run_dir),
    }

    # ---------- Detect format ----------
    is_baseline = "metrics" in s and "results" not in s
    results = s.get("results", [])
    ok = [r for r in results if r.get("success", False)]

    # ---------- Method ----------
    method = s.get("method", "")
    if not method and config_path.exists():
        with open(config_path) as f:
            cfg = json.load(f)
        method = cfg.get("method", "")
    rec["method"] = method

    # ---------- Video counts ----------
    if is_baseline:
        rec["n_ok"] = s.get("num_successful", 0)
        rec["n_total"] = s.get("num_videos", 0)
    else:
        rec["n_ok"] = len(ok)
        rec["n_total"] = s.get("num_videos", len(results))

    # ---------- Metrics ----------
    if is_baseline:
        m = s.get("metrics", {})
        for key in ("psnr", "ssim", "lpips"):
            if key in m and m[key]:
                rec[f"{key}_mean"] = m[key].get("mean")
                rec[f"{key}_std"] = m[key].get("std")
                rec[f"{key}_median"] = m[key].get("median")
        t = s.get("timing", {}).get("per_video_inference_s", {})
        if t:
            rec["gen_time_mean"] = t.get("mean")
            rec["gen_time_std"] = t.get("std")
        rec["train_time_mean"] = 0.0
        rec["train_time_std"] = 0.0
    else:
        psnrs = [r["psnr"] for r in ok if "psnr" in r and r["psnr"] is not None and not math.isnan(r["psnr"])]
        ssims = [r["ssim"] for r in ok if "ssim" in r and r["ssim"] is not None and not math.isnan(r["ssim"])]
        lpipss = [r["lpips"] for r in ok if "lpips" in r and r["lpips"] is not None and not math.isnan(r["lpips"])]
        train_ts = [r["train_time"] for r in ok if "train_time" in r]
        gen_ts = [r["gen_time"] for r in ok if "gen_time" in r]
        total_ts = [r.get("total_time", r.get("train_time", 0) + r.get("gen_time", 0))
                    for r in ok if "train_time" in r]

        rec["psnr_mean"] = safe_mean(psnrs)
        rec["psnr_std"] = safe_std(psnrs)
        rec["psnr_median"] = safe_median(psnrs)
        rec["ssim_mean"] = safe_mean(ssims)
        rec["ssim_std"] = safe_std(ssims)
        rec["lpips_mean"] = safe_mean(lpipss)
        rec["lpips_std"] = safe_std(lpipss)
        rec["train_time_mean"] = safe_mean(train_ts)
        rec["train_time_std"] = safe_std(train_ts)
        rec["gen_time_mean"] = safe_mean(gen_ts)
        rec["gen_time_std"] = safe_std(gen_ts)
        rec["total_time_mean"] = safe_mean(total_ts)

        # Per-video losses
        losses = [r["final_loss"] for r in ok
                  if "final_loss" in r and r["final_loss"] is not None
                  and not math.isnan(r["final_loss"])]
        rec["final_loss_mean"] = safe_mean(losses)

    # ---------- Early stopping ----------
    if not is_baseline:
        es_infos = [r.get("early_stopping_info") for r in ok
                    if r.get("early_stopping_info")]
        if es_infos:
            stopped = [e for e in es_infos if e.get("stopped_early", False)]
            best_steps = [e["best_step"] for e in es_infos if "best_step" in e]
            rec["es_stopped_count"] = len(stopped)
            rec["es_total_count"] = len(es_infos)
            rec["es_best_step_mean"] = safe_mean(best_steps)

    # ---------- Config from summary.json top-level ----------
    config_keys = [
        "delta_steps", "delta_lr", "num_groups", "delta_target",
        "delta_mode", "delta_dim",
        "learning_rate", "num_steps", "warmup_steps",
        "lora_rank", "lora_alpha",
        "total_params", "trainable_params",
        "norm_target", "norm_steps", "norm_lr",
        "film_mode", "film_steps", "film_lr",
        "num_cond_frames", "num_frames", "gen_start_frame",
        "tta_total_frames", "tta_context_frames",
        "num_inference_steps", "guidance_scale", "resolution",
        "max_videos", "seed",
        "batch_videos", "retrieval_pool_dir",
        "optimizer",
    ]
    for k in config_keys:
        if k in s and s[k] is not None:
            rec[k] = s[k]

    # ---------- Config from config.json (may have nested structure) ----------
    if config_path.exists():
        try:
            with open(config_path) as f:
                cfg = json.load(f)
            # Flatten nested config
            for k, v in cfg.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        full_key = k2
                        if full_key not in rec or rec[full_key] is None:
                            rec[full_key] = v2
                else:
                    if k not in rec or rec[k] is None:
                        rec[k] = v
        except (json.JSONDecodeError, OSError):
            pass

    # ---------- Derived fields ----------
    trainable = rec.get("trainable_params")
    if trainable is None:
        method_lower = str(rec.get("method", "")).lower()
        if "delta_a" in method_lower:
            rec["trainable_params"] = 512
        elif "delta_b" in method_lower:
            groups = rec.get("num_groups", 1)
            dim = rec.get("delta_dim") or 512
            rec["trainable_params"] = groups * dim
        elif "delta_c" in method_lower:
            rec["trainable_params"] = 4096

    # Train/gen time ratio
    if rec.get("train_time_mean") and rec.get("gen_time_mean"):
        rec["train_gen_ratio"] = rec["train_time_mean"] / rec["gen_time_mean"]

    return rec


def main():
    all_records = []
    seen_paths = set()

    for results_dir in RESULTS_DIRS:
        if not results_dir.exists():
            continue
        for series_dir in sorted(results_dir.iterdir()):
            if not series_dir.is_dir():
                continue
            # Check if this is a single-run directory (has summary.json directly)
            if (series_dir / "summary.json").exists():
                rec = extract_run(series_dir)
                if rec and str(series_dir) not in seen_paths:
                    seen_paths.add(str(series_dir))
                    all_records.append(rec)
                continue
            # Otherwise scan subdirectories
            for run_dir in sorted(series_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                if str(run_dir) in seen_paths:
                    continue
                rec = extract_run(run_dir)
                if rec:
                    seen_paths.add(str(run_dir))
                    all_records.append(rec)

    json.dump(all_records, sys.stdout, indent=2, default=str)
    print(file=sys.stderr)
    complete = sum(1 for r in all_records if r.get("status") == "complete")
    progress = sum(1 for r in all_records if r.get("status") == "in_progress")
    print(f"Exported {complete} complete + {progress} in-progress = "
          f"{len(all_records)} total records", file=sys.stderr)


if __name__ == "__main__":
    main()
