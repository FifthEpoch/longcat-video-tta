#!/usr/bin/env python3
"""Extract and display comprehensive metrics from a TTA experiment summary.json."""
import json
import math
import os
import statistics
import sys


def fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "—"
    return f"{val:.{decimals}f}"


def extract_tta(path):
    """Extract from TTA sweep summary.json (results array format)."""
    with open(path) as f:
        d = json.load(f)

    results = d.get("results", [])
    ok = [r for r in results if r.get("success", False)]
    n_total = d.get("num_videos", len(results))
    n_ok = len(ok)

    psnrs = [r["psnr"] for r in ok if "psnr" in r and not math.isnan(r["psnr"])]
    ssims = [r["ssim"] for r in ok if "ssim" in r and not math.isnan(r["ssim"])]
    lpipss = [r["lpips"] for r in ok if "lpips" in r and not math.isnan(r["lpips"])]
    train_ts = [r["train_time"] for r in ok if "train_time" in r]
    gen_ts = [r["gen_time"] for r in ok if "gen_time" in r]

    out = {"n_ok": n_ok, "n_total": n_total, "format": "tta"}

    if psnrs:
        out["psnr_mean"] = statistics.mean(psnrs)
        out["psnr_std"] = statistics.stdev(psnrs) if len(psnrs) > 1 else 0
    if ssims:
        out["ssim_mean"] = statistics.mean(ssims)
        out["ssim_std"] = statistics.stdev(ssims) if len(ssims) > 1 else 0
    if lpipss:
        out["lpips_mean"] = statistics.mean(lpipss)
        out["lpips_std"] = statistics.stdev(lpipss) if len(lpipss) > 1 else 0
    if train_ts:
        out["train_mean"] = statistics.mean(train_ts)
        out["train_std"] = statistics.stdev(train_ts) if len(train_ts) > 1 else 0
    if gen_ts:
        out["gen_mean"] = statistics.mean(gen_ts)
        out["gen_std"] = statistics.stdev(gen_ts) if len(gen_ts) > 1 else 0

    # Config from top-level
    cfg_keys = [
        "method", "delta_steps", "delta_lr", "num_groups", "delta_target",
        "delta_mode", "learning_rate", "num_steps", "lora_rank", "lora_alpha",
        "total_params", "trainable_params", "norm_target", "norm_steps",
        "norm_lr", "film_mode", "film_steps", "film_lr",
    ]
    cfg = {}
    for k in cfg_keys:
        if k in d and d[k] is not None:
            cfg[k] = d[k]
    out["config"] = cfg

    # Early stopping
    es_infos = [r.get("early_stopping_info") for r in ok if r.get("early_stopping_info")]
    if es_infos:
        stopped = [e for e in es_infos if e.get("stopped_early", False)]
        best_steps = [e["best_step"] for e in es_infos if "best_step" in e]
        out["es_total"] = len(es_infos)
        out["es_stopped"] = len(stopped)
        if best_steps:
            out["es_best_step_mean"] = statistics.mean(best_steps)
            out["es_best_step_min"] = min(best_steps)
            out["es_best_step_max"] = max(best_steps)

    return out


def extract_baseline(path):
    """Extract from baseline summary.json (metrics dict format)."""
    with open(path) as f:
        d = json.load(f)

    out = {
        "n_ok": d.get("num_successful", "?"),
        "n_total": d.get("num_videos", "?"),
        "format": "baseline",
    }

    m = d.get("metrics", {})
    if "psnr" in m and m["psnr"]:
        out["psnr_mean"] = m["psnr"].get("mean", 0)
        out["psnr_std"] = m["psnr"].get("std", 0)
    if "ssim" in m and m["ssim"]:
        out["ssim_mean"] = m["ssim"].get("mean", 0)
        out["ssim_std"] = m["ssim"].get("std", 0)
    if "lpips" in m and m["lpips"]:
        out["lpips_mean"] = m["lpips"].get("mean", 0)
        out["lpips_std"] = m["lpips"].get("std", 0)

    t = d.get("timing", {})
    pv = t.get("per_video_inference_s", {})
    if pv and pv.get("mean") is not None:
        out["gen_mean"] = pv["mean"]
        out["gen_std"] = pv.get("std", 0)

    cfg = {}
    for k in ["num_cond_frames", "num_gen_frames", "num_inference_steps",
              "guidance_scale", "resolution"]:
        if k in d:
            cfg[k] = d[k]
    out["config"] = cfg

    return out


def extract_checkpoint(path):
    with open(path) as f:
        d = json.load(f)
    idx = d.get("next_idx", 0)
    results = d.get("results", [])
    ok = [r for r in results if r.get("success", False)]
    return {"n_done": len(ok), "next_idx": idx}


def format_line(run_id, data):
    """Format a single run's metrics into a compact multi-line display."""
    lines = []

    # Line 1: Status + PSNR/SSIM/LPIPS
    status = f"{data['n_ok']}/{data['n_total']} ok"
    metrics = []
    if "psnr_mean" in data:
        metrics.append(f"PSNR={fmt(data['psnr_mean'])}±{fmt(data.get('psnr_std', 0))}")
    if "ssim_mean" in data:
        metrics.append(f"SSIM={fmt(data['ssim_mean'], 4)}±{fmt(data.get('ssim_std', 0), 4)}")
    if "lpips_mean" in data:
        metrics.append(f"LPIPS={fmt(data['lpips_mean'], 4)}±{fmt(data.get('lpips_std', 0), 4)}")

    metric_str = ", ".join(metrics) if metrics else "no metric data"
    lines.append(f"  {run_id}: {status} | {metric_str}")

    # Line 2: Config
    cfg = data.get("config", {})
    if cfg:
        cfg_parts = []
        for k, v in cfg.items():
            if k == "method":
                continue
            label = k.replace("_", " ")
            if isinstance(v, float):
                cfg_parts.append(f"{label}={v:g}")
            else:
                cfg_parts.append(f"{label}={v}")
        method = cfg.get("method", "")
        if method:
            cfg_parts.insert(0, f"method={method}")
        if cfg_parts:
            lines.append(f"         config: {', '.join(cfg_parts)}")

    # Line 3: Timing
    timing = []
    if "train_mean" in data:
        timing.append(f"train={fmt(data['train_mean'], 1)}±{fmt(data.get('train_std', 0), 1)}s")
    if "gen_mean" in data:
        timing.append(f"gen={fmt(data['gen_mean'], 1)}±{fmt(data.get('gen_std', 0), 1)}s")
    if timing:
        lines.append(f"         timing: {', '.join(timing)}")

    # Line 4: Early stopping
    if "es_total" in data:
        es_str = f"ES: {data['es_stopped']}/{data['es_total']} stopped early"
        if "es_best_step_mean" in data:
            es_str += (f", best_step mean={fmt(data['es_best_step_mean'], 1)}"
                       f" min={data['es_best_step_min']}"
                       f" max={data['es_best_step_max']}")
        lines.append(f"         {es_str}")

    return "\n".join(lines)


def process_dir(dir_path):
    """Process a single experiment directory."""
    summary = os.path.join(dir_path, "summary.json")
    checkpoint = os.path.join(dir_path, "checkpoint.json")

    if os.path.isfile(summary):
        try:
            with open(summary) as f:
                d = json.load(f)
            if "results" in d and isinstance(d["results"], list):
                return extract_tta(summary)
            elif "metrics" in d:
                return extract_baseline(summary)
            else:
                return {"n_ok": "?", "n_total": "?", "error": "unknown format"}
        except Exception as e:
            return {"error": str(e)}
    elif os.path.isfile(checkpoint):
        ck = extract_checkpoint(checkpoint)
        return {"in_progress": True, **ck}
    elif os.path.isdir(dir_path):
        return {"empty": True}
    else:
        return {"not_found": True}


def print_run(run_id, dir_path):
    data = process_dir(dir_path)
    if "error" in data:
        print(f"  {run_id}: ERROR - {data['error']}")
    elif data.get("in_progress"):
        print(f"  {run_id}: in-progress ({data['n_done']} videos, checkpoint next_idx={data['next_idx']})")
    elif data.get("empty"):
        print(f"  {run_id}: dir exists, no data")
    elif data.get("not_found"):
        print(f"  {run_id}: NOT STARTED")
    else:
        print(format_line(run_id, data))


def scan_series(base_dir, pattern=""):
    """Scan a series directory and print all runs."""
    if not os.path.isdir(base_dir):
        print(f"  (directory not found: {base_dir})")
        return

    entries = sorted(os.listdir(base_dir))
    if not entries:
        print(f"  (empty directory)")
        return

    for name in entries:
        full = os.path.join(base_dir, name)
        if os.path.isdir(full):
            if pattern and not name.startswith(pattern.rstrip("*")):
                continue
            print_run(name, full)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_summary.py <dir> [pattern]")
        sys.exit(1)

    base = sys.argv[1]
    pat = sys.argv[2] if len(sys.argv) > 2 else ""

    if os.path.isfile(os.path.join(base, "summary.json")) or \
       os.path.isfile(os.path.join(base, "checkpoint.json")):
        print_run(os.path.basename(base), base)
    else:
        scan_series(base, pat)
