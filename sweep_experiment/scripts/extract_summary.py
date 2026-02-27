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

    # Config from top-level (incl. cond/gen for display and baseline lookup)
    cfg_keys = [
        "method", "delta_steps", "delta_lr", "num_groups", "delta_target",
        "delta_mode", "learning_rate", "num_steps", "lora_rank", "lora_alpha",
        "total_params", "trainable_params", "norm_target", "norm_steps",
        "norm_lr", "film_mode", "film_steps", "film_lr",
        "num_cond_frames", "num_frames", "gen_start_frame",
    ]
    cfg = {}
    for k in cfg_keys:
        if k in d and d[k] is not None:
            cfg[k] = d[k]
    out["config"] = cfg

    # Cond/gen from top-level or first result (for older summaries)
    n_cond = d.get("num_cond_frames")
    n_frames = d.get("num_frames")
    if (n_cond is None or n_frames is None) and results:
        r0 = results[0]
        n_cond = n_cond if n_cond is not None else r0.get("num_cond_frames")
        n_frames = n_frames if n_frames is not None else r0.get("num_frames")
    out["num_cond_frames"] = n_cond
    out["num_frames"] = n_frames
    out["num_gen_frames"] = (int(n_frames) - int(n_cond)) if (n_frames is not None and n_cond is not None) else None

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
    out["num_cond_frames"] = d.get("num_cond_frames")
    out["num_gen_frames"] = d.get("num_gen_frames")

    return out


def extract_checkpoint(path):
    with open(path) as f:
        d = json.load(f)
    idx = d.get("next_idx", 0)
    results = d.get("results", [])
    ok = [r for r in results if r.get("success", False)]
    return {"n_done": len(ok), "next_idx": idx}


def load_baseline_metrics(baseline_dir, num_cond, num_gen):
    """Load No-TTA baseline metrics for the same cond/gen. Returns dict with psnr_mean, etc. or None."""
    if baseline_dir is None or num_cond is None or num_gen is None:
        return None
    for subdir in (f"cond{num_cond}_gen{num_gen}", f"ucf101_cond{num_cond}_gen{num_gen}"):
        path = os.path.join(baseline_dir, subdir, "summary.json")
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    d = json.load(f)
                m = d.get("metrics", {})
                out = {}
                if "psnr" in m and m["psnr"]:
                    out["psnr_mean"] = m["psnr"].get("mean")
                if "ssim" in m and m["ssim"]:
                    out["ssim_mean"] = m["ssim"].get("mean")
                if "lpips" in m and m["lpips"]:
                    out["lpips_mean"] = m["lpips"].get("mean")
                if out:
                    return out
            except Exception:
                pass
    return None


def format_line(run_id, data, baseline_dir=None):
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

    # Frames: cond / gen (for every run)
    n_cond = data.get("num_cond_frames")
    n_gen = data.get("num_gen_frames")
    if n_cond is not None and n_gen is not None:
        lines.append(f"         frames: {n_cond} cond, {n_gen} gen")
    elif n_cond is not None and data.get("num_frames") is not None:
        lines.append(f"         frames: {n_cond} cond, {int(data['num_frames']) - int(n_cond)} gen")

    # Line 2: Config
    cfg = data.get("config", {})
    if cfg:
        cfg_parts = []
        for k, v in cfg.items():
            if k in ("method", "num_cond_frames", "num_frames", "gen_start_frame"):
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

    # vs No-TTA baseline (all three metrics)
    if data.get("format") == "tta" and baseline_dir and "psnr_mean" in data:
        bl = load_baseline_metrics(
            baseline_dir,
            data.get("num_cond_frames"),
            data.get("num_gen_frames"),
        )
        if bl:
            parts = []
            if bl.get("psnr_mean") is not None:
                d_psnr = data["psnr_mean"] - bl["psnr_mean"]
                parts.append(f"PSNR {fmt(bl['psnr_mean'])}→Δ{'+' if d_psnr >= 0 else ''}{fmt(d_psnr)}")
            if bl.get("ssim_mean") is not None:
                d_ssim = data["ssim_mean"] - bl["ssim_mean"]
                parts.append(f"SSIM {fmt(bl['ssim_mean'], 4)}→Δ{'+' if d_ssim >= 0 else ''}{fmt(d_ssim, 4)}")
            if bl.get("lpips_mean") is not None:
                d_lpips = data["lpips_mean"] - bl["lpips_mean"]
                parts.append(f"LPIPS {fmt(bl['lpips_mean'], 4)}→Δ{'+' if d_lpips >= 0 else ''}{fmt(d_lpips, 4)}")
            if parts:
                lines.append(f"         vs No-TTA: {', '.join(parts)}")

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


# Abbreviations for config keys to save horizontal space
_CONFIG_ABBREV = {
    "learning_rate": "lr",
    "num_steps": "steps",
    "warmup_steps": "warmup",
    "weight_decay": "wd",
    "max_grad_norm": "mgn",
    "delta_steps": "d_steps",
    "delta_lr": "d_lr",
    "num_groups": "grp",
    "delta_target": "d_tgt",
    "delta_mode": "d_mode",
    "lora_rank": "r",
    "lora_alpha": "a",
    "total_params": "params",
    "trainable_params": "trainable",
    "norm_target": "n_tgt",
    "norm_steps": "n_steps",
    "norm_lr": "n_lr",
    "film_mode": "f_mode",
    "film_steps": "f_steps",
    "film_lr": "f_lr",
}


def _config_short(cfg):
    """Short config string (abbreviated keys, exclude cond/gen, truncate)."""
    skip = ("method", "num_cond_frames", "num_frames", "gen_start_frame")
    parts = []
    for k, v in (cfg or {}).items():
        if k in skip:
            continue
        key = _CONFIG_ABBREV.get(k, k)
        if isinstance(v, float):
            parts.append(f"{key}={v:g}")
        else:
            parts.append(f"{key}={v}")
    s = ", ".join(parts)
    return s[:44] + "…" if len(s) > 44 else s


def _infer_cond_gen_exp3(run_id, series_dir):
    """Infer (num_cond, num_gen) for Exp 3 Training Frames Ablation from run_id when summary lacks them.
    Exp 3 run_ids: TF_F1..TF_F4, TF_DA1..TF_DA4, etc. Suffix 1→2 cond, 2→7, 3→14, 4→24; gen=14."""
    if not series_dir or "exp3_train_frames" not in os.path.normpath(series_dir):
        return None, None
    s = run_id.strip()
    if not s:
        return None, None
    last = s[-1]
    if last not in "1234":
        return None, None
    cond_map = {"1": 2, "2": 7, "3": 14, "4": 24}
    return cond_map[last], 14


def _infer_cond_gen_exp4(run_id, series_dir):
    """Exp 4 Generation Horizon: 14 cond fixed, num_frames swept 16/28/44/72 → gen = total - 14."""
    if not series_dir or "exp4_gen_horizon" not in os.path.normpath(series_dir):
        return None, None
    # GH_F1→16, GH_F2→28, GH_F3→44, GH_F4→72
    total_map = {"1": 16, "2": 28, "3": 44, "4": 72}
    s = run_id.strip()
    if len(s) < 2:
        return None, None
    suffix = s[-1] if s[-1] in "1234" else None
    if suffix is None:
        return None, None
    total = total_map.get(suffix)
    if total is None:
        return None, None
    return 14, total - 14


def _infer_cond_gen_delta_a_optimized(run_id, series_dir):
    """Delta-A optimized: DAO1 = 24 cond, 28 total → 4 gen; DAO2 = 14 cond, 14 gen."""
    if not series_dir or "delta_a_optimized" not in os.path.normpath(series_dir):
        return None, None
    if run_id == "DAO1":
        return 24, 4
    if run_id == "DAO2":
        return 14, 14
    return None, None


def _infer_cond_gen_ratio_sweep(run_id, series_dir):
    """AdaSteer ratio sweep: num_frames=28, cond 2/14/24 → gen = 28 - cond."""
    if not series_dir or "adasteer_ratio_sweep" not in os.path.normpath(series_dir):
        return None, None
    # AR1,AR2,AR3 → 2 cond, 26 gen; AR4,AR5 → 14,14; AR6,AR7,AR8 → 24,4
    if run_id in ("AR1", "AR2", "AR3"):
        return 2, 26
    if run_id in ("AR4", "AR5"):
        return 14, 14
    if run_id in ("AR6", "AR7", "AR8"):
        return 24, 4
    return None, None


# Series that use standard 14 cond, 14 gen (num_frames=28) when not in summary
_SERIES_14_14 = frozenset([
    "full_lr_sweep", "full_iter_sweep", "full_long_train", "full_long_train_100v",
    "delta_a_lr_sweep", "delta_a_iter_sweep", "delta_a_long_train", "delta_a_norm_combined",
    "delta_a_equiv_verify", "delta_b_groups_sweep", "delta_b_lr_sweep", "delta_b_iter_sweep",
    "delta_b_low_lr", "delta_b_hidden_sweep", "adasteer_groups_5step",
    "lora_rank_sweep", "lora_iter_sweep", "lora_constrained_sweep", "lora_ultra_constrained",
    "lora_builtin_comparison", "delta_c_lr_sweep", "delta_c_iter_sweep",
    "es_ablation_disable", "es_ablation_holdout", "es_ablation_noise_draws",
    "es_ablation_patience", "es_ablation_sigmas", "es_ablation_check_freq",
    "norm_tune_sweep", "film_adapter_sweep", "best_methods_proper_es",
    "ucf101_full", "ucf101_delta_a", "ucf101_lora",
    "panda_no_tta_continuation", "ucf101_no_tta",
])


def _infer_cond_gen_from_series(series_dir, run_id):
    """Infer (num_cond, num_gen) from series path and run_id when summary lacks them."""
    if not series_dir:
        return None, None
    path_norm = os.path.normpath(series_dir)
    base_name = os.path.basename(path_norm)

    # Exp 3 (training frames ablation)
    c, g = _infer_cond_gen_exp3(run_id, path_norm)
    if c is not None:
        return c, g

    # Exp 4 (generation horizon)
    c, g = _infer_cond_gen_exp4(run_id, path_norm)
    if c is not None:
        return c, g

    # Delta-A optimized
    c, g = _infer_cond_gen_delta_a_optimized(run_id, path_norm)
    if c is not None:
        return c, g

    # AdaSteer ratio sweep
    c, g = _infer_cond_gen_ratio_sweep(run_id, path_norm)
    if c is not None:
        return c, g

    # No-TTA and standard TTA series: 14 cond, 14 gen
    if base_name in _SERIES_14_14:
        return 14, 14

    return None, None


def _bl_cell(baseline_val, run_val, decimals=2):
    """Format baseline column: baseline value and delta, e.g. '21.5 (+0.9)'."""
    if baseline_val is None or run_val is None:
        return "—"
    delta = run_val - baseline_val
    sign = "+" if delta >= 0 else ""
    if decimals <= 2:
        return f"{fmt(baseline_val, decimals)} ({sign}{fmt(delta, decimals)})"
    return f"{fmt(baseline_val, 4)} ({sign}{fmt(delta, 4)})"


def data_to_row(run_id, data, baseline_dir=None, series_dir=None):
    """Turn one run's data into a dict of table column values (strings)."""
    row = {}
    all_cols = ("ok", "PSNR", "PSNR_bl", "SSIM", "SSIM_bl", "LPIPS", "LPIPS_bl",
                "cond", "gen", "config", "train_s", "gen_s", "ES")
    if "error" in data:
        row["run_id"] = run_id
        row["ok"] = f"ERROR: {data['error'][:24]}"
        for k in all_cols:
            row[k] = "—"
        return row
    if data.get("in_progress"):
        row["run_id"] = run_id
        row["ok"] = f"in-prog ({data.get('n_done', 0)} vids)"
        for k in all_cols:
            row[k] = "—"
        return row
    if data.get("empty") or data.get("not_found"):
        row["run_id"] = run_id
        row["ok"] = "empty" if data.get("empty") else "NOT STARTED"
        for k in all_cols:
            row[k] = "—"
        return row

    row["run_id"] = run_id
    row["ok"] = f"{data['n_ok']}/{data['n_total']}"
    row["PSNR"] = f"{fmt(data.get('psnr_mean'))}±{fmt(data.get('psnr_std', 0))}" if data.get("psnr_mean") is not None else "—"
    row["SSIM"] = f"{fmt(data.get('ssim_mean'), 4)}±{fmt(data.get('ssim_std', 0), 4)}" if data.get("ssim_mean") is not None else "—"
    row["LPIPS"] = f"{fmt(data.get('lpips_mean'), 4)}±{fmt(data.get('lpips_std', 0), 4)}" if data.get("lpips_mean") is not None else "—"
    n_cond = data.get("num_cond_frames")
    n_gen = data.get("num_gen_frames")
    # Infer cond/gen when summary was written before we added these fields
    if (n_cond is None or n_gen is None) and series_dir:
        i_cond, i_gen = _infer_cond_gen_from_series(series_dir, run_id)
        if n_cond is None and i_cond is not None:
            n_cond = i_cond
        if n_gen is None and i_gen is not None:
            n_gen = i_gen
    row["cond"] = str(n_cond) if n_cond is not None else "—"
    row["gen"] = str(n_gen) if n_gen is not None else "—"
    row["config"] = _config_short(data.get("config"))
    row["train_s"] = f"{fmt(data.get('train_mean'), 1)}±{fmt(data.get('train_std', 0), 1)}" if data.get("train_mean") is not None else "—"
    row["gen_s"] = f"{fmt(data.get('gen_mean'), 1)}±{fmt(data.get('gen_std', 0), 1)}" if data.get("gen_mean") is not None else "—"
    if "es_total" in data:
        es = f"{data['es_stopped']}/{data['es_total']} early"
        if "es_best_step_mean" in data:
            es += f", μ={fmt(data['es_best_step_mean'], 1)}"
        row["ES"] = es
    else:
        row["ES"] = "—"

    bl = None
    is_no_tta = series_dir and ("no_tta" in os.path.normpath(series_dir))
    if (
        data.get("format") == "tta"
        and not is_no_tta
        and baseline_dir
        and n_cond is not None
        and n_gen is not None
    ):
        bl = load_baseline_metrics(baseline_dir, n_cond, n_gen)
    if bl:
        row["PSNR_bl"] = _bl_cell(bl.get("psnr_mean"), data.get("psnr_mean"), 2)
        row["SSIM_bl"] = _bl_cell(bl.get("ssim_mean"), data.get("ssim_mean"), 4)
        row["LPIPS_bl"] = _bl_cell(bl.get("lpips_mean"), data.get("lpips_mean"), 4)
    else:
        row["PSNR_bl"] = row["SSIM_bl"] = row["LPIPS_bl"] = "—"

    return row


# Column order and display widths (no status; PSNR | PSNR bl | SSIM | SSIM bl | LPIPS | LPIPS bl)
TABLE_COLUMNS = [
    ("run_id", 10),
    ("ok", 12),
    ("PSNR", 14),
    ("PSNR_bl", 14),
    ("SSIM", 14),
    ("SSIM_bl", 14),
    ("LPIPS", 14),
    ("LPIPS_bl", 14),
    ("cond", 4),
    ("gen", 4),
    ("config", 48),
    ("train_s", 12),
    ("gen_s", 10),
    ("ES", 18),
]


def print_table(rows):
    """Print a single table for a series (one row per run)."""
    if not rows:
        return
    col_names = [c[0] for c in TABLE_COLUMNS]
    widths = {c[0]: c[1] for c in TABLE_COLUMNS}
    # Header
    header = "  " + " | ".join(cn.ljust(widths[cn]) for cn in col_names)
    sep = "  " + "-+-".join("-" * widths[cn] for cn in col_names)
    print(header)
    print(sep)
    for r in rows:
        cells = []
        for cn in col_names:
            val = r.get(cn, "—")
            s = str(val)[:widths[cn]]
            cells.append(s.ljust(widths[cn]))
        print("  " + " | ".join(cells))
    print()


def print_run(run_id, dir_path, baseline_dir=None):
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
        print(format_line(run_id, data, baseline_dir=baseline_dir))


def scan_series(base_dir, pattern="", baseline_dir=None, table=True):
    """Scan a series directory and print all runs (as one table if table=True)."""
    if not os.path.isdir(base_dir):
        print(f"  (directory not found: {base_dir})")
        return

    entries = sorted(os.listdir(base_dir))
    if not entries:
        print(f"  (empty directory)")
        return

    if table:
        rows = []
        for name in entries:
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full):
                continue
            if pattern and not name.startswith(pattern.rstrip("*")):
                continue
            data = process_dir(full)
            row = data_to_row(name, data, baseline_dir=baseline_dir, series_dir=base_dir)
            rows.append(row)
        if rows:
            print_table(rows)
    else:
        for name in entries:
            full = os.path.join(base_dir, name)
            if os.path.isdir(full):
                if pattern and not name.startswith(pattern.rstrip("*")):
                    continue
                print_run(name, full, baseline_dir=baseline_dir)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Extract and display metrics from TTA/baseline summary.json")
    ap.add_argument("dir", help="Series directory or single run directory")
    ap.add_argument("pattern", nargs="?", default="", help="Optional run name prefix filter (e.g. DA)")
    ap.add_argument("--baseline-dir", type=str, default=None,
                    help="Baseline results root (e.g. baseline_experiment/results) to compare vs No-TTA")
    ap.add_argument("--no-table", action="store_true",
                    help="Use multi-line format per run instead of one table per series")
    args = ap.parse_args()

    base = args.dir
    pat = args.pattern
    baseline_dir = args.baseline_dir
    table = not args.no_table

    if os.path.isfile(os.path.join(base, "summary.json")) or \
       os.path.isfile(os.path.join(base, "checkpoint.json")):
        print_run(os.path.basename(base), base, baseline_dir=baseline_dir)
    else:
        scan_series(base, pat, baseline_dir=baseline_dir, table=table)
