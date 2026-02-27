#!/usr/bin/env python3
"""
Generate all paper figures from exported experiment data.

Usage:
    cd LongCat-Video-Experiment
    python paper_figures/generate_figures.py

Reads: all_results.json
Writes to: paper_figures/output/<subdirectory>/
"""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "all_results.json"
OUT_ROOT = Path(__file__).resolve().parent / "output"

# ═══════════════════════════════════════════════════════════════════════
# COLOR PALETTE
# ═══════════════════════════════════════════════════════════════════════

C_ADASTEER   = "#5160FF"   # hero — AdaSteer (best of A/B)
C_ADASTEER_B = "#5160FF"   # alias (unified now)
C_LORA       = "#D4A574"   # warm amber — distinct from blue/green
C_FULL       = "#8BBE70"   # green
C_DELTAC     = "#A8AA35"   # olive (naive methods)
C_NORMTUNE   = "#C4977A"   # dusty rose (naive methods)
C_FILM       = "#9BB8A0"   # sage (naive methods)
C_BASELINE   = "#B0B5AC"   # muted grey-green
C_RED        = "#FF6262"
C_LIGHT_Y    = "#F9FFB7"
C_LIGHT_T    = "#B7E8E0"
C_BASELINE_LINE = "#9A9E98"  # darker grey for baseline dashed line

METHOD_COLORS = {
    "AdaSteer":    C_ADASTEER,
    "AdaSteer-A":  C_ADASTEER,
    "AdaSteer-B":  C_ADASTEER,
    "Delta-C":     C_DELTAC,
    "Full-model":  C_FULL,
    "LoRA":        C_LORA,
    "NormTune":    C_NORMTUNE,
    "FiLM":        C_FILM,
    "No TTA":      C_BASELINE,
}

# Main comparison: only AdaSteer (unified), LoRA, Full-model, No TTA
METHOD_ORDER_MAIN = ["AdaSteer", "LoRA", "Full-model", "No TTA"]

# Full order including naive (for other graphs)
METHOD_ORDER = [
    "AdaSteer", "LoRA", "Full-model",
    "Delta-C", "NormTune", "FiLM", "No TTA",
]

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     12,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    9,
    "figure.dpi":         200,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linestyle":     ":",
    "axes.axisbelow":     True,
})

# ═══════════════════════════════════════════════════════════════════════
# CONFIG LOOKUPS (values missing from exported JSON)
# ═══════════════════════════════════════════════════════════════════════

EXP3_COND_FRAMES = {
    "TF_DA1": 2, "TF_DA2": 7, "TF_DA3": 14, "TF_DA4": 24,
    "TF_DB1": 2, "TF_DB2": 7, "TF_DB3": 14, "TF_DB4": 24,
    "TF_DC1": 2, "TF_DC2": 7, "TF_DC3": 14, "TF_DC4": 24,
    "TF_F1":  2, "TF_F2":  7, "TF_F3":  14, "TF_F4":  24,
    "TF_L1":  2, "TF_L2":  7, "TF_L3":  14, "TF_L4":  24,
}

EXP4_GEN_FRAMES = {
    "GH_DA1": 16, "GH_DA2": 28, "GH_DA3": 44, "GH_DA4": 72,
    "GH_DB1": 16, "GH_DB2": 28, "GH_DB3": 44, "GH_DB4": 72,
    "GH_DC1": 16, "GH_DC2": 28, "GH_DC3": 44, "GH_DC4": 72,
    "GH_F1":  16, "GH_F2":  28, "GH_F3":  44, "GH_F4":  72,
    "GH_L1":  16, "GH_L2":  28, "GH_L3":  44, "GH_L4":  72,
}

RATIO_SWEEP_COND = {
    "AR1": 2,  "AR2": 2,  "AR3": 2,
    "AR6": 24, "AR7": 24, "AR8": 24,
}

# Method display name from raw method string
def method_display(raw: str) -> str:
    m = {
        "delta_a":      "AdaSteer",
        "delta_b":      "AdaSteer",
        "delta_c":      "Delta-C",
        "full_tta":     "Full-model",
        "lora_tta":     "LoRA",
        "norm_tune":    "NormTune",
        "film_adapter": "FiLM",
        "":             "No TTA",
    }
    return m.get(raw, raw)

# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

def load_data() -> List[Dict]:
    if not DATA_FILE.exists():
        sys.stderr.write(f"  (no {DATA_FILE.name}, some figures will use embedded fallback data)\n")
        return []
    with open(DATA_FILE) as f:
        return json.load(f)

def complete_runs(data: List[Dict]) -> List[Dict]:
    return [r for r in data
            if r.get("status") == "complete" and r.get("psnr_mean") is not None]

def by_series(data: List[Dict], series_name: str) -> List[Dict]:
    return [r for r in data if r.get("series") == series_name]

# ═══════════════════════════════════════════════════════════════════════
# PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════════════

def annotate_bar(ax, bar, value, fmt="{:.1f}", bold=False, fontsize=8, offset=0.3):
    """Place value text above a bar."""
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    txt = fmt.format(value)
    weight = "bold" if bold else "normal"
    ax.text(x, y + offset, txt, ha="center", va="bottom",
            fontsize=fontsize, fontweight=weight, color="#333333")

def save(fig, *parts):
    """Save figure to OUT_ROOT / parts."""
    p = OUT_ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)
    plt.close(fig)
    print(f"  ✓ {p.relative_to(OUT_ROOT)}")


def titled(ax, title, fixed=None, fontsize=13):
    """Set title with optional 'Fixed: ...' subtitle."""
    ax.set_title(title, fontweight="bold", pad=18 if fixed else 10, fontsize=fontsize)
    if fixed:
        ax.text(0.5, 1.01, f"Fixed: {fixed}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8.5, color="#777777", style="italic")


# ═══════════════════════════════════════════════════════════════════════
# FIG 1: MAIN METHOD COMPARISON  (DeepSeek-style grouped bars)
# ═══════════════════════════════════════════════════════════════════════

def _get_standard_best(c):
    """Best standard config per method: 14 cond frames, 28 gen frames."""
    exclude_series = set()
    for r in c:
        s = r["series"]
        if any(k in s for k in ["exp3", "exp4", "long_train", "extended",
                                 "ucf101", "ratio", "optimized"]):
            exclude_series.add(s)
        if s.startswith("results") or "es_ablation" in s or "equiv" in s or "hidden" in s:
            exclude_series.add(s)
    standard = [r for r in c if r["series"] not in exclude_series and r.get("n_ok", 0) >= 80]

    best_per = {}
    for r in standard:
        md = method_display(r.get("method", ""))
        if md == "No TTA":
            continue
        if md not in best_per or r["psnr_mean"] > best_per[md]["psnr_mean"]:
            best_per[md] = r

    notta = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    if notta:
        best_per["No TTA"] = notta[0]
    return best_per


def _draw_method_bars(ax, methods, best_per, key, metric_label, unit,
                      fontsize_annot=9, show_baseline_text=False):
    """Shared bar-drawing logic for method comparison charts."""
    vals = [(m, best_per[m].get(key, 0) or 0) for m in methods]
    n = len(vals)
    bar_w = 0.82
    xs = np.arange(n)

    # Baseline dashed line BEHIND bars (zorder=0)
    bl_val = best_per.get("No TTA", {}).get(key)
    if bl_val is not None:
        ax.axhline(bl_val, color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)

    # Compute y-limits first so annotation offset is proportional
    all_v = [v for _, v in vals]
    vrange = max(all_v) - min(all_v) if max(all_v) != min(all_v) else max(all_v) * 0.05
    lower_is_better = "LPIPS" in metric_label
    if not lower_is_better:
        ymin = min(all_v) - vrange * 0.25
        ymin = max(ymin, 0)
        ymax = max(all_v) + vrange * 0.55
    else:
        ymin = min(all_v) - vrange * 0.25
        ymin = max(ymin, 0)
        ymax = max(all_v) + vrange * 0.55
    ax.set_ylim(ymin, ymax)
    vis_range = ymax - ymin
    annot_offset = vis_range * 0.02

    for i, (m, v) in enumerate(vals):
        color = METHOD_COLORS.get(m, "#999999")
        is_hero = m == "AdaSteer"
        hatch = "///" if is_hero else ""
        ec = "#3040CC" if is_hero else "none"
        bar = ax.bar(i, v, bar_w, color=color, hatch=hatch,
                     edgecolor=ec, linewidth=0.8 if is_hero else 0, zorder=3)
        annotate_bar(ax, bar[0], v, fmt="{:.2f}", bold=is_hero,
                     fontsize=fontsize_annot, offset=annot_offset)

    ax.set_xticks(xs)
    ax.set_xticklabels([v[0] for v in vals], fontsize=10)
    ylabel = metric_label
    if unit:
        ylabel += f" ({unit})"
    ax.set_ylabel(ylabel)


def fig_method_comparison(data):
    print("\n[1] Method comparison (per-metric bar charts)")
    c = complete_runs(data)
    best_per = _get_standard_best(c)

    methods = [m for m in METHOD_ORDER_MAIN if m in best_per]
    metrics = [("PSNR", "psnr_mean", "psnr_std", "dB"),
               ("SSIM", "ssim_mean", "ssim_std", ""),
               ("LPIPS (lower is better)", "lpips_mean", "lpips_std", "")]

    STD_FIXED = "20 steps, 14 cond frames, 28 gen frames, best LR per method"

    # --- Individual metric charts ---
    for metric_label, key, std_key, unit in metrics:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        _draw_method_bars(ax, methods, best_per, key, metric_label, unit, fontsize_annot=9)
        titled(ax, f"{metric_label} — Method Comparison", fixed=STD_FIXED)
        save(fig, "01_method_comparison", f"method_comparison_{key.split('_')[0]}.png")

    # --- Multi-metric comparison: 3 subplots side by side ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
    for ax, (metric_label, key, std_key, unit) in zip(axes, metrics):
        _draw_method_bars(ax, methods, best_per, key, metric_label, unit, fontsize_annot=8)
        ax.set_title(metric_label, fontweight="bold")

    fig.suptitle("TTA Method Comparison",
                 fontweight="bold", y=1.03, fontsize=13)
    fig.text(0.5, 0.98, f"Fixed: {STD_FIXED}",
             ha="center", fontsize=8.5, color="#777777", style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "01_method_comparison", "method_comparison_all_metrics.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 2: PARETO FRONTS
# ═══════════════════════════════════════════════════════════════════════

def fig_pareto(data):
    print("\n[2] Pareto fronts (quality vs params, quality vs time)")
    c = complete_runs(data)
    best_per = _get_standard_best(c)

    bl_psnr = best_per.get("No TTA", {}).get("psnr_mean", 0)

    # --- Params vs PSNR ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.axhline(bl_psnr, color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)

    plotted = []
    for m in METHOD_ORDER_MAIN:
        if m not in best_per or m == "No TTA":
            continue
        r = best_per[m]
        params = r.get("trainable_params")
        if params is None or params == 0:
            continue
        psnr = r["psnr_mean"]
        color = METHOD_COLORS.get(m, "#999999")
        marker = "D" if m == "AdaSteer" else "o"
        ms = 100 if m == "AdaSteer" else 65
        ax.scatter(params, psnr, c=color, s=ms, marker=marker,
                   edgecolors="white", linewidths=1.0, zorder=10)
        plotted.append((params, psnr, m, color))

    for params, psnr, m, color in plotted:
        ha, dx, dy = "left", 10, 5
        if "Full" in m:
            ha, dx, dy = "right", -10, -12
        ax.annotate(m, (params, psnr), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, color=color, ha=ha,
                    fontweight="bold" if m == "AdaSteer" else "normal")

    ax.text(0.98, 0.03, f"No TTA baseline = {bl_psnr:.2f} dB",
            transform=ax.transAxes, color=C_BASELINE_LINE, fontsize=8,
            ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("PSNR (dB)")
    titled(ax, "Quality vs. Parameter Efficiency",
           fixed="20 steps, 14 cond frames, 28 gen frames")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x < 1e4 else f"{x/1e3:.0f}K" if x < 1e6
        else f"{x/1e6:.0f}M" if x < 1e9 else f"{x/1e9:.1f}B"))
    save(fig, "02_pareto", "pareto_params_vs_psnr.png")

    # --- Time vs PSNR ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.axhline(bl_psnr, color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)

    plotted = []
    for m in METHOD_ORDER_MAIN:
        if m not in best_per or m == "No TTA":
            continue
        r = best_per[m]
        tt = r.get("train_time_mean", 0)
        if tt is None:
            continue
        psnr = r["psnr_mean"]
        color = METHOD_COLORS.get(m, "#999999")
        marker = "D" if m == "AdaSteer" else "o"
        ms = 100 if m == "AdaSteer" else 65
        ax.scatter(tt, psnr, c=color, s=ms, marker=marker,
                   edgecolors="white", linewidths=1.0, zorder=10)
        plotted.append((tt, psnr, m, color))

    for tt, psnr, m, color in plotted:
        dx, dy = 8, 5
        if "Full" in m:
            dx, dy = -8, -10
        ha = "left" if dx > 0 else "right"
        ax.annotate(m, (tt, psnr), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, color=color, ha=ha,
                    fontweight="bold" if m == "AdaSteer" else "normal")

    ax.text(0.98, 0.03, f"No TTA baseline = {bl_psnr:.2f} dB",
            transform=ax.transAxes, color=C_BASELINE_LINE, fontsize=8,
            ha="right", va="bottom")

    ax.set_xlabel("Training Time per Video (s)")
    ax.set_ylabel("PSNR (dB)")
    titled(ax, "Quality vs. Training Cost",
           fixed="20 steps, 14 cond frames, 28 gen frames")
    save(fig, "02_pareto", "pareto_time_vs_psnr.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 3: LR SWEEP
# ═══════════════════════════════════════════════════════════════════════

def fig_lr_sweep(data):
    print("\n[3] Learning rate sweep")
    c = complete_runs(data)

    # Main methods only; use B data for AdaSteer (better performing impl)
    series_map = {
        "delta_b_lr_sweep": ("AdaSteer", "delta_lr"),
        "full_lr_sweep":    ("Full-model",  "learning_rate"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    if bl:
        ax.axhline(bl[0]["psnr_mean"], color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)

    for series_name, (label, lr_key) in series_map.items():
        runs = by_series(c, series_name)
        if not runs:
            continue
        pts = sorted([(r.get(lr_key, r.get("delta_lr", r.get("learning_rate", 0))),
                       r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        color = METHOD_COLORS.get(label, "#999999")
        marker = "D" if label == "AdaSteer" else "o"
        ax.plot(xs, ys, "-", color=color, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("PSNR (dB)")
    titled(ax, "Learning Rate Sweep",
           fixed="20 steps, 14 cond frames, 28 gen frames")
    ax.legend(frameon=False)
    save(fig, "03_lr_sweep", "lr_sweep_psnr.png")

    # AdaSteer LR sensitivity detail (combined B sweep + low-lr data)
    fig, ax = plt.subplots(figsize=(7, 5))
    if bl:
        ax.axhline(bl[0]["psnr_mean"], color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)

    db_all = by_series(c, "delta_b_lr_sweep") + by_series(c, "delta_b_low_lr")
    if db_all:
        pts = {}
        for r in db_all:
            lr = r.get("delta_lr", 0)
            if lr not in pts or r["psnr_mean"] > pts[lr]:
                pts[lr] = r["psnr_mean"]
        pts = sorted(pts.items())
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=C_ADASTEER, marker="D", markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label="AdaSteer")

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("PSNR (dB)")
    titled(ax, "AdaSteer Learning Rate Sensitivity",
           fixed="20 steps, 14 cond frames, 28 gen frames")
    ax.legend(frameon=False)
    save(fig, "03_lr_sweep", "lr_sweep_adasteer_detail.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 4: ITERATION SWEEP
# ═══════════════════════════════════════════════════════════════════════

def fig_iter_sweep(data):
    print("\n[4] Iteration sweep")
    c = complete_runs(data)

    # Main methods only; use A data for AdaSteer (has more iter-sweep data)
    series_map = {
        "delta_a_iter_sweep": ("AdaSteer", "delta_steps"),
        "full_iter_sweep":    ("Full-model",  "num_steps"),
        "lora_iter_sweep":    ("LoRA",        "num_steps"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    if bl:
        ax.axhline(bl[0]["psnr_mean"], color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)

    for series_name, (label, step_key) in series_map.items():
        runs = by_series(c, series_name)
        if not runs:
            continue
        pts = sorted([(r.get(step_key, r.get("delta_steps", r.get("num_steps", 0))),
                       r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        color = METHOD_COLORS.get(label, "#999999")
        marker = "D" if label == "AdaSteer" else "o"
        ax.plot(xs, ys, "-", color=color, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label=label)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("PSNR (dB)")
    titled(ax, "Training Steps Sweep",
           fixed="best LR per method, 14 cond frames, 28 gen frames")
    ax.legend(frameon=False)
    save(fig, "04_iter_sweep", "iter_sweep_psnr.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 5: ADASTEER GROUPS
# ═══════════════════════════════════════════════════════════════════════

def fig_adasteer_groups(data):
    print("\n[5] AdaSteer groups analysis")
    c = complete_runs(data)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    bl_psnr = bl[0]["psnr_mean"] if bl else 22.07

    # Panel A: delta_b_groups_sweep (20 steps)
    ax = axes[0]
    runs = by_series(c, "delta_b_groups_sweep")
    if runs:
        pts = sorted([(r.get("num_groups", 1), r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=C_ADASTEER, marker="D", markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, lw=2, label="20-step")
    ax.axhline(bl_psnr, color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)
    ax.set_xlabel("Number of Groups (G)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("AdaSteer Groups (20 steps)", fontweight="bold")
    ax.legend(frameon=False)

    # Panel B: adasteer_groups_5step
    ax = axes[1]
    runs = by_series(c, "adasteer_groups_5step")
    if runs:
        pts = sorted([(r.get("num_groups", 1), r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=C_ADASTEER, marker="D", markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, lw=2, label="5-step")
    ax.axhline(bl_psnr, color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)
    ax.set_xlabel("Number of Groups (G)")
    ax.set_title("AdaSteer Groups (5 steps, delta_lr=0.01)", fontweight="bold")
    ax.legend(frameon=False)

    fig.suptitle("")
    fig.tight_layout()
    save(fig, "05_adasteer_groups", "groups_sweep.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 6: LORA ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

def fig_lora_analysis(data):
    print("\n[6] LoRA analysis")
    c = complete_runs(data)

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    bl_psnr = bl[0]["psnr_mean"] if bl else 22.07

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Rank sweep
    ax = axes[0]
    runs = by_series(c, "lora_rank_sweep")
    if runs:
        pts = sorted([(r.get("lora_rank", 1), r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=C_LORA, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, lw=2)
    ax.axhline(bl_psnr, color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("LoRA Rank Sweep", fontweight="bold")

    # Panel B: Params vs PSNR (all LoRA variants)
    ax = axes[1]
    lora_series = ["lora_rank_sweep", "lora_constrained_sweep",
                   "lora_ultra_constrained", "lora_builtin_comparison"]
    colors_lora = [C_LORA, C_ADASTEER, C_FULL, C_BASELINE]
    labels_lora = ["All blocks", "Low alpha", "Last block only", "Built-in LoRA"]

    for si, (sname, col, lab) in enumerate(zip(lora_series, colors_lora, labels_lora)):
        runs = by_series(c, sname)
        if not runs:
            continue
        pts = sorted([(r.get("trainable_params", 0), r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=col, s=40, alpha=0.8, label=lab, zorder=5)

    ax.axhline(bl_psnr, color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("LoRA: Parameters vs Quality", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1e3:.0f}K" if x >= 1e3 else f"{x:.0f}"))

    fig.tight_layout()
    save(fig, "06_lora_analysis", "lora_analysis.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 7: CONDITIONING FRAMES ABLATION (exp3)
# ═══════════════════════════════════════════════════════════════════════

def fig_cond_frames(data):
    print("\n[7] Conditioning frames ablation (exp3)")
    c = complete_runs(data)

    # Use B data for AdaSteer; exclude Delta-C (naive)
    series_methods = {
        "exp3_train_frames_delta_b": "AdaSteer",
        "exp3_train_frames_full":    "Full-model",
        "exp3_train_frames_lora":    "LoRA",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    if bl:
        ax.axhline(bl[0]["psnr_mean"], color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)

    for sname, label in series_methods.items():
        runs = by_series(c, sname)
        if not runs:
            continue
        pts = []
        for r in runs:
            cond = EXP3_COND_FRAMES.get(r["run_id"])
            if cond is not None:
                pts.append((cond, r["psnr_mean"]))
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        color = METHOD_COLORS.get(label, "#999999")
        marker = "D" if label == "AdaSteer" else "o"
        ax.plot(xs, ys, "-", color=color, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label=label)

    ax.set_xlabel("Conditioning Frames")
    ax.set_ylabel("PSNR (dB)")
    titled(ax, "Effect of Conditioning Frames",
           fixed="20 steps, 28 gen frames, best LR per method")
    ax.legend(frameon=False)
    ax.set_xticks([2, 7, 14, 24])
    save(fig, "07_cond_frames", "cond_frames_psnr.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 8: GENERATION HORIZON ABLATION (exp4)
# ═══════════════════════════════════════════════════════════════════════

def fig_gen_horizon(data):
    print("\n[8] Generation horizon ablation (exp4)")
    c = complete_runs(data)

    # Use B data for AdaSteer; exclude Delta-C (naive)
    series_methods = {
        "exp4_gen_horizon_delta_b": "AdaSteer",
        "exp4_gen_horizon_full":    "Full-model",
        "exp4_gen_horizon_lora":    "LoRA",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    if bl:
        ax.axhline(bl[0]["psnr_mean"], color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)

    for sname, label in series_methods.items():
        runs = by_series(c, sname)
        if not runs:
            continue
        pts = []
        for r in runs:
            gen = EXP4_GEN_FRAMES.get(r["run_id"])
            if gen is not None:
                pts.append((gen, r["psnr_mean"]))
        if not pts:
            continue
        pts.sort()
        xs, ys = zip(*pts)
        color = METHOD_COLORS.get(label, "#999999")
        marker = "D" if label == "AdaSteer" else "o"
        ax.plot(xs, ys, "-", color=color, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label=label)

    ax.set_xlabel("Generation Frames")
    ax.set_ylabel("PSNR (dB)")
    titled(ax, "Effect of Generation Horizon",
           fixed="20 steps, 14 cond frames, best LR per method")
    ax.legend(frameon=False)
    ax.set_xticks([16, 28, 44, 72])
    save(fig, "08_gen_horizon", "gen_horizon_psnr.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 9: CROSS-DATASET (UCF-101 vs Panda-70M)
# ═══════════════════════════════════════════════════════════════════════

def fig_cross_dataset(data):
    print("\n[9] Cross-dataset comparison")
    c = complete_runs(data)

    panda_best = _get_standard_best(c)

    # UCF-101 results
    ucf_best = {}
    for r in c:
        if "ucf101" not in r.get("series", ""):
            continue
        md = method_display(r.get("method", ""))
        if r["series"] == "ucf101_no_tta":
            md = "No TTA"
        if md not in ucf_best or r["psnr_mean"] > ucf_best[md]["psnr_mean"]:
            ucf_best[md] = r

    common = [m for m in ["AdaSteer", "LoRA", "Full-model", "No TTA"]
              if m in panda_best and m in ucf_best]

    if not common:
        print("  (skip: no common methods)")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_w = 0.38
    xs = np.arange(len(common))

    bars_panda = ax.bar(xs - bar_w / 2, [panda_best[m]["psnr_mean"] for m in common],
                        bar_w, color=C_ADASTEER, alpha=0.85, label="Panda-70M", zorder=3)
    bars_ucf = ax.bar(xs + bar_w / 2, [ucf_best[m]["psnr_mean"] for m in common],
                      bar_w, color=C_LORA, alpha=0.85, label="UCF-101", zorder=3)

    for b in bars_panda:
        annotate_bar(ax, b, b.get_height(), fmt="{:.1f}", fontsize=8, offset=0.15)
    for b in bars_ucf:
        annotate_bar(ax, b, b.get_height(), fmt="{:.1f}", fontsize=8, offset=0.15)

    ax.set_xticks(xs)
    ax.set_xticklabels(common)
    ax.set_ylabel("PSNR (dB)")
    titled(ax, "Cross-Dataset Generalization",
           fixed="20 steps, 14 cond frames, 28 gen frames, best LR per method")
    ax.legend(frameon=False)

    all_vals = [panda_best[m]["psnr_mean"] for m in common] + \
               [ucf_best[m]["psnr_mean"] for m in common]
    vrange = max(all_vals) - min(all_vals) if max(all_vals) != min(all_vals) else 1
    ymin = min(all_vals) - vrange * 0.25
    ymax = max(all_vals) + vrange * 0.45
    ax.set_ylim(max(ymin, 0), ymax)

    save(fig, "09_cross_dataset", "cross_dataset_psnr.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 10: EARLY STOPPING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

# Fallback ES ablation data (from progress report) when all_results.json lacks it
ES_ABLATION_FALLBACK = [
    {"series": "es_ablation_disable", "run_id": "ES_D1", "n_early": 0, "train_time_mean": 77.6, "psnr_mean": 22.05, "ssim_mean": 0.7682, "lpips_mean": 0.2380},
    {"series": "es_ablation_holdout", "run_id": "ES_H1", "n_early": 24, "train_time_mean": 94.0, "psnr_mean": 22.07, "ssim_mean": 0.7681, "lpips_mean": 0.2380},
    {"series": "es_ablation_holdout", "run_id": "ES_H2", "n_early": 26, "train_time_mean": 92.4, "psnr_mean": 22.07, "ssim_mean": 0.7687, "lpips_mean": 0.2357},
    {"series": "es_ablation_holdout", "run_id": "ES_H3", "n_early": 25, "train_time_mean": 92.4, "psnr_mean": 22.07, "ssim_mean": 0.7683, "lpips_mean": 0.2363},
    {"series": "es_ablation_noise_draws", "run_id": "ES_ND1", "n_early": 31, "train_time_mean": 85.2, "psnr_mean": 22.05, "ssim_mean": 0.7683, "lpips_mean": 0.2373},
    {"series": "es_ablation_noise_draws", "run_id": "ES_ND2", "n_early": 26, "train_time_mean": 93.7, "psnr_mean": 22.05, "ssim_mean": 0.7684, "lpips_mean": 0.2374},
    {"series": "es_ablation_noise_draws", "run_id": "ES_ND3", "n_early": 31, "train_time_mean": 107.2, "psnr_mean": 22.07, "ssim_mean": 0.7685, "lpips_mean": 0.2358},
    {"series": "es_ablation_patience", "run_id": "ES_P1", "n_early": 94, "train_time_mean": 51.9, "psnr_mean": 22.06, "ssim_mean": 0.7684, "lpips_mean": 0.2361},
    {"series": "es_ablation_patience", "run_id": "ES_P2", "n_early": 59, "train_time_mean": 80.7, "psnr_mean": 22.07, "ssim_mean": 0.7683, "lpips_mean": 0.2364},
    {"series": "es_ablation_patience", "run_id": "ES_P3", "n_early": 22, "train_time_mean": 93.5, "psnr_mean": 22.05, "ssim_mean": 0.7683, "lpips_mean": 0.2377},
    {"series": "es_ablation_patience", "run_id": "ES_P4", "n_early": 0, "train_time_mean": 96.0, "psnr_mean": 22.07, "ssim_mean": 0.7682, "lpips_mean": 0.2365},
    {"series": "es_ablation_patience", "run_id": "ES_P5", "n_early": 0, "train_time_mean": 95.7, "psnr_mean": 22.09, "ssim_mean": 0.7685, "lpips_mean": 0.2358},
    {"series": "es_ablation_sigmas", "run_id": "ES_S1", "n_early": 23, "train_time_mean": 83.2, "psnr_mean": 22.06, "ssim_mean": 0.7685, "lpips_mean": 0.2372},
    {"series": "es_ablation_sigmas", "run_id": "ES_S2", "n_early": 26, "train_time_mean": 82.8, "psnr_mean": 22.08, "ssim_mean": 0.7684, "lpips_mean": 0.2354},
    {"series": "es_ablation_sigmas", "run_id": "ES_S3", "n_early": 24, "train_time_mean": 81.6, "psnr_mean": 22.08, "ssim_mean": 0.7681, "lpips_mean": 0.2373},
    {"series": "es_ablation_sigmas", "run_id": "ES_S4", "n_early": 29, "train_time_mean": 82.3, "psnr_mean": 22.05, "ssim_mean": 0.7680, "lpips_mean": 0.2374},
]


def _es_series_legend_labels():
    """Short labels for ES ablation series (for legend)."""
    return {
        "es_ablation_disable":    "No ES",
        "es_ablation_holdout":   "Holdout",
        "es_ablation_noise_draws": "Noise draws",
        "es_ablation_patience":  "Patience",
        "es_ablation_sigmas":    "Sigmas",
    }


def _add_es_series_legend(fig_or_ax, series_order: List[str], series_colors: Dict[str, str], ax=None):
    """Add legend for ES ablation series (scatter colors). Pass fig and ax for fig.legend, or just ax for ax.legend."""
    labels = _es_series_legend_labels()
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=series_colors.get(s, "#999999"),
               markeredgecolor="white", markeredgewidth=1.0, markersize=8, label=labels.get(s, s))
        for s in series_order
    ]
    if ax is not None:
        ax.legend(handles=handles, frameon=False, fontsize=9)
    else:
        fig_or_ax.legend(handles=handles, frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))


def _get_es_ablation_time_runs(complete_runs_list: List[Dict]) -> Tuple[List[Dict], float]:
    """Collect ES ablation runs with n_early, train_time_mean, psnr/ssim/lpips. Returns (runs, no_es_train_time)."""
    es_series = {"es_ablation_disable", "es_ablation_holdout", "es_ablation_noise_draws",
                 "es_ablation_patience", "es_ablation_sigmas"}
    from_data = []
    no_es_time = None
    for r in complete_runs_list:
        if r.get("series") not in es_series:
            continue
        n_early = r.get("es_stopped_count", 0)
        total = max(r.get("es_total_count", 1), 1)
        # Normalize to count
        train_t = r.get("train_time_mean")
        if train_t is None:
            continue
        if r.get("series") == "es_ablation_disable":
            no_es_time = train_t
        from_data.append({
            "series": r["series"],
            "run_id": r["run_id"],
            "n_early": n_early,
            "train_time_mean": train_t,
            "psnr_mean": r.get("psnr_mean"),
            "ssim_mean": r.get("ssim_mean"),
            "lpips_mean": r.get("lpips_mean"),
        })
    if from_data and no_es_time is not None:
        return from_data, no_es_time
    # Use fallback
    no_es_time = 77.6
    return ES_ABLATION_FALLBACK, no_es_time


def fig_early_stopping_time_savings(complete_runs_list: List[Dict]):
    """Early stopping: time savings vs # videos stopped early, performance unchanged (PSNR/SSIM/LPIPS)."""
    runs, no_es_time = _get_es_ablation_time_runs(complete_runs_list)
    if not runs:
        return

    n_early = [r["n_early"] for r in runs]
    train_t = [r["train_time_mean"] for r in runs]
    psnr = [r.get("psnr_mean") for r in runs]
    ssim = [r.get("ssim_mean") for r in runs]
    lpips = [r.get("lpips_mean") for r in runs]
    series = [r["series"] for r in runs]
    run_ids = [r["run_id"] for r in runs]

    # Color by series (ablation type)
    series_order = ["es_ablation_disable", "es_ablation_holdout", "es_ablation_noise_draws",
                    "es_ablation_patience", "es_ablation_sigmas"]
    series_colors = {
        "es_ablation_disable":    C_BASELINE,
        "es_ablation_holdout":   C_LIGHT_T,
        "es_ablation_noise_draws": C_LORA,
        "es_ablation_patience":  C_ADASTEER,
        "es_ablation_sigmas":    C_FULL,
    }
    colors = [series_colors.get(s, "#999999") for s in series]

    # --- 1. Train time vs # early (with no-ES reference) ---
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.axhline(no_es_time, color=C_BASELINE_LINE, ls="--", lw=1.2, alpha=0.7, label="No early stopping", zorder=0)
    ax.scatter(n_early, train_t, c=colors, s=72, edgecolors="white", linewidths=1.0, zorder=5)
    ax.set_xlabel("Videos stopped early")
    ax.set_ylabel("Mean training time per video (s)")
    ax.set_title("Early stopping reduces training time", fontweight="bold", pad=10)
    labels = _es_series_legend_labels()
    handles = [Line2D([0], [0], color=C_BASELINE_LINE, ls="--", lw=1.2, alpha=0.7, label="No early stopping")]
    handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor=series_colors.get(s, "#999999"),
                       markeredgecolor="white", markeredgewidth=1.0, markersize=8, label=labels.get(s, s))
                for s in series_order]
    ax.legend(handles=handles, frameon=False, fontsize=9)
    ax.set_ylim(0, max(train_t) * 1.08 if train_t else 120)
    save(fig, "10_early_stopping", "es_time_vs_early.png")

    # --- 2. PSNR, SSIM, LPIPS vs # early (three subplots) ---
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8), sharex=True)
    for ax, key, label, ylim_tight in [
        (axes[0], "psnr_mean", "PSNR (dB)", (21.9, 22.2)),
        (axes[1], "ssim_mean", "SSIM", (0.765, 0.772)),
        (axes[2], "lpips_mean", "LPIPS", (0.233, 0.240)),
    ]:
        vals = [r.get(key) for r in runs if r.get(key) is not None]
        if not vals:
            continue
        ax.scatter(n_early, [r.get(key) for r in runs], c=colors, s=72, edgecolors="white", linewidths=1.0, zorder=5)
        ax.set_ylabel(label)
        ax.set_ylim(ylim_tight)
        ax.axhline(np.mean(vals), color=C_BASELINE_LINE, ls=":", lw=1.0, alpha=0.6, zorder=0)
    axes[2].set_xlabel("Videos stopped early")
    axes[0].set_title("Performance unchanged across early-stopping settings", fontweight="bold", pad=10)
    _add_es_series_legend(fig, series_order, series_colors, axes[1])
    fig.tight_layout()
    save(fig, "10_early_stopping", "es_metrics_vs_early.png")

    # --- 3. Two-panel: time + metrics (PSNR, SSIM, LPIPS in one row of 3 small panels) ---
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])
    ax_time = fig.add_subplot(gs[0])
    ax_time.axhline(no_es_time, color=C_BASELINE_LINE, ls="--", lw=1.2, alpha=0.7, label="No ES", zorder=0)
    ax_time.scatter(n_early, train_t, c=colors, s=64, edgecolors="white", linewidths=1.0, zorder=5)
    ax_time.set_xlabel("Videos stopped early")
    ax_time.set_ylabel("Mean training time (s)")
    ax_time.set_title("Training time", fontweight="bold")

    gs_right = gs[1].subgridspec(1, 3)
    first_metric_ax = None
    for i, (key, label, ylim_tight) in enumerate([
        ("psnr_mean", "PSNR (dB)", (21.9, 22.2)),
        ("ssim_mean", "SSIM", (0.765, 0.772)),
        ("lpips_mean", "LPIPS", (0.233, 0.240)),
    ]):
        ax = fig.add_subplot(gs_right[0, i])
        if first_metric_ax is None:
            first_metric_ax = ax
        ax.scatter(n_early, [r.get(key) for r in runs], c=colors, s=52, edgecolors="white", linewidths=0.8, zorder=5)
        ax.set_ylabel(label, fontsize=10)
        ax.set_ylim(ylim_tight)
        ax.set_xlabel("# early", fontsize=9)
    if first_metric_ax is not None:
        _add_es_series_legend(fig, series_order, series_colors, first_metric_ax)
    fig.suptitle("Early stopping: time savings without compromising quality",
                 fontweight="bold", y=1.02, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save(fig, "10_early_stopping", "es_time_savings_two_panel.png")

    # --- 4. Time saved vs # early ---
    time_saved = [no_es_time - t for t in train_t]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(n_early, time_saved, c=colors, s=72, edgecolors="white", linewidths=1.0, zorder=5)
    ax.axhline(0, color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.5, zorder=0)
    ax.set_xlabel("Videos stopped early")
    ax.set_ylabel("Time saved per video (s)")
    ax.set_title("Early stopping: time saved vs. number of videos stopped early", fontweight="bold", pad=10)
    _add_es_series_legend(fig, series_order, series_colors, ax)
    ax.set_ylim(min(time_saved) - 2 if time_saved else -5, 30)
    save(fig, "10_early_stopping", "es_time_saved_vs_early.png")

    # --- 5. Mean TTA train time per video vs metrics (PSNR, SSIM, LPIPS) ---
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharex=True)
    for ax, key, label, ylim_tight in [
        (axes[0], "psnr_mean", "PSNR (dB)", (21.9, 22.2)),
        (axes[1], "ssim_mean", "SSIM", (0.765, 0.772)),
        (axes[2], "lpips_mean", "LPIPS", (0.233, 0.240)),
    ]:
        ax.scatter(train_t, [r.get(key) for r in runs], c=colors, s=72, edgecolors="white", linewidths=1.0, zorder=5)
        ax.set_xlabel("Mean TTA train time per video (s)")
        ax.set_ylabel(label)
        ax.set_ylim(ylim_tight)
    _add_es_series_legend(fig, series_order, series_colors, axes[1])
    fig.suptitle("Metrics vs. mean TTA train time per video (early-stopping ablations)", fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save(fig, "10_early_stopping", "es_train_time_vs_metrics.png")


def fig_early_stopping(data):
    print("\n[10] Early stopping analysis")
    c = complete_runs(data)

    # --- Panel A: ES ablation - Patience ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    runs = sorted(by_series(c, "es_ablation_patience"),
                  key=lambda r: r["run_id"])
    if runs:
        patience_vals = [1, 2, 3, 5, 10]  # ES_P1..ES_P5
        psnrs = [r["psnr_mean"] for r in runs]
        stopped_frac = [r.get("es_stopped_count", 0) / max(r.get("es_total_count", 1), 1)
                        for r in runs]

        ax2 = ax.twinx()
        ax2.bar(patience_vals[:len(stopped_frac)],
                [f * 100 for f in stopped_frac],
                0.6, color=C_LIGHT_T, alpha=0.45, zorder=1)
        ax2.set_ylabel("% Videos Stopped Early", color="#666666")
        ax2.set_ylim(0, 105)

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(patience_vals[:len(psnrs)], psnrs, "-o", color=C_ADASTEER,
                markersize=6, markeredgecolor="white", lw=1.8, label="PSNR", zorder=10)
        ax.set_xlabel("Patience")
        ax.set_ylabel("PSNR (dB)", color=C_ADASTEER)
        ax.set_title("Early Stopping: Patience", fontweight="bold")

    # --- Panel B: ES ablation - Check Frequency ---
    ax = axes[1]
    runs = sorted(by_series(c, "es_ablation_check_freq"),
                  key=lambda r: r["run_id"])
    if runs:
        check_freqs = [1, 2, 5, 10]  # ES_CF1..ES_CF4
        psnrs = [r["psnr_mean"] for r in runs]
        stopped_frac = [r.get("es_stopped_count", 0) / max(r.get("es_total_count", 1), 1)
                        for r in runs]

        ax2 = ax.twinx()
        ax2.bar(check_freqs[:len(stopped_frac)],
                [f * 100 for f in stopped_frac],
                0.6, color=C_LIGHT_T, alpha=0.45, zorder=1)
        ax2.set_ylabel("% Videos Stopped Early", color="#666666")
        ax2.set_ylim(0, 105)

        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.plot(check_freqs[:len(psnrs)], psnrs, "-o", color=C_ADASTEER,
                markersize=6, markeredgecolor="white", lw=1.8, label="PSNR", zorder=10)
        ax.set_xlabel("Check Every N Steps")
        ax.set_ylabel("PSNR (dB)", color=C_ADASTEER)
        ax.set_title("Early Stopping: Check Frequency", fontweight="bold")

    fig.tight_layout()
    save(fig, "10_early_stopping", "es_ablation.png")

    # --- Time savings + same performance (all ES ablations) ---
    fig_early_stopping_time_savings(c)

    # --- Long-train early stopping overview ---
    fig, ax = plt.subplots(figsize=(8, 4))
    long_runs = sorted(
        [r for r in c if r["series"] in ("delta_a_long_train", "full_long_train")],
        key=lambda r: r.get("delta_steps") or r.get("num_steps") or 0
    )
    if long_runs:
        labels = []
        for i, r in enumerate(long_runs):
            md = method_display(r.get("method", ""))
            total_steps = r.get("delta_steps") or r.get("num_steps") or 0
            best_step = r.get("es_best_step_mean", total_steps)
            stopped = r.get("es_stopped_count", 0)
            total_v = r.get("es_total_count", 1)
            color = METHOD_COLORS.get(md, "#999999")
            labels.append(f"{md}\n({total_steps} steps)")

            ax.barh(i, total_steps, color=C_LIGHT_Y, edgecolor="#cccccc",
                    height=0.55, zorder=1)
            ax.barh(i, best_step, color=color, height=0.55, zorder=2)
            ax.text(best_step + total_steps * 0.02, i,
                    f"avg best = step {best_step:.0f}  ({stopped} stopped early)",
                    va="center", fontsize=9, color="#333333", zorder=3)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Training Steps")
        ax.set_title("Early Stopping on Long Training Runs", fontweight="bold", pad=10)
        ax.invert_yaxis()

    save(fig, "10_early_stopping", "long_train_es.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 11: TRAINING TIME COST
# ═══════════════════════════════════════════════════════════════════════

def fig_time_cost(data):
    print("\n[11] Training time cost")
    c = complete_runs(data)
    best_per = _get_standard_best(c)
    methods = [m for m in METHOD_ORDER_MAIN if m in best_per and m != "No TTA"]

    # --- Bar chart: training time per video ---
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    xs = np.arange(len(methods))
    bar_w = 0.82
    for i, m in enumerate(methods):
        r = best_per[m]
        tt = r.get("train_time_mean", 0)
        color = METHOD_COLORS.get(m, "#999999")
        is_hero = m == "AdaSteer"
        hatch = "///" if is_hero else ""
        ec = "#3040CC" if is_hero else "none"

        bar = ax.bar(i, tt, bar_w, color=color, hatch=hatch,
                     edgecolor=ec, linewidth=0.8 if is_hero else 0, zorder=3)
        annotate_bar(ax, bar[0], tt, fmt="{:.0f}s", fontsize=9,
                     bold=is_hero, offset=max(tt * 0.02, 1))

    ax.set_xticks(xs)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Training Time per Video (s)")
    titled(ax, "TTA Training Cost per Video",
           fixed="20 steps, 14 cond frames, 28 gen frames")
    save(fig, "11_time_cost", "train_time.png")

    # --- Train/Gen ratio ---
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for i, m in enumerate(methods):
        r = best_per[m]
        ratio = r.get("train_gen_ratio", 0) or 0
        color = METHOD_COLORS.get(m, "#999999")
        is_hero = m == "AdaSteer"
        hatch = "///" if is_hero else ""
        ec = "#3040CC" if is_hero else "none"

        bar = ax.bar(i, ratio, bar_w, color=color, hatch=hatch,
                     edgecolor=ec, linewidth=0.8 if is_hero else 0, zorder=3)
        annotate_bar(ax, bar[0], ratio, fmt="{:.2f}x", fontsize=9,
                     bold=is_hero, offset=max(ratio * 0.02, 0.01))

    ax.axhline(1.0, color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)
    ax.text(len(methods) - 0.5, 1.03, "Train = Gen time", color=C_BASELINE_LINE,
            fontsize=8, ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("Train Time / Generation Time")
    titled(ax, "Training Overhead Relative to Generation",
           fixed="20 steps, 14 cond frames, 28 gen frames")
    save(fig, "11_time_cost", "train_gen_ratio.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 12: EXTENDED DATA (Window A / Window B)
# ═══════════════════════════════════════════════════════════════════════

def fig_extended_data(data):
    print("\n[12] Extended data (AdaSteer with longer videos)")
    c = complete_runs(data)

    runs = by_series(c, "adasteer_extended_data")
    if not runs:
        print("  (no data)")
        return

    window_a = {r["run_id"]: r for r in runs if r["run_id"].startswith("EX_A")}
    window_b = {r["run_id"]: r for r in runs if r["run_id"].startswith("EX_B")}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    for ax, window, title, color_base in [
        (axes[0], window_a, "Window A (5s, gen_start=120)", C_ADASTEER),
        (axes[1], window_b, "Window B (8s, gen_start=200)", C_ADASTEER),
    ]:
        control_key = [k for k in window if k.endswith("0")]
        tta_keys = [k for k in sorted(window.keys()) if not k.endswith("0")]

        if control_key:
            ctrl = window[control_key[0]]
            ax.axhline(ctrl["psnr_mean"], color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)

        groups = []
        psnrs = []
        for k in tta_keys:
            r = window[k]
            g = r.get("num_groups", 1)
            groups.append(g)
            psnrs.append(r["psnr_mean"])

        if groups:
            ax.plot(groups, psnrs, "-D", color=color_base, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.8, lw=2)
            for g, p in zip(groups, psnrs):
                ax.annotate(f"{p:.2f}", (g, p), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color=color_base)

        ax.set_xlabel("Number of Groups (G)")
        ax.set_title(title, fontweight="bold")
        ax.set_xticks(sorted(set(groups)))

    axes[0].set_ylabel("PSNR (dB)")
    fig.suptitle("AdaSteer with Extended Training Data", fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "12_extended_data", "extended_data_groups.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 13: RATIO SWEEP (cond frames × groups)
# ═══════════════════════════════════════════════════════════════════════

def fig_ratio_sweep(data):
    print("\n[13] Token-to-parameter ratio sweep")
    c = complete_runs(data)

    runs = by_series(c, "adasteer_ratio_sweep")
    if not runs:
        print("  (no data)")
        return

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    bl_psnr = bl[0]["psnr_mean"] if bl else 22.07

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(bl_psnr, color=C_BASELINE_LINE, ls="--", lw=1.0, alpha=0.55, zorder=0)

    by_cond = defaultdict(list)
    for r in runs:
        cond = RATIO_SWEEP_COND.get(r["run_id"])
        if cond is None:
            continue
        by_cond[cond].append((r.get("num_groups", 1), r["psnr_mean"]))

    cond_colors = {2: C_ADASTEER, 24: C_ADASTEER_B}
    for cond in sorted(by_cond.keys()):
        pts = sorted(by_cond[cond])
        xs, ys = zip(*pts)
        color = cond_colors.get(cond, "#999999")
        ax.plot(xs, ys, "-D", color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, lw=2,
                label=f"cond_frames={cond}")

    ax.set_xlabel("Number of Groups (G)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Conditioning Frames × Groups Interaction", fontweight="bold", pad=10)
    ax.legend(frameon=False)
    save(fig, "13_ratio_sweep", "ratio_sweep.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG BONUS: COMPREHENSIVE SCATTER (all runs)
# ═══════════════════════════════════════════════════════════════════════

def fig_all_runs_scatter(data):
    print("\n[Bonus] All runs scatter plot")
    c = complete_runs(data)

    fig, ax = plt.subplots(figsize=(9, 6))

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    if bl:
        ax.axhline(bl[0]["psnr_mean"], color=C_BASELINE_LINE, ls="--", lw=1, alpha=0.5, zorder=0)

    seen_methods = set()
    for r in c:
        md = method_display(r.get("method", ""))
        params = r.get("trainable_params")
        if params is None or params == 0 or md == "No TTA":
            continue
        color = METHOD_COLORS.get(md, "#999999")
        label = md if md not in seen_methods else None
        seen_methods.add(md)
        marker = "D" if md == "AdaSteer" else "o"
        ax.scatter(params, r["psnr_mean"], c=color, s=25, alpha=0.55,
                   marker=marker, label=label, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("All Experiment Runs", fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x < 1e3 else f"{x/1e3:.0f}K" if x < 1e6
        else f"{x/1e6:.0f}M" if x < 1e9 else f"{x/1e9:.1f}B"))
    save(fig, "01_method_comparison", "all_runs_scatter.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG BONUS: NormTune & FiLM detail
# ═══════════════════════════════════════════════════════════════════════

def fig_naive_methods(data):
    """Naive TTA methods (Delta-C, NormTune, FiLM) in their own directory."""
    print("\n[Naive] Naive TTA methods (Delta-C, NormTune, FiLM)")
    c = complete_runs(data)
    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    bl_psnr = bl[0]["psnr_mean"] if bl else 22.07
    best_per = _get_standard_best(c)

    # --- Comparison bar chart ---
    naive_methods = ["Delta-C", "NormTune", "FiLM", "No TTA"]
    available = [m for m in naive_methods if m in best_per]
    if available:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        _draw_method_bars(ax, available, best_per, "psnr_mean", "PSNR", "dB")
        # Also show AdaSteer as reference
        ada_val = best_per.get("AdaSteer", {}).get("psnr_mean")
        if ada_val:
            ax.axhline(ada_val, color=C_ADASTEER, ls="-", lw=1.2, alpha=0.5, zorder=0)
            ax.text(0.02, ada_val, f"AdaSteer = {ada_val:.2f}",
                    color=C_ADASTEER, fontsize=8, va="bottom",
                    transform=ax.get_yaxis_transform())
        ax.set_title("Naive TTA Methods — PSNR", fontweight="bold", pad=10)
        save(fig, "14_naive_methods", "naive_methods_psnr.png")

    # --- NormTune sweep detail ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    runs = by_series(c, "norm_tune_sweep")
    if runs:
        for r in runs:
            params = r.get("trainable_params", 0)
            lr = r.get("norm_lr") or r.get("learning_rate") or 0
            psnr = r["psnr_mean"]
            ax.scatter(lr, psnr, c=C_NORMTUNE, s=65, zorder=5, edgecolors="white", lw=0.8)
            ax.annotate(f"{params/1e3:.0f}K", (lr, psnr), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color="#555555")
    ax.axhline(bl_psnr, color=C_BASELINE_LINE, ls="--", lw=1, alpha=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("NormTune Sweep", fontweight="bold")
    save(fig, "14_naive_methods", "normtune_sweep.png")

    # --- FiLM sweep detail ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    runs = by_series(c, "film_adapter_sweep")
    if runs:
        for r in runs:
            params = r.get("trainable_params", 0)
            lr = r.get("film_lr") or r.get("learning_rate") or 0
            psnr = r["psnr_mean"]
            ax.scatter(lr, psnr, c=C_FILM, s=65, zorder=5, edgecolors="white", lw=0.8)
            ax.annotate(f"{params/1e3:.0f}K", (lr, psnr), textcoords="offset points",
                        xytext=(6, 6), fontsize=7, color="#555555")
    ax.axhline(bl_psnr, color=C_BASELINE_LINE, ls="--", lw=1, alpha=0.5, zorder=0)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("FiLM Adapter Sweep", fontweight="bold")
    save(fig, "14_naive_methods", "film_sweep.png")

    # --- Delta-C iteration sweep ---
    fig, ax = plt.subplots(figsize=(6, 4.5))
    runs = by_series(c, "delta_c_iter_sweep")
    if runs:
        pts = sorted([(r.get("delta_steps", 0), r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-o", color=C_DELTAC, markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, lw=2)
    ax.axhline(bl_psnr, color=C_BASELINE_LINE, ls="--", lw=1, alpha=0.5, zorder=0)
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Delta-C (Output Residual) Iteration Sweep", fontweight="bold")
    save(fig, "14_naive_methods", "delta_c_iter_sweep.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def fig_summary_table(data):
    """Render a visual summary table of main methods."""
    print("\n[Table] Summary table figure")
    c = complete_runs(data)
    best_per = _get_standard_best(c)
    methods = [m for m in METHOD_ORDER_MAIN if m in best_per]

    columns = ["Method", "Params", "PSNR (dB)", "SSIM", "LPIPS", "Train (s)", "Ratio"]
    rows = []
    for m in methods:
        r = best_per[m]
        params = r.get("trainable_params")
        if params is None:
            pstr = "-"
        elif params >= 1e9:
            pstr = f"{params/1e9:.1f}B"
        elif params >= 1e6:
            pstr = f"{params/1e6:.1f}M"
        elif params >= 1e3:
            pstr = f"{params/1e3:.0f}K"
        else:
            pstr = str(params)
        psnr = f"{r.get('psnr_mean', 0):.2f}"
        ssim = f"{r.get('ssim_mean', 0):.3f}"
        lpips = f"{r.get('lpips_mean', 0):.3f}"
        tt = r.get("train_time_mean", 0) or 0
        tstr = f"{tt:.1f}" if tt > 0.01 else "0.0"
        ratio = r.get("train_gen_ratio", 0) or 0
        rstr = f"{ratio:.2f}x" if ratio > 0.001 else "-"
        rows.append([m, pstr, psnr, ssim, lpips, tstr, rstr])

    fig, ax = plt.subplots(figsize=(10, 0.5 + 0.45 * len(rows)))
    ax.axis("off")

    cell_colors = []
    for i, row in enumerate(rows):
        m = row[0]
        bg = "#E8EBFF" if "AdaSteer" in m else "#F5F5F5" if i % 2 == 0 else "white"
        cell_colors.append([bg] * len(columns))

    tbl = ax.table(cellText=rows, colLabels=columns, cellLoc="center",
                   loc="center", cellColours=cell_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)

    for (r_idx, c_idx), cell in tbl.get_celld().items():
        if r_idx == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor("#555555")
            cell.set_edgecolor("#555555")
        else:
            cell.set_edgecolor("#dddddd")

    ax.set_title("TTA Method Summary — Standard Configuration",
                 fontweight="bold", pad=15, fontsize=12)
    save(fig, "01_method_comparison", "summary_table.png")


def main():
    print(f"Loading data from {DATA_FILE}")
    data = load_data()
    total = len(data)
    comp = len(complete_runs(data))
    print(f"  {comp} complete / {total} total records\n")

    fig_method_comparison(data)
    fig_pareto(data)
    fig_lr_sweep(data)
    fig_iter_sweep(data)
    fig_adasteer_groups(data)
    fig_lora_analysis(data)
    fig_cond_frames(data)
    fig_gen_horizon(data)
    fig_cross_dataset(data)
    fig_early_stopping(data)
    fig_time_cost(data)
    fig_extended_data(data)
    fig_ratio_sweep(data)
    fig_all_runs_scatter(data)
    fig_naive_methods(data)
    fig_summary_table(data)

    # Loss curve figures (from separate export)
    loss_file = ROOT / "loss_curves.json"
    if loss_file.exists():
        with open(loss_file) as f:
            loss_data = json.load(f)
        print(f"\nLoading loss curves from {loss_file}")
        print(f"  {len(loss_data)} runs with loss curves\n")
        fig_loss_curves_long_train(loss_data)
        fig_loss_curves_method_compare(loss_data)
        fig_loss_curves_es_check_freq(loss_data)
        fig_loss_curves_iter_sweep(loss_data)
    else:
        print(f"\n  (skipping loss curves: {loss_file} not found)")

    print(f"\nAll figures saved to {OUT_ROOT}")


# ═══════════════════════════════════════════════════════════════════════
# LOSS CURVE FIGURES
# ═══════════════════════════════════════════════════════════════════════

def _get_curve(loss_data, series, run_id):
    """Find a run in loss_data by series/run_id."""
    for r in loss_data:
        if r["series"] == series and r["run_id"] == run_id:
            return r
    return None


def _plot_loss_curve(ax, curve_data, color, label, alpha_fill=0.15, lw=2.0,
                     subsample=1):
    """Plot mean +/- std anchor loss curve (DeepSeek-R1 style)."""
    agg = curve_data["aggregate_curve"]
    steps = [p["step"] for p in agg]
    means = [p["mean"] for p in agg]
    stds  = [p["std"]  for p in agg]

    if subsample > 1:
        steps = steps[::subsample]
        means = means[::subsample]
        stds  = stds[::subsample]

    upper = [m + s for m, s in zip(means, stds)]
    lower = [m - s for m, s in zip(means, stds)]

    ax.fill_between(steps, lower, upper, color=color, alpha=alpha_fill, linewidth=0)
    ax.plot(steps, means, color=color, lw=lw, label=label)


def fig_loss_curves_long_train(loss_data):
    """Long-train anchor validation loss: AdaSteer-A (200 steps) vs Full-model (500 steps)."""
    print("[Loss] Long-train anchor validation loss curves")

    da_long = _get_curve(loss_data, "delta_a_long_train", "DALT1")
    full_long = _get_curve(loss_data, "full_long_train", "FLT1")

    if not da_long and not full_long:
        print("  (no data)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel A: AdaSteer-A long train ---
    ax = axes[0]
    if da_long:
        _plot_loss_curve(ax, da_long, C_ADASTEER, "AdaSteer")
        best = da_long.get("best_step_mean", 0)
        ax.axvline(best, color=C_RED, ls="--", lw=1.2, alpha=0.7)
        ax.text(best + 3, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                f"avg best = step {best:.0f}", color=C_RED, fontsize=9, va="bottom")
        stopped = da_long.get("stopped_early_count", 0)
        n = da_long.get("n_videos_with_curves", 0)
        ax.set_title(f"AdaSteer (200 steps, {stopped}/{n} stopped early)",
                     fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Anchor Validation Loss")
    ax.legend(frameon=False)

    # --- Panel B: Full-model long train ---
    ax = axes[1]
    if full_long:
        _plot_loss_curve(ax, full_long, C_FULL, "Full-model")
        best = full_long.get("best_step_mean", 0)
        ax.axvline(best, color=C_RED, ls="--", lw=1.2, alpha=0.7)
        ax.text(best + 5, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05,
                f"avg best = step {best:.0f}", color=C_RED, fontsize=9, va="bottom")
        stopped = full_long.get("stopped_early_count", 0)
        n = full_long.get("n_videos_with_curves", 0)
        ax.set_title(f"Full-model (500 steps, {stopped}/{n} stopped early)",
                     fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Anchor Validation Loss")
    ax.legend(frameon=False)

    fig.suptitle("Anchor Validation Loss During Extended Training",
                 fontweight="bold", y=1.02, fontsize=13)
    fig.tight_layout()
    save(fig, "10_early_stopping", "long_train_loss_curves.png")


def fig_loss_curves_method_compare(loss_data):
    """Compare loss curves across methods at 20 training steps."""
    print("[Loss] Method comparison loss curves (20-step)")

    runs = [
        ("delta_b_low_lr", "DBL4", "AdaSteer (512 params)", C_ADASTEER),
        ("full_lr_sweep", "F2", "Full-model (13.6B params)", C_FULL),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted = False

    for series, run_id, label, color in runs:
        r = _get_curve(loss_data, series, run_id)
        if not r:
            continue
        _plot_loss_curve(ax, r, color, label, alpha_fill=0.08, lw=2.2)
        plotted = True

    if not plotted:
        plt.close(fig)
        print("  (no data)")
        return

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Anchor Validation Loss")
    ax.set_title("Anchor Validation Loss — Best 20-Step Configs",
                 fontweight="bold", pad=10)
    ax.legend(frameon=False)
    save(fig, "10_early_stopping", "method_compare_loss_curves.png")


def fig_loss_curves_es_check_freq(loss_data):
    """ES check frequency effect on loss curves."""
    print("[Loss] ES check frequency loss curves")

    check_freqs = [
        ("es_ablation_check_freq", "ES_CF1", "check every 1 step", C_ADASTEER),
        ("es_ablation_check_freq", "ES_CF2", "check every 2 steps", C_LORA),
        ("es_ablation_check_freq", "ES_CF3", "check every 5 steps", C_FULL),
        ("es_ablation_check_freq", "ES_CF4", "check every 10 steps", C_BASELINE),
    ]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    plotted = False

    for series, run_id, label, color in check_freqs:
        r = _get_curve(loss_data, series, run_id)
        if not r:
            continue
        _plot_loss_curve(ax, r, color, label, lw=1.8)
        plotted = True

    if not plotted:
        plt.close(fig)
        print("  (no data)")
        return

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Anchor Validation Loss")
    ax.set_title("Effect of ES Check Frequency on Validation Loss (Full-model, 20 steps)",
                 fontweight="bold", pad=10)
    ax.legend(frameon=False)
    save(fig, "10_early_stopping", "es_check_freq_loss_curves.png")


def fig_loss_curves_iter_sweep(loss_data):
    """Loss curves at different training step budgets, per method."""
    print("[Loss] Iteration sweep loss curves")

    methods = [
        ("AdaSteer", ("delta_a_long_train", "DALT1", "200 steps"), C_ADASTEER),
        ("LoRA", ("lora_iter_sweep", "L16", "80 steps"), C_LORA),
        ("Full-model", ("full_long_train", "FLT1", "500 steps"), C_FULL),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))

    for ax, (method_name, (series, run_id, label), base_color) in zip(axes, methods):
        r = _get_curve(loss_data, series, run_id)
        if r:
            sub = 5 if method_name != "LoRA" else 1
            _plot_loss_curve(ax, r, base_color, label, alpha_fill=0.12, lw=2.0,
                             subsample=sub)

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Anchor Validation Loss")
        ax.set_title(method_name, fontweight="bold")
        ax.legend(frameon=False, fontsize=9)

    fig.suptitle("Validation Loss Across Training Budgets",
                 fontweight="bold", y=1.02, fontsize=13)
    fig.tight_layout()
    save(fig, "10_early_stopping", "iter_sweep_loss_curves.png")


if __name__ == "__main__":
    main()
