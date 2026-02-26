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

C_ADASTEER   = "#5160FF"   # hero — AdaSteer (Delta-A / Delta-B)
C_ADASTEER_B = "#7B86FF"   # lighter variant for Delta-B when shown separately
C_LORA       = "#88AADE"
C_FULL       = "#8BBE70"
C_DELTAC     = "#A8AA35"
C_NORMTUNE   = "#B0B5AC"
C_FILM       = "#C5C0A0"
C_BASELINE   = "#B0B5AC"
C_RED        = "#FF6262"
C_LIGHT_Y    = "#F9FFB7"
C_LIGHT_T    = "#B7E8E0"

METHOD_COLORS = {
    "AdaSteer-A":  C_ADASTEER,
    "AdaSteer-B":  C_ADASTEER_B,
    "AdaSteer":    C_ADASTEER,
    "Delta-C":     C_DELTAC,
    "Full-model":  C_FULL,
    "LoRA":        C_LORA,
    "NormTune":    C_NORMTUNE,
    "FiLM":        C_FILM,
    "No TTA":      C_BASELINE,
}

METHOD_ORDER = [
    "AdaSteer-A", "AdaSteer-B", "Full-model",
    "LoRA", "Delta-C", "NormTune", "FiLM", "No TTA",
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
        "delta_a":      "AdaSteer-A",
        "delta_b":      "AdaSteer-B",
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


def fig_method_comparison(data):
    print("\n[1] Method comparison (per-metric bar charts)")
    c = complete_runs(data)
    best_per = _get_standard_best(c)

    methods = [m for m in METHOD_ORDER if m in best_per]
    metrics = [("PSNR", "psnr_mean", "psnr_std", "dB"),
               ("SSIM", "ssim_mean", "ssim_std", ""),
               ("LPIPS (lower is better)", "lpips_mean", "lpips_std", "")]

    for metric_label, key, std_key, unit in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))

        vals = [(m, best_per[m].get(key, 0) or 0, best_per[m].get(std_key, 0) or 0)
                for m in methods]

        xs = np.arange(len(vals))
        for i, (m, v, s) in enumerate(vals):
            color = METHOD_COLORS.get(m, "#999999")
            is_hero = "AdaSteer" in m
            hatch = "///" if is_hero else ""
            ec = "#3040CC" if is_hero else "none"
            bar = ax.bar(i, v, 0.65, color=color, hatch=hatch,
                         edgecolor=ec, linewidth=0.7 if is_hero else 0)
            fmtstr = "{:.2f}"
            offset = max(v * 0.008, 0.1) if "psnr" in key else max(v * 0.01, 0.002)
            annotate_bar(ax, bar[0], v, fmt=fmtstr, bold=is_hero, fontsize=8, offset=offset)

        bl_val = best_per.get("No TTA", {}).get(key)
        if bl_val is not None:
            ax.axhline(bl_val, color=C_RED, ls="--", lw=1.2, alpha=0.7, zorder=5)
            side = "bottom" if "LPIPS" not in metric_label else "top"
            ax.text(len(vals) - 0.5, bl_val, f"No TTA = {bl_val:.2f}",
                    color=C_RED, fontsize=8, va=side, ha="right")

        ax.set_xticks(xs)
        ax.set_xticklabels([v[0] for v in vals], rotation=25, ha="right")
        ylabel = f"{metric_label}"
        if unit:
            ylabel += f" ({unit})"
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric_label} — Best Standard Configuration per Method",
                     fontweight="bold", pad=10)

        lower_is_better = "LPIPS" in metric_label
        if not lower_is_better:
            ymin = min(v for _, v, _ in vals) * 0.9
            ax.set_ylim(ymin, ax.get_ylim()[1] * 1.08)
        else:
            ax.set_ylim(0, max(v for _, v, _ in vals) * 1.15)

        save(fig, "01_method_comparison", f"method_comparison_{key.split('_')[0]}.png")

    # --- Multi-metric comparison: 3 subplots side by side ---
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, (metric_label, key, std_key, unit) in zip(axes, metrics):
        vals = [(m, best_per[m].get(key, 0) or 0) for m in methods]
        xs = np.arange(len(vals))
        for i, (m, v) in enumerate(vals):
            color = METHOD_COLORS.get(m, "#999999")
            is_hero = "AdaSteer" in m
            hatch = "///" if is_hero else ""
            ec = "#3040CC" if is_hero else "none"
            bar = ax.bar(i, v, 0.65, color=color, hatch=hatch,
                         edgecolor=ec, linewidth=0.7 if is_hero else 0)
            fmtstr = "{:.2f}"
            offset = max(v * 0.008, 0.06) if "psnr" in key else max(v * 0.01, 0.002)
            annotate_bar(ax, bar[0], v, fmt=fmtstr, bold=is_hero, fontsize=7, offset=offset)

        bl_val = best_per.get("No TTA", {}).get(key)
        if bl_val is not None:
            ax.axhline(bl_val, color=C_RED, ls="--", lw=1, alpha=0.6, zorder=5)

        ax.set_xticks(xs)
        ax.set_xticklabels([v[0] for v in vals], rotation=35, ha="right", fontsize=8)
        ylabel = metric_label
        if unit:
            ylabel += f" ({unit})"
        ax.set_ylabel(ylabel)
        ax.set_title(metric_label, fontweight="bold")

        lower_is_better = "LPIPS" in metric_label
        if not lower_is_better:
            ymin = min(v for _, v in vals) * 0.9
            ax.set_ylim(ymin, ax.get_ylim()[1] * 1.08)
        else:
            ax.set_ylim(0, max(v for _, v in vals) * 1.15)

    fig.suptitle("TTA Method Comparison — Standard Config (14 cond, 28 gen frames)",
                 fontweight="bold", y=1.02, fontsize=13)
    fig.tight_layout()
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

    ax.axhline(bl_psnr, color=C_RED, ls="--", lw=1.2, alpha=0.6, zorder=0)

    plotted = []
    for m in METHOD_ORDER:
        if m not in best_per or m == "No TTA":
            continue
        r = best_per[m]
        params = r.get("trainable_params")
        if params is None or params == 0:
            continue
        psnr = r["psnr_mean"]
        color = METHOD_COLORS.get(m, "#999999")
        marker = "D" if "AdaSteer" in m else "o"
        ms = 100 if "AdaSteer" in m else 65
        ax.scatter(params, psnr, c=color, s=ms, marker=marker,
                   edgecolors="white", linewidths=1.0, zorder=10)
        plotted.append((params, psnr, m, color))

    # Smart annotation placement
    for params, psnr, m, color in plotted:
        ha, dx, dy = "left", 10, 5
        if "Full" in m:
            ha, dx, dy = "right", -10, -12
        ax.annotate(m, (params, psnr), textcoords="offset points",
                    xytext=(dx, dy), fontsize=9, color=color, ha=ha,
                    fontweight="bold" if "AdaSteer" in m else "normal")

    ax.text(0.98, 0.03, f"No TTA baseline = {bl_psnr:.2f} dB",
            transform=ax.transAxes, color=C_RED, fontsize=8,
            ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Quality vs. Parameter Efficiency", fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x < 1e4 else f"{x/1e3:.0f}K" if x < 1e6
        else f"{x/1e6:.0f}M" if x < 1e9 else f"{x/1e9:.1f}B"))
    save(fig, "02_pareto", "pareto_params_vs_psnr.png")

    # --- Time vs PSNR ---
    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.axhline(bl_psnr, color=C_RED, ls="--", lw=1.2, alpha=0.6, zorder=0)

    plotted = []
    for m in METHOD_ORDER:
        if m not in best_per or m == "No TTA":
            continue
        r = best_per[m]
        tt = r.get("train_time_mean", 0)
        if tt is None:
            continue
        psnr = r["psnr_mean"]
        color = METHOD_COLORS.get(m, "#999999")
        marker = "D" if "AdaSteer" in m else "o"
        ms = 100 if "AdaSteer" in m else 65
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
                    fontweight="bold" if "AdaSteer" in m else "normal")

    ax.text(0.98, 0.03, f"No TTA baseline = {bl_psnr:.2f} dB",
            transform=ax.transAxes, color=C_RED, fontsize=8,
            ha="right", va="bottom")

    ax.set_xlabel("Training Time per Video (s)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Quality vs. Training Cost", fontweight="bold", pad=10)
    save(fig, "02_pareto", "pareto_time_vs_psnr.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 3: LR SWEEP
# ═══════════════════════════════════════════════════════════════════════

def fig_lr_sweep(data):
    print("\n[3] Learning rate sweep")
    c = complete_runs(data)

    series_map = {
        "delta_a_lr_sweep": ("AdaSteer-A", "delta_lr"),
        "delta_b_lr_sweep": ("AdaSteer-B", "delta_lr"),
        "delta_c_lr_sweep": ("Delta-C",    "delta_lr"),
        "full_lr_sweep":    ("Full-model",  "learning_rate"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    if bl:
        ax.axhline(bl[0]["psnr_mean"], color=C_RED, ls="--", lw=1.2, alpha=0.6)
        ax.text(1e-6, bl[0]["psnr_mean"] + 0.15, "No TTA",
                color=C_RED, fontsize=8, va="bottom")

    for series_name, (label, lr_key) in series_map.items():
        runs = by_series(c, series_name)
        if not runs:
            continue
        pts = sorted([(r.get(lr_key, r.get("delta_lr", r.get("learning_rate", 0))),
                       r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        color = METHOD_COLORS.get(label, "#999999")
        marker = "D" if "AdaSteer" in label else "o"
        ax.plot(xs, ys, "-", color=color, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label=label)

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Learning Rate Sweep", fontweight="bold", pad=10)
    ax.legend(frameon=False)
    save(fig, "03_lr_sweep", "lr_sweep_psnr.png")

    # Also add delta_b_low_lr combined with delta_b_lr_sweep
    fig, ax = plt.subplots(figsize=(7, 5))
    if bl:
        ax.axhline(bl[0]["psnr_mean"], color=C_RED, ls="--", lw=1.2, alpha=0.6)

    db_all = by_series(c, "delta_b_lr_sweep") + by_series(c, "delta_b_low_lr")
    if db_all:
        pts = {}
        for r in db_all:
            lr = r.get("delta_lr", 0)
            if lr not in pts or r["psnr_mean"] > pts[lr]:
                pts[lr] = r["psnr_mean"]
        pts = sorted(pts.items())
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=C_ADASTEER_B, marker="D", markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label="AdaSteer-B")

    da = by_series(c, "delta_a_lr_sweep")
    if da:
        pts = sorted([(r.get("delta_lr", 0), r["psnr_mean"]) for r in da])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=C_ADASTEER, marker="D", markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label="AdaSteer-A")

    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("AdaSteer Learning Rate Sensitivity", fontweight="bold", pad=10)
    ax.legend(frameon=False)
    save(fig, "03_lr_sweep", "lr_sweep_adasteer_detail.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 4: ITERATION SWEEP
# ═══════════════════════════════════════════════════════════════════════

def fig_iter_sweep(data):
    print("\n[4] Iteration sweep")
    c = complete_runs(data)

    series_map = {
        "delta_a_iter_sweep": ("AdaSteer-A", "delta_steps"),
        "delta_b_iter_sweep": ("AdaSteer-B", "delta_steps"),
        "delta_c_iter_sweep": ("Delta-C",    "delta_steps"),
        "full_iter_sweep":    ("Full-model",  "num_steps"),
        "lora_iter_sweep":    ("LoRA",        "num_steps"),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    if bl:
        ax.axhline(bl[0]["psnr_mean"], color=C_RED, ls="--", lw=1.2, alpha=0.6)
        ax.text(82, bl[0]["psnr_mean"] + 0.15, "No TTA",
                color=C_RED, fontsize=8, va="bottom")

    for series_name, (label, step_key) in series_map.items():
        runs = by_series(c, series_name)
        if not runs:
            continue
        pts = sorted([(r.get(step_key, r.get("delta_steps", r.get("num_steps", 0))),
                       r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        color = METHOD_COLORS.get(label, "#999999")
        marker = "D" if "AdaSteer" in label else "o"
        ax.plot(xs, ys, "-", color=color, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label=label)

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Training Steps Sweep", fontweight="bold", pad=10)
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
        ax.plot(xs, ys, "-", color=C_ADASTEER_B, marker="D", markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, lw=2, label="20-step")
    ax.axhline(bl_psnr, color=C_RED, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Number of Groups (G)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("AdaSteer-B Groups (20 steps)", fontweight="bold")
    ax.legend(frameon=False)

    # Panel B: adasteer_groups_5step
    ax = axes[1]
    runs = by_series(c, "adasteer_groups_5step")
    if runs:
        pts = sorted([(r.get("num_groups", 1), r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        ax.plot(xs, ys, "-", color=C_ADASTEER, marker="D", markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, lw=2, label="5-step")
    ax.axhline(bl_psnr, color=C_RED, ls="--", lw=1, alpha=0.6)
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
    ax.axhline(bl_psnr, color=C_RED, ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("LoRA Rank")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("LoRA Rank Sweep", fontweight="bold")

    # Panel B: Params vs PSNR (all LoRA variants)
    ax = axes[1]
    lora_series = ["lora_rank_sweep", "lora_constrained_sweep",
                   "lora_ultra_constrained", "lora_builtin_comparison"]
    colors_lora = [C_LORA, "#6690C0", "#5577AA", "#99BBDD"]
    labels_lora = ["Standard rank", "Constrained", "Ultra-constrained", "Built-in comparison"]

    for si, (sname, col, lab) in enumerate(zip(lora_series, colors_lora, labels_lora)):
        runs = by_series(c, sname)
        if not runs:
            continue
        pts = sorted([(r.get("trainable_params", 0), r["psnr_mean"]) for r in runs])
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=col, s=40, alpha=0.8, label=lab, zorder=5)

    ax.axhline(bl_psnr, color=C_RED, ls="--", lw=1, alpha=0.6)
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

    series_methods = {
        "exp3_train_frames_delta_a": "AdaSteer-A",
        "exp3_train_frames_delta_b": "AdaSteer-B",
        "exp3_train_frames_delta_c": "Delta-C",
        "exp3_train_frames_full":    "Full-model",
        "exp3_train_frames_lora":    "LoRA",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    if bl:
        ax.axhline(bl[0]["psnr_mean"], color=C_RED, ls="--", lw=1, alpha=0.6)
        ax.text(23, bl[0]["psnr_mean"] + 0.15, "No TTA",
                color=C_RED, fontsize=8, va="bottom")

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
        marker = "D" if "AdaSteer" in label else "o"
        ax.plot(xs, ys, "-", color=color, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label=label)

    ax.set_xlabel("Conditioning Frames (num_cond_frames)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Effect of Conditioning Frames on TTA Quality", fontweight="bold", pad=10)
    ax.legend(frameon=False)
    ax.set_xticks([2, 7, 14, 24])
    save(fig, "07_cond_frames", "cond_frames_psnr.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 8: GENERATION HORIZON ABLATION (exp4)
# ═══════════════════════════════════════════════════════════════════════

def fig_gen_horizon(data):
    print("\n[8] Generation horizon ablation (exp4)")
    c = complete_runs(data)

    series_methods = {
        "exp4_gen_horizon_delta_a": "AdaSteer-A",
        "exp4_gen_horizon_delta_b": "AdaSteer-B",
        "exp4_gen_horizon_delta_c": "Delta-C",
        "exp4_gen_horizon_full":    "Full-model",
        "exp4_gen_horizon_lora":    "LoRA",
    }

    fig, ax = plt.subplots(figsize=(7, 5))

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
        marker = "D" if "AdaSteer" in label else "o"
        ax.plot(xs, ys, "-", color=color, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=0.8, lw=1.8, label=label)

    ax.set_xlabel("Generation Frames (num_frames)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Effect of Generation Horizon on TTA Quality", fontweight="bold", pad=10)
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

    common = [m for m in ["AdaSteer-A", "Full-model", "LoRA", "No TTA"]
              if m in panda_best and m in ucf_best]

    if not common:
        print("  (skip: no common methods)")
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bar_w = 0.32
    xs = np.arange(len(common))

    bars_panda = ax.bar(xs - bar_w / 2, [panda_best[m]["psnr_mean"] for m in common],
                        bar_w, color=C_ADASTEER, alpha=0.85, label="Panda-70M")
    bars_ucf = ax.bar(xs + bar_w / 2, [ucf_best[m]["psnr_mean"] for m in common],
                      bar_w, color=C_DELTAC, alpha=0.85, label="UCF-101")

    for b in bars_panda:
        annotate_bar(ax, b, b.get_height(), fmt="{:.1f}", fontsize=8, offset=0.15)
    for b in bars_ucf:
        annotate_bar(ax, b, b.get_height(), fmt="{:.1f}", fontsize=8, offset=0.15)

    ax.set_xticks(xs)
    ax.set_xticklabels(common)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Cross-Dataset Generalization", fontweight="bold", pad=10)
    ax.legend(frameon=False)

    ymin = min(panda_best[m]["psnr_mean"] for m in common) * 0.9
    ymin = min(ymin, min(ucf_best[m]["psnr_mean"] for m in common) * 0.9)
    ax.set_ylim(ymin, ax.get_ylim()[1] * 1.08)

    save(fig, "09_cross_dataset", "cross_dataset_psnr.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG 10: EARLY STOPPING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════

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

        ax.plot(patience_vals[:len(psnrs)], psnrs, "-o", color=C_ADASTEER,
                markersize=6, markeredgecolor="white", lw=1.8, label="PSNR")
        ax.set_xlabel("Patience")
        ax.set_ylabel("PSNR (dB)", color=C_ADASTEER)

        ax2 = ax.twinx()
        ax2.bar(patience_vals[:len(stopped_frac)],
                [f * 100 for f in stopped_frac],
                0.6, color=C_LIGHT_T, alpha=0.6, label="% Stopped")
        ax2.set_ylabel("% Videos Stopped Early", color="#666666")
        ax2.set_ylim(0, 105)
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

        ax.plot(check_freqs[:len(psnrs)], psnrs, "-o", color=C_ADASTEER,
                markersize=6, markeredgecolor="white", lw=1.8, label="PSNR")
        ax.set_xlabel("Check Every N Steps")
        ax.set_ylabel("PSNR (dB)", color=C_ADASTEER)

        ax2 = ax.twinx()
        ax2.bar(check_freqs[:len(stopped_frac)],
                [f * 100 for f in stopped_frac],
                0.6, color=C_LIGHT_T, alpha=0.6, label="% Stopped")
        ax2.set_ylabel("% Videos Stopped Early", color="#666666")
        ax2.set_ylim(0, 105)
        ax.set_title("Early Stopping: Check Frequency", fontweight="bold")

    fig.tight_layout()
    save(fig, "10_early_stopping", "es_ablation.png")

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
                    f"avg best = step {best_step:.0f}  ({stopped}/{total_v} stopped)",
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
    methods = [m for m in METHOD_ORDER if m in best_per and m != "No TTA"]

    # --- Bar chart: training time per video ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(methods))
    for i, m in enumerate(methods):
        r = best_per[m]
        tt = r.get("train_time_mean", 0)
        gt = r.get("gen_time_mean", 80)
        color = METHOD_COLORS.get(m, "#999999")
        hatch = "///" if "AdaSteer" in m else ""
        edgecolor = color if hatch else "none"

        bar = ax.bar(i, tt, 0.55, color=color, hatch=hatch,
                     edgecolor=edgecolor, linewidth=0.5)
        annotate_bar(ax, bar[0], tt, fmt="{:.0f}s", fontsize=8,
                     bold="AdaSteer" in m, offset=max(tt * 0.02, 1))

    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Training Time per Video (s)")
    ax.set_title("TTA Training Cost per Video", fontweight="bold", pad=10)
    save(fig, "11_time_cost", "train_time.png")

    # --- Train/Gen ratio ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, m in enumerate(methods):
        r = best_per[m]
        ratio = r.get("train_gen_ratio", 0) or 0
        color = METHOD_COLORS.get(m, "#999999")
        hatch = "///" if "AdaSteer" in m else ""
        edgecolor = color if hatch else "none"

        bar = ax.bar(i, ratio, 0.55, color=color, hatch=hatch,
                     edgecolor=edgecolor, linewidth=0.5)
        annotate_bar(ax, bar[0], ratio, fmt="{:.2f}x", fontsize=8,
                     bold="AdaSteer" in m, offset=max(ratio * 0.02, 0.01))

    ax.axhline(1.0, color=C_RED, ls="--", lw=1, alpha=0.6)
    ax.text(len(methods) - 0.5, 1.03, "Train = Gen time", color=C_RED,
            fontsize=8, ha="right")

    ax.set_xticks(xs)
    ax.set_xticklabels(methods, rotation=25, ha="right")
    ax.set_ylabel("Train Time / Generation Time")
    ax.set_title("Training Overhead Relative to Generation", fontweight="bold", pad=10)
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
        (axes[1], window_b, "Window B (8s, gen_start=200)", C_ADASTEER_B),
    ]:
        # Separate no-TTA control
        control_key = [k for k in window if k.endswith("0")]
        tta_keys = [k for k in sorted(window.keys()) if not k.endswith("0")]

        if control_key:
            ctrl = window[control_key[0]]
            ax.axhline(ctrl["psnr_mean"], color=C_RED, ls="--", lw=1, alpha=0.6)
            ax.text(0.5, ctrl["psnr_mean"] + 0.05, f"No TTA = {ctrl['psnr_mean']:.2f}",
                    color=C_RED, fontsize=8, transform=ax.get_yaxis_transform(), va="bottom")

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
    ax.axhline(bl_psnr, color=C_RED, ls="--", lw=1, alpha=0.6)

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
        ax.axhline(bl[0]["psnr_mean"], color=C_RED, ls="--", lw=1, alpha=0.5)

    seen_methods = set()
    for r in c:
        md = method_display(r.get("method", ""))
        params = r.get("trainable_params")
        if params is None or params == 0 or md == "No TTA":
            continue
        color = METHOD_COLORS.get(md, "#999999")
        label = md if md not in seen_methods else None
        seen_methods.add(md)
        marker = "D" if "AdaSteer" in md else "o"
        ax.scatter(params, r["psnr_mean"], c=color, s=20, alpha=0.5,
                   marker=marker, label=label, zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Trainable Parameters")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("All Experiment Runs", fontweight="bold", pad=10)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x:.0f}" if x < 1e3 else f"{x/1e3:.0f}K" if x < 1e6
        else f"{x/1e6:.0f}M" if x < 1e9 else f"{x/1e9:.1f}B"))
    save(fig, "01_method_comparison", "all_runs_scatter.png")


# ═══════════════════════════════════════════════════════════════════════
# FIG BONUS: NormTune & FiLM detail
# ═══════════════════════════════════════════════════════════════════════

def fig_normtune_film(data):
    print("\n[Bonus] NormTune & FiLM sweep detail")
    c = complete_runs(data)
    bl = [r for r in c if r["series"] == "panda_no_tta_continuation"]
    bl_psnr = bl[0]["psnr_mean"] if bl else 22.07

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # NormTune
    ax = axes[0]
    runs = by_series(c, "norm_tune_sweep")
    if runs:
        for r in runs:
            params = r.get("trainable_params", 0)
            lr = r.get("norm_lr") or r.get("learning_rate") or 0
            psnr = r["psnr_mean"]
            ax.scatter(lr, psnr, c=C_NORMTUNE, s=60, zorder=5, edgecolors="white")
            ax.annotate(f"{params/1e3:.0f}K", (lr, psnr), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color="#666666")
    ax.axhline(bl_psnr, color=C_RED, ls="--", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("NormTune Sweep", fontweight="bold")

    # FiLM
    ax = axes[1]
    runs = by_series(c, "film_adapter_sweep")
    if runs:
        for r in runs:
            params = r.get("trainable_params", 0)
            lr = r.get("film_lr") or r.get("learning_rate") or 0
            psnr = r["psnr_mean"]
            ax.scatter(lr, psnr, c=C_FILM, s=60, zorder=5, edgecolors="white")
            ax.annotate(f"{params/1e3:.0f}K", (lr, psnr), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color="#666666")
    ax.axhline(bl_psnr, color=C_RED, ls="--", lw=1, alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("FiLM Adapter Sweep", fontweight="bold")

    fig.tight_layout()
    save(fig, "01_method_comparison", "normtune_film_detail.png")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def fig_summary_table(data):
    """Render a visual summary table of all methods."""
    print("\n[Table] Summary table figure")
    c = complete_runs(data)
    best_per = _get_standard_best(c)
    methods = [m for m in METHOD_ORDER if m in best_per]

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
    fig_normtune_film(data)
    fig_summary_table(data)

    print(f"\nAll figures saved to {OUT_ROOT}")


if __name__ == "__main__":
    main()
