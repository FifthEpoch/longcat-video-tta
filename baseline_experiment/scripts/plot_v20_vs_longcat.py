#!/usr/bin/env python3
"""
Plot comparing Open-Sora v2.0 vs LongCat-Video baseline performance.

Produces three separate plots (one per metric): PSNR, SSIM, LPIPS.
Each plot shows the mean ± std as a bar with error bars.

Usage:
  python baseline_experiment/scripts/plot_v20_vs_longcat.py
  python baseline_experiment/scripts/plot_v20_vs_longcat.py --output-dir my_plots/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Data ──────────────────────────────────────────────────────────────────

# Open-Sora v2.0 baseline  (2 cond, 16 gen, 256px 16:9, Panda-70M 100 videos)
V20 = {
    "psnr_mean": 12.03, "psnr_std": 3.53,
    "ssim_mean": 0.4123, "ssim_std": 0.1599,
    "lpips_mean": 0.4764, "lpips_std": 0.1467,
}

# LongCat-Video baseline  (14 cond, 14 gen, 480p 16:9, Panda-70M 100 videos)
LONGCAT = {
    "psnr_mean": 14.98, "psnr_std": 7.24,
    "ssim_mean": 0.5732, "ssim_std": 0.2209,
    "lpips_mean": 0.4297, "lpips_std": 0.2436,
}

V20_LABEL = "Open-Sora v2.0\n(2 cond / 16 gen, 256px)"
LONGCAT_LABEL = "LongCat-Video\n(14 cond / 14 gen, 480p)"

V20_COLOR = "#3B7DD8"
LONGCAT_COLOR = "#E8573A"

BAR_WIDTH = 0.45


# ── Plotting ──────────────────────────────────────────────────────────────

def make_metric_plot(
    out_path: Path,
    metric: str,
    ylabel: str,
    title: str,
    lower_is_better: bool = False,
):
    v20_mean = V20[f"{metric}_mean"]
    v20_std = V20[f"{metric}_std"]
    lc_mean = LONGCAT[f"{metric}_mean"]
    lc_std = LONGCAT[f"{metric}_std"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    xs = [0, 1]
    means = [v20_mean, lc_mean]
    stds = [v20_std, lc_std]
    colors = [V20_COLOR, LONGCAT_COLOR]
    labels = [V20_LABEL, LONGCAT_LABEL]

    bars = ax.bar(xs, means, width=BAR_WIDTH, color=colors, edgecolor="white",
                  linewidth=0.8, zorder=3)
    ax.errorbar(xs, means, yerr=stds, fmt="none", ecolor="black",
                capsize=6, capthick=1.5, linewidth=1.5, zorder=4)

    # Value annotations
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + (0.3 if metric == "psnr" else 0.015),
                f"{m:.2f} ± {s:.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.25, zorder=0)

    # Better axis
    if metric == "psnr":
        ymin = max(0, min(means) - max(stds) - 2)
        ymax = max(means) + max(stds) + 3
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0, 1.0)

    # Arrow annotation indicating better direction
    direction = "↓ lower is better" if lower_is_better else "↑ higher is better"
    ax.annotate(direction, xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, fontstyle="italic",
                color="gray")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def print_table():
    metrics = [
        ("PSNR",  "psnr",  "higher↑"),
        ("SSIM",  "ssim",  "higher↑"),
        ("LPIPS", "lpips", "lower↓"),
    ]
    print()
    print("Baseline Comparison: Open-Sora v2.0 vs LongCat-Video")
    print("-" * 80)
    print(f"{'Metric':<8} {'Open-Sora v2.0':>22} {'LongCat-Video':>22} {'Δ':>10} {'Dir':>10}")
    print("-" * 80)
    for name, key, direction in metrics:
        v20_m = V20[f"{key}_mean"]
        v20_s = V20[f"{key}_std"]
        lc_m = LONGCAT[f"{key}_mean"]
        lc_s = LONGCAT[f"{key}_std"]
        delta = lc_m - v20_m
        print(f"{name:<8} {v20_m:>8.2f} ± {v20_s:<8.2f}  {lc_m:>8.2f} ± {lc_s:<8.2f} {delta:>+8.2f}  {direction:>8}")
    print("-" * 80)
    print()
    print("Open-Sora v2.0 config : 2 cond frames, 16 gen frames, 256px, 16:9")
    print("LongCat-Video config  : 14 cond frames, 14 gen frames, 480p, 16:9")
    print()


def main():
    p = argparse.ArgumentParser(description="Plot Open-Sora v2.0 vs LongCat baseline comparison")
    p.add_argument("--output-dir", type=str,
                   default="baseline_experiment/plots",
                   help="Directory to save plots")
    args = p.parse_args()

    print_table()

    out_dir = Path(args.output_dir)

    make_metric_plot(
        out_dir / "v20_vs_longcat_psnr.png",
        metric="psnr", ylabel="PSNR (dB)",
        title="Baseline PSNR — Open-Sora v2.0 vs LongCat-Video",
    )
    make_metric_plot(
        out_dir / "v20_vs_longcat_ssim.png",
        metric="ssim", ylabel="SSIM",
        title="Baseline SSIM — Open-Sora v2.0 vs LongCat-Video",
    )
    make_metric_plot(
        out_dir / "v20_vs_longcat_lpips.png",
        metric="lpips", ylabel="LPIPS",
        title="Baseline LPIPS — Open-Sora v2.0 vs LongCat-Video",
        lower_is_better=True,
    )

    print(f"All plots saved to: {out_dir}/")


if __name__ == "__main__":
    main()
