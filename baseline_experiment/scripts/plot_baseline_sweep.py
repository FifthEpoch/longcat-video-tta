#!/usr/bin/env python3
"""
Plot baseline sweep results: 6 graphs comparing metrics across configs.

  Row 1: PSNR  — (a) vs conditioning frames, (b) vs generated frames
  Row 2: SSIM  — (a) vs conditioning frames, (b) vs generated frames
  Row 3: LPIPS — (a) vs conditioning frames, (b) vs generated frames

Usage:
  python plot_baseline_sweep.py --results-root baseline_experiment/results
  python plot_baseline_sweep.py --results-root baseline_experiment/results --out-dir plots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────

# Varying conditioning frames (gen fixed at 14)
COND_SWEEP = [
    {"label": "2",  "dir": "cond2_gen14",  "cond": 2,  "gen": 14},
    {"label": "14", "dir": "cond14_gen14", "cond": 14, "gen": 14},
    {"label": "28", "dir": "cond28_gen14", "cond": 28, "gen": 14},
]

# Varying generated frames (cond fixed at 2)
GEN_SWEEP = [
    {"label": "7",  "dir": "cond2_gen7",  "cond": 2, "gen": 7},
    {"label": "14", "dir": "cond2_gen14", "cond": 2, "gen": 14},
    {"label": "28", "dir": "cond2_gen28", "cond": 2, "gen": 28},
]

METRICS = ["psnr", "ssim", "lpips"]
METRIC_LABELS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS"}


# ── Helpers ───────────────────────────────────────────────────────────────

def load_summary(results_root: Path, subdir: str) -> dict | None:
    p = results_root / subdir / "summary.json"
    if not p.exists():
        print(f"  WARNING: {p} not found, skipping")
        return None
    with open(p) as f:
        return json.load(f)


def extract_metric(summary: dict, metric: str) -> tuple[float, float]:
    """Return (mean, std) for a metric from summary.json."""
    m = summary.get("metrics", {}).get(metric, {})
    return m.get("mean", float("nan")), m.get("std", float("nan"))


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_sweep(ax, sweep_configs, results_root, metric, x_label):
    """Plot a single metric across a sweep on the given axes."""
    x_labels = []
    means = []
    stds = []

    for cfg in sweep_configs:
        summary = load_summary(results_root, cfg["dir"])
        if summary is None:
            x_labels.append(cfg["label"])
            means.append(float("nan"))
            stds.append(0)
            continue

        mean, std = extract_metric(summary, metric)
        x_labels.append(cfg["label"])
        means.append(mean)
        stds.append(std)

    x = np.arange(len(x_labels))
    bars = ax.bar(x, means, yerr=stds, capsize=6, width=0.5,
                  color="#4C72B0", edgecolor="black", linewidth=0.8, alpha=0.85)

    # Value labels on bars
    for i, (bar, m, s) in enumerate(zip(bars, means, stds)):
        if not np.isnan(m):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.01,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
    ax.grid(axis="y", alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-root", type=str, required=True,
                   help="Path to baseline_experiment/results/")
    p.add_argument("--out-dir", type=str, default=None,
                   help="Directory to save plots (default: results-root)")
    args = p.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check that at least some results exist
    found = sum(1 for c in COND_SWEEP + GEN_SWEEP
                if (results_root / c["dir"] / "summary.json").exists())
    if found == 0:
        print(f"ERROR: No summary.json files found under {results_root}")
        sys.exit(1)
    print(f"Found {found}/{len(COND_SWEEP) + len(GEN_SWEEP)} result directories")

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Baseline Sweep: LongCat-Video Continuation on Panda-70M (480p)",
                 fontsize=14, fontweight="bold", y=0.98)

    for row, metric in enumerate(METRICS):
        # Left column: varying conditioning frames
        plot_sweep(axes[row, 0], COND_SWEEP, results_root, metric,
                   "Conditioning Frames (gen=14 fixed)")
        axes[row, 0].set_title(f"{METRIC_LABELS[metric]} vs Conditioning Frames",
                               fontsize=11)

        # Right column: varying generated frames
        plot_sweep(axes[row, 1], GEN_SWEEP, results_root, metric,
                   "Generated Frames (cond=2 fixed)")
        axes[row, 1].set_title(f"{METRIC_LABELS[metric]} vs Generated Frames",
                               fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = out_dir / "baseline_sweep_metrics.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    # Also save individual plots for flexibility
    for row, metric in enumerate(METRICS):
        for col, (sweep, xlabel, suffix) in enumerate([
            (COND_SWEEP, "Conditioning Frames (gen=14 fixed)", "cond"),
            (GEN_SWEEP, "Generated Frames (cond=2 fixed)", "gen"),
        ]):
            fig_i, ax_i = plt.subplots(figsize=(6, 4))
            plot_sweep(ax_i, sweep, results_root, metric, xlabel)
            ax_i.set_title(f"{METRIC_LABELS[metric]} vs {'Conditioning' if col == 0 else 'Generated'} Frames",
                           fontsize=12)
            fig_i.tight_layout()
            ind_path = out_dir / f"{metric}_vs_{suffix}_frames.png"
            fig_i.savefig(str(ind_path), dpi=150, bbox_inches="tight")
            plt.close(fig_i)

    print(f"Saved 6 individual plots to {out_dir}/")
    plt.close("all")


if __name__ == "__main__":
    main()
