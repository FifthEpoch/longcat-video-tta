#!/usr/bin/env python3
"""
Plot baseline sweep results with Panda-70M and UCF-101 side by side.

Produces 6 individual plots:
  psnr_vs_cond_frames.png   psnr_vs_gen_frames.png
  ssim_vs_cond_frames.png   ssim_vs_gen_frames.png
  lpips_vs_cond_frames.png  lpips_vs_gen_frames.png

Each plot shows grouped bars: Panda-70M (#4169E1) and UCF-101 (#0000FF).

Usage:
  python plot_baseline_sweep_dual.py \
      --results-root /scratch/wc3013/longcat-video-tta/baseline_experiment/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ── Configuration ─────────────────────────────────────────────────────────

METRICS = ["psnr", "ssim", "lpips"]
METRIC_LABELS = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS"}

PANDA_COLOR = "#4169E1"
UCF101_COLOR = "#0000FF"

BAR_WIDTH = 0.30


def build_sweep_pair(sweep_type: str):
    """Return list of dicts with x-labels and dir names for both datasets.

    sweep_type: 'cond' or 'gen'
    """
    if sweep_type == "cond":
        labels = ["2", "14", "28"]
        panda_dirs = ["cond2_gen14", "cond14_gen14", "cond28_gen14"]
        ucf_dirs = ["ucf101_cond2_gen14", "ucf101_cond14_gen14", "ucf101_cond28_gen14"]
    else:  # gen
        labels = ["7", "14", "28"]
        panda_dirs = ["cond2_gen7", "cond2_gen14", "cond2_gen28"]
        ucf_dirs = ["ucf101_cond2_gen7", "ucf101_cond2_gen14", "ucf101_cond2_gen28"]
    return labels, panda_dirs, ucf_dirs


# ── Helpers ───────────────────────────────────────────────────────────────

def load_summary(results_root: Path, subdir: str) -> dict | None:
    p = results_root / subdir / "summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def extract_metric(summary: dict | None, metric: str) -> tuple[float, float]:
    """Return (mean, std). Returns (nan, 0) if unavailable."""
    if summary is None:
        return float("nan"), 0.0
    m = summary.get("metrics", {}).get(metric, {})
    return m.get("mean", float("nan")), m.get("std", float("nan"))


# ── Plotting ──────────────────────────────────────────────────────────────

def plot_dual_sweep(
    ax,
    labels: list[str],
    panda_dirs: list[str],
    ucf_dirs: list[str],
    results_root: Path,
    metric: str,
    x_label: str,
):
    """Plot grouped bars for Panda-70M and UCF-101 on the same axes."""
    x = np.arange(len(labels))

    panda_means, panda_stds = [], []
    ucf_means, ucf_stds = [], []

    for pd, ud in zip(panda_dirs, ucf_dirs):
        pm, ps = extract_metric(load_summary(results_root, pd), metric)
        um, us = extract_metric(load_summary(results_root, ud), metric)
        panda_means.append(pm)
        panda_stds.append(ps if not np.isnan(ps) else 0)
        ucf_means.append(um)
        ucf_stds.append(us if not np.isnan(us) else 0)

    bars_panda = ax.bar(
        x - BAR_WIDTH / 2, panda_means, BAR_WIDTH,
        yerr=panda_stds, capsize=5,
        color=PANDA_COLOR, edgecolor="black", linewidth=0.6, alpha=0.85,
        label="Panda-70M",
    )
    bars_ucf = ax.bar(
        x + BAR_WIDTH / 2, ucf_means, BAR_WIDTH,
        yerr=ucf_stds, capsize=5,
        color=UCF101_COLOR, edgecolor="black", linewidth=0.6, alpha=0.85,
        label="UCF-101",
    )

    # Value labels on bars
    for bars, means, stds in [(bars_panda, panda_means, panda_stds),
                               (bars_ucf, ucf_means, ucf_stds)]:
        for bar, m, s in zip(bars, means, stds):
            if not np.isnan(m):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + s + 0.005,
                    f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(METRIC_LABELS[metric], fontsize=11)
    ax.legend(fontsize=9, loc="best")
    ax.grid(axis="y", alpha=0.3)


def main():
    p = argparse.ArgumentParser(
        description="Plot Panda-70M vs UCF-101 baseline sweep side by side"
    )
    p.add_argument(
        "--results-root", type=str, required=True,
        help="Path to baseline_experiment/results/",
    )
    p.add_argument(
        "--out-dir", type=str, default=None,
        help="Directory to save plots (default: results-root)",
    )
    args = p.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir) if args.out_dir else results_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check that at least some results exist
    test_dirs = ["cond2_gen14", "ucf101_cond2_gen14"]
    found = sum(
        1 for d in test_dirs
        if (results_root / d / "summary.json").exists()
    )
    if found == 0:
        print(f"ERROR: No summary.json found under {results_root}")
        sys.exit(1)

    sweep_configs = [
        ("cond", "Conditioning Frames (gen=14 fixed)", "cond_frames"),
        ("gen", "Generated Frames (cond=2 fixed)", "gen_frames"),
    ]

    for sweep_type, xlabel, suffix in sweep_configs:
        labels, panda_dirs, ucf_dirs = build_sweep_pair(sweep_type)

        for metric in METRICS:
            fig, ax = plt.subplots(figsize=(7, 4.5))

            plot_dual_sweep(
                ax, labels, panda_dirs, ucf_dirs,
                results_root, metric, xlabel,
            )

            vary = "Conditioning" if sweep_type == "cond" else "Generated"
            ax.set_title(
                f"{METRIC_LABELS[metric]} vs {vary} Frames — "
                f"Panda-70M vs UCF-101 (480p)",
                fontsize=12,
            )

            fig.tight_layout()
            out_path = out_dir / f"{metric}_vs_{suffix}.png"
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_path}")

    # Also produce combined 3x2 grid
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(
        "Baseline Sweep: Panda-70M vs UCF-101 (LongCat-Video, 480p)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    for col, (sweep_type, xlabel, _) in enumerate(sweep_configs):
        labels, panda_dirs, ucf_dirs = build_sweep_pair(sweep_type)
        for row, metric in enumerate(METRICS):
            plot_dual_sweep(
                axes[row, col], labels, panda_dirs, ucf_dirs,
                results_root, metric, xlabel,
            )
            vary = "Conditioning" if sweep_type == "cond" else "Generated"
            axes[row, col].set_title(
                f"{METRIC_LABELS[metric]} vs {vary} Frames", fontsize=11,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    grid_path = out_dir / "baseline_sweep_metrics_dual.png"
    fig.savefig(str(grid_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {grid_path}")

    print(f"\nAll 7 plots saved to {out_dir}/")


if __name__ == "__main__":
    main()
