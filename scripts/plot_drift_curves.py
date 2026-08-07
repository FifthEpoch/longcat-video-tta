#!/usr/bin/env python3
"""
Plot per-chunk drift curves from the long-horizon NOTTA diagnostic.

Reads the ``summary.json`` written by
``delta_experiment/scripts/diag_longhorizon_drift.py`` and emits one PNG per
metric (metric vs chunk index): a faint line per video plus a bold mean +/- std
band. Over-smoothing shows as sharpness/motion DECREASING with chunk index;
over-saturation as colorfulness INCREASING; GT metrics (psnr/ssim/lpips) are
plotted only over the chunks where GT still overlapped the rollout.

Usage
-----
    python scripts/plot_drift_curves.py \
        --summary sweep_experiment/results/diag_longhorizon_drift/summary.json \
        --out-dir sweep_experiment/results/diag_longhorizon_drift/plots
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# (key, human label, expected drift direction if the model degrades)
_METRICS = [
    ("sharpness", "Sharpness (Laplacian var)  \u2014 over-smoothing \u2193", "down"),
    ("temporal_motion", "Temporal motion (|\u0394frame|)  \u2014 motion collapse \u2193", "down"),
    ("colorfulness", "Colorfulness (Hasler-S\u00fcsstrunk)  \u2014 over-saturation \u2191", "up"),
    ("saturation", "Mean saturation", None),
    ("contrast", "Contrast (luma std)", None),
    ("brightness", "Brightness (mean luma)", None),
    ("seam_ratio", "Cross-chunk seam / motion ratio", "up"),
    ("psnr", "PSNR vs GT (dB)  \u2014 where GT overlaps", None),
    ("ssim", "SSIM vs GT  \u2014 where GT overlaps", None),
    ("lpips", "LPIPS vs GT  \u2014 where GT overlaps", None),
]


def _series_per_video(results: List[Dict], key: str, num_chunks: int):
    out = []
    for r in results:
        if not r.get("success"):
            continue
        ys = [None] * num_chunks
        for ch in r.get("chunks", []):
            ci = ch["chunk"] - 1
            if 0 <= ci < num_chunks:
                v = ch.get(key)
                ys[ci] = v if (v is not None and v == v) else None
        out.append(ys)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)

    num_chunks = int(summary["num_chunks"])
    results = [r for r in summary.get("results", []) if r.get("success")]
    curves = summary.get("drift_curves", {})
    os.makedirs(args.out_dir, exist_ok=True)
    xs = np.arange(1, num_chunks + 1)

    written = []
    for key, label, _direction in _METRICS:
        cur = curves.get(key)
        if cur is None:
            continue
        means = [m if m is not None else np.nan for m in cur["mean"]]
        stds = [s if s is not None else np.nan for s in cur["std"]]
        if all(np.isnan(means)):
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for ys in _series_per_video(results, key, num_chunks):
            yv = np.array([y if y is not None else np.nan for y in ys], dtype=float)
            ax.plot(xs, yv, color="0.75", linewidth=0.8, alpha=0.6, zorder=1)

        m = np.array(means, dtype=float)
        s = np.array(stds, dtype=float)
        ax.plot(xs, m, color="C0", linewidth=2.4, marker="o", zorder=3, label="mean")
        ax.fill_between(xs, m - s, m + s, color="C0", alpha=0.18, zorder=2)

        ax.set_xlabel("autoregressive chunk index")
        ax.set_ylabel(key)
        ax.set_title(label, fontsize=10)
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)

        v = summary.get("drift_verdict", {}).get(key)
        if v is not None:
            ax.annotate(
                f"{v['pct_change']:+.1f}% over rollout\nslope={v['slope_per_chunk']:+.4g}/chunk",
                xy=(0.98, 0.03), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
            )
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out = os.path.join(args.out_dir, f"drift_{key}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        written.append(out)
        print(f"wrote {out}")

    # combined headline panel
    headline = ["sharpness", "temporal_motion", "colorfulness", "psnr"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, key in zip(axes.ravel(), headline):
        cur = curves.get(key)
        if cur is None or all(m is None for m in cur["mean"]):
            ax.set_visible(False)
            continue
        m = np.array([x if x is not None else np.nan for x in cur["mean"]], dtype=float)
        s = np.array([x if x is not None else np.nan for x in cur["std"]], dtype=float)
        ax.plot(xs, m, color="C0", linewidth=2.2, marker="o")
        ax.fill_between(xs, m - s, m + s, color="C0", alpha=0.18)
        ax.set_title(key)
        ax.set_xlabel("chunk index")
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"NOTTA long-horizon drift ({summary.get('num_successful', '?')} videos "
        f"x {num_chunks} chunks, LongCat-Video)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    combined = os.path.join(args.out_dir, "drift_headline.png")
    fig.savefig(combined, dpi=140)
    plt.close(fig)
    written.append(combined)
    print(f"wrote {combined}")
    print(f"\n{len(written)} plots -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
