#!/usr/bin/env python3
"""Slide 1c figure: native drift GROWS with horizon (30s -> 60s).

Directly supports the claim "drift grows monotonically with length, so a
correction has more room the longer you roll." Uses LongCat's NATIVE window
(13-cond/80-gen) at two horizons, so the magnitudes match the honest Slide 1c /
Slide 3 numbers (NOT the overstated reencode 14/14 protocol).

Data (GT-free signal drift, chunk 1 -> last, % change):
  native 30 s = 6 chunks,  N=12  (longhorizon_sweep_notta_native_6ch)
  native 60 s = 12 chunks, N=8   (longhorizon_sweep_notta_native_12ch, 2026-08-09)

Run:
    python3 scripts/make_drift_horizon_growth_fig.py \
        --out-dir sweep_experiment/reports/paper_figures/2026-08-08_longhorizon_drift
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["Sharpness\n(HF artifacts)", "Temporal\nmotion", "Contrast", "Colorfulness\n(saturation)"]
NATIVE_30S = np.array([27.9, 8.2, 2.8, 4.1])    # 6 chunks,  N=12
NATIVE_60S = np.array([48.0, 45.1, -16.4, 5.7])  # 12 chunks, N=8

C30, C60 = "#8ecae6", "#023047"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="sweep_experiment/reports/paper_figures/2026-08-08_longhorizon_drift")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 12,
        "axes.titlesize": 14, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "legend.frameon": False,
    })
    x = np.arange(len(LABELS)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5.6))
    b1 = ax.bar(x - w/2, NATIVE_30S, w, color=C30,
                label="Native 30 s (6 chunks, N=12)")
    b2 = ax.bar(x + w/2, NATIVE_60S, w, color=C60,
                label="Native 60 s (12 chunks, N=8)")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(LABELS)
    ax.set_ylabel("GT-free drift (% change, chunk 1 → last)")
    ax.set_title("Native drift GROWS with horizon (30 s → 60 s)\n"
                 "more headroom for a correction the longer you roll")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:+.0f}%", (r.get_x()+r.get_width()/2, h),
                        textcoords="offset points",
                        xytext=(0, 4 if h >= 0 else -13),
                        ha="center", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right")
    # honest annotation: contrast develops a fade only at long horizon
    ax.annotate("contrast fade\nappears only at 60 s", (2 + w/2, -16.4),
                textcoords="offset points", xytext=(6, -6),
                ha="left", va="top", fontsize=8.5, color="0.35")
    fig.tight_layout()
    p = os.path.join(args.out_dir, "drift_native_horizon_growth.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
