#!/usr/bin/env python3
"""Intervention comparison figure: NOTTA vs a FIXED AdaSteer delta held across
an 8-chunk rollout (both reencode geometry, N=24, paired seeds).

Data embedded verbatim from summary.json verdicts/curves:
  NOTTA  = job 15497180 (diag_longhorizon_drift)
  DELTA  = diag_longhorizon_drift_delta_reencode (delta_norm mean 0.139)

Run:
    python3 scripts/make_drift_intervention_figs.py \
        --out-dir sweep_experiment/reports/paper_figures/2026-08-08_longhorizon_drift
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHUNK = np.arange(1, 9)

NOTTA = {
    "sharpness":    np.array([0.0070, 0.0088, 0.0108, 0.0138, 0.0168, 0.0200, 0.0223, 0.0251]),
    "colorfulness": np.array([0.1488, 0.1556, 0.1624, 0.1693, 0.1762, 0.1894, 0.2149, 0.2354]),
    "psnr":         np.array([19.0211, 14.8659, 12.5109, 11.6459, 11.1058, 10.5834, 10.1578, 9.8217]),
    "lpips":        np.array([0.2478, 0.4093, 0.5199, 0.5971, 0.6376, 0.6908, 0.7149, 0.7458]),
}
DELTA = {
    "sharpness":    np.array([0.0071, 0.0090, 0.0107, 0.0148, 0.0178, 0.0208, 0.0235, 0.0266]),
    "colorfulness": np.array([0.1485, 0.1548, 0.1609, 0.1728, 0.1816, 0.1919, 0.2023, 0.2190]),
    "psnr":         np.array([19.0226, 14.8228, 12.5132, 11.3673, 10.7918, 10.6018, 10.3143, 10.0582]),
    "lpips":        np.array([0.2484, 0.4148, 0.5309, 0.6133, 0.6475, 0.6885, 0.7081, 0.7389]),
}

CN, CD = "#2a6f97", "#d1495b"  # NOTTA blue, DELTA red


def _style():
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 12,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "legend.frameon": False,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="sweep_experiment/reports/paper_figures/2026-08-08_longhorizon_drift")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _style()

    panels = [
        ("colorfulness", "Colorfulness (lower = less over-saturation)"),
        ("sharpness",    "Sharpness / HF artifacts (lower = fewer)"),
        ("psnr",         "PSNR (dB, higher = better)"),
        ("lpips",        "LPIPS (lower = better)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (key, title) in zip(axes.ravel(), panels):
        ax.plot(CHUNK, NOTTA[key], "-o", color=CN, lw=2.6, ms=6, label="NOTTA")
        ax.plot(CHUNK, DELTA[key], "--s", color=CD, lw=2.6, ms=6, label="fixed delta")
        ax.set_title(title)
        ax.set_xlabel("chunk")
        ax.set_xticks(CHUNK)
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("A FIXED AdaSteer delta does not flatten long-horizon drift\n"
                 "curves stay parallel to NOTTA (N=24, reencode geometry, paired seeds) "
                 "-> motivates a streaming per-chunk delta",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(args.out_dir, "drift_intervention_notta_vs_delta.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
