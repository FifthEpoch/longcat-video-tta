#!/usr/bin/env python3
"""Native-vs-reencode geometry control figure.

Shows that most of the reencode-geometry "drift" is a short-window
re-conditioning artifact: at LongCat's native 13-cond/80-gen window the model is
far more robust EVEN over ~5.7x more generated frames.

Data (chunk 1 -> chunk 6 % change, matched at 6 chunks):
  reencode = job 15497180 (N=24, 14-cond/14-gen; 6 chunks = 84 gen frames)
  native   = diag_longhorizon_drift_notta_native checkpoint (N=12, 13-cond/80-gen;
             6 chunks = 480 gen frames). PRELIMINARY (arm timed out at 12/16 videos).

Run:
    python3 scripts/make_drift_geometry_control_fig.py \
        --out-dir sweep_experiment/reports/paper_figures/2026-08-08_longhorizon_drift
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABELS = ["Sharpness\n(HF artifacts)", "Colorfulness\n(saturation)", "Contrast",
          "Temporal\nmotion", "PSNR", "SSIM", "LPIPS"]
REENCODE = np.array([186.0, 27.3, 9.5, -9.1, -44.4, -50.6, 178.8])
NATIVE   = np.array([27.9, 4.1, 2.8, 8.2, -20.8, -39.6, 96.1])

CRE, CNAT = "#d1495b", "#2a6f97"


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
    fig, ax = plt.subplots(figsize=(11, 5.8))
    b1 = ax.bar(x - w/2, REENCODE, w, color=CRE,
                label="Reencode 14/14 window (N=24) — 6 chunks = 84 gen frames")
    b2 = ax.bar(x + w/2, NATIVE, w, color=CNAT,
                label="Native 13/80 window (N=12) — 6 chunks = 480 gen frames")
    ax.axhline(0, color="0.3", lw=1)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, fontsize=10)
    ax.set_ylabel("Drift over 6 chunks (% change, chunk 1 → 6)")
    ax.set_title("Native protocol drifts far less — most reencode 'drift' was a\n"
                 "short-window re-conditioning artifact (native covers 5.7x more frames)")
    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.annotate(f"{h:+.0f}%", (r.get_x()+r.get_width()/2, h),
                        textcoords="offset points",
                        xytext=(0, 4 if h >= 0 else -12),
                        ha="center", fontsize=8.5, fontweight="bold")
    ax.legend(loc="upper center", fontsize=9.5)
    fig.tight_layout()
    p = os.path.join(args.out_dir, "drift_geometry_control_native_vs_reencode.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
