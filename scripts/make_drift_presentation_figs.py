#!/usr/bin/env python3
"""Presentation figures for the long-horizon drift finding (job 15497180).

Standalone: the per-chunk numbers are embedded verbatim from
``sweep_experiment/reports/experiment_outputs/2026-08-07.md`` (NOTTA reencode
rollout, N=24 videos x 8 chunks, cond=14 / frames=28 / gen_start=48). Run:

    python3 scripts/make_drift_presentation_figs.py \
        --out-dir sweep_experiment/reports/paper_figures/2026-08-08_longhorizon_drift

Produces three slide-ready PNGs (300 dpi):
  1. drift_gtfree_normalized.png  -- headline: GT-free signals, % of chunk 1
  2. drift_gtfree_raw.png         -- GT-free signals in raw units (2x2)
  3. drift_gt_fidelity.png        -- PSNR + LPIPS collapse (with GT coverage)
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- embedded data (job 15497180, chunks 1..8) ----------------------------
CHUNK = np.array([1, 2, 3, 4, 5, 6, 7, 8])
SHARPNESS = np.array([0.0070, 0.0088, 0.0108, 0.0138, 0.0168, 0.0200, 0.0223, 0.0251])
MOTION    = np.array([0.0232, 0.0218, 0.0238, 0.0210, 0.0228, 0.0211, 0.0202, 0.0204])
COLORFUL  = np.array([0.1488, 0.1556, 0.1624, 0.1693, 0.1762, 0.1894, 0.2149, 0.2354])
CONTRAST  = np.array([0.2358, 0.2402, 0.2465, 0.2515, 0.2563, 0.2583, 0.2629, 0.2674])
PSNR      = np.array([19.02, 14.87, 12.51, 11.65, 11.11, 10.58, 10.16, 9.82])
SSIM      = np.array([0.710, 0.587, 0.505, 0.432, 0.390, 0.351, 0.321, 0.311])
LPIPS     = np.array([0.248, 0.409, 0.520, 0.597, 0.638, 0.691, 0.715, 0.746])
PSNR_N    = np.array([24, 24, 19, 18, 16, 15, 14, 13])  # videos with GT still overlapping

N_VIDEOS = 24

C = {
    "colorful": "#d1495b",   # red  – over-saturation
    "contrast": "#edae49",   # amber
    "sharp":    "#9b5de5",   # purple – HF artifacts
    "motion":   "#00a6a6",   # teal – motion (control, flat)
    "psnr":     "#2a6f97",   # blue
    "lpips":    "#d1495b",   # red
}


def _style():
    plt.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
    })


def _pct(arr):
    return arr / arr[0] * 100.0


def fig_normalized(out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    series = [
        ("Colorfulness (over-saturation)", COLORFUL, C["colorful"], "o"),
        ("Sharpness / HF-artifacts (Laplacian var)", SHARPNESS, C["sharp"], "s"),
        ("Contrast (luma std)", CONTRAST, C["contrast"], "^"),
        ("Temporal motion (control)", MOTION, C["motion"], "D"),
    ]
    for label, arr, col, mk in series:
        y = _pct(arr)
        ax.plot(CHUNK, y, marker=mk, color=col, lw=2.6, ms=7, label=label)
        ax.annotate(f"{y[-1]-100:+.0f}%", (CHUNK[-1], y[-1]),
                    textcoords="offset points", xytext=(8, 0),
                    color=col, fontsize=12, fontweight="bold", va="center")
    ax.axhline(100, color="0.4", lw=1, ls=":")
    ax.set_xlabel("Autoregressive chunk index")
    ax.set_ylabel("Signal, % of chunk 1")
    ax.set_title("LongCat drifts monotonically over an 8-chunk rollout\n"
                 f"GT-free quality signals (N={N_VIDEOS}, NOTTA)")
    ax.set_xticks(CHUNK)
    ax.set_xlim(0.8, 8.9)
    ax.legend(loc="upper left")
    fig.tight_layout()
    p = os.path.join(out, "drift_gtfree_normalized.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_raw(out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (axes[0, 0], "Colorfulness", COLORFUL, C["colorful"], "o", "+58% (monotone)"),
        (axes[0, 1], "Sharpness (Laplacian var)", SHARPNESS, C["sharp"], "s", "+258% (monotone)"),
        (axes[1, 0], "Contrast (luma std)", CONTRAST, C["contrast"], "^", "+13% (monotone)"),
        (axes[1, 1], "Temporal motion", MOTION, C["motion"], "D", "-12% (flat/noisy)"),
    ]
    for ax, title, arr, col, mk, tag in panels:
        ax.plot(CHUNK, arr, marker=mk, color=col, lw=2.6, ms=7)
        ax.set_title(f"{title}   [{tag}]", fontsize=13)
        ax.set_xlabel("chunk")
        ax.set_xticks(CHUNK)
    fig.suptitle("GT-free drift signals in raw units — over-saturation + HF-artifact "
                 "accumulation;\nmotion stays flat (drift mode is NOT motion collapse)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out, "drift_gtfree_raw.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_fidelity(out):
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax1.plot(CHUNK, PSNR, marker="o", color=C["psnr"], lw=2.8, ms=7, label="PSNR (dB)")
    ax1.set_xlabel("Autoregressive chunk index")
    ax1.set_ylabel("PSNR (dB)", color=C["psnr"])
    ax1.tick_params(axis="y", labelcolor=C["psnr"])
    ax1.set_xticks(CHUNK)
    ax1.set_ylim(8, 20)

    ax2 = ax1.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.plot(CHUNK, LPIPS, marker="s", color=C["lpips"], lw=2.8, ms=7, label="LPIPS")
    ax2.set_ylabel("LPIPS (lower = better)", color=C["lpips"])
    ax2.tick_params(axis="y", labelcolor=C["lpips"])
    ax2.set_ylim(0.2, 0.8)
    ax2.grid(False)

    # GT coverage annotation (videos whose GT still overlaps the rollout)
    for x, n in zip(CHUNK, PSNR_N):
        ax1.annotate(f"n={n}", (x, PSNR[np.where(CHUNK == x)[0][0]]),
                     textcoords="offset points", xytext=(0, 10),
                     ha="center", fontsize=8, color="0.45")

    ax1.set_title("GT-referenced fidelity collapses over the rollout\n"
                  "PSNR 19.0 -> 9.8 dB (-48%), LPIPS +201%  (GT coverage n falls 24 -> 13)")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right")
    fig.tight_layout()
    p = os.path.join(out, "drift_gt_fidelity.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="sweep_experiment/reports/paper_figures/2026-08-08_longhorizon_drift")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    _style()
    for fn in (fig_normalized, fig_raw, fig_fidelity):
        print("wrote", fn(args.out_dir))


if __name__ == "__main__":
    main()
