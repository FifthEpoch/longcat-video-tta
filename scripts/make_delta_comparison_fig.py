#!/usr/bin/env python3
"""Slide 4 figure: delta recipes vs No-TTA at native 60s (N=8, paired).

Two panels that together explain the null:
  A. POPULATION endpoint drift (chunk1->last, % change) per GT-free signal for
     NOTTA vs streaming-generated (EXP4) vs streaming-clean. The deltas LOOK like
     they reduce drift here.
  B. PER-VIDEO paired |drift| reduction vs NOTTA (positive = less drift than
     NOTTA = better) with 95% bootstrap CIs. EVERY CI crosses 0 -> null. The
     population "flattening" in A was cancellation of opposite per-video effects.

Only the two recipes actually RUN at native 60s (12 autoregressive chunks) are
shown. Fixed (EXP-B) ran at the earlier reencode geometry (different protocol);
the time-scheduled ramp was contraindicated by the chunk-interaction gate and
never run.

Data:
  population endpoints -- experiment_outputs/2026-08-09.md (NOTTA, gen),
    2026-08-10.md (clean).
  paired |drift| reduction + 95% CI + p -- compare_drift_paired.py:
    .../longhorizon_sweep_delta_stream_native_12ch/paired  (gen)
    .../longhorizon_sweep_delta_stream_clean_native_12ch/paired  (clean)

Run:
    python3 scripts/make_delta_comparison_fig.py \
        --out-dir sweep_experiment/reports/paper_figures/2026-08-08_longhorizon_drift
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGNALS = ["Sharpness", "Temporal\nmotion", "Colorfulness", "Contrast"]

# --- Panel A: population endpoint drift (chunk1 -> last, % change), native 60s N=8
POP_NOTTA = np.array([48.0, 45.1, 5.7, -16.4])
POP_GEN = np.array([24.8, 40.8, 0.4, -11.5])
POP_CLEAN = np.array([34.7, 5.7, -8.5, -20.9])

# --- Panel B: per-video paired |drift| reduction vs NOTTA (positive = beats NOTTA)
GEN_RED = np.array([-0.0015, 0.0008, -0.0078, -0.0029])
GEN_LO = np.array([-0.0038, -0.0061, -0.0199, -0.0148])
GEN_HI = np.array([0.0007, 0.0074, 0.0051, 0.0081])
GEN_P = [0.26, 0.88, 0.32, 0.66]

CLEAN_RED = np.array([-0.0014, 0.0010, 0.0015, -0.0177])
CLEAN_LO = np.array([-0.0051, -0.0077, -0.0107, -0.0659])
CLEAN_HI = np.array([0.0028, 0.0087, 0.0158, 0.0164])
CLEAN_P = [0.53, 0.83, 0.84, 0.70]

C_NOTTA, C_GEN, C_CLEAN = "#6c757d", "#e07a5f", "#3d5a80"


def _yerr(red, lo, hi):
    return np.vstack([red - lo, hi - red])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="sweep_experiment/reports/paper_figures/2026-08-08_longhorizon_drift")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 300, "savefig.dpi": 300, "font.size": 11,
        "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "legend.frameon": False,
    })
    x = np.arange(len(SIGNALS))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.8))

    # Panel A: population endpoint drift, 3 arms
    w = 0.26
    axA.bar(x - w, POP_NOTTA, w, color=C_NOTTA, label="No-TTA")
    axA.bar(x, POP_GEN, w, color=C_GEN, label="Streaming-generated (EXP4)")
    axA.bar(x + w, POP_CLEAN, w, color=C_CLEAN, label="Streaming-clean")
    axA.axhline(0, color="0.3", lw=1)
    axA.set_xticks(x); axA.set_xticklabels(SIGNALS)
    axA.set_ylabel("Population endpoint drift (%, chunk 1 → last)")
    axA.set_title("A. Population means: deltas LOOK like they reduce drift")
    for arr, off in [(POP_NOTTA, -w), (POP_GEN, 0.0), (POP_CLEAN, w)]:
        for xi, v in zip(x, arr):
            axA.annotate(f"{v:+.0f}", (xi + off, v), textcoords="offset points",
                         xytext=(0, 3 if v >= 0 else -11), ha="center",
                         fontsize=8, fontweight="bold")
    axA.legend(loc="upper right", fontsize=9)

    # Panel B: per-video paired |drift| reduction vs NOTTA, 2 arms, with CI
    w2 = 0.34
    axB.bar(x - w2 / 2, GEN_RED, w2, yerr=_yerr(GEN_RED, GEN_LO, GEN_HI),
            color=C_GEN, capsize=4, error_kw={"lw": 1.4, "ecolor": "0.25"},
            label="Streaming-generated (EXP4)")
    axB.bar(x + w2 / 2, CLEAN_RED, w2, yerr=_yerr(CLEAN_RED, CLEAN_LO, CLEAN_HI),
            color=C_CLEAN, capsize=4, error_kw={"lw": 1.4, "ecolor": "0.25"},
            label="Streaming-clean")
    axB.axhline(0, color="#c1121f", lw=1.6)
    axB.annotate("No-TTA (0 = no better than baseline)", (len(SIGNALS) - 0.5, 0),
                 textcoords="offset points", xytext=(0, 4), ha="right",
                 fontsize=8.5, color="#c1121f", fontweight="bold")
    axB.set_xticks(x); axB.set_xticklabels(SIGNALS)
    axB.set_ylim(-0.078, 0.024)
    axB.set_ylabel("Paired |drift| reduction vs No-TTA\n(positive = less drift, better)")
    axB.set_title("B. Per-video paired test: every 95% CI crosses 0 → NULL")
    for xi, v, lo, p in zip(x - w2 / 2, GEN_RED, GEN_LO, GEN_P):
        axB.annotate(f"p={p:.2f}", (xi, lo), textcoords="offset points",
                     xytext=(0, -13), ha="center", fontsize=7.5, color="0.3")
    for xi, v, lo, p in zip(x + w2 / 2, CLEAN_RED, CLEAN_LO, CLEAN_P):
        axB.annotate(f"p={p:.2f}", (xi, lo), textcoords="offset points",
                     xytext=(0, -13), ha="center", fontsize=7.5, color="0.3")
    axB.legend(loc="lower left", fontsize=9)

    fig.suptitle("Delta recipes vs No-TTA at native 60 s (12 autoregressive chunks, N=8, paired)\n"
                 "Fixed (EXP-B) ran at earlier reencode geometry (not shown); time-scheduled ramp was "
                 "contraindicated and never run",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    p = os.path.join(args.out_dir, "delta_recipes_vs_notta_native60s.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
