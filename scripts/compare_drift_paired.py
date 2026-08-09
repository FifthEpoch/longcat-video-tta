#!/usr/bin/env python3
"""Paired NOTTA-vs-intervention comparison for long-horizon drift.

The merged verdicts only give population endpoint means; this answers the real
question -- "does the intervention reduce per-video drift beyond N=8 noise?" --
with the same rigor we used to rule out the PSNR router:

  * pair videos by name across the two merged_summary.json files (same seed/pool
    => same videos),
  * per video + per signal compute drift = |last_finite - chunk1| (deviation from
    the chunk-1 baseline; sign-agnostic so it works for rising sharpness AND
    falling contrast),
  * report mean drift per arm, the paired reduction (NOTTA - intervention),
    a bootstrap 95% CI over videos, and a sign-flip permutation p-value.

Also writes a per-chunk overlay figure (NOTTA vs intervention) for the headline
GT-free signals.

Usage:
  python scripts/compare_drift_paired.py \
    --notta sweep_experiment/results/longhorizon_sweep_notta_native_12ch/merged_summary.json \
    --delta sweep_experiment/results/longhorizon_sweep_delta_stream_native_12ch/merged_summary.json \
    --out-dir sweep_experiment/results/longhorizon_sweep_delta_stream_native_12ch/paired
"""
import argparse
import json
import os

import numpy as np

GEN_FREE = ["sharpness", "temporal_motion", "colorfulness", "contrast"]
GT = ["psnr", "ssim", "lpips"]
SIGNALS = GEN_FREE + GT
FIG_SIGNALS = ["sharpness", "colorfulness", "contrast", "temporal_motion"]
FIG_TITLES = {
    "sharpness": "Sharpness / HF artifacts (lower drift better)",
    "colorfulness": "Colorfulness / saturation (lower drift better)",
    "contrast": "Contrast (less fade better)",
    "temporal_motion": "Temporal motion (less inflation better)",
}


def _load(path):
    with open(path) as f:
        return json.load(f)


def _series(chunks, key):
    """Ordered (chunk_idx, value) for finite values of `key`."""
    out = []
    for ch in chunks:
        v = ch.get(key)
        if isinstance(v, (int, float)) and v == v:
            out.append((ch["chunk"], float(v)))
    out.sort()
    return out


def per_video_drift(summary, key):
    """video_name -> |last_finite - first_finite| for `key` (None if <2 points)."""
    d = {}
    for r in summary.get("results", []):
        if not r.get("success"):
            continue
        s = _series(r.get("chunks", []), key)
        if len(s) >= 2:
            d[r["video_name"]] = abs(s[-1][1] - s[0][1])
    return d


def boot_ci(x, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def signflip_p(diffs, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(float(np.mean(diffs)))
    signs = rng.choice([-1.0, 1.0], size=(n, len(diffs)))
    perm = np.abs((signs * diffs[None, :]).mean(axis=1))
    return float((perm >= obs).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notta", required=True, help="baseline merged_summary.json")
    ap.add_argument("--delta", required=True, help="intervention merged_summary.json")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--label-a", default="NOTTA")
    ap.add_argument("--label-b", default="stream-delta")
    args = ap.parse_args()

    A, B = _load(args.notta), _load(args.delta)
    out_dir = args.out_dir or os.path.dirname(args.delta)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Paired drift comparison: {args.label_a} vs {args.label_b}")
    print(f"  A: {args.notta}")
    print(f"  B: {args.delta}\n")
    print(f"{'signal':16s} {'n':>3s} {'|drift|_A':>10s} {'|drift|_B':>10s} "
          f"{'reduction':>10s} {'95% CI':>22s} {'p(signflip)':>12s}")
    print("-" * 90)

    rows = []
    for key in SIGNALS:
        da, db = per_video_drift(A, key), per_video_drift(B, key)
        common = sorted(set(da) & set(db))
        if len(common) < 2:
            print(f"{key:16s} {len(common):>3d}  (too few paired videos)")
            continue
        a = np.array([da[v] for v in common], float)
        b = np.array([db[v] for v in common], float)
        red = a - b                      # >0 => intervention reduces drift
        mred = float(np.mean(red))
        lo, hi = boot_ci(red)
        p = signflip_p(red)
        flag = "  *" if (lo > 0 or hi < 0) else ""
        print(f"{key:16s} {len(common):>3d} {a.mean():>10.4f} {b.mean():>10.4f} "
              f"{mred:>+10.4f} [{lo:>+8.4f},{hi:>+8.4f}] {p:>12.4f}{flag}")
        rows.append({
            "signal": key, "n_paired": len(common),
            "mean_abs_drift_A": a.mean(), "mean_abs_drift_B": b.mean(),
            "mean_reduction_A_minus_B": mred, "ci95": [lo, hi],
            "signflip_p": p, "gt_free": key in GEN_FREE,
        })

    print("\n  reduction > 0 => intervention shrinks per-video drift.")
    print("  * = bootstrap 95% CI excludes 0. Judge GT-free signals; GT metrics")
    print("    (psnr/ssim/lpips) span only ~1-2 chunks here (GT runs out).")

    with open(os.path.join(out_dir, "paired_stats.json"), "w") as f:
        json.dump({"label_a": args.label_a, "label_b": args.label_b,
                   "notta": args.notta, "delta": args.delta, "rows": rows}, f, indent=2)

    # ---- overlay figure (per-chunk means) --------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ca, cb = A.get("drift_curves", {}), B.get("drift_curves", {})
        nchunks = int(A.get("num_chunks", 12))
        x = np.arange(1, nchunks + 1)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, key in zip(axes.ravel(), FIG_SIGNALS):
            ya = ca.get(key, {}).get("mean", [])
            yb = cb.get(key, {}).get("mean", [])
            ax.plot(x[:len(ya)], [np.nan if v is None else v for v in ya],
                    "-o", color="#2a6f97", lw=2.4, ms=5, label=args.label_a)
            ax.plot(x[:len(yb)], [np.nan if v is None else v for v in yb],
                    "--s", color="#d1495b", lw=2.4, ms=5, label=args.label_b)
            ax.set_title(FIG_TITLES.get(key, key))
            ax.set_xlabel("autoregressive chunk")
            ax.set_xticks(x)
        axes[0, 0].legend(loc="upper left")
        fig.suptitle(f"Streaming per-chunk delta vs NOTTA (native ~60s, N={rows[0]['n_paired']})\n"
                     "anchored re-fit flattens HF-artifact + saturation drift",
                     fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        p = os.path.join(out_dir, "paired_notta_vs_streamdelta.png")
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"\nwrote {p}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(figure skipped: {e})")

    print(f"saved paired_stats.json -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
