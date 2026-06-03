#!/usr/bin/env python3
"""Smoking-gun overfitting test: correlate per-video TTA update magnitude
with video dynamicness AND with ΔPSNR vs No-TTA.

The hypothesis (from partner review, 2026-06-01):
    "TTA is overfitting; high-motion videos are harder, the model takes
     bigger steps trying to fit them, and those bigger steps don't help."

This script tests it directly. For each TTA method we extract the
``delta_norm`` field from ``chunk_*/summary.json["results"]`` (already saved
by the runner — represents the L2 norm of the trained adapter parameters
after TTA), then plot:

    Panel A:  delta_norm vs video dynamicness
              "Are bigger updates triggered by more dynamic content?"

    Panel B:  ΔPSNR (method - NoTTA) vs delta_norm
              "Do bigger updates help, hurt, or neither?"

    Panel C:  ΔPSNR vs dynamicness, scatter coloured by delta_norm
              The 2D version: shows whether the high-motion / large-update
              corner of the joint distribution is the negative-Δ region.

A strict overfitting story would predict:
    - Panel A: positive correlation (more motion -> more update)
    - Panel B: negative correlation (more update -> worse Δ)
    - Panel C: high-motion + large-update points cluster in negative Δ

A null story (the saturation finding from this week's data) would predict:
    - Panel A: low correlation (delta_norm is roughly content-independent)
    - Panel B: zero correlation (all the variance in Δ is noise, unrelated
      to update size)
    - Panel C: no spatial structure
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def _canonical_video_id(s: str) -> str:
    if s is None:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


def load_per_video_records(method_dir: Path) -> Dict[str, Dict]:
    """Return {canonical_video_id -> raw record dict} from chunk_*/summary.json["results"]."""
    pv: Dict[str, Dict] = {}
    for cf in sorted(method_dir.glob("chunk_*/summary.json")):
        try:
            d = json.load(open(cf))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {cf}: {e}", file=sys.stderr)
            continue
        items = d.get("results", []) if isinstance(d, dict) else []
        if not isinstance(items, list):
            continue
        for r in items:
            if not isinstance(r, dict):
                continue
            vid_raw = (r.get("video_name")
                       or r.get("video_id")
                       or r.get("video_path"))
            if vid_raw is None:
                continue
            vid = _canonical_video_id(str(vid_raw))
            if vid:
                pv[vid] = r
    return pv


def _safe_corr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    """Return (pearson_r, spearman_rho, n_used) on overlap-finite samples."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan"), float("nan"), int(mask.sum())
    xs, ys = x[mask], y[mask]
    # Pearson
    pearson = float(np.corrcoef(xs, ys)[0, 1])
    # Spearman via ranks
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return pearson, spearman, int(mask.sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-root", required=True, type=Path)
    ap.add_argument("--baseline-method", default="NOTTA")
    ap.add_argument("--tta-methods", nargs="+", required=True,
                    help="TTA methods to analyse (must have non-zero delta_norm).")
    ap.add_argument("--extra-method-root", type=Path, default=None)
    ap.add_argument("--extra-tta-methods", nargs="*", default=[])
    ap.add_argument("--dynamic-degree-json", required=True, type=Path)
    ap.add_argument("--flow-key", default="mean_flow",
                    choices=["mean_flow", "max_flow"])
    ap.add_argument("--output-png", required=True, type=Path)
    ap.add_argument("--title", default="")
    ap.add_argument("--save-stats-json", type=Path, default=None)
    ap.add_argument("--delta-norm-key", default="delta_norm",
                    help="Field name in per-video records. Default: delta_norm.")
    args = ap.parse_args()

    # ---- load dynamicness scores ------------------------------------------
    dd = json.load(open(args.dynamic_degree_json))
    flow_by_vid: Dict[str, float] = {}
    for vid, info in dd["videos"].items():
        if "error" in info or info.get(args.flow_key) is None:
            continue
        flow_by_vid[_canonical_video_id(vid)] = float(info[args.flow_key])
    print(f"Loaded {len(flow_by_vid)} dynamicness scores.")

    # ---- baseline records --------------------------------------------------
    baseline_dir = args.series_root / args.baseline_method
    baseline_pv = load_per_video_records(baseline_dir)
    if not baseline_pv:
        print(f"[error] no records under {baseline_dir}", file=sys.stderr)
        return 2
    print(f"Baseline {args.baseline_method}: {len(baseline_pv)} records "
          f"(from {baseline_dir})")

    # ---- TTA records -------------------------------------------------------
    tta_specs: List[Tuple[str, Path]] = [
        (m, args.series_root / m) for m in args.tta_methods
    ]
    if args.extra_method_root:
        tta_specs += [
            (m, args.extra_method_root / m) for m in args.extra_tta_methods
        ]

    tta_pv: Dict[str, Dict[str, Dict]] = {}
    for name, mdir in tta_specs:
        if not mdir.exists():
            print(f"[warn] {mdir} missing — skipping {name}", file=sys.stderr)
            continue
        recs = load_per_video_records(mdir)
        if not recs:
            print(f"[warn] no records under {mdir}", file=sys.stderr)
            continue
        # Quick check: any non-zero delta_norm?
        norms = [r.get(args.delta_norm_key) for r in recs.values()]
        norms = [float(n) for n in norms if n is not None]
        if not norms:
            print(f"[warn] {name} has no '{args.delta_norm_key}' field — "
                  "skipping. Available keys (first record): "
                  f"{sorted(next(iter(recs.values())).keys())[:30]}",
                  file=sys.stderr)
            continue
        nz = sum(1 for n in norms if n > 1e-12)
        print(f"  {name:20s} records={len(recs)}  delta_norms: "
              f"min={min(norms):.4g} mean={np.mean(norms):.4g} "
              f"max={max(norms):.4g}  nonzero={nz}/{len(norms)}")
        if nz == 0:
            print(f"[warn] {name} has all-zero delta_norm — TTA effectively "
                  "no-op? Skipping panels for this method.", file=sys.stderr)
            continue
        tta_pv[name] = recs

    if not tta_pv:
        print("[error] no TTA methods with usable delta_norm — abort.",
              file=sys.stderr)
        return 2

    # ---- intersect: only videos shared across (baseline + all TTA + flow) -
    common = set(flow_by_vid.keys()) & set(baseline_pv.keys())
    for name, recs in tta_pv.items():
        common &= set(recs.keys())
    common = sorted(common)
    print(f"\nCommon videos for analysis: {len(common)}")
    if len(common) < 50:
        print("[warn] fewer than 50 common videos — correlations will be noisy.",
              file=sys.stderr)

    flows = np.array([flow_by_vid[v] for v in common], dtype=float)

    # ---- plot --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_methods = len(tta_pv)
    fig = plt.figure(figsize=(15, 4.2 * n_methods))

    # one row per method, three columns
    gs = fig.add_gridspec(n_methods, 3, hspace=0.45, wspace=0.30)
    stats_record: Dict[str, Dict] = {}

    for row, (name, recs) in enumerate(tta_pv.items()):
        delta_norm = np.array(
            [float(recs[v].get(args.delta_norm_key))
             if recs[v].get(args.delta_norm_key) is not None else np.nan
             for v in common], dtype=float,
        )
        psnr_method = np.array(
            [float(recs[v]["psnr"]) if recs[v].get("psnr") is not None else np.nan
             for v in common], dtype=float,
        )
        psnr_baseline = np.array(
            [float(baseline_pv[v]["psnr"])
             if baseline_pv[v].get("psnr") is not None else np.nan
             for v in common], dtype=float,
        )
        delta_psnr = psnr_method - psnr_baseline

        # ---- Panel A: delta_norm vs flow ---------------------------------
        axA = fig.add_subplot(gs[row, 0])
        axA.scatter(flows, delta_norm, s=8, alpha=0.4, edgecolors="none")
        rA, sA, nA = _safe_corr(flows, delta_norm)
        axA.set_xlabel(f"Video dynamicness ({args.flow_key}, RAFT)")
        axA.set_ylabel(f"{name}: {args.delta_norm_key}")
        axA.set_xscale("symlog", linthresh=0.1)
        axA.set_title(f"A. update size vs motion\n"
                      f"Pearson={rA:+.3f}  Spearman={sA:+.3f}  n={nA}",
                      fontsize=10)
        axA.grid(True, alpha=0.3)

        # ---- Panel B: ΔPSNR vs delta_norm --------------------------------
        axB = fig.add_subplot(gs[row, 1])
        axB.scatter(delta_norm, delta_psnr, s=8, alpha=0.4, edgecolors="none")
        axB.axhline(0.0, color="grey", lw=0.8, alpha=0.6)
        rB, sB, nB = _safe_corr(delta_norm, delta_psnr)
        axB.set_xlabel(f"{name}: {args.delta_norm_key}")
        axB.set_ylabel(f"Δ PSNR ({name} − {args.baseline_method}) [dB]")
        axB.set_title(f"B. effect of update on Δ\n"
                      f"Pearson={rB:+.3f}  Spearman={sB:+.3f}  n={nB}",
                      fontsize=10)
        axB.grid(True, alpha=0.3)

        # ---- Panel C: 2D scatter, ΔPSNR colored ------------------------
        axC = fig.add_subplot(gs[row, 2])
        # color by ΔPSNR sign-magnitude
        sc = axC.scatter(flows, delta_norm, c=delta_psnr, s=10, alpha=0.7,
                         cmap="RdBu_r",
                         vmin=-np.nanpercentile(np.abs(delta_psnr), 95),
                         vmax=+np.nanpercentile(np.abs(delta_psnr), 95),
                         edgecolors="none")
        axC.set_xscale("symlog", linthresh=0.1)
        axC.set_xlabel(f"Video dynamicness ({args.flow_key}, RAFT)")
        axC.set_ylabel(f"{name}: {args.delta_norm_key}")
        rC, sC, nC = _safe_corr(flows + 0.001 * delta_norm,
                                delta_psnr)  # joint loose
        axC.set_title(f"C. joint (color = Δ PSNR)\n"
                      f"red = method better, blue = baseline better",
                      fontsize=10)
        axC.grid(True, alpha=0.3)
        cb = fig.colorbar(sc, ax=axC, fraction=0.05, pad=0.04)
        cb.set_label("Δ PSNR (dB)", fontsize=8)

        stats_record[name] = {
            "n": len(common),
            "panel_A_flow_vs_delta_norm":      {"pearson": rA, "spearman": sA, "n": nA},
            "panel_B_delta_norm_vs_dPSNR":     {"pearson": rB, "spearman": sB, "n": nB},
            "delta_norm_quartiles": np.nanquantile(delta_norm, [0, .25, .5, .75, 1]).tolist(),
            "delta_psnr_quartiles": np.nanquantile(delta_psnr, [0, .25, .5, .75, 1]).tolist(),
            "delta_psnr_mean": float(np.nanmean(delta_psnr)),
            "delta_psnr_sem":  float(np.nanstd(delta_psnr, ddof=1) / np.sqrt(np.isfinite(delta_psnr).sum())),
        }

    if args.title:
        fig.suptitle(args.title, fontsize=12, y=1.00)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if args.title else None)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160, bbox_inches="tight")
    print(f"\nWrote figure: {args.output_png}")

    # ---- save stats --------------------------------------------------------
    if args.save_stats_json is None:
        args.save_stats_json = args.output_png.with_suffix(".stats.json")
    with open(args.save_stats_json, "w") as f:
        json.dump({
            "title": args.title,
            "flow_key": args.flow_key,
            "baseline_method": args.baseline_method,
            "tta_methods": list(tta_pv.keys()),
            "n_common_videos": len(common),
            "stats": stats_record,
        }, f, indent=2)
    print(f"Wrote stats JSON: {args.save_stats_json}")

    # ---- console summary ---------------------------------------------------
    print("\n" + "=" * 70)
    print("OVERFITTING TEST — interpretation guide")
    print("=" * 70)
    print(f"For each method, A>0 + B<0 (Spearman) supports overfitting.")
    print(f"Magnitudes |r|<0.05 = effectively zero correlation.\n")
    for name, st in stats_record.items():
        rA = st["panel_A_flow_vs_delta_norm"]["spearman"]
        rB = st["panel_B_delta_norm_vs_dPSNR"]["spearman"]
        verdict = "?"
        if abs(rA) < 0.05 and abs(rB) < 0.05:
            verdict = "NULL — neither motion nor update size predicts Δ"
        elif rA > 0.05 and rB < -0.05:
            verdict = "OVERFITTING signature (motion → big update → worse)"
        elif rA > 0.05 and rB > 0.05:
            verdict = "POSITIVE — motion → big update → BETTER"
        elif abs(rA) < 0.05 and rB < -0.05:
            verdict = "Update size predicts harm — but not driven by motion"
        else:
            verdict = "MIXED"
        print(f"  {name:20s}  ρ(flow, Δnorm)={rA:+.3f}  "
              f"ρ(Δnorm, ΔPSNR)={rB:+.3f}  →  {verdict}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
