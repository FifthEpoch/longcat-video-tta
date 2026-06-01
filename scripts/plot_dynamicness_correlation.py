#!/usr/bin/env python3
"""Plot per-method per-video metric performance vs video dynamicness.

For each {dataset, horizon} series this loads:
  1) per-video metrics (PSNR/SSIM/LPIPS) from each method's chunk results
  2) per-video dynamicness scores (RAFT mean flow) precomputed by
     scripts/compute_dynamic_degree.py

and produces a multi-panel figure where:
  - x-axis : video dynamicness quintile (or raw mean-flow if --no-bin)
  - y-axis : metric value (PSNR ↑, SSIM ↑, LPIPS ↓)
  - lines  : one per TTA method, all on the same axes

Optionally also produces an FVD-per-bin companion plot using the
per-chunk fvd_fid_stats.npz files if present.

Standard usage:
    python scripts/plot_dynamicness_correlation.py \
        --series-root sweep_experiment/results/panda_1000v_standard \
        --methods NOTTA ADA LORA_R8_TTA \
        --extra-method-root delta_experiment/results/tinylora_panda_1000v_standard \
        --extra-methods TL_BARE_R2 \
        --dynamic-degree-json datasets/panda_1000_480p/dynamic_degree.json \
        --output-png reports/figures/dynamicness_panda_1000v_std.png \
        --title "Panda-70M N=999, standard horizon"

Output: PNG figure + a sidecar JSON of binned numerical values for later
inclusion in the recap.
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


# ---------------------------------------------------------------------------
# Canonical video-id extraction
# ---------------------------------------------------------------------------
# Per-video records may carry method suffixes added by the runner, e.g.
# saved generated videos look like ``panda_0010_delta_a.mp4`` while the
# source dataset clips are just ``panda_0010.mp4``. We normalise both sides
# to ``panda_0010`` so the dynamicness scores (computed on source clips)
# can be joined with the per-video metric records (computed on generated
# clips) by video identity.
_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def _canonical_video_id(s: str) -> str:
    """Strip directory, extension and method suffixes, keeping ``<prefix>_<num>``."""
    if s is None:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


# ---------------------------------------------------------------------------
# Per-video metric loading
# ---------------------------------------------------------------------------
def load_per_video_metrics(method_dir: Path) -> Dict[str, Dict[str, float]]:
    """Return {canonical_video_id -> {psnr, ssim, lpips}}.

    Scans ``chunk_*/summary.json``; per-video records live under the
    ``results`` key as a list of dicts with at least ``video_name`` (or
    ``video_path``) and ``psnr/ssim/lpips``. Falls back to a flat
    ``summary.json`` if no chunk dirs exist.

    For backward compatibility, also tries ``chunk_*/results.json`` and the
    older nested ``results``-wrapper schema.
    """
    pv: Dict[str, Dict[str, float]] = {}
    candidates: List[Path] = sorted(method_dir.glob("chunk_*/summary.json"))
    if not candidates:
        candidates = sorted(method_dir.glob("chunk_*/results.json"))
    if not candidates:
        flat = method_dir / "summary.json"
        if flat.exists():
            candidates = [flat]

    for cf in candidates:
        try:
            d = json.load(open(cf))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {cf}: {e}", file=sys.stderr)
            continue
        items = d.get("results", d) if isinstance(d, dict) else d
        if not isinstance(items, list):
            continue
        for r in items:
            if not isinstance(r, dict):
                continue
            vid_raw = (r.get("video_name")
                       or r.get("video_id")
                       or r.get("video")
                       or r.get("video_path")
                       or r.get("path"))
            if vid_raw is None:
                continue
            vid = _canonical_video_id(str(vid_raw))
            if not vid:
                continue
            row = {
                "psnr":  r.get("psnr",  r.get("avg_psnr")),
                "ssim":  r.get("ssim",  r.get("avg_ssim")),
                "lpips": r.get("lpips", r.get("avg_lpips")),
            }
            pv[vid] = row
    return pv


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def quantile_bin_assign(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_index_per_sample, bin_edges). Edges include outer endpoints."""
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    # avoid degenerate edges with ties
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros_like(values, dtype=int), edges
    idx = np.clip(np.searchsorted(edges[1:-1], values, side="right"), 0, edges.size - 2)
    return idx, edges


def bin_means_sem(
    x: np.ndarray, y: np.ndarray, bin_idx: np.ndarray, n_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_y_mean, bin_y_sem, bin_count)."""
    centers = np.full(n_bins, np.nan)
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        xs, ys = x[mask], y[mask]
        ys = ys[~np.isnan(ys)]
        if ys.size == 0:
            continue
        centers[b] = float(np.mean(xs))
        means[b]   = float(np.mean(ys))
        sems[b]    = float(np.std(ys, ddof=1) / np.sqrt(ys.size)) if ys.size > 1 else 0.0
        counts[b]  = int(ys.size)
    return centers, means, sems, counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
METRIC_DEFS = [
    ("psnr",  "PSNR (dB)",   True),   # higher is better
    ("ssim",  "SSIM",        True),
    ("lpips", "LPIPS",       False),  # lower is better
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-root", required=True, type=Path,
                    help="Directory containing method subdirs "
                         "(e.g. sweep_experiment/results/panda_1000v_standard)")
    ap.add_argument("--methods", nargs="+", required=True,
                    help="Method subdir names under --series-root.")
    ap.add_argument("--extra-method-root", type=Path, default=None,
                    help="Optional second series root (e.g. for TinyLoRA "
                         "under delta_experiment/results/...)")
    ap.add_argument("--extra-methods", nargs="*", default=[],
                    help="Method subdir names under --extra-method-root.")
    ap.add_argument("--dynamic-degree-json", required=True, type=Path,
                    help="Output JSON from scripts/compute_dynamic_degree.py")
    ap.add_argument("--output-png", required=True, type=Path)
    ap.add_argument("--title", default="")
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--flow-key", default="mean_flow",
                    choices=["mean_flow", "max_flow"])
    ap.add_argument("--save-binned-json", type=Path, default=None,
                    help="Optional: save binned numerical values for tables.")
    args = ap.parse_args()

    # ---- load dynamicness scores ------------------------------------------
    dd = json.load(open(args.dynamic_degree_json))
    flow_by_vid: Dict[str, float] = {}
    for vid, info in dd["videos"].items():
        if "error" in info or info.get(args.flow_key) is None:
            continue
        flow_by_vid[_canonical_video_id(vid)] = float(info[args.flow_key])
    print(f"Loaded {len(flow_by_vid)} dynamicness scores "
          f"(key={args.flow_key}, model={dd.get('model')}).")

    # ---- load per-video metrics per method --------------------------------
    method_specs: List[Tuple[str, Path]] = [
        (m, args.series_root / m) for m in args.methods
    ]
    if args.extra_method_root:
        method_specs += [
            (m, args.extra_method_root / m) for m in args.extra_methods
        ]

    per_method_pv: Dict[str, Dict[str, Dict[str, float]]] = {}
    for name, mdir in method_specs:
        if not mdir.exists():
            print(f"[warn] {mdir} does not exist — skipping {name}",
                  file=sys.stderr)
            continue
        pv = load_per_video_metrics(mdir)
        if not pv:
            print(f"[warn] no per-video records under {mdir}", file=sys.stderr)
            continue
        per_method_pv[name] = pv
        print(f"  {name:18s} per-video records: {len(pv)}  (from {mdir})")

    if not per_method_pv:
        print("[error] no methods loaded — abort.", file=sys.stderr)
        return 2

    # ---- intersect: only videos present in (all methods) AND flow ---------
    common = set(flow_by_vid.keys())
    for name, pv in per_method_pv.items():
        common &= set(pv.keys())
    common = sorted(common)
    print(f"\nCommon videos across all methods + flow scores: {len(common)}")
    if len(common) < args.n_bins * 5:
        print(f"[warn] only {len(common)} common videos for "
              f"{args.n_bins} bins — bins will be sparse.", file=sys.stderr)

    flows = np.array([flow_by_vid[v] for v in common], dtype=float)
    bin_idx, edges = quantile_bin_assign(flows, args.n_bins)
    n_bins_eff = max(int(bin_idx.max()) + 1, 1)
    print(f"Bin edges ({args.flow_key}): "
          f"{', '.join(f'{e:.3f}' for e in edges)}")

    # ---- plot --------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(METRIC_DEFS), figsize=(15, 4.2),
                             sharex=True)
    if len(METRIC_DEFS) == 1:
        axes = [axes]

    binned_record: Dict[str, Dict] = {}

    for ax, (key, label, higher_better) in zip(axes, METRIC_DEFS):
        for name, pv in per_method_pv.items():
            y = np.array([pv[v][key] if pv[v][key] is not None else np.nan
                          for v in common], dtype=float)
            centers, means, sems, counts = bin_means_sem(
                flows, y, bin_idx, n_bins_eff
            )
            valid = ~np.isnan(means)
            ax.errorbar(centers[valid], means[valid], yerr=sems[valid],
                        marker="o", label=name, linewidth=1.6,
                        capsize=2, alpha=0.95)
            binned_record.setdefault(key, {})[name] = {
                "bin_centers": centers.tolist(),
                "bin_means":   means.tolist(),
                "bin_sems":    sems.tolist(),
                "bin_counts":  counts.tolist(),
            }

        ax.set_xlabel(f"Video dynamicness "
                      f"({args.flow_key.replace('_', ' ')}, RAFT)")
        ax.set_ylabel(label + (" ↑" if higher_better else " ↓"))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    if args.title:
        fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if args.title else None)

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=160, bbox_inches="tight")
    print(f"\nWrote figure: {args.output_png}")

    # ---- save binned values for tables ------------------------------------
    if args.save_binned_json is None:
        args.save_binned_json = args.output_png.with_suffix(".binned.json")
    args.save_binned_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.save_binned_json, "w") as f:
        json.dump({
            "title": args.title,
            "flow_key": args.flow_key,
            "n_bins": n_bins_eff,
            "edges": edges.tolist(),
            "n_common_videos": len(common),
            "methods": list(per_method_pv.keys()),
            "binned": binned_record,
        }, f, indent=2)
    print(f"Wrote binned JSON: {args.save_binned_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
