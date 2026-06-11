#!/usr/bin/env python3
"""Per-chunk ΔFVD sign analysis across TTA methods (the deferred TODO).

Background
----------
Each TTA series on the cluster is laid out as

    <series_path>/<METHOD>/chunk_0/summary.json
    <series_path>/<METHOD>/chunk_1/summary.json
    ...
    <series_path>/<METHOD>/chunk_9/summary.json
    <series_path>/<METHOD>/merged_summary.json

``merge_chunks.py`` populates ``merged_summary.json['fvd_per_chunk']``
with the 10 per-chunk FVD values (computed correctly per-chunk by
``OnlineFrechetAccumulator.compute`` then written into the chunk's
``summary.json['fvd']``). The 1000-video global FVD is also recomputed
from the merged sufficient statistics (``fvd_fid_stats.npz``) and lands
under ``merged_summary.json['fvd']``.

The deferred TODO from the 2026-06-09 prompt-vs-NOPROMPT work is the
per-chunk ΔFVD sign analysis: for each (method ≠ NOTTA), how many of
the 10 chunks have lower FVD than the NOTTA baseline's same chunk? At
N=100 videos per chunk, the per-chunk FVD has sample-size variance —
the SIGN-COUNT across 10 chunks is a non-parametric significance test
for a per-method FVD effect that is invisible at the population level.

Specifically this script computes, for every (series, method ≠ baseline):

  • per-chunk ΔFVD_c = FVD_method(chunk_c) − FVD_baseline(chunk_c)
  • sign count: X/10 chunks improve under TTA (ΔFVD < 0)
  • mean/std/min/max ΔFVD across chunks
  • a sign-test p-value (Wilson) for X/10 deviation from 5/10

Outputs
-------
    per_chunk_fvd.csv             — long format: series, method, chunk,
                                     fvd_method, fvd_baseline, dfvd
    per_chunk_fvd_summary.csv     — wide: series, method, n_chunks,
                                     wins, mean_dfvd, std_dfvd, p_value
    summary.md                     — paragraph + table per series
    boxplot_<series>.png          — boxplot of ΔFVD per method, with
                                     0.0 reference line

CPU-only — numpy + matplotlib only. Reads JSON files directly.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Loading per-chunk FVD
# ---------------------------------------------------------------------------
def _load_chunk_fvds_from_chunks(method_dir: Path) -> List[Optional[float]]:
    """Read ``chunk_*/summary.json`` and return list of per-chunk FVD values
    (None where missing). Sorted by chunk index (parsed from ``chunk_N``)."""
    out: List[Tuple[int, Optional[float]]] = []
    for c in sorted(method_dir.glob("chunk_*/summary.json")):
        try:
            idx = int(c.parent.name.split("_", 1)[1])
        except ValueError:
            continue
        try:
            with c.open() as f:
                blob = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {c}: {e}", file=sys.stderr)
            out.append((idx, None))
            continue
        v = blob.get("fvd")
        if v is None:
            out.append((idx, None))
            continue
        try:
            out.append((idx, float(v)))
        except (TypeError, ValueError):
            out.append((idx, None))
    out.sort(key=lambda kv: kv[0])
    return [v for _, v in out]


def _load_chunk_fvds_from_merged(method_dir: Path) -> List[Optional[float]]:
    """Fallback: read ``merged_summary.json['fvd_per_chunk']`` if individual
    chunk summary.json files are missing. Ordered list."""
    mp = method_dir / "merged_summary.json"
    if not mp.exists():
        return []
    try:
        with mp.open() as f:
            blob = json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {mp}: {e}", file=sys.stderr)
        return []
    vals = blob.get("fvd_per_chunk", []) or []
    out: List[Optional[float]] = []
    for v in vals:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(None)
    return out


def load_per_chunk_fvds(method_dir: Path) -> List[Optional[float]]:
    """Try chunk_* first; fall back to merged_summary.json['fvd_per_chunk']."""
    fvds = _load_chunk_fvds_from_chunks(method_dir)
    if any(v is not None for v in fvds):
        return fvds
    return _load_chunk_fvds_from_merged(method_dir)


def autodiscover_methods(series_path: Path) -> List[str]:
    """Return method subdir names that have either chunk_*/summary.json or
    a merged_summary.json with fvd_per_chunk."""
    if not series_path.exists():
        return []
    out: List[str] = []
    for sub in sorted(p for p in series_path.iterdir() if p.is_dir()):
        if any(sub.glob("chunk_*/summary.json")):
            out.append(sub.name)
            continue
        if (sub / "merged_summary.json").exists():
            try:
                with (sub / "merged_summary.json").open() as f:
                    blob = json.load(f)
                if blob.get("fvd_per_chunk"):
                    out.append(sub.name)
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# Sign-test (two-sided binomial)
# ---------------------------------------------------------------------------
def _log_binom(n: int, k: int) -> float:
    """log(C(n, k)) via lgamma — numerically stable."""
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def _binom_pmf(n: int, k: int, p: float = 0.5) -> float:
    if k < 0 or k > n:
        return 0.0
    return math.exp(_log_binom(n, k)) * (p ** k) * ((1 - p) ** (n - k))


def two_sided_sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact sign-test p-value for wins / n under p=0.5.

    Reports the probability of observing a deviation at least as extreme
    (in either direction) as the observed wins under H0: p = 0.5. For n=10
    the right tail at wins=10 is 1/1024 (one-sided 0.00098 -> two-sided
    0.00195).
    """
    if n <= 0:
        return 1.0
    expected = n / 2.0
    extreme_distance = abs(wins - expected)
    p = 0.0
    for k in range(0, n + 1):
        if abs(k - expected) >= extreme_distance:
            p += _binom_pmf(n, k, 0.5)
    return float(min(p, 1.0))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


def plot_boxplot(
    plt, out_path: Path, series_name: str,
    methods: List[str], dfvd_lists: Dict[str, List[float]],
):
    methods_plot = [m for m in methods if dfvd_lists.get(m)]
    if not methods_plot:
        return False
    data = [dfvd_lists[m] for m in methods_plot]
    fig, ax = plt.subplots(figsize=(max(6.0, 0.8 * len(methods_plot) + 3), 5.0))
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6,
               label="No-TTA baseline (ΔFVD=0)")
    bp = ax.boxplot(data, labels=methods_plot, showmeans=True,
                    meanline=True, widths=0.55)
    for i, vals in enumerate(data, start=1):
        ax.scatter([i] * len(vals), vals, color="tab:blue", alpha=0.6, s=14)
    ax.set_ylabel("per-chunk ΔFVD vs No-TTA (lower is better)")
    ax.set_title(f"{series_name} — per-chunk ΔFVD across 10 chunks")
    ax.legend(loc="best")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Series processing
# ---------------------------------------------------------------------------
def process_series(
    series_path: Path, baseline: str, explicit_methods: Optional[List[str]],
) -> Tuple[
    Dict[str, List[Optional[float]]],
    List[str],
    Dict[str, List[float]],
    Dict[str, dict],
]:
    """Return (per_method_fvds, methods_evaluated, dfvd_lists, per_method_stats).

    per_method_fvds: method -> list-of-FVDs aligned with baseline chunk index.
    dfvd_lists: method (≠baseline) -> list-of-finite-ΔFVDs.
    per_method_stats: method (≠baseline) -> dict with wins/mean/std/p_value/etc.
    """
    if not series_path.exists():
        print(f"[warn] series path {series_path} does not exist; skipping",
              file=sys.stderr)
        return {}, [], {}, {}

    methods = explicit_methods or autodiscover_methods(series_path)
    if baseline not in methods:
        if (series_path / baseline).exists():
            methods = [baseline] + [m for m in methods if m != baseline]
        else:
            print(f"[warn] baseline {baseline!r} not found under {series_path}; "
                  f"skipping series", file=sys.stderr)
            return {}, [], {}, {}

    per_method: Dict[str, List[Optional[float]]] = {}
    for m in methods:
        per_method[m] = load_per_chunk_fvds(series_path / m)

    base = per_method.get(baseline, [])
    if not base:
        print(f"[warn] baseline {baseline!r} has 0 per-chunk FVDs under "
              f"{series_path}; skipping series", file=sys.stderr)
        return per_method, methods, {}, {}

    dfvd_lists: Dict[str, List[float]] = {}
    per_method_stats: Dict[str, dict] = {}
    for m in methods:
        if m == baseline:
            continue
        ms = per_method.get(m, [])
        ds: List[float] = []
        for i in range(min(len(ms), len(base))):
            if ms[i] is None or base[i] is None:
                continue
            ds.append(ms[i] - base[i])
        dfvd_lists[m] = ds
        if ds:
            arr = np.array(ds, dtype=float)
            wins = int((arr < 0).sum())
            n = int(arr.size)
            per_method_stats[m] = {
                "n_chunks": n,
                "wins": wins,
                "losses": int((arr > 0).sum()),
                "ties": int((arr == 0).sum()),
                "mean_dfvd": float(arr.mean()),
                "std_dfvd":  float(arr.std(ddof=1)) if n > 1 else 0.0,
                "min_dfvd":  float(arr.min()),
                "max_dfvd":  float(arr.max()),
                "median_dfvd": float(np.median(arr)),
                "sign_test_p": two_sided_sign_test_p(wins, n),
                "baseline_fvd_mean": float(np.mean([b for b in base if b is not None])),
            }
        else:
            per_method_stats[m] = {
                "n_chunks": 0, "wins": 0, "losses": 0, "ties": 0,
                "mean_dfvd": float("nan"), "std_dfvd": float("nan"),
                "min_dfvd": float("nan"), "max_dfvd": float("nan"),
                "median_dfvd": float("nan"), "sign_test_p": 1.0,
                "baseline_fvd_mean": float("nan"),
            }
    return per_method, methods, dfvd_lists, per_method_stats


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------
def write_long_csv(
    out_path: Path,
    per_series: Dict[str, Tuple[Dict[str, List[Optional[float]]], str]],
):
    """Long format: series, method, chunk, fvd_method, fvd_baseline, dfvd."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["series", "method", "chunk",
                    "fvd_method", "fvd_baseline", "dfvd"])
        for series_name, (per_method, baseline) in per_series.items():
            base = per_method.get(baseline, [])
            for m, vals in per_method.items():
                if m == baseline:
                    continue
                for i in range(min(len(vals), len(base))):
                    mv = vals[i]
                    bv = base[i]
                    dv = (mv - bv) if (mv is not None and bv is not None) else None
                    w.writerow([
                        series_name, m, i,
                        "" if mv is None else f"{mv:.6f}",
                        "" if bv is None else f"{bv:.6f}",
                        "" if dv is None else f"{dv:.6f}",
                    ])


def write_summary_csv(
    out_path: Path,
    per_series_stats: Dict[str, Dict[str, dict]],
):
    """Wide format: one row per (series, method)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "series", "method", "n_chunks",
        "wins", "losses", "ties",
        "mean_dfvd", "std_dfvd", "median_dfvd",
        "min_dfvd", "max_dfvd",
        "sign_test_p", "baseline_fvd_mean",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for series_name, stats_by_m in per_series_stats.items():
            for m, s in stats_by_m.items():
                w.writerow({
                    "series": series_name,
                    "method": m,
                    "n_chunks": s.get("n_chunks", 0),
                    "wins":     s.get("wins", 0),
                    "losses":   s.get("losses", 0),
                    "ties":     s.get("ties", 0),
                    "mean_dfvd":   "" if math.isnan(s.get("mean_dfvd", float("nan"))) else f"{s['mean_dfvd']:.6f}",
                    "std_dfvd":    "" if math.isnan(s.get("std_dfvd",  float("nan"))) else f"{s['std_dfvd']:.6f}",
                    "median_dfvd": "" if math.isnan(s.get("median_dfvd", float("nan"))) else f"{s['median_dfvd']:.6f}",
                    "min_dfvd":    "" if math.isnan(s.get("min_dfvd", float("nan"))) else f"{s['min_dfvd']:.6f}",
                    "max_dfvd":    "" if math.isnan(s.get("max_dfvd", float("nan"))) else f"{s['max_dfvd']:.6f}",
                    "sign_test_p": f"{s.get('sign_test_p', 1.0):.6f}",
                    "baseline_fvd_mean": "" if math.isnan(s.get("baseline_fvd_mean", float("nan"))) else f"{s['baseline_fvd_mean']:.4f}",
                })


def write_summary_md(
    out_path: Path,
    per_series: Dict[str, Tuple[Dict[str, List[Optional[float]]], str]],
    per_series_stats: Dict[str, Dict[str, dict]],
):
    lines: List[str] = []
    lines.append("# Per-chunk ΔFVD sign analysis")
    lines.append("")
    lines.append(
        "Each TTA series has 10 chunks of ~100 videos each. Per-chunk FVD "
        "is computed correctly inside the chunk (Frechet distance of 100-"
        "video I3D feature distributions), so ΔFVD_c = FVD_method(chunk_c) "
        "− FVD_baseline(chunk_c) is a paired difference at chunk granularity. "
        "Wins are chunks where TTA improved FVD (ΔFVD<0). Sign-test p is the "
        "two-sided binomial probability under H0: p(improve)=0.5."
    )
    lines.append("")
    for series_name, (per_method, baseline) in per_series.items():
        stats_by_m = per_series_stats.get(series_name, {})
        base_vals = [v for v in per_method.get(baseline, []) if v is not None]
        base_str = (f"mean={np.mean(base_vals):.2f}, "
                    f"min={min(base_vals):.2f}, max={max(base_vals):.2f}"
                    if base_vals else "n/a")
        lines.append(f"## {series_name}")
        lines.append("")
        lines.append(f"- Baseline ({baseline}) per-chunk FVD: {base_str}  "
                     f"(N_chunks={len(base_vals)})")
        lines.append("")
        if not stats_by_m:
            lines.append("- (no non-baseline methods with valid per-chunk FVD)")
            lines.append("")
            continue
        lines.append("| method | N | wins | losses | ties | mean ΔFVD | "
                     "median ΔFVD | std | min/max | sign-test p |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|---:|")
        for m, s in stats_by_m.items():
            if s.get("n_chunks", 0) == 0:
                lines.append(
                    f"| `{m}` | 0 | — | — | — | — | — | — | — | — |"
                )
                continue
            lines.append(
                f"| `{m}` | {s['n_chunks']} | {s['wins']} | {s['losses']} | "
                f"{s['ties']} | {s['mean_dfvd']:+.2f} | "
                f"{s['median_dfvd']:+.2f} | {s['std_dfvd']:.2f} | "
                f"{s['min_dfvd']:+.2f} / {s['max_dfvd']:+.2f} | "
                f"{s['sign_test_p']:.4f} |"
            )
        lines.append("")
        lines.append(
            "Interpretation: a non-baseline method with **wins ≥ 8/10** "
            "and **sign-test p < 0.11** is evidence that TTA has a non-"
            "trivial per-chunk FVD effect even when the global merged-FVD "
            "is statistically indistinguishable from No-TTA. Conversely, "
            "**wins ≈ 5/10 and p ≈ 1.0** means the per-chunk effect is "
            "indistinguishable from coin-flip — consistent with the "
            "headline ΔFVD ≈ 0 dB story."
        )
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--series-paths", nargs="+", type=Path, required=True,
        help="One or more series roots (each contains <METHOD>/chunk_*/...). "
             "E.g. sweep_experiment/results/panda_1000v_standard "
             "delta_experiment/results/tinylora_panda_1000v_standard "
             "sweep_experiment/results/panda_longctx_1000v "
             "delta_experiment/results/tinylora_longctx_1000v",
    )
    ap.add_argument(
        "--baseline-method", default="NOTTA",
        help="Method name used as the per-chunk baseline. Default: NOTTA.",
    )
    ap.add_argument(
        "--methods", nargs="*", default=None,
        help="Optional explicit list of method subdir names. Default: "
             "auto-detect per-series.",
    )
    ap.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write per_chunk_fvd.csv, per_chunk_fvd_summary.csv, "
             "summary.md, and the boxplot PNGs.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    print("=== analyze_per_chunk_fvd ===")
    print(f"Baseline: {args.baseline_method}")
    print(f"Output dir: {args.output_dir}")
    print("Series:")
    for s in args.series_paths:
        print(f"  - {s}")
    print()

    per_series: Dict[str, Tuple[Dict[str, List[Optional[float]]], str]] = {}
    per_series_stats: Dict[str, Dict[str, dict]] = {}
    per_series_dfvd: Dict[str, Dict[str, List[float]]] = {}
    series_methods_kept: Dict[str, List[str]] = {}

    for sp in args.series_paths:
        series_name = sp.name
        per_method, methods, dfvd_lists, stats = process_series(
            sp, args.baseline_method, args.methods,
        )
        if not per_method:
            continue
        per_series[series_name] = (per_method, args.baseline_method)
        per_series_stats[series_name] = stats
        per_series_dfvd[series_name] = dfvd_lists
        series_methods_kept[series_name] = methods
        print(f"[{series_name}] methods={methods}; "
              f"per-method chunk counts: "
              f"{ {m: sum(1 for v in vals if v is not None) for m, vals in per_method.items()} }")

    if not per_series:
        print("[error] no series produced valid per-chunk FVD; abort.",
              file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    long_csv = args.output_dir / "per_chunk_fvd.csv"
    write_long_csv(long_csv, per_series)
    print(f"Wrote {long_csv}")

    summary_csv = args.output_dir / "per_chunk_fvd_summary.csv"
    write_summary_csv(summary_csv, per_series_stats)
    print(f"Wrote {summary_csv}")

    md_path = args.output_dir / "summary.md"
    write_summary_md(md_path, per_series, per_series_stats)
    print(f"Wrote {md_path}")

    plt = _setup_matplotlib()
    for series_name, dfvd_lists in per_series_dfvd.items():
        ordered_methods = [m for m in series_methods_kept[series_name]
                           if m != args.baseline_method]
        out_p = args.output_dir / f"boxplot_{series_name}.png"
        if plot_boxplot(plt, out_p, series_name, ordered_methods, dfvd_lists):
            print(f"Wrote {out_p}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
