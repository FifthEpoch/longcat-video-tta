#!/usr/bin/env python3
"""Side-by-side standard- vs long-horizon per-video distribution comparison.

Background
----------
At the population level (`paper_tables/2026-06-08_headline_1000v.md`),
standard-horizon Panda 1000v (28 frames, 17-frame generation) and long-
horizon Panda 1000v (76 frames) BOTH show ΔPSNR ≈ 0 vs No-TTA for every
TTA method. The user's hypothesis (locked 2026-06-11) is that long-
horizon may have *fatter tails in both directions* — a larger winner
percentage paid for by an equally larger loser percentage — even when
the population mean is identical to standard horizon.

This script tests that hypothesis directly. It consumes the
``per_video_gains.csv`` produced by
``scripts/analyze_per_video_tta_gain.py`` for the two regimes and
produces:

    summary.md                — tail-breakdown side-by-side table per
                                 method shared between regimes, plus
                                 mean/median/Q1/Q3 and a 3-line
                                 hypothesis verdict per method
    side_by_side_tails.csv    — same tail-breakdown table in long format
                                 (one row per method × regime × metric ×
                                 threshold)
    overlay_dpsnr_<METHOD>.png — overlaid ΔPSNR histograms for the two
                                 regimes, per method
    overlay_dlpips_<METHOD>.png — same for ΔLPIPS (if available)

Inputs
------
Two pre-computed analysis bundles, each containing a
``per_video_gains.csv`` with columns
``video_id, <METHOD>_psnr, <METHOD>_dpsnr, <METHOD>_dlpips, ...``.

CPU-only — pandas + numpy + matplotlib. No torch / no model loading.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def _coerce_float(s: str) -> float:
    if s is None or s == "":
        return float("nan")
    try:
        x = float(s)
    except ValueError:
        return float("nan")
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    return x


def load_per_video_gains_csv(path: Path) -> Tuple[List[dict], List[str]]:
    """Return (rows, fieldnames). Rows are dicts of strings (raw)."""
    if not path.exists():
        raise FileNotFoundError(f"Per-video gains CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def methods_in_csv(fieldnames: List[str], suffix: str = "_dpsnr") -> List[str]:
    """Methods present in CSV are anything with a ``<METHOD>_dpsnr`` column."""
    return sorted({f[: -len(suffix)] for f in fieldnames if f.endswith(suffix)})


def extract_delta_arrays(
    rows: List[dict], method: str, metric: str,
) -> np.ndarray:
    """Return a float array of ``<METHOD>_d<metric>`` values across rows."""
    col = f"{method}_d{metric}"
    return np.array([_coerce_float(r.get(col, "")) for r in rows], dtype=float)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def tail_breakdown(
    d: np.ndarray, thresholds: List[float], *,
    lower_is_better: bool = False,
) -> dict:
    """Return tail counts/percentages at each threshold + summary stats."""
    arr = d[~np.isnan(d)]
    n = int(arr.size)
    out: dict = {"n": n}
    if n == 0:
        return out
    out["mean"]   = float(arr.mean())
    out["median"] = float(np.median(arr))
    out["q1"]     = float(np.percentile(arr, 25))
    out["q3"]     = float(np.percentile(arr, 75))
    out["std"]    = float(arr.std(ddof=1)) if n > 1 else 0.0
    for t in thresholds:
        if lower_is_better:
            # winners are < -t, losers are > +t
            wins = int((arr < -t).sum())
            ties = int((np.abs(arr) <= t).sum())
            losses = int((arr > t).sum())
        else:
            wins = int((arr > t).sum())
            ties = int((np.abs(arr) <= t).sum())
            losses = int((arr < -t).sum())
        out[f"wins_{t:g}"]   = wins
        out[f"ties_{t:g}"]   = ties
        out[f"losses_{t:g}"] = losses
        out[f"wins_pct_{t:g}"]   = 100.0 * wins   / n
        out[f"ties_pct_{t:g}"]   = 100.0 * ties   / n
        out[f"losses_pct_{t:g}"] = 100.0 * losses / n
    return out


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


def plot_overlay(
    plt, out_path: Path, method: str, metric: str,
    std_delta: np.ndarray, long_delta: np.ndarray,
    std_label: str, long_label: str,
):
    """Overlay translucent histograms for the two regimes."""
    std_clean = std_delta[~np.isnan(std_delta)]
    long_clean = long_delta[~np.isnan(long_delta)]
    if std_clean.size == 0 and long_clean.size == 0:
        return False
    all_finite = np.concatenate([std_clean, long_clean])
    if all_finite.size < 2:
        return False
    lo = float(np.percentile(all_finite, 1))
    hi = float(np.percentile(all_finite, 99))
    span = max(hi - lo, 1e-3)
    pad = 0.05 * span
    edges = np.linspace(lo - pad, hi + pad, 41)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axvline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    if std_clean.size:
        ax.hist(std_clean, bins=edges, alpha=0.45, color="tab:blue",
                label=f"{std_label}  μ={std_clean.mean():+.3f} "
                      f"med={np.median(std_clean):+.3f} N={std_clean.size}",
                edgecolor="black", linewidth=0.4)
    if long_clean.size:
        ax.hist(long_clean, bins=edges, alpha=0.45, color="tab:red",
                label=f"{long_label}  μ={long_clean.mean():+.3f} "
                      f"med={np.median(long_clean):+.3f} N={long_clean.size}",
                edgecolor="black", linewidth=0.4)
    ax.set_xlabel(rf"per-video $\Delta${metric.upper()} vs No-TTA")
    ax.set_ylabel("# videos")
    ax.set_title(f"{method} — {metric.upper()} distribution: standard vs long-horizon")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
def _fmt_pct(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:5.1f}%"


def _verdict_line(
    std_b: dict, long_b: dict, t: float, *, lower_is_better: bool,
) -> str:
    """Return a one-line hypothesis-verdict per method.

    The user's hypothesis is: at long-horizon, BOTH wins% and losses% at
    threshold t go UP vs standard horizon, and the ties% (within ±t) goes DOWN.
    The verdict labels each method as ``fatter-tails / shrinking-tails /
    mixed / inconclusive (one regime missing)``.
    """
    needed = (f"wins_pct_{t:g}", f"losses_pct_{t:g}", f"ties_pct_{t:g}")
    if not all(k in std_b for k in needed) or not all(k in long_b for k in needed):
        return "inconclusive (one regime missing)"
    dw = long_b[f"wins_pct_{t:g}"]   - std_b[f"wins_pct_{t:g}"]
    dl = long_b[f"losses_pct_{t:g}"] - std_b[f"losses_pct_{t:g}"]
    dt = long_b[f"ties_pct_{t:g}"]   - std_b[f"ties_pct_{t:g}"]
    if dw > 0.5 and dl > 0.5 and dt < -0.5:
        return ("**fatter both tails** "
                f"(Δwin={dw:+.1f}pp Δlose={dl:+.1f}pp Δties={dt:+.1f}pp)")
    if dw < -0.5 and dl < -0.5 and dt > 0.5:
        return ("**shrinking both tails** "
                f"(Δwin={dw:+.1f}pp Δlose={dl:+.1f}pp Δties={dt:+.1f}pp)")
    if (dw > 0.5 and dl < -0.5) or (dw < -0.5 and dl > 0.5):
        return ("**asymmetric shift** "
                f"(Δwin={dw:+.1f}pp Δlose={dl:+.1f}pp Δties={dt:+.1f}pp)")
    return ("inside ±0.5 pp noise band "
            f"(Δwin={dw:+.1f}pp Δlose={dl:+.1f}pp Δties={dt:+.1f}pp)")


def write_summary_md(
    out_path: Path,
    std_bundle_path: Path, long_bundle_path: Path,
    shared_methods: List[str], unique_to_std: List[str], unique_to_long: List[str],
    breakdowns_psnr: Dict[str, Tuple[dict, dict]],
    breakdowns_lpips: Dict[str, Tuple[dict, dict]],
    thresholds_psnr: List[float], thresholds_lpips: List[float],
):
    """Write the side-by-side comparison summary."""
    lines: List[str] = []
    lines.append("# Standard- vs long-horizon per-video distribution comparison")
    lines.append("")
    lines.append(f"- Standard-horizon bundle:  `{std_bundle_path}`")
    lines.append(f"- Long-horizon bundle:      `{long_bundle_path}`")
    lines.append(f"- Methods shared by both bundles: "
                 + (", ".join(f"`{m}`" for m in shared_methods)
                    if shared_methods else "_(none)_"))
    if unique_to_std:
        lines.append(f"- Methods only in standard bundle (skipped): "
                     + ", ".join(f"`{m}`" for m in unique_to_std))
    if unique_to_long:
        lines.append(f"- Methods only in long-horizon bundle (skipped): "
                     + ", ".join(f"`{m}`" for m in unique_to_long))
    lines.append("")
    lines.append(
        "**User hypothesis under test (2026-06-11):** at long-horizon, both "
        "the winner tail (Δ>+t) and loser tail (Δ<−t) percentages are larger "
        "than at standard horizon, while the |Δ|≤t band shrinks correspondingly. "
        "Mean ΔPSNR can still be ≈ 0 in both regimes while the long-horizon "
        "tails carry more probability mass."
    )
    lines.append("")

    # ---- PSNR side-by-side table ------------------------------------------
    if breakdowns_psnr:
        lines.append("## ΔPSNR tail breakdown — side by side")
        lines.append("")
        lines.append("Counts (and % of N) of videos in each ΔPSNR tail at the "
                     "± 0.5 dB and ± 1.0 dB thresholds. **bold** = better tail "
                     "shape under the hypothesis (i.e. fatter winner tail or "
                     "fatter loser tail at long-horizon).")
        lines.append("")
        # one block per threshold
        for t in thresholds_psnr:
            lines.append(f"### threshold |Δ| > {t:g} dB")
            lines.append("")
            lines.append("| method | regime | N | mean Δ | median Δ | "
                         f"Δ>+{t:g} | \\|Δ\\|≤{t:g} | Δ<−{t:g} |")
            lines.append("|---|---|---:|---:|---:|---|---|---|")
            for m in shared_methods:
                std_b, long_b = breakdowns_psnr[m]
                for label, b in (("std", std_b), ("long", long_b)):
                    if b.get("n", 0) == 0:
                        lines.append(f"| `{m}` | {label} | 0 | — | — | — | — | — |")
                        continue
                    wins   = b[f"wins_{t:g}"]
                    ties   = b[f"ties_{t:g}"]
                    losses = b[f"losses_{t:g}"]
                    wp = b[f"wins_pct_{t:g}"]
                    tp = b[f"ties_pct_{t:g}"]
                    lp = b[f"losses_pct_{t:g}"]
                    lines.append(
                        f"| `{m}` | {label} | {b['n']} | "
                        f"{b['mean']:+.3f} | {b['median']:+.3f} | "
                        f"{wins} ({wp:.1f}%) | {ties} ({tp:.1f}%) | "
                        f"{losses} ({lp:.1f}%) |"
                    )
            lines.append("")
        # verdict per method
        lines.append("### Per-method hypothesis verdict (PSNR)")
        lines.append("")
        lines.append("Per-method delta-of-percentages between long-horizon and "
                     "standard-horizon regimes at the ±0.5 dB threshold. "
                     "Positive Δwin% AND positive Δlose% (with negative Δties%) "
                     "supports the user's hypothesis on that method.")
        lines.append("")
        lines.append("| method | verdict @ |Δ|>0.5 dB | verdict @ |Δ|>1.0 dB |")
        lines.append("|---|---|---|")
        for m in shared_methods:
            std_b, long_b = breakdowns_psnr[m]
            v05 = _verdict_line(std_b, long_b, 0.5, lower_is_better=False)
            v10 = _verdict_line(std_b, long_b, 1.0, lower_is_better=False)
            lines.append(f"| `{m}` | {v05} | {v10} |")
        lines.append("")

    # ---- summary stats (Q1/median/Q3) -------------------------------------
    if breakdowns_psnr:
        lines.append("## ΔPSNR distribution summary (Q1 / median / Q3)")
        lines.append("")
        lines.append("| method | regime | N | mean | std | Q1 | median | Q3 |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
        for m in shared_methods:
            std_b, long_b = breakdowns_psnr[m]
            for label, b in (("std", std_b), ("long", long_b)):
                if b.get("n", 0) == 0:
                    continue
                lines.append(
                    f"| `{m}` | {label} | {b['n']} | "
                    f"{b['mean']:+.3f} | {b['std']:.3f} | "
                    f"{b['q1']:+.3f} | {b['median']:+.3f} | {b['q3']:+.3f} |"
                )
        lines.append("")

    # ---- LPIPS side-by-side table -----------------------------------------
    if breakdowns_lpips:
        lines.append("## ΔLPIPS tail breakdown — side by side")
        lines.append("")
        lines.append("LPIPS is lower-is-better, so 'wins' = Δ<−t and "
                     "'losses' = Δ>+t. Same hypothesis: long-horizon should "
                     "show fatter tails in both directions.")
        lines.append("")
        for t in thresholds_lpips:
            lines.append(f"### threshold |Δ| > {t:g}")
            lines.append("")
            lines.append("| method | regime | N | mean Δ | median Δ | "
                         f"Δ<−{t:g} | \\|Δ\\|≤{t:g} | Δ>+{t:g} |")
            lines.append("|---|---|---:|---:|---:|---|---|---|")
            for m in shared_methods:
                std_b, long_b = breakdowns_lpips.get(m, ({}, {}))
                for label, b in (("std", std_b), ("long", long_b)):
                    if b.get("n", 0) == 0:
                        lines.append(f"| `{m}` | {label} | 0 | — | — | — | — | — |")
                        continue
                    wins   = b[f"wins_{t:g}"]
                    ties   = b[f"ties_{t:g}"]
                    losses = b[f"losses_{t:g}"]
                    wp = b[f"wins_pct_{t:g}"]
                    tp = b[f"ties_pct_{t:g}"]
                    lp = b[f"losses_pct_{t:g}"]
                    lines.append(
                        f"| `{m}` | {label} | {b['n']} | "
                        f"{b['mean']:+.5f} | {b['median']:+.5f} | "
                        f"{wins} ({wp:.1f}%) | {ties} ({tp:.1f}%) | "
                        f"{losses} ({lp:.1f}%) |"
                    )
            lines.append("")

        lines.append("### Per-method hypothesis verdict (LPIPS)")
        lines.append("")
        lines.append("| method | verdict @ |Δ|>0.005 | verdict @ |Δ|>0.01 |")
        lines.append("|---|---|---|")
        for m in shared_methods:
            std_b, long_b = breakdowns_lpips.get(m, ({}, {}))
            v0 = _verdict_line(std_b, long_b, 0.005, lower_is_better=True)
            v1 = _verdict_line(std_b, long_b, 0.01,  lower_is_better=True)
            lines.append(f"| `{m}` | {v0} | {v1} |")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CSV emission
# ---------------------------------------------------------------------------
def write_long_csv(
    out_path: Path, shared_methods: List[str],
    breakdowns_psnr: Dict[str, Tuple[dict, dict]],
    breakdowns_lpips: Dict[str, Tuple[dict, dict]],
    thresholds_psnr: List[float], thresholds_lpips: List[float],
):
    """One row per (method, regime, metric, threshold)."""
    fieldnames = [
        "method", "regime", "metric", "threshold",
        "n", "mean", "median", "std", "q1", "q3",
        "wins", "ties", "losses",
        "wins_pct", "ties_pct", "losses_pct",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        def _emit(metric: str, breakdowns, thresholds):
            for m in shared_methods:
                std_b, long_b = breakdowns.get(m, ({}, {}))
                for label, b in (("std", std_b), ("long", long_b)):
                    if b.get("n", 0) == 0:
                        continue
                    for t in thresholds:
                        writer.writerow({
                            "method": m,
                            "regime": label,
                            "metric": metric,
                            "threshold": f"{t:g}",
                            "n": b["n"],
                            "mean":   f"{b['mean']:.6f}",
                            "median": f"{b['median']:.6f}",
                            "std":    f"{b['std']:.6f}",
                            "q1":     f"{b['q1']:.6f}",
                            "q3":     f"{b['q3']:.6f}",
                            "wins":   b[f"wins_{t:g}"],
                            "ties":   b[f"ties_{t:g}"],
                            "losses": b[f"losses_{t:g}"],
                            "wins_pct":   f"{b[f'wins_pct_{t:g}']:.4f}",
                            "ties_pct":   f"{b[f'ties_pct_{t:g}']:.4f}",
                            "losses_pct": f"{b[f'losses_pct_{t:g}']:.4f}",
                        })

        _emit("psnr",  breakdowns_psnr,  thresholds_psnr)
        _emit("lpips", breakdowns_lpips, thresholds_lpips)


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--standard-bundle", type=Path, required=True,
        help="Directory containing the standard-horizon "
             "`per_video_gains.csv` (e.g. "
             "`sweep_experiment/reports/per_video_analysis/2026-06-09`).",
    )
    ap.add_argument(
        "--longhorizon-bundle", type=Path, required=True,
        help="Directory containing the long-horizon `per_video_gains.csv` "
             "(e.g. "
             "`sweep_experiment/reports/per_video_analysis/2026-06-12_longhorizon`).",
    )
    ap.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write summary.md, side_by_side_tails.csv, and the "
             "overlay PNGs.",
    )
    ap.add_argument(
        "--psnr-thresholds", nargs="+", type=float, default=[0.5, 1.0],
        help="Tail thresholds for ΔPSNR (in dB). Default: 0.5 and 1.0.",
    )
    ap.add_argument(
        "--lpips-thresholds", nargs="+", type=float, default=[0.005, 0.01],
        help="Tail thresholds for ΔLPIPS (absolute units). "
             "Default: 0.005 and 0.01.",
    )
    ap.add_argument(
        "--methods", nargs="*", default=None,
        help="Optional explicit list of methods. Default: intersection of "
             "methods present in both bundles' per_video_gains.csv.",
    )
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    std_csv = args.standard_bundle / "per_video_gains.csv"
    long_csv = args.longhorizon_bundle / "per_video_gains.csv"

    print("=== compare_horizons_per_video ===")
    print(f"Standard bundle:    {args.standard_bundle}")
    print(f"Long-horizon bundle:{args.longhorizon_bundle}")
    print(f"Output dir:         {args.output_dir}")
    print(f"PSNR thresholds:    {args.psnr_thresholds}")
    print(f"LPIPS thresholds:   {args.lpips_thresholds}")
    print()

    std_rows,  std_fields  = load_per_video_gains_csv(std_csv)
    long_rows, long_fields = load_per_video_gains_csv(long_csv)
    print(f"Loaded {len(std_rows)} std rows ({len(std_fields)} cols)  "
          f"from {std_csv}")
    print(f"Loaded {len(long_rows)} long rows ({len(long_fields)} cols)  "
          f"from {long_csv}")

    std_methods  = set(methods_in_csv(std_fields,  "_dpsnr"))
    long_methods = set(methods_in_csv(long_fields, "_dpsnr"))

    if args.methods:
        candidate = list(args.methods)
    else:
        candidate = sorted(std_methods & long_methods)

    shared_methods = [m for m in candidate
                      if m in std_methods and m in long_methods]
    unique_to_std  = sorted(std_methods  - long_methods)
    unique_to_long = sorted(long_methods - std_methods)
    print(f"Shared methods: {shared_methods}")
    print(f"Only in std:    {unique_to_std}")
    print(f"Only in long:   {unique_to_long}")
    if not shared_methods:
        print("[error] no methods shared between the two bundles — abort.",
              file=sys.stderr)
        return 2

    # ---- per-method tail breakdowns ---------------------------------------
    breakdowns_psnr: Dict[str, Tuple[dict, dict]] = {}
    breakdowns_lpips: Dict[str, Tuple[dict, dict]] = {}

    has_lpips_std = any(f.endswith("_dlpips") for f in std_fields)
    has_lpips_long = any(f.endswith("_dlpips") for f in long_fields)
    has_lpips = has_lpips_std and has_lpips_long
    if not has_lpips:
        print("[info] one bundle is missing _dlpips columns; LPIPS tables "
              "will be omitted.")

    for m in shared_methods:
        std_d  = extract_delta_arrays(std_rows,  m, "psnr")
        long_d = extract_delta_arrays(long_rows, m, "psnr")
        breakdowns_psnr[m] = (
            tail_breakdown(std_d,  args.psnr_thresholds, lower_is_better=False),
            tail_breakdown(long_d, args.psnr_thresholds, lower_is_better=False),
        )
        if has_lpips:
            std_dl  = extract_delta_arrays(std_rows,  m, "lpips")
            long_dl = extract_delta_arrays(long_rows, m, "lpips")
            breakdowns_lpips[m] = (
                tail_breakdown(std_dl,  args.lpips_thresholds, lower_is_better=True),
                tail_breakdown(long_dl, args.lpips_thresholds, lower_is_better=True),
            )

    # ---- write CSV --------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "side_by_side_tails.csv"
    write_long_csv(
        csv_path, shared_methods, breakdowns_psnr, breakdowns_lpips,
        args.psnr_thresholds, args.lpips_thresholds,
    )
    print(f"\nWrote {csv_path}")

    # ---- plots ------------------------------------------------------------
    plt = _setup_matplotlib()
    std_label  = f"standard ({args.standard_bundle.name})"
    long_label = f"long-horizon ({args.longhorizon_bundle.name})"
    for m in shared_methods:
        std_d  = extract_delta_arrays(std_rows,  m, "psnr")
        long_d = extract_delta_arrays(long_rows, m, "psnr")
        out_p = args.output_dir / f"overlay_dpsnr_{m}.png"
        if plot_overlay(plt, out_p, m, "psnr", std_d, long_d, std_label, long_label):
            print(f"Wrote {out_p}")
        if has_lpips:
            std_dl  = extract_delta_arrays(std_rows,  m, "lpips")
            long_dl = extract_delta_arrays(long_rows, m, "lpips")
            out_l = args.output_dir / f"overlay_dlpips_{m}.png"
            if plot_overlay(plt, out_l, m, "lpips",
                            std_dl, long_dl, std_label, long_label):
                print(f"Wrote {out_l}")

    # ---- summary.md -------------------------------------------------------
    md_path = args.output_dir / "summary.md"
    write_summary_md(
        md_path, args.standard_bundle, args.longhorizon_bundle,
        shared_methods, unique_to_std, unique_to_long,
        breakdowns_psnr, breakdowns_lpips,
        args.psnr_thresholds, args.lpips_thresholds,
    )
    print(f"Wrote {md_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
