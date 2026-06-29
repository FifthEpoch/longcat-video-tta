#!/usr/bin/env python3
"""Win/loss magnitude tables from ``per_video_vbench_gains.csv``.

Use when the agreement CSV already exists (no need to re-read chunk VBench files).

    python3 scripts/analyze_vbench_magnitude_from_csv.py \\
        sweep_experiment/reports/per_video_analysis/2026-06-28/vbench_agreement/per_video_vbench_gains.csv \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-28/vbench_agreement/vbench_magnitude_summary.md
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_per_video_vbench_agreement import VBENCH_DIMS  # noqa: E402
from scripts.summarize_vbench_population_per_video import (  # noqa: E402
    DIM_SHORT,
    _fmt,
    _fmt_delta,
    _magnitude_stats,
)

# Delta columns: ``{METHOD}_d{metric}`` where metric is psnr/ssim/lpips or a
# full VBench dim name (e.g. ``ADA_daesthetic_quality``).  Do NOT use a
# generic ``_d(.+)`` regex — raw score columns like ``ADA_dynamic_degree``
# would be misparsed as method ``ADA_dynamic``.
_DELTA_METRICS = ("psnr", "ssim", "lpips") + tuple(VBENCH_DIMS)
_DELTA_COL = re.compile(
    r"^(.+)_d(" + "|".join(re.escape(m) for m in _DELTA_METRICS) + r")$"
)


def _load_csv(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            parsed: dict = {}
            for k, v in row.items():
                if v == "" or v is None:
                    parsed[k] = float("nan")
                else:
                    try:
                        parsed[k] = float(v)
                    except ValueError:
                        parsed[k] = v
            rows.append(parsed)
    return rows, fieldnames


def _methods_and_metrics(fieldnames: Sequence[str]) -> Tuple[List[str], List[str]]:
    methods: List[str] = []
    metrics: List[str] = []
    seen_m: set = set()
    seen_metric: set = set()
    for col in fieldnames:
        m = _DELTA_COL.match(col)
        if not m:
            continue
        method, metric = m.group(1), m.group(2)
        if method not in seen_m:
            seen_m.add(method)
            methods.append(method)
        if metric not in seen_metric:
            seen_metric.add(metric)
            metrics.append(metric)
    return methods, metrics


def build_magnitude_report(
    rows: List[dict],
    methods: Sequence[str],
    *,
    vbench_threshold: float = 0.01,
    psnr_threshold: float = 0.1,
) -> str:
    lines: List[str] = []
    lines.append("# VBench++ win/loss magnitude vs NOTTA")
    lines.append("")
    lines.append(f"- **Videos:** {len(rows)}")
    lines.append(f"- **VBench threshold:** ±{vbench_threshold}")
    lines.append(f"- **PSNR threshold:** ±{psnr_threshold} dB")
    lines.append("")
    lines.append(
        "Among videos classified **win** or **loss**, report mean/median Δ and "
        "**cancel_ratio** = mean|Δ| on wins ÷ mean|Δ| on losses (≈1 with balanced "
        "counts ⇒ net population mean ≈ 0)."
    )
    lines.append("")

    metric_order: List[Tuple[str, str, float, bool]] = [
        ("psnr", "PSNR (dB)", psnr_threshold, True),
    ]
    for d in VBENCH_DIMS:
        metric_order.append((d, DIM_SHORT.get(d, d), vbench_threshold, True))

    for key, label, thr, hib in metric_order:
        lines.append(f"## {label} (±{thr}{' dB' if key == 'psnr' else ''})")
        lines.append("")
        lines.append(
            "| Method | n_win | mean Δ win | med win | p90 win | "
            "n_loss | mean Δ loss | med loss | p10 loss | cancel_ratio | net mean Δ |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        nd = 2 if key == "psnr" else 3
        for method in methods:
            col = f"{method}_d{key}"
            deltas = np.array([float(r.get(col, float("nan"))) for r in rows], dtype=float)
            st = _magnitude_stats(deltas, thr, higher_is_better=hib)
            if st.get("n", 0) == 0:
                continue
            lines.append(
                f"| `{method}` | {st['n_win']} | {_fmt_delta(st['mean_win'], nd)} "
                f"| {_fmt_delta(st['median_win'], nd)} | {_fmt_delta(st['p90_win'], nd)} | "
                f"{st['n_loss']} | {_fmt_delta(st['mean_loss'], nd)} "
                f"| {_fmt_delta(st['median_loss'], nd)} | {_fmt_delta(st['p10_loss'], nd)} "
                f"| {_fmt(st['cancel_ratio'], 2)} | {_fmt_delta(st['mean_all'], nd)} |"
            )
        lines.append("")

    lines.append("## Compact VBench (mean Δ on wins vs losses)")
    lines.append("")
    lines.append("| Dim | Method | mean Δ win | mean Δ loss | |win|/|loss| | net mean Δ |")
    lines.append("|---|---|---:|---:|---:|---:|")
    for d in VBENCH_DIMS:
        for method in methods:
            col = f"{method}_d{d}"
            deltas = np.array([float(r.get(col, float("nan"))) for r in rows], dtype=float)
            st = _magnitude_stats(deltas, vbench_threshold, higher_is_better=True)
            if st.get("n_win", 0) == 0 and st.get("n_loss", 0) == 0:
                continue
            lines.append(
                f"| {DIM_SHORT.get(d, d)} | `{method}` | {_fmt_delta(st['mean_win'])} "
                f"| {_fmt_delta(st['mean_loss'])} | {_fmt(st['cancel_ratio'], 2)} "
                f"| {_fmt_delta(st['mean_all'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="per_video_vbench_gains.csv from agreement run")
    ap.add_argument("--output", type=Path, default=None,
                    help="Write markdown report (default: print to stdout)")
    ap.add_argument("--vbench-threshold", type=float, default=0.01)
    ap.add_argument("--psnr-threshold", type=float, default=0.1)
    args = ap.parse_args()

    rows, fieldnames = _load_csv(args.csv)
    methods, _ = _methods_and_metrics(fieldnames)
    # Drop baseline if present (deltas are vs baseline; NOTTA_d* should be ~0)
    methods = [m for m in methods if not m.upper().startswith("NOTTA")]

    report = build_magnitude_report(
        rows, methods,
        vbench_threshold=args.vbench_threshold,
        psnr_threshold=args.psnr_threshold,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
