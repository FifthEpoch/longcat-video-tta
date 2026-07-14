#!/usr/bin/env python3
"""Audit preview 1000v budget sweep completeness and per-video PSNR coverage."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    PILOT_GRID_RUN_ORDER,
    discover_runs,
    load_run_all_metrics,
)
from scripts.analyze_per_video_tta_gain import load_per_video_metrics  # noqa: E402


def _finite_psnr_count(run_dir: Path) -> tuple[int, int, int]:
    """Return (n_total, n_finite_psnr, n_chunks_with_summary)."""
    chunks = sorted(run_dir.glob("chunk_*/summary.json"))
    all_rows = load_per_video_metrics(run_dir)
    finite = sum(
        1 for m in all_rows.values()
        if m.get("psnr") is not None and not math.isnan(m["psnr"])
    )
    return len(all_rows), finite, len(chunks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--series-root",
        type=Path,
        default=_REPO / "sweep_experiment/results/panda_ood_budget_1000v_preview",
    )
    ap.add_argument(
        "--min-intersection",
        type=int,
        default=900,
        help="Minimum videos with finite PSNR in ALL configs (default: 900)",
    )
    args = ap.parse_args()

    print(f"Series: {args.series_root}")
    print(f"{'run_id':<14} {'chunks':>6} {'videos':>7} {'finite_psnr':>12}")
    print("-" * 44)

    problems = []
    for run_id in PILOT_GRID_RUN_ORDER:
        run_dir = args.series_root / run_id
        if not run_dir.is_dir():
            problems.append(f"{run_id}: missing dir")
            print(f"{run_id:<14} {'—':>6} {'—':>7} {'—':>12}")
            continue
        n_vid, n_fin, n_chunk = _finite_psnr_count(run_dir)
        print(f"{run_id:<14} {n_chunk:>6} {n_vid:>7} {n_fin:>12}")
        if n_chunk < 10:
            problems.append(f"{run_id}: only {n_chunk}/10 chunks")
        if n_vid < 950:
            problems.append(f"{run_id}: only {n_vid} videos (expected ~1000)")
        if n_fin < 950:
            problems.append(f"{run_id}: only {n_fin} finite PSNR rows")

    discovered = discover_runs(args.series_root)
    print()
    print(f"discover_runs(): {len(discovered)} configs")

    metric_maps = {
        rid: load_run_all_metrics(args.series_root / rid)
        for rid in PILOT_GRID_RUN_ORDER
        if (args.series_root / rid).is_dir()
    }
    n_intersection = 0
    if metric_maps:
        common = None
        for rid, m in metric_maps.items():
            ids = set(m.keys())
            common = ids if common is None else common & ids
        n_intersection = len(common or [])
        print(f"videos with finite PSNR in ALL present runs: {n_intersection}")
        if n_intersection < args.min_intersection:
            problems.append(
                f"intersection {n_intersection} < {args.min_intersection} "
                "(mixed-era chunk data — wipe all runs and resubmit full grid)"
            )

    if problems:
        print("\nWARN:")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nOK — ready for router eval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
