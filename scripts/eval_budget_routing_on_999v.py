#!/usr/bin/env python3
"""Offline eval: NR-proxy routing on 999v 3-config budget VBench (no new videos)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.analyze_adasteer_budget_oracle import discover_runs, build_video_table, FIXED_ADA_RUN_ID
from scripts.analyze_adasteer_budget_vbench_oracle import (
    build_score_table,
    filter_vbench_grid_runs,
    load_vbench_by_run,
)
from scripts.analyze_per_video_vbench_agreement import VBENCH_DIMS
from scripts.budget_routing_common import BESTOF3_RUNS, bootstrap_captured, load_metric_matrix
from scripts.train_vbench_headroom_router import eval_config_pick_policy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--series-root",
        type=Path,
        default=_REPO / "sweep_experiment/results/panda_1000v_adasteer_budget_vbench",
    )
    ap.add_argument("--fixed-run", type=str, default=FIXED_ADA_RUN_ID)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    if not args.series_root.is_dir():
        print(f"[skip] series not found: {args.series_root}", file=sys.stderr)
        return 0

    runs = discover_runs(args.series_root)
    grid = [r for r in runs if r.startswith("S")]
    if args.fixed_run not in grid:
        print(f"[error] fixed run {args.fixed_run} not in series", file=sys.stderr)
        return 2

    _ids, psnr_table = build_video_table(runs)
    vids = sorted(psnr_table.keys())
    vb = load_vbench_by_run(runs, list(runs.keys()))
    grid, _ = filter_vbench_grid_runs(vb, grid, min_videos=50)
    total_table, _ = build_score_table(vb, grid, vids, list(VBENCH_DIMS))

    n = len(vids)
    k = len(grid)
    Y = np.full((n, k), np.nan)
    fixed_vb = np.full(n, np.nan)
    for i, vid in enumerate(vids):
        row = total_table.get(vid, {})
        if args.fixed_run in row:
            fixed_vb[i] = row[args.fixed_run]
        for j, rid in enumerate(grid):
            if rid in row:
                Y[i, j] = row[rid]

    psnr = load_metric_matrix(runs, grid, vids, "psnr")
    picks = np.full(n, -1, dtype=int)
    bestof_cols = [grid.index(r) for r in BESTOF3_RUNS if r in grid]
    for i in range(n):
        if bestof_cols:
            scores = [(j, psnr[i, j]) for j in bestof_cols if np.isfinite(psnr[i, j])]
            if scores:
                picks[i] = max(scores, key=lambda x: x[1])[0]
        else:
            picks[i] = int(np.nanargmax(psnr[i]))

    policy = eval_config_pick_policy(picks, Y, fixed_vb, grid)
    cap = policy["fraction_oracle_captured"]

    mask = np.isfinite(fixed_vb) & (picks >= 0)
    pv = np.array([Y[i, picks[i]] for i in range(n) if mask[i]])
    ov = np.array([np.nanmax(Y[i]) for i in range(n) if mask[i]])
    fv = fixed_vb[mask]
    _, dlo, dhi, clo, chi = bootstrap_captured(pv, ov, fv)

    lines = [
        "# 999v offline proxy routing (best-of-3 PSNR)",
        "",
        f"**Series:** `{args.series_root}`",
        f"**Configs:** {', '.join(grid)}",
        f"**Fixed:** `{args.fixed_run}`",
        f"**N:** {int(np.sum(np.isfinite(fixed_vb)))}",
        "",
        f"- Proxy captured: **{100 * cap:.1f}%**",
        f"- Bootstrap captured: **[{100 * clo:.1f}%, {100 * chi:.1f}%]**",
        f"- Bootstrap Δ vs fixed: **[{dlo:+.4f}, {dhi:+.4f}]**",
        "",
    ]
    out = args.output or (
        _REPO / "sweep_experiment/reports/per_video_analysis/2026-07-05/budget_routing_experiments/999v_proxy_bestof3.md"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
