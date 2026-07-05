#!/usr/bin/env python3
"""Exp13: route budget pilot via DOVER scores on S2/S10 probe mp4s → eval VBench total.

Compares deployable DOVER probe routing against exp10 GT upper bound (18.4%).

Usage:
  python3 scripts/run_dover_probe_routing_eval.py \\
      --dover-csv-dir sweep_experiment/reports/dover_scores \\
      --series-root sweep_experiment/results/panda_ood_budget_pilot \\
      --output-dir sweep_experiment/reports/per_video_analysis/2026-07-05/dover_probe_routing
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.budget_routing_common import load_pilot_bundle, labeled_mask  # noqa: E402
from scripts.train_vbench_headroom_router import eval_config_pick_policy  # noqa: E402

PROBE2 = ("S2_LR5e3", "S10_LR5e3")
FULL_MAP = {"S2_LR5e3": "S5_LR5e3", "S10_LR5e3": "S10_LR5e3"}


def load_dover_csv(path: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["video_id"]] = float(row["fused"])
    return out


def load_dover_scores_dir(csv_dir: Path, run_id: str) -> Dict[str, float]:
    """Merge shard CSVs for one probe run."""
    merged: Dict[str, float] = {}
    for p in sorted(csv_dir.glob(f"{run_id}*.csv")):
        merged.update(load_dover_csv(p))
    return merged


def route_from_dover(
    video_ids: Sequence[str],
    grid: Sequence[str],
    scores_by_run: Dict[str, Dict[str, float]],
) -> np.ndarray:
    n = len(video_ids)
    picks = np.full(n, -1, dtype=int)
    probe_js = [grid.index(r) for r in PROBE2 if r in grid]
    for i, vid in enumerate(video_ids):
        best_j, best_s = -1, float("-inf")
        for j in probe_js:
            rid = grid[j]
            s = scores_by_run.get(rid, {}).get(vid)
            if s is None or not np.isfinite(s):
                continue
            if s > best_s:
                best_s, best_j = s, j
        if best_j < 0:
            continue
        rid = grid[best_j]
        target = FULL_MAP.get(rid, rid)
        if target in grid:
            picks[i] = grid.index(target)
    return picks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dover-csv-dir", type=Path, required=True)
    ap.add_argument("--series-root", type=Path, default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot")
    ap.add_argument(
        "--feature-date",
        type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-06",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    scores_by_run: Dict[str, Dict[str, float]] = {}
    missing: List[str] = []
    for rid in PROBE2:
        scores_by_run[rid] = load_dover_scores_dir(args.dover_csv_dir, rid)
        if not scores_by_run[rid]:
            missing.append(rid)

    if missing:
        print(f"ERROR: missing DOVER CSV for {missing}", file=sys.stderr)
        return 2

    bundle = load_pilot_bundle(args.series_root, args.feature_date)
    grid = bundle["grid_runs"]
    Y = bundle["Y_total"]
    fixed_vb = bundle["fixed_vb"]
    mask = labeled_mask(fixed_vb, Y)
    picks = route_from_dover(bundle["video_ids"], grid, scores_by_run)
    valid = mask & (picks >= 0)
    pol = eval_config_pick_policy(picks[valid], Y[valid], fixed_vb[valid], grid)
    oracle_idx = np.nanargmax(Y[valid], axis=1)
    match_rate = float(np.mean(picks[valid] == oracle_idx))

    row = {
        "experiment": "exp13_dover_probe_route",
        "n_videos": int(mask.sum()),
        "n_scored": int(valid.sum()),
        "match_rate": match_rate,
        "captured_pct": 100 * pol["fraction_oracle_captured"],
        "policy_gain": pol["mean_policy_vbench"] - pol["mean_fixed_vbench"],
        "headroom": pol["oracle_headroom"],
        "top_picks": pol.get("top_picks"),
        "reference_exp10_upper_bound_pct": 18.4,
        "reference_exp7_best_oof_pct": 12.8,
    }

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "exp13_dover_probe_route.json").write_text(json.dumps(row, indent=2), encoding="utf-8")

    lines = [
        "# Exp13: DOVER probe routing @ N=200",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Captured % (total VBench) | {row['captured_pct']:.1f} |",
        f"| Oracle match % | {100 * match_rate:.1f} |",
        f"| Videos scored | {row['n_scored']} / {row['n_videos']} |",
        "",
        "## Reference",
        "",
        f"- exp10 GT Aes+IQ upper bound: **18.4%**",
        f"- exp7 best honest OOF: **12.8%**",
        f"- Success bar: **>25%**",
        "",
    ]
    (out / "exp13_dover_probe_route.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(row, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
