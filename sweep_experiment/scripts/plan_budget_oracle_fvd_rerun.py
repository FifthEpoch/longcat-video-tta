#!/usr/bin/env python3
"""Plan minimal mp4 re-run for budget-grid oracle FVD.

The oracle needs one mp4 per eval video — from whichever grid config won
PSNR for that video. Metrics-only sweeps (NO_SAVE_VIDEOS=1) still have PSNR
in summary.json; this script lists which run_ids must save mp4s and how many
videos each config wins.

Usage:
    python sweep_experiment/scripts/plan_budget_oracle_fvd_rerun.py \\
        --series-root sweep_experiment/results/panda_ood_budget_pilot

    # Minimal re-run on cluster (after git pull):
    ONLY_RUNS="S2_LR1e2 S10_LR1e2 ..." NO_SAVE_VIDEOS=0 \\
        bash sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    discover_runs,
    load_run_psnr,
    oracle_winner,
)
from sweep_experiment.scripts.build_budget_oracle_policy_dirs import (  # noqa: E402
    _index_grid_videos,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Plan budget oracle FVD mp4 re-run")
    ap.add_argument(
        "--series-root",
        type=Path,
        default=Path("sweep_experiment/results/panda_ood_budget_pilot"),
    )
    ap.add_argument("--grid-runs", nargs="*", default=None)
    args = ap.parse_args()

    if not args.series_root.is_dir():
        print(f"ERROR: not found: {args.series_root}", file=sys.stderr)
        return 2

    runs = discover_runs(args.series_root)
    grid_runs = sorted(r for r in runs if r.startswith("S"))
    if args.grid_runs:
        grid_runs = [r for r in args.grid_runs if r in runs]

    psnr_by_run = {rid: load_run_psnr(runs[rid]) for rid in grid_runs}
    all_vids = sorted(set().union(*[set(d.keys()) for d in psnr_by_run.values()]))
    video_index = {rid: _index_grid_videos(args.series_root, rid) for rid in grid_runs}

    winners: Counter[str] = Counter()
    missing_mp4 = 0
    have_mp4 = 0
    for vid in all_vids:
        row = {rid: psnr_by_run[rid].get(vid) for rid in grid_runs}
        winner = oracle_winner(row, grid_runs)
        if winner is None:
            continue
        winners[winner] += 1
        if video_index.get(winner, {}).get(vid):
            have_mp4 += 1
        else:
            missing_mp4 += 1

    print(f"Series: {args.series_root}")
    print(f"Videos with PSNR oracle winner: {sum(winners.values())}")
    print(f"Oracle mp4s already on disk: {have_mp4}")
    print(f"Oracle mp4s missing (need re-run): {missing_mp4}")
    print()
    print("Winner run_id -> video count (configs to re-run with NO_SAVE_VIDEOS=0):")
    for rid, n in winners.most_common():
        indexed = len(video_index.get(rid, {}))
        print(f"  {rid:12s}  wins={n:4d}  mp4s_on_disk={indexed:4d}")

    needed = [rid for rid, n in winners.items() if n > len(video_index.get(rid, {}))]
    if needed:
        only_runs = " ".join(sorted(needed, key=lambda r: -winners[r]))
        print()
        print("Suggested minimal re-run:")
        print(f'  ONLY_RUNS="{only_runs}" NO_SAVE_VIDEOS=0 \\')
        print("    bash sweep_experiment/sbatch/submit_adasteer_budget_pilot.sh")
    elif missing_mp4 == 0 and have_mp4 > 0:
        print()
        print("All oracle mp4s present — run:")
        print("  python3 sweep_experiment/scripts/run_budget_oracle_fvd.py \\")
        print(f"      --series-root {args.series_root} \\")
        print("      --gt-cache gt_caches/panda_1000_longcat.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
