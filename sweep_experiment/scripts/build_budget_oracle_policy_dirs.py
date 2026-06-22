#!/usr/bin/env python3
"""Build symlinked video dirs for budget-grid oracle FVD (H9).

Per eval video, pick the grid config with highest PSNR (same rule as
``scripts/analyze_adasteer_budget_oracle.py``), then symlink that config's
generated mp4 into ``<output-root>/oracle_best_psnr/videos/``.

Requires saved mp4s (``NO_SAVE_VIDEOS=0`` during the budget sweep). Pilot
runs with metrics-only mode cannot populate this dir — re-run best configs
with video retention or use the 1000v incremental jobs.

Usage:
    python sweep_experiment/scripts/build_budget_oracle_policy_dirs.py \\
        --series-root sweep_experiment/results/panda_ood_budget_pilot \\
        --output-root sweep_experiment/reports/budget_oracle_fvd
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    discover_runs,
    load_run_psnr,
    oracle_winner,
)
from scripts.caption_utils import canonical_video_id
from sweep_experiment.scripts.build_oracle_policy_dirs import (  # noqa: E402
    find_mp4,
    _load_chunk_summary_order,
    _mp4_readable,
)


def _index_grid_videos(series_root: Path, run_id: str) -> Dict[str, Path]:
    """Map canonical video_id -> generated mp4 for one grid config."""
    run_dir = series_root / run_id
    if not run_dir.is_dir():
        return {}

    out: Dict[str, Path] = {}
    chunk_dirs = sorted(run_dir.glob("chunk_*/"))
    if not chunk_dirs:
        chunk_dirs = [run_dir]

    for chunk_dir in chunk_dirs:
        videos_dir = chunk_dir / "videos"
        summary_path = chunk_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        idx_by_name = _load_chunk_summary_order(summary)

        for rec in summary.get("per_video_results", summary.get("results", [])):
            if not rec.get("success", False):
                continue
            vname = rec.get("video_name", "")
            vid = canonical_video_id(vname)
            if not vid:
                continue
            mp4 = find_mp4(videos_dir, vname, idx_by_name)
            if mp4 is None:
                op = rec.get("output_path")
                if op:
                    p = Path(op)
                    if p.exists() and p.suffix.lower() == ".mp4":
                        mp4 = p
            if mp4 is not None and _mp4_readable(mp4):
                out[vid] = mp4.resolve()
    return out


def build_oracle_dir(
    *,
    series_root: Path,
    output_root: Path,
    grid_runs: Optional[List[str]] = None,
    policy_name: str = "oracle_best_psnr",
) -> Tuple[int, int]:
    runs = discover_runs(series_root)
    if grid_runs is None:
        grid_runs = sorted(r for r in runs if r.startswith("S"))
    else:
        grid_runs = [r for r in grid_runs if r in runs]

    psnr_by_run: Dict[str, Dict[str, float]] = {
        rid: load_run_psnr(runs[rid]) for rid in grid_runs
    }
    all_vids = sorted(set().union(*[set(d.keys()) for d in psnr_by_run.values()]))

    video_index: Dict[str, Dict[str, Path]] = {}
    for rid in grid_runs:
        video_index[rid] = _index_grid_videos(series_root, rid)

    out_dir = output_root / policy_name
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    skipped = 0
    manifest: List[dict] = []

    for vid in all_vids:
        row = {rid: psnr_by_run[rid].get(vid) for rid in grid_runs}
        winner = oracle_winner(row, grid_runs)
        if winner is None:
            skipped += 1
            continue
        src = video_index.get(winner, {}).get(vid)
        if src is None:
            skipped += 1
            continue
        dst = videos_dir / f"{vid}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        linked += 1
        manifest.append({
            "video_id": vid,
            "winner_run": winner,
            "psnr": row.get(winner),
            "source_mp4": str(src),
        })

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": policy_name,
                "series_root": str(series_root),
                "grid_runs": grid_runs,
                "linked_videos": linked,
                "skipped_videos": skipped,
                "entries": manifest,
            },
            f,
            indent=2,
        )
    print(f"Built {policy_name}: linked={linked} skipped={skipped} -> {videos_dir}")
    return linked, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Build budget-grid oracle policy video dirs")
    ap.add_argument(
        "--series-root",
        type=Path,
        default=Path("sweep_experiment/results/panda_ood_budget_pilot"),
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("sweep_experiment/reports/budget_oracle_fvd"),
    )
    ap.add_argument(
        "--grid-runs", nargs="*", default=None,
        help="Subset of run IDs (default: all S* dirs with PSNR under series-root)",
    )
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if not args.series_root.is_dir():
        print(f"ERROR: series root not found: {args.series_root}", file=sys.stderr)
        return 2

    if args.clean and args.output_root.exists():
        import shutil
        shutil.rmtree(args.output_root)

    linked, skipped = build_oracle_dir(
        series_root=args.series_root,
        output_root=args.output_root,
        grid_runs=args.grid_runs,
    )
    if linked == 0:
        print(
            "ERROR: no videos linked — budget sweep likely used NO_SAVE_VIDEOS=1. "
            "Re-run with saved mp4s or use run_budget_oracle_fvd --skip-build after "
            "populating videos.",
            file=sys.stderr,
        )
        return 1
    if skipped:
        print(f"WARN: skipped {skipped} videos (missing PSNR winner mp4)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
