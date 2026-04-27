#!/usr/bin/env python3
"""Prune saved videos to keep only the top-N by PSNR (or another metric).

Usage:
    python prune_videos.py --results-dir path/to/METHOD_DIR --top-n 200
    python prune_videos.py --results-dir path/to/parent --recursive --top-n 200

For chunked runs, operates on merged_summary.json (or per-chunk summary.json
files if merged is unavailable).  Deletes video files not in the top-N and
writes a retained_videos.json manifest.
"""

import argparse
import json
import os
import sys
from pathlib import Path


def load_per_video_results(run_dir: Path):
    """Collect per-video results from merged or chunked summaries."""
    merged = run_dir / "merged_summary.json"
    if merged.exists():
        with open(merged) as f:
            data = json.load(f)
        return data.get("results", [])

    results = []
    for chunk_dir in sorted(run_dir.glob("chunk_*")):
        summary = chunk_dir / "summary.json"
        if summary.exists():
            with open(summary) as f:
                data = json.load(f)
            results.extend(data.get("results", []))
    return results


def prune(run_dir: Path, top_n: int, metric: str, dry_run: bool):
    results = load_per_video_results(run_dir)
    successful = [r for r in results if r.get("success", False) and r.get(metric) is not None]

    if not successful:
        print(f"  {run_dir.name}: no successful videos with '{metric}', skipping")
        return

    successful.sort(key=lambda r: r[metric], reverse=True)

    retain = set()
    for r in successful[:top_n]:
        name = r.get("video_name", "")
        if name:
            retain.add(name)

    video_dirs = []
    for chunk_dir in sorted(run_dir.glob("chunk_*/videos")):
        video_dirs.append(chunk_dir)
    top_level = run_dir / "videos"
    if top_level.is_dir():
        video_dirs.append(top_level)

    deleted = 0
    kept = 0
    for vdir in video_dirs:
        for mp4 in sorted(vdir.glob("*.mp4")):
            stem = mp4.stem
            base_name = stem.split("_psnr")[0].split("_ssim")[0]
            if base_name in retain or stem in retain:
                kept += 1
            else:
                if dry_run:
                    print(f"    [dry-run] would delete {mp4}")
                else:
                    mp4.unlink()
                deleted += 1

    manifest_path = run_dir / "retained_videos.json"
    manifest = {
        "top_n": top_n,
        "metric": metric,
        "total_successful": len(successful),
        "retained": sorted(retain),
        "deleted_count": deleted,
        "kept_count": kept,
    }
    if not dry_run:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    action = "would delete" if dry_run else "deleted"
    print(f"  {run_dir.name}: kept {kept}, {action} {deleted} "
          f"(top {top_n} by {metric} out of {len(successful)})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Method directory (or parent with --recursive)")
    parser.add_argument("--recursive", action="store_true",
                        help="Process all subdirectories")
    parser.add_argument("--top-n", type=int, default=200,
                        help="Number of videos to keep (default: 200)")
    parser.add_argument("--metric", type=str, default="psnr",
                        help="Metric to rank by (default: psnr)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without deleting")
    args = parser.parse_args()

    root = Path(args.results_dir)
    if not root.exists():
        print(f"ERROR: {root} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.recursive:
        method_dirs = sorted(d for d in root.iterdir() if d.is_dir())
    else:
        method_dirs = [root]

    for mdir in method_dirs:
        has_chunks = any(mdir.glob("chunk_*"))
        has_videos = (mdir / "videos").is_dir() or has_chunks
        if has_videos:
            prune(mdir, args.top_n, args.metric, args.dry_run)
        else:
            print(f"  {mdir.name}: no videos directory, skipping")


if __name__ == "__main__":
    main()
