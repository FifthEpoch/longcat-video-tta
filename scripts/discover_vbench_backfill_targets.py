#!/usr/bin/env python3
"""Discover every method dir that has saved generated videos and is missing
one or more VBench dimensions.

Output: a TSV (and optional JSON) with one row per method dir, listing:
  - path
  - n_chunks_with_videos
  - total_videos
  - existing_dims (intersect of dims with at least one chunk_*/vbench_results/<dim> file)
  - missing_dims (target list minus existing_dims)
  - has_merged_summary

This drives ``submit_vbench_backfill_all.sh`` which submits one sbatch per
non-empty missing-dims row.

Run:
    python3 scripts/discover_vbench_backfill_targets.py \
        --root /scratch/$USER/longcat-video-tta \
        --target-dims motion_smoothness dynamic_degree imaging_quality temporal_flickering \
        --output sweep_experiment/reports/vbench_backfill_targets.tsv
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


SEARCH_GLOBS = [
    "sweep_experiment/results/*/*",
    "delta_experiment/results/*/*",
]

DEFAULT_TARGET_DIMS = [
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
    "temporal_flickering",
]

ALL_VBENCH_DIMS = [
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "motion_smoothness", "dynamic_degree", "imaging_quality",
    "temporal_flickering",
]


def discover_method_dirs(root: Path) -> List[Path]:
    out: List[Path] = []
    for g in SEARCH_GLOBS:
        out += [p for p in root.glob(g) if p.is_dir()]
    return sorted(set(out))


def inspect_method_dir(method_dir: Path,
                       target_dims: List[str]) -> Dict[str, object]:
    """Return per-method-dir summary."""
    chunks_with_videos = []
    total_videos = 0
    n_chunks = 0
    existing_dims_per_chunk: List[Set[str]] = []

    chunk_dirs = sorted(method_dir.glob("chunk_*"))
    if chunk_dirs:
        n_chunks = len(chunk_dirs)
        for cd in chunk_dirs:
            videos_dir = cd / "videos"
            if videos_dir.is_dir():
                n_mp4 = len(list(videos_dir.glob("*.mp4")))
                if n_mp4 > 0:
                    chunks_with_videos.append(cd)
                    total_videos += n_mp4
            vb_dir = cd / "vbench_results"
            cur: Set[str] = set()
            if vb_dir.is_dir():
                for dim in ALL_VBENCH_DIMS:
                    rf = vb_dir / f"vbench_{dim}_eval_results.json"
                    if rf.exists():
                        cur.add(dim)
            existing_dims_per_chunk.append(cur)
    else:
        # older single-job layout
        videos_dir = method_dir / "videos"
        if videos_dir.is_dir():
            n_mp4 = len(list(videos_dir.glob("*.mp4")))
            if n_mp4 > 0:
                chunks_with_videos.append(method_dir)
                total_videos = n_mp4
                n_chunks = 1
        vb_dir = method_dir / "vbench_results"
        cur: Set[str] = set()
        if vb_dir.is_dir():
            for dim in ALL_VBENCH_DIMS:
                rf = vb_dir / f"vbench_{dim}_eval_results.json"
                if rf.exists():
                    cur.add(dim)
        existing_dims_per_chunk.append(cur)

    # A dim is "existing" only if all chunks have it (intersect)
    if existing_dims_per_chunk:
        existing_dims = set.intersection(*existing_dims_per_chunk) if all(existing_dims_per_chunk) else set()
        # Also report dims present in *any* chunk (union) for diagnostics
        present_any = set.union(*existing_dims_per_chunk) if existing_dims_per_chunk else set()
    else:
        existing_dims = set()
        present_any = set()

    missing_dims = [d for d in target_dims if d not in existing_dims]

    return {
        "method_dir":          str(method_dir),
        "n_chunks":            n_chunks,
        "n_chunks_with_videos": len(chunks_with_videos),
        "total_videos":        total_videos,
        "existing_dims_all":   sorted(existing_dims),
        "existing_dims_any":   sorted(present_any),
        "missing_dims":        missing_dims,
        "has_merged_summary":  (method_dir / "merged_summary.json").exists() or
                                (method_dir / "summary.json").exists(),
        "needs_backfill":      bool(missing_dims) and len(chunks_with_videos) > 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", type=Path,
                    help="Project root (default: cwd, i.e. longcat-video-tta).")
    ap.add_argument("--target-dims", nargs="+", default=DEFAULT_TARGET_DIMS,
                    help="Dimensions we want to backfill. Default = the 4 "
                         "that are typically missing.")
    ap.add_argument("--output", type=Path, default=None,
                    help="TSV output path. If omitted, prints to stdout only.")
    ap.add_argument("--json-output", type=Path, default=None,
                    help="Optional JSON output path with full per-dir details.")
    ap.add_argument("--only-needs-backfill", action="store_true",
                    help="Filter to dirs that need backfill (have videos AND "
                         "missing target dims).")
    args = ap.parse_args()

    method_dirs = discover_method_dirs(args.root)
    print(f"Discovered {len(method_dirs)} method dir candidates", file=sys.stderr)

    rows = [inspect_method_dir(md, args.target_dims) for md in method_dirs]
    if args.only_needs_backfill:
        rows = [r for r in rows if r["needs_backfill"]]

    # Print TSV
    cols = ["method_dir", "n_chunks", "n_chunks_with_videos", "total_videos",
            "existing_dims_all", "missing_dims", "needs_backfill"]
    header = "\t".join(cols)
    lines = [header]
    for r in rows:
        lines.append("\t".join([
            r["method_dir"],
            str(r["n_chunks"]),
            str(r["n_chunks_with_videos"]),
            str(r["total_videos"]),
            ",".join(r["existing_dims_all"]) or "-",
            ",".join(r["missing_dims"]) or "-",
            str(r["needs_backfill"]),
        ]))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines) + "\n")
        print(f"Wrote TSV: {args.output}", file=sys.stderr)
    print("\n".join(lines))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(rows, indent=2))
        print(f"Wrote JSON: {args.json_output}", file=sys.stderr)

    n_need = sum(1 for r in rows if r["needs_backfill"])
    total_videos_need = sum(r["total_videos"] for r in rows if r["needs_backfill"])
    print("", file=sys.stderr)
    print(f"Summary: {n_need}/{len(rows)} method dirs need backfill, "
          f"covering {total_videos_need} total saved videos", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
