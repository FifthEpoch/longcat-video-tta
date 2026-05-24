#!/usr/bin/env python3
"""Create a deterministic subset from an existing video dataset.

The input dataset is expected to have a ``videos/`` directory and optionally a
``metadata.csv`` with a ``filename`` column. The output dataset symlinks (or
copies) selected videos and writes a filtered metadata.csv. This is intended for
cheap discovery sweeps where we want a fixed 200-video subset from a larger
Panda-70M or UCF-101 pool.
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def read_entries(src_dir: Path) -> List[Dict[str, str]]:
    meta_path = src_dir / "metadata.csv"
    entries: List[Dict[str, str]] = []
    if meta_path.exists():
        with meta_path.open(newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            for row in reader:
                fname = row.get("filename") or row.get("video_path") or row.get("path")
                if not fname:
                    continue
                video_path = src_dir / "videos" / fname
                if not video_path.exists():
                    video_path = src_dir / fname
                if not video_path.exists():
                    continue
                row = dict(row)
                row["filename"] = video_path.name
                row["_video_path"] = str(video_path)
                row["_fieldnames"] = ",".join(fieldnames)
                entries.append(row)
    else:
        for vp in sorted((src_dir / "videos").glob("*")):
            if vp.suffix.lower() in VIDEO_EXTS:
                entries.append({
                    "filename": vp.name,
                    "caption": "A video clip",
                    "category": "unknown",
                    "_video_path": str(vp),
                    "_fieldnames": "filename,caption,category",
                })
    if not entries:
        raise SystemExit(f"No videos found in {src_dir}")
    return entries


def select_entries(entries: List[Dict[str, str]], num_videos: int, seed: int, stratify_by: str | None) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    if not stratify_by:
        shuffled = list(entries)
        rng.shuffle(shuffled)
        return shuffled[:num_videos]

    groups = defaultdict(list)
    for row in entries:
        groups[row.get(stratify_by) or "unknown"].append(row)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    selected: List[Dict[str, str]] = []
    keys = sorted(groups.keys())
    cursor = 0
    while len(selected) < num_videos and any(groups.values()):
        key = keys[cursor % len(keys)]
        if groups[key]:
            selected.append(groups[key].pop())
        cursor += 1
    rng.shuffle(selected)
    return selected[:num_videos]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--dst-dir", required=True)
    parser.add_argument("--num-videos", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stratify-by", default=None,
                        help="Metadata column to stratify by, e.g. category/class_name. Default: random sample.")
    parser.add_argument("--copy", action="store_true",
                        help="Copy video files instead of symlinking.")
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    videos_dir = dst_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    entries = read_entries(src_dir)
    selected = select_entries(entries, args.num_videos, args.seed, args.stratify_by)

    fieldnames = []
    for raw in selected[0].get("_fieldnames", "").split(","):
        if raw and raw not in fieldnames and not raw.startswith("_"):
            fieldnames.append(raw)
    for required in ("filename", "caption", "category", "class_name", "original"):
        if any(required in row for row in selected) and required not in fieldnames:
            fieldnames.append(required)
    if "filename" not in fieldnames:
        fieldnames.insert(0, "filename")

    output_rows = []
    for row in selected:
        src_path = Path(row["_video_path"])
        dst_path = videos_dir / src_path.name
        if not dst_path.exists():
            if args.copy:
                shutil.copy2(src_path, dst_path)
            else:
                os.symlink(src_path, dst_path)
        out = {k: row.get(k, "") for k in fieldnames}
        out["filename"] = dst_path.name
        output_rows.append(out)

    with (dst_dir / "metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Created subset: {dst_dir}")
    print(f"Videos: {len(output_rows)}")
    print(f"Mode: {'copy' if args.copy else 'symlink'}")
    if args.stratify_by:
        print(f"Stratified by: {args.stratify_by}")


if __name__ == "__main__":
    main()
