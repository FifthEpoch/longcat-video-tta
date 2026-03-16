#!/usr/bin/env python3
"""
Sample a stratified subset of UCF101 videos and resize to 832x480 (16:9 480p).

Reads originals from ucf101_org/ (category subdirs with .avi files),
picks --videos-per-category videos from each category, converts to mp4
at 832x480, and writes metadata.csv.

Usage:
  # 100 videos (1 per category, original behavior):
  python prepare_ucf101_subset.py \
      --src-dir /path/to/ucf101_org \
      --dst-dir /path/to/datasets/ucf101_100_480p \
      --num-videos 100 --seed 42

  # 1000 videos (10 per category, for batch-size ablation):
  python prepare_ucf101_subset.py \
      --src-dir /path/to/ucf101_org \
      --dst-dir /path/to/datasets/ucf101_1000_480p \
      --num-videos 1000 --videos-per-category 10 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path


TARGET_W, TARGET_H = 832, 480


def class_name_to_caption(class_name: str) -> str:
    """Convert CamelCase action name to a readable caption.
    e.g. 'ApplyEyeMakeup' -> 'apply eye makeup'
    """
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", class_name)
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", words)
    return words.lower().strip()


def resize_video(src: Path, dst: Path, w: int = TARGET_W, h: int = TARGET_H) -> bool:
    """Resize and convert a single video to mp4 using ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=disable",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        stderr_lines = r.stderr.strip().splitlines()
        print(f"  ffmpeg error: {stderr_lines[-3:]}", flush=True)
    return r.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=str, required=True,
                   help="Path to ucf101_org/ with category subdirectories")
    p.add_argument("--dst-dir", type=str, required=True,
                   help="Output directory (e.g. datasets/ucf101_1000_480p)")
    p.add_argument("--num-videos", type=int, default=100)
    p.add_argument("--videos-per-category", type=int, default=None,
                   help="Number of videos to sample per category. "
                        "Default: auto (num_videos // num_categories)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    videos_dir = dst / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not found in PATH", flush=True)
        sys.exit(1)

    categories = sorted([d.name for d in src.iterdir() if d.is_dir()])
    print(f"Found {len(categories)} categories", flush=True)

    rng = random.Random(args.seed)

    vpc = args.videos_per_category
    if vpc is None:
        vpc = max(1, args.num_videos // len(categories))
    print(f"Sampling up to {vpc} video(s) per category (target: {args.num_videos})", flush=True)

    selected = []
    for cat in categories:
        cat_dir = src / cat
        vids = sorted(cat_dir.glob("*.avi"))
        if not vids:
            print(f"  WARNING: no .avi files in {cat}, skipping", flush=True)
            continue
        rng.shuffle(vids)
        caption = class_name_to_caption(cat)
        for v in vids[:vpc]:
            selected.append({
                "src_path": v,
                "category": cat,
                "caption": caption,
            })

    rng.shuffle(selected)
    selected = selected[:args.num_videos]

    n_cats_used = len(set(e["category"] for e in selected))
    print(f"Selected {len(selected)} videos from {n_cats_used} categories", flush=True)

    # Resize and convert
    metadata = []
    ok = fail = 0
    for i, entry in enumerate(selected):
        out_name = f"ucf101_{i:04d}.mp4"
        out_path = videos_dir / out_name

        # Resume: skip if already exists
        if out_path.exists() and out_path.stat().st_size > 1000:
            ok += 1
            metadata.append({
                "filename": out_name,
                "category": entry["category"],
                "caption": entry["caption"],
                "original": entry["src_path"].name,
            })
            continue

        if resize_video(entry["src_path"], out_path):
            ok += 1
            metadata.append({
                "filename": out_name,
                "category": entry["category"],
                "caption": entry["caption"],
                "original": entry["src_path"].name,
            })
        else:
            fail += 1
            print(f"  FAILED: {entry['src_path'].name}", flush=True)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(selected)}] done", flush=True)

    # Write metadata.csv
    csv_path = dst / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "category", "caption", "original"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nComplete: {ok} ok, {fail} failed", flush=True)
    print(f"Videos:   {videos_dir}", flush=True)
    print(f"Metadata: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
