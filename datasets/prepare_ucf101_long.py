#!/usr/bin/env python3
"""
Prepare a stratified UCF-101 subset for long-context generation experiments.

Stratified sampling across categories, with minimum frame count filtering
to ensure all videos are long enough for extended generation horizons.

Usage (on cluster):
    python datasets/prepare_ucf101_long.py \
        --src-dir /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org \
        --dst-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_1000_long_480p \
        --num-videos 1000 \
        --min-frames 93 \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import random
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

TARGET_W, TARGET_H = 832, 480


def count_frames(video_path: str) -> int:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_packets",
        "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0",
        str(video_path),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return -1
    try:
        return int(r.stdout.strip())
    except ValueError:
        return -1


def resize_video(src: str, dst: str) -> bool:
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=disable",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        dst,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def class_name_to_caption(name: str) -> str:
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", words)
    return words.lower().strip()


def main():
    p = argparse.ArgumentParser(
        description="Prepare stratified UCF-101 subset with min-frame filtering")
    p.add_argument("--src-dir", type=str, required=True,
                   help="UCF-101 root with category subdirs (e.g. ucf101_org/)")
    p.add_argument("--dst-dir", type=str, required=True,
                   help="Output directory (e.g. datasets/ucf101_1000_long_480p)")
    p.add_argument("--num-videos", type=int, default=1000)
    p.add_argument("--min-frames", type=int, default=93,
                   help="Minimum frames after resize (must be >= num_frames in generation config)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    videos_dir = dst / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # Discover all source videos grouped by category
    by_cat: dict[str, list[Path]] = defaultdict(list)
    for cat_dir in sorted(src.iterdir()):
        if not cat_dir.is_dir():
            continue
        for vf in sorted(cat_dir.glob("*.avi")):
            by_cat[cat_dir.name].append(vf)

    categories = sorted(by_cat.keys())
    total_src = sum(len(v) for v in by_cat.values())
    print(f"Source: {total_src} videos across {len(categories)} categories")
    print(f"Target: {args.num_videos} videos, min {args.min_frames} frames")

    rng = random.Random(args.seed)
    for cat in categories:
        rng.shuffle(by_cat[cat])

    # Phase 1: resize + frame-count all candidates, keep those >= min_frames
    print(f"\nPhase 1: Resize and filter (min {args.min_frames} frames)...")
    eligible: dict[str, list[dict]] = defaultdict(list)
    n_resized = 0
    n_short = 0
    n_fail = 0

    for cat in categories:
        for vf in by_cat[cat]:
            tmp_path = videos_dir / f"_tmp_{vf.stem}.mp4"
            final_ok = False

            if not resize_video(str(vf), str(tmp_path)):
                n_fail += 1
                continue

            nf = count_frames(str(tmp_path))
            n_resized += 1

            if nf < args.min_frames:
                n_short += 1
                tmp_path.unlink(missing_ok=True)
                continue

            eligible[cat].append({
                "src_path": vf,
                "tmp_path": tmp_path,
                "num_frames": nf,
                "category": cat,
                "caption": class_name_to_caption(cat),
            })

        if n_resized % 200 == 0 and n_resized > 0:
            n_eligible = sum(len(v) for v in eligible.values())
            print(f"  Resized {n_resized}: {n_eligible} eligible, "
                  f"{n_short} short, {n_fail} failed", flush=True)

    n_eligible_total = sum(len(v) for v in eligible.values())
    n_cats_with_eligible = sum(1 for v in eligible.values() if v)
    print(f"\nPhase 1 done: {n_eligible_total} eligible videos "
          f"from {n_cats_with_eligible}/{len(categories)} categories")
    print(f"  Short (<{args.min_frames}f): {n_short}, Failed: {n_fail}")

    if n_eligible_total < args.num_videos:
        print(f"\nWARNING: Only {n_eligible_total} eligible videos available, "
              f"fewer than requested {args.num_videos}")

    # Phase 2: Stratified selection
    print(f"\nPhase 2: Stratified selection...")
    target = min(args.num_videos, n_eligible_total)
    base_per_cat = target // len(categories) if categories else 0

    selected = []
    overflow_pool = []

    for cat in categories:
        pool = eligible.get(cat, [])
        take = min(base_per_cat, len(pool))
        selected.extend(pool[:take])
        overflow_pool.extend(pool[take:])

    remaining = target - len(selected)
    if remaining > 0:
        rng.shuffle(overflow_pool)
        selected.extend(overflow_pool[:remaining])

    rng.shuffle(selected)
    print(f"Selected {len(selected)} videos from "
          f"{len(set(e['category'] for e in selected))} categories")

    # Phase 3: Rename to final paths and write metadata
    print(f"\nPhase 3: Finalizing...")
    metadata = []
    for i, entry in enumerate(selected):
        final_name = f"ucf101_{i:04d}.mp4"
        final_path = videos_dir / final_name
        tmp = entry["tmp_path"]
        if tmp.exists():
            tmp.rename(final_path)
        metadata.append({
            "filename": final_name,
            "category": entry["category"],
            "caption": entry["caption"],
            "original": entry["src_path"].name,
            "num_frames": entry["num_frames"],
        })

    # Clean up leftover tmp files
    for tmp in videos_dir.glob("_tmp_*.mp4"):
        tmp.unlink(missing_ok=True)

    csv_path = dst / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "category", "caption", "original", "num_frames"])
        writer.writeheader()
        writer.writerows(metadata)

    # Summary
    cats_used = defaultdict(int)
    for m in metadata:
        cats_used[m["category"]] += 1
    frame_counts = [m["num_frames"] for m in metadata]

    print(f"\nDone: {len(metadata)} videos")
    print(f"  Categories: {len(cats_used)}")
    print(f"  Frames: min={min(frame_counts)} max={max(frame_counts)} "
          f"mean={sum(frame_counts)/len(frame_counts):.0f}")
    print(f"  Videos per category: min={min(cats_used.values())} "
          f"max={max(cats_used.values())} "
          f"mean={sum(cats_used.values())/len(cats_used):.1f}")
    print(f"\n  Metadata: {csv_path}")
    print(f"  Videos:   {videos_dir}")


if __name__ == "__main__":
    main()
