#!/usr/bin/env python3
"""
Prepare UCF-101 dataset for TTA evaluation with reliable FVD.

Walks the original UCF-101 class directories, resizes videos to
LongCat 480p bucket (832x480), filters by minimum frame count,
and produces a metadata.csv with class-based captions.

Usage (on cluster):
    python sweep_experiment/scripts/prepare_ucf101_500.py \
        --src-dir /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org \
        --dst-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_500_480p \
        --num-videos 500 \
        --min-frames 62 \
        --seed 42
"""
import argparse
import csv
import os
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
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", src,
        "-vf", f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=disable",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        dst,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def make_caption(class_name: str) -> str:
    words = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", class_name).lower()
    return f"a person performing {words}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=str, required=True,
                        help="UCF-101 root with class subdirs (e.g. ucf101_org/)")
    parser.add_argument("--dst-dir", type=str, required=True,
                        help="Output dir (e.g. datasets/ucf101_500_480p)")
    parser.add_argument("--num-videos", type=int, default=500)
    parser.add_argument("--min-frames", type=int, default=62,
                        help="Minimum frames after resize (gen_start + gen_frames)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    videos_dir = dst / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    all_videos = []
    for class_dir in sorted(src.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for vf in sorted(class_dir.glob("*.avi")):
            all_videos.append((class_name, vf))

    print(f"Found {len(all_videos)} source videos across "
          f"{len(set(c for c, _ in all_videos))} classes")

    random.seed(args.seed)
    random.shuffle(all_videos)

    metadata = []
    skipped_short = 0
    skipped_fail = 0

    for i, (class_name, src_path) in enumerate(all_videos):
        if len(metadata) >= args.num_videos:
            break

        stem = src_path.stem
        dst_path = videos_dir / f"{stem}.mp4"

        already_exists = dst_path.exists() and dst_path.stat().st_size > 0
        if not already_exists:
            if not resize_video(str(src_path), str(dst_path)):
                skipped_fail += 1
                print(f"  SKIP (resize failed): {src_path.name}")
                continue

        nf = count_frames(str(dst_path))
        if nf < args.min_frames:
            skipped_short += 1
            dst_path.unlink(missing_ok=True)
            if (skipped_short % 50) == 0:
                print(f"  ... {skipped_short} short videos skipped so far")
            continue

        caption = make_caption(class_name)
        metadata.append({
            "filename": f"videos/{stem}.mp4",
            "caption": caption,
            "class_name": class_name,
            "num_frames": nf,
        })

        if len(metadata) % 50 == 0:
            print(f"  [{len(metadata)}/{args.num_videos}] accepted "
                  f"(scanned {i+1}, short={skipped_short}, fail={skipped_fail})")

    meta_path = dst / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "caption", "class_name", "num_frames"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nDone: {len(metadata)} videos accepted")
    print(f"  Skipped (short <{args.min_frames}f): {skipped_short}")
    print(f"  Skipped (resize failed): {skipped_fail}")
    print(f"  Metadata: {meta_path}")
    print(f"  Videos:   {videos_dir}")

    by_class = defaultdict(int)
    for m in metadata:
        by_class[m["class_name"]] += 1
    print(f"\n{len(by_class)} classes represented, top 10:")
    for cls, count in sorted(by_class.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cls}: {count}")

    frame_counts = [m["num_frames"] for m in metadata]
    if frame_counts:
        print(f"\nFrame stats: min={min(frame_counts)} max={max(frame_counts)} "
              f"mean={sum(frame_counts)/len(frame_counts):.0f}")


if __name__ == "__main__":
    main()
