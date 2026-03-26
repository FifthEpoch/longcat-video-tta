#!/usr/bin/env python3
"""
Prepare UCF-101 test split for TTA evaluation.

Downloads UCF-101, parses the official test split, resizes videos to 480p,
and generates a metadata.csv compatible with our TTA scripts.

Usage:
    python sweep_experiment/scripts/prepare_ucf101.py \
        --ucf101-dir /scratch/wc3013/datasets/ucf101 \
        --output-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_test_480p \
        --split-file /scratch/wc3013/datasets/ucf101/ucfTrainTestlist/testlist01.txt \
        --max-videos 2500
"""
import argparse, csv, os, subprocess, sys
from pathlib import Path
from collections import defaultdict


def parse_split_file(split_path: str) -> list:
    """Parse UCF-101 test split file. Returns list of (class_name, filename)."""
    entries = []
    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Format: ClassName/v_ClassName_g01_c01.avi [label]
            parts = line.split()
            rel_path = parts[0]
            class_name = rel_path.split("/")[0]
            entries.append((class_name, rel_path))
    return entries


def resize_video(input_path: str, output_path: str, height: int = 480):
    """Resize video to target height using ffmpeg, maintaining aspect ratio."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-an",  # no audio
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Prepare UCF-101 for TTA evaluation")
    parser.add_argument("--ucf101-dir", type=str, required=True,
                        help="Root directory containing UCF-101 videos (e.g. UCF-101/)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for resized videos and metadata")
    parser.add_argument("--split-file", type=str, required=True,
                        help="Path to testlist01.txt")
    parser.add_argument("--max-videos", type=int, default=2500)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--min-frames", type=int, default=32,
                        help="Skip videos shorter than this many frames")
    args = parser.parse_args()

    entries = parse_split_file(args.split_file)
    print(f"Found {len(entries)} test videos in split file")

    os.makedirs(args.output_dir, exist_ok=True)
    videos_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    metadata = []
    processed = 0

    for class_name, rel_path in entries:
        if processed >= args.max_videos:
            break

        src = os.path.join(args.ucf101_dir, rel_path)
        if not os.path.exists(src):
            # Try with UCF-101 subdirectory
            src = os.path.join(args.ucf101_dir, "UCF-101", rel_path)
        if not os.path.exists(src):
            continue

        # Output as mp4
        stem = Path(rel_path).stem
        dst = os.path.join(videos_dir, f"{stem}.mp4")

        if os.path.exists(dst) or resize_video(src, dst, args.height):
            caption = class_name.replace("_", " ").lower()
            metadata.append({
                "filename": f"videos/{stem}.mp4",
                "caption": f"a person performing {caption}",
                "class_name": class_name,
            })
            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{min(len(entries), args.max_videos)}")

    # Write metadata CSV
    meta_path = os.path.join(args.output_dir, "metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "caption", "class_name"])
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nDone: {processed} videos processed")
    print(f"Metadata: {meta_path}")
    print(f"Videos: {videos_dir}")

    # Print class distribution
    by_class = defaultdict(int)
    for m in metadata:
        by_class[m["class_name"]] += 1
    print(f"\n{len(by_class)} classes, top 10:")
    for cls, count in sorted(by_class.items(), key=lambda x: -x[1])[:10]:
        print(f"  {cls}: {count}")


if __name__ == "__main__":
    main()
