#!/usr/bin/env python3
"""
Prepare UCF-101 videos for DFoT evaluation.

DFoT expects:
  - Videos as .mp4 in data/{dataset}/{split}/ directories
  - metadata/{split}.pt with list of video metadata dicts
  - Resolution: 128x128 (matching Kinetics-600 training config)
  - At least 17 frames (context_length=5 + 12 prediction)

Reads ucf101_500_480p/metadata.csv and creates DFoT-compatible data.

Usage:
    python prepare_ucf101_dfot.py \
        --src-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_500_480p \
        --dst-dir /scratch/wc3013/longcat-video-tta/comparison_methods/data/ucf101_dfot
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

TARGET_SIZE = 128
MIN_FRAMES = 17
TARGET_FPS = 10


def get_frame_count(video_path):
    cmd = [
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", str(video_path),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip())
    except Exception:
        return 0


def center_crop_resize(src, dst, size, fps):
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "fps=%d,crop=min(iw\\,ih):min(iw\\,ih),scale=%d:%d" % (fps, size, size),
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
        str(dst),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", required=True)
    parser.add_argument("--dst-dir", required=True)
    parser.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    args = parser.parse_args()

    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    metadata_path = src_dir / "metadata.csv"

    if not metadata_path.exists():
        print("ERROR: metadata.csv not found at %s" % metadata_path)
        sys.exit(1)

    video_dir = dst_dir / "test"
    meta_dir = dst_dir / "metadata"
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        rows = list(csv.DictReader(f))

    print("Found %d videos in metadata.csv" % len(rows))
    print("Target: %dx%d @ %d FPS" % (TARGET_SIZE, TARGET_SIZE, TARGET_FPS))
    print("Minimum frames: %d" % args.min_frames)
    print()

    metadata_entries = []
    converted = skipped = failed = 0

    for i, row in enumerate(rows):
        filename = row["filename"]
        category = row["category"]
        src_path = src_dir / "videos" / filename
        if not src_path.exists():
            src_path = src_dir / filename
        if not src_path.exists():
            print("  [%d/%d] SKIP (not found): %s" % (i+1, len(rows), filename))
            failed += 1
            continue

        dst_name = Path(filename).stem + ".mp4"
        dst_path = video_dir / dst_name

        if not dst_path.exists():
            ok = center_crop_resize(str(src_path), str(dst_path), TARGET_SIZE, TARGET_FPS)
            if not ok:
                print("  [%d/%d] FAIL: %s" % (i+1, len(rows), filename))
                failed += 1
                continue

        nframes = get_frame_count(str(dst_path))
        if nframes < args.min_frames:
            print("  [%d/%d] SKIP (%d frames): %s" % (i+1, len(rows), nframes, filename))
            dst_path.unlink(missing_ok=True)
            skipped += 1
            continue

        metadata_entries.append({
            "path": str(dst_path),
            "relative_path": dst_name,
            "num_frames": nframes,
            "category": category,
            "original_filename": filename,
        })
        converted += 1
        if converted % 50 == 0:
            print("  [%d/%d] Converted %d videos..." % (i+1, len(rows), converted))

    print()
    print("=" * 60)
    print("DFoT Data Preparation Complete")
    print("  Converted: %d" % converted)
    print("  Skipped: %d" % skipped)
    print("  Failed: %d" % failed)
    print("  Output: %s" % video_dir)
    print("=" * 60)

    if torch is not None:
        meta_path = meta_dir / "test.pt"
        torch.save(metadata_entries, meta_path)
        print("  Metadata (torch): %s" % meta_path)
    else:
        print("  WARNING: torch not available, skipping .pt metadata save")

    mapping_path = dst_dir / "video_mapping.csv"
    with open(mapping_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dfot_filename", "original_filename", "category", "num_frames"])
        for entry in metadata_entries:
            writer.writerow([
                entry["relative_path"],
                entry["original_filename"],
                entry["category"],
                entry["num_frames"],
            ])
    print("  Mapping (CSV): %s" % mapping_path)


if __name__ == "__main__":
    main()
