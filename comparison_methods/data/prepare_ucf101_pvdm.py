#!/usr/bin/env python3
"""
Prepare UCF-101 videos for PVDM evaluation.

PVDM expects:
  - Videos in class-based directories: UCF-101/<class>/<video>.mp4
  - Resolution: 256x256 (center-crop + resize)
  - At least 32 consecutive frames per video (16 cond + 16 pred)

Reads ucf101_500_480p/metadata.csv and creates re-encoded copies at 256x256.

Usage:
    python prepare_ucf101_pvdm.py \
        --src-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_500_480p \
        --dst-dir /scratch/wc3013/longcat-video-tta/comparison_methods/data/ucf101_pvdm
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

TARGET_SIZE = 256
MIN_FRAMES = 32


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


def center_crop_resize(src, dst, size):
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", "crop=min(iw\\,ih):min(iw\\,ih),scale=%d:%d" % (size, size),
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

    ucf_dir = dst_dir / "UCF-101"
    ucf_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        rows = list(csv.DictReader(f))

    print("Found %d videos in metadata.csv" % len(rows))
    print("Target resolution: %dx%d" % (TARGET_SIZE, TARGET_SIZE))
    print("Minimum frames: %d" % args.min_frames)
    print()

    converted = skipped_frames = failed = 0

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

        class_dir = ucf_dir / category
        class_dir.mkdir(parents=True, exist_ok=True)
        dst_name = Path(filename).stem + ".mp4"
        dst_path = class_dir / dst_name

        if dst_path.exists():
            converted += 1
            continue

        nframes = get_frame_count(str(src_path))
        if nframes < args.min_frames:
            print("  [%d/%d] SKIP (%d frames): %s" % (i+1, len(rows), nframes, filename))
            skipped_frames += 1
            continue

        ok = center_crop_resize(str(src_path), str(dst_path), TARGET_SIZE)
        if ok:
            converted += 1
            if converted % 50 == 0:
                print("  [%d/%d] Converted %d videos..." % (i+1, len(rows), converted))
        else:
            print("  [%d/%d] FAIL: %s" % (i+1, len(rows), filename))
            failed += 1

    print()
    print("=" * 60)
    print("PVDM Data Preparation Complete")
    print("  Converted: %d" % converted)
    print("  Skipped (< %d frames): %d" % (args.min_frames, skipped_frames))
    print("  Failed: %d" % failed)
    print("  Output: %s" % ucf_dir)
    print("=" * 60)

    mapping_path = dst_dir / "video_mapping.csv"
    with open(mapping_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pvdm_path", "original_filename", "category", "caption"])
        for row in rows:
            pvdm_rel = "UCF-101/%s/%s.mp4" % (row["category"], Path(row["filename"]).stem)
            writer.writerow([pvdm_rel, row["filename"], row["category"], row.get("caption", "")])
    print("  Mapping: %s" % mapping_path)


if __name__ == "__main__":
    main()
