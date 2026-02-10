#!/usr/bin/env python3
"""Resize Panda-70M videos to LongCat 480p 16:9 bucket (832x480)."""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path


TARGET_W, TARGET_H = 832, 480  # 480p bucket for ratio ~1.73 (16:9)


def resize_video(src: Path, dst: Path, w: int = TARGET_W, h: int = TARGET_H) -> bool:
    """Resize a single video using ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={w}:{h}:force_original_aspect_ratio=disable",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",  # drop audio
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=str, required=True, help="e.g. datasets/panda_100")
    p.add_argument("--dst-dir", type=str, required=True, help="e.g. datasets/panda_100_480p")
    args = p.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    src_videos = src / "videos"
    dst_videos = dst / "videos"
    dst_videos.mkdir(parents=True, exist_ok=True)

    # Copy metadata
    meta_src = src / "metadata.csv"
    if meta_src.exists():
        shutil.copy2(meta_src, dst / "metadata.csv")
        print(f"Copied metadata.csv")

    # Collect video files
    files = sorted(src_videos.glob("*.mp4"))
    print(f"Resizing {len(files)} videos to {TARGET_W}x{TARGET_H} ...")

    ok, fail = 0, 0
    for i, f in enumerate(files):
        out = dst_videos / f.name
        if out.exists():
            ok += 1
            continue
        if resize_video(f, out):
            ok += 1
        else:
            fail += 1
            print(f"  FAILED: {f.name}")
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}] done")

    print(f"\nComplete: {ok} ok, {fail} failed")
    print(f"Output:   {dst_videos}")


if __name__ == "__main__":
    main()
