#!/usr/bin/env python3
"""Resize Panda-70M videos to LongCat 480p 16:9 bucket (832x480)."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


TARGET_W, TARGET_H = 832, 480  # 480p bucket for ratio ~1.73 (16:9)


def check_ffmpeg():
    """Verify ffmpeg is available."""
    r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    if r.returncode != 0:
        print("ERROR: ffmpeg not found in PATH", flush=True)
        sys.exit(1)
    # Print first line (version info)
    print(f"ffmpeg: {r.stdout.splitlines()[0]}", flush=True)


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
    if r.returncode != 0:
        # Print last few lines of stderr for debugging
        err_lines = r.stderr.strip().splitlines()[-5:]
        print(f"  ffmpeg error: {'; '.join(err_lines)}", flush=True)
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

    check_ffmpeg()

    # Copy metadata
    meta_src = src / "metadata.csv"
    if meta_src.exists():
        shutil.copy2(meta_src, dst / "metadata.csv")
        print("Copied metadata.csv", flush=True)

    # Collect video files
    files = sorted(src_videos.glob("*.mp4"))
    print(f"Resizing {len(files)} videos to {TARGET_W}x{TARGET_H} ...", flush=True)

    if not files:
        print(f"ERROR: No .mp4 files found in {src_videos}", flush=True)
        sys.exit(1)

    ok, fail = 0, 0
    for i, f in enumerate(files):
        out = dst_videos / f.name
        if out.exists() and out.stat().st_size > 0:
            ok += 1
            continue
        if resize_video(f, out):
            ok += 1
        else:
            fail += 1
            print(f"  FAILED: {f.name}", flush=True)
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(files)}] done", flush=True)

    print(f"\nComplete: {ok} ok, {fail} failed", flush=True)
    print(f"Output:   {dst_videos}", flush=True)


if __name__ == "__main__":
    main()
