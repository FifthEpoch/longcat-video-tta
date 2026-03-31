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
import shutil
import sys
from pathlib import Path

TARGET_SIZE = 256
MIN_FRAMES = 32


def get_frame_count_csv(row):
    """Get frame count from metadata CSV (preferred over ffprobe)."""
    for key in ("num_frames", "n_frames", "frame_count"):
        val = row.get(key)
        if val is not None:
            try:
                return int(val)
            except (ValueError, TypeError):
                pass
    return None


def get_frame_count_ffprobe(video_path):
    """Fallback: count frames via ffprobe (only if available)."""
    if not shutil.which("ffprobe"):
        return 0
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


def get_frame_count_av(video_path):
    """Fallback: count frames via Python av library."""
    try:
        import av
        with av.open(str(video_path)) as container:
            stream = container.streams.video[0]
            if stream.frames > 0:
                return stream.frames
            count = 0
            for _ in container.decode(video=0):
                count += 1
            return count
    except Exception:
        return 0


def center_crop_resize(src, dst, size):
    """Center-crop and resize video. Tries ffmpeg, falls back to Python av."""
    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-vf",
            "crop=min(iw\\,ih):min(iw\\,ih),scale=%d:%d" % (size, size),
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
            str(dst),
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return True
        except Exception:
            pass

    try:
        return _center_crop_resize_av(src, dst, size)
    except Exception:
        return False


def _center_crop_resize_av(src, dst, size):
    """Pure-Python fallback using av for center-crop + resize."""
    import av
    from PIL import Image

    in_container = av.open(str(src))
    in_stream = in_container.streams.video[0]
    fps = float(in_stream.average_rate) if in_stream.average_rate else 30.0

    out_container = av.open(str(dst), mode='w')
    out_stream = out_container.add_stream('libx264', rate=fps)
    out_stream.width = size
    out_stream.height = size
    out_stream.pix_fmt = 'yuv420p'
    out_stream.options = {'crf': '18', 'preset': 'fast'}

    for frame in in_container.decode(video=0):
        img = frame.to_image()
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        img = img.resize((size, size), Image.BILINEAR)
        out_frame = av.VideoFrame.from_image(img)
        for packet in out_stream.encode(out_frame):
            out_container.mux(packet)

    for packet in out_stream.encode():
        out_container.mux(packet)
    out_container.close()
    in_container.close()
    return True


def resolve_src_path(src_dir, filename):
    """Resolve video path, handling 'videos/' prefix in filename."""
    basename = Path(filename).name
    candidates = [
        src_dir / filename,
        src_dir / "videos" / basename,
        src_dir / "videos" / filename,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


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

    sample_keys = list(rows[0].keys()) if rows else []
    print("CSV columns: %s" % sample_keys)

    has_ffprobe = shutil.which("ffprobe") is not None
    has_ffmpeg = shutil.which("ffmpeg") is not None
    print("ffprobe available: %s" % has_ffprobe)
    print("ffmpeg available: %s" % has_ffmpeg)
    print()

    def get_category(row):
        return row.get("category", row.get("class_name", "unknown"))

    def get_filename(row):
        return row.get("filename", row.get("video_path", ""))

    converted = skipped_frames = failed = 0

    for i, row in enumerate(rows):
        filename = get_filename(row)
        category = get_category(row)

        src_path = resolve_src_path(src_dir, filename)
        if src_path is None:
            print("  [%d/%d] SKIP (not found): %s" % (i + 1, len(rows), filename))
            failed += 1
            continue

        class_dir = ucf_dir / category
        class_dir.mkdir(parents=True, exist_ok=True)
        dst_name = Path(filename).stem + ".mp4"
        dst_path = class_dir / dst_name

        if dst_path.exists():
            converted += 1
            continue

        nframes = get_frame_count_csv(row)
        if nframes is None:
            if has_ffprobe:
                nframes = get_frame_count_ffprobe(str(src_path))
            else:
                nframes = get_frame_count_av(str(src_path))
        if nframes < args.min_frames:
            print("  [%d/%d] SKIP (%d frames): %s" % (
                i + 1, len(rows), nframes, filename))
            skipped_frames += 1
            continue

        ok = center_crop_resize(str(src_path), str(dst_path), TARGET_SIZE)
        if ok:
            converted += 1
            if converted % 50 == 0:
                print("  [%d/%d] Converted %d videos..." % (
                    i + 1, len(rows), converted))
        else:
            print("  [%d/%d] FAIL: %s" % (i + 1, len(rows), filename))
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
        writer.writerow([
            "pvdm_path", "original_filename", "category", "caption"])
        for row in rows:
            cat = get_category(row)
            fname = get_filename(row)
            pvdm_rel = "UCF-101/%s/%s.mp4" % (cat, Path(fname).stem)
            writer.writerow([
                pvdm_rel, fname, cat, row.get("caption", "")])
    print("  Mapping: %s" % mapping_path)


if __name__ == "__main__":
    main()
