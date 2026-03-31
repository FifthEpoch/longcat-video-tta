#!/usr/bin/env python3
"""
Prepare UCF-101 videos for DFoT evaluation.

DFoT expects:
  - Videos as .mp4 in data/{dataset}/{split}/ directories
  - metadata/{split}.pt with list of video metadata dicts
  - Resolution: 128x128 (matching Kinetics-600 training config)
  - At least 17 frames (context_length=5 + 12 prediction)

Reads ucf101_500_480p/metadata.csv and creates DFoT-compatible data.
"""

import argparse
import csv
import subprocess
import shutil
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

TARGET_SIZE = 128
MIN_FRAMES = 17
TARGET_FPS = 10


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


def center_crop_resize(src, dst, size, fps):
    """Center-crop, resize, and re-fps. Tries ffmpeg, falls back to av."""
    if shutil.which("ffmpeg"):
        vf = "fps=%d,crop=min(iw\\,ih):min(iw\\,ih),scale=%d:%d" % (
            fps, size, size)
        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
            str(dst),
        ]
        try:
            r = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return True
        except Exception:
            pass

    try:
        return _center_crop_resize_av(src, dst, size, fps)
    except Exception:
        return False


def _center_crop_resize_av(src, dst, size, fps):
    """Pure-Python fallback using av."""
    import av
    from PIL import Image

    in_c = av.open(str(src))
    in_s = in_c.streams.video[0]
    src_fps = float(in_s.average_rate) if in_s.average_rate else 30.0

    out_c = av.open(str(dst), mode='w')
    out_s = out_c.add_stream('libx264', rate=fps)
    out_s.width = size
    out_s.height = size
    out_s.pix_fmt = 'yuv420p'
    out_s.options = {'crf': '18', 'preset': 'fast'}

    interval = src_fps / fps
    target = 0.0

    for idx, frame in enumerate(in_c.decode(video=0)):
        if idx >= target:
            img = frame.to_image()
            w, h = img.size
            s = min(w, h)
            left = (w - s) // 2
            top = (h - s) // 2
            img = img.crop((left, top, left + s, top + s))
            img = img.resize((size, size), Image.BILINEAR)
            of = av.VideoFrame.from_image(img)
            for pkt in out_s.encode(of):
                out_c.mux(pkt)
            target += interval

    for pkt in out_s.encode():
        out_c.mux(pkt)
    out_c.close()
    in_c.close()
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

    video_dir = dst_dir / "test"
    meta_dir = dst_dir / "metadata"
    video_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        rows = list(csv.DictReader(f))

    print("Found %d videos in metadata.csv" % len(rows))
    print("Target: %dx%d @ %d FPS" % (TARGET_SIZE, TARGET_SIZE, TARGET_FPS))
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

    metadata_entries = []
    converted = skipped = failed = 0

    for i, row in enumerate(rows):
        filename = get_filename(row)
        category = get_category(row)

        src_path = resolve_src_path(src_dir, filename)
        if src_path is None:
            print("  [%d/%d] SKIP (not found): %s" % (
                i + 1, len(rows), filename))
            failed += 1
            continue

        dst_name = Path(filename).stem + ".mp4"
        dst_path = video_dir / dst_name

        if not dst_path.exists():
            ok = center_crop_resize(
                str(src_path), str(dst_path),
                TARGET_SIZE, TARGET_FPS)
            if not ok:
                print("  [%d/%d] FAIL: %s" % (
                    i + 1, len(rows), filename))
                failed += 1
                continue

        # Use CSV num_frames first; only fall back to probing the output
        nframes_csv = get_frame_count_csv(row)
        if nframes_csv is not None and nframes_csv >= args.min_frames:
            nframes = nframes_csv
        elif has_ffprobe:
            nframes = get_frame_count_ffprobe(str(dst_path))
        else:
            nframes = get_frame_count_av(str(dst_path))

        if nframes < args.min_frames:
            print("  [%d/%d] SKIP (%d frames): %s" % (
                i + 1, len(rows), nframes, filename))
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
            print("  [%d/%d] Converted %d videos..." % (
                i + 1, len(rows), converted))

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
        writer.writerow([
            "dfot_filename", "original_filename",
            "category", "num_frames"])
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
