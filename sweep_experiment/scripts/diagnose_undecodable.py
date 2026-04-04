#!/usr/bin/env python3
"""
Diagnose undecodable videos in a dataset directory.

Tests each video with both PyAV (matching the runtime check in common.py)
and ffprobe, reporting which videos fail and why.

Usage:
    python sweep_experiment/scripts/diagnose_undecodable.py \
        --data-dir /scratch/wc3013/longcat-video-tta/datasets/ucf101_500_480p \
        --output sweep_experiment/reports/ucf101_bad_videos.json
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def test_pyav(video_path):
    """Test if PyAV can open and decode at least one frame."""
    try:
        import av
        container = av.open(str(video_path))
        try:
            frame = next(container.decode(video=0))
            n_frames = 0
            for _ in container.decode(video=0):
                n_frames += 1
            container.close()
            return True, n_frames + 1, None
        except StopIteration:
            container.close()
            return False, 0, "empty_stream"
    except Exception as e:
        return False, 0, str(e)


def test_ffprobe(video_path):
    """Test if ffprobe can read the video and count frames."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets,codec_name,width,height",
            "-of", "csv=p=0",
            str(video_path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False, 0, "ffprobe_error: %s" % r.stderr.strip()[:200]
        parts = r.stdout.strip().split(",")
        if len(parts) < 4:
            return False, 0, "ffprobe_parse_error: %s" % r.stdout.strip()[:200]
        codec = parts[0]
        w, h = int(parts[1]), int(parts[2])
        n_packets = int(parts[3])
        return True, n_packets, None
    except Exception as e:
        return False, 0, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose undecodable videos in a dataset")
    parser.add_argument("--data-dir", required=True,
                        help="Path to dataset directory with metadata.csv and videos/")
    parser.add_argument("--output", required=True,
                        help="Path to write bad_videos.json")
    parser.add_argument("--min-frames", type=int, default=62,
                        help="Minimum frames required (default 62)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta_path = data_dir / "metadata.csv"

    video_paths = []
    if meta_path.exists():
        with open(meta_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", row.get("video_path", ""))
                vp = data_dir / "videos" / fname
                if not vp.exists():
                    vp = data_dir / fname
                video_paths.append((fname, vp))
        print("Loaded %d entries from %s" % (len(video_paths), meta_path))
    else:
        for mp4 in sorted((data_dir / "videos").glob("*.mp4")):
            video_paths.append((mp4.name, mp4))
        print("Found %d MP4 files in %s/videos/" % (len(video_paths), data_dir))

    bad_videos = []
    good_count = 0
    pyav_fail = 0
    ffprobe_fail = 0
    short_count = 0

    for i, (fname, vp) in enumerate(video_paths):
        if not vp.exists():
            bad_videos.append({
                "filename": fname,
                "path": str(vp),
                "error": "file_not_found",
                "pyav_ok": False,
                "ffprobe_ok": False,
            })
            continue

        pyav_ok, pyav_frames, pyav_err = test_pyav(vp)
        ffprobe_ok, ffprobe_frames, ffprobe_err = test_ffprobe(vp)

        is_bad = False
        errors = []

        if not pyav_ok:
            is_bad = True
            errors.append("pyav: %s" % pyav_err)
            pyav_fail += 1

        if not ffprobe_ok:
            is_bad = True
            errors.append("ffprobe: %s" % ffprobe_err)
            ffprobe_fail += 1

        frame_count = pyav_frames if pyav_ok else ffprobe_frames
        if frame_count < args.min_frames:
            is_bad = True
            errors.append("short: %d < %d frames" % (frame_count, args.min_frames))
            short_count += 1

        if is_bad:
            bad_videos.append({
                "filename": fname,
                "path": str(vp),
                "pyav_ok": pyav_ok,
                "pyav_frames": pyav_frames,
                "ffprobe_ok": ffprobe_ok,
                "ffprobe_frames": ffprobe_frames,
                "errors": errors,
            })
        else:
            good_count += 1

        if (i + 1) % 100 == 0:
            print("  Checked %d/%d videos..." % (i + 1, len(video_paths)))

    output = {
        "data_dir": str(data_dir),
        "total_videos": len(video_paths),
        "good_videos": good_count,
        "bad_videos_count": len(bad_videos),
        "pyav_failures": pyav_fail,
        "ffprobe_failures": ffprobe_fail,
        "short_videos": short_count,
        "min_frames": args.min_frames,
        "bad_videos": bad_videos,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 60)
    print("Diagnosis Complete")
    print("  Total: %d" % len(video_paths))
    print("  Good:  %d" % good_count)
    print("  Bad:   %d" % len(bad_videos))
    print("    PyAV failures:    %d" % pyav_fail)
    print("    ffprobe failures: %d" % ffprobe_fail)
    print("    Too short:        %d" % short_count)
    print("  Output: %s" % args.output)
    print("=" * 60)

    if bad_videos:
        print("\nFirst 10 bad videos:")
        for bv in bad_videos[:10]:
            print("  %s: %s" % (bv["filename"], ", ".join(bv.get("errors", [bv.get("error", "unknown")]))))


if __name__ == "__main__":
    main()
