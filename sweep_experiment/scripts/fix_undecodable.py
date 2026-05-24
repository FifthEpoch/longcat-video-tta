#!/usr/bin/env python3
"""
Re-encode undecodable videos using ffmpeg and re-validate with PyAV.

Reads a bad_videos.json produced by diagnose_undecodable.py, attempts
to re-encode each problematic video, then validates the result.

Usage:
    python sweep_experiment/scripts/fix_undecodable.py \
        --bad-videos sweep_experiment/reports/ucf101_bad_videos.json \
        --output sweep_experiment/reports/ucf101_fix_results.json
"""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def test_pyav(video_path, min_frames=32):
    """Validate video is decodable by PyAV with sufficient frames."""
    try:
        import av
        container = av.open(str(video_path))
        n_frames = 0
        for _ in container.decode(video=0):
            n_frames += 1
        container.close()
        if n_frames < min_frames:
            return False, n_frames, "short: %d frames" % n_frames
        return True, n_frames, None
    except Exception as e:
        return False, 0, str(e)


def reencode(src, dst):
    """Re-encode video with libx264 for maximum compatibility."""
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-c:v", "libx264", "-crf", "18", "-preset", "medium",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return r.returncode == 0, r.stderr[:500] if r.returncode != 0 else None


def main():
    parser = argparse.ArgumentParser(
        description="Re-encode bad videos and validate")
    parser.add_argument("--bad-videos", required=True,
                        help="Path to bad_videos.json from diagnose script")
    parser.add_argument("--output", required=True,
                        help="Path to write fix_results.json")
    parser.add_argument("--min-frames", type=int, default=32,
                        help="Minimum frames required after re-encode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without modifying files")
    parser.add_argument("--backup-dir", type=str, default=None,
                        help="Directory to copy originals before overwriting")
    args = parser.parse_args()

    with open(args.bad_videos, "r") as f:
        report = json.load(f)

    bad_list = report["bad_videos"]
    print("Loaded %d bad videos from %s" % (len(bad_list), args.bad_videos))

    if args.backup_dir:
        Path(args.backup_dir).mkdir(parents=True, exist_ok=True)

    results = []
    fixed = 0
    failed = 0
    skipped = 0

    for i, entry in enumerate(bad_list):
        src = Path(entry["path"])
        fname = entry["filename"]

        if not src.exists():
            print("  [%d/%d] SKIP %s (file not found)" %
                  (i + 1, len(bad_list), fname))
            results.append({
                "filename": fname,
                "status": "skipped",
                "reason": "file_not_found",
            })
            skipped += 1
            continue

        if args.dry_run:
            print("  [%d/%d] DRY-RUN would re-encode %s" %
                  (i + 1, len(bad_list), fname))
            results.append({
                "filename": fname,
                "status": "dry_run",
            })
            continue

        if args.backup_dir:
            backup_path = Path(args.backup_dir) / fname
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(backup_path))

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        ok, ffmpeg_err = reencode(src, tmp_path)
        if not ok:
            print("  [%d/%d] FAIL ffmpeg re-encode %s: %s" %
                  (i + 1, len(bad_list), fname, ffmpeg_err))
            results.append({
                "filename": fname,
                "status": "reencode_failed",
                "error": ffmpeg_err,
            })
            tmp_path.unlink(missing_ok=True)
            failed += 1
            continue

        pyav_ok, n_frames, pyav_err = test_pyav(tmp_path, args.min_frames)
        if not pyav_ok:
            print("  [%d/%d] FAIL post-reencode validation %s: %s" %
                  (i + 1, len(bad_list), fname, pyav_err))
            results.append({
                "filename": fname,
                "status": "validation_failed",
                "error": pyav_err,
                "frames_after_reencode": n_frames,
            })
            tmp_path.unlink(missing_ok=True)
            failed += 1
            continue

        shutil.move(str(tmp_path), str(src))
        print("  [%d/%d] FIXED %s (%d frames)" %
              (i + 1, len(bad_list), fname, n_frames))
        results.append({
            "filename": fname,
            "status": "fixed",
            "frames": n_frames,
        })
        fixed += 1

    output = {
        "source": args.bad_videos,
        "total_bad": len(bad_list),
        "fixed": fixed,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 60)
    print("Fix Results")
    print("  Total bad:  %d" % len(bad_list))
    print("  Fixed:      %d" % fixed)
    print("  Failed:     %d" % failed)
    print("  Skipped:    %d" % skipped)
    print("  Output:     %s" % args.output)
    print("=" * 60)


if __name__ == "__main__":
    main()
