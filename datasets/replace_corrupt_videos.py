#!/usr/bin/env python3
"""
Replace corrupt videos in panda_100 by downloading new ones from Panda-70M.

Reads the existing manifest to find which videoIDs are already used,
picks new candidates from metadata, downloads replacements.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def validate_video(path: Path, min_frames: int = 33) -> bool:
    """Check if a video file is valid and has enough frames."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames,width,height",
            "-of", "csv=p=0",
            str(path),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            return False
        parts = r.stdout.strip().split(",")
        if len(parts) < 3:
            return False
        w, h, nframes = int(parts[0]), int(parts[1]), int(parts[2])
        return nframes >= min_frames and w > 0 and h > 0
    except Exception:
        return False


def download_video(video_id: str, out_path: Path, start=None, end=None, timeout=120) -> bool:
    """Download a YouTube video clip using yt-dlp."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4/best",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
    ]
    if start is not None and end is not None:
        cmd += ["--download-sections", f"*{start}-{end}"]
    cmd.append(url)

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode == 0 and out_path.exists() and out_path.stat().st_size > 500_000
    except Exception:
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="datasets/panda_100")
    p.add_argument("--min-frames", type=int, default=33)
    p.add_argument("--timeout", type=int, default=120)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    videos_dir = data_dir / "videos"
    manifest_path = data_dir / "manifest.jsonl"
    metadata_path = data_dir / "metadata.csv"

    # 1. Find corrupt videos
    print("Scanning for corrupt videos...", flush=True)
    corrupt = []
    for mp4 in sorted(videos_dir.glob("*.mp4")):
        if not validate_video(mp4, args.min_frames):
            corrupt.append(mp4)
            print(f"  Corrupt: {mp4.name} ({mp4.stat().st_size / 1024:.0f} KB)", flush=True)

    if not corrupt:
        print("All videos are valid!")
        return

    print(f"\n{len(corrupt)} corrupt videos to replace", flush=True)

    # 2. Load existing manifest to get used videoIDs
    used_ids = set()
    if manifest_path.exists():
        with open(manifest_path) as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    used_ids.add(entry.get("videoID", ""))
                except (json.JSONDecodeError, KeyError):
                    continue
    print(f"Already used {len(used_ids)} videoIDs", flush=True)

    # 3. Load metadata for replacement candidates
    candidates = []
    if metadata_path.exists():
        with open(metadata_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                candidates.append(row)

    # Also try the clean JSONL if available
    clean_jsonl = data_dir / "panda70m_metadata_clean.jsonl"
    if clean_jsonl.exists() and not candidates:
        with open(clean_jsonl) as f:
            for line in f:
                try:
                    candidates.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue

    # Filter to candidates NOT already used
    new_candidates = [c for c in candidates if c.get("videoID", c.get("video_id", "")) not in used_ids]
    print(f"Available replacement candidates: {len(new_candidates)}", flush=True)

    if not new_candidates:
        # Fall back: load from the big metadata file
        big_csv = data_dir / "panda70m_training_2m.csv"
        if big_csv.exists():
            print("Loading candidates from panda70m_training_2m.csv...", flush=True)
            with open(big_csv) as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    vid = row.get("videoID", row.get("url", ""))
                    if vid and vid not in used_ids:
                        new_candidates.append(row)
                    if len(new_candidates) >= 500:
                        break
            print(f"Loaded {len(new_candidates)} candidates from big CSV", flush=True)

    if not new_candidates:
        print("ERROR: No replacement candidates available!", flush=True)
        sys.exit(1)

    # 4. Replace each corrupt video
    replaced = 0
    ci = 0  # candidate index

    for corrupt_path in corrupt:
        print(f"\nReplacing {corrupt_path.name}...", flush=True)

        # Delete corrupt file
        corrupt_path.unlink()

        success = False
        while ci < len(new_candidates) and not success:
            cand = new_candidates[ci]
            ci += 1
            vid_id = cand.get("videoID", cand.get("video_id", ""))
            if not vid_id:
                continue

            start = cand.get("start", cand.get("timestamp_start"))
            end = cand.get("end", cand.get("timestamp_end"))

            print(f"  Trying {vid_id}...", end=" ", flush=True)
            ok = download_video(vid_id, corrupt_path, start=start, end=end, timeout=args.timeout)

            if ok and validate_video(corrupt_path, args.min_frames):
                print("OK", flush=True)
                success = True
                replaced += 1

                # Update manifest
                with open(manifest_path, "a") as mf:
                    entry = {"videoID": vid_id, "filename": corrupt_path.name,
                             "replaced": True}
                    mf.write(json.dumps(entry) + "\n")
            else:
                print("FAILED", flush=True)
                if corrupt_path.exists():
                    corrupt_path.unlink()

        if not success:
            print(f"  Could not replace {corrupt_path.name} — ran out of candidates", flush=True)

    print(f"\nDone: {replaced}/{len(corrupt)} replaced", flush=True)


if __name__ == "__main__":
    main()
