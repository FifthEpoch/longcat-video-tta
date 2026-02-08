#!/usr/bin/env python3
"""
Download a stratified subset of Panda-70M videos for LongCat-Video TTA experiments.

Strategy:
  1. Load Panda-70M metadata from HuggingFace (iejMac/Panda-70M)
  2. Stratify by broad caption categories (action, nature, people, etc.)
  3. Download videos via yt-dlp with robust retry logic
  4. Validate each download (ffprobe check, min frames, min duration)
  5. Stop when we reach --num-videos successful downloads

YouTube availability is ~20-40% for Panda-70M, so we sample a large
candidate pool (10x target) and iterate until we hit the target count.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional


# ── Caption-based stratification ─────────────────────────────────────────
# Broad categories derived from common Panda-70M caption patterns.
CATEGORY_KEYWORDS = {
    "sports":   ["sport", "game", "play", "match", "goal", "score", "team",
                 "basketball", "football", "soccer", "tennis", "baseball",
                 "swim", "run", "race", "boxing", "wrestling", "golf"],
    "nature":   ["nature", "animal", "bird", "fish", "ocean", "sea", "river",
                 "forest", "mountain", "sky", "sunset", "sunrise", "rain",
                 "snow", "flower", "tree", "garden", "wildlife", "landscape"],
    "people":   ["person", "man", "woman", "child", "kid", "people", "crowd",
                 "dance", "sing", "talk", "walk", "interview", "speak",
                 "conversation", "family", "group"],
    "food":     ["food", "cook", "kitchen", "eat", "recipe", "bake", "chef",
                 "restaurant", "meal", "dish", "drink", "coffee", "cake"],
    "vehicles": ["car", "truck", "bus", "train", "plane", "boat", "ship",
                 "motorcycle", "bike", "drive", "road", "highway", "traffic"],
    "urban":    ["city", "building", "street", "house", "room", "office",
                 "store", "shop", "market", "bridge", "tower", "construction"],
    "music":    ["music", "guitar", "piano", "drum", "violin", "concert",
                 "band", "orchestra", "song", "melody", "instrument"],
    "tech":     ["computer", "phone", "screen", "video game", "gaming",
                 "robot", "technology", "digital", "software", "app"],
}


def categorize_caption(caption: str) -> str:
    """Assign a caption to a broad category, or 'other'."""
    caption_lower = caption.lower()
    scores = {}
    for cat, keywords in CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in caption_lower)
    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best
    return "other"


# ── Video download & validation ──────────────────────────────────────────

def download_youtube_video(video_id: str, start: float, end: float,
                           out_path: Path, timeout: int = 120) -> bool:
    """Download a YouTube video clip using yt-dlp."""
    yt_dlp = shutil.which("yt-dlp")
    if not yt_dlp:
        print("ERROR: yt-dlp not found in PATH")
        return False

    url = f"https://www.youtube.com/watch?v={video_id}"

    # Build download section arg for trimming
    section_args = []
    if start is not None and end is not None:
        section_args = [
            "--download-sections", f"*{start}-{end}",
            "--force-keyframes-at-cuts",
        ]

    cmd = [
        yt_dlp,
        "-f", "bv*[ext=mp4][height<=720]+ba[ext=m4a]/bv*[height<=720]+ba/b[height<=720]/bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "--merge-output-format", "mp4",
        "--no-part",
        "--no-playlist",
        "--no-overwrites",
        "--retries", "3",
        "--fragment-retries", "3",
        "--socket-timeout", "30",
        "--extractor-args", "youtube:player_client=default,web",
        "-o", str(out_path),
        *section_args,
        url,
    ]

    try:
        result = subprocess.run(
            cmd, check=False, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            # Clean up partial downloads
            for p in out_path.parent.glob(f"{out_path.stem}*"):
                p.unlink(missing_ok=True)
            return False
        return out_path.exists() and out_path.stat().st_size > 10_000
    except subprocess.TimeoutExpired:
        for p in out_path.parent.glob(f"{out_path.stem}*"):
            p.unlink(missing_ok=True)
        return False
    except Exception as e:
        print(f"  Download error: {e}")
        return False


def validate_video(path: Path, min_duration: float = 3.0,
                   min_frames: int = 26) -> dict | None:
    """Validate a video file and return its info, or None if invalid."""
    if not path.exists() or path.stat().st_size < 10_000:
        return None

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return {"duration": None, "fps": None, "frames": None,
                "width": None, "height": None}

    cmd = [
        ffprobe, "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames:format=duration",
        "-of", "json",
        str(path),
    ]

    try:
        result = subprocess.run(
            cmd, check=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=30
        )
        info = json.loads(result.stdout)
    except Exception:
        return None

    # Parse duration
    duration = None
    try:
        duration = float(info.get("format", {}).get("duration", 0))
    except (ValueError, TypeError):
        pass
    if duration is None or duration < min_duration:
        return None

    # Parse stream info
    streams = info.get("streams", [{}])
    stream = streams[0] if streams else {}

    width = stream.get("width")
    height = stream.get("height")

    # Parse fps
    fps = None
    fps_str = stream.get("r_frame_rate", "")
    if "/" in fps_str:
        try:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            pass

    # Parse or estimate frame count
    frames = None
    nb = stream.get("nb_frames")
    if nb and str(nb).isdigit():
        frames = int(nb)
    elif fps and duration:
        frames = int(round(fps * duration))

    if frames is not None and frames < min_frames:
        return None

    return {
        "duration": duration,
        "fps": fps,
        "frames": frames,
        "width": width,
        "height": height,
    }


# ── Metadata loading ────────────────────────────────────────────────────

def load_metadata_from_hf(split: str = "train", max_rows: int = 50_000) -> list[dict]:
    """Load Panda-70M metadata from HuggingFace datasets library."""
    try:
        from datasets import load_dataset
        print(f"Loading Panda-70M metadata from HuggingFace (split={split}, max_rows={max_rows})...")
        ds = load_dataset("iejMac/Panda-70M", split=split, streaming=True)
        rows = []
        for i, item in enumerate(ds):
            if i >= max_rows:
                break
            rows.append({
                "videoID": item.get("videoID", ""),
                "url": item.get("url", ""),
                "caption": item.get("caption", ""),
                "timestamp": item.get("timestamp", {}),
            })
            if (i + 1) % 10_000 == 0:
                print(f"  Loaded {i + 1} metadata rows...")
        print(f"  Total metadata rows loaded: {len(rows)}")
        return rows
    except Exception as e:
        print(f"WARNING: Failed to load from HuggingFace: {e}")
        return []


def load_metadata_from_jsonl(path: Path) -> list[dict]:
    """Load metadata from a local JSONL file (plain text or gzipped)."""
    import gzip

    # Detect gzip by magic bytes
    with open(path, "rb") as f:
        magic = f.read(2)

    if magic == b"\x1f\x8b":
        print(f"Detected gzipped file: {path}")
        opener = lambda: gzip.open(path, "rt", encoding="utf-8", errors="replace")
    else:
        # Try utf-8, fall back to latin-1 (never fails)
        opener = lambda: open(path, "r", encoding="utf-8", errors="replace")

    rows = []
    with opener() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # skip malformed lines
    print(f"Loaded {len(rows)} rows from {path}")
    return rows


# ── Main download pipeline ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Download Panda-70M subset")
    parser.add_argument("--out-dir", type=str, required=True,
                        help="Output directory for videos and metadata")
    parser.add_argument("--num-videos", type=int, default=100,
                        help="Target number of videos to download")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for stratified sampling")
    parser.add_argument("--min-duration", type=float, default=4.0,
                        help="Minimum video duration in seconds")
    parser.add_argument("--max-duration", type=float, default=30.0,
                        help="Maximum video duration in seconds")
    parser.add_argument("--min-frames", type=int, default=33,
                        help="Minimum number of frames (33 = enough for LongCat conditioning)")
    parser.add_argument("--meta-path", type=str, default=None,
                        help="Local metadata JSONL path (skip HuggingFace)")
    parser.add_argument("--hf-split", type=str, default="train",
                        help="HuggingFace dataset split to use")
    parser.add_argument("--hf-max-rows", type=int, default=50_000,
                        help="Max metadata rows to load from HuggingFace")
    parser.add_argument("--candidate-multiplier", type=int, default=15,
                        help="Sample N * num_videos candidates to account for failures")
    parser.add_argument("--download-timeout", type=int, default=120,
                        help="Timeout per video download in seconds")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing partial download")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ────────────────────────────────────────────────────
    rows = []
    if args.meta_path and Path(args.meta_path).exists():
        try:
            rows = load_metadata_from_jsonl(Path(args.meta_path))
        except Exception as e:
            print(f"WARNING: Failed to load local metadata: {e}")
            rows = []

    # Fallback to HuggingFace if local metadata failed or was empty
    if not rows:
        print("Falling back to HuggingFace metadata...")
        rows = load_metadata_from_hf(split=args.hf_split,
                                     max_rows=args.hf_max_rows)

    if not rows:
        print("ERROR: No metadata loaded. Provide --meta-path or check HuggingFace.")
        sys.exit(1)

    # ── Normalize rows ───────────────────────────────────────────────────
    normalized = []
    for r in rows:
        if not isinstance(r, dict):
            continue  # skip malformed entries (ints, strings, etc.)
        video_id = r.get("videoID", "")
        url = r.get("url", "") or r.get("video_url", "") or r.get("video", "")

        # Extract video ID from YouTube URL if not provided
        if not video_id and url:
            m = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
            if m:
                video_id = m.group(1)

        if not video_id:
            continue

        caption = r.get("caption", "") or r.get("text", "") or "video"
        timestamp = r.get("timestamp", {})
        start = timestamp.get("start") if isinstance(timestamp, dict) else None
        end = timestamp.get("end") if isinstance(timestamp, dict) else None

        # Filter by timestamp duration if available
        if start is not None and end is not None:
            try:
                clip_dur = float(end) - float(start)
                if clip_dur < args.min_duration or clip_dur > args.max_duration:
                    continue
            except (ValueError, TypeError):
                pass

        normalized.append({
            "videoID": video_id,
            "caption": caption,
            "start": start,
            "end": end,
            "category": categorize_caption(caption),
        })

    print(f"\nNormalized {len(normalized)} candidates after filtering")

    # ── Stratified sampling ──────────────────────────────────────────────
    random.seed(args.seed)

    # Group by category
    by_cat = defaultdict(list)
    for r in normalized:
        by_cat[r["category"]].append(r)

    print(f"\nCategory distribution:")
    for cat in sorted(by_cat):
        print(f"  {cat}: {len(by_cat[cat])}")

    # Target: equal samples per category, with overflow going to 'other'
    n_cats = len(by_cat)
    target_total = args.num_videos * args.candidate_multiplier
    per_cat = max(1, target_total // n_cats)

    candidates = []
    for cat, items in by_cat.items():
        random.shuffle(items)
        candidates.extend(items[:per_cat])

    # Shuffle the combined candidate list
    random.shuffle(candidates)
    print(f"\nSampled {len(candidates)} candidates ({args.candidate_multiplier}x target)")

    # ── Check for existing downloads (resume support) ────────────────────
    existing = {}
    if args.resume:
        manifest_path = out_dir / "manifest.jsonl"
        if manifest_path.exists():
            with open(manifest_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    existing[entry["videoID"]] = entry
            print(f"\nResuming: found {len(existing)} existing downloads")

    # ── Download loop ────────────────────────────────────────────────────
    downloaded = list(existing.values())
    failures = 0
    skipped_dup = 0
    seen_ids = set(e["videoID"] for e in existing.values())
    manifest_path = out_dir / "manifest.jsonl"

    # Open manifest for appending
    manifest_f = open(manifest_path, "a" if args.resume else "w")

    print(f"\nStarting downloads (target: {args.num_videos})...")
    print("=" * 70)

    for i, cand in enumerate(candidates):
        if len(downloaded) >= args.num_videos:
            break

        vid_id = cand["videoID"]
        if vid_id in seen_ids:
            skipped_dup += 1
            continue
        seen_ids.add(vid_id)

        idx = len(downloaded)
        filename = f"panda_{idx:04d}.mp4"
        video_path = videos_dir / filename

        print(f"\n[{idx + 1}/{args.num_videos}] Trying {vid_id} "
              f"(cat={cand['category']}, attempt {i + 1})...")

        start_time = time.time()
        ok = download_youtube_video(
            video_id=vid_id,
            start=cand.get("start"),
            end=cand.get("end"),
            out_path=video_path,
            timeout=args.download_timeout,
        )

        if not ok:
            failures += 1
            print(f"  FAILED (download). Total failures: {failures}")
            continue

        # Validate
        info = validate_video(video_path,
                              min_duration=args.min_duration,
                              min_frames=args.min_frames)
        if info is None:
            failures += 1
            video_path.unlink(missing_ok=True)
            print(f"  FAILED (validation). Total failures: {failures}")
            continue

        elapsed = time.time() - start_time
        entry = {
            "index": idx,
            "filename": filename,
            "videoID": vid_id,
            "caption": cand["caption"],
            "category": cand["category"],
            "duration": info["duration"],
            "fps": info["fps"],
            "frames": info["frames"],
            "width": info["width"],
            "height": info["height"],
            "path": str(video_path),
        }
        downloaded.append(entry)

        # Write to manifest incrementally (crash-safe)
        manifest_f.write(json.dumps(entry) + "\n")
        manifest_f.flush()

        print(f"  OK: {filename} | {info['duration']:.1f}s, "
              f"{info['frames']} frames, {info['width']}x{info['height']} "
              f"| {elapsed:.1f}s")

    manifest_f.close()

    # ── Write CSV metadata ───────────────────────────────────────────────
    metadata_csv = out_dir / "metadata.csv"
    with open(metadata_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "index", "filename", "videoID", "caption", "category",
            "duration", "fps", "frames", "width", "height", "path",
        ])
        writer.writeheader()
        writer.writerows(downloaded)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"DOWNLOAD COMPLETE")
    print(f"=" * 70)
    print(f"  Target:     {args.num_videos}")
    print(f"  Downloaded: {len(downloaded)}")
    print(f"  Failures:   {failures}")
    print(f"  Skipped:    {skipped_dup} (duplicates)")
    print(f"  Success rate: {len(downloaded) / max(1, len(downloaded) + failures) * 100:.1f}%")
    print()

    # Category breakdown
    cat_counts = defaultdict(int)
    for d in downloaded:
        cat_counts[d["category"]] += 1
    print("Category breakdown:")
    for cat in sorted(cat_counts):
        print(f"  {cat}: {cat_counts[cat]}")
    print()

    print(f"Metadata: {metadata_csv}")
    print(f"Manifest: {manifest_path}")
    print(f"Videos:   {videos_dir}")

    if len(downloaded) < args.num_videos:
        print(f"\nWARNING: Only got {len(downloaded)}/{args.num_videos} videos.")
        print("Consider increasing --hf-max-rows or --candidate-multiplier")
        sys.exit(1)


if __name__ == "__main__":
    main()
