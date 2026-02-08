#!/usr/bin/env python3
"""
Download a stratified subset of Panda-70M videos for LongCat-Video TTA experiments.

Strategy:
  1. Load Panda-70M metadata (CSV/JSONL local file, Google Drive, or HuggingFace)
  2. Stratify by broad caption categories (action, nature, people, etc.)
  3. Download videos via yt-dlp with robust retry logic
  4. Validate each download (ffprobe check, min frames, min duration)
  5. Stop when we reach --num-videos successful downloads

YouTube availability is ~20-40% for Panda-70M, so we sample a large
candidate pool (15x target) and iterate until we hit the target count.
"""

from __future__ import annotations

import argparse
import csv
import gzip
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

def load_metadata_from_local(path: Path) -> list[dict]:
    """Load metadata from a local CSV or JSONL file (plain or gzipped).

    Supports:
      - .csv / .tsv  (with headers)
      - .jsonl / .jsonl.gz  (one JSON object per line)
      - .gz  (gzipped JSONL or CSV)
    """
    if not path.exists():
        print(f"  Local file not found: {path}")
        return []

    # Detect gzip by magic bytes
    with open(path, "rb") as f:
        magic = f.read(2)

    is_gz = (magic == b"\x1f\x8b")
    suffix = path.suffix.lower()

    rows = []

    # --- Try JSONL ---
    if suffix in {".jsonl", ".json"} or (is_gz and ".jsonl" in path.name.lower()):
        try:
            if is_gz:
                opener = lambda: gzip.open(path, "rt", encoding="utf-8", errors="replace")
            else:
                opener = lambda: open(path, "r", encoding="utf-8", errors="replace")
            with opener() as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict):
                            rows.append(obj)
                    except json.JSONDecodeError:
                        continue
            if rows:
                print(f"  Loaded {len(rows)} rows from JSONL: {path}")
                return rows
        except Exception as e:
            print(f"  JSONL parse failed: {e}")

    # --- Try CSV ---
    if suffix in {".csv", ".tsv"} or (not rows):
        try:
            if is_gz:
                opener = lambda: gzip.open(path, "rt", encoding="utf-8", errors="replace")
            else:
                opener = lambda: open(path, "r", encoding="utf-8", errors="replace")
            with opener() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
                    if len(rows) >= 200_000:  # cap for safety
                        break
            if rows:
                print(f"  Loaded {len(rows)} rows from CSV: {path}")
                return rows
        except Exception as e:
            print(f"  CSV parse failed: {e}")

    print(f"  Failed to parse: {path}")
    return []


def load_metadata_from_hf(max_rows: int = 50_000) -> list[dict]:
    """Load Panda-70M metadata from HuggingFace.

    The canonical dataset is multimodalart/panda-70m.
    Each row may contain MULTIPLE clips (timestamp/caption are lists),
    so we flatten them into individual clip entries.

    Splits:
      - train_2m: ~800K rows (main data)
      - validation: ~2K rows
      - test: ~2K rows
      - train: only 4 rows (index, not useful)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  HuggingFace 'datasets' library not installed")
        return []

    # Try datasets in order of reliability
    candidates = [
        ("multimodalart/panda-70m", "train_2m"),   # canonical source
        ("multimodalart/panda-70m", "validation"),  # fallback (smaller)
        ("multimodalart/panda-70m", "test"),        # fallback (smaller)
    ]

    for name, split in candidates:
        try:
            print(f"  Trying HuggingFace: {name} (split={split}) ...")
            ds = load_dataset(name, split=split, streaming=True)
            rows = []
            raw_count = 0
            for item in ds:
                raw_count += 1
                if len(rows) >= max_rows:
                    break
                # Flatten: each row has lists of timestamps/captions
                flattened = _flatten_panda70m_row(dict(item))
                rows.extend(flattened)
                if raw_count % 10_000 == 0:
                    print(f"    Processed {raw_count} raw rows -> {len(rows)} clips...")
            if rows:
                print(f"  Success: {len(rows)} clips from {raw_count} rows ({name}/{split})")
                return rows[:max_rows]
        except Exception as e:
            print(f"  Failed: {name}/{split} — {e}")
            continue

    print("  All HuggingFace sources failed")
    return []


def _flatten_panda70m_row(row: dict) -> list[dict]:
    """Flatten a single Panda-70M row into individual clip entries.

    In multimodalart/panda-70m, each row has:
      - videoID: str (YouTube video ID)
      - url: str (YouTube URL)
      - timestamp: list of [start, end] pairs (as strings like '0:00:16.300')
      - caption: list of caption strings (one per clip)
      - matching_score: list of floats
    """
    video_id = row.get("videoID", "")
    url = row.get("url", "")

    if not video_id and url:
        m = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
        if m:
            video_id = m.group(1)

    if not video_id:
        return []

    timestamps = row.get("timestamp", [])
    captions = row.get("caption", [])

    # Parse timestamps/captions if they're strings (e.g., from CSV)
    if isinstance(timestamps, str):
        try:
            timestamps = json.loads(timestamps.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            timestamps = []
    if isinstance(captions, str):
        try:
            captions = json.loads(captions.replace("'", '"'))
        except (json.JSONDecodeError, ValueError):
            captions = [captions] if captions else []

    if not isinstance(timestamps, list):
        timestamps = []
    if not isinstance(captions, list):
        captions = [captions] if captions else []

    # If no timestamps, return a single entry with just the video
    if not timestamps:
        cap = captions[0] if captions else "video"
        return [{"videoID": video_id, "url": url, "caption": cap,
                 "start": None, "end": None}]

    entries = []
    for i, ts in enumerate(timestamps):
        # Parse timestamp pair: ['0:00:16.300', '0:00:32.566']
        start_sec, end_sec = None, None
        if isinstance(ts, (list, tuple)) and len(ts) >= 2:
            start_sec = _parse_timestamp(ts[0])
            end_sec = _parse_timestamp(ts[1])

        cap = captions[i] if i < len(captions) else "video"

        entries.append({
            "videoID": video_id,
            "url": url,
            "caption": cap,
            "start": start_sec,
            "end": end_sec,
        })

    return entries


def _parse_timestamp(ts_str) -> Optional[float]:
    """Parse a timestamp string like '0:00:16.300' into seconds."""
    if ts_str is None:
        return None
    if isinstance(ts_str, (int, float)):
        return float(ts_str)
    ts_str = str(ts_str).strip()
    try:
        # Try direct float first
        return float(ts_str)
    except ValueError:
        pass
    # Parse H:MM:SS.mmm or MM:SS.mmm
    parts = ts_str.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    except (ValueError, TypeError):
        pass
    return None


def _clip_duration(start, end) -> Optional[float]:
    """Compute clip duration in seconds, or None if unknown."""
    if start is None or end is None:
        return None
    try:
        return float(end) - float(start)
    except (ValueError, TypeError):
        return None


def download_metadata_from_gdrive(out_path: Path) -> list[dict]:
    """Download official Panda-70M metadata CSV from Google Drive."""
    # Official Panda-70M Google Drive file ID
    # (the same one used in the Open-Sora-v2.0 setup)
    GDRIVE_ID = "1k7NzU6wVNZYl6NxOhLXE7Hz7OrpzNLgB"

    try:
        import gdown
    except ImportError:
        print("  gdown not installed, trying pip install...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "gdown", "--quiet"],
                           check=True, timeout=120)
            import gdown
        except Exception as e:
            print(f"  Could not install gdown: {e}")
            return []

    tmp_path = out_path.parent / "panda70m_gdrive_download.tmp"
    try:
        print(f"  Downloading from Google Drive (id={GDRIVE_ID})...")
        gdown.download(id=GDRIVE_ID, output=str(tmp_path), quiet=False, fuzzy=True)

        if not tmp_path.exists() or tmp_path.stat().st_size < 1000:
            print(f"  Download too small or failed")
            return []

        # Verify it's not an HTML error page
        with open(tmp_path, "rb") as f:
            head = f.read(100)
        if b"<!DOCTYPE html" in head or b"<html" in head:
            print(f"  Downloaded HTML instead of data (Google Drive quota exceeded?)")
            tmp_path.unlink(missing_ok=True)
            return []

        # Try to parse the downloaded file
        rows = load_metadata_from_local(tmp_path)
        if rows:
            # Save as clean JSONL for next time
            with open(out_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            tmp_path.unlink(missing_ok=True)
            print(f"  Saved clean metadata to {out_path}")
            return rows
        else:
            tmp_path.unlink(missing_ok=True)
            return []
    except Exception as e:
        print(f"  Google Drive download failed: {e}")
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return []


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
                        help="Local metadata CSV/JSONL path (skip online sources)")
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

    # ── Load metadata (try multiple sources) ─────────────────────────────
    rows = []

    # Source 1: explicit local file
    if args.meta_path:
        meta_path = Path(args.meta_path)
        print(f"\n[Source 1] Loading local metadata: {meta_path}")
        rows = load_metadata_from_local(meta_path)

    # Source 2: Google Drive (official Panda-70M metadata)
    if not rows:
        clean_meta = out_dir / "panda70m_metadata_clean.jsonl"
        if clean_meta.exists() and clean_meta.stat().st_size > 1000:
            print(f"\n[Source 2a] Loading cached clean metadata: {clean_meta}")
            rows = load_metadata_from_local(clean_meta)
        if not rows:
            print(f"\n[Source 2b] Downloading from Google Drive...")
            rows = download_metadata_from_gdrive(clean_meta)

    # Source 3: HuggingFace datasets
    if not rows:
        print(f"\n[Source 3] Trying HuggingFace...")
        rows = load_metadata_from_hf(max_rows=args.hf_max_rows)

    if not rows:
        print("\nERROR: No metadata loaded from any source!")
        print("Options:")
        print("  1. Download the Panda-70M CSV from Google Drive manually:")
        print("     https://drive.google.com/file/d/1k7NzU6wVNZYl6NxOhLXE7Hz7OrpzNLgB")
        print(f"     Save to: {out_dir / 'panda70m_metadata_clean.jsonl'}")
        print("  2. Provide --meta-path <path-to-csv-or-jsonl>")
        print("  3. pip install datasets && check huggingface.co accessibility")
        sys.exit(1)

    print(f"\n✓ Loaded {len(rows)} metadata rows")

    # ── Normalize rows ───────────────────────────────────────────────────
    # Rows from HuggingFace are already flattened (one clip per entry).
    # Rows from local files may still need flattening.
    normalized = []
    for r in rows:
        if not isinstance(r, dict):
            continue  # skip malformed entries (ints, strings, etc.)

        # Check if this row has list-valued timestamps/captions (multi-clip)
        # and needs flattening (e.g., from local CSV/JSONL)
        ts_raw = r.get("timestamp")
        cap_raw = r.get("caption")
        if isinstance(ts_raw, list) and len(ts_raw) > 0 and isinstance(ts_raw[0], list):
            # Multi-clip row — flatten it
            flat = _flatten_panda70m_row(r)
            for entry in flat:
                clip_dur = _clip_duration(entry.get("start"), entry.get("end"))
                if clip_dur is not None:
                    if clip_dur < args.min_duration or clip_dur > args.max_duration:
                        continue
                caption = entry.get("caption", "video") or "video"
                normalized.append({
                    "videoID": entry["videoID"],
                    "caption": caption,
                    "start": entry.get("start"),
                    "end": entry.get("end"),
                    "category": categorize_caption(caption),
                })
            continue

        # Single-clip row (already flattened, e.g., from HuggingFace loader)
        video_id = (r.get("videoID", "") or r.get("video_id", "")
                    or r.get("id", "") or r.get("youtube_id", ""))
        url = r.get("url", "") or r.get("video_url", "") or r.get("video", "")

        if not video_id and url:
            m = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", url)
            if m:
                video_id = m.group(1)

        if not video_id:
            continue

        caption = r.get("caption", "") or r.get("text", "") or "video"
        start = r.get("start")
        end = r.get("end")

        # Filter by clip duration if available
        clip_dur = _clip_duration(start, end)
        if clip_dur is not None:
            if clip_dur < args.min_duration or clip_dur > args.max_duration:
                continue

        normalized.append({
            "videoID": video_id,
            "caption": caption,
            "start": start,
            "end": end,
            "category": categorize_caption(caption),
        })

    print(f"Normalized {len(normalized)} candidates after filtering")

    if not normalized:
        print("ERROR: No valid candidates after normalization!")
        print("The metadata might not have 'videoID'/'url' fields.")
        # Show a sample of what we got
        for r in rows[:5]:
            print(f"  Sample row keys: {list(r.keys()) if isinstance(r, dict) else type(r)}")
        sys.exit(1)

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
                    try:
                        entry = json.loads(line.strip())
                        existing[entry["videoID"]] = entry
                    except (json.JSONDecodeError, KeyError):
                        continue
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
