#!/usr/bin/env python3
"""Build a UCF-101 subset dataset from a retain.json list.

Reads a retain.json (one ID per line, or {"all": [...]}) plus the source
dataset's ``metadata.csv`` and writes a minimal subset dataset with the
same layout (``videos/`` symlinks + filtered ``metadata.csv``).

Used to assemble the small subset on which we regenerate No-TTA / LoRA /
AdaSteer videos for the LoRA-collapse cover image.

Idempotent: re-running overwrites prior symlinks and metadata.csv. Skips
videos whose source MP4 is missing and prints a warning.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import List, Set


def load_retain(p: Path) -> List[str]:
    text = p.read_text().strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [line.strip() for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        return list(data.get("all", []))
    if isinstance(data, list):
        return list(data)
    raise SystemExit(f"Unrecognised retain.json shape in {p}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--retain-json", required=True, type=Path)
    ap.add_argument("--src-dataset", required=True, type=Path,
                    help="Parent dataset dir containing videos/ + metadata.csv.")
    ap.add_argument("--out-dataset", required=True, type=Path,
                    help="Output dataset dir (created if missing).")
    args = ap.parse_args()

    retain = load_retain(args.retain_json)
    retain_stems: Set[str] = {Path(v).stem for v in retain}
    print(f"Retain list: {len(retain_stems)} videos")

    src_videos = args.src_dataset / "videos"
    src_meta = args.src_dataset / "metadata.csv"
    if not src_videos.is_dir():
        raise SystemExit(f"No videos/ under {args.src_dataset}")
    if not src_meta.exists():
        raise SystemExit(f"No metadata.csv under {args.src_dataset}")

    out_videos = args.out_dataset / "videos"
    out_videos.mkdir(parents=True, exist_ok=True)
    out_meta = args.out_dataset / "metadata.csv"

    # Symlink the selected MP4s.
    linked = 0
    missing: List[str] = []
    for stem in sorted(retain_stems):
        src = src_videos / f"{stem}.mp4"
        if not src.exists():
            missing.append(stem)
            continue
        dst = out_videos / f"{stem}.mp4"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        os.symlink(src.resolve(), dst)
        linked += 1
    print(f"Symlinked {linked} video files; missing for {missing}")

    # Filter metadata.csv.
    with src_meta.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        kept_rows = []
        for row in reader:
            fname = row.get("filename") or row.get("video_path") or row.get("path") or ""
            stem = Path(fname).stem
            if stem in retain_stems:
                kept_rows.append(row)
    with out_meta.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in kept_rows:
            writer.writerow(row)
    print(f"Wrote {len(kept_rows)}-row metadata.csv at {out_meta}")


if __name__ == "__main__":
    main()
