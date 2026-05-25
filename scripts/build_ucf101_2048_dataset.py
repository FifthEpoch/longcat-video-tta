#!/usr/bin/env python3
"""
Build the UCF-101 2048-video dataset for the headline 5-method comparison.

Strategy: extension of the existing 1000-video dataset, identical in
spirit to `build_panda_2048_dataset.py`. The new dataset contains:

  - indices 0..999    : symlinks/copies to the existing ucf101_*_480p set.
  - indices 1000..N   : freshly transcoded clips from ucf101_org/, drawn
                        in stratified fashion across categories.

UCF101 has 101 action categories. The existing 1000-set uses ~10 clips
per category. To extend to 2048 we sample an additional ~12 clips per
category, oversampling to ~13 to absorb validation failures. We exclude
clip filenames already present in the existing set so the 0..999 prefix
is exactly preserved.

The script:

  1. Pre-seeds the new dataset dir with symlinks to existing videos and
     reads which raw filenames have been used.
  2. Samples additional raw clips from ucf101_org/<class>/, excluding
     already-used filenames, using `videos-per-category` candidates per
     class (default 13 = ~1313 candidates above 1000, sufficient
     oversampling for 2048).
  3. Transcodes each new clip to 832x480 mp4 via ffmpeg.
  4. Validates everything with `scripts/validate_dataset.py`.
  5. Trims `metadata.csv` to the first 2048 valid rows.

Usage:

  python scripts/build_ucf101_2048_dataset.py \
      --src-dir /scratch/wc3013/open-sora-v1.3-experiment/env_setup/download_ucf101/ucf101_org \
      --existing-dataset /scratch/wc3013/longcat-video-tta/datasets/ucf101_test_480p \
      --new-dataset /scratch/wc3013/longcat-video-tta/datasets/ucf101_2048_480p \
      --target-valid 2048
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_W, TARGET_H = 832, 480


def class_name_to_caption(class_name: str) -> str:
    """Match the helper in datasets/prepare_ucf101_subset.py."""
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", class_name)
    words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", words)
    return words.lower().strip()


def resize_video(src: Path, dst: Path) -> bool:
    """Transcode a single .avi to 832x480 mp4 via ffmpeg."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"scale={TARGET_W}:{TARGET_H}:force_original_aspect_ratio=disable",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-an",
        str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        last = r.stderr.strip().splitlines()[-3:] if r.stderr else []
        print(f"  ffmpeg error: {last}", flush=True)
    return r.returncode == 0


def _read_existing_meta(existing_dir: Path) -> List[dict]:
    meta_path = existing_dir / "metadata.csv"
    if not meta_path.exists():
        print(f"ERROR: no metadata.csv in {existing_dir}", file=sys.stderr)
        sys.exit(2)
    rows = []
    with open(meta_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _preseed_new_dataset(
    existing_dir: Path,
    new_dir: Path,
    existing_rows: List[dict],
    use_symlink: bool,
) -> List[dict]:
    """Symlink/copy the 1000 existing videos and return their metadata
    rows with consistent fieldnames."""
    new_videos = new_dir / "videos"
    new_videos.mkdir(parents=True, exist_ok=True)

    print(f"Pre-seeding new dataset with {len(existing_rows)} existing videos...")
    kept = []
    for row in existing_rows:
        fname = row.get("filename", "")
        if not fname:
            continue
        src = existing_dir / "videos" / fname
        if not src.exists():
            src_alt = existing_dir / fname
            if src_alt.exists():
                src = src_alt
            else:
                print(f"  WARN: source video missing for {fname}")
                continue
        dst = new_videos / fname
        if not dst.exists() and not dst.is_symlink():
            if use_symlink:
                os.symlink(os.path.relpath(src.resolve(), new_videos), dst)
            else:
                shutil.copy2(src, dst)
        kept.append({
            "filename": fname,
            "category": row.get("category", ""),
            "caption": row.get("caption", class_name_to_caption(
                row.get("category", "")
            )),
            "original": row.get("original", ""),
        })
    print(f"  Seeded {len(kept)} videos into {new_videos}")
    return kept


def _sample_additional_clips(
    src_dir: Path,
    used_originals: set,
    used_categories_counts: dict,
    target_extra: int,
    videos_per_category: int,
    seed: int,
) -> List[dict]:
    """Sample additional .avi files from ucf101_org/, excluding ones we
    already have. Returns a stratified list of dicts with src_path,
    category, caption."""
    rng = random.Random(seed)
    categories = sorted([d.name for d in src_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(categories)} categories in {src_dir}")

    # First pass: pick `videos_per_category` new clips per category.
    selected: List[dict] = []
    for cat in categories:
        cat_dir = src_dir / cat
        avis = sorted(cat_dir.glob("*.avi"))
        if not avis:
            continue
        candidates = [v for v in avis if v.name not in used_originals]
        if not candidates:
            continue
        rng.shuffle(candidates)
        for v in candidates[:videos_per_category]:
            selected.append({
                "src_path": v,
                "category": cat,
                "caption": class_name_to_caption(cat),
            })

    # Second pass: if still short, keep filling from remaining pools.
    rng.shuffle(selected)
    if len(selected) < target_extra:
        print(f"  WARNING: per-category pool yielded only {len(selected)} "
              f"new clips; need {target_extra}. Filling from leftover pool.")
        leftover: List[dict] = []
        seen_so_far = {entry["src_path"].name for entry in selected}
        for cat in categories:
            cat_dir = src_dir / cat
            for v in cat_dir.glob("*.avi"):
                if v.name in used_originals or v.name in seen_so_far:
                    continue
                leftover.append({
                    "src_path": v,
                    "category": cat,
                    "caption": class_name_to_caption(cat),
                })
        rng.shuffle(leftover)
        needed = target_extra - len(selected)
        selected.extend(leftover[:needed])

    return selected


def _transcode_clips(
    selected: List[dict],
    new_videos: Path,
    start_index: int,
) -> List[dict]:
    """Transcode selected .avi sources to mp4 in `new_videos`, returning
    metadata rows for those that succeeded."""
    out_rows: List[dict] = []
    ok = fail = 0
    print(f"\nTranscoding {len(selected)} new clips (starting index "
          f"{start_index})...")
    for i, entry in enumerate(selected):
        idx = start_index + i
        out_name = f"ucf101_{idx:04d}.mp4"
        out_path = new_videos / out_name
        if out_path.exists() and out_path.stat().st_size > 1000:
            ok += 1
            out_rows.append({
                "filename": out_name,
                "category": entry["category"],
                "caption": entry["caption"],
                "original": entry["src_path"].name,
            })
            continue
        if resize_video(entry["src_path"], out_path):
            ok += 1
            out_rows.append({
                "filename": out_name,
                "category": entry["category"],
                "caption": entry["caption"],
                "original": entry["src_path"].name,
            })
        else:
            fail += 1
            print(f"  FAILED {entry['src_path'].name}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(selected)}]  ok={ok} fail={fail}",
                  flush=True)
    print(f"  Transcoded: ok={ok}  fail={fail}")
    return out_rows


def _write_metadata(new_dir: Path, rows: List[dict]) -> Path:
    csv_path = new_dir / "metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "category", "caption", "original"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Wrote {len(rows)} rows -> {csv_path}")
    return csv_path


def _run_validate(new_dir: Path, required_valid: int, min_frames: int) -> int:
    validate_script = REPO_ROOT / "scripts" / "validate_dataset.py"
    cmd = [
        sys.executable, str(validate_script),
        "--dataset-dir", str(new_dir),
        "--required-valid", str(required_valid),
        "--min-frames", str(min_frames),
        "--write-valid-subset", "valid_subset.csv",
    ]
    print()
    print("Invoking validator:")
    print("  " + " ".join(cmd))
    return int(subprocess.run(cmd).returncode)


def _finalize_to_n(new_dir: Path, target_valid: int) -> int:
    valid_path = new_dir / "valid_subset.csv"
    if not valid_path.exists():
        print(f"ERROR: validator did not produce valid_subset.csv at "
              f"{valid_path}", file=sys.stderr)
        return -1
    with open(valid_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if len(rows) < target_valid:
        print(f"WARN: only {len(rows)} valid rows; target was {target_valid}")
        kept = rows
    else:
        kept = rows[:target_valid]
    final_path = new_dir / "metadata.csv"
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    print(f"\nFinal metadata.csv written ({len(kept)} rows) -> {final_path}")
    return len(kept)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src-dir", type=str, required=True,
                        help="Path to ucf101_org/ with category subdirectories")
    parser.add_argument("--existing-dataset", type=str, required=True,
                        help="Path to the existing ucf101_*_480p directory "
                             "whose first 1000 videos will be inherited")
    parser.add_argument("--new-dataset", type=str, required=True,
                        help="Path to the new ucf101_2048_480p directory")
    parser.add_argument("--target-valid", type=int, default=2048)
    parser.add_argument("--videos-per-category", type=int, default=13,
                        help="New clips to sample per category (101 cats * 13 "
                             "= 1313 candidates above the existing 1000; "
                             "enough oversampling for ~10%% failure rate)")
    parser.add_argument("--copy-instead-of-symlink", action="store_true")
    parser.add_argument("--seed", type=int, default=43,
                        help="Different from the 1000-set seed (42) so the "
                             "newly drawn clips are disjoint by construction")
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--skip-transcode", action="store_true",
                        help="Skip transcoding step (use after a partial run)")
    parser.add_argument("--skip-seeding", action="store_true")
    args = parser.parse_args()

    src_dir = Path(args.src_dir).resolve()
    existing_dir = Path(args.existing_dataset).resolve()
    new_dir = Path(args.new_dataset).resolve()
    if not src_dir.is_dir():
        print(f"ERROR: src dir {src_dir} not found", file=sys.stderr)
        return 2
    if not existing_dir.is_dir():
        print(f"ERROR: existing dataset {existing_dir} not found",
              file=sys.stderr)
        return 2
    if shutil.which("ffmpeg") is None:
        print("ERROR: ffmpeg not in PATH", file=sys.stderr)
        return 2

    new_dir.mkdir(parents=True, exist_ok=True)
    new_videos = new_dir / "videos"
    new_videos.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Build UCF-101 2048-video dataset")
    print("=" * 70)
    print(f"  src        : {src_dir}")
    print(f"  existing   : {existing_dir}")
    print(f"  new        : {new_dir}")
    print(f"  target     : {args.target_valid} valid")
    print(f"  per-cat new: {args.videos_per_category}")
    print()

    t0 = time.time()

    existing_rows = _read_existing_meta(existing_dir)
    if args.skip_seeding:
        seeded_rows = [
            {
                "filename": r["filename"],
                "category": r.get("category", ""),
                "caption": r.get("caption", ""),
                "original": r.get("original", ""),
            }
            for r in existing_rows
        ]
    else:
        seeded_rows = _preseed_new_dataset(
            existing_dir=existing_dir,
            new_dir=new_dir,
            existing_rows=existing_rows,
            use_symlink=not args.copy_instead_of_symlink,
        )

    used_originals = {r["original"] for r in seeded_rows if r["original"]}
    used_categories_counts: dict = {}
    for r in seeded_rows:
        c = r.get("category", "")
        used_categories_counts[c] = used_categories_counts.get(c, 0) + 1

    target_extra = max(0, args.target_valid - len(seeded_rows)) + 200
    print(f"\nSampling additional clips (target_extra={target_extra})...")
    selected_new = _sample_additional_clips(
        src_dir=src_dir,
        used_originals=used_originals,
        used_categories_counts=used_categories_counts,
        target_extra=target_extra,
        videos_per_category=args.videos_per_category,
        seed=args.seed,
    )
    print(f"  Selected {len(selected_new)} new clips")

    if not args.skip_transcode:
        new_rows = _transcode_clips(
            selected=selected_new,
            new_videos=new_videos,
            start_index=len(seeded_rows),
        )
    else:
        print("Skipping transcode; reading existing mp4s in new_videos/")
        existing_in_new = sorted(new_videos.glob("ucf101_*.mp4"))
        new_rows = []
        for vp in existing_in_new:
            name = vp.name
            if name in {r["filename"] for r in seeded_rows}:
                continue
            new_rows.append({
                "filename": name, "category": "", "caption": "", "original": ""
            })

    all_rows = seeded_rows + new_rows
    _write_metadata(new_dir, all_rows)

    rc = _run_validate(new_dir, args.target_valid, args.min_frames)
    if rc != 0:
        print(f"\nVALIDATION FAILED. Re-run with --videos-per-category bigger.",
              file=sys.stderr)
        return rc

    n_final = _finalize_to_n(new_dir, args.target_valid)
    if n_final < args.target_valid:
        return 1

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"DONE in {elapsed/60:.1f} min")
    print("=" * 70)
    print(f"  dataset    : {new_dir}")
    print(f"  videos     : {new_dir / 'videos'}")
    print(f"  metadata   : {new_dir / 'metadata.csv'} ({n_final} rows)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
