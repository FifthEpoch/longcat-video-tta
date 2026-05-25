#!/usr/bin/env python3
"""
Build the Panda-70M 2048-video dataset for the headline 5-method comparison.

Strategy: this is an EXTENSION of the existing 1000-video dataset, not a
fresh resample. The new dataset contains:

  - indices 0..999    : symlinks/copies to the existing panda_1000_480p videos
                        (preserving filename, caption, category, videoID, etc.)
  - indices 1000..N   : freshly downloaded clips with new videoIDs.

Why extension instead of fresh sample:

  - The first-1000 prefix of the 2048 dataset is bit-identical to the
    existing 1000-set, so prior experiments at N=1000 are reproducible as
    a strict prefix.
  - Saves ~1000 yt-dlp downloads and the associated wall time.
  - Deterministic resume: the existing `download_panda70m_subset.py`
    already supports `--resume`, which reads manifest.jsonl and skips
    already-downloaded videoIDs. Pre-seeding the new directory with the
    1000-set's manifest is sufficient.

The script:

  1. Pre-seeds the new dataset dir with symlinks to the 1000-set videos
     and a copy of the 1000-set's manifest.jsonl.
  2. Invokes `datasets/download_panda70m_subset.py --resume` with a
     larger target (default 2300) to oversample for validation failures.
  3. Runs `scripts/validate_dataset.py` on the resulting directory.
  4. If the validator finds >= 2048 valid videos, trims metadata.csv to
     the first 2048 valid rows (preserving index order). Otherwise reports
     the shortfall and exits non-zero so the user can re-run with more
     oversampling.

Usage:

  # Run from the cluster project root after `git pull`:
  python scripts/build_panda_2048_dataset.py \
      --existing-dataset /scratch/wc3013/longcat-video-tta/datasets/panda_1000_480p \
      --new-dataset    /scratch/wc3013/longcat-video-tta/datasets/panda_2048_480p \
      --target-valid 2048 \
      --oversample-to 2300

  # Resume after a partial run (skips redownload):
  python scripts/build_panda_2048_dataset.py \
      --existing-dataset .../panda_1000_480p \
      --new-dataset    .../panda_2048_480p \
      --target-valid 2048 --oversample-to 2300 --skip-download

This is a CPU + network job. Submit via the standard panda download
sbatch (datasets/download_panda70m.sbatch) or run directly on a login
node if the network is permitted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read_existing_manifest(existing_dir: Path) -> List[dict]:
    manifest_path = existing_dir / "manifest.jsonl"
    if not manifest_path.exists():
        print(f"  WARNING: no manifest.jsonl in {existing_dir}; will rebuild "
              f"from metadata.csv (videoID may be missing).")
        meta_path = existing_dir / "metadata.csv"
        if not meta_path.exists():
            print(f"ERROR: neither manifest.jsonl nor metadata.csv exist in "
                  f"{existing_dir}", file=sys.stderr)
            sys.exit(2)
        entries = []
        with open(meta_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append(dict(row))
        return entries

    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def _preseed_new_dataset(
    existing_dir: Path,
    new_dir: Path,
    existing_entries: List[dict],
    use_symlink: bool,
) -> None:
    new_videos = new_dir / "videos"
    new_videos.mkdir(parents=True, exist_ok=True)

    print(f"Pre-seeding new dataset with {len(existing_entries)} existing videos...")
    seeded = 0
    for entry in existing_entries:
        fname = entry.get("filename", "")
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
        if dst.exists() or dst.is_symlink():
            seeded += 1
            continue
        if use_symlink:
            os.symlink(os.path.relpath(src.resolve(), new_videos), dst)
        else:
            shutil.copy2(src, dst)
        seeded += 1
    print(f"  Seeded {seeded} videos into {new_videos}")

    new_manifest = new_dir / "manifest.jsonl"
    if not new_manifest.exists():
        with open(new_manifest, "w") as f_out:
            for entry in existing_entries:
                if "videoID" not in entry:
                    continue
                if "filename" not in entry:
                    continue
                payload = {
                    "index": int(entry.get("index", 0))
                    if str(entry.get("index", "")).isdigit() else 0,
                    "filename": entry["filename"],
                    "videoID": entry["videoID"],
                    "caption": entry.get("caption", ""),
                    "category": entry.get("category", ""),
                    "duration": entry.get("duration", ""),
                    "fps": entry.get("fps", ""),
                    "frames": entry.get("frames", ""),
                    "width": entry.get("width", ""),
                    "height": entry.get("height", ""),
                    "path": str(new_videos / entry["filename"]),
                }
                f_out.write(json.dumps(payload) + "\n")
        print(f"  Wrote pre-seed manifest -> {new_manifest}")


def _run_download(
    new_dir: Path,
    target_total: int,
    cookies_file: Optional[str],
    candidate_multiplier: int,
    download_timeout: int,
    hf_max_rows: int,
    seed: int,
) -> int:
    """Run the existing download_panda70m_subset.py with --resume to fill
    up the new dataset directory."""
    download_script = REPO_ROOT / "datasets" / "download_panda70m_subset.py"
    if not download_script.exists():
        print(f"ERROR: download script not found at {download_script}",
              file=sys.stderr)
        return 2

    cmd = [
        sys.executable, str(download_script),
        "--out-dir", str(new_dir),
        "--num-videos", str(target_total),
        "--seed", str(seed),
        "--candidate-multiplier", str(candidate_multiplier),
        "--download-timeout", str(download_timeout),
        "--hf-max-rows", str(hf_max_rows),
        "--resume",
    ]
    if cookies_file:
        cmd += ["--cookies-file", cookies_file]

    print()
    print("Invoking download script:")
    print("  " + " ".join(cmd))
    print()

    result = subprocess.run(cmd)
    return int(result.returncode)


def _run_validate(
    new_dir: Path,
    required_valid: int,
    min_frames: int,
) -> int:
    validate_script = REPO_ROOT / "scripts" / "validate_dataset.py"
    cmd = [
        sys.executable, str(validate_script),
        "--dataset-dir", str(new_dir),
        "--required-valid", str(required_valid),
        "--min-frames", str(min_frames),
        "--write-valid-subset", "valid_subset.csv",
        "--no-require-category",  # Panda manifests sometimes drop category.
    ]
    print()
    print("Invoking validator:")
    print("  " + " ".join(cmd))
    result = subprocess.run(cmd)
    return int(result.returncode)


def _finalize_metadata_to_n(
    new_dir: Path,
    target_valid: int,
) -> int:
    """Read valid_subset.csv produced by the validator, keep the first N
    rows, and overwrite metadata.csv with the trimmed list. Returns the
    final video count."""
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
        print(f"WARN: only {len(rows)} valid rows available, fewer than "
              f"target {target_valid}", file=sys.stderr)
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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--existing-dataset", type=str, required=True,
                        help="Path to existing panda_1000_480p directory")
    parser.add_argument("--new-dataset", type=str, required=True,
                        help="Path to new panda_2048_480p directory to build")
    parser.add_argument("--target-valid", type=int, default=2048,
                        help="Required number of valid videos in the final "
                             "dataset (default 2048)")
    parser.add_argument("--oversample-to", type=int, default=2300,
                        help="Total number of videos to download before "
                             "validation (default 2300 = ~12%% oversampling)")
    parser.add_argument("--copy-instead-of-symlink", action="store_true",
                        help="Copy existing videos instead of symlinking "
                             "(safer if the existing dir might be deleted, "
                             "but uses 2x disk)")
    parser.add_argument("--cookies-file", type=str, default=None,
                        help="Pass-through to download_panda70m_subset.py")
    parser.add_argument("--candidate-multiplier", type=int, default=15,
                        help="Pass-through (default 15)")
    parser.add_argument("--download-timeout", type=int, default=120,
                        help="Pass-through (default 120s/video)")
    parser.add_argument("--hf-max-rows", type=int, default=80_000,
                        help="Increase metadata pool because we need more "
                             "candidates than the 1000-set required")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default 42, matches "
                             "the original 1000-set seed for determinism)")
    parser.add_argument("--min-frames", type=int, default=50,
                        help="Validator min-frames threshold (default 50)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip the download step (use when resuming from "
                             "a previously-completed download to re-validate "
                             "or re-finalize)")
    parser.add_argument("--skip-seeding", action="store_true",
                        help="Skip pre-seeding from the existing dataset "
                             "(use only if you have already seeded)")
    args = parser.parse_args()

    existing_dir = Path(args.existing_dataset).resolve()
    new_dir = Path(args.new_dataset).resolve()
    if not existing_dir.is_dir():
        print(f"ERROR: existing dataset not found at {existing_dir}",
              file=sys.stderr)
        return 2

    new_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Build Panda-70M 2048-video dataset")
    print("=" * 70)
    print(f"  existing  : {existing_dir}")
    print(f"  new       : {new_dir}")
    print(f"  target    : {args.target_valid} valid")
    print(f"  oversample: {args.oversample_to} total")
    print()

    t0 = time.time()

    if not args.skip_seeding:
        existing_entries = _read_existing_manifest(existing_dir)
        _preseed_new_dataset(
            existing_dir=existing_dir,
            new_dir=new_dir,
            existing_entries=existing_entries,
            use_symlink=not args.copy_instead_of_symlink,
        )

    if not args.skip_download:
        rc = _run_download(
            new_dir=new_dir,
            target_total=args.oversample_to,
            cookies_file=args.cookies_file,
            candidate_multiplier=args.candidate_multiplier,
            download_timeout=args.download_timeout,
            hf_max_rows=args.hf_max_rows,
            seed=args.seed,
        )
        if rc != 0:
            print(f"\nWARN: download exited with code {rc}. Continuing to "
                  f"validate what we have.", file=sys.stderr)

    rc = _run_validate(
        new_dir=new_dir,
        required_valid=args.target_valid,
        min_frames=args.min_frames,
    )
    if rc != 0:
        print(f"\nVALIDATION FAILED. Fewer than {args.target_valid} valid "
              f"videos. Re-run with --skip-seeding --oversample-to <larger>",
              file=sys.stderr)
        return rc

    n_final = _finalize_metadata_to_n(new_dir, args.target_valid)
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
    print(f"  report     : {new_dir / 'validation_report.json'}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
