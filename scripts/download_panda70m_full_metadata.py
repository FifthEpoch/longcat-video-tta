#!/usr/bin/env python3
"""
Download Panda-70M training-set metadata CSV from Google Drive.

Why we go through Google Drive:
  Panda-70M's full segment-level metadata is NOT mirrored on HuggingFace.
  The official distribution (per snap-research/Panda-70M README) is
  Google Drive links for each split. We use gdown -- the same tool the
  earlier panda_pool_10k build used to fetch panda70m_training_2m.csv.

Splits available (from snap-research/Panda-70M README):

  +--------+-----------+----------------------+-------------------+
  | Split  | Drive sz  | # Source Videos      | # Segments        |
  +--------+-----------+----------------------+-------------------+
  | full   | 2.73 GB   | 3,779,763            | 70,723,513        |
  | 10m    | 504 MB    | 3,755,240            | 10,473,922        |
  | 2m     | 118 MB    | 800,000              |  2,400,000        |
  | val    | 1.2 MB    | 2,000                |     6,000         |
  | test   | 1.2 MB    | 2,000                |     6,000         |
  +--------+-----------+----------------------+-------------------+

Average segments/video:
  full -> ~18.7   10m -> ~2.8   2m -> ~3.0  (the 2m subset already
  on disk caps each video at 3 curated segments).

For 2048 source videos in our pool:
  full -> ~38K segments BEFORE filtering  ->  ~25-30K after the
                                              `desirable_filtering`
                                              quality gate.
  10m  -> ~5.7K
  2m   -> ~6.1K  (3,302 actually emitted in Phase 2A)

Usage:
  python scripts/download_panda70m_full_metadata.py \
      --out-dir /scratch/wc3013/longcat-video-tta/datasets/panda_metadata_full \
      --split full

Environment fallbacks (so the sbatch wrapper can pass them via --export):
  SPLIT     : --split    (full / 10m / 2m / val / test)
  OUT_DIR   : --out-dir
  FORCE     : 1 = re-download even if file exists
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


# Google Drive file IDs from snap-research/Panda-70M README.md
PANDA_DRIVE_FILES = {
    "full": (
        "1pbh8W3qgst9CD7nlPhsH9wmUSWjQlGdW",
        "panda70m_training_full.csv",
        2.73e9,
        70_723_513,
    ),
    "10m": (
        "1LLOFeYw9nZzjT5aA1Wj4oGi5yHUzwSk5",
        "panda70m_training_10m.csv",
        504e6,
        10_473_922,
    ),
    "2m": (
        "1k7NzU6wVNZYl6NxOhLXE7Hz7OrpzNLgB",
        "panda70m_training_2m.csv",
        118e6,
        2_400_000,
    ),
    "val": (
        "1uHR5iXS3Sftzw6AwEhyZ9RefipNzBAzt",
        "panda70m_validation.csv",
        1.2e6,
        6_000,
    ),
    "test": (
        "1BZ9L-157Au1TwmkwlJV8nZQvSRLIiFhq",
        "panda70m_testing.csv",
        1.2e6,
        6_000,
    ),
}


def _resolve_arg(arg_value, env_name, default=None):
    if arg_value is not None:
        return arg_value
    return os.environ.get(env_name, default)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        choices=list(PANDA_DRIVE_FILES.keys()),
        default=None,
        help="Panda-70M split to download (env: SPLIT, default 'full').",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Destination directory (env: OUT_DIR).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if output file already exists.",
    )
    args = parser.parse_args()

    split = _resolve_arg(args.split, "SPLIT", "full")
    out_dir_str = _resolve_arg(args.out_dir, "OUT_DIR")
    if out_dir_str is None:
        print("ERROR: --out-dir or OUT_DIR env var required.", file=sys.stderr)
        return 2
    if split not in PANDA_DRIVE_FILES:
        print(f"ERROR: unknown split {split!r}; valid: "
              f"{list(PANDA_DRIVE_FILES.keys())}", file=sys.stderr)
        return 2

    force = args.force or os.environ.get("FORCE", "0") == "1"

    out_dir = Path(out_dir_str).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    file_id, filename, expected_bytes, expected_rows = PANDA_DRIVE_FILES[split]
    out_path = out_dir / filename

    print("=" * 78)
    print("Panda-70M metadata download")
    print("=" * 78)
    print(f"  split            : {split}")
    print(f"  Google Drive ID  : {file_id}")
    print(f"  destination      : {out_path}")
    print(f"  expected size    : {expected_bytes / 1e9:.2f} GB")
    print(f"  expected segments: {expected_rows:,}")
    print(f"  force            : {force}")
    print("=" * 78)
    print()

    if out_path.exists() and not force:
        actual_size = out_path.stat().st_size
        print(f"  Destination file already exists "
              f"({actual_size / 1e9:.2f} GB on disk).")
        if actual_size > 0.9 * expected_bytes:
            print(f"  Size matches (>=90% of expected). Skipping download.")
            print(f"  Use --force / FORCE=1 to re-download.")
            _quick_row_count(out_path)
            return 0
        else:
            print(f"  WARN: file is much smaller than expected; "
                  f"re-downloading.")
            out_path.unlink()

    # Lazy import so the help-text path doesn't require gdown installed.
    print("Importing gdown ...")
    try:
        import gdown  # type: ignore
    except ImportError:
        print("ERROR: gdown not installed. The sbatch wrapper "
              "should `pip install gdown --quiet` before running this "
              "script.", file=sys.stderr)
        return 2

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Starting download from {url}")
    print("(gdown handles the >100MB confirmation token automatically.)")
    print()

    t0 = time.time()
    try:
        gdown.download(url, str(out_path), quiet=False, fuzzy=True)
    except Exception as exc:
        print(f"\nERROR: gdown.download raised: {exc}", file=sys.stderr)
        return 1

    elapsed = time.time() - t0
    if not out_path.exists():
        print(f"ERROR: gdown finished without creating {out_path}",
              file=sys.stderr)
        return 1

    actual_size = out_path.stat().st_size
    print()
    print("=" * 78)
    print(f"DOWNLOAD COMPLETE in {elapsed / 60:.1f} min")
    print("=" * 78)
    print(f"  file        : {out_path}")
    print(f"  size        : {actual_size / 1e9:.3f} GB "
          f"(expected ~{expected_bytes / 1e9:.2f} GB)")
    if actual_size < 0.5 * expected_bytes:
        print(f"  WARN: actual size much smaller than expected -- "
              f"download may have been truncated.")

    _quick_row_count(out_path)

    print()
    print("=" * 78)
    print("Next step: rebuild the segment pool with this metadata")
    print("=" * 78)
    print(f"  sbatch --account=torch_pr_36_mren \\")
    print(f"      --export=ALL,SOURCE_METADATA={out_path} \\")
    print(f"      datasets/build_panda_segment_pool.sbatch")
    print()
    print("  (resume support means previously-cut segments are skipped;")
    print("   only NEW segments from the expanded metadata get encoded.)")
    print("=" * 78)
    return 0


def _quick_row_count(path: Path) -> None:
    """Print row count + first non-header line for a quick sanity check."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            header = f.readline().rstrip("\n")
            n_rows = 1  # header line
            sample_row = None
            for line in f:
                n_rows += 1
                if sample_row is None:
                    sample_row = line.rstrip("\n")
                if n_rows > 5_000_000:
                    print(f"  rows: >{n_rows:,} (stopped counting; file is large)")
                    print(f"  header: {header[:120]}")
                    return
        print(f"  rows: {n_rows:,} (incl. header)")
        print(f"  header: {header[:120]}")
        if sample_row:
            print(f"  sample: {sample_row[:120]}...")
    except OSError as exc:
        print(f"  WARN: could not read {path} for row count: {exc}")


if __name__ == "__main__":
    sys.exit(main())
