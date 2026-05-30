#!/usr/bin/env python3
"""
Build a head-trimmed Panda-70M retrieval pool for similarity- and random-
augmented TTA.

Why this exists separately from ``build_panda_2048_dataset.py``:

  - The 2048-video EVAL set keeps full-length Panda clips (avg ~5 min,
    73 MB / clip) because eval treats each clip as a single sample.
  - The retrieval POOL only needs ~5 s of usable content per entry (TTA
    reads the first 48 frames, ~1.92 s at 25 fps), so we filter HF
    candidates to short clips and let yt-dlp's ``--download-sections``
    pull only that range. End result: ~3 MB / clip, ~30 GB total at 10 K
    entries instead of ~730 GB at full-length.

This script is a thin orchestrator. The heavy lifting (HF metadata stream,
yt-dlp + retry logic, ffprobe validation) lives in
``datasets/download_panda70m_subset.py`` -- the same script that built
``panda_2048_480p``. We just call it with retrieval-pool-appropriate
defaults and follow up with ``scripts/validate_dataset.py`` + a final
trim of metadata.csv to ``--target-valid`` rows.

Pipeline:
  1. Invoke ``datasets/download_panda70m_subset.py`` with:
       --num-videos <oversample-to>
       --min-duration <window-min>
       --max-duration <window-max>
       --candidate-multiplier <mult>
       --hf-max-rows <rows>
       --resume   (always; safe-by-default since output_dir is fresh)
  2. Run ``scripts/validate_dataset.py`` (>= --min-frames per clip).
  3. Truncate metadata.csv to the first ``--target-valid`` valid rows
     (preserving the manifest's index order).

Resumable:
  yt-dlp success rate on Panda-70M URLs is ~30-40%, so building 10 K
  successful clips takes 50-80 h of serial yt-dlp time, which exceeds
  the cluster's 24 h SLURM cap. The companion sbatch wrapper sets
  ``--time=20:00:00`` and ``--resume`` so the same sbatch can be
  re-submitted multiple times until the manifest reaches
  ``--target-valid``. Each invocation skips already-downloaded video IDs.

Usage:

  python scripts/build_panda_retrieval_pool.py \\
      --new-dataset /scratch/wc3013/longcat-video-tta/datasets/panda_pool_10k \\
      --target-valid 10000 \\
      --oversample-to 12000

  # Resume after a 20-h SLURM timeout (no flag needed; the underlying
  # download script always reads/writes manifest.jsonl):
  sbatch --account=torch_pr_36_mren \\
      datasets/build_panda_retrieval_pool.sbatch

  # Skip download and just (re)validate + finalize:
  python scripts/build_panda_retrieval_pool.py ... --skip-download

CPU + network only -- no GPU. Submit via
``datasets/build_panda_retrieval_pool.sbatch``.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[1]


def _count_manifest_entries(new_dir: Path) -> int:
    manifest = new_dir / "manifest.jsonl"
    if not manifest.exists():
        return 0
    n = 0
    with open(manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                n += 1
    return n


def _run_download(
    out_dir: Path,
    num_videos: int,
    candidate_multiplier: int,
    hf_max_rows: int,
    min_duration: float,
    max_duration: float,
    download_timeout: int,
    cookies_file: Optional[str],
    seed: int,
    resume: bool,
    min_frames: int,
) -> int:
    download_script = REPO_ROOT / "datasets" / "download_panda70m_subset.py"
    if not download_script.exists():
        print(f"ERROR: {download_script} does not exist", file=sys.stderr)
        return 2

    cmd = [
        sys.executable, str(download_script),
        "--out-dir", str(out_dir),
        "--num-videos", str(num_videos),
        "--seed", str(seed),
        "--min-duration", f"{min_duration:.3f}",
        "--max-duration", f"{max_duration:.3f}",
        "--min-frames", str(min_frames),
        "--hf-max-rows", str(hf_max_rows),
        "--candidate-multiplier", str(candidate_multiplier),
        "--download-timeout", str(download_timeout),
    ]
    if cookies_file:
        cmd += ["--cookies-file", cookies_file]
    if resume:
        cmd.append("--resume")

    print("=" * 70)
    print("Invoking download_panda70m_subset.py")
    print("=" * 70)
    for tok in cmd:
        print(f"  {tok}" if tok.startswith("--") else f"    {tok}")
    print()
    return int(subprocess.run(cmd).returncode)


def _run_validate(new_dir: Path, required_valid: int, min_frames: int) -> int:
    validate_script = REPO_ROOT / "scripts" / "validate_dataset.py"
    if not validate_script.exists():
        print(f"  validate_dataset.py not found at {validate_script}; "
              f"skipping validation step.")
        return 0
    cmd = [
        sys.executable, str(validate_script),
        "--dataset-dir", str(new_dir),
        "--required-valid", str(required_valid),
        "--min-frames", str(min_frames),
        "--write-valid-subset", "valid_subset.csv",
    ]
    print()
    print("=" * 70)
    print("Invoking validator")
    print("=" * 70)
    print("  " + " ".join(cmd))
    print()
    return int(subprocess.run(cmd).returncode)


def _finalize_metadata_to_n(new_dir: Path, target_valid: int) -> int:
    """Trim metadata.csv to the first ``target_valid`` rows from
    valid_subset.csv. Returns the row count actually written."""
    valid_path = new_dir / "valid_subset.csv"
    meta_path = new_dir / "metadata.csv"
    if not valid_path.exists():
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                n = sum(1 for _ in csv.DictReader(f))
            print(f"\n(No valid_subset.csv; keeping metadata.csv with "
                  f"{n} rows -- did the validator run?)")
            return n
        return -1

    with open(valid_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if len(rows) < target_valid:
        print(f"\nWARNING: only {len(rows)} valid rows; target "
              f"{target_valid} not reached. Re-submit the sbatch (it "
              f"resumes); the underlying download script will skip "
              f"already-downloaded videoIDs and continue sampling new "
              f"candidates.")
        kept = rows
    else:
        kept = rows[:target_valid]
    with open(meta_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    print(f"\nFinal metadata.csv written ({len(kept)} rows) -> {meta_path}")
    return len(kept)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--new-dataset", type=str, required=True,
        help="Output dir for the pool, e.g. .../datasets/panda_pool_10k",
    )
    parser.add_argument(
        "--target-valid", type=int, default=10000,
        help="Final pool size after validation+trim (default 10000). "
             "Increase to 25000 if you have ~80 h of cumulative SLURM "
             "wall-time (~4 sbatch resubmissions of the 20 h wrapper).",
    )
    parser.add_argument(
        "--oversample-to", type=int, default=12000,
        help="Pass-through to download_panda70m_subset.py --num-videos. "
             "Setting this slightly higher than --target-valid gives "
             "headroom for the validator filtering out a small fraction "
             "of corrupt downloads. Default 12000.",
    )
    parser.add_argument(
        "--candidate-multiplier", type=int, default=3,
        help="The download script samples num_videos * multiplier "
             "candidate URLs and iterates until num_videos succeed. "
             "yt-dlp success rate is ~30-40 percent. Default 3 (down "
             "from the script's default 15) because we apply tight "
             "duration filters that reject many fewer post-sample.",
    )
    parser.add_argument(
        "--hf-max-rows", type=int, default=300_000,
        help="Cap on rows pulled from the HF Panda-70M stream. Higher "
             "values give more candidates after duration filtering, at "
             "the cost of a longer initial metadata pass. Default "
             "300000 (covers ~3-5 percent of Panda-70M).",
    )
    parser.add_argument(
        "--min-duration", type=float, default=2.5,
        help="Drop HF candidates whose Panda-70M clip range is shorter "
             "than this (default 2.5 s).",
    )
    parser.add_argument(
        "--max-duration", type=float, default=6.0,
        help="Drop HF candidates whose Panda-70M clip range is longer "
             "than this. THIS IS THE HEAD-TRIM MECHANISM: a smaller "
             "max keeps disk usage low but reduces the candidate pool. "
             "Default 6.0 s (TTA only reads first 48 frames ~ 1.92 s, "
             "so 6 s is generous margin).",
    )
    parser.add_argument(
        "--min-frames", type=int, default=48,
        help="Validator min-frames threshold. Default 48 (matches "
             "tta_total_frames).",
    )
    parser.add_argument(
        "--download-timeout", type=int, default=120,
        help="Per-clip yt-dlp timeout in seconds. Default 120.",
    )
    parser.add_argument(
        "--cookies-file", type=str, default=None,
        help="Optional cookies.txt for YouTube (helps with age-gated / "
             "regionally blocked clips). Pass-through to "
             "download_panda70m_subset.py; can also be set via the "
             "COOKIES_FILE env var read by that script.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for stratified candidate sampling (default 42).",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip the download step (use after a partial run when only "
             "validation+finalization need rerunning).",
    )
    parser.add_argument(
        "--skip-validate", action="store_true",
        help="Skip the validator (and the metadata.csv trim).",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Don't pass --resume to the download script. Default is to "
             "always resume so re-submissions of the sbatch wrapper "
             "make forward progress.",
    )
    args = parser.parse_args()

    new_dir = Path(args.new_dataset).resolve()
    new_dir.mkdir(parents=True, exist_ok=True)

    est_disk_gb = args.target_valid * 3.0 / 1024.0
    pre_existing = _count_manifest_entries(new_dir)

    print("=" * 70)
    print("Build Panda-70M retrieval pool (head-trimmed)")
    print("=" * 70)
    print(f"  new dir              : {new_dir}")
    print(f"  target valid         : {args.target_valid}")
    print(f"  oversample-to        : {args.oversample_to}")
    print(f"  candidate-multiplier : {args.candidate_multiplier}")
    print(f"  hf-max-rows          : {args.hf_max_rows}")
    print(f"  duration window      : [{args.min_duration}, "
          f"{args.max_duration}] s")
    print(f"  est. disk usage      : ~{est_disk_gb:.1f} GB "
          f"(at ~3 MB / clip)")
    print(f"  manifest pre-run     : {pre_existing} entries already present")
    print(f"  resume               : {'no' if args.no_resume else 'yes'}")
    print(f"  skip download / val. : {args.skip_download} / "
          f"{args.skip_validate}")
    print()

    t0 = time.time()
    if not args.skip_download:
        rc = _run_download(
            out_dir=new_dir,
            num_videos=args.oversample_to,
            candidate_multiplier=args.candidate_multiplier,
            hf_max_rows=args.hf_max_rows,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            download_timeout=args.download_timeout,
            cookies_file=args.cookies_file,
            seed=args.seed,
            resume=not args.no_resume,
            min_frames=args.min_frames,
        )
        if rc != 0:
            print(f"\nDownload script exited with rc={rc}. Continuing to "
                  f"the validator so partial progress can still be "
                  f"finalized.", file=sys.stderr)
    else:
        print("(--skip-download set; reusing existing manifest.jsonl.)")

    if not args.skip_validate:
        _run_validate(
            new_dir=new_dir,
            required_valid=args.target_valid,
            min_frames=args.min_frames,
        )
    else:
        print("(--skip-validate set; not running validator.)")

    n_final = _finalize_metadata_to_n(new_dir, args.target_valid)

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"DONE in {elapsed/60:.1f} min")
    print("=" * 70)
    print(f"  pool       : {new_dir}")
    print(f"  metadata   : {new_dir / 'metadata.csv'} ({n_final} rows)")
    if 0 < n_final < args.target_valid:
        print(f"  ** Pool short of target ({n_final} / {args.target_valid}). "
              f"Re-submit the sbatch to continue downloading.")
    print()
    print("Next step: pre-compute caption embeddings.")
    print(f"  sbatch --account=torch_pr_36_mren \\")
    print(f"      --export=ALL,POOL_DIR={new_dir} \\")
    print(f"      delta_experiment/sbatch/precompute_pool_embeddings.sbatch")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
