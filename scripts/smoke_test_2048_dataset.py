#!/usr/bin/env python3
"""
Pre-flight smoke test for a freshly-built 2048-video dataset.

Runs NoTTA (delta_steps=0) on the first N videos (default 20) of a
dataset, with the exact frame geometry the 2048v sweep will use. The
goal is to catch dataset issues BEFORE submitting the 5-method ×
2-dataset batch: missing metadata fields, decode failures the validator
didn't catch, caption-guard failures, etc.

Acceptance criteria (the script exits 0 only if all are met):

  - All N videos produce a per-video entry in summary.json.
  - No video reports an exception in summary["errors"].
  - All N videos have non-null PSNR/SSIM/LPIPS values.
  - The pipeline runs end-to-end without crashing.

Submit via sbatch on the cluster:

  sbatch --account=torch_pr_36_mren \
         --export=ALL,DATASET_DIR=/scratch/wc3013/longcat-video-tta/datasets/panda_2048_480p,SMOKE_N=20 \
         sweep_experiment/sbatch/smoke_test_2048v.sbatch

Or run directly on a GPU node (login or interactive):

  python scripts/smoke_test_2048_dataset.py \
      --dataset-dir /scratch/wc3013/longcat-video-tta/datasets/panda_2048_480p \
      --num-videos 20

This script delegates the actual TTA invocation to the AdaSteer runner
(`delta_experiment/scripts/run_delta_a.py`) with `--delta-steps 0` so
the smoke test is identical to the NoTTA path of the real sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_notta_smoke(
    dataset_dir: Path,
    output_dir: Path,
    num_videos: int,
    num_cond_frames: int,
    num_frames: int,
    gen_start_frame: int,
    num_inference_steps: int,
    guidance_scale: float,
    resolution: str,
    seed: int,
    checkpoint_dir: Path,
) -> int:
    """Invoke run_delta_a.py with delta_steps=0 on the first N videos."""
    run_script = REPO_ROOT / "delta_experiment" / "scripts" / "run_delta_a.py"
    if not run_script.exists():
        print(f"ERROR: run_delta_a.py not found at {run_script}",
              file=sys.stderr)
        return 2

    cmd = [
        sys.executable, str(run_script),
        "--checkpoint-dir", str(checkpoint_dir),
        "--data-dir", str(dataset_dir),
        "--output-dir", str(output_dir),
        "--max-videos", str(num_videos),
        "--start-video-idx", "0",
        "--delta-steps", "0",
        "--delta-lr", "5e-3",
        "--num-cond-frames", str(num_cond_frames),
        "--num-frames", str(num_frames),
        "--gen-start-frame", str(gen_start_frame),
        "--tta-total-frames", str(gen_start_frame),
        "--tta-context-frames", str(num_cond_frames),
        "--num-inference-steps", str(num_inference_steps),
        "--guidance-scale", str(guidance_scale),
        "--resolution", resolution,
        "--seed", str(seed),
        "--no-save-videos",
        "--es-disable",
        "--caption-guard-mode", "warn",
        "--feature-frame-guard-mode", "warn",
    ]

    print()
    print("Running NoTTA smoke:")
    print("  " + " ".join(cmd))
    print()
    return int(subprocess.run(cmd).returncode)


def _check_summary(
    output_dir: Path,
    expected_n: int,
) -> int:
    summary_path = output_dir / "summary.json"
    if not summary_path.exists():
        print(f"FAIL: summary.json missing at {summary_path}",
              file=sys.stderr)
        return 1
    with open(summary_path) as f:
        summary = json.load(f)

    per_video = summary.get("per_video", [])
    if isinstance(per_video, dict):
        per_video = list(per_video.values())
    n_entries = len(per_video)
    if n_entries < expected_n:
        print(f"FAIL: expected {expected_n} per-video entries, got "
              f"{n_entries}", file=sys.stderr)
        return 1

    bad_psnr = [e for e in per_video[:expected_n]
                if e.get("psnr") in (None, "", float("nan"))]
    bad_ssim = [e for e in per_video[:expected_n]
                if e.get("ssim") in (None, "", float("nan"))]
    bad_lpips = [e for e in per_video[:expected_n]
                 if e.get("lpips") in (None, "", float("nan"))]

    print()
    print("Smoke summary:")
    print(f"  per_video entries     : {n_entries}")
    print(f"  null PSNR (top-N)     : {len(bad_psnr)}")
    print(f"  null SSIM (top-N)     : {len(bad_ssim)}")
    print(f"  null LPIPS (top-N)    : {len(bad_lpips)}")
    if summary.get("errors"):
        print(f"  reported errors       : {len(summary['errors'])}")
    print()

    if bad_psnr or bad_ssim or bad_lpips:
        print("FAIL: at least one video produced null pointwise metrics",
              file=sys.stderr)
        return 1
    if summary.get("errors"):
        print("FAIL: pipeline reported per-video errors", file=sys.stderr)
        return 1

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Default: <dataset-dir>/_smoke_test_notta")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/scratch/wc3013/longcat-video-checkpoints")
    parser.add_argument("--num-videos", type=int, default=20)
    parser.add_argument("--num-cond-frames", type=int, default=14)
    parser.add_argument("--num-frames", type=int, default=28)
    parser.add_argument("--gen-start-frame", type=int, default=48)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else dataset_dir / "_smoke_test_notta"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("2048-video dataset smoke test (NoTTA on first N)")
    print("=" * 70)
    print(f"  dataset    : {dataset_dir}")
    print(f"  output     : {output_dir}")
    print(f"  N          : {args.num_videos}")
    print(f"  frame geom : cond={args.num_cond_frames} num={args.num_frames} "
          f"gen_start={args.gen_start_frame}")
    print()

    t0 = time.time()
    rc = _run_notta_smoke(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        num_videos=args.num_videos,
        num_cond_frames=args.num_cond_frames,
        num_frames=args.num_frames,
        gen_start_frame=args.gen_start_frame,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        resolution=args.resolution,
        seed=args.seed,
        checkpoint_dir=Path(args.checkpoint_dir),
    )
    if rc != 0:
        print(f"FAIL: smoke runner exited with code {rc}", file=sys.stderr)
        return rc

    rc = _check_summary(output_dir=output_dir, expected_n=args.num_videos)
    elapsed = time.time() - t0
    if rc == 0:
        print(f"PASS in {elapsed/60:.1f} min")
    return rc


if __name__ == "__main__":
    sys.exit(main())
