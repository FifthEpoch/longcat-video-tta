#!/usr/bin/env python3
"""Create GENERATED-ONLY clips by trimming conditioning frames from saved mp4s.

Background / why this exists
----------------------------
Saved LongCat continuation mp4s contain ``[cond | gen]`` frames (for the 1000v
preview geometry: 14 conditioning + 15 generated = 29 frames). VBench was being
run on the FULL mp4, so every VBench score was ~half real conditioning footage
(see sweep_experiment/reports/per_video_analysis/2026-07-22/eval_metric_audit.md).
Pixel metrics and FVD already score gen-only (they skip the first
``num_cond_frames``), but VBench takes a directory of mp4s and cannot skip
leading frames itself.

This script writes gen-only clips (dropping the first ``--num-cond-frames``
frames) into a sibling ``videos_geneval/`` dir per chunk, preserving filenames
so downstream canonical-id joins are unchanged. Encoding matches the pipeline's
own writer exactly (imageio + libx264, quality=9) so VBench sees identical codec
characteristics. The original ``videos/`` dir is never modified.

Usage:
    python3 scripts/make_geneval_clips.py \
        --method-dir sweep_experiment/results/panda_ood_budget_1000v_preview/NOTTA \
        --num-cond-frames 14

Then run VBench on the trimmed clips:
    python3 scripts/run_vbench_backfill.py \
        --method-dir <same> --videos-subdir videos_geneval \
        --out-subdir vbench_results_geneval --dimensions <dims> --mode custom_input
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _save_video_u8(frames_u8: np.ndarray, output_path: Path, fps: int = 24) -> None:
    """Write [T,H,W,3] uint8 frames to mp4 — identical settings to the pipeline's
    ``save_video_from_numpy`` (imageio + libx264, quality=9)."""
    import imageio.v2 as imageio

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_path), frames_u8, fps=fps, codec="libx264", quality=9)


def _read_frames(src: Path) -> np.ndarray:
    """Decode an mp4 to [T,H,W,3] uint8."""
    import imageio.v3 as iio

    arr = iio.imread(str(src), plugin="pyav")  # [T,H,W,C] uint8
    if arr.ndim == 3:  # grayscale safety -> stack to 3ch
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:  # drop alpha
        arr = arr[..., :3]
    return arr


def _trim_one(src: Path, dst: Path, num_cond: int, fps: int) -> Tuple[bool, str]:
    """Return (ok, message)."""
    try:
        arr = _read_frames(src)
    except Exception as exc:  # noqa: BLE001
        return False, f"decode failed: {exc}"
    T = arr.shape[0]
    if T <= num_cond:
        return False, f"too short: {T} frames <= num_cond={num_cond}"
    gen = arr[num_cond:]
    try:
        _save_video_u8(gen, dst, fps=fps)
    except Exception as exc:  # noqa: BLE001
        return False, f"write failed: {exc}"
    # verify frame-exact output
    try:
        n_out = _read_frames(dst).shape[0]
    except Exception as exc:  # noqa: BLE001
        return False, f"verify decode failed: {exc}"
    if n_out != T - num_cond:
        return False, f"frame count mismatch: wrote {n_out}, expected {T - num_cond}"
    return True, f"{T} -> {n_out} frames"


def process_method_dir(method_dir: Path, num_cond: int, videos_subdir: str,
                       out_subdir: str, fps: int, force: bool,
                       dry_run: bool) -> int:
    if not method_dir.exists():
        print(f"[error] method dir does not exist: {method_dir}", file=sys.stderr)
        return 2

    chunks = sorted(method_dir.glob("chunk_*"))
    if not chunks and (method_dir / videos_subdir).is_dir():
        chunks = [method_dir]
    if not chunks:
        print(f"[error] no chunks (or {videos_subdir}/) under {method_dir}", file=sys.stderr)
        return 2

    print(f"Method dir      : {method_dir}")
    print(f"num_cond_frames : {num_cond}  (dropping first {num_cond} frames)")
    print(f"in / out subdir : {videos_subdir} -> {out_subdir}")
    print(f"chunks          : {len(chunks)}")
    print()

    n_ok = n_skip = n_fail = 0
    failures: List[Tuple[str, str]] = []
    for chunk in chunks:
        vdir = chunk / videos_subdir
        if not vdir.is_dir():
            print(f"  {chunk.name}: no {videos_subdir}/ — skipping")
            continue
        odir = chunk / out_subdir
        mp4s = sorted(vdir.glob("*.mp4"))
        made = 0
        for src in mp4s:
            dst = odir / src.name
            if dst.exists() and not force:
                n_skip += 1
                continue
            if dry_run:
                made += 1
                continue
            ok, msg = _trim_one(src, dst, num_cond, fps)
            if ok:
                n_ok += 1
                made += 1
            else:
                n_fail += 1
                failures.append((str(src), msg))
        print(f"  {chunk.name}: {len(mp4s)} src mp4  -> {made} trimmed "
              f"({'dry-run' if dry_run else 'written'})")

    print()
    print("=" * 60)
    print(f"trimmed OK : {n_ok}")
    print(f"skipped    : {n_skip} (already existed)")
    print(f"failed     : {n_fail}")
    if failures:
        print("failures (up to 10):")
        for p, m in failures[:10]:
            print(f"  {p}: {m}")
    return 0 if n_fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Trim conditioning frames -> gen-only clips")
    ap.add_argument("--method-dir", required=True, type=Path)
    ap.add_argument("--num-cond-frames", type=int, default=14,
                    help="Number of leading conditioning frames to drop (default 14).")
    ap.add_argument("--videos-subdir", default="videos",
                    help="Input clip subdir per chunk (default 'videos').")
    ap.add_argument("--out-subdir", default="videos_geneval",
                    help="Output gen-only clip subdir per chunk (default 'videos_geneval').")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--force", action="store_true", help="Overwrite existing trimmed clips.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return process_method_dir(args.method_dir, args.num_cond_frames,
                              args.videos_subdir, args.out_subdir, args.fps,
                              args.force, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
