#!/usr/bin/env python3
"""Run VBench++ backfill for a single method dir.

For each ``chunk_<i>/videos/`` containing saved generated mp4s, runs VBench
for the requested dimensions and writes the per-chunk
``vbench_<dim>_eval_results.json`` files alongside the existing 3 dimensions
that did succeed in the original runs.

After backfill, run ``scripts/update_merged_with_vbench.py`` to fold the new
dimensions into the merged ``merged_summary.json["vbench"]`` dict.

This script is idempotent: chunks whose result file already exists for a
given dimension are skipped unless ``--force`` is passed.

Run:
    python scripts/run_vbench_backfill.py \
        --method-dir sweep_experiment/results/panda_1000v_standard/NOTTA \
        --dimensions motion_smoothness dynamic_degree imaging_quality temporal_flickering \
        --mode custom_input
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional


DEFAULT_DIMS = [
    "motion_smoothness",
    "dynamic_degree",
    "imaging_quality",
    "temporal_flickering",
]


def _find_full_info_json() -> str:
    """Return path to VBench_full_info.json shipped with the vbench pkg."""
    import vbench as _v
    pkg_dir = os.path.dirname(_v.__file__)
    candidates = [
        os.path.join(pkg_dir, "VBench_full_info.json"),
        os.path.join(os.path.dirname(pkg_dir), "vbench", "VBench_full_info.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "VBench_full_info.json not found alongside vbench package. "
        f"Tried: {candidates}"
    )


def _result_file(chunk_vbench_dir: Path, dim: str) -> Path:
    return chunk_vbench_dir / f"vbench_{dim}_eval_results.json"


def run_backfill(method_dir: Path, dimensions: List[str],
                 mode: str = "custom_input",
                 force: bool = False,
                 dry_run: bool = False,
                 videos_subdir: str = "videos",
                 out_subdir: str = "vbench_results") -> int:
    """Returns 0 on success.

    videos_subdir : which per-chunk clip dir to evaluate (default 'videos').
                    Use 'videos_geneval' for generated-only clips produced by
                    scripts/make_geneval_clips.py.
    out_subdir    : where per-chunk vbench_<dim>_eval_results.json are written
                    (default 'vbench_results'; use 'vbench_results_geneval' to
                    keep gen-only results separate from the old full-clip ones).
    """

    if not method_dir.exists():
        print(f"[error] method dir does not exist: {method_dir}", file=sys.stderr)
        return 2

    chunks = sorted(method_dir.glob("chunk_*"))
    if not chunks:
        # support older single-job layout: method_dir/<videos_subdir>/*.mp4
        if (method_dir / videos_subdir).is_dir():
            chunks = [method_dir]
        else:
            print(f"[error] no chunks under {method_dir}", file=sys.stderr)
            return 2

    print(f"Method dir   : {method_dir}")
    print(f"Dimensions   : {dimensions}")
    print(f"Mode         : {mode}")
    print(f"Force        : {force}")
    print(f"Videos subdir: {videos_subdir}")
    print(f"Out subdir   : {out_subdir}")
    print(f"Chunks       : {len(chunks)}")
    print()

    if dry_run:
        for chunk_dir in chunks:
            videos_dir = chunk_dir / videos_subdir
            n = len(list(videos_dir.glob("*.mp4"))) if videos_dir.is_dir() else 0
            vb_dir = chunk_dir / out_subdir
            existing = []
            for dim in dimensions:
                if _result_file(vb_dir, dim).exists():
                    existing.append(dim)
            print(f"  {chunk_dir.name}  videos={n:>3}  "
                  f"existing={existing}  to_run={[d for d in dimensions if d not in existing]}")
        return 0

    # Lazy import VBench so dry-run doesn't need the env activated.
    import torch
    from vbench import VBench

    full_info = _find_full_info_json()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device           : {device}")
    print(f"  full_info_json   : {full_info}")
    print()

    t_start = time.time()
    n_run = 0
    n_skip = 0
    n_fail = 0
    failed: List = []

    for chunk_idx, chunk_dir in enumerate(chunks):
        videos_dir = chunk_dir / videos_subdir
        if not videos_dir.is_dir():
            print(f"  [{chunk_idx+1}/{len(chunks)}] {chunk_dir.name}: "
                  f"no {videos_subdir}/ subdir — skipping")
            continue

        mp4_count = len(list(videos_dir.glob("*.mp4")))
        if mp4_count == 0:
            print(f"  [{chunk_idx+1}/{len(chunks)}] {chunk_dir.name}: "
                  f"0 mp4 in {videos_subdir}/ — skipping")
            continue

        vb_dir = chunk_dir / out_subdir
        vb_dir.mkdir(parents=True, exist_ok=True)

        # Initialise a fresh VBench per chunk so output_path is set correctly.
        try:
            vb = VBench(device, full_info, str(vb_dir))
        except Exception as exc:
            print(f"  [{chunk_idx+1}/{len(chunks)}] {chunk_dir.name}: "
                  f"VBench init failed: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            traceback.print_exc()
            n_fail += 1
            failed.append((chunk_dir.name, "init", str(exc)))
            continue

        for dim in dimensions:
            res_path = _result_file(vb_dir, dim)
            if res_path.exists() and not force:
                print(f"    {chunk_dir.name} / {dim}: SKIP (exists)")
                n_skip += 1
                continue

            print(f"    {chunk_dir.name} / {dim}: running on {mp4_count} videos ...",
                  end=" ", flush=True)
            t0 = time.time()
            try:
                vb.evaluate(
                    videos_path=str(videos_dir),
                    name=f"vbench_{dim}",
                    dimension_list=[dim],
                    mode=mode,
                )
                dt = time.time() - t0
                if res_path.exists():
                    print(f"OK ({dt:.1f}s)")
                    n_run += 1
                else:
                    print(f"WARN: completed but no result file at {res_path}")
                    n_fail += 1
                    failed.append((chunk_dir.name, dim, "no result file written"))
            except Exception as exc:
                dt = time.time() - t0
                print(f"FAIL ({dt:.1f}s): {type(exc).__name__}: {exc}")
                traceback.print_exc()
                n_fail += 1
                failed.append((chunk_dir.name, dim, f"{type(exc).__name__}: {exc}"))

    elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"Backfill summary for {method_dir.name}")
    print("=" * 70)
    print(f"  ran     : {n_run}")
    print(f"  skipped : {n_skip} (already existed)")
    print(f"  failed  : {n_fail}")
    print(f"  elapsed : {elapsed:.1f}s")
    if failed:
        print()
        print("  failures:")
        for c, d, m in failed:
            print(f"    {c} / {d}: {m}")
    return 0 if n_fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-dir", required=True, type=Path)
    ap.add_argument("--dimensions", nargs="+", default=DEFAULT_DIMS,
                    help=f"VBench dims to backfill (default: {DEFAULT_DIMS})")
    ap.add_argument("--mode", default="custom_input",
                    choices=["custom_input", "i2v", "t2v"])
    ap.add_argument("--force", action="store_true",
                    help="Re-run even if result file already exists.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print plan without invoking VBench.")
    ap.add_argument("--videos-subdir", default="videos",
                    help="Per-chunk clip dir to evaluate (default 'videos'; "
                         "use 'videos_geneval' for generated-only clips).")
    ap.add_argument("--out-subdir", default="vbench_results",
                    help="Per-chunk output dir for results (default "
                         "'vbench_results'; use 'vbench_results_geneval').")
    args = ap.parse_args()
    return run_backfill(args.method_dir, args.dimensions, args.mode,
                        args.force, args.dry_run,
                        videos_subdir=args.videos_subdir,
                        out_subdir=args.out_subdir)


if __name__ == "__main__":
    sys.exit(main())
