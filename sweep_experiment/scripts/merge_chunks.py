#!/usr/bin/env python3
"""
Merge chunked experiment results into a single summary.

Reads summary.json from each chunk_N/ subdirectory, combines per-video
results, and writes a merged_summary.json in the parent directory.

Usage:
    python sweep_experiment/scripts/merge_chunks.py \
        --results-dir sweep_experiment/results/panda_longctx_1000v/NOTTA

    # Or merge all runs under a dataset directory:
    python sweep_experiment/scripts/merge_chunks.py \
        --results-dir sweep_experiment/results/panda_longctx_1000v --recursive
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


def load_chunk_summaries(run_dir: Path):
    """Load summary.json from all chunk_N subdirs, sorted by chunk index."""
    chunks = []
    for d in sorted(run_dir.iterdir()):
        if d.is_dir() and d.name.startswith("chunk_"):
            summary_path = d / "summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    data = json.load(f)
                data["_chunk_dir"] = str(d)
                chunks.append(data)
    return chunks


def merge_summaries(chunks):
    """Merge chunk summaries into a single aggregate summary."""
    all_results = []
    for c in chunks:
        all_results.extend(c.get("per_video_results", c.get("results", [])))

    successful = [r for r in all_results if r.get("psnr") is not None
                  or r.get("ssim") is not None]

    metric_keys = ["psnr", "ssim", "lpips"]
    metrics = {}
    for k in metric_keys:
        vals = [r[k] for r in successful if k in r and r[k] is not None]
        if vals:
            metrics[k] = float(np.mean(vals))
            metrics[f"{k}_std"] = float(np.std(vals))

    total_train = [r.get("train_time", 0) for r in successful]
    total_gen = [r.get("gen_time", 0) for r in successful]
    total_all = [r.get("total_time", 0) for r in successful]

    chunk_fvds = [c.get("fvd") for c in chunks if c.get("fvd") is not None]
    chunk_fids = [c.get("fid") for c in chunks if c.get("fid") is not None]

    merged = {
        "num_chunks": len(chunks),
        "num_videos": sum(c.get("num_videos", 0) for c in chunks),
        "num_successful": len(successful),
        **metrics,
        "avg_train_time": float(np.mean(total_train)) if total_train else 0,
        "avg_gen_time": float(np.mean(total_gen)) if total_gen else 0,
        "avg_total_time": float(np.mean(total_all)) if total_all else 0,
    }

    if chunk_fvds:
        merged["fvd_per_chunk"] = chunk_fvds
        merged["fvd_mean"] = float(np.mean(chunk_fvds))
        merged["fvd_std"] = float(np.std(chunk_fvds))
    if chunk_fids:
        merged["fid_per_chunk"] = chunk_fids
        merged["fid_mean"] = float(np.mean(chunk_fids))
        merged["fid_std"] = float(np.std(chunk_fids))

    config = chunks[0].get("config", chunks[0].get("experiment_config", {}))
    if config:
        merged["config"] = config

    return merged


def process_run_dir(run_dir: Path):
    """Process a single run directory with chunk_N subdirs."""
    chunks = load_chunk_summaries(run_dir)
    if not chunks:
        return None

    merged = merge_summaries(chunks)
    out_path = run_dir / "merged_summary.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    n = merged["num_successful"]
    print(f"  {run_dir.name}: {len(chunks)} chunks, {n} videos")
    print(f"    PSNR={merged.get('psnr', 0):.3f}  "
          f"SSIM={merged.get('ssim', 0):.4f}  "
          f"LPIPS={merged.get('lpips', 0):.4f}")
    if "fvd_mean" in merged:
        print(f"    FVD={merged['fvd_mean']:.1f}±{merged['fvd_std']:.1f}  "
              f"FID={merged.get('fid_mean', 0):.1f}±{merged.get('fid_std', 0):.1f}")
    print(f"    Avg time: train={merged['avg_train_time']:.1f}s  "
          f"gen={merged['avg_gen_time']:.1f}s  "
          f"total={merged['avg_total_time']:.1f}s")
    print(f"    → {out_path}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge chunked experiment results")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Run directory containing chunk_N/ subdirs, "
                             "or parent directory with --recursive")
    parser.add_argument("--recursive", action="store_true",
                        help="Process all subdirectories that contain chunk_N/ dirs")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    if args.recursive:
        found = False
        for sub in sorted(results_dir.iterdir()):
            if sub.is_dir():
                chunks = list(sub.glob("chunk_*/summary.json"))
                if chunks:
                    found = True
                    process_run_dir(sub)
        if not found:
            print(f"No chunk_N/summary.json found under {results_dir}")
    else:
        result = process_run_dir(results_dir)
        if result is None:
            print(f"No chunk_N/summary.json found in {results_dir}")
            sys.exit(1)


if __name__ == "__main__":
    main()
