#!/usr/bin/env python3
"""
Merge chunked experiment results into a single summary.

Reads summary.json and fvd_fid_stats.npz from each chunk_N/ subdirectory,
combines per-video results and FVD/FID sufficient statistics, then writes
a merged_summary.json in the parent directory.

FVD/FID are distributional metrics that require comparing feature
distributions across ALL videos.  This script merges the per-chunk
sufficient statistics (running sum, sum-of-outer-products, count) and
computes a single global FVD/FID from the combined 1000-video distribution
rather than averaging per-chunk FVD/FID values (which is incorrect).

Usage:
    python sweep_experiment/scripts/merge_chunks.py \
        --results-dir sweep_experiment/results/panda_longctx_1000v/NOTTA

    # Or merge all runs under a dataset directory:
    python sweep_experiment/scripts/merge_chunks.py \
        --results-dir sweep_experiment/results/panda_longctx_1000v --recursive
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import sqrtm


_COV_EPS = 1e-6


def _compute_frechet_distance(
    sum_a, cov_sum_a, n_a,
    sum_b, cov_sum_b, n_b,
    eps=_COV_EPS,
):
    """Frechet distance from running sums (float64).
    Mirrors _compute_frechet_distance in common.py."""
    mu_a = sum_a / n_a
    mu_b = sum_b / n_b
    sigma_a = cov_sum_a / n_a - np.outer(mu_a, mu_a)
    sigma_b = cov_sum_b / n_b - np.outer(mu_b, mu_b)
    sigma_a += eps * np.eye(sigma_a.shape[0])
    sigma_b += eps * np.eye(sigma_b.shape[0])
    diff = mu_a - mu_b
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean))


def load_chunk_data(run_dir):
    """Load summary.json and fvd_fid_stats.npz from all chunk_N subdirs."""
    chunks = []
    for d in sorted(run_dir.iterdir()):
        if not (d.is_dir() and d.name.startswith("chunk_")):
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue
        with open(summary_path) as f:
            data = json.load(f)
        data["_chunk_dir"] = str(d)
        stats_path = d / "fvd_fid_stats.npz"
        data["_has_stats"] = stats_path.exists()
        if stats_path.exists():
            data["_stats"] = dict(np.load(stats_path, allow_pickle=True))
        chunks.append(data)
    return chunks


def merge_frechet_stats(chunks):
    """Merge sufficient statistics across chunks and compute global FVD/FID."""
    stats_chunks = [c["_stats"] for c in chunks if c.get("_has_stats")]
    if not stats_chunks:
        return {}

    gen_sum = sum(s["gen_sum"] for s in stats_chunks)
    gen_cov = sum(s["gen_cov"] for s in stats_chunks)
    gen_count = sum(int(s["gen_count"]) for s in stats_chunks)
    ref_sum = sum(s["ref_sum"] for s in stats_chunks)
    ref_cov = sum(s["ref_cov"] for s in stats_chunks)
    ref_count = sum(int(s["ref_count"]) for s in stats_chunks)

    result = {
        "fvd_num_videos": gen_count,
        "fvd_num_ref_videos": ref_count,
        "fvd_num_chunks": len(stats_chunks),
    }

    if gen_count >= 2 and ref_count >= 2:
        fvd = _compute_frechet_distance(
            gen_sum, gen_cov, gen_count,
            ref_sum, ref_cov, ref_count,
        )
        result["fvd"] = round(fvd, 6)
    else:
        result["fvd"] = None
        result["fvd_error"] = "Not enough videos (gen=%d, ref=%d)" % (gen_count, ref_count)

    has_fid = all("fid_gen_sum" in s for s in stats_chunks)
    if has_fid:
        fid_gen_sum = sum(s["fid_gen_sum"] for s in stats_chunks)
        fid_gen_cov = sum(s["fid_gen_cov"] for s in stats_chunks)
        fid_gen_frames = sum(int(s["fid_gen_frames"]) for s in stats_chunks)
        fid_ref_sum = sum(s["fid_ref_sum"] for s in stats_chunks)
        fid_ref_cov = sum(s["fid_ref_cov"] for s in stats_chunks)
        fid_ref_frames = sum(int(s["fid_ref_frames"]) for s in stats_chunks)
        if fid_gen_frames >= 2 and fid_ref_frames >= 2:
            fid = _compute_frechet_distance(
                fid_gen_sum, fid_gen_cov, fid_gen_frames,
                fid_ref_sum, fid_ref_cov, fid_ref_frames,
            )
            result["fid"] = round(fid, 6)
            result["fid_num_frames_gen"] = fid_gen_frames
            result["fid_num_frames_ref"] = fid_ref_frames

    chunk_fvds = [c.get("fvd") for c in chunks if c.get("fvd") is not None]
    chunk_fids = [c.get("fid") for c in chunks if c.get("fid") is not None]
    if chunk_fvds:
        result["fvd_per_chunk"] = chunk_fvds
    if chunk_fids:
        result["fid_per_chunk"] = chunk_fids
    return result


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
            metrics[k + "_std"] = float(np.std(vals))

    total_train = [r.get("train_time", 0) for r in successful]
    total_gen = [r.get("gen_time", 0) for r in successful]
    total_all = [r.get("total_time", 0) for r in successful]

    merged = {
        "num_chunks": len(chunks),
        "num_videos": sum(c.get("num_videos", 0) for c in chunks),
        "num_successful": len(successful),
    }
    merged.update(metrics)
    merged["avg_train_time"] = float(np.mean(total_train)) if total_train else 0
    merged["avg_gen_time"] = float(np.mean(total_gen)) if total_gen else 0
    merged["avg_total_time"] = float(np.mean(total_all)) if total_all else 0

    frechet = merge_frechet_stats(chunks)
    if frechet:
        merged.update(frechet)
    else:
        chunk_fvds = [c.get("fvd") for c in chunks if c.get("fvd") is not None]
        chunk_fids = [c.get("fid") for c in chunks if c.get("fid") is not None]
        if chunk_fvds:
            merged["fvd_per_chunk"] = chunk_fvds
            merged["fvd_mean_of_chunks"] = float(np.mean(chunk_fvds))
            merged["fvd_WARNING"] = (
                "No fvd_fid_stats.npz found; fvd_mean_of_chunks is the "
                "average of per-chunk FVDs (NOT a valid global FVD). "
                "Re-run experiments with updated code to get proper stats."
            )
        if chunk_fids:
            merged["fid_per_chunk"] = chunk_fids
            merged["fid_mean_of_chunks"] = float(np.mean(chunk_fids))

    vbench_chunks = [c.get("vbench") for c in chunks
                     if c.get("vbench") and not c.get("vbench_skipped", True)]
    if vbench_chunks:
        all_dims = set()
        for vc in vbench_chunks:
            if isinstance(vc, dict):
                all_dims.update(vc.keys())
        vbench_merged = {}
        for dim in sorted(all_dims):
            vals = []
            for vc in vbench_chunks:
                if not isinstance(vc, dict) or dim not in vc:
                    continue
                v = vc[dim]
                if isinstance(v, (int, float)):
                    vals.append(float(v))
                elif isinstance(v, dict):
                    inner = v.get(dim, next(iter(v.values()), None))
                    if isinstance(inner, (list, tuple)) and inner and isinstance(inner[0], (int, float)):
                        vals.append(float(inner[0]))
            if vals:
                vbench_merged[dim] = float(np.mean(vals))
                vbench_merged[dim + "_std"] = float(np.std(vals))
                vbench_merged[dim + "_per_chunk"] = vals
        if vbench_merged:
            merged["vbench"] = vbench_merged
            merged["vbench_num_chunks"] = len(vbench_chunks)

    config = chunks[0].get("config", chunks[0].get("experiment_config", {}))
    if config:
        merged["config"] = config
    return merged


def process_run_dir(run_dir):
    """Process a single run directory with chunk_N subdirs."""
    chunks = load_chunk_data(run_dir)
    if not chunks:
        return None

    merged = merge_summaries(chunks)
    out_path = run_dir / "merged_summary.json"
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)

    n = merged["num_successful"]
    has_global = "fvd" in merged and merged["fvd"] is not None
    print("  %s: %d chunks, %d videos" % (run_dir.name, len(chunks), n))
    print("    PSNR=%.3f  SSIM=%.4f  LPIPS=%.4f" % (
        merged.get("psnr", 0), merged.get("ssim", 0), merged.get("lpips", 0)))
    if has_global:
        fid_str = ("  FID=%.1f" % merged["fid"]) if merged.get("fid") else ""
        print("    FVD=%.1f (global, %s videos)%s" % (
            merged["fvd"], merged.get("fvd_num_videos", "?"), fid_str))
    elif "fvd_mean_of_chunks" in merged:
        print("    FVD~%.1f (WARNING: avg of chunk FVDs, not true global)" %
              merged["fvd_mean_of_chunks"])
    if "vbench" in merged and isinstance(merged["vbench"], dict):
        vb = merged["vbench"]
        dims = [k for k in vb if not k.endswith("_std") and not k.endswith("_per_chunk")]
        parts = ["    VBench:"] + ["%s=%.3f" % (d, vb[d]) for d in dims]
        print("  ".join(parts))
    print("    Avg time: train=%.1fs  gen=%.1fs  total=%.1fs" % (
        merged["avg_train_time"], merged["avg_gen_time"], merged["avg_total_time"]))
    print("    -> %s" % out_path)
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
        print("ERROR: %s does not exist" % results_dir, file=sys.stderr)
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
            print("No chunk_N/summary.json found under %s" % results_dir)
    else:
        result = process_run_dir(results_dir)
        if result is None:
            print("No chunk_N/summary.json found in %s" % results_dir)
            sys.exit(1)


if __name__ == "__main__":
    main()
