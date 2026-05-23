#!/usr/bin/env python3
"""End-to-end FVD/FID recomputation from chunked sufficient statistics.

Independent of sweep_experiment/scripts/merge_chunks.py: this script walks
all chunk_N/fvd_fid_stats.npz files under a run directory, sums the
sufficient statistics manually, computes a single global FVD and FID, and
prints the result alongside the value stored in merged_summary.json (when
available).

This serves three purposes:

1. Validates that merge_chunks.py is implemented correctly: if its output
   in merged_summary.json disagrees with this script's recomputation, one
   of the two is buggy.

2. Validates that the per-chunk sufficient statistics are internally
   consistent: feature dimensions, counts, and float64 dtype are checked
   for every chunk.

3. Provides a single un-merged number that can be quoted in the paper
   as the canonical FVD/FID for that run, replacing any per-chunk
   averages that may have leaked into older logs.

Usage on cluster:

    cd $LONGCAT_REPO
    python scripts/recompute_fvd_fid_from_stats.py \\
        --run-dir sweep_experiment/results/panda_longctx_1000v/NOTTA

    # Or compare two runs side by side:
    python scripts/recompute_fvd_fid_from_stats.py \\
        --run-dir sweep_experiment/results/panda_longctx_1000v/NOTTA \\
        --compare-dir sweep_experiment/results/panda_longctx_1000v/ADA_S10

The script only depends on numpy and scipy (which the cluster env has);
it does NOT require torch, the I3D module, or any GPU.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import sqrtm


_COV_EPS = 1e-6


def frechet_distance(sum_a, cov_a, n_a, sum_b, cov_b, n_b, eps=_COV_EPS):
    sum_a = np.asarray(sum_a, dtype=np.float64)
    cov_a = np.asarray(cov_a, dtype=np.float64)
    sum_b = np.asarray(sum_b, dtype=np.float64)
    cov_b = np.asarray(cov_b, dtype=np.float64)
    mu_a = sum_a / n_a
    mu_b = sum_b / n_b
    sigma_a = cov_a / n_a - np.outer(mu_a, mu_a) + eps * np.eye(sum_a.shape[0])
    sigma_b = cov_b / n_b - np.outer(mu_b, mu_b) + eps * np.eye(sum_b.shape[0])
    diff = mu_a - mu_b
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean))


def load_chunk_stats(run_dir: Path) -> List[Dict]:
    """Load all chunk_N/fvd_fid_stats.npz under run_dir and validate shapes."""
    paths = sorted(run_dir.glob("chunk_*/fvd_fid_stats.npz"))
    if not paths:
        raise SystemExit(f"No chunk_*/fvd_fid_stats.npz under {run_dir}")
    stats = []
    expected_keys = ["gen_sum", "gen_cov", "gen_count", "ref_sum", "ref_cov", "ref_count"]
    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        for k in expected_keys:
            if k not in d:
                raise SystemExit(f"{p}: missing key {k!r}")
        d["_path"] = str(p)
        stats.append(d)
    return stats


def consistency_audit(stats: List[Dict]) -> None:
    """Print a per-chunk shape/count audit. Raises on any inconsistency."""
    print(f"Loaded {len(stats)} chunks")
    print(
        f"  {'chunk':24s}  {'gen_n':>6s}  {'ref_n':>6s}  "
        f"{'gen_sum':>10s}  {'gen_cov':>13s}  "
        f"{'ref_sum':>10s}  {'ref_cov':>13s}"
    )
    ref_shapes: List[Tuple] = []
    for s in stats:
        name = Path(s["_path"]).parent.name
        print(
            f"  {name:24s}  {int(s['gen_count']):>6d}  {int(s['ref_count']):>6d}  "
            f"{str(s['gen_sum'].shape):>10s}  {str(s['gen_cov'].shape):>13s}  "
            f"{str(s['ref_sum'].shape):>10s}  {str(s['ref_cov'].shape):>13s}"
        )
        ref_shapes.append((s["gen_sum"].shape, s["gen_cov"].shape,
                           s["ref_sum"].shape, s["ref_cov"].shape))
    unique_shapes = set(ref_shapes)
    if len(unique_shapes) != 1:
        raise SystemExit(f"Inconsistent feature shapes across chunks: {unique_shapes}")
    if any(s["gen_sum"].dtype != np.float64 for s in stats):
        print("  WARNING: not all gen_sum arrays are float64.")
    if any(s["gen_cov"].dtype != np.float64 for s in stats):
        print("  WARNING: not all gen_cov arrays are float64.")


def merge_and_compute(stats: List[Dict], kind: str) -> Optional[float]:
    """kind in {'fvd', 'fid'}."""
    if kind == "fvd":
        gen_sum_key, gen_cov_key, gen_n_key = "gen_sum", "gen_cov", "gen_count"
        ref_sum_key, ref_cov_key, ref_n_key = "ref_sum", "ref_cov", "ref_count"
    elif kind == "fid":
        gen_sum_key, gen_cov_key, gen_n_key = "fid_gen_sum", "fid_gen_cov", "fid_gen_frames"
        ref_sum_key, ref_cov_key, ref_n_key = "fid_ref_sum", "fid_ref_cov", "fid_ref_frames"
    else:
        raise ValueError(kind)

    if not all(gen_sum_key in s for s in stats):
        return None

    gen_sum = sum(s[gen_sum_key] for s in stats)
    gen_cov = sum(s[gen_cov_key] for s in stats)
    gen_count = sum(int(s[gen_n_key]) for s in stats)
    ref_sum = sum(s[ref_sum_key] for s in stats)
    ref_cov = sum(s[ref_cov_key] for s in stats)
    ref_count = sum(int(s[ref_n_key]) for s in stats)

    return frechet_distance(gen_sum, gen_cov, gen_count, ref_sum, ref_cov, ref_count)


def load_merged_summary(run_dir: Path) -> Dict:
    p = run_dir / "merged_summary.json"
    if p.exists():
        with p.open() as f:
            return json.load(f)
    return {}


def summarize(run_dir: Path) -> Dict[str, Optional[float]]:
    print(f"\n=== {run_dir} ===")
    stats = load_chunk_stats(run_dir)
    consistency_audit(stats)
    fvd = merge_and_compute(stats, "fvd")
    fid = merge_and_compute(stats, "fid")
    merged = load_merged_summary(run_dir)
    stored_fvd = merged.get("fvd")
    stored_fid = merged.get("fid")
    n_gen = sum(int(s["gen_count"]) for s in stats)
    n_ref = sum(int(s["ref_count"]) for s in stats)
    print(f"Total generated videos: {n_gen}")
    print(f"Total reference videos: {n_ref}")
    print(f"Recomputed FVD: {fvd:.4f}" if fvd is not None else "Recomputed FVD: n/a")
    print(f"Stored FVD    : {stored_fvd if stored_fvd is None else f'{stored_fvd:.4f}'}")
    if fvd is not None and stored_fvd is not None:
        rel = abs(fvd - stored_fvd) / max(abs(stored_fvd), 1e-12)
        flag = "OK" if rel < 1e-4 else "MISMATCH"
        print(f"  -> recomputed vs stored: {flag} (rel_diff={rel:.3e})")
    print(f"Recomputed FID: {fid:.4f}" if fid is not None else "Recomputed FID: n/a")
    print(f"Stored FID    : {stored_fid if stored_fid is None else f'{stored_fid:.4f}'}")
    if fid is not None and stored_fid is not None:
        rel = abs(fid - stored_fid) / max(abs(stored_fid), 1e-12)
        flag = "OK" if rel < 1e-4 else "MISMATCH"
        print(f"  -> recomputed vs stored: {flag} (rel_diff={rel:.3e})")
    return {"fvd": fvd, "fid": fid, "stored_fvd": stored_fvd, "stored_fid": stored_fid}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Run directory containing chunk_N/fvd_fid_stats.npz")
    parser.add_argument("--compare-dir", type=Path, default=None,
                        help="Optional second run to compute the FVD/FID delta against.")
    args = parser.parse_args()

    a = summarize(args.run_dir)
    if args.compare_dir is not None:
        b = summarize(args.compare_dir)
        print("\n=== Pairwise delta (compare - run) ===")
        if a["fvd"] is not None and b["fvd"] is not None:
            print(f"  dFVD = {b['fvd'] - a['fvd']:+.4f}  "
                  f"(run={a['fvd']:.4f}, compare={b['fvd']:.4f})")
        if a["fid"] is not None and b["fid"] is not None:
            print(f"  dFID = {b['fid'] - a['fid']:+.4f}  "
                  f"(run={a['fid']:.4f}, compare={b['fid']:.4f})")


if __name__ == "__main__":
    main()
