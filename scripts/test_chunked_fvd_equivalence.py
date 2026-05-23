#!/usr/bin/env python3
"""Numerical unit test: chunked vs single-pass Frechet distance.

Generates two synthetic distributions of 400-D features (mimicking the I3D
output dimension and the typical FVD scale used in the AdaSteer paper),
computes the Frechet distance two ways:

  (A) Single pass on the full 999-sample stream.
  (B) Split into 10 chunks of 99-100 samples, accumulate sufficient
      statistics per chunk, sum them, and compute Frechet distance from
      the merged totals.

The two values must agree to <1e-6 relative tolerance, otherwise the
chunked merge in sweep_experiment/scripts/merge_chunks.py is buggy.

Why this test matters: the +5.4 FVD regression on long-context Panda
999-video was computed via chunked merge. If chunked != single-pass, the
result could be an artifact. This test isolates the math, removing all
I/O, all I3D-extraction, and all reference-distribution-mismatch concerns,
so we have one un-contaminated answer about the merge implementation.

Run locally (no cluster needed):
    python3 scripts/test_chunked_fvd_equivalence.py
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import sqrtm


_COV_EPS = 1e-6


def frechet_from_sums(sum_a, cov_a, n_a, sum_b, cov_b, n_b, eps=_COV_EPS):
    """Same closed-form as merge_chunks._compute_frechet_distance."""
    mu_a = sum_a / n_a
    mu_b = sum_b / n_b
    sigma_a = cov_a / n_a - np.outer(mu_a, mu_a)
    sigma_b = cov_b / n_b - np.outer(mu_b, mu_b)
    sigma_a = sigma_a + eps * np.eye(sigma_a.shape[0])
    sigma_b = sigma_b + eps * np.eye(sigma_b.shape[0])
    diff = mu_a - mu_b
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean))


def single_pass(features_gen, features_ref):
    n_g = features_gen.shape[0]
    n_r = features_ref.shape[0]
    sum_g = features_gen.sum(axis=0)
    cov_g = features_gen.T @ features_gen
    sum_r = features_ref.sum(axis=0)
    cov_r = features_ref.T @ features_ref
    return frechet_from_sums(sum_g, cov_g, n_g, sum_r, cov_r, n_r)


def chunked_pass(features_gen, features_ref, chunk_size):
    """Mirror the production merge: per-chunk sufficient statistics, then sum."""
    d = features_gen.shape[1]
    sum_g = np.zeros(d, dtype=np.float64)
    cov_g = np.zeros((d, d), dtype=np.float64)
    sum_r = np.zeros(d, dtype=np.float64)
    cov_r = np.zeros((d, d), dtype=np.float64)
    n_g = 0
    n_r = 0
    for start in range(0, features_gen.shape[0], chunk_size):
        end = min(start + chunk_size, features_gen.shape[0])
        chunk_g = features_gen[start:end]
        chunk_r = features_ref[start:end]
        sum_g = sum_g + chunk_g.sum(axis=0)
        cov_g = cov_g + chunk_g.T @ chunk_g
        sum_r = sum_r + chunk_r.sum(axis=0)
        cov_r = cov_r + chunk_r.T @ chunk_r
        n_g += chunk_g.shape[0]
        n_r += chunk_r.shape[0]
    return frechet_from_sums(sum_g, cov_g, n_g, sum_r, cov_r, n_r)


def run_one(seed, n_total, dim, chunk_size, mean_shift, label):
    rng = np.random.default_rng(seed)
    cov = np.diag(rng.uniform(0.5, 1.5, size=dim))
    sample_gen = rng.multivariate_normal(np.zeros(dim) + mean_shift, cov, size=n_total)
    sample_ref = rng.multivariate_normal(np.zeros(dim), cov, size=n_total)
    fvd_single = single_pass(sample_gen, sample_ref)
    fvd_chunked = chunked_pass(sample_gen, sample_ref, chunk_size)
    rel = abs(fvd_single - fvd_chunked) / max(abs(fvd_single), 1e-12)
    ok = rel < 1e-6
    status = "PASS" if ok else "FAIL"
    print(
        f"[{status}] {label}: single={fvd_single:.6f}  chunked={fvd_chunked:.6f}  "
        f"abs_diff={fvd_single - fvd_chunked:+.3e}  rel_diff={rel:.3e}"
    )
    return ok


def main() -> None:
    print("Chunked vs single-pass Frechet-distance unit test.")
    print("=" * 60)
    cases = [
        # (seed, n, dim, chunk, mean_shift, label)
        (0, 999, 400, 100, 0.0, "FVD shape, 999 vs 999, chunk=100, distributions match"),
        (1, 999, 400, 100, 0.10, "FVD shape, mean shift +0.10 (typical TTA-vs-NoTTA scale)"),
        (2, 999, 400,  50, 0.05, "FVD shape, smaller chunks"),
        (3, 999, 400, 250, 0.05, "FVD shape, larger chunks"),
        (4, 999, 2048, 100, 0.0, "FID shape (Inception-2048), no shift"),
        (5, 999, 2048, 100, 0.05, "FID shape (Inception-2048), small shift"),
        (6, 500, 400,  37, 0.20, "Uneven chunk sizes (500 / 37 leaves remainder)"),
    ]
    results = [run_one(*c) for c in cases]
    print("=" * 60)
    n_pass = sum(results)
    print(f"Result: {n_pass}/{len(results)} cases passed.")
    if n_pass != len(results):
        raise SystemExit("FAIL: chunked merge does NOT equal single-pass FVD.")
    print(
        "PASS: chunked sufficient-statistics merge is mathematically "
        "equivalent to single-pass Frechet distance to <1e-6 relative tolerance."
    )


if __name__ == "__main__":
    main()
