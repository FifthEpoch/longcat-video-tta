#!/usr/bin/env python3
"""Compute per-video difficulty signals from conditioning frames.

For each video in --videos-dir, load the first --num-cond-frames, downscale
to --target-size, and compute pair-wise (adjacent-frame) signals that
distinguish smooth motion from cuts and rapid editing:

  - cut_count        : adjacent pairs with SSIM(t, t+1) < --ssim-cut-threshold
  - min_pair_ssim    : minimum SSIM across adjacent pairs
  - mean_pair_ssim   : mean SSIM across adjacent pairs
  - mean_motion      : mean L1 pixel distance between adjacent frames (in [0,1])
  - max_motion       : max  L1 pixel distance between adjacent frames
  - motion_std       : std of L1 pixel distance (bursty motion vs steady)
  - max_hist_chi2    : max chi-squared distance between adjacent RGB hists
  - mean_hist_chi2   : mean chi-squared distance between adjacent RGB hists

Optionally joins with --gains-csv (produced by
diagnose_long_horizon_failures.py) and reports:

  - Pearson and Spearman correlations between each difficulty signal and
    each per-video gain (dPSNR / dSSIM / dLPIPS)
  - Mean gain by cut_count bucket (0 / 1 / 2+)
  - Mean gain by mean_motion quintile

Dependencies: numpy, opencv-python. skimage is used opportunistically for
multi-channel SSIM; a manual single-channel SSIM is used as a fallback.
"""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as _skimage_ssim
except ImportError:
    _skimage_ssim = None


SUPPORTED_EXTS = (".mp4", ".avi", ".mkv", ".mov", ".webm")


def load_first_n_frames(path: Path, n: int, target_size: int) -> Optional[np.ndarray]:
    """Decode the first n frames; returns (n, H, W, 3) uint8 RGB or None."""
    cap = cv2.VideoCapture(str(path))
    frames: List[np.ndarray] = []
    try:
        while len(frames) < n:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if target_size > 0 and (rgb.shape[0] != target_size or rgb.shape[1] != target_size):
                rgb = cv2.resize(rgb, (target_size, target_size),
                                 interpolation=cv2.INTER_AREA)
            frames.append(rgb)
    finally:
        cap.release()
    if len(frames) < n:
        return None
    return np.stack(frames, axis=0)


def _manual_ssim_gray(a: np.ndarray, b: np.ndarray) -> float:
    """Single-channel SSIM without windowing. Inputs are uint8 HxW arrays."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    var_a = a.var()
    var_b = b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    L = 255.0
    c1 = (0.01 * L) ** 2
    c2 = (0.03 * L) ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    return float(num / den) if den > 0 else 0.0


def pair_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """RGB SSIM in [-1, 1]. Uses skimage when available, manual gray SSIM otherwise."""
    if _skimage_ssim is not None:
        return float(_skimage_ssim(a, b, channel_axis=-1, data_range=255))
    a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
    return _manual_ssim_gray(a_gray, b_gray)


def pair_motion(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute pixel difference scaled to [0, 1]."""
    return float(np.abs(a.astype(np.float32) / 255.0
                        - b.astype(np.float32) / 255.0).mean())


def pair_hist_chi2(a: np.ndarray, b: np.ndarray, n_bins: int = 32) -> float:
    """Symmetric chi-squared histogram distance averaged over RGB channels."""
    out = 0.0
    eps = 1e-10
    for c in range(3):
        h1, _ = np.histogram(a[..., c], bins=n_bins, range=(0, 255))
        h2, _ = np.histogram(b[..., c], bins=n_bins, range=(0, 255))
        h1 = h1.astype(np.float64); h1 /= max(h1.sum(), 1.0)
        h2 = h2.astype(np.float64); h2 /= max(h2.sum(), 1.0)
        out += float(0.5 * np.sum(((h1 - h2) ** 2) / (h1 + h2 + eps)))
    return out / 3.0


def per_video_signals(frames: np.ndarray, ssim_cut_threshold: float) -> Dict[str, float]:
    """Aggregate signals from (N, H, W, 3) frames; N >= 2."""
    n_pairs = frames.shape[0] - 1
    ssims, motions, hists = [], [], []
    for t in range(n_pairs):
        ssims.append(pair_ssim(frames[t], frames[t + 1]))
        motions.append(pair_motion(frames[t], frames[t + 1]))
        hists.append(pair_hist_chi2(frames[t], frames[t + 1]))
    return {
        "n_pairs": n_pairs,
        "cut_count": sum(1 for s in ssims if s < ssim_cut_threshold),
        "min_pair_ssim": float(min(ssims)),
        "mean_pair_ssim": float(statistics.fmean(ssims)),
        "mean_motion": float(statistics.fmean(motions)),
        "max_motion": float(max(motions)),
        "motion_std": float(statistics.pstdev(motions) if n_pairs > 1 else 0.0),
        "max_hist_chi2": float(max(hists)),
        "mean_hist_chi2": float(statistics.fmean(hists)),
    }


def pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    sx, sy = x - x.mean(), y - y.mean()
    den = math.sqrt(float((sx * sx).sum()) * float((sy * sy).sum()))
    return float((sx * sy).sum() / den) if den > 0 else None


def spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3:
        return None
    def rank(arr: List[float]) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float64)
        order = np.argsort(a)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a))
        return ranks
    return pearson(rank(xs).tolist(), rank(ys).tolist())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos-dir", required=True, type=Path,
                   help="Directory containing video files.")
    p.add_argument("--out-csv", required=True, type=Path,
                   help="Output CSV with per-video signals.")
    p.add_argument("--gains-csv", type=Path, default=None,
                   help="Optional per-video gains CSV to join and correlate.")
    p.add_argument("--num-cond-frames", type=int, default=14,
                   help="Leading frames per video to analyze (default 14, matches AdaSteer setup).")
    p.add_argument("--target-size", type=int, default=224,
                   help="Resize to (target-size)x(target-size) for speed (default 224).")
    p.add_argument("--ssim-cut-threshold", type=float, default=0.5,
                   help="Adjacent SSIM below this counts as a cut (default 0.5).")
    p.add_argument("--max-videos", type=int, default=None,
                   help="Optional cap for quick testing.")
    p.add_argument("--progress-every", type=int, default=50,
                   help="Print progress every N videos.")
    args = p.parse_args()

    if not args.videos_dir.is_dir():
        raise SystemExit(f"--videos-dir not a directory: {args.videos_dir}")

    paths = sorted(p for p in args.videos_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS)
    if args.max_videos is not None:
        paths = paths[: args.max_videos]
    if not paths:
        raise SystemExit(f"No video files in {args.videos_dir}")

    print(f"Processing {len(paths)} videos from {args.videos_dir}")
    if _skimage_ssim is None:
        print("[note] skimage not available; using manual gray SSIM (slightly less accurate)")

    rows: List[Dict[str, float]] = []
    skipped = 0
    for i, vp in enumerate(paths, 1):
        frames = load_first_n_frames(vp, n=args.num_cond_frames,
                                     target_size=args.target_size)
        if frames is None:
            skipped += 1
            continue
        sig = per_video_signals(frames, ssim_cut_threshold=args.ssim_cut_threshold)
        sig["video"] = vp.stem
        rows.append(sig)
        if i == 1 or i % args.progress_every == 0 or i == len(paths):
            print(f"  [{i}/{len(paths)}] {vp.name}: "
                  f"cuts={sig['cut_count']} min_ssim={sig['min_pair_ssim']:.3f} "
                  f"mean_motion={sig['mean_motion']:.4f}")

    if not rows:
        raise SystemExit("No valid videos processed.")
    print(f"\nProcessed {len(rows)} videos ({skipped} skipped for insufficient frames).")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["video", "n_pairs", "cut_count", "min_pair_ssim",
                  "mean_pair_ssim", "mean_motion", "max_motion", "motion_std",
                  "max_hist_chi2", "mean_hist_chi2"]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
    print(f"Wrote per-video signals: {args.out_csv}")

    if args.gains_csv is None:
        return
    if not args.gains_csv.exists():
        print(f"\n[warn] --gains-csv {args.gains_csv} does not exist; skipping correlation analysis")
        return

    with args.gains_csv.open() as f:
        gains_rows = list(csv.DictReader(f))
    by_video = {r["video"]: r for r in gains_rows}
    joined: List[Dict[str, float]] = []
    for r in rows:
        g = by_video.get(r["video"])
        if not g:
            continue
        merged = dict(r)
        for k in ("dpsnr", "dssim", "dlpips"):
            v = g.get(k)
            try:
                merged[k] = float(v) if v not in ("", None) else None
            except (TypeError, ValueError):
                merged[k] = None
        merged["theme"] = g.get("theme", "")
        joined.append(merged)
    print(f"\nJoined {len(joined)} videos with gains CSV")

    signal_keys = ["cut_count", "min_pair_ssim", "mean_pair_ssim",
                   "mean_motion", "max_motion", "motion_std",
                   "max_hist_chi2", "mean_hist_chi2"]
    target_keys = ["dpsnr", "dssim", "dlpips"]

    def _corr_table(corr_fn, name: str) -> None:
        print(f"\n=== {name} (signal vs delta) ===")
        header = f"  {'signal':<18s}" + "".join(f"  {k:>10s}" for k in target_keys)
        print(header)
        for sk in signal_keys:
            line = f"  {sk:<18s}"
            for tk in target_keys:
                pairs = [(r[sk], r[tk]) for r in joined
                         if r.get(sk) is not None and r.get(tk) is not None]
                if len(pairs) < 3:
                    line += f"  {'n/a':>10s}"
                    continue
                xs = [pp[0] for pp in pairs]
                ys = [pp[1] for pp in pairs]
                c = corr_fn(xs, ys)
                line += f"  {c:+.4f}".rjust(12) if c is not None else f"  {'n/a':>10s}"
            print(line)

    _corr_table(pearson, "Pearson r")
    _corr_table(spearman, "Spearman rho")

    print("\n=== Mean delta by cut_count bucket ===")
    cut_buckets: Dict[str, List[Dict]] = defaultdict(list)
    for r in joined:
        cc = int(r.get("cut_count") or 0)
        key = "0" if cc == 0 else ("1" if cc == 1 else "2+")
        cut_buckets[key].append(r)
    print(f"  {'cuts':<6s}  {'N':>5s}  {'dPSNR':>10s}  {'dSSIM':>10s}  {'dLPIPS':>10s}")
    for key in ("0", "1", "2+"):
        items = cut_buckets.get(key, [])
        if not items:
            continue
        means = {}
        for tk in target_keys:
            vs = [r[tk] for r in items if r.get(tk) is not None]
            means[tk] = statistics.fmean(vs) if vs else None
        fp = lambda v: f"{v:+.4f}" if v is not None else "  n/a   "
        print(f"  {key:<6s}  {len(items):>5d}  "
              f"{fp(means['dpsnr']):>10s}  {fp(means['dssim']):>10s}  {fp(means['dlpips']):>10s}")

    motions = sorted(r["mean_motion"] for r in joined)
    n = len(motions)
    if n >= 5:
        edges = [motions[int(n * q)] for q in (0.2, 0.4, 0.6, 0.8)]
        def bucket(v: float) -> int:
            for i, e in enumerate(edges):
                if v <= e:
                    return i
            return len(edges)
        motion_buckets: Dict[int, List[Dict]] = defaultdict(list)
        for r in joined:
            motion_buckets[bucket(r["mean_motion"])].append(r)
        print("\n=== Mean delta by mean_motion quintile ===")
        print(f"  {'quint':<6s}  {'range':>22s}  {'N':>5s}  "
              f"{'dPSNR':>10s}  {'dSSIM':>10s}  {'dLPIPS':>10s}")
        edges_full = [motions[0]] + edges + [motions[-1]]
        for i in range(5):
            items = motion_buckets.get(i, [])
            if not items:
                continue
            rng = f"[{edges_full[i]:.4f}, {edges_full[i+1]:.4f}]"
            means = {}
            for tk in target_keys:
                vs = [r[tk] for r in items if r.get(tk) is not None]
                means[tk] = statistics.fmean(vs) if vs else None
            fp = lambda v: f"{v:+.4f}" if v is not None else "  n/a   "
            print(f"  Q{i+1:<5d}  {rng:>22s}  {len(items):>5d}  "
                  f"{fp(means['dpsnr']):>10s}  {fp(means['dssim']):>10s}  {fp(means['dlpips']):>10s}")

    out_joined = args.out_csv.with_suffix(".joined.csv")
    joined_fields = ["video", "theme"] + signal_keys + target_keys
    with out_joined.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=joined_fields)
        w.writeheader()
        for r in joined:
            w.writerow({k: r.get(k) for k in joined_fields})
    print(f"\nWrote joined CSV: {out_joined}")


if __name__ == "__main__":
    main()
