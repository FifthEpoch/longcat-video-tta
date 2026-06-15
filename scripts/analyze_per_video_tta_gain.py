#!/usr/bin/env python3
"""Per-video TTA-gain analysis: who wins, who loses, and what predicts it.

Background: the headline ``panda_1000v_standard`` table shows AdaSteer ≈
NoTTA at the population level (ΔPSNR ≈ 0). A discovery smoke run on a
single chunk_0 of the no-prompt ADA ablation looked +0.68 dB better than
the headline ADA but turned out to be sampling noise after all 10 chunks
were merged (ΔPSNR ≈ 0 again). That raises the diagnostic question this
script answers:

    Do NO videos benefit from TTA at all, or do roughly equal numbers
    of videos win big and lose big, and the mean washes out? If the
    latter, what video-level features predict the winners?

For every TTA method under a series root, compute per-video
ΔPSNR = method_psnr − baseline_psnr (default baseline ``NOTTA``), join
with per-video dynamicness (RAFT mean optical flow) and per-video
captions from ``metadata.csv``, then emit:

  (a) ``per_video_gains.csv``                 (the raw long-format table)
  (b) ``delta_psnr_vs_dynamicness.png``       (ΔPSNR vs RAFT mean-flow quintile)
  (c) ``delta_psnr_vs_baseline_psnr.png``     (per-method scatter + regression)
  (d) ``delta_psnr_histogram.png``            (overlaid per-video ΔPSNR histograms)
  (e) ``delta_psnr_vs_caption_length.png``    (ΔPSNR vs caption-words quintile)
  (f) ``summary.md``                          (tails, top winners/losers,
                                               Pearson + Spearman correlations)

This is a SIBLING of ``scripts/plot_dynamicness_correlation.py`` (which
plots raw per-method metric curves vs dynamicness bins). It is not a
replacement: the existing script is per-bin per-metric for headline
methods only, while this one focuses on ΔPSNR distributions and adds
caption-length / baseline-PSNR axes that the older script does not have.

Standard cluster usage (the user runs this; do NOT submit slurm jobs):

    python3 scripts/analyze_per_video_tta_gain.py \
        --series-path sweep_experiment/results/panda_1000v_standard \
        --tinylora-series-path delta_experiment/results/tinylora_panda_1000v_standard \
        --output-dir sweep_experiment/reports/per_video_analysis/$(date +%Y-%m-%d)

The script auto-detects every method subdir (under either series root)
that has at least one ``chunk_*/summary.json`` (or a flat
``merged_summary.json`` as fallback). Pass ``--methods`` to restrict.

Dependencies: numpy, matplotlib. No pandas / seaborn / scipy required.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.caption_utils import load_resolved_captions_csv


# ---------------------------------------------------------------------------
# Canonical video-id extraction (mirrors plot_dynamicness_correlation.py)
# ---------------------------------------------------------------------------
# Generated clips look like ``panda_0010_delta_a.mp4``; source clips are
# ``panda_0010.mp4``. We strip directory, extension, and method suffix down
# to ``<prefix>_<number>`` so the metric rows, dynamicness rows and
# metadata.csv rows all join on the same key.
_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def _canonical_video_id(s: Optional[str]) -> str:
    if not s:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


# ---------------------------------------------------------------------------
# Per-video metric loading
# ---------------------------------------------------------------------------
def _coerce_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _records_from_blob(blob) -> List[dict]:
    """Pull the per-video record list out of a summary.json / merged_summary.json."""
    if isinstance(blob, list):
        return [r for r in blob if isinstance(r, dict)]
    if not isinstance(blob, dict):
        return []
    for key in ("results", "per_video_results", "per_video"):
        v = blob.get(key)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
    return []


def load_per_video_metrics(method_dir: Path) -> Dict[str, Dict[str, Optional[float]]]:
    """Return {canonical_video_id -> {psnr, ssim, lpips}}.

    Scans ``chunk_*/summary.json`` first. Falls back to ``chunk_*/results.json``,
    then to the flat ``merged_summary.json`` / ``summary.json`` at the method
    root (whose per-video records also live under the ``results`` key).

    Later chunk records overwrite earlier ones if a video id appears twice
    (which should not happen for paper-grade series but is harmless).
    """
    candidates: List[Path] = sorted(method_dir.glob("chunk_*/summary.json"))
    if not candidates:
        candidates = sorted(method_dir.glob("chunk_*/results.json"))
    if not candidates:
        for flat_name in ("merged_summary.json", "summary.json"):
            flat = method_dir / flat_name
            if flat.exists():
                candidates = [flat]
                break

    pv: Dict[str, Dict[str, Optional[float]]] = {}
    for cf in candidates:
        try:
            with cf.open() as f:
                blob = json.load(f)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {cf}: {e}", file=sys.stderr)
            continue
        for r in _records_from_blob(blob):
            vid_raw = (r.get("video_name")
                       or r.get("video_id")
                       or r.get("video")
                       or r.get("video_path")
                       or r.get("path"))
            vid = _canonical_video_id(vid_raw if vid_raw is not None else "")
            if not vid:
                continue
            pv[vid] = {
                "psnr":  _coerce_float(r.get("psnr",  r.get("avg_psnr"))),
                "ssim":  _coerce_float(r.get("ssim",  r.get("avg_ssim"))),
                "lpips": _coerce_float(r.get("lpips", r.get("avg_lpips"))),
            }
    return pv


def autodiscover_methods(series_path: Path) -> List[str]:
    """Return alphabetised method subdir names under ``series_path`` that have
    at least one ``chunk_*/summary.json`` or a flat ``merged_summary.json``."""
    if not series_path.exists():
        return []
    out: List[str] = []
    for sub in sorted(p for p in series_path.iterdir() if p.is_dir()):
        if (sub / "merged_summary.json").exists() or (sub / "summary.json").exists():
            out.append(sub.name)
            continue
        if any(sub.glob("chunk_*/summary.json")):
            out.append(sub.name)
            continue
        if any(sub.glob("chunk_*/results.json")):
            out.append(sub.name)
    return out


# ---------------------------------------------------------------------------
# Sidecar (dynamicness + captions) loading
# ---------------------------------------------------------------------------
def load_dynamicness(path: Path, flow_key: str = "mean_flow") -> Dict[str, float]:
    """Return {canonical_video_id -> mean_flow}. Skips entries with errors."""
    if not path.exists():
        print(f"[warn] dynamicness JSON not found at {path}; mean_flow will be NaN",
              file=sys.stderr)
        return {}
    with path.open() as f:
        blob = json.load(f)
    videos = blob.get("videos", {})
    out: Dict[str, float] = {}
    for vid, info in videos.items():
        if not isinstance(info, dict) or "error" in info:
            continue
        v = info.get(flow_key)
        if v is None:
            continue
        try:
            out[_canonical_video_id(vid)] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def load_captions(path: Path) -> Dict[str, str]:
    """Return {canonical_video_id -> resolved caption}. Tolerant of missing files."""
    return load_resolved_captions_csv(path, canonical_id=_canonical_video_id)


# ---------------------------------------------------------------------------
# Stats helpers (stdlib + numpy, no scipy)
# ---------------------------------------------------------------------------
def pearson_r(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3:
        return None
    x = xs[mask].astype(np.float64)
    y = ys[mask].astype(np.float64)
    sx, sy = x - x.mean(), y - y.mean()
    den = math.sqrt(float((sx * sx).sum()) * float((sy * sy).sum()))
    if den <= 0:
        return None
    return float((sx * sy).sum() / den)


def spearman_rho(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3:
        return None
    def _ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(a.size, dtype=np.float64)
        # average ties
        uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sum_ranks = np.zeros(uniq.size, dtype=np.float64)
            np.add.at(sum_ranks, inv, ranks)
            avg_ranks = sum_ranks / counts
            ranks = avg_ranks[inv]
        return ranks
    return pearson_r(_ranks(xs[mask]), _ranks(ys[mask]))


def quantile_bin_assign(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_index_per_sample, unique-edges). Identical idea to the
    sibling script: handles ties by collapsing duplicate edges."""
    if values.size == 0:
        return np.zeros(0, dtype=int), np.array([])
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros_like(values, dtype=int), edges
    idx = np.clip(np.searchsorted(edges[1:-1], values, side="right"),
                  0, edges.size - 2)
    return idx, edges


def bin_means_sem(
    y: np.ndarray, bin_idx: np.ndarray, n_bins: int,
    x_for_centers: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (bin_centers, bin_mean, bin_sem, bin_count) over y, gated by NaN."""
    centers = np.full(n_bins, np.nan)
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    if x_for_centers is None:
        x_for_centers = bin_idx.astype(np.float64)
    for b in range(n_bins):
        mask = (bin_idx == b) & ~np.isnan(y)
        n = int(mask.sum())
        if n == 0:
            continue
        ys = y[mask]
        centers[b] = float(np.mean(x_for_centers[mask]))
        means[b] = float(np.mean(ys))
        sems[b] = float(np.std(ys, ddof=1) / math.sqrt(n)) if n > 1 else 0.0
        counts[b] = n
    return centers, means, sems, counts


def linear_fit(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (slope, intercept) for a 1-D least-squares fit, ignoring NaN."""
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return None
    xs = x[mask].astype(np.float64)
    ys = y[mask].astype(np.float64)
    x_mean = xs.mean()
    y_mean = ys.mean()
    num = float(((xs - x_mean) * (ys - y_mean)).sum())
    den = float(((xs - x_mean) ** 2).sum())
    if den <= 0:
        return None
    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope, intercept


def topk_indices(delta: np.ndarray, k: int, *, lower_is_better: bool) -> np.ndarray:
    """Return the indices of the top-k rows of ``delta`` (NaN-safe).

    ``lower_is_better=False`` -> winners are largest values (e.g. ΔPSNR > 0).
    ``lower_is_better=True``  -> winners are smallest values (e.g. ΔLPIPS < 0).
    """
    finite = np.where(~np.isnan(delta))[0]
    if finite.size == 0:
        return np.empty(0, dtype=int)
    vals = delta[finite]
    sign = 1.0 if lower_is_better else -1.0
    order = finite[np.argsort(sign * vals, kind="mergesort")]
    return order[: min(k, order.size)]


def jaccard_matrix_topk(
    delta_by_method: Dict[str, np.ndarray], k: int, *,
    lower_is_better: bool = False,
) -> Tuple[List[str], np.ndarray]:
    """Cross-method top-k winner Jaccard matrix.

    Returns (method_names, J) where J[i,j] = |top-k(i) ∩ top-k(j)| / |top-k(i) ∪ top-k(j)|.
    Empty union -> NaN. Random-overlap baseline for two top-k sets out of N
    videos is k/(2N-k) -> Jaccard ≈ k/(2N) when k << N. For paper output we
    contrast the observed Jaccard against this baseline.
    """
    methods = list(delta_by_method.keys())
    tops: Dict[str, set] = {}
    for m in methods:
        idxs = topk_indices(delta_by_method[m], k, lower_is_better=lower_is_better)
        tops[m] = set(idxs.tolist())
    n = len(methods)
    J = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a = tops[methods[i]]
            b = tops[methods[j]]
            u = a | b
            if not u:
                continue
            J[i, j] = len(a & b) / len(u)
    return methods, J


def sign_agreement_stats(
    delta_by_method: Dict[str, np.ndarray], *,
    favourable_sign: int = +1,
) -> dict:
    """Count how often the per-video ΔPSNR sign agrees across ALL methods.

    ``favourable_sign=+1`` -> "improved" means delta > 0 (PSNR / SSIM).
    ``favourable_sign=-1`` -> "improved" means delta < 0 (LPIPS).

    Returns a dict with both the observed unanimous counts and the
    independence-baseline expectation (used as the "Nx lift" headline number
    in the 2026-06-09 analysis).
    """
    methods = list(delta_by_method.keys())
    if not methods:
        return {"methods": [], "n_eval": 0}
    arrs = [delta_by_method[m] for m in methods]
    M = np.column_stack(arrs)
    finite_mask = ~np.isnan(M).any(axis=1)
    Mf = M[finite_mask]
    if Mf.size == 0:
        return {"methods": methods, "n_eval": 0}

    if favourable_sign > 0:
        win = Mf > 0.0
        lose = Mf < 0.0
    else:
        win = Mf < 0.0
        lose = Mf > 0.0

    all_win = win.all(axis=1)
    all_lose = lose.all(axis=1)
    unanimous = all_win | all_lose

    p_win = win.mean(axis=0)
    p_lose = lose.mean(axis=0)
    expected_all_win_frac = float(np.prod(p_win))
    expected_all_lose_frac = float(np.prod(p_lose))
    expected_unanimous_frac = expected_all_win_frac + expected_all_lose_frac

    n_eval = int(Mf.shape[0])
    obs_unanimous = int(unanimous.sum())
    exp_unanimous = expected_unanimous_frac * n_eval
    lift = (obs_unanimous / exp_unanimous) if exp_unanimous > 0 else float("inf")

    return {
        "methods": methods,
        "n_eval": n_eval,
        "p_win_per_method": {m: float(p_win[i]) for i, m in enumerate(methods)},
        "p_lose_per_method": {m: float(p_lose[i]) for i, m in enumerate(methods)},
        "n_all_win":  int(all_win.sum()),
        "n_all_lose": int(all_lose.sum()),
        "n_unanimous": obs_unanimous,
        "expected_n_all_win_under_indep":  expected_all_win_frac * n_eval,
        "expected_n_all_lose_under_indep": expected_all_lose_frac * n_eval,
        "expected_n_unanimous_under_indep": exp_unanimous,
        "lift_unanimous": lift,
        "favourable_sign": int(favourable_sign),
    }


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--series-path", type=Path,
        default=Path("sweep_experiment/results/panda_1000v_standard"),
        help="Primary series root (one subdir per method).",
    )
    ap.add_argument(
        "--tinylora-series-path", type=Path,
        default=Path("delta_experiment/results/tinylora_panda_1000v_standard"),
        help="Optional second series root (for TL_BARE_R2 / TL_TIED_R2 and "
             "their _NOPROMPT siblings). Pass an empty string to disable.",
    )
    ap.add_argument(
        "--baseline-method", default="NOTTA",
        help="Method name used as baseline for ΔPSNR / ΔSSIM / ΔLPIPS.",
    )
    ap.add_argument(
        "--methods", nargs="*", default=None,
        help="Explicit list of method subdir names. Default: auto-detect every "
             "subdir under --series-path AND --tinylora-series-path that has "
             "at least one chunk_*/summary.json (or a merged_summary.json).",
    )
    ap.add_argument(
        "--dynamicness-json", type=Path,
        default=Path("datasets/panda_1000_480p/dynamic_degree.json"),
        help="Output JSON from scripts/compute_dynamic_degree.py (RAFT flows).",
    )
    ap.add_argument(
        "--captions-csv", type=Path,
        default=Path("datasets/panda_1000_480p/metadata.csv"),
        help="Panda-style metadata.csv with at least filename + caption columns.",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=Path("sweep_experiment/reports/per_video_analysis/_unspecified_date"),
        help="Where to write per_video_gains.csv, the four plots, and summary.md.",
    )
    ap.add_argument("--n-bins", type=int, default=5,
                    help="Number of dynamicness / caption-length quantile bins.")
    ap.add_argument("--top-k", type=int, default=10,
                    help="How many top winners / losers to include in summary.md.")
    ap.add_argument(
        "--tails-thresholds", nargs="+", type=float,
        default=[1.0, 0.5],
        help="Per-method tail thresholds (in dB) listed in summary.md as "
             "|Δ|>t. Default: 1.0 and 0.5.",
    )
    ap.add_argument(
        "--flow-key", default="mean_flow", choices=["mean_flow", "max_flow"],
        help="Which dynamicness scalar to use (matches "
             "plot_dynamicness_correlation.py).",
    )
    return ap.parse_args()


def _resolve_method_paths(
    series_path: Path,
    tinylora_series_path: Optional[Path],
    explicit_methods: Optional[List[str]],
) -> List[Tuple[str, Path]]:
    """Return [(method_name, method_dir)] using auto-detect if methods is None.

    When explicit methods are given, they are looked up in both roots: the
    primary root wins on collision.
    """
    if explicit_methods:
        out: List[Tuple[str, Path]] = []
        for name in explicit_methods:
            cand_primary = series_path / name
            if cand_primary.exists():
                out.append((name, cand_primary))
                continue
            if tinylora_series_path and (tinylora_series_path / name).exists():
                out.append((name, tinylora_series_path / name))
                continue
            print(f"[warn] requested method {name} not found under "
                  f"{series_path} or {tinylora_series_path}; skipping",
                  file=sys.stderr)
        return out

    seen: Dict[str, Path] = {}
    for name in autodiscover_methods(series_path):
        seen.setdefault(name, series_path / name)
    if tinylora_series_path is not None:
        for name in autodiscover_methods(tinylora_series_path):
            seen.setdefault(name, tinylora_series_path / name)
    return sorted(seen.items(), key=lambda kv: kv[0])


def _truncate(s: str, n: int = 80) -> str:
    s = (s or "").replace("\n", " ").replace("\r", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _color_cycle(plt, n: int) -> List:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def plot_delta_psnr_vs_bins(
    plt, out_path: Path, methods: List[str], data: Dict[str, np.ndarray],
    bin_x: np.ndarray, n_bins: int, x_label: str, log_x: bool = False,
    title: str = "",
):
    """Bin per-video ΔPSNR by ``bin_x`` quintile; one line per method.

    ``data[method]`` is the per-video ΔPSNR vector aligned to ``bin_x``;
    rows where ``bin_x`` is NaN are excluded BEFORE quantile-edge selection
    (otherwise NaNs would propagate into edges and silently break the
    binning), and rows where ΔPSNR is NaN are dropped per-bin inside
    ``bin_means_sem``.
    """
    valid_x = ~np.isnan(bin_x)
    if not valid_x.any():
        print(f"[warn] {out_path.name}: bin_x is all NaN; skipping plot",
              file=sys.stderr)
        return None
    bin_x_clean = bin_x[valid_x]
    bin_idx_clean, edges = quantile_bin_assign(bin_x_clean, n_bins)
    n_bins_eff = max(int(bin_idx_clean.max()) + 1, 1) if bin_idx_clean.size else 1
    # broadcast clean bin_idx back to the full-length array with -1 sentinel
    bin_idx = np.full(bin_x.shape, -1, dtype=int)
    bin_idx[valid_x] = bin_idx_clean

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    colors = _color_cycle(plt, len(methods))
    counts_per_bin: List[int] = [0] * n_bins_eff
    for color, name in zip(colors, methods):
        y = data[name]
        centers, means, sems, counts = bin_means_sem(
            y, bin_idx, n_bins_eff, x_for_centers=bin_x,
        )
        valid = ~np.isnan(means)
        ax.errorbar(centers[valid], means[valid], yerr=sems[valid],
                    marker="o", capsize=2, linewidth=1.6,
                    label=name, color=color)
        for b in range(n_bins_eff):
            counts_per_bin[b] = max(counts_per_bin[b], int(counts[b]))
    # annotate per-bin sample counts on the x-axis (after artists exist so
    # ylim is data-driven)
    annot_y = ax.get_ylim()[0]
    for b in range(n_bins_eff):
        mask = (bin_idx == b)
        if mask.any() and counts_per_bin[b] > 0:
            cx = float(np.mean(bin_x[mask]))
            ax.annotate(f"n={counts_per_bin[b]}", (cx, annot_y),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=7, color="grey")
    if log_x:
        positive = bin_x[(bin_x > 0) & ~np.isnan(bin_x)]
        if positive.size > 1 and positive.max() / max(positive.min(), 1e-6) > 5:
            ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(r"per-video $\Delta$PSNR vs baseline (dB)")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return edges


def plot_delta_psnr_vs_baseline(
    plt, out_path: Path, methods: List[str],
    baseline_psnr: np.ndarray, delta_by_method: Dict[str, np.ndarray],
    title: str = "",
):
    """One subplot per method: per-video (baseline PSNR, ΔPSNR) scatter + LS fit."""
    n = len(methods)
    if n == 0:
        return
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 3.6 * nrows),
        squeeze=False, sharex=True, sharey=True,
    )
    for k, name in enumerate(methods):
        ax = axes[k // ncols][k % ncols]
        d = delta_by_method[name]
        mask = ~(np.isnan(baseline_psnr) | np.isnan(d))
        x = baseline_psnr[mask]
        y = d[mask]
        ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.scatter(x, y, s=8, alpha=0.45, color="tab:blue", edgecolor="none")
        fit = linear_fit(x, y)
        r = pearson_r(x, y)
        if fit is not None:
            slope, intercept = fit
            xs_line = np.linspace(x.min(), x.max(), 64) if x.size else np.array([])
            if xs_line.size:
                ax.plot(xs_line, slope * xs_line + intercept,
                        color="tab:red", linewidth=1.4,
                        label=f"slope={slope:+.3f}  r={r:+.3f}"
                              if r is not None else f"slope={slope:+.3f}")
                ax.legend(loc="best")
        ax.set_title(f"{name}  (N={int(mask.sum())})")
        if k // ncols == nrows - 1:
            ax.set_xlabel("baseline PSNR (dB)")
        if k % ncols == 0:
            ax.set_ylabel(r"$\Delta$PSNR (dB)")
    # blank out unused axes
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)
    if title:
        fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 1, 0.96] if title else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_delta_psnr_histogram(
    plt, out_path: Path, methods: List[str],
    delta_by_method: Dict[str, np.ndarray], title: str = "",
):
    """Overlaid translucent histograms — distribution shape per method."""
    n = len(methods)
    if n == 0:
        return
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.axvline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    # shared bin edges so the overlay is fair
    all_finite: List[float] = []
    for name in methods:
        d = delta_by_method[name]
        all_finite.extend(d[~np.isnan(d)].tolist())
    if not all_finite:
        plt.close(fig)
        return
    lo, hi = float(np.percentile(all_finite, 1)), float(np.percentile(all_finite, 99))
    span = max(hi - lo, 1e-3)
    pad = 0.05 * span
    edges = np.linspace(lo - pad, hi + pad, 41)
    colors = _color_cycle(plt, n)
    for color, name in zip(colors, methods):
        d = delta_by_method[name]
        d = d[~np.isnan(d)]
        if d.size == 0:
            continue
        ax.hist(d, bins=edges, alpha=0.45, color=color,
                label=f"{name}  μ={d.mean():+.3f}  med={np.median(d):+.3f}  N={d.size}",
                edgecolor="black", linewidth=0.4)
    ax.set_xlabel(r"per-video $\Delta$PSNR vs baseline (dB)")
    ax.set_ylabel("# videos")
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary markdown
# ---------------------------------------------------------------------------
def write_summary_md(
    out_path: Path, args: argparse.Namespace, baseline_name: str,
    methods: List[str], rows: List[dict],
    delta_by_method: Dict[str, np.ndarray],
    mean_flow: np.ndarray, baseline_psnr: np.ndarray,
    caption_word_count: np.ndarray, video_ids: List[str],
    captions: List[str], dropped_per_method: Dict[str, int],
    intersection_size: int,
    intersection_missing_flow: int, intersection_missing_caption: int,
    delta_lpips_by_method: Optional[Dict[str, np.ndarray]] = None,
):
    lines: List[str] = []
    lines.append(f"# Per-video TTA-gain analysis  (baseline = {baseline_name})")
    lines.append("")
    lines.append(f"- Series: `{args.series_path}`")
    if args.tinylora_series_path:
        lines.append(f"- TinyLoRA series: `{args.tinylora_series_path}`")
    lines.append(f"- Dynamicness JSON: `{args.dynamicness_json}` "
                 f"(flow key = `{args.flow_key}`)")
    lines.append(f"- Captions CSV: `{args.captions_csv}`")
    lines.append(f"- Methods analysed: {', '.join(methods)}  "
                 f"(non-baseline: {', '.join(m for m in methods if m != baseline_name)})")
    lines.append(f"- Common video_id intersection (across baseline + every method): "
                 f"**{intersection_size}**")
    lines.append("")

    # ----- counts: dropped NaN PSNR rows + missing sidecar joins ------------
    lines.append("## Data integrity")
    lines.append("")
    lines.append(f"- Intersection rows missing `mean_flow`: "
                 f"**{intersection_missing_flow}** of {intersection_size}")
    lines.append(f"- Intersection rows missing caption: "
                 f"**{intersection_missing_caption}** of {intersection_size}")
    lines.append("")
    lines.append("| method | NaN-PSNR rows dropped before intersection |")
    lines.append("|---|---:|")
    for m in methods:
        lines.append(f"| `{m}` | {dropped_per_method.get(m, 0)} |")
    lines.append("")

    # ----- tail counts ------------------------------------------------------
    lines.append("## Per-method ΔPSNR tail counts")
    lines.append("")
    lines.append("Interpretation: large |Δ| tails mean TTA has real per-video "
                 "effects even when the population mean is ≈ 0. A symmetric "
                 "spread implies wins are paid for by equal-sized losses; a "
                 "right-skewed spread means TTA is a net positive on a subset.")
    lines.append("")
    thresholds = list(args.tails_thresholds)
    header = "| method | N | mean Δ | median Δ |"
    sep = "|---|---:|---:|---:|"
    for t in thresholds:
        header += f" Δ>+{t:.1f} | \\|Δ\\|≤{t:.1f} | Δ<−{t:.1f} |"
        sep += "---:|---:|---:|"
    lines.append(header)
    lines.append(sep)
    for m in methods:
        if m == baseline_name:
            continue
        delta = delta_by_method[m]
        d = delta[~np.isnan(delta)]
        n = int(d.size)
        mu = float(d.mean()) if n else float("nan")
        med = float(np.median(d)) if n else float("nan")
        row = f"| `{m}` | {n} | {mu:+.4f} | {med:+.4f} |"
        for t in thresholds:
            wins = int((d > t).sum())
            ties = int((np.abs(d) <= t).sum())
            losses = int((d < -t).sum())
            row += f" {wins} | {ties} | {losses} |"
        lines.append(row)
    lines.append("")

    # ----- correlations -----------------------------------------------------
    lines.append("## Correlation between ΔPSNR and per-video features")
    lines.append("")
    lines.append("Reported as Pearson r (Spearman ρ). Both use the intersection "
                 "of finite ΔPSNR with finite feature; per-method N may differ "
                 "slightly when individual rows have NaN features.")
    lines.append("")
    lines.append("| method | r(Δ, mean_flow) ρ | r(Δ, baseline PSNR) ρ | r(Δ, caption words) ρ |")
    lines.append("|---|---|---|---|")
    for m in methods:
        if m == baseline_name:
            continue
        d = delta_by_method[m]
        cells = []
        for feat in (mean_flow, baseline_psnr, caption_word_count):
            r = pearson_r(feat, d)
            rho = spearman_rho(feat, d)
            cells.append(
                f"{r:+.3f} ({rho:+.3f})"
                if r is not None and rho is not None
                else "n/a"
            )
        lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
    lines.append("")

    # ----- top winners / losers per method ----------------------------------
    lines.append(f"## Top {args.top_k} winners / losers per method")
    lines.append("")
    lines.append(f"Rank by ΔPSNR. Caption truncated to 80 chars; mean_flow / "
                 f"baseline PSNR pulled from this video's row in the intersection.")
    lines.append("")
    for m in methods:
        if m == baseline_name:
            continue
        d = delta_by_method[m]
        finite = np.where(~np.isnan(d))[0]
        if finite.size == 0:
            continue
        order_desc = finite[np.argsort(-d[finite], kind="mergesort")]
        order_asc = finite[np.argsort(d[finite], kind="mergesort")]
        for label, order in (("winners", order_desc[: args.top_k]),
                             ("losers", order_asc[: args.top_k])):
            lines.append(f"### `{m}` — top {len(order)} {label}")
            lines.append("")
            lines.append("| # | ΔPSNR | baseline PSNR | mean_flow | video_id | caption |")
            lines.append("|---:|---:|---:|---:|---|---|")
            for i, idx in enumerate(order, 1):
                vid = video_ids[idx]
                cap = _truncate(captions[idx], 80)
                bp = baseline_psnr[idx]
                mf = mean_flow[idx]
                bp_s = "n/a" if np.isnan(bp) else f"{bp:.2f}"
                mf_s = "n/a" if np.isnan(mf) else f"{mf:.3f}"
                lines.append(
                    f"| {i} | {d[idx]:+.3f} | {bp_s} | {mf_s} | `{vid}` | {cap} |"
                )
            lines.append("")

    # ----- ΔLPIPS tail breakdown (perceptual analog of ΔPSNR tails) ---------
    non_baseline = [m for m in methods if m != baseline_name]
    if delta_lpips_by_method:
        lines.append("## Per-method ΔLPIPS tail counts")
        lines.append("")
        lines.append("LPIPS is lower-is-better, so Δ<0 are winners (perceptual "
                     "improvement) and Δ>0 are losers. Thresholds are listed in "
                     "absolute LPIPS units (NOT dB); the headline ±0.005 band is "
                     "the per-video LPIPS noise floor on this eval set.")
        lines.append("")
        lpips_thresholds = [0.01, 0.005]
        header = "| method | N | mean Δ | median Δ |"
        sep = "|---|---:|---:|---:|"
        for t in lpips_thresholds:
            header += f" Δ<−{t:.3f} | \\|Δ\\|≤{t:.3f} | Δ>+{t:.3f} |"
            sep += "---:|---:|---:|"
        lines.append(header)
        lines.append(sep)
        for m in non_baseline:
            d = delta_lpips_by_method.get(m)
            if d is None:
                continue
            arr = d[~np.isnan(d)]
            n = int(arr.size)
            mu = float(arr.mean()) if n else float("nan")
            med = float(np.median(arr)) if n else float("nan")
            row = f"| `{m}` | {n} | {mu:+.5f} | {med:+.5f} |"
            for t in lpips_thresholds:
                wins = int((arr < -t).sum())
                ties = int((np.abs(arr) <= t).sum())
                losses = int((arr > t).sum())
                row += f" {wins} | {ties} | {losses} |"
            lines.append(row)
        lines.append("")

    # ----- Cross-method top-K winner Jaccard matrix (ΔPSNR) -----------------
    if len(non_baseline) >= 2:
        k_jacc = max(50, args.top_k)
        N_total = intersection_size
        baseline_overlap = k_jacc / (2 * max(N_total, 1) - k_jacc) if N_total > 0 else 0.0
        lines.append(f"## Cross-method top-{k_jacc} winner Jaccard matrix (ΔPSNR)")
        lines.append("")
        lines.append(f"Indices are the top-{k_jacc} videos by ΔPSNR for each "
                     f"non-baseline method (largest gains). Jaccard = "
                     f"|A∩B|/|A∪B|. Random-overlap baseline at this k and "
                     f"N={N_total} is **{baseline_overlap:.3f}** (k/(2N−k) under "
                     f"independent uniform sampling). Diagonal = 1.0 trivially.")
        lines.append("")
        jm_methods, J = jaccard_matrix_topk(
            {m: delta_by_method[m] for m in non_baseline},
            k=k_jacc, lower_is_better=False,
        )
        hdr = "| | " + " | ".join(f"`{m}`" for m in jm_methods) + " |"
        sep = "|---|" + "---:|" * len(jm_methods)
        lines.append(hdr)
        lines.append(sep)
        for i, m in enumerate(jm_methods):
            cells = []
            for j in range(len(jm_methods)):
                v = J[i, j]
                cells.append("n/a" if np.isnan(v) else f"{v:.3f}")
            lines.append(f"| `{m}` | " + " | ".join(cells) + " |")
        lines.append("")

        # off-diagonal mean to surface the "lift over random" headline number
        if J.shape[0] > 1:
            mask = ~np.eye(J.shape[0], dtype=bool)
            off_vals = J[mask]
            off_vals = off_vals[~np.isnan(off_vals)]
            if off_vals.size:
                mean_off = float(off_vals.mean())
                lift = mean_off / baseline_overlap if baseline_overlap > 0 else float("inf")
                lines.append(
                    f"- Off-diagonal mean Jaccard: **{mean_off:.3f}** "
                    f"({lift:.2f}× the random baseline {baseline_overlap:.3f})."
                )
                lines.append("")

    # ----- Sign-agreement across all non-baseline methods (ΔPSNR) -----------
    if len(non_baseline) >= 2:
        sa = sign_agreement_stats(
            {m: delta_by_method[m] for m in non_baseline},
            favourable_sign=+1,
        )
        if sa.get("n_eval", 0) > 0:
            lines.append("## Sign agreement across all non-baseline methods (ΔPSNR)")
            lines.append("")
            lines.append(
                f"For each of the {sa['n_eval']} videos with finite ΔPSNR for "
                f"all {len(non_baseline)} non-baseline methods, count how often "
                f"all methods agree on the sign of ΔPSNR. The independence "
                f"baseline is the product of per-method positive-fraction "
                f"(p_win). Observed-over-expected is the headline 'Nx lift' "
                f"number cited as evidence that the winning-subset story is "
                f"not method-specific noise."
            )
            lines.append("")
            lines.append("| method | p(Δ>0) | p(Δ<0) |")
            lines.append("|---|---:|---:|")
            for m in non_baseline:
                pw = sa["p_win_per_method"].get(m, float("nan"))
                pl = sa["p_lose_per_method"].get(m, float("nan"))
                lines.append(f"| `{m}` | {pw:.4f} | {pl:.4f} |")
            lines.append("")
            lines.append(
                f"- Videos where **all** methods improved (Δ>0): "
                f"**{sa['n_all_win']}** "
                f"(expected under independence: "
                f"{sa['expected_n_all_win_under_indep']:.2f}).\n"
                f"- Videos where **all** methods regressed (Δ<0): "
                f"**{sa['n_all_lose']}** "
                f"(expected under independence: "
                f"{sa['expected_n_all_lose_under_indep']:.2f}).\n"
                f"- Unanimous (either all-win or all-lose): "
                f"**{sa['n_unanimous']}** "
                f"(expected under independence: "
                f"{sa['expected_n_unanimous_under_indep']:.2f}; "
                f"lift = **{sa['lift_unanimous']:.2f}×**)."
            )
            lines.append("")

    # ----- Sign-agreement across all non-baseline methods (ΔLPIPS) ----------
    if delta_lpips_by_method and len(non_baseline) >= 2:
        sa_l = sign_agreement_stats(
            {m: delta_lpips_by_method[m] for m in non_baseline
             if delta_lpips_by_method.get(m) is not None},
            favourable_sign=-1,  # LPIPS lower-is-better
        )
        if sa_l.get("n_eval", 0) > 0:
            lines.append("## Sign agreement across all non-baseline methods (ΔLPIPS)")
            lines.append("")
            lines.append(
                "For LPIPS, lower-is-better, so 'win' means Δ<0. Otherwise the "
                "table reads identically to the ΔPSNR sign-agreement table above."
            )
            lines.append("")
            lines.append("| method | p(Δ<0, win) | p(Δ>0, lose) |")
            lines.append("|---|---:|---:|")
            for m in non_baseline:
                if delta_lpips_by_method.get(m) is None:
                    continue
                pw = sa_l["p_win_per_method"].get(m, float("nan"))
                pl = sa_l["p_lose_per_method"].get(m, float("nan"))
                lines.append(f"| `{m}` | {pw:.4f} | {pl:.4f} |")
            lines.append("")
            lines.append(
                f"- Unanimous-win (all methods improved LPIPS): "
                f"**{sa_l['n_all_win']}** "
                f"(expected under independence: "
                f"{sa_l['expected_n_all_win_under_indep']:.2f}).\n"
                f"- Unanimous-lose (all methods worsened LPIPS): "
                f"**{sa_l['n_all_lose']}** "
                f"(expected under independence: "
                f"{sa_l['expected_n_all_lose_under_indep']:.2f}).\n"
                f"- Unanimous total: **{sa_l['n_unanimous']}** "
                f"(expected: {sa_l['expected_n_unanimous_under_indep']:.2f}; "
                f"lift = **{sa_l['lift_unanimous']:.2f}×**)."
            )
            lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CSV emission
# ---------------------------------------------------------------------------
def write_long_csv(
    out_path: Path, methods: List[str], baseline_name: str,
    rows: List[dict],
):
    """Write per_video_gains.csv. One row per video_id in the intersection.

    Columns:
        video_id, caption, caption_len_chars, caption_len_words, mean_flow,
        <method>_psnr / _ssim / _lpips for every method,
        <method>_dpsnr / _dssim / _dlpips for every non-baseline method.
    """
    fieldnames = ["video_id", "caption", "caption_len_chars",
                  "caption_len_words", "mean_flow"]
    for m in methods:
        for metric in ("psnr", "ssim", "lpips"):
            fieldnames.append(f"{m}_{metric}")
    for m in methods:
        if m == baseline_name:
            continue
        for metric in ("psnr", "ssim", "lpips"):
            fieldnames.append(f"{m}_d{metric}")

    def fmt(v) -> str:
        if v is None:
            return ""
        if isinstance(v, float):
            if math.isnan(v):
                return ""
            return f"{v:.6f}"
        return str(v)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in fieldnames})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = _parse_args()
    # Empty-string / sentinel values disable the optional TinyLoRA root.
    # Path("") collapses to Path(".") which is almost certainly not what the
    # user meant; treat as "no second root".
    if args.tinylora_series_path is not None:
        s = str(args.tinylora_series_path).strip()
        if s in ("", ".", "none", "null", "None") or not args.tinylora_series_path.exists():
            if s not in ("", "none", "null", "None"):
                print(f"[info] tinylora series path {args.tinylora_series_path} "
                      f"does not exist; disabling second root",
                      file=sys.stderr)
            args.tinylora_series_path = None

    print(f"=== Per-video TTA-gain analysis ===")
    print(f"Series:           {args.series_path}")
    print(f"TinyLoRA series:  {args.tinylora_series_path}")
    print(f"Baseline:         {args.baseline_method}")
    print(f"Output:           {args.output_dir}")
    print()

    # ---- discover methods --------------------------------------------------
    method_specs = _resolve_method_paths(
        args.series_path, args.tinylora_series_path, args.methods,
    )
    if not method_specs:
        print("[error] no methods discovered — abort.", file=sys.stderr)
        return 2
    print("Discovered methods:")
    for name, mdir in method_specs:
        n_chunks = len(list(mdir.glob("chunk_*/summary.json")))
        flat = "yes" if (mdir / "merged_summary.json").exists() else "no"
        print(f"  {name:30s}  chunks={n_chunks:>2d}  merged={flat}  ({mdir})")
    method_names = [n for n, _ in method_specs]
    if args.baseline_method not in method_names:
        print(f"[error] baseline {args.baseline_method!r} not among discovered "
              f"methods {method_names}; pass --baseline-method or add the dir.",
              file=sys.stderr)
        return 2
    # put baseline first for prettier output ordering
    method_names = ([args.baseline_method]
                    + [m for m in method_names if m != args.baseline_method])
    method_dir_by_name = {n: p for n, p in method_specs}

    # ---- load per-video metrics per method ---------------------------------
    pv_by_method: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {}
    dropped_per_method: Dict[str, int] = {}
    for m in method_names:
        pv = load_per_video_metrics(method_dir_by_name[m])
        before = len(pv)
        pv = {vid: row for vid, row in pv.items() if row.get("psnr") is not None}
        dropped_per_method[m] = before - len(pv)
        if dropped_per_method[m]:
            print(f"  [info] {m}: dropped {dropped_per_method[m]} rows with NaN/missing PSNR")
        pv_by_method[m] = pv
        print(f"  loaded {m:30s}  per-video records: {len(pv)}")
    print()

    # ---- intersection across methods ---------------------------------------
    common_ids: Optional[set] = None
    for m, pv in pv_by_method.items():
        ids = set(pv.keys())
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = sorted(common_ids or set())
    print(f"Intersection across all methods (with finite baseline PSNR): "
          f"{len(common_ids)} videos")
    if not common_ids:
        print("[error] empty intersection — nothing to plot.", file=sys.stderr)
        return 2

    # ---- load sidecars -----------------------------------------------------
    flow_by_vid = load_dynamicness(args.dynamicness_json, args.flow_key)
    captions_by_vid = load_captions(args.captions_csv)
    print(f"Dynamicness entries: {len(flow_by_vid)}  "
          f"(flow key={args.flow_key})")
    print(f"Caption entries:     {len(captions_by_vid)}")

    flow_keys = set(flow_by_vid.keys())
    cap_keys = set(captions_by_vid.keys())
    intersection_missing_flow = sum(1 for v in common_ids if v not in flow_keys)
    intersection_missing_caption = sum(1 for v in common_ids if v not in cap_keys)
    if intersection_missing_flow:
        print(f"  [warn] {intersection_missing_flow}/{len(common_ids)} intersection "
              f"video_ids absent from dynamicness JSON (mean_flow=NaN for those rows)")
    if intersection_missing_caption:
        print(f"  [warn] {intersection_missing_caption}/{len(common_ids)} intersection "
              f"video_ids absent from captions CSV (caption empty for those rows)")

    # ---- assemble per-video rows ------------------------------------------
    rows: List[dict] = []
    for vid in common_ids:
        cap = captions_by_vid.get(vid, "") or ""
        cap_chars = len(cap)
        cap_words = len(cap.split())
        row: dict = {
            "video_id": vid,
            "caption": cap,
            "caption_len_chars": cap_chars,
            "caption_len_words": cap_words,
            "mean_flow": flow_by_vid.get(vid, float("nan")),
        }
        for m in method_names:
            mr = pv_by_method[m].get(vid, {})
            for metric in ("psnr", "ssim", "lpips"):
                v = mr.get(metric)
                row[f"{m}_{metric}"] = (
                    float("nan") if v is None else float(v)
                )
        base = row[f"{args.baseline_method}_psnr"]
        base_ssim = row[f"{args.baseline_method}_ssim"]
        base_lpips = row[f"{args.baseline_method}_lpips"]
        for m in method_names:
            if m == args.baseline_method:
                continue
            mp = row[f"{m}_psnr"]
            ms = row[f"{m}_ssim"]
            ml = row[f"{m}_lpips"]
            row[f"{m}_dpsnr"] = (
                float("nan") if (math.isnan(mp) or math.isnan(base)) else mp - base
            )
            row[f"{m}_dssim"] = (
                float("nan") if (math.isnan(ms) or math.isnan(base_ssim)) else ms - base_ssim
            )
            row[f"{m}_dlpips"] = (
                float("nan") if (math.isnan(ml) or math.isnan(base_lpips)) else ml - base_lpips
            )
        rows.append(row)

    # ---- assemble numpy arrays for plots -----------------------------------
    video_ids = [r["video_id"] for r in rows]
    captions = [r["caption"] for r in rows]
    mean_flow = np.array([r["mean_flow"] for r in rows], dtype=float)
    baseline_psnr = np.array(
        [r[f"{args.baseline_method}_psnr"] for r in rows], dtype=float,
    )
    caption_word_count = np.array(
        [r["caption_len_words"] for r in rows], dtype=float,
    )

    non_baseline = [m for m in method_names if m != args.baseline_method]
    delta_by_method: Dict[str, np.ndarray] = {
        m: np.array([r[f"{m}_dpsnr"] for r in rows], dtype=float)
        for m in non_baseline
    }
    delta_lpips_by_method: Dict[str, np.ndarray] = {
        m: np.array([r[f"{m}_dlpips"] for r in rows], dtype=float)
        for m in non_baseline
    }

    # ---- write CSV ---------------------------------------------------------
    csv_path = args.output_dir / "per_video_gains.csv"
    write_long_csv(csv_path, method_names, args.baseline_method, rows)
    print(f"\nWrote {csv_path}  ({len(rows)} rows, {len(method_names)} methods)")

    # ---- plots -------------------------------------------------------------
    plt = _setup_matplotlib()

    # (b) ΔPSNR vs dynamicness quintile
    title_b = f"Per-video ΔPSNR vs dynamicness — {args.series_path.name}"
    plot_delta_psnr_vs_bins(
        plt, args.output_dir / "delta_psnr_vs_dynamicness.png",
        non_baseline, delta_by_method,
        bin_x=np.where(np.isnan(mean_flow), 0.0, mean_flow),
        n_bins=args.n_bins,
        x_label=f"dynamicness ({args.flow_key}, RAFT)",
        log_x=True, title=title_b,
    )
    print(f"Wrote {args.output_dir / 'delta_psnr_vs_dynamicness.png'}")

    # (c) ΔPSNR vs baseline PSNR scatter
    title_c = f"Per-video ΔPSNR vs baseline (NOTTA) PSNR — {args.series_path.name}"
    plot_delta_psnr_vs_baseline(
        plt, args.output_dir / "delta_psnr_vs_baseline_psnr.png",
        non_baseline, baseline_psnr, delta_by_method, title=title_c,
    )
    print(f"Wrote {args.output_dir / 'delta_psnr_vs_baseline_psnr.png'}")

    # (d) ΔPSNR histograms overlaid
    title_d = f"Per-video ΔPSNR distributions — {args.series_path.name}"
    plot_delta_psnr_histogram(
        plt, args.output_dir / "delta_psnr_histogram.png",
        non_baseline, delta_by_method, title=title_d,
    )
    print(f"Wrote {args.output_dir / 'delta_psnr_histogram.png'}")

    # (e) ΔPSNR vs caption-words quintile
    title_e = f"Per-video ΔPSNR vs caption word count — {args.series_path.name}"
    plot_delta_psnr_vs_bins(
        plt, args.output_dir / "delta_psnr_vs_caption_length.png",
        non_baseline, delta_by_method,
        bin_x=caption_word_count, n_bins=args.n_bins,
        x_label="caption word count",
        log_x=False, title=title_e,
    )
    print(f"Wrote {args.output_dir / 'delta_psnr_vs_caption_length.png'}")

    # ---- summary.md --------------------------------------------------------
    md_path = args.output_dir / "summary.md"
    write_summary_md(
        md_path, args, args.baseline_method, method_names, rows,
        delta_by_method, mean_flow, baseline_psnr, caption_word_count,
        video_ids, captions, dropped_per_method, len(common_ids),
        intersection_missing_flow, intersection_missing_caption,
        delta_lpips_by_method=delta_lpips_by_method,
    )
    print(f"Wrote {md_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
