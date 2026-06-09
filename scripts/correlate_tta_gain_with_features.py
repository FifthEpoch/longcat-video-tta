#!/usr/bin/env python3
"""Correlate per-video TTA gain (ΔPSNR) against the feature battery emitted
by ``scripts/extract_video_features_for_tta.py`` AND (optionally) the
diffusion-OOD-score battery emitted by
``scripts/compute_diffusion_ood_score.py``.

Reads:
  * ``--gains-csv``     : the per_video_gains.csv produced by
                          ``scripts/analyze_per_video_tta_gain.py``  (one row
                          per video, ``<METHOD>_dpsnr`` per non-baseline method).
  * ``--features-csv``  : the video_features.csv produced by the sibling
                          extract_video_features_for_tta.py script (one row
                          per video, Tier-1 + Tier-3 feature columns).
  * ``--ood-csv``       : OPTIONAL. The diffusion_ood_scores.csv produced by
                          scripts/compute_diffusion_ood_score.py (one row per
                          video, per-timestep flow-matching MSE + summary
                          columns + VAE-latent statistics). When present, its
                          numeric columns are joined alongside the feature CSV
                          and analysed in the same correlation tables; OOD-
                          derived columns are flagged ``OOD`` in the tier
                          column so the combined report shows
                          ρ(ΔPSNR, X) for X in {feature-extractor columns} ∪
                          {OOD columns}.

Emits, under ``--output-dir``:
  1.  ``correlation_table.md``           markdown ρ table, |ρ| highlights
  2.  ``correlation_table.csv``          raw ρ values + N
  3.  ``top_features_per_method.md``     top-3 features per method by |ρ|
                                          (Tier-1 + OOD, online-actionable)
  4.  ``plot_<feature>.png``             per-online-actionable feature:
                                          ΔPSNR vs quintile
  5.  ``winners_losers_by_top_feature.md`` top/loser cohorts for the strongest
                                            feature
  6.  ``summary.md``                     narrative + paper-claim recommendation

The script is intentionally pure stdlib + numpy + matplotlib (no scipy /
pandas) so it runs anywhere the existing analyze_per_video_tta_gain.py runs.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Schema knowledge
# ---------------------------------------------------------------------------
TIER1_FEATURES: Tuple[str, ...] = (
    "cut_count_pyscenedetect",
    "cut_count_histogram",
    "cut_density_per_frame",
    "clip_text_image_sim_mean",
    "clip_text_image_sim_var",
    "clip_text_image_sim_min",
    "dino_temporal_l2_mean",
    "laplacian_variance_mean",
    "rgb_histogram_entropy_mean",
)
TIER3_FEATURES: Tuple[str, ...] = (
    "dino_tta_vs_genregion_sim",
    "clip_text_genregion_sim_mean",
)

# OOD columns we ALWAYS recognise when --ood-csv is provided. The
# per-timestep loss columns (`diffusion_loss_caption_t{T}` /
# `diffusion_loss_uncond_t{T}`) are auto-discovered at load time, since
# the user controls --timesteps when generating the OOD CSV.
OOD_SUMMARY_COLUMNS: Tuple[str, ...] = (
    "mean_diffusion_loss_caption",
    "mean_diffusion_loss_uncond",
    "delta_caption_minus_uncond",
    "latent_norm_mean",
    "latent_norm_std",
    "latent_kurtosis",
)

# Non-feature OOD columns: documentation / book-keeping that should NOT be
# correlated against ΔPSNR even when present in the OOD CSV.
OOD_NON_FEATURE_COLUMNS: Tuple[str, ...] = (
    "video_id", "n_visible_frames", "n_gen_target_frames", "seed",
)

# Short one-liner interpretations used by top_features_per_method.md.
FEATURE_INTERPRETATIONS: Dict[str, str] = {
    "cut_count_pyscenedetect":
        "more PySceneDetect cuts in the visible window — content discontinuity",
    "cut_count_histogram":
        "more RGB-histogram cuts in the visible window — content discontinuity",
    "cut_density_per_frame":
        "scene cuts normalised by visible length — content-density agnostic to window length",
    "clip_text_image_sim_mean":
        "higher average CLIP image↔caption alignment — prompt describes visible content well",
    "clip_text_image_sim_var":
        "higher CLIP-alignment variance — prompt fits some visible frames better than others",
    "clip_text_image_sim_min":
        "higher worst-frame CLIP alignment — every visible frame plausibly matches the prompt",
    "dino_temporal_l2_mean":
        "higher DINOv2 temporal-jump magnitude — more visible-motion / scene change",
    "laplacian_variance_mean":
        "higher Laplacian variance — sharper / more-textured visible frames",
    "rgb_histogram_entropy_mean":
        "higher RGB-histogram entropy — more colour diversity in the visible window",
    "dino_tta_vs_genregion_sim":
        "TIER 3: TTA-region DINO mean ≈ generation-region DINO mean (continuity)",
    "clip_text_genregion_sim_mean":
        "TIER 3: caption describes the GENERATION target frames well",
    # OOD-derived (compute_diffusion_ood_score.py)
    "mean_diffusion_loss_caption":
        "OOD: mean per-video flow-matching MSE (caption-conditioned) — high ⇒ "
        "base model is more 'surprised' by this video (out-of-distribution)",
    "mean_diffusion_loss_uncond":
        "OOD: mean per-video flow-matching MSE (unconditional) — caption-free "
        "OOD signal, factors out caption-conditioning quality",
    "delta_caption_minus_uncond":
        "OOD: caption-vs-uncond loss delta — large positive ⇒ caption hurts "
        "the prediction (caption disagrees with visuals)",
    "latent_norm_mean":
        "OOD (Tier-1 cheap): mean per-spatiotemporal-token VAE-latent L2 norm "
        "in the channel dim — visible latents that look unusual to the VAE",
    "latent_norm_std":
        "OOD (Tier-1 cheap): per-token VAE-latent L2 norm spread — high ⇒ "
        "latents are spatially heterogeneous",
    "latent_kurtosis":
        "OOD (Tier-1 cheap): excess kurtosis of VAE-latent values (fisher=True) "
        "— >0 ⇒ heavier-tailed than Gaussian, common for OOD content",
}


# ---------------------------------------------------------------------------
# Stats helpers (no scipy, matches analyze_per_video_tta_gain.py)
# ---------------------------------------------------------------------------
def _pearson_r(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
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


def _ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(a.size, dtype=np.float64)
    uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
    if (counts > 1).any():
        sum_ranks = np.zeros(uniq.size, dtype=np.float64)
        np.add.at(sum_ranks, inv, ranks)
        avg_ranks = sum_ranks / counts
        ranks = avg_ranks[inv]
    return ranks


def spearman_rho(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3:
        return None
    return _pearson_r(_ranks(xs[mask]), _ranks(ys[mask]))


def quantile_bin(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (bin_idx, edges) ignoring NaNs (NaN -> -1 in bin_idx)."""
    bin_idx = np.full(values.shape, -1, dtype=int)
    mask = ~np.isnan(values)
    if not mask.any():
        return bin_idx, np.array([])
    v = values[mask]
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        bin_idx[mask] = 0
        return bin_idx, edges
    idx = np.clip(np.searchsorted(edges[1:-1], v, side="right"), 0, edges.size - 2)
    bin_idx[mask] = idx
    return bin_idx, edges


# ---------------------------------------------------------------------------
# CSV loaders
# ---------------------------------------------------------------------------
def _coerce(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def load_gains_csv(path: Path) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Return ({video_id -> {col -> float}}, list_of_method_names).

    A "method name" is the prefix M of any column named ``<M>_dpsnr``.
    """
    if not path.exists():
        raise FileNotFoundError(f"--gains-csv not found: {path}")
    rows: Dict[str, Dict[str, float]] = {}
    methods: List[str] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        for fn in reader.fieldnames:
            if fn.endswith("_dpsnr"):
                methods.append(fn[: -len("_dpsnr")])
        methods = sorted(set(methods))
        for r in reader:
            vid = (r.get("video_id") or "").strip()
            if not vid:
                continue
            rows[vid] = {k: v for k, v in r.items()}
    return rows, methods


def load_features_csv(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"--features-csv not found: {path}")
    out: Dict[str, Dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for r in reader:
            vid = (r.get("video_id") or "").strip()
            if not vid:
                continue
            out[vid] = dict(r)
    return out


def load_ood_csv(path: Path) -> Tuple[Dict[str, Dict[str, str]], List[str]]:
    """Return ({video_id -> row dict}, list_of_ood_feature_columns).

    The OOD feature columns are every numeric column in the OOD CSV header
    EXCEPT the documentation columns listed in OOD_NON_FEATURE_COLUMNS
    (video_id, n_visible_frames, n_gen_target_frames, seed).  This makes
    the per-timestep loss columns auto-discoverable so the correlation
    runs even when the user re-generates the OOD CSV with a different
    --timesteps schedule.
    """
    if not path.exists():
        raise FileNotFoundError(f"--ood-csv not found: {path}")
    out: Dict[str, Dict[str, str]] = {}
    feature_cols: List[str] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header row")
        for fn in reader.fieldnames:
            if fn in OOD_NON_FEATURE_COLUMNS:
                continue
            feature_cols.append(fn)
        for r in reader:
            vid = (r.get("video_id") or "").strip()
            if not vid:
                continue
            out[vid] = dict(r)
    return out, feature_cols


# ---------------------------------------------------------------------------
# Feature source registry — tracks which CSV each feature came from + its
# tier label for the summary tables. Used by build_join() so the joined
# rows look up each feature in the correct source dict.
# ---------------------------------------------------------------------------
class FeatureSource:
    """Lightweight bundle of (rows_by_video_id, feature_list, tier_by_feature)."""
    __slots__ = ("rows", "features", "tier_by_feature", "label")

    def __init__(
        self,
        rows: Dict[str, Dict[str, str]],
        features: List[str],
        tier_by_feature: Dict[str, str],
        label: str,
    ):
        self.rows = rows
        self.features = features
        self.tier_by_feature = tier_by_feature
        self.label = label


# ---------------------------------------------------------------------------
# Join + extract numpy matrices
# ---------------------------------------------------------------------------
def build_join(
    gains: Dict[str, Dict[str, str]],
    sources: List[FeatureSource],
    methods: List[str],
) -> Tuple[
    List[str],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    int,
    List[str],
    Dict[str, str],
]:
    """Intersect on video_id across gains AND every FeatureSource.

    Returns
    -------
    (video_ids, delta_by_method, feature_by_name, n_missing_gain,
     ordered_features, tier_by_feature)
        ``ordered_features`` is the concatenation of each source's
        ``features`` list in registration order so the downstream
        markdown / CSV tables present features grouped by their source
        (feature-extractor T1, T3, then OOD).  ``tier_by_feature`` maps
        feature name -> tier label (e.g. "T1", "T3", "OOD").
    """
    common_set = set(gains.keys())
    for src in sources:
        common_set &= set(src.rows.keys())
    common = sorted(common_set)

    delta: Dict[str, np.ndarray] = {
        m: np.zeros(len(common), dtype=float) for m in methods
    }

    # Build per-source (orig_name -> disambiguated_name) lookup so we can
    # read the original column name out of the source's row dict while
    # storing it under a unique key in feat_vecs.
    ordered_features: List[str] = []
    tier_by_feature: Dict[str, str] = {}
    src_to_pairs: List[List[Tuple[str, str]]] = []
    for src in sources:
        pairs: List[Tuple[str, str]] = []
        for fname in src.features:
            disamb = (
                f"{fname}__{src.label}" if fname in tier_by_feature else fname
            )
            pairs.append((fname, disamb))
            ordered_features.append(disamb)
            tier_by_feature[disamb] = src.tier_by_feature.get(fname, "?")
        src_to_pairs.append(pairs)

    feat_vecs: Dict[str, np.ndarray] = {
        f: np.zeros(len(common), dtype=float) for f in ordered_features
    }

    n_missing_gain = 0
    for i, vid in enumerate(common):
        g = gains[vid]
        for m in methods:
            v = _coerce(g.get(f"{m}_dpsnr"))
            delta[m][i] = float("nan") if v is None else float(v)
        for src, pairs in zip(sources, src_to_pairs):
            fr = src.rows[vid]
            for orig, disamb in pairs:
                v = _coerce(fr.get(orig))
                feat_vecs[disamb][i] = float("nan") if v is None else float(v)
        if all(np.isnan(delta[m][i]) for m in methods):
            n_missing_gain += 1
    return common, delta, feat_vecs, n_missing_gain, ordered_features, tier_by_feature


# ---------------------------------------------------------------------------
# Highlight formatting
# ---------------------------------------------------------------------------
def _fmt_rho(rho: Optional[float]) -> str:
    if rho is None or np.isnan(rho):
        return "n/a"
    a = abs(rho)
    if a >= 0.3:
        return f"**_{rho:+.3f}_**"
    if a >= 0.2:
        return f"**{rho:+.3f}**"
    return f"{rho:+.3f}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _setup_mpl():
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


def plot_feature_quintile(
    plt,
    out_path: Path,
    feature_name: str,
    feature_vec: np.ndarray,
    methods: List[str],
    delta_by_method: Dict[str, np.ndarray],
    n_bins: int = 5,
    title: Optional[str] = None,
):
    bin_idx, edges = quantile_bin(feature_vec, n_bins)
    n_bins_eff = max(int(bin_idx.max()) + 1 if (bin_idx >= 0).any() else 0, 1)

    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)

    per_bin_n: List[int] = [0] * n_bins_eff
    per_bin_center: List[float] = [float("nan")] * n_bins_eff

    for i, m in enumerate(methods):
        d = delta_by_method[m]
        means = np.full(n_bins_eff, np.nan)
        sems = np.full(n_bins_eff, np.nan)
        centers = np.full(n_bins_eff, np.nan)
        for b in range(n_bins_eff):
            mask = (bin_idx == b) & ~np.isnan(d) & ~np.isnan(feature_vec)
            n = int(mask.sum())
            if n == 0:
                continue
            ys = d[mask]
            means[b] = float(ys.mean())
            sems[b] = float(ys.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            centers[b] = float(feature_vec[mask].mean())
            per_bin_n[b] = max(per_bin_n[b], n)
            per_bin_center[b] = centers[b]
        ax.errorbar(
            centers, means, yerr=1.96 * sems,
            marker="o", capsize=2, linewidth=1.6,
            label=m, color=cmap(i % 10),
        )
    annot_y = ax.get_ylim()[0]
    for b in range(n_bins_eff):
        if per_bin_n[b] > 0 and not math.isnan(per_bin_center[b]):
            ax.annotate(
                f"n={per_bin_n[b]}",
                (per_bin_center[b], annot_y),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=7, color="grey",
            )
    ax.set_xlabel(f"{feature_name} (quintile-binned, n≈{int(np.mean([n for n in per_bin_n if n>0])) if any(per_bin_n) else 0}/bin)")
    ax.set_ylabel(r"per-video $\Delta$PSNR vs NOTTA (dB; 95% CI)")
    ax.set_title(title or f"ΔPSNR vs {feature_name}")
    ax.legend(loc="best", ncols=2 if len(methods) > 4 else 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------
def _interpretation_for(fname: str) -> str:
    """Return a short sign-aware interpretation, robust to disambiguated keys.

    For columns disambiguated by the join (e.g. ``y__features`` or
    ``y__ood``), strip the trailing ``__<label>`` suffix so we still hit
    the FEATURE_INTERPRETATIONS lookup. Falls back to the raw column name.
    """
    if "__" in fname:
        base = fname.split("__", 1)[0]
    else:
        base = fname
    return FEATURE_INTERPRETATIONS.get(base, base)


def write_correlation_table(
    md_path: Path, csv_path: Path,
    methods: List[str], features: List[str],
    tier_by_feature: Dict[str, str],
    rho_table: Dict[Tuple[str, str], Tuple[Optional[float], int]],
):
    """Write both the markdown table (with highlights) and the raw CSV."""
    # ----- markdown -----
    lines: List[str] = []
    lines.append("# Spearman ρ between ΔPSNR and per-video features")
    lines.append("")
    lines.append("Cells are Spearman ρ (sample size N in parentheses).  "
                 "Bold = |ρ| ≥ 0.2 ; bold+italic = |ρ| ≥ 0.3.  Tier `OOD` "
                 "columns come from the diffusion-OOD-score CSV (per-video "
                 "flow-matching MSE against the LongCat-Video base model); "
                 "`T1` / `T3` columns come from the feature-extractor CSV.")
    lines.append("")
    header = "| feature | tier | " + " | ".join(f"`{m}`" for m in methods) + " |"
    sep = "|---|---|" + "|".join(["---:"] * len(methods)) + "|"
    lines.append(header)
    lines.append(sep)
    for fname in features:
        tier = tier_by_feature.get(fname, "?")
        cells: List[str] = []
        for m in methods:
            rho, n = rho_table.get((m, fname), (None, 0))
            if rho is None:
                cells.append(f"n/a (N={n})")
            else:
                cells.append(f"{_fmt_rho(rho)} (N={n})")
        lines.append(f"| `{fname}` | {tier} | " + " | ".join(cells) + " |")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ----- raw CSV -----
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header_cols = ["feature", "tier"]
        for m in methods:
            header_cols.extend([f"{m}_rho", f"{m}_n"])
        writer.writerow(header_cols)
        for fname in features:
            tier = tier_by_feature.get(fname, "?")
            row = [fname, tier]
            for m in methods:
                rho, n = rho_table.get((m, fname), (None, 0))
                row.append("" if rho is None else f"{rho:.6f}")
                row.append(str(n))
            writer.writerow(row)


def write_top_features_per_method(
    md_path: Path, methods: List[str],
    features: List[str],
    tier_by_feature: Dict[str, str],
    rho_table: Dict[Tuple[str, str], Tuple[Optional[float], int]],
    top_k: int = 3,
    online_tiers: Tuple[str, ...] = ("T1", "OOD"),
):
    lines: List[str] = []
    lines.append("# Top features per method (by |Spearman ρ|)")
    lines.append("")
    lines.append(
        "Restricted to ONLINE-ACTIONABLE features: Tier-1 visible-window "
        "features (`T1`) and diffusion-OOD-score columns (`OOD`).  Both "
        "are computable at deployment time from the TTA-visible frames "
        "alone, so a per-video selection rule based on either is "
        "implementable without privileged access to the generation target."
    )
    lines.append("")
    online_features = [
        f for f in features if tier_by_feature.get(f) in online_tiers
    ]
    for m in methods:
        scored: List[Tuple[str, float, int]] = []
        for fname in online_features:
            rho, n = rho_table.get((m, fname), (None, 0))
            if rho is None:
                continue
            scored.append((fname, rho, n))
        scored.sort(key=lambda t: abs(t[1]), reverse=True)
        lines.append(f"## `{m}`")
        lines.append("")
        if not scored:
            lines.append("_No finite Spearman ρ available._")
            lines.append("")
            continue
        lines.append("| rank | feature | tier | ρ | N | interpretation (sign-aware) |")
        lines.append("|---:|---|---|---:|---:|---|")
        for i, (fname, rho, n) in enumerate(scored[:top_k], 1):
            base = _interpretation_for(fname)
            tier = tier_by_feature.get(fname, "?")
            direction = "↑ feature ⇒ ↑ ΔPSNR" if rho > 0 else "↑ feature ⇒ ↓ ΔPSNR"
            lines.append(
                f"| {i} | `{fname}` | {tier} | {_fmt_rho(rho)} | {n} | "
                f"{base}; sign: {direction} |"
            )
        lines.append("")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_winners_losers(
    md_path: Path,
    methods: List[str],
    video_ids: List[str],
    delta_by_method: Dict[str, np.ndarray],
    feature_name: str,
    feature_vec: np.ndarray,
    top_k: int,
):
    lines: List[str] = []
    lines.append(f"# Winners & losers cohort sorted by `{feature_name}`")
    lines.append("")
    lines.append(
        f"Top {top_k} highest-feature videos vs top {top_k} lowest-feature videos.  "
        "Median ΔPSNR per method per cohort.  Feature chosen as the one with "
        "the highest mean |ρ| across the analysed methods."
    )
    lines.append("")
    n = feature_vec.size
    if n == 0:
        lines.append("_Empty join; nothing to report._")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    finite = np.where(~np.isnan(feature_vec))[0]
    if finite.size == 0:
        lines.append(f"_All `{feature_name}` rows are NaN; nothing to report._")
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    order_desc = finite[np.argsort(-feature_vec[finite], kind="mergesort")]
    order_asc = finite[np.argsort(feature_vec[finite], kind="mergesort")]
    high_idx = order_desc[:top_k]
    low_idx = order_asc[:top_k]
    lines.append(f"## Cohort medians ΔPSNR (high-`{feature_name}` minus low-`{feature_name}`)")
    lines.append("")
    lines.append("| method | high cohort median Δ | low cohort median Δ | high − low |")
    lines.append("|---|---:|---:|---:|")
    for m in methods:
        d = delta_by_method[m]
        hi = d[high_idx]; hi = hi[~np.isnan(hi)]
        lo = d[low_idx]; lo = lo[~np.isnan(lo)]
        hi_med = float(np.median(hi)) if hi.size else float("nan")
        lo_med = float(np.median(lo)) if lo.size else float("nan")
        diff = (hi_med - lo_med) if not (math.isnan(hi_med) or math.isnan(lo_med)) else float("nan")
        def _fmt(x):
            return "n/a" if math.isnan(x) else f"{x:+.3f}"
        lines.append(f"| `{m}` | {_fmt(hi_med)} | {_fmt(lo_med)} | {_fmt(diff)} |")
    lines.append("")

    lines.append(f"## Top {top_k} highest-`{feature_name}` videos")
    lines.append("")
    cols = ["#", "video_id", f"`{feature_name}`"] + [f"`{m}` Δ" for m in methods]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|---:|---|---:|" + "---:|" * len(methods))
    for i, idx in enumerate(high_idx, 1):
        cells = [
            str(i),
            f"`{video_ids[idx]}`",
            f"{feature_vec[idx]:+.4f}",
        ]
        for m in methods:
            v = delta_by_method[m][idx]
            cells.append("n/a" if math.isnan(v) else f"{v:+.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append(f"## Top {top_k} lowest-`{feature_name}` videos")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|---:|---|---:|" + "---:|" * len(methods))
    for i, idx in enumerate(low_idx, 1):
        cells = [
            str(i),
            f"`{video_ids[idx]}`",
            f"{feature_vec[idx]:+.4f}",
        ]
        for m in methods:
            v = delta_by_method[m][idx]
            cells.append("n/a" if math.isnan(v) else f"{v:+.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    md_path: Path,
    args: argparse.Namespace,
    methods: List[str],
    features: List[str],
    tier_by_feature: Dict[str, str],
    rho_table: Dict[Tuple[str, str], Tuple[Optional[float], int]],
    n_videos: int,
    strongest_feature: Optional[str],
    strongest_mean_abs_rho: float,
    online_tiers: Tuple[str, ...] = ("T1", "OOD"),
):
    lines: List[str] = []
    lines.append("# Per-video TTA-gain ↔ feature-battery correlation summary")
    lines.append("")
    lines.append(f"- Gains CSV: `{args.gains_csv}`")
    lines.append(f"- Features CSV: `{args.features_csv}`")
    if getattr(args, "ood_csv", None):
        lines.append(f"- OOD CSV: `{args.ood_csv}`")
    lines.append(f"- Methods analysed (non-baseline): {', '.join('`' + m + '`' for m in methods)}")
    lines.append(f"- Joined videos (intersection of gains ∩ all feature sources): **{n_videos}**")
    lines.append("")

    # Per-feature mean |ρ| ranking, broken out by tier so OOD-derived columns
    # are visually distinguishable from feature-extractor columns.
    online_features = [
        f for f in features if tier_by_feature.get(f) in online_tiers
    ]
    lines.append(
        "## Feature ranking (mean |ρ| across analysed methods, online-actionable only)"
    )
    lines.append("")
    lines.append(
        "Online-actionable = Tier-1 visible-window features (`T1`) + "
        "diffusion-OOD-score columns (`OOD`).  Tier-3 columns "
        "(diagnostic-only, use GT frames) are excluded from the ranking "
        "but still appear in `correlation_table.{md,csv}`."
    )
    lines.append("")
    ranking: List[Tuple[str, str, float, int]] = []
    for fname in online_features:
        vals: List[float] = []
        cleared = 0
        for m in methods:
            rho, _n = rho_table.get((m, fname), (None, 0))
            if rho is None or np.isnan(rho):
                continue
            vals.append(abs(rho))
            if abs(rho) >= 0.2:
                cleared += 1
        if not vals:
            continue
        ranking.append(
            (fname, tier_by_feature.get(fname, "?"), float(np.mean(vals)), cleared)
        )
    ranking.sort(key=lambda t: t[2], reverse=True)
    lines.append("| rank | feature | tier | mean \\|ρ\\| | # methods with \\|ρ\\|≥0.2 |")
    lines.append("|---:|---|---|---:|---:|")
    for i, (fname, tier, mu, c) in enumerate(ranking, 1):
        lines.append(
            f"| {i} | `{fname}` | {tier} | {mu:.3f} | {c} / {len(methods)} |"
        )
    lines.append("")

    # Bar bar: how many features clear |ρ| >= 0.2 for >= 2 methods?
    cleared_for_two = [(f, t) for (f, t, _mu, c) in ranking if c >= 2]
    lines.append("## Headline: features that cleared |ρ| ≥ 0.2 for ≥ 2 methods")
    lines.append("")
    if cleared_for_two:
        for f, t in cleared_for_two:
            lines.append(f"- `{f}`  (`{t}`)")
    else:
        lines.append(
            "_No online-actionable feature cleared the |ρ| ≥ 0.2 bar "
            "for ≥ 2 methods._"
        )
    lines.append("")

    # OOD-specific headline (the original hypothesis under test).
    lines.append("## OOD-hypothesis headline")
    lines.append("")
    ood_keys = [
        f for f in features if tier_by_feature.get(f) == "OOD"
    ]
    if not ood_keys:
        lines.append(
            "_No OOD CSV joined; pass `--ood-csv` to test the OOD-score "
            "hypothesis._"
        )
    else:
        ood_uncond_keys = [
            f for f in ood_keys if f.startswith("mean_diffusion_loss_uncond")
        ]
        focal = ood_uncond_keys[0] if ood_uncond_keys else None
        if focal is None:
            lines.append(
                "_OOD CSV is missing the `mean_diffusion_loss_uncond` summary "
                "column; cannot evaluate the headline hypothesis._"
            )
        else:
            lines.append(
                f"Hypothesis under test: ρ(ΔPSNR, `{focal}`) > 0 — videos the "
                "base diffusion model finds unfamiliar should TTA better."
            )
            lines.append("")
            lines.append("| method | ρ | N | verdict |")
            lines.append("|---|---:|---:|---|")
            for m in methods:
                rho, n = rho_table.get((m, focal), (None, 0))
                if rho is None or (isinstance(rho, float) and math.isnan(rho)):
                    verdict = "n/a"
                elif rho > 0.2:
                    verdict = "supports hypothesis (|ρ| ≥ 0.2 positive)"
                elif rho < -0.2:
                    verdict = "refutes hypothesis (|ρ| ≥ 0.2 negative)"
                else:
                    verdict = "no signal at |ρ| ≥ 0.2"
                lines.append(
                    f"| `{m}` | {_fmt_rho(rho)} | {n} | {verdict} |"
                )
    lines.append("")

    # Recommended paper claim.
    lines.append("## Recommended paper claim")
    lines.append("")
    if cleared_for_two:
        f0, t0 = cleared_for_two[0]
        lines.append(
            f"The TTA-gain distribution is non-random: it correlates with `{f0}` "
            f"(`{t0}`; and {len(cleared_for_two) - 1} additional online-actionable "
            "feature(s)) at |ρ| ≥ 0.2 across multiple TTA methods, suggesting a "
            "deployment-time selection rule based on this signal would beat "
            "applying TTA uniformly. See `top_features_per_method.md` for the "
            "sign and which methods are most affected."
        )
    elif ranking and ranking[0][2] >= 0.15:
        f0, t0, mu, _c = ranking[0]
        lines.append(
            f"No single online-actionable feature cleared |ρ| ≥ 0.2 for ≥ 2 "
            f"methods.  The strongest candidate is `{f0}` (`{t0}`) at mean "
            f"|ρ| = {mu:.3f}; the per-video TTA-gain signal is mostly noise "
            "at this feature-battery resolution.  Reproduce with a richer "
            "battery before claiming a per-video selection rule."
        )
    else:
        lines.append(
            "No Tier-1 feature in this battery predicts per-video ΔPSNR at the "
            "|ρ| ≥ 0.2 / 2-methods threshold.  The honest paper claim is still "
            "_population-mean TTA gain ≈ 0 with no per-video predictor in this "
            "feature battery_.  Next iteration should extend the battery before "
            "the per-video story is salvageable."
        )
    lines.append("")

    # Honest read.
    lines.append("## Cross-feature honest read")
    lines.append("")
    lines.append(
        "* `cut_count_*` and `cut_density_per_frame` are scene-cut signals: the "
        "hypothesis was that high-cut videos break TTA's continuity assumption.  "
        "Look at the Tier-1 table to confirm.\n"
        "* `clip_text_image_sim_*` test whether the caption actually describes the "
        "visible window: poor alignment means the TTA loss term that uses the "
        "caption is mis-supervised, predicting TTA hurts.\n"
        "* `dino_temporal_l2_mean` captures visible-region motion magnitude using "
        "DINOv2 instead of RAFT.  Compare to the existing `mean_flow` correlation "
        "in `summary.md` (per-video analysis): DINOv2 features may pick up "
        "structural change RAFT misses on slow zooms.\n"
        "* `laplacian_variance_mean` / `rgb_histogram_entropy_mean` are visible-"
        "region appearance complexity proxies.  If sharp / colourful clips "
        "correlate with positive ΔPSNR, that suggests TTA over-smooths and "
        "thus only helps clips that are already smooth.\n"
        "* `dino_tta_vs_genregion_sim` (Tier 3) tests whether the TTA window and "
        "generation target look the same; high values would mean the TTA loss "
        "is a good proxy for the generation loss.  This is NOT online-actionable "
        "because the gen-region frames are GT.\n"
    )
    lines.append("")

    # Next-iteration features (mention explicitly per user's "be honest" ask).
    lines.append("## What we would test if nothing clears |ρ| ≥ 0.2")
    lines.append("")
    lines.append(
        "Adding these features to the next iteration is the explicit fallback if "
        "no Tier-1 column in THIS battery clears the bar:\n\n"
        "* Caption language-model perplexity (a confused / template-y caption is a "
        "  diagnostic of poor supervision signal).\n"
        "* Action-vs-object caption classification (verb-density per word).\n"
        "* Optical flow SECOND moments (variance / max) in addition to the "
        "  current mean flow.\n"
        "* CLIP-vs-DINO disagreement (semantic vs structural feature mismatch on "
        "  the visible window).\n"
        "* Per-frame VAE reconstruction error of the BASE model on the visible "
        "  window (proxy for 'is this clip in-distribution').\n"
        "* SLIDING-window CLIP alignment within the visible frames (captures "
        "  prompt drift within a single TTA window).\n"
        "\n"
        "Document that THIS battery did not find a signal and ship the next "
        "battery; do NOT silently widen the threshold post-hoc."
    )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gains-csv", type=Path, required=True,
                    help="per_video_gains.csv from analyze_per_video_tta_gain.py")
    ap.add_argument("--features-csv", type=Path, required=True,
                    help="video_features.csv from extract_video_features_for_tta.py")
    ap.add_argument("--ood-csv", type=Path, default=None,
                    help="OPTIONAL diffusion_ood_scores.csv from "
                         "compute_diffusion_ood_score.py. When present, its "
                         "feature columns (per-timestep + summary diffusion-"
                         "loss columns plus the cheap VAE-latent statistics) "
                         "are joined alongside the feature CSV and analysed "
                         "in the same correlation tables under the `OOD` tier.")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--top-k", type=int, default=10,
                    help="Top-K winners / losers cohort size for the strongest feature.")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="Restrict to these method names (must match the "
                         "<METHOD>_dpsnr column prefix in --gains-csv). "
                         "Default: all methods present.")
    ap.add_argument("--n-bins", type=int, default=5,
                    help="Number of quantile bins for the per-feature plot.")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ΔPSNR ↔ video-feature correlation")
    print("=" * 70)
    print(f"Gains CSV    : {args.gains_csv}")
    print(f"Features CSV : {args.features_csv}")
    print(f"OOD CSV      : {args.ood_csv if args.ood_csv else '(not provided)'}")
    print(f"Output dir   : {args.output_dir}")
    print("=" * 70)

    gains_rows, gains_methods = load_gains_csv(args.gains_csv)
    feats_rows = load_features_csv(args.features_csv)
    print(f"Gains rows   : {len(gains_rows)}  (methods in CSV: {gains_methods})")
    print(f"Feature rows : {len(feats_rows)}")

    # ---- Build feature sources -------------------------------------------
    feat_tier_by_name: Dict[str, str] = {}
    for f in TIER1_FEATURES:
        feat_tier_by_name[f] = "T1"
    for f in TIER3_FEATURES:
        feat_tier_by_name[f] = "T3"
    feat_features = list(TIER1_FEATURES) + list(TIER3_FEATURES)
    feat_source = FeatureSource(
        rows=feats_rows,
        features=feat_features,
        tier_by_feature=feat_tier_by_name,
        label="features",
    )

    sources: List[FeatureSource] = [feat_source]
    if args.ood_csv is not None:
        ood_rows, ood_feature_cols = load_ood_csv(args.ood_csv)
        print(f"OOD rows     : {len(ood_rows)}  "
              f"(feature columns: {ood_feature_cols})")
        ood_tier = {f: "OOD" for f in ood_feature_cols}
        sources.append(FeatureSource(
            rows=ood_rows,
            features=ood_feature_cols,
            tier_by_feature=ood_tier,
            label="ood",
        ))

    if args.methods:
        methods = [m for m in args.methods if m in gains_methods]
        missing = sorted(set(args.methods) - set(gains_methods))
        if missing:
            print(f"[warn] requested methods not in gains CSV: {missing}",
                  file=sys.stderr)
    else:
        methods = list(gains_methods)
    if not methods:
        print("[error] no methods to analyse; aborting.", file=sys.stderr)
        return 2
    print(f"Methods used : {methods}")

    (
        video_ids, delta_by_method, feature_by_name, n_missing_gain,
        ordered_features, tier_by_feature,
    ) = build_join(gains_rows, sources, methods)
    print(f"Joined videos: {len(video_ids)}  "
          f"(rows with all-NaN ΔPSNR across selected methods: {n_missing_gain})")

    # ---- correlation table -----------------------------------------------
    rho_table: Dict[Tuple[str, str], Tuple[Optional[float], int]] = {}
    for fname in ordered_features:
        fv = feature_by_name[fname]
        for m in methods:
            d = delta_by_method[m]
            rho = spearman_rho(fv, d)
            mask = ~(np.isnan(fv) | np.isnan(d))
            n = int(mask.sum())
            rho_table[(m, fname)] = (rho, n)

    write_correlation_table(
        args.output_dir / "correlation_table.md",
        args.output_dir / "correlation_table.csv",
        methods, ordered_features, tier_by_feature, rho_table,
    )
    write_top_features_per_method(
        args.output_dir / "top_features_per_method.md",
        methods, ordered_features, tier_by_feature, rho_table, top_k=3,
    )
    print(f"Wrote correlation_table.{{md,csv}} and top_features_per_method.md")

    # ---- per-feature plots (online-actionable: T1 + OOD) -----------------
    plt = _setup_mpl()
    online_features = [
        f for f in ordered_features if tier_by_feature.get(f) in ("T1", "OOD")
    ]
    for fname in online_features:
        fv = feature_by_name[fname]
        out = args.output_dir / f"plot_{fname}.png"
        plot_feature_quintile(
            plt, out, fname, fv, methods, delta_by_method,
            n_bins=args.n_bins,
            title=f"ΔPSNR vs {fname} (quintile, 95% CI per method)",
        )
    print(f"Wrote {len(online_features)} per-online-actionable-feature plots "
          "(plot_<feature>.png; tiers T1 + OOD)")

    # ---- strongest online-actionable feature + winners/losers ------------
    feature_mean_abs_rho: List[Tuple[str, float]] = []
    for fname in online_features:
        rhos = [
            abs(rho_table[(m, fname)][0])
            for m in methods
            if rho_table[(m, fname)][0] is not None
            and not np.isnan(rho_table[(m, fname)][0])
        ]
        if rhos:
            feature_mean_abs_rho.append((fname, float(np.mean(rhos))))
    feature_mean_abs_rho.sort(key=lambda t: t[1], reverse=True)
    strongest_name: Optional[str] = (
        feature_mean_abs_rho[0][0] if feature_mean_abs_rho else None
    )
    strongest_mu: float = (
        feature_mean_abs_rho[0][1] if feature_mean_abs_rho else 0.0
    )
    if strongest_name is None:
        print("[warn] no online-actionable feature had finite ρ; "
              "skipping winners/losers")
    else:
        print(
            f"Strongest online-actionable feature : `{strongest_name}` "
            f"(`{tier_by_feature.get(strongest_name, '?')}`, "
            f"mean |ρ|={strongest_mu:.3f})"
        )
        write_winners_losers(
            args.output_dir / "winners_losers_by_top_feature.md",
            methods, video_ids, delta_by_method,
            strongest_name, feature_by_name[strongest_name], top_k=args.top_k,
        )

    # ---- summary.md ------------------------------------------------------
    write_summary(
        args.output_dir / "summary.md",
        args, methods, ordered_features, tier_by_feature, rho_table,
        len(video_ids), strongest_name, strongest_mu,
    )
    print(f"Wrote summary.md")
    print(f"\nAll outputs under {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
