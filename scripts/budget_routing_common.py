"""Shared loaders for budget-grid routing experiments (pilot N=200, no new videos)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from scripts.analyze_adasteer_budget_oracle import (
    FIXED_ADA_RUN_ID,
    NOTTA_RUN_ID,
    PILOT_GRID_RUN_ORDER,
    build_video_table,
    discover_runs,
    load_run_psnr,
)
from scripts.analyze_adasteer_budget_vbench_oracle import (
    build_score_table,
    filter_vbench_grid_runs,
    load_vbench_by_run,
    vbench_total_score,
)
from scripts.analyze_per_video_vbench_agreement import VBENCH_DIMS
from scripts.analyze_per_video_tta_gain import load_per_video_metrics
from scripts.predictor_analysis_common import join_feature_tables

FIXED_BUDGET = FIXED_ADA_RUN_ID
PROBE_RUNS = ("S2_LR5e3", "S10_LR1e3", "S10_LR5e3", "S20_LR1e3")
BESTOF3_RUNS = ("S2_LR5e3", "S10_LR5e3", "S20_LR1e3")

# Deploy-strict router: pixels + caption stats only (no DiT forwards, no LoRA probes).
VIDEO_CORE_FEATURES: Tuple[str, ...] = (
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

# Optional cheap pixel-side stats (separate CSVs; no extra neural nets beyond CLIP/DINO above).
FAST_PIXEL_FEATURES: Tuple[str, ...] = (
    "bpp_h264",
    "bpp_png_avg",
    "hf_energy_ratio_3d",
    "hf_energy_ratio_spatial_only",
)

VAE_RECERR_FEATURES: Tuple[str, ...] = (
    "rec_err_l1",
    "rec_err_lpips",
)

# Ordered deploy router blocks (see run_deploy_strict_router_experiments.py).
DEPLOY_BLOCK_VIDEO_CORE = "video_caption"
DEPLOY_BLOCK_OOD = "diffusion_ood"
DEPLOY_BLOCK_VAE = "vae_inference"


def load_video_core_features(
    feature_date: Path,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """9-d video + caption stats from ``video_features.csv`` only."""
    from scripts.predictor_analysis_common import load_features_csv

    base = load_features_csv(feature_date / "video_features.csv")
    cols = [c for c in VIDEO_CORE_FEATURES if any(c in r for r in base.values())]
    out: Dict[str, Dict[str, float]] = {}
    for vid, row in base.items():
        out[vid] = {c: _coerce(row.get(c)) for c in cols}
    return out, cols


def load_ood_features(
    feature_date: Path,
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """~20-d diffusion-OOD block (Slide 2 / ``compute_diffusion_ood_score.py``)."""
    from scripts.correlate_tta_gain_with_features import load_ood_csv

    path = feature_date / "diffusion_ood_scores.csv"
    if not path.exists():
        raise FileNotFoundError(f"OOD CSV missing: {path}")
    rows, cols = load_ood_csv(path)
    out: Dict[str, Dict[str, float]] = {}
    for vid, row in rows.items():
        out[vid] = {c: _coerce(row.get(c)) for c in cols}
    return out, cols


def merge_feature_blocks(
    *blocks: Tuple[Dict[str, Dict[str, float]], List[str], str],
) -> Tuple[Dict[str, Dict[str, float]], List[str], Dict[str, str]]:
    """Concatenate feature blocks in order; ``blocks`` = (rows, cols, tier)."""
    merged: Dict[str, Dict[str, float]] = {}
    feature_names: List[str] = []
    tiers: Dict[str, str] = {}
    for rows, cols, tier in blocks:
        for c in cols:
            if c not in feature_names:
                feature_names.append(c)
            tiers[c] = tier
        for vid, row in rows.items():
            bucket = merged.setdefault(vid, {})
            for c in cols:
                bucket[c] = row.get(c, float("nan"))
    return merged, feature_names, tiers


def build_deploy_feature_keep(
    feature_date: Path,
    *,
    video_core: bool = False,
    ood: bool = False,
    vae_latent_profile: bool = False,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Return ordered feature list + block → column names for reporting."""
    blocks: Dict[str, List[str]] = {}
    order: List[str] = []
    if video_core:
        _, cols = load_video_core_features(feature_date)
        blocks[DEPLOY_BLOCK_VIDEO_CORE] = cols
        order.extend(cols)
    if ood:
        _, cols = load_ood_features(feature_date)
        blocks[DEPLOY_BLOCK_OOD] = cols
        order.extend(cols)
    if vae_latent_profile:
        from scripts.correlate_tta_gain_with_features import load_vae_latent_profile_csv

        path = feature_date / "vae_latent_profile_features.csv"
        _, cols = load_vae_latent_profile_csv(path)
        blocks[DEPLOY_BLOCK_VAE] = cols
        order.extend(cols)
    return order, blocks


def _coerce(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def steps_bucket(run_id: str) -> str:
    if run_id.startswith("S2_"):
        return "S2"
    if run_id.startswith("S5_"):
        return "S5"
    if run_id.startswith("S10_"):
        return "S10"
    if run_id.startswith("S20_"):
        return "S20"
    return "other"


def load_metric_matrix(
    runs: Dict[str, Path],
    grid_runs: Sequence[str],
    video_ids: Sequence[str],
    metric: str,
) -> np.ndarray:
    """metric in psnr, ssim, lpips from chunk summary.json."""
    k = len(grid_runs)
    Y = np.full((len(video_ids), k), np.nan, dtype=float)
    idx = {rid: j for j, rid in enumerate(grid_runs)}
    for rid in grid_runs:
        if rid not in runs:
            continue
        j = idx[rid]
        per_vid = load_per_video_metrics(runs[rid])
        for i, vid in enumerate(video_ids):
            row = per_vid.get(vid, {})
            v = row.get(metric)
            if v is not None:
                Y[i, j] = float(v)
    return Y


def join_pilot_feature_tables(
    feature_date: Path,
    *,
    include_video_features: bool = True,
    video_core: bool = False,
    fast_pixel: bool = False,
    vae_latent_profile: bool = False,
    vae_recerr: bool = False,
    ood: bool = False,
    tier3: bool = False,
    motion: bool = False,
    loss_var: bool = False,
    flow: bool = False,
) -> Tuple[Dict[str, Dict[str, float]], List[str], Dict[str, str]]:
    """Selectively merge pilot feature CSVs for deploy vs lab routers.

    When ``include_video_features=False``, load only explicit blocks via
    ``video_core`` / ``ood`` / ``vae_latent_profile`` (structured concat).
    """
    if not include_video_features and (video_core or ood or vae_latent_profile):
        block_specs = []
        if video_core:
            block_specs.append((*load_video_core_features(feature_date), DEPLOY_BLOCK_VIDEO_CORE))
        if ood:
            block_specs.append((*load_ood_features(feature_date), DEPLOY_BLOCK_OOD))
        if vae_latent_profile:
            from scripts.correlate_tta_gain_with_features import load_vae_latent_profile_csv

            path = feature_date / "vae_latent_profile_features.csv"
            if not path.exists():
                raise FileNotFoundError(f"vae_latent_profile CSV missing: {path}")
            rows, cols = load_vae_latent_profile_csv(path)
            vae_rows = {vid: {c: _coerce(row.get(c)) for c in cols} for vid, row in rows.items()}
            block_specs.append((vae_rows, cols, DEPLOY_BLOCK_VAE))
        if not block_specs:
            return {}, [], {}
        return merge_feature_blocks(*block_specs)

    if include_video_features:
        return join_feature_tables(
            features_csv=feature_date / "video_features.csv",
            ood_csv=(feature_date / "diffusion_ood_scores.csv") if ood else None,
            tier3_csv=(feature_date / "tier3_probe_features.csv") if tier3 else None,
            flow_csv=(feature_date / "flow_shape_features.csv") if flow else None,
            bpp_csv=(feature_date / "bpp_features.csv") if fast_pixel else None,
            fft_csv=(feature_date / "fft_features.csv") if fast_pixel else None,
            vae_recerr_csv=(feature_date / "vae_recerr_features.csv") if vae_recerr else None,
            vae_latent_profile_csv=(
                feature_date / "vae_latent_profile_features.csv"
            ) if vae_latent_profile else None,
            motion_csv=(feature_date / "latent_motion_features.csv") if motion else None,
            loss_var_csv=(feature_date / "loss_variance_features.csv") if loss_var else None,
        )

    from scripts.correlate_tta_gain_with_features import load_vae_latent_profile_csv

    merged: Dict[str, Dict[str, float]] = {}
    feature_names: List[str] = []
    tiers: Dict[str, str] = {}

    def _merge_source(src: Dict[str, Dict], cols: Iterable[str], tier: str) -> None:
        for c in cols:
            if c not in feature_names:
                feature_names.append(c)
            tiers[c] = tier
        for vid, row in src.items():
            bucket = merged.setdefault(vid, {})
            for c in cols:
                bucket[c] = _coerce(row.get(c))

    if vae_latent_profile:
        path = feature_date / "vae_latent_profile_features.csv"
        if not path.exists():
            raise FileNotFoundError(f"vae_latent_profile CSV missing: {path}")
        rows, cols = load_vae_latent_profile_csv(path)
        _merge_source(rows, cols, DEPLOY_BLOCK_VAE)

    return merged, feature_names, tiers


def filter_feature_names(
    feat_names: Sequence[str],
    keep: Sequence[str],
) -> List[int]:
    """Column indices whose names appear in *keep* (preserves *keep* order)."""
    index = {n: i for i, n in enumerate(feat_names)}
    return [index[n] for n in keep if n in index]


def subset_feature_bundle(
    features: Dict[str, Dict[str, float]],
    feat_names: Sequence[str],
    keep: Sequence[str],
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    """Drop columns not in *keep*; order follows *keep*."""
    keep_present = [n for n in keep if n in feat_names]
    out: Dict[str, Dict[str, float]] = {}
    for vid, row in features.items():
        out[vid] = {n: row.get(n, float("nan")) for n in keep_present}
    return out, keep_present


def load_pilot_bundle(
    series_root: Path,
    feature_date: Path,
    *,
    min_videos: int = 10,
    require_vbench: bool = True,
    feature_sources: Optional[dict] = None,
    feature_keep: Optional[Sequence[str]] = None,
) -> dict:
    """Load Y (total + per-dim), PSNR/SSIM, Phase-0 features for pilot grid.

    When ``require_vbench=False``, include every present pilot-grid config that
    has per-video PSNR (no VBench backfill required). VBench arrays are still
    loaded when available but may be all-NaN.
    """
    runs = discover_runs(series_root)
    if FIXED_BUDGET not in runs:
        raise FileNotFoundError(f"fixed run {FIXED_BUDGET} missing under {series_root}")

    grid_all = [r for r in runs if r.startswith("S")]
    order = {rid: i for i, rid in enumerate(PILOT_GRID_RUN_ORDER)}
    grid_all = sorted(grid_all, key=lambda r: (order.get(r, 999), r))

    _run_ids, psnr_table = build_video_table(runs)
    video_ids = sorted(psnr_table.keys())

    vb_by_run = load_vbench_by_run(runs, list(runs.keys()))
    if require_vbench:
        grid_runs, excluded = filter_vbench_grid_runs(
            vb_by_run, grid_all, min_videos=min_videos,
        )
    else:
        grid_runs = [r for r in PILOT_GRID_RUN_ORDER if r in runs]
        excluded = [r for r in grid_all if r not in grid_runs]
        if FIXED_BUDGET not in grid_runs:
            raise FileNotFoundError(
                f"fixed run {FIXED_BUDGET} missing from PSNR grid under {series_root}"
            )
    active_dims = list(VBENCH_DIMS)
    total_table, dim_tables = build_score_table(
        vb_by_run, grid_runs, video_ids, active_dims,
    )

    k = len(grid_runs)
    n = len(video_ids)
    Y_total = np.full((n, k), np.nan, dtype=float)
    Y_dim: Dict[str, np.ndarray] = {d: np.full((n, k), np.nan) for d in active_dims}
    fixed_vb = np.full(n, np.nan, dtype=float)
    run_idx = {rid: j for j, rid in enumerate(grid_runs)}

    for i, vid in enumerate(video_ids):
        row = total_table.get(vid, {})
        if FIXED_BUDGET in row:
            fixed_vb[i] = row[FIXED_BUDGET]
        for rid, j in run_idx.items():
            if rid in row:
                Y_total[i, j] = row[rid]
        for d in active_dims:
            drow = dim_tables[d].get(vid, {})
            for rid, j in run_idx.items():
                if rid in drow:
                    Y_dim[d][i, j] = drow[rid]

    psnr = load_metric_matrix(runs, grid_runs, video_ids, "psnr")
    ssim = load_metric_matrix(runs, grid_runs, video_ids, "ssim")

    if feature_sources is None:
        features, feat_names, _ = join_feature_tables(
            features_csv=feature_date / "video_features.csv",
            ood_csv=feature_date / "diffusion_ood_scores.csv",
            tier3_csv=feature_date / "tier3_probe_features.csv",
            flow_csv=feature_date / "flow_shape_features.csv",
            bpp_csv=feature_date / "bpp_features.csv",
            fft_csv=feature_date / "fft_features.csv",
            vae_recerr_csv=feature_date / "vae_recerr_features.csv",
            vae_latent_profile_csv=feature_date / "vae_latent_profile_features.csv",
            motion_csv=feature_date / "latent_motion_features.csv",
            loss_var_csv=feature_date / "loss_variance_features.csv",
        )
    else:
        features, feat_names, _ = join_pilot_feature_tables(
            feature_date, **feature_sources,
        )
    if feature_keep is not None:
        features, feat_names = subset_feature_bundle(features, feat_names, feature_keep)

    return {
        "series_root": series_root,
        "video_ids": video_ids,
        "grid_runs": grid_runs,
        "excluded_runs": excluded,
        "Y_total": Y_total,
        "Y_dim": Y_dim,
        "fixed_vb": fixed_vb,
        "psnr": psnr,
        "ssim": ssim,
        "features": features,
        "feat_names": feat_names,
        "run_idx": run_idx,
        "fixed_run": FIXED_BUDGET,
    }


def labeled_mask(fixed_vb: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return ~np.isnan(fixed_vb) & ~np.all(np.isnan(Y), axis=1)


def bootstrap_captured(
    policy_vb: np.ndarray,
    oracle_vb: np.ndarray,
    fixed_vb: np.ndarray,
    *,
    n_boot: int = 5000,
    seed: int = 42,
) -> Tuple[float, float, float, float, float]:
    """Return mean captured, lo, hi, delta_lo, delta_hi."""
    d = policy_vb - fixed_vb
    h = oracle_vb - fixed_vb
    valid = np.isfinite(d) & np.isfinite(h) & (np.abs(h) > 1e-9)
    d, h, fv, ov, pv = d[valid], h[valid], fixed_vb[valid], oracle_vb[valid], policy_vb[valid]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    cap = float(d.mean() / h.mean())
    rng = np.random.default_rng(seed)
    boot_d, boot_c = [], []
    for _ in range(n_boot):
        ix = rng.integers(0, len(d), len(d))
        dm, hm = d[ix].mean(), h[ix].mean()
        boot_d.append(dm)
        boot_c.append(dm / hm if abs(hm) > 1e-9 else float("nan"))
    boot_c = [x for x in boot_c if math.isfinite(x)]
    return (
        cap,
        float(np.percentile(boot_d, 2.5)),
        float(np.percentile(boot_d, 97.5)),
        float(np.percentile(boot_c, 2.5)) if boot_c else float("nan"),
        float(np.percentile(boot_c, 97.5)) if boot_c else float("nan"),
    )
