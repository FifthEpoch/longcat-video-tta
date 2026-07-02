#!/usr/bin/env python3
"""Train offline routers to predict per-video VBench++ headroom (Phase-0 features → ΔVBench).

Targets (supervised from completed sweeps — no feature leakage):
  * ``method_gain``     — ΔVBench total for AdaSteer vs NOTTA (apply/skip router)
  * ``budget_headroom`` — max grid VBench total − fixed S10/LR5e-3 (budget oracle uplift)
  * ``budget_config``   — pick step×LR config maximizing predicted VBench total (TTA router)

Models: ridge regression + optional win/loss logistic (numpy only, no sklearn).
Evaluation: held-out Spearman ρ, deployable policy mean VBench, fraction of oracle headroom.

Example (999v method router):
    python3 scripts/train_vbench_headroom_router.py \\
        --gains-csv sweep_experiment/reports/per_video_analysis/2026-07-02/vbench_agreement/per_video_vbench_gains.csv \\
        --features-csv sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --task method_gain \\
        --output-dir sweep_experiment/reports/per_video_analysis/2026-07-02/vbench_headroom_router

Budget headroom (pilot grid with per-config VBench):
    python3 scripts/train_vbench_headroom_router.py \\
        --task budget_headroom \\
        --budget-series-root sweep_experiment/results/panda_ood_budget_pilot \\
        --features-csv ... --ood-csv ... \\
        --output-dir ...
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_adasteer_budget_vbench_oracle import (  # noqa: E402
    FIXED_ADA_RUN_ID,
    NOTTA_RUN_ID,
    PILOT_GRID_RUN_ORDER,
    build_score_table,
    discover_runs,
    filter_vbench_grid_runs,
    load_vbench_by_run,
    vbench_total_score,
)
from scripts.analyze_per_video_vbench_agreement import VBENCH_DIMS  # noqa: E402
from scripts.analyze_router_auc import binary_auc  # noqa: E402
from scripts.correlate_tta_gain_with_features import spearman_rho  # noqa: E402
from scripts.predictor_analysis_common import (  # noqa: E402
    enrich_vbench_totals,
    intersect_videos,
    join_feature_tables,
    load_vbench_gains,
)
from scripts.per_video_metric_store import load_gains_csv, vbench_total  # noqa: E402

METHOD_DEFAULT = "ADA"
FIXED_BUDGET = FIXED_ADA_RUN_ID


def _coerce(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def abs_vbench_total_row(row: Dict[str, float], method: str) -> float:
    vals = [_coerce(row.get(f"{method}_{d}")) for d in VBENCH_DIMS]
    if any(math.isnan(v) for v in vals):
        return float("nan")
    return float(np.mean(vals))


def method_gain_labels(
    gains: Dict[str, Dict[str, float]],
    video_ids: Sequence[str],
    *,
    method: str = METHOD_DEFAULT,
) -> np.ndarray:
    col = f"{method}_dvbench_total"
    out = np.array(
        [_coerce(gains.get(vid, {}).get(col)) for vid in video_ids], dtype=float
    )
    missing = np.isnan(out)
    if missing.any():
        for i, vid in enumerate(video_ids):
            if not missing[i]:
                continue
            row = gains.get(vid, {})
            m = abs_vbench_total_row(row, method)
            n = abs_vbench_total_row(row, "NOTTA")
            if not (math.isnan(m) or math.isnan(n)):
                out[i] = m - n
    return out


def load_budget_headroom_labels(
    series_root: Path,
    video_ids: Sequence[str],
    *,
    fixed_run: str = FIXED_BUDGET,
    min_videos: int = 10,
) -> Tuple[np.ndarray, np.ndarray, List[str], int]:
    """Return (headroom, fixed_vbench, grid_runs_used, n_labeled)."""
    runs = discover_runs(series_root)
    if fixed_run not in runs:
        raise FileNotFoundError(f"fixed run {fixed_run} not under {series_root}")
    grid_all = [r for r in runs if r.startswith("S")]
    order = {rid: i for i, rid in enumerate(PILOT_GRID_RUN_ORDER)}
    grid_all = sorted(grid_all, key=lambda r: (order.get(r, 999), r))
    vb_by_run = load_vbench_by_run(runs, list(runs.keys()))
    grid_runs, _excluded = filter_vbench_grid_runs(
        vb_by_run, grid_all, min_videos=min_videos,
    )
    active_dims = list(VBENCH_DIMS)
    total_table, _ = build_score_table(vb_by_run, grid_runs, list(video_ids), active_dims)

    headroom = np.full(len(video_ids), np.nan, dtype=float)
    fixed_vb = np.full(len(video_ids), np.nan, dtype=float)
    for i, vid in enumerate(video_ids):
        row = total_table.get(vid, {})
        if fixed_run not in row:
            continue
        fixed_vb[i] = row[fixed_run]
        if not row:
            continue
        best = max(row.get(r, float("-inf")) for r in grid_runs if r in row)
        if best > float("-inf"):
            headroom[i] = best - fixed_vb[i]
    n = int(np.sum(~np.isnan(headroom)))
    return headroom, fixed_vb, grid_runs, n


def load_budget_score_matrix(
    series_root: Path,
    video_ids: Sequence[str],
    *,
    fixed_run: str = FIXED_BUDGET,
    min_videos: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], int]:
    """Return (Y scores n×k, fixed_vb, notta_vb, grid_runs, n_labeled)."""
    runs = discover_runs(series_root)
    if fixed_run not in runs:
        raise FileNotFoundError(f"fixed run {fixed_run} not under {series_root}")
    grid_all = [r for r in runs if r.startswith("S")]
    order = {rid: i for i, rid in enumerate(PILOT_GRID_RUN_ORDER)}
    grid_all = sorted(grid_all, key=lambda r: (order.get(r, 999), r))
    vb_by_run = load_vbench_by_run(runs, list(runs.keys()))
    grid_runs, _excluded = filter_vbench_grid_runs(
        vb_by_run, grid_all, min_videos=min_videos,
    )
    active_dims = list(VBENCH_DIMS)
    total_table, _ = build_score_table(vb_by_run, grid_runs, list(video_ids), active_dims)

    notta_vb = np.full(len(video_ids), np.nan, dtype=float)
    notta_map = vb_by_run.get(NOTTA_RUN_ID, {})
    for i, vid in enumerate(video_ids):
        tot = vbench_total_score(notta_map.get(vid, {}), active_dims)
        if tot is not None:
            notta_vb[i] = tot

    k = len(grid_runs)
    Y = np.full((len(video_ids), k), np.nan, dtype=float)
    fixed_vb = np.full(len(video_ids), np.nan, dtype=float)
    run_idx = {rid: j for j, rid in enumerate(grid_runs)}
    for i, vid in enumerate(video_ids):
        row = total_table.get(vid, {})
        if fixed_run in row:
            fixed_vb[i] = row[fixed_run]
        for rid, j in run_idx.items():
            if rid in row:
                Y[i, j] = row[rid]
    labeled = ~np.isnan(fixed_vb) & ~np.all(np.isnan(Y), axis=1)
    n = int(np.sum(labeled))
    return Y, fixed_vb, notta_vb, grid_runs, n


def pct_gain(delta: float, baseline: float) -> Optional[float]:
    if baseline is None or math.isnan(baseline) or abs(baseline) < 1e-9:
        return None
    return 100.0 * delta / baseline


def build_feature_matrix(
    video_ids: Sequence[str],
    features: Dict[str, Dict[str, float]],
    feature_names: Sequence[str],
    *,
    impute: Dict[str, float],
) -> np.ndarray:
    X = np.zeros((len(video_ids), len(feature_names)), dtype=float)
    for i, vid in enumerate(video_ids):
        row = features.get(vid, {})
        for j, name in enumerate(feature_names):
            v = _coerce(row.get(name))
            if math.isnan(v):
                v = impute.get(name, 0.0)
            X[i, j] = v
    return X


def compute_impute(
    video_ids: Sequence[str],
    features: Dict[str, Dict[str, float]],
    feature_names: Sequence[str],
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name in feature_names:
        vals = [
            _coerce(features.get(vid, {}).get(name))
            for vid in video_ids
            if not math.isnan(_coerce(features.get(vid, {}).get(name)))
        ]
        out[name] = float(np.median(vals)) if vals else 0.0
    return out


def standardize_train_test(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X_train, axis=0)
    sd = np.nanstd(X_train, axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    p = Xb.shape[1]
    return np.linalg.solve(Xb.T @ Xb + lam * np.eye(p), Xb.T @ y)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    return Xb @ w


def logistic_fit(
    X: np.ndarray, y: np.ndarray, *, lam: float = 1.0, steps: int = 400, lr: float = 0.1
) -> np.ndarray:
    """Binary logistic with L2; y in {0,1}. Returns weights [bias, ...]."""
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1], dtype=float)
    for _ in range(steps):
        z = np.clip(Xb @ w, -20, 20)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = Xb.T @ (p - y) / len(y) + lam * np.r_[0.0, w[1:]]
        w -= lr * grad
    return w


def logistic_predict_proba(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    z = np.clip(Xb @ w, -20, 20)
    return 1.0 / (1.0 + np.exp(-z))


def kfold_indices(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return [np.asarray(f, dtype=int) for f in folds]


def select_ridge_lambda(
    X: np.ndarray,
    y: np.ndarray,
    folds: List[np.ndarray],
    lambdas: Sequence[float],
) -> float:
    best_lam, best_mse = lambdas[0], float("inf")
    for lam in lambdas:
        mses: List[float] = []
        for i, test_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)
            w = ridge_fit(X_tr_s, y_tr, lam)
            pred = ridge_predict(X_te_s, w)
            mses.append(float(np.mean((pred - y_te) ** 2)))
        mse = float(np.mean(mses))
        if mse < best_mse:
            best_mse, best_lam = mse, lam
    return best_lam


def eval_method_policy(
    pred_gain: np.ndarray,
    y_gain: np.ndarray,
    *,
    notta_abs: Optional[np.ndarray] = None,
    tau: float = 0.0,
) -> dict:
    """Apply ADA if pred > tau else NOTTA (0 gain). Oracle = per-video max(0, gain)."""
    apply = pred_gain > tau
    realized = np.where(apply, y_gain, 0.0)
    oracle = np.maximum(y_gain, 0.0)
    fixed_mean = float(np.nanmean(y_gain))
    policy_mean = float(np.nanmean(realized))
    oracle_mean = float(np.nanmean(oracle))
    headroom = oracle_mean - fixed_mean
    captured = (policy_mean - fixed_mean) / headroom if abs(headroom) > 1e-9 else float("nan")
    notta_mean = float(np.nanmean(notta_abs)) if notta_abs is not None else None
    return {
        "tau": tau,
        "apply_rate": float(np.mean(apply)),
        "mean_realized_gain": policy_mean,
        "mean_fixed_gain": fixed_mean,
        "mean_oracle_gain": oracle_mean,
        "oracle_headroom": headroom,
        "fraction_oracle_captured": captured,
        "pct_vs_notta": pct_gain(policy_mean, notta_mean) if notta_mean else None,
        "pct_oracle_vs_notta": pct_gain(oracle_mean, notta_mean) if notta_mean else None,
    }


def eval_budget_policy(
    pred_headroom: np.ndarray,
    y_headroom: np.ndarray,
    fixed_vb: np.ndarray,
    *,
    notta_vb: Optional[np.ndarray] = None,
    tau: float = 0.0,
) -> dict:
    """If pred > tau use oracle headroom else 0 uplift (stay on fixed)."""
    apply = pred_headroom > tau
    realized_h = np.where(apply, y_headroom, 0.0)
    oracle_h = np.maximum(y_headroom, 0.0)
    fixed_mean = float(np.nanmean(fixed_vb))
    policy_mean = float(np.nanmean(fixed_vb + realized_h))
    oracle_mean = float(np.nanmean(fixed_vb + oracle_h))
    headroom = oracle_mean - fixed_mean
    captured = (policy_mean - fixed_mean) / headroom if headroom > 1e-9 else float("nan")
    notta_mean = float(np.nanmean(notta_vb)) if notta_vb is not None else None
    return {
        "tau": tau,
        "apply_rate": float(np.mean(apply)),
        "mean_policy_vbench": policy_mean,
        "mean_fixed_vbench": fixed_mean,
        "mean_oracle_vbench": oracle_mean,
        "oracle_headroom": headroom,
        "fraction_oracle_captured": captured,
        "pct_vs_notta": pct_gain(policy_mean - (notta_mean or 0.0), notta_mean)
        if notta_mean
        else None,
        "pct_vs_fixed": pct_gain(policy_mean - fixed_mean, fixed_mean),
    }


def eval_config_pick_policy(
    picked_idx: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid_runs: Sequence[str],
    *,
    notta_vb: Optional[np.ndarray] = None,
) -> dict:
    """Deployable TTA config router: realized VBench from predicted best config."""
    n, k = Y.shape
    realized = np.full(n, np.nan, dtype=float)
    oracle = np.full(n, np.nan, dtype=float)
    for i in range(n):
        row = Y[i]
        if np.all(np.isnan(row)):
            continue
        j = int(picked_idx[i])
        if 0 <= j < k and not math.isnan(row[j]):
            realized[i] = row[j]
        oracle[i] = float(np.nanmax(row))
    fixed_mean = float(np.nanmean(fixed_vb))
    policy_mean = float(np.nanmean(realized))
    oracle_mean = float(np.nanmean(oracle))
    headroom = oracle_mean - fixed_mean
    captured = (policy_mean - fixed_mean) / headroom if headroom > 1e-9 else float("nan")
    notta_mean = float(np.nanmean(notta_vb)) if notta_vb is not None else None
    pick_counts: Dict[str, int] = {}
    for j in picked_idx.astype(int):
        if 0 <= j < len(grid_runs):
            pick_counts[grid_runs[j]] = pick_counts.get(grid_runs[j], 0) + 1
    return {
        "mean_policy_vbench": policy_mean,
        "mean_fixed_vbench": fixed_mean,
        "mean_oracle_vbench": oracle_mean,
        "oracle_headroom": headroom,
        "fraction_oracle_captured": captured,
        "pct_vs_notta": pct_gain(policy_mean - (notta_mean or 0.0), notta_mean)
        if notta_mean
        else None,
        "pct_vs_fixed": pct_gain(policy_mean - fixed_mean, fixed_mean),
        "top_picks": dict(sorted(pick_counts.items(), key=lambda kv: -kv[1])[:5]),
    }


def run_task(
    *,
    task: str,
    video_ids: List[str],
    X: np.ndarray,
    y: np.ndarray,
    aux: Optional[np.ndarray],
    output_dir: Path,
    seed: int,
    n_folds: int,
    notta_aux: Optional[np.ndarray] = None,
) -> dict:
    mask = ~np.isnan(y)
    if mask.sum() < 30:
        raise ValueError(f"{task}: only {mask.sum()} labeled videos (need ≥30)")

    vid = [video_ids[i] for i in range(len(video_ids)) if mask[i]]
    X = X[mask]
    y = y[mask]
    aux = aux[mask] if aux is not None else None

    folds = kfold_indices(len(y), n_folds, seed)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    best_lam = select_ridge_lambda(X, y, folds, lambdas)

    oof_pred = np.full(len(y), np.nan, dtype=float)
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)
        w = ridge_fit(X_tr_s, y_tr, best_lam)
        oof_pred[test_idx] = ridge_predict(X_te_s, w)

    rho = spearman_rho(oof_pred, y)
    mse = float(np.mean((oof_pred - y) ** 2))
    mae = float(np.mean(np.abs(oof_pred - y)))

    win_y = (y > 0.01).astype(float)
    if len(np.unique(win_y)) == 2:
        X_s = (X - X.mean(0)) / np.where(X.std(0) < 1e-8, 1.0, X.std(0))
        w_log = logistic_fit(X_s, win_y, lam=0.01)
        proba = logistic_predict_proba(X_s, w_log)
        auc = binary_auc(proba, win_y)
    else:
        auc = None

    policy_rows: List[dict] = []
    for tau in (0.0, 0.01, 0.02, 0.05):
        if task == "method_gain":
            pol = eval_method_policy(oof_pred, y, notta_abs=notta_aux, tau=tau)
        else:
            pol = eval_budget_policy(
                oof_pred, y, aux, notta_vb=notta_aux, tau=tau,
            )
        policy_rows.append(pol)

    impute = {str(j): float(np.median(X[:, j])) for j in range(X.shape[1])}
    X_all_s = (X - X.mean(0)) / np.where(X.std(0) < 1e-8, 1.0, X.std(0))
    w_final = ridge_fit(X_all_s, y, best_lam)

    out = {
        "task": task,
        "n_videos": len(y),
        "ridge_lambda": best_lam,
        "oof_spearman_rho": rho,
        "oof_mse": mse,
        "oof_mae": mae,
        "win_auc": auc,
        "policy_by_tau": policy_rows,
        "weights": {"bias": float(w_final[0]), "coef": w_final[1:].tolist()},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / task
    with (prefix.with_suffix(".json")).open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    pred_csv = prefix.with_name(f"{task}_oof_predictions.csv")
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "y_true", "y_pred"])
        for v, yt, yp in zip(vid, y, oof_pred):
            w.writerow([v, f"{yt:.6f}", f"{yp:.6f}"])

    return out


def run_budget_config_task(
    *,
    video_ids: List[str],
    X: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    notta_vb: np.ndarray,
    grid_runs: Sequence[str],
    output_dir: Path,
    seed: int,
    n_folds: int,
) -> dict:
    """Per-config ridge regressors; deploy by argmax predicted VBench total."""
    mask = ~np.isnan(fixed_vb) & ~np.all(np.isnan(Y), axis=1)
    if mask.sum() < 30:
        raise ValueError(f"budget_config: only {mask.sum()} labeled videos (need ≥30)")

    vid = [video_ids[i] for i in range(len(video_ids)) if mask[i]]
    X = X[mask]
    Y = Y[mask]
    fixed_vb = fixed_vb[mask]
    notta_vb = notta_vb[mask]
    n, k = Y.shape

    folds = kfold_indices(n, n_folds, seed)
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    y_flat = Y[~np.isnan(Y)]
    best_lam = lambdas[0]
    if len(y_flat) >= 30:
        flat_y = Y.T.reshape(-1)
        flat_mask = ~np.isnan(flat_y)
        flat_y = flat_y[flat_mask]
        flat_X = np.repeat(X, k, axis=0)[flat_mask]
        if len(flat_y) >= 30:
            ff = kfold_indices(len(flat_y), min(n_folds, 5), seed)
            best_lam = select_ridge_lambda(flat_X, flat_y, ff, lambdas)

    oof_pick = np.full(n, -1, dtype=int)
    oof_pred_scores = np.full((n, k), np.nan, dtype=float)
    oracle_idx = np.nanargmax(Y, axis=1)

    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        X_tr, X_te = X[train_idx], X[test_idx]
        X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)
        pred_te = np.full((len(test_idx), k), np.nan, dtype=float)
        for j in range(k):
            y_col = Y[train_idx, j]
            m = ~np.isnan(y_col)
            if m.sum() < 10:
                continue
            w = ridge_fit(X_tr_s[m], y_col[m], best_lam)
            pred_te[:, j] = ridge_predict(X_te_s, w)
        oof_pred_scores[test_idx] = pred_te
        with np.errstate(all="ignore"):
            oof_pick[test_idx] = np.nanargmax(pred_te, axis=1)

    acc = float(np.mean(oof_pick == oracle_idx))
    policy = eval_config_pick_policy(oof_pick, Y, fixed_vb, grid_runs, notta_vb=notta_vb)

    X_s = (X - X.mean(0)) / np.where(X.std(0) < 1e-8, 1.0, X.std(0))
    weights: Dict[str, dict] = {}
    for j, rid in enumerate(grid_runs):
        y_col = Y[:, j]
        m = ~np.isnan(y_col)
        if m.sum() < 10:
            continue
        w = ridge_fit(X_s[m], y_col[m], best_lam)
        weights[rid] = {"bias": float(w[0]), "coef": w[1:].tolist()}

    out = {
        "task": "budget_config",
        "n_videos": n,
        "n_configs": k,
        "grid_runs": list(grid_runs),
        "ridge_lambda": best_lam,
        "oof_oracle_match_rate": acc,
        "policy": policy,
        "weights_by_config": weights,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "budget_config"
    with (prefix.with_suffix(".json")).open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    pred_csv = prefix.with_name("budget_config_oof_predictions.csv")
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "picked_run", "oracle_run", "policy_vbench", "oracle_vbench", "fixed_vbench"])
        for idx, v in enumerate(vid):
            pi = int(oof_pick[idx])
            oi = int(oracle_idx[idx])
            pv = Y[idx, pi] if 0 <= pi < k else float("nan")
            ov = Y[idx, oi] if 0 <= oi < k else float("nan")
            w.writerow([
                v,
                grid_runs[pi] if 0 <= pi < k else "",
                grid_runs[oi] if 0 <= oi < k else "",
                f"{pv:.6f}",
                f"{ov:.6f}",
                f"{fixed_vb[idx]:.6f}",
            ])
    return out


def build_report(results: List[dict], *, feature_count: int) -> str:
    lines = [
        "# VBench++ headroom router (learned offline)",
        "",
        "Objective: predict **per-video VBench++ gain / headroom** from Phase-0 features",
        "(no post-TTA leakage). Evaluation uses **out-of-fold** predictions.",
        "",
        f"- **Features:** {feature_count} Phase-0 predictors (median-imputed)",
        "",
    ]
    for r in results:
        if r["task"] == "budget_config":
            pol = r["policy"]
            cap = pol.get("fraction_oracle_captured")
            cap_s = f"{cap * 100:.1f}%" if cap is not None and not math.isnan(cap) else "—"
            pct_n = pol.get("pct_vs_notta")
            pct_f = pol.get("pct_vs_fixed")
            lines += [
                f"## Task: `{r['task']}` (TTA step×LR picker)",
                "",
                f"- **N:** {r['n_videos']} videos × {r['n_configs']} configs",
                f"- **Ridge λ:** {r['ridge_lambda']}",
                f"- **OOF oracle-config match rate:** {r['oof_oracle_match_rate'] * 100:.1f}%",
                f"- **Policy VBench total:** {pol['mean_policy_vbench']:.4f}",
                f"- **Fixed S10/LR5e-3:** {pol['mean_fixed_vbench']:.4f}",
                f"- **Oracle VBench total:** {pol['mean_oracle_vbench']:.4f}",
                f"- **Oracle headroom:** {pol['oracle_headroom']:.4f}",
                f"- **Fraction oracle captured:** {cap_s}",
            ]
            if pct_n is not None:
                lines.append(f"- **Δ vs NOTTA (relative):** {pct_n:+.2f}%")
            if pct_f is not None:
                lines.append(f"- **Δ vs fixed (relative):** {pct_f:+.2f}%")
            if pol.get("top_picks"):
                lines += ["", "**Top OOF config picks:**"]
                for rid, cnt in pol["top_picks"].items():
                    lines.append(f"- `{rid}`: {cnt} videos ({100 * cnt / r['n_videos']:.1f}%)")
            lines.append("")
            continue

        lines += [
            f"## Task: `{r['task']}`",
            "",
            f"- **N:** {r['n_videos']}",
            f"- **Ridge λ (CV):** {r['ridge_lambda']}",
            f"- **OOF Spearman ρ(pred, true):** {r['oof_spearman_rho']:+.3f}"
            if r["oof_spearman_rho"] is not None
            else "- **OOF Spearman:** n/a",
            f"- **OOF MAE:** {r['oof_mae']:.4f}",
            f"- **Win/loss AUC (Δ>0.01):** {r['win_auc']:.3f}"
            if r.get("win_auc") is not None
            else "- **Win/loss AUC:** n/a",
            "",
            "### Deployable policy (OOF predictions, **VBench++ objective**)",
            "",
            "| τ | Apply rate | Oracle headroom | Policy VBench / Δ | Captured | Δ vs NOTTA |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for pol in r["policy_by_tau"]:
            if r["task"] == "method_gain":
                obj = pol["mean_realized_gain"]
            else:
                obj = pol["mean_policy_vbench"]
            cap = pol["fraction_oracle_captured"]
            cap_s = f"{cap * 100:.1f}%" if cap is not None and not math.isnan(cap) else "—"
            pct = pol.get("pct_vs_notta")
            pct_s = f"{pct:+.2f}%" if pct is not None and not math.isnan(pct) else "—"
            lines.append(
                f"| {pol['tau']:.2f} | {pol['apply_rate'] * 100:.1f}% | "
                f"{pol['oracle_headroom']:.4f} | {obj:.4f} | {cap_s} | {pct_s} |"
            )
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        "- Compare **fraction oracle captured** to quintile-adaptive rules (~8–17% on budget pilot).",
        "- **Primary objective:** maximize population **VBench++ total** (report Δ vs NOTTA where available).",
        "- If OOF ρ ≪ 0.2 and captured ≈ 0, linear routers fail — need richer features or nonlinear model.",
        "- **method_gain:** apply/skip AdaSteer; **budget_headroom:** switch vs fixed; **budget_config:** pick step×LR.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Train VBench++ headroom routers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--gains-csv", type=Path, default=None)
    ap.add_argument("--features-csv", type=Path, required=True)
    ap.add_argument("--ood-csv", type=Path, default=None)
    ap.add_argument("--tier3-csv", type=Path, default=None)
    ap.add_argument("--flow-csv", type=Path, default=None)
    ap.add_argument("--bpp-csv", type=Path, default=None)
    ap.add_argument("--fft-csv", type=Path, default=None)
    ap.add_argument("--vae-recerr-csv", type=Path, default=None)
    ap.add_argument("--motion-csv", type=Path, default=None)
    ap.add_argument("--loss-var-csv", type=Path, default=None)
    ap.add_argument("--budget-series-root", type=Path, default=None)
    ap.add_argument("--method", type=str, default=METHOD_DEFAULT)
    ap.add_argument(
        "--task",
        choices=("method_gain", "budget_headroom", "budget_config", "both", "all"),
        default="all",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    task_map = {
        "both": ["method_gain", "budget_headroom"],
        "all": ["method_gain", "budget_headroom", "budget_config"],
    }
    tasks: List[str] = task_map.get(args.task, [args.task])

    features, feature_names, _tiers = join_feature_tables(
        features_csv=args.features_csv,
        ood_csv=args.ood_csv,
        tier3_csv=args.tier3_csv,
        flow_csv=args.flow_csv,
        bpp_csv=args.bpp_csv,
        fft_csv=args.fft_csv,
        vae_recerr_csv=args.vae_recerr_csv,
        motion_csv=args.motion_csv,
        loss_var_csv=args.loss_var_csv,
    )
    if not feature_names:
        print("[error] no features loaded", file=sys.stderr)
        return 2

    video_ids: List[str] = sorted(features.keys())
    y_method = y_budget = aux_budget = notta_method = None
    Y_budget = notta_budget = None
    grid_runs: List[str] = []

    if "method_gain" in tasks:
        if not args.gains_csv or not args.gains_csv.is_file():
            print("[error] --gains-csv required for method_gain", file=sys.stderr)
            return 2
        _vids, gains, _methods = load_vbench_gains(args.gains_csv)
        video_ids = intersect_videos(video_ids, _vids)
        y_method = method_gain_labels(gains, video_ids, method=args.method)
        notta_method = np.array(
            [abs_vbench_total_row(gains.get(vid, {}), "NOTTA") for vid in video_ids],
            dtype=float,
        )

    budget_tasks = [t for t in tasks if t in ("budget_headroom", "budget_config")]
    if budget_tasks:
        if not args.budget_series_root or not args.budget_series_root.is_dir():
            print("[error] --budget-series-root required for budget tasks", file=sys.stderr)
            return 2
        Y_budget, aux_budget, notta_budget, grid_runs, n_b = load_budget_score_matrix(
            args.budget_series_root, video_ids,
        )
        y_budget = Y_budget.max(axis=1) - aux_budget
        print(f"[info] budget labels: {n_b} videos, grid={len(grid_runs)} configs", file=sys.stderr)
        if n_b < 30:
            print(
                f"[warn] only {n_b} budget-labeled videos — "
                "finish S5_LR1e2 VBench backfill for fuller grid",
                file=sys.stderr,
            )

    impute = compute_impute(video_ids, features, feature_names)
    X = build_feature_matrix(video_ids, features, feature_names, impute=impute)

    results: List[dict] = []
    for task in tasks:
        if task == "budget_config":
            if Y_budget is None:
                continue
            print("[train] budget_config ...", file=sys.stderr)
            try:
                res = run_budget_config_task(
                    video_ids=video_ids,
                    X=X,
                    Y=Y_budget,
                    fixed_vb=aux_budget,
                    notta_vb=notta_budget,
                    grid_runs=grid_runs,
                    output_dir=args.output_dir,
                    seed=args.seed,
                    n_folds=args.n_folds,
                )
                results.append(res)
                cap = res["policy"].get("fraction_oracle_captured")
                cap_s = f"{cap:.1%}" if cap is not None and not math.isnan(cap) else "n/a"
                print(
                    f"  match={res['oof_oracle_match_rate']:.1%}  captured={cap_s}",
                    file=sys.stderr,
                )
            except ValueError as e:
                print(f"[skip] budget_config: {e}", file=sys.stderr)
            continue

        y = y_method if task == "method_gain" else y_budget
        aux = None if task == "method_gain" else aux_budget
        notta_aux = notta_method if task == "method_gain" else notta_budget
        if y is None:
            continue
        print(f"[train] {task} ...", file=sys.stderr)
        try:
            res = run_task(
                task=task,
                video_ids=video_ids,
                X=X,
                y=y,
                aux=aux,
                output_dir=args.output_dir,
                seed=args.seed,
                n_folds=args.n_folds,
                notta_aux=notta_aux,
            )
            results.append(res)
            cap = res["policy_by_tau"][0].get("fraction_oracle_captured")
            cap_s = f"{cap:.1%}" if cap is not None and not math.isnan(cap) else "n/a"
            print(
                f"  OOF ρ={res['oof_spearman_rho']:+.3f}  captured@τ=0={cap_s}",
                file=sys.stderr,
            )
        except ValueError as e:
            print(f"[skip] {task}: {e}", file=sys.stderr)

    if not results:
        print("[error] no tasks completed", file=sys.stderr)
        return 2

    report = build_report(results, feature_count=len(feature_names))
    report_path = args.output_dir / "vbench_headroom_router_summary.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
