#!/usr/bin/env python3
"""Diagnose routability: decompose oracle headroom into REAL signal vs NOISE.

Before hunting for "what signal retrieves the oracle headroom", measure how much
of that headroom is even real. For each metric (psnr / vbench) on the paired pool:

  1. within-video config spread  — does config choice move the metric for a fixed
     video? (mean over videos of std across the 12 configs)
  2. mean pairwise Pearson corr among the 12 configs — shared signal vs per-config noise
  3. corr(NOTTA, config-mean) — is no-TTA an INDEPENDENT draw (→ augmented oracle is
     max-of-noise, not routable headroom)?
  4. observed config-oracle gain vs a PURE-NOISE floor E[max of k iid] — how much of the
     oracle gain is explainable by taking a max over noisy measurements
  5. augmented-oracle gain predicted from independence  (mean + sigma/sqrt(pi))
  6. OOF ridge R^2 predicting (a) per-video quality (sanity) and (b) per-video oracle
     gain over fixed (the ROUTABLE part) from all features — the routability ceiling
  7. top single-feature |Pearson| with the per-video oracle gain (incl. probe features)

Offline, ~1 min, no new generation.

Usage:
  python3 scripts/diagnose_routability.py \
    --series-root sweep_experiment/results/panda_ood_budget_1000v_preview \
    --feature-date sweep_experiment/reports/per_video_analysis/2026-07-12 \
    --output-dir sweep_experiment/reports/per_video_analysis/2026-07-21/routability_diag_1000v
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
    kfold_indices,
    ridge_fit,
    ridge_predict,
    select_ridge_lambda,
    standardize_train_test,
)
from scripts.budget_routing_common import load_pilot_bundle  # noqa: E402
from scripts.run_budget_routing_experiments import build_probe_features  # noqa: E402
from scripts.run_router_full_matrix import _load_notta  # noqa: E402

_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _emax_iid(k: int, n_draw: int = 400_000, seed: int = 0) -> float:
    """E[max of k iid standard normals] via Monte Carlo."""
    rng = np.random.default_rng(seed)
    return float(np.mean(np.max(rng.standard_normal((n_draw, k)), axis=1)))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    m = ~np.isnan(a) & ~np.isnan(b)
    if m.sum() < 10:
        return float("nan")
    a2, b2 = a[m], b[m]
    if np.std(a2) < 1e-12 or np.std(b2) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a2, b2)[0, 1])


def _oof_ridge_r2(X: np.ndarray, y: np.ndarray, n_folds: int, seed: int) -> float:
    m = ~np.isnan(y)
    X, y = X[m], y[m]
    if len(y) < 40:
        return float("nan")
    folds = kfold_indices(len(y), n_folds, seed)
    lam = select_ridge_lambda(X, y, kfold_indices(len(y), min(5, n_folds), seed), list(_LAMBDAS))
    pred = np.full(len(y), np.nan)
    for i, te in enumerate(folds):
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        Xtr, Xte, _, _ = standardize_train_test(X[tr], X[te])
        w = ridge_fit(Xtr, y[tr], lam)
        pred[te] = ridge_predict(Xte, w)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")


def diagnose(metric: str, args: argparse.Namespace) -> Tuple[dict, List[str]]:
    bundle = load_pilot_bundle(
        args.series_root, args.feature_date, require_vbench=(metric == "vbench")
    )
    vids: List[str] = bundle["video_ids"]
    Y = np.array(bundle["psnr" if metric == "psnr" else "Y_total"], dtype=float)
    notta = _load_notta(args.series_root, vids, metric)

    # paired pool: all 12 configs present + NOTTA present (clean correlation structure)
    pool = np.all(~np.isnan(Y), axis=1) & ~np.isnan(notta)
    n = int(pool.sum())
    Ym = Y[pool]
    nt = notta[pool]
    vids_p = [vids[i] for i in range(len(vids)) if pool[i]]
    k = Ym.shape[1]

    cfg_mean = np.nanmean(Ym, axis=1)
    within_sigma = float(np.mean(np.nanstd(Ym, axis=1)))          # config spread per video
    between_sigma = float(np.nanstd(cfg_mean))                     # quality spread across videos

    # mean pairwise config correlation
    corrs = []
    for a in range(k):
        for b in range(a + 1, k):
            corrs.append(_pearson(Ym[:, a], Ym[:, b]))
    mean_cc = float(np.nanmean(corrs))
    corr_notta_cfg = _pearson(nt, cfg_mean)

    # best single population config (= "fixed") on this pool
    col_means = np.nanmean(Ym, axis=0)
    best_j = int(np.argmax(col_means))
    fixed = Ym[:, best_j]

    config_oracle = np.nanmax(Ym, axis=1)
    aug_oracle = np.maximum(config_oracle, nt)
    gain_cfg = config_oracle - fixed                              # routable config headroom
    gain_aug = aug_oracle - nt

    # pure-noise floor for config-oracle gain over the per-video config MEAN
    emax = _emax_iid(k)
    noise_floor_gain = within_sigma * emax                        # if 12 configs were iid noise
    obs_gain_over_mean = float(np.mean(config_oracle - cfg_mean))
    # independence prediction for augmented oracle over NOTTA
    indep_aug_pred = (between_sigma + within_sigma) * 0.0 + (np.hypot(between_sigma, within_sigma))
    indep_gain_pred = np.hypot(np.nanstd(cfg_mean), np.nanstd(nt)) / math.sqrt(math.pi)

    # ---- signal scan: predict per-video oracle gain from features -------------
    feat_names = bundle["feat_names"]
    impute = compute_impute(vids_p, bundle["features"], feat_names)
    X_all = build_feature_matrix(vids_p, bundle["features"], feat_names, impute=impute)
    try:
        X_probe, probe_names = build_probe_features(bundle, vids_p, X_all, feat_names)
    except Exception:
        X_probe, probe_names = X_all, feat_names

    r2_quality = _oof_ridge_r2(X_all, cfg_mean, args.n_folds, args.seed)     # sanity (should be >0)
    r2_gain_feat = _oof_ridge_r2(X_all, gain_cfg, args.n_folds, args.seed)   # ROUTABLE part, features
    r2_gain_probe = _oof_ridge_r2(X_probe, gain_cfg, args.n_folds, args.seed)  # + probe outcomes

    # top single-feature |corr| with the oracle gain
    single = []
    for j, name in enumerate(feat_names):
        r = _pearson(X_all[:, j], gain_cfg)
        if not math.isnan(r):
            single.append((abs(r), r, name))
    single.sort(reverse=True)
    top = single[:8]

    res = {
        "metric": metric,
        "n": n,
        "k_configs": k,
        "within_video_config_sigma": within_sigma,
        "between_video_sigma": between_sigma,
        "mean_pairwise_config_corr": mean_cc,
        "corr_notta_vs_configmean": corr_notta_cfg,
        "best_config_idx": best_j,
        "best_config_mean": float(col_means[best_j]),
        "notta_mean": float(np.mean(nt)),
        "config_oracle_gain_over_fixed": float(np.mean(gain_cfg)),
        "config_oracle_gain_over_mean": obs_gain_over_mean,
        "pure_noise_floor_gain_over_mean": noise_floor_gain,
        "aug_oracle_gain_over_notta": float(np.mean(gain_aug)),
        "aug_gain_predicted_if_independent": float(indep_gain_pred),
        "r2_predict_quality_features": r2_quality,
        "r2_predict_oracle_gain_features": r2_gain_feat,
        "r2_predict_oracle_gain_with_probe": r2_gain_probe,
        "top_single_feature_corr_with_gain": [
            {"feature": nm, "corr": rr} for _, rr, nm in top
        ],
    }
    lines = _fmt_metric(res)
    return res, lines


def _f(x: Optional[float], nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _fmt_metric(r: dict) -> List[str]:
    unit = "dB" if r["metric"] == "psnr" else "raw-total"
    L = [
        f"## metric = {r['metric']} ({unit}) · N={r['n']} · {r['k_configs']} configs",
        "",
        "| quantity | value | reading |",
        "|---|---:|---|",
        f"| within-video config σ | {_f(r['within_video_config_sigma'])} | how much config "
        "choice moves the metric for a fixed video (small ⇒ config barely matters) |",
        f"| between-video σ | {_f(r['between_video_sigma'])} | quality spread across videos |",
        f"| mean pairwise config corr | {_f(r['mean_pairwise_config_corr'])} | configs' shared "
        "signal (high ⇒ configs agree per video) |",
        f"| corr(NOTTA, config-mean) | {_f(r['corr_notta_vs_configmean'])} | is no-TTA an "
        "INDEPENDENT draw? (≈0 ⇒ aug-oracle is max-of-noise) |",
        f"| config-oracle gain / fixed | {_f(r['config_oracle_gain_over_fixed'])} | routable "
        "per-video config headroom |",
        f"| config-oracle gain / mean | {_f(r['config_oracle_gain_over_mean'])} | vs pure-noise "
        f"floor {_f(r['pure_noise_floor_gain_over_mean'])} (≈ ⇒ headroom is noise) |",
        f"| aug-oracle gain / NOTTA | {_f(r['aug_oracle_gain_over_notta'])} | vs independence "
        f"prediction {_f(r['aug_gain_predicted_if_independent'])} (≈ ⇒ no-TTA is noise draw) |",
        "",
        "**Routability (OOF ridge R², leakage-free):**",
        "",
        f"- predict per-video QUALITY from features: R² = **{_f(r['r2_predict_quality_features'])}** "
        "(sanity — features should predict overall quality)",
        f"- predict per-video ORACLE GAIN from features: R² = **{_f(r['r2_predict_oracle_gain_features'])}** "
        "(≤0 ⇒ no routable signal in static features)",
        f"- predict per-video ORACLE GAIN + probe outcomes: R² = **{_f(r['r2_predict_oracle_gain_with_probe'])}** "
        "(observing probe metrics; needs GT ⇒ upper bound, not deployable)",
        "",
        "Top single features by |corr| with per-video oracle gain:",
        "",
        "| feature | corr |",
        "|---|---:|",
    ]
    for e in r["top_single_feature_corr_with_gain"]:
        L.append(f"| `{e['feature']}` | {e['corr']:+.4f} |")
    L.append("")
    return L


def main() -> int:
    ap = argparse.ArgumentParser(description="Routability diagnostic")
    ap.add_argument(
        "--series-root",
        type=Path,
        default=_REPO / "sweep_experiment/results/panda_ood_budget_1000v_preview",
    )
    ap.add_argument(
        "--feature-date",
        type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-12",
    )
    ap.add_argument("--metrics", nargs="+", default=["psnr", "vbench"], choices=("psnr", "vbench"))
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_res = {}
    lines = [
        "# Routability diagnostic — real signal vs noise in the oracle headroom",
        "",
        f"**Series:** `{args.series_root.name}`  ·  **Features:** `{args.feature_date.name}`.",
        "",
    ]
    for metric in args.metrics:
        res, ls = diagnose(metric, args)
        all_res[metric] = res
        lines.extend(ls)
        # stderr digest
        print(
            f"[{metric}] N={res['n']} within_cfg_sigma={res['within_video_config_sigma']:.4f} "
            f"corr_cc={res['mean_pairwise_config_corr']:.3f} "
            f"corr(notta,cfg)={res['corr_notta_vs_configmean']:.3f} "
            f"oracle_gain/fixed={res['config_oracle_gain_over_fixed']:.4f} "
            f"R2_gain_feat={res['r2_predict_oracle_gain_features']:.4f} "
            f"R2_gain_probe={res['r2_predict_oracle_gain_with_probe']:.4f}",
            file=sys.stderr,
        )

    (args.output_dir / "routability_diag.json").write_text(
        json.dumps(all_res, indent=2), encoding="utf-8"
    )
    report = args.output_dir / "routability_diag_summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
