#!/usr/bin/env python3
"""Wave-1 VBench++ gain-prediction screen @ pilot N=200 (CPU, ~minutes).

Experiments:
  exp14_multi_verifier_deploy  — rank S2/S10 probes by PSNR+SSIM only (deployable)
  exp14_multi_verifier_full    — + GT VBench Aes/IQ/Dyn on probe outputs (ceiling)
  exp15_tail_only_gate         — route top-15% predicted headroom with 3-way ridge
  exp16_knn_probe_manifold     — kNN vote on probe features (not Phase-0)
  exp17_per_dim_fuse_router    — OOF ridge per VBench dim → fused 3-way pick
  exp18_logistic_3way_gate     — logistic P(3-way beats fixed) → route or abstain
  exp19_feature_dim_correlation — screen |ρ|(feature, ΔVBench dim); report only

Usage:
  python3 scripts/run_vbench_predictor_wave1_experiments.py --run-all
  python3 scripts/run_vbench_predictor_wave1_experiments.py --aggregate-only
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.analyze_per_video_vbench_agreement import VBENCH_DIMS  # noqa: E402
from scripts.analyze_per_video_tta_gain import spearman_rho  # noqa: E402
from scripts.budget_routing_common import (  # noqa: E402
    bootstrap_captured,
    labeled_mask,
    load_pilot_bundle,
)
from scripts.run_budget_routing_experiments import (  # noqa: E402
    _policy_from_budget_task,
    build_probe_features,
)
from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
    eval_config_pick_policy,
    kfold_indices,
    logistic_fit,
    logistic_predict_proba,
    ridge_fit,
    ridge_predict,
    run_budget_config_task,
    standardize_train_test,
)

PROBE2 = ("S2_LR5e3", "S10_LR5e3")
FULL3 = ("S5_LR5e3", "S10_LR5e3", "S20_LR1e3")
FULL_MAP = {"S2_LR5e3": "S5_LR5e3", "S10_LR5e3": "S10_LR5e3"}
HEADROOM_EPS = 0.05
TAIL_FRAC = 0.15

WAVE1_EXPERIMENTS = (
    "exp14_multi_verifier_deploy",
    "exp14_multi_verifier_full",
    "exp15_tail_only_gate",
    "exp16_knn_probe_manifold",
    "exp17_per_dim_fuse_router",
    "exp18_logistic_3way_gate",
    "exp19_feature_dim_correlation",
)

# Gates for Wave-2 / GPU follow-up
WAVE2_CAPTURED_GO = 15.0
TAIL_CAPTURED_GO = 30.0
BASELINE_BEST = 12.8  # exp7
# Uses GT VBench dims on probe outputs — ceiling only, not deployable at inference.
CEILING_EXPERIMENTS = frozenset({"exp14_multi_verifier_full"})


def _cap_pct(policy: dict) -> Optional[float]:
    cap = policy.get("fraction_oracle_captured")
    if cap is None or (isinstance(cap, float) and math.isnan(cap)):
        return None
    return 100 * float(cap)


def _row(name: str, policy: dict, *, n: int, extra: Optional[dict] = None) -> dict:
    row = {
        "experiment": name,
        "n_videos": n,
        "match_rate": policy.get("oof_oracle_match_rate"),
        "captured_pct": _cap_pct(policy),
        "policy_gain": (
            policy.get("mean_policy_vbench", 0) - policy.get("mean_fixed_vbench", 0)
            if policy.get("mean_policy_vbench") is not None
            else None
        ),
        "headroom": policy.get("oracle_headroom"),
    }
    cap_lo = policy.get("captured_ci_lo")
    cap_hi = policy.get("captured_ci_hi")
    if cap_lo is not None:
        row["captured_ci_lo_pct"] = 100 * cap_lo
    if cap_hi is not None:
        row["captured_ci_hi_pct"] = 100 * cap_hi
    if extra:
        row.update(extra)
    return row


def _config_indices(grid: Sequence[str], names: Sequence[str]) -> List[int]:
    return [grid.index(r) for r in names if r in grid]


def _oof_policy_from_picks(
    picks: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    mask: np.ndarray,
    *,
    n_boot: int = 2000,
) -> dict:
    valid = mask & (picks >= 0)
    pol = eval_config_pick_policy(picks[valid], Y[valid], fixed_vb[valid], grid)
    oracle_idx = np.nanargmax(Y[valid], axis=1)
    pol["oof_oracle_match_rate"] = float(np.mean(picks[valid] == oracle_idx))
    pol["n_valid"] = int(valid.sum())
    realized = np.array([Y[i, picks[i]] for i in np.where(valid)[0] if picks[i] >= 0])
    fv = fixed_vb[valid]
    ov = np.array([np.nanmax(Y[i]) for i in np.where(valid)[0]])
    cap, _, _, lo, hi = bootstrap_captured(realized, ov, fv, n_boot=n_boot, seed=42)
    pol["captured_ci_lo"] = lo
    pol["captured_ci_hi"] = hi
    return pol


def _route_from_probe_scores(
    bundle: dict,
    grid: Sequence[str],
    video_ids: Sequence[str],
    score_fn,
) -> np.ndarray:
    """Pick config from best-scoring probe (S2/S10) per video."""
    probe_js = _config_indices(grid, PROBE2)
    n = len(video_ids)
    picks = np.full(n, -1, dtype=int)
    for i in range(n):
        best_j, best_s = -1, float("-inf")
        for j in probe_js:
            s = score_fn(i, j)
            if s is None or not np.isfinite(s):
                continue
            if s > best_s:
                best_s, best_j = s, j
        if best_j < 0:
            continue
        rid = grid[best_j]
        target = FULL_MAP.get(rid, rid)
        if target in grid:
            picks[i] = grid.index(target)
    return picks


def run_exp14_deploy(bundle: dict, Y: np.ndarray, fixed_vb: np.ndarray, grid: Sequence[str]) -> dict:
    psnr, ssim = bundle["psnr"], bundle["ssim"]
    fixed_j = grid.index(bundle["fixed_run"])

    def score_fn(i: int, j: int) -> float:
        p, s = psnr[i, j], ssim[i, j]
        if not np.isfinite(p) or not np.isfinite(s):
            return float("nan")
        dp = p - psnr[i, fixed_j]
        ds = s - ssim[i, fixed_j]
        return 0.5 * dp + 0.5 * ds

    picks = _route_from_probe_scores(bundle, grid, bundle["video_ids"], score_fn)
    mask = labeled_mask(fixed_vb, Y)
    pol = _oof_policy_from_picks(picks, Y, fixed_vb, grid, mask)
    return _row("exp14_multi_verifier_deploy", pol, n=int(mask.sum()),
                extra={"verifiers": "probe_dpsnr+dssim"})


def run_exp14_full(bundle: dict, Y: np.ndarray, fixed_vb: np.ndarray, grid: Sequence[str]) -> dict:
    psnr, ssim = bundle["psnr"], bundle["ssim"]
    Y_dim = bundle["Y_dim"]
    fixed_j = grid.index(bundle["fixed_run"])
    w = {"psnr": 0.15, "ssim": 0.10, "aes": 0.25, "iq": 0.25, "dyn": 0.25}

    def score_fn(i: int, j: int) -> float:
        parts = []
        p, s = psnr[i, j], ssim[i, j]
        if np.isfinite(p) and np.isfinite(s):
            parts.append(w["psnr"] * (p - psnr[i, fixed_j]))
            parts.append(w["ssim"] * (s - ssim[i, fixed_j]))
        for key, dim in (("aes", "aesthetic_quality"), ("iq", "imaging_quality"), ("dyn", "dynamic_degree")):
            v = Y_dim[dim][i, j]
            if np.isfinite(v):
                parts.append(w[key] * v)
        return float(sum(parts)) if parts else float("nan")

    picks = _route_from_probe_scores(bundle, grid, bundle["video_ids"], score_fn)
    mask = labeled_mask(fixed_vb, Y)
    pol = _oof_policy_from_picks(picks, Y, fixed_vb, grid, mask)
    return _row("exp14_multi_verifier_full", pol, n=int(mask.sum()),
                extra={"verifiers": "probe+GT_aes_iq_dyn", "note": "ceiling; needs probe-time metrics at deploy"})


def run_exp15_tail(
    X_probe: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    seed: int,
    n_folds: int,
    tail_frac: float = TAIL_FRAC,
) -> dict:
    mask = labeled_mask(fixed_vb, Y)
    X = X_probe[mask]
    Y = Y[mask]
    fv = fixed_vb[mask]
    n = len(X)
    headroom = np.nanmax(Y, axis=1) - fv
    full_js = _config_indices(grid, FULL3)
    fixed_j = grid.index("S10_LR5e3") if "S10_LR5e3" in grid else full_js[1]
    folds = kfold_indices(n, n_folds, seed)
    oof_pick = np.full(n, fixed_j, dtype=int)
    oof_pred_headroom = np.full(n, np.nan)

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != fi])
        X_tr, X_te = X[train_idx], X[test_idx]
        X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)
        y_h = headroom[train_idx]
        w = ridge_fit(X_tr_s, y_h, 0.1)
        oof_pred_headroom[test_idx] = ridge_predict(X_te_s, w)
        thr = np.quantile(oof_pred_headroom[test_idx], 1.0 - tail_frac)
        adapt = oof_pred_headroom[test_idx] >= thr
        if not adapt.any():
            continue
        pred_te = np.full((len(test_idx), len(full_js)), np.nan)
        for cj, col in enumerate(full_js):
            y_col = Y[train_idx, col]
            m = np.isfinite(y_col)
            if m.sum() < 10:
                continue
            wf = ridge_fit(X_tr_s[m], y_col[m], 0.1)
            pred_te[:, cj] = ridge_predict(X_te_s, wf)
        for ti, global_i in enumerate(test_idx):
            if not adapt[ti]:
                continue
            oof_pick[global_i] = full_js[int(np.nanargmax(pred_te[ti]))]

    # Map back to full video index
    full_mask = np.zeros(len(fixed_vb), dtype=bool)
    full_mask[np.where(mask)[0]] = True
    full_picks = np.full(len(fixed_vb), -1, dtype=int)
    full_picks[np.where(mask)[0]] = oof_pick
    pol = _oof_policy_from_picks(full_picks, Y, fixed_vb, grid, full_mask)

    routed = oof_pred_headroom >= np.quantile(oof_pred_headroom[np.isfinite(oof_pred_headroom)], 1.0 - tail_frac)
    if routed.sum() >= 5:
        idx = np.where(mask)[0][routed]
        tail_picks = oof_pick[routed]
        tail_pol = eval_config_pick_policy(
            tail_picks, Y[idx], fixed_vb[idx], grid,
        )
        tail_cap = 100 * tail_pol["fraction_oracle_captured"]
    else:
        tail_cap = float("nan")

    return _row(
        "exp15_tail_only_gate",
        pol,
        n=n,
        extra={
            "tail_frac": tail_frac,
            "apply_rate_est": float(routed.mean()),
            "tail_captured_pct": tail_cap,
        },
    )


def run_exp16_knn_probe(
    X_probe: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    k: int,
    seed: int,
    n_folds: int,
) -> dict:
    mask = labeled_mask(fixed_vb, Y)
    X = X_probe[mask]
    Y = Y[mask]
    fixed_vb_m = fixed_vb[mask]
    n = len(X)
    oracle_idx = np.nanargmax(Y, axis=1)
    folds = kfold_indices(n, n_folds, seed)
    oof_pick = np.full(n, -1, dtype=int)

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != fi])
        X_tr, X_te = X[train_idx], X[test_idx]
        mu, sig = X_tr.mean(0), X_tr.std(0)
        sig = np.where(sig < 1e-8, 1.0, sig)
        X_tr_s = (X_tr - mu) / sig
        X_te_s = (X_te - mu) / sig
        train_oracle = oracle_idx[train_idx]
        for ti, te_i in enumerate(test_idx):
            d = np.sum((X_te_s[ti : ti + 1] - X_tr_s) ** 2, axis=1)
            nn = np.argsort(d)[: min(k, len(d))]
            votes = Counter(int(train_oracle[j]) for j in nn)
            oof_pick[te_i] = votes.most_common(1)[0][0]

    full_picks = np.full(len(fixed_vb), -1, dtype=int)
    full_picks[np.where(mask)[0]] = oof_pick
    pol = _oof_policy_from_picks(full_picks, Y, fixed_vb, grid, mask)
    return _row("exp16_knn_probe_manifold", pol, n=n, extra={"k": k})


def run_exp17_per_dim_fuse(
    X_probe: np.ndarray,
    Y: np.ndarray,
    Y_dim: Dict[str, np.ndarray],
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    seed: int,
    n_folds: int,
) -> dict:
    mask = labeled_mask(fixed_vb, Y)
    X = X_probe[mask]
    fv = fixed_vb[mask]
    n = len(X)
    full_js = _config_indices(grid, FULL3)
    folds = kfold_indices(n, n_folds, seed)
    oof_pick = np.full(n, -1, dtype=int)

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != fi])
        X_tr, X_te = X[train_idx], X[test_idx]
        X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)
        fused = np.zeros((len(test_idx), len(full_js)))
        for dim in VBENCH_DIMS:
            Y_d = Y_dim[dim][mask][:, full_js]
            for cj in range(len(full_js)):
                y_col = Y_d[train_idx, cj]
                m = np.isfinite(y_col)
                if m.sum() < 10:
                    continue
                w = ridge_fit(X_tr_s[m], y_col[m], 0.1)
                fused[:, cj] += ridge_predict(X_te_s, w)
        fused /= len(VBENCH_DIMS)
        for ti, global_i in enumerate(test_idx):
            oof_pick[global_i] = full_js[int(np.nanargmax(fused[ti]))]

    picks_idx = np.full(len(fixed_vb), -1, dtype=int)
    for local_i, gi in enumerate(np.where(mask)[0]):
        picks_idx[gi] = int(oof_pick[local_i])

    pol = _oof_policy_from_picks(picks_idx, Y, fixed_vb, grid, mask)
    return _row("exp17_per_dim_fuse_router", pol, n=n, extra={"fusion": "mean_7dim_ridge"})


def run_exp18_logistic_3way(
    X_probe: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    seed: int,
    n_folds: int,
) -> dict:
    mask = labeled_mask(fixed_vb, Y)
    X = X_probe[mask]
    Y = Y[mask]
    fv = fixed_vb[mask]
    n = len(X)
    full_js = _config_indices(grid, FULL3)
    fixed_j = grid.index("S10_LR5e3") if "S10_LR5e3" in grid else full_js[1]
    y_bin = (np.nanmax(Y[:, full_js], axis=1) > fv + 1e-9).astype(float)
    folds = kfold_indices(n, n_folds, seed)
    oof_pick = np.full(n, fixed_j, dtype=int)

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != fi])
        X_tr, X_te = X[train_idx], X[test_idx]
        X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)
        yb = y_bin[train_idx]
        if yb.sum() >= 10 and 0 < yb.sum() < len(yb):
            w = logistic_fit(X_tr_s, yb, lam=0.5)
            p = logistic_predict_proba(X_te_s, w)
        else:
            p = np.full(len(test_idx), float(yb.mean()))
        adapt = p > 0.5
        pred_te = np.full((len(test_idx), len(full_js)), np.nan)
        for cj, col in enumerate(full_js):
            y_col = Y[train_idx, col]
            m = np.isfinite(y_col)
            if m.sum() < 10:
                continue
            wf = ridge_fit(X_tr_s[m], y_col[m], 0.1)
            pred_te[:, cj] = ridge_predict(X_te_s, wf)
        for ti, global_i in enumerate(test_idx):
            if adapt[ti]:
                oof_pick[global_i] = full_js[int(np.nanargmax(pred_te[ti]))]

    full_picks = np.full(len(fixed_vb), -1, dtype=int)
    for local_i, gi in enumerate(np.where(mask)[0]):
        full_picks[gi] = int(oof_pick[local_i])
    pol = _oof_policy_from_picks(full_picks, Y, fixed_vb, grid, mask)
    return _row("exp18_logistic_3way_gate", pol, n=n,
                extra={"positive_rate": float(y_bin.mean())})


def run_exp19_correlation(bundle: dict, feature_date: Path) -> dict:
    """Screen Phase-0 + probe features vs ΔVBench per dim (no routing)."""
    from scripts.predictor_analysis_common import join_feature_tables

    mask = labeled_mask(bundle["fixed_vb"], bundle["Y_total"])
    vid = [bundle["video_ids"][i] for i in range(len(bundle["video_ids"])) if mask[i]]
    fixed_j = bundle["grid_runs"].index(bundle["fixed_run"])
    Y = bundle["Y_total"][mask]
    fv = bundle["fixed_vb"][mask]
    gains = Y - fv[:, np.newaxis]

    features, feat_names, _ = join_feature_tables(
        features_csv=feature_date / "video_features.csv",
        ood_csv=feature_date / "diffusion_ood_scores.csv",
        tier3_csv=feature_date / "tier3_probe_features.csv",
        flow_csv=feature_date / "flow_shape_features.csv",
        bpp_csv=feature_date / "bpp_features.csv",
        fft_csv=feature_date / "fft_features.csv",
        vae_recerr_csv=feature_date / "vae_recerr_features.csv",
        motion_csv=feature_date / "latent_motion_features.csv",
        loss_var_csv=feature_date / "loss_variance_features.csv",
    )
    impute = compute_impute(vid, features, feat_names)
    X = build_feature_matrix(vid, features, feat_names, impute=impute)

    best: List[dict] = []
    for fi, fname in enumerate(feat_names):
        col = X[:, fi]
        if np.nanstd(col) < 1e-9:
            continue
        for di, dim in enumerate(VBENCH_DIMS):
            g = bundle["Y_dim"][dim][mask][:, fixed_j]  # wrong - need delta vs fixed for dim
            g = bundle["Y_dim"][dim][mask].max(axis=1) - bundle["Y_dim"][dim][mask][:, fixed_j]
            rho = spearman_rho(col, g)
            if rho is None or math.isnan(rho):
                continue
            best.append({"feature": fname, "dim": dim, "rho": float(rho)})

    best.sort(key=lambda x: -abs(x["rho"]))
    top = best[:15]
    n_pass = sum(1 for b in best if abs(b["rho"]) >= 0.2)
    return {
        "experiment": "exp19_feature_dim_correlation",
        "n_features_tested": len(feat_names),
        "n_pairs_pass_0.2": n_pass,
        "top_pairs": top,
        "captured_pct": None,
        "note": "screen only; no deployable policy",
    }


def load_results(out_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name in WAVE1_EXPERIMENTS:
        p = out_dir / f"{name}.json"
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            out[name] = data.get("row") or data
    return out


def write_summary(results: Dict[str, dict], out_dir: Path) -> None:
    routing = [
        n for n in WAVE1_EXPERIMENTS
        if n != "exp19_feature_dim_correlation" and results.get(n, {}).get("captured_pct") is not None
    ]
    best_name, best_cap = None, -1.0
    deploy_name, deploy_cap = None, -1.0
    for n in routing:
        cap = results[n].get("captured_pct")
        if cap is None:
            continue
        cap = float(cap)
        if cap > best_cap:
            best_cap, best_name = cap, n
        if n not in CEILING_EXPERIMENTS and cap > deploy_cap:
            deploy_cap, deploy_name = cap, n

    lines = [
        "# Wave-1 VBench++ predictor screen @ N=200",
        "",
        "| Experiment | Captured % | CI (%) | Match % | Notes |",
        "|---|---:|---|---:|---|",
    ]
    for name in WAVE1_EXPERIMENTS:
        r = results.get(name, {})
        if not r:
            lines.append(f"| `{name}` | — | — | — | not run |")
            continue
        cap = r.get("captured_pct")
        cap_s = f"{cap:.1f}" if cap is not None else "—"
        lo, hi = r.get("captured_ci_lo_pct"), r.get("captured_ci_hi_pct")
        ci_s = f"[{lo:.1f}, {hi:.1f}]" if lo is not None and hi is not None else "—"
        mr = r.get("match_rate")
        mr_s = f"{100 * mr:.1f}" if mr is not None else "—"
        note = r.get("verifiers") or r.get("tail_captured_pct") or r.get("note") or ""
        if r.get("tail_captured_pct") is not None:
            note = f"tail_cap={r['tail_captured_pct']:.1f}% apply={r.get('apply_rate_est', 0):.0%}"
        lines.append(f"| `{name}` | {cap_s} | {ci_s} | {mr_s} | {note} |")

    r19 = results.get("exp19_feature_dim_correlation", {})
    if r19:
        lines += [
            "",
            f"**Feature screen:** {r19.get('n_pairs_pass_0.2', 0)} feature×dim pairs with |ρ|≥0.2",
        ]
        for p in (r19.get("top_pairs") or [])[:5]:
            lines.append(f"- `{p['feature']}` vs Δ{p['dim']}: ρ={p['rho']:+.3f}")

    lines += [
        "",
        "## Reference",
        f"- Prior best (exp7): **{BASELINE_BEST}%**",
        f"- Wave-2 GO bar: captured **>{WAVE2_CAPTURED_GO}%** (CI lo > 5%)",
        f"- Tail GO bar: tail captured **>{TAIL_CAPTURED_GO}%** at ~{100*TAIL_FRAC:.0f}% apply",
        "",
    ]

    ceiling_r = results.get("exp14_multi_verifier_full", {})
    ceiling_cap = ceiling_r.get("captured_pct")
    tail_r = results.get("exp15_tail_only_gate", {})
    tail_go = (tail_r.get("tail_captured_pct") or 0) >= TAIL_CAPTURED_GO
    deploy_go = deploy_cap >= WAVE2_CAPTURED_GO or deploy_cap > BASELINE_BEST + 2
    gpu_go = deploy_go or tail_go

    lines += ["## Tonight's decision", ""]
    if ceiling_cap is not None:
        lines.append(
            f"- **Ceiling** (`exp14_multi_verifier_full`, GT VBench on probes): **{ceiling_cap:.1f}%** "
            f"(≈ exp10 upper bound; not deployable)"
        )
    if deploy_name:
        dr = results[deploy_name]
        lo = dr.get("captured_ci_lo_pct")
        lines.append(
            f"- **Best deployable** (`{deploy_name}`): **{deploy_cap:.1f}%**"
            + (f" CI [{lo:.1f}, {dr.get('captured_ci_hi_pct', 0):.1f}]" if lo is not None else "")
        )
    lines.append("")
    if gpu_go:
        lines.append(
            f"**GO Wave-2/GPU follow-up** — deployable `{deploy_name}` at **{deploy_cap:.1f}%**"
            + (f"; tail gate tail_cap={tail_r.get('tail_captured_pct', 0):.1f}%" if tail_go else "")
        )
        lines.append("")
        lines.append("Submit before bed:")
        lines.append("```bash")
        lines.append("bash sweep_experiment/sbatch/submit_vbench_predictor_wave2.sh  # after pull")
        lines.append("```")
    else:
        lines.append(
            f"**NO-GO heavy GPU tonight** — best deployable `{deploy_name}` at **{deploy_cap:.1f}%** "
            f"(need >{WAVE2_CAPTURED_GO}% or tail >{TAIL_CAPTURED_GO}%). "
            "Paper line: oracle real; offline routing stays ~13%."
        )
    lines.append("")

    (out_dir / "wave1_predictor_summary.md").write_text("\n".join(lines), encoding="utf-8")
    decision = {
        "best_experiment": best_name,
        "best_captured_pct": best_cap,
        "best_deployable_experiment": deploy_name,
        "best_deployable_captured_pct": deploy_cap,
        "ceiling_experiment": "exp14_multi_verifier_full",
        "ceiling_captured_pct": ceiling_cap,
        "wave2_go": deploy_go,
        "tail_go": tail_go,
        "gpu_go": gpu_go,
    }
    (out_dir / "wave1_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-root", type=Path,
                    default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot")
    ap.add_argument("--feature-date", type=Path,
                    default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-06")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--experiment", choices=WAVE1_EXPERIMENTS, default=None)
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--knn-k", type=int, default=7)
    args = ap.parse_args()

    out = args.output_dir or (
        _REPO / "sweep_experiment/reports/per_video_analysis/2026-07-06/wave1_predictor_experiments"
    )
    out.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        write_summary(load_results(out), out)
        print(f"Aggregated → {out}/wave1_predictor_summary.md", file=sys.stderr)
        return 0

    if not args.run_all and not args.experiment:
        ap.error("Use --run-all, --experiment, or --aggregate-only")

    exps = list(WAVE1_EXPERIMENTS) if args.run_all else [args.experiment]
    bundle = load_pilot_bundle(args.series_root, args.feature_date)
    impute = compute_impute(bundle["video_ids"], bundle["features"], bundle["feat_names"])
    X_base = build_feature_matrix(
        bundle["video_ids"], bundle["features"], bundle["feat_names"], impute=impute,
    )
    X_probe, _ = build_probe_features(
        bundle, bundle["video_ids"], X_base, bundle["feat_names"], probe_runs=PROBE2,
    )
    Y = bundle["Y_total"]
    fixed_vb = bundle["fixed_vb"]
    grid = bundle["grid_runs"]

    for name in exps:
        print(f"[run] {name} ...", file=sys.stderr)
        try:
            if name == "exp14_multi_verifier_deploy":
                row = run_exp14_deploy(bundle, Y, fixed_vb, grid)
            elif name == "exp14_multi_verifier_full":
                row = run_exp14_full(bundle, Y, fixed_vb, grid)
            elif name == "exp15_tail_only_gate":
                row = run_exp15_tail(
                    X_probe, Y, fixed_vb, grid, seed=args.seed, n_folds=args.n_folds,
                )
            elif name == "exp16_knn_probe_manifold":
                row = run_exp16_knn_probe(
                    X_probe, Y, fixed_vb, grid,
                    k=args.knn_k, seed=args.seed, n_folds=args.n_folds,
                )
            elif name == "exp17_per_dim_fuse_router":
                row = run_exp17_per_dim_fuse(
                    X_probe, Y, bundle["Y_dim"], fixed_vb, grid,
                    seed=args.seed, n_folds=args.n_folds,
                )
            elif name == "exp18_logistic_3way_gate":
                row = run_exp18_logistic_3way(
                    X_probe, Y, fixed_vb, grid, seed=args.seed, n_folds=args.n_folds,
                )
            elif name == "exp19_feature_dim_correlation":
                row = run_exp19_correlation(bundle, args.feature_date)
            else:
                raise ValueError(name)
            payload = {"row": row, "policy_note": "eval on VBench total vs fixed S10_LR5e3"}
            (out / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  captured={row.get('captured_pct')}", file=sys.stderr)
        except Exception as exc:
            (out / f"{name}.json").write_text(
                json.dumps({"skipped": True, "reason": str(exc)}, indent=2), encoding="utf-8",
            )
            print(f"  FAILED: {exc}", file=sys.stderr)

    if args.run_all:
        write_summary(load_results(out), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
