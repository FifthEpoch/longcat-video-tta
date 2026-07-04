#!/usr/bin/env python3
"""Budget-grid routing experiment suite @ pilot N=200 (no new video generation).

Uses existing 12-config VBench + PSNR/SSIM from ``panda_ood_budget_pilot``.

Experiments:
  baseline_linear_total   — ridge argmax VBench total (replicates budget_config)
  dim_<name>              — ridge argmax single VBench dimension
  coarse_steps_lr         — predict step bucket then LR within bucket
  probe_simulated         — Phase-0 + probe-config PSNR/SSIM deltas (S2/S10/S20)
  proxy_psnr_all          — pick config with max PSNR (NR proxy ceiling)
  proxy_bestof3_psnr      — best-of-3 PSNR among S2/S10/S20 LR5e3/1e3
  pairwise_logistic_top4  — OOF pairwise logistic among top-4 train configs
  pairwise_gbm_top4       — HistGradientBoosting pairs (if sklearn installed)
  composite_psnr_ridge    — ridge on Phase-0 + all-config PSNR columns
  mlp_shallow             — 1-hidden-layer ReLU config picker

Usage:
    python3 scripts/run_budget_routing_experiments.py --run-all
    python3 scripts/run_budget_routing_experiments.py --experiment dim_dynamic_degree
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

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.budget_routing_common import (  # noqa: E402
    BESTOF3_RUNS,
    PROBE_RUNS,
    bootstrap_captured,
    labeled_mask,
    load_pilot_bundle,
    steps_bucket,
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
    select_ridge_lambda,
    standardize_train_test,
)

try:
    from sklearn.ensemble import HistGradientBoostingClassifier

    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

DIM_EXPERIMENTS = (
    "dim_dynamic_degree",
    "dim_aesthetic_quality",
    "dim_imaging_quality",
    "dim_subject_consistency",
)

ALL_EXPERIMENTS = (
    "baseline_linear_total",
    *DIM_EXPERIMENTS,
    "coarse_steps_lr",
    "probe_simulated",
    "proxy_psnr_all",
    "proxy_bestof3_psnr",
    "pairwise_logistic_top4",
    "pairwise_gbm_top4",
    "composite_psnr_ridge",
    "mlp_shallow",
)


def _num(v: Optional[float], default: float = 0.0) -> float:
    return default if v is None else float(v)


def _policy_from_budget_task(res: dict) -> dict:
    policy = dict(res["policy"])
    policy["oof_oracle_match_rate"] = res.get("oof_oracle_match_rate")
    return policy


def _write_experiment_json(output_dir: Path, name: str, payload: dict) -> None:
    (output_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _submatrix(Y: np.ndarray, grid_runs: Sequence[str], picks: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    idx = [grid_runs.index(p) for p in picks if p in grid_runs]
    names = [grid_runs[i] for i in idx]
    return Y[:, idx], names


def build_probe_features(
    bundle: dict,
    video_ids: Sequence[str],
    base_X: np.ndarray,
    feat_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Simulate probe-and-route: S2/S10/S20 PSNR+SSIM + deltas vs fixed."""
    grid = bundle["grid_runs"]
    psnr = bundle["psnr"]
    ssim = bundle["ssim"]
    fixed_j = grid.index(bundle["fixed_run"])
    cols: List[np.ndarray] = []
    names: List[str] = list(feat_names)
    out = base_X.copy()
    for rid in PROBE_RUNS:
        if rid not in grid:
            continue
        j = grid.index(rid)
        p = psnr[:, j]
        s = ssim[:, j]
        cols.extend([p, s, p - psnr[:, fixed_j], s - ssim[:, fixed_j]])
        names.extend([f"probe_{rid}_psnr", f"probe_{rid}_ssim", f"probe_{rid}_dpsnr", f"probe_{rid}_dssim"])
    if cols:
        out = np.column_stack([out] + cols)
    return out, names


def build_psnr_feature_block(bundle: dict, base_X: np.ndarray, feat_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    grid = bundle["grid_runs"]
    psnr = bundle["psnr"]
    names = list(feat_names) + [f"psnr_{rid}" for rid in grid]
    block = np.column_stack([base_X] + [psnr[:, j] for j in range(len(grid))])
    return block, names


def run_proxy_pick(
    Y: np.ndarray,
    proxy: np.ndarray,
    fixed_vb: np.ndarray,
    grid_runs: Sequence[str],
    *,
    allowed_cols: Optional[Sequence[int]] = None,
) -> dict:
    n, k = Y.shape
    picks = np.full(n, -1, dtype=int)
    for i in range(n):
        if allowed_cols is not None:
            scores = [(j, proxy[i, j]) for j in allowed_cols if np.isfinite(proxy[i, j])]
        else:
            scores = [(j, proxy[i, j]) for j in range(k) if np.isfinite(proxy[i, j])]
        if scores:
            picks[i] = max(scores, key=lambda x: x[1])[0]
    policy = eval_config_pick_policy(picks, Y, fixed_vb, grid_runs)
    oracle_idx = np.nanargmax(Y, axis=1)
    policy["oof_oracle_match_rate"] = float(np.mean(picks == oracle_idx))
    return policy


def top_config_indices(Y_train: np.ndarray, k_top: int = 4) -> List[int]:
    means = np.nanmean(Y_train, axis=0)
    order = np.argsort(-means)
    return [int(i) for i in order if np.isfinite(means[i])][:k_top]


def run_pairwise_oof(
    X: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid_runs: Sequence[str],
    *,
    seed: int,
    n_folds: int,
    use_gbm: bool,
    k_top: int = 4,
) -> dict:
    mask = labeled_mask(fixed_vb, Y)
    X = X[mask]
    Y = Y[mask]
    fixed_vb = fixed_vb[mask]
    n, k = Y.shape
    folds = kfold_indices(n, n_folds, seed)
    oof_pick = np.full(n, -1, dtype=int)

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != fi])
        top_idx = top_config_indices(Y[train_idx], k_top=k_top)
        if len(top_idx) < 2:
            continue
        X_tr, X_te = X[train_idx], X[test_idx]
        X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)
        scores = np.zeros((len(test_idx), len(top_idx)), dtype=float)

        for a_pos, ia in enumerate(top_idx):
            for b_pos, ib in enumerate(top_idx):
                if ia == ib:
                    continue
                ya = Y[train_idx, ia]
                yb = Y[train_idx, ib]
                m = np.isfinite(ya) & np.isfinite(yb)
                if m.sum() < 15:
                    continue
                y_bin = (ya[m] > yb[m]).astype(float)
                if y_bin.sum() in (0, m.sum()):
                    continue
                if use_gbm and _HAS_SKLEARN:
                    clf = HistGradientBoostingClassifier(
                        max_depth=3, max_iter=80, learning_rate=0.08, random_state=seed,
                    )
                    clf.fit(X_tr_s[m], y_bin.astype(int))
                    p = clf.predict_proba(X_te_s)[:, 1]
                else:
                    w = logistic_fit(X_tr_s[m], y_bin, lam=0.5)
                    p = logistic_predict_proba(X_te_s, w)
                scores[:, a_pos] += p
                scores[:, b_pos] += 1.0 - p

        with np.errstate(all="ignore"):
            local_pick = np.argmax(scores, axis=1)
        oof_pick[test_idx] = [top_idx[int(li)] for li in local_pick]

    oracle_idx = np.nanargmax(Y, axis=1)
    valid = oof_pick >= 0
    policy = eval_config_pick_policy(oof_pick[valid], Y[valid], fixed_vb[valid], grid_runs)
    policy["oof_oracle_match_rate"] = float(np.mean(oof_pick[valid] == oracle_idx[valid]))
    policy["n_valid"] = int(valid.sum())
    return policy


def run_coarse_steps_lr(
    X: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid_runs: Sequence[str],
    *,
    seed: int,
    n_folds: int,
) -> dict:
    """Two-stage: bucket (S2/S5/S10/S20) then LR within bucket."""
    buckets = ["S2", "S5", "S10", "S20"]
    bucket_to_cols: Dict[str, List[int]] = {b: [] for b in buckets}
    for j, rid in enumerate(grid_runs):
        b = steps_bucket(rid)
        if b in bucket_to_cols:
            bucket_to_cols[b].append(j)

    mask = labeled_mask(fixed_vb, Y)
    X = X[mask]
    Y = Y[mask]
    fixed_vb = fixed_vb[mask]
    n = len(X)
    folds = kfold_indices(n, n_folds, seed)
    oof_pick = np.full(n, -1, dtype=int)
    lambdas = [1e-3, 1e-2, 1e-1, 1.0]

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != fi])
        X_tr, X_te = X[train_idx], X[test_idx]
        X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)

        # Stage 1: bucket scores = max predicted within bucket
        bucket_pred = np.full((len(test_idx), len(buckets)), np.nan, dtype=float)
        for bi, b in enumerate(buckets):
            cols = bucket_to_cols[b]
            if not cols:
                continue
            preds = []
            for j in cols:
                y_col = Y[train_idx, j]
                m = np.isfinite(y_col)
                if m.sum() < 10:
                    continue
                lam = select_ridge_lambda(X_tr_s[m], y_col[m], kfold_indices(int(m.sum()), 3, seed), lambdas)
                w = ridge_fit(X_tr_s[m], y_col[m], lam)
                preds.append(ridge_predict(X_te_s, w))
            if preds:
                bucket_pred[:, bi] = np.nanmax(np.column_stack(preds), axis=1)

        best_bucket = np.nanargmax(bucket_pred, axis=1)
        for ti, b_idx in enumerate(best_bucket):
            b = buckets[int(b_idx)]
            cols = bucket_to_cols[b]
            if not cols:
                continue
            preds = []
            for j in cols:
                y_col = Y[train_idx, j]
                m = np.isfinite(y_col)
                if m.sum() < 10:
                    continue
                w = ridge_fit(X_tr_s[m], y_col[m], 0.1)
                preds.append((j, ridge_predict(X_te_s[ti : ti + 1], w)[0]))
            if preds:
                oof_pick[test_idx[ti]] = max(preds, key=lambda x: x[1])[0]

    oracle_idx = np.nanargmax(Y, axis=1)
    policy = eval_config_pick_policy(oof_pick, Y, fixed_vb, grid_runs)
    policy["oof_oracle_match_rate"] = float(np.mean(oof_pick == oracle_idx))
    return policy


def run_mlp_shallow(
    X: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid_runs: Sequence[str],
    *,
    seed: int,
    n_folds: int,
    hidden: int = 16,
    steps: int = 400,
    lr: float = 0.05,
) -> dict:
    """One-hidden-layer softmax over configs (numpy only)."""
    mask = labeled_mask(fixed_vb, Y)
    X = X[mask]
    Y = Y[mask]
    fixed_vb = fixed_vb[mask]
    n, k = Y.shape
    folds = kfold_indices(n, n_folds, seed)
    oof_pick = np.full(n, -1, dtype=int)
    rng = np.random.default_rng(seed)

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != fi])
        X_tr, X_te = X[train_idx], X[test_idx]
        X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)
        d_in = X_tr_s.shape[1]
        W1 = rng.normal(0, 0.05, size=(d_in, hidden))
        b1 = np.zeros(hidden)
        W2 = rng.normal(0, 0.05, size=(hidden, k))
        b2 = np.zeros(k)

        # Targets: soft one-hot from normalized Y per row
        target = np.zeros((len(train_idx), k), dtype=float)
        for ri, row in enumerate(Y[train_idx]):
            m = np.isfinite(row)
            if not m.any():
                continue
            r = row.copy()
            r[~m] = -np.inf
            ex = np.exp(r - np.nanmax(r))
            ex[~m] = 0.0
            s = ex.sum()
            if s > 0:
                target[ri] = ex / s

        for _ in range(steps):
            h = np.tanh(X_tr_s @ W1 + b1)
            logits = h @ W2 + b2
            logits -= logits.max(axis=1, keepdims=True)
            prob = np.exp(logits)
            prob /= prob.sum(axis=1, keepdims=True)
            grad_logits = (prob - target) / len(train_idx)
            grad_W2 = h.T @ grad_logits
            grad_b2 = grad_logits.sum(axis=0)
            grad_h = grad_logits @ W2.T
            grad_pre = grad_h * (1.0 - np.tanh(X_tr_s @ W1 + b1) ** 2)
            grad_W1 = X_tr_s.T @ grad_pre
            grad_b1 = grad_pre.sum(axis=0)
            W2 -= lr * grad_W2
            b2 -= lr * grad_b2
            W1 -= lr * grad_W1
            b1 -= lr * grad_b1

        h_te = np.tanh(X_te_s @ W1 + b1)
        logits_te = h_te @ W2 + b2
        oof_pick[test_idx] = np.argmax(logits_te, axis=1)

    oracle_idx = np.nanargmax(Y, axis=1)
    policy = eval_config_pick_policy(oof_pick, Y, fixed_vb, grid_runs)
    policy["oof_oracle_match_rate"] = float(np.mean(oof_pick == oracle_idx))
    return policy


def policy_to_row(name: str, policy: dict, *, n_videos: int, extra: Optional[dict] = None) -> dict:
    cap = policy.get("fraction_oracle_captured")
    row = {
        "experiment": name,
        "n_videos": n_videos,
        "match_rate": policy.get("oof_oracle_match_rate"),
        "captured_pct": None if cap is None or (isinstance(cap, float) and math.isnan(cap)) else 100 * cap,
        "policy_vbench": policy.get("mean_policy_vbench"),
        "fixed_vbench": policy.get("mean_fixed_vbench"),
        "oracle_vbench": policy.get("mean_oracle_vbench"),
        "headroom": policy.get("oracle_headroom"),
        "policy_gain": (
            policy.get("mean_policy_vbench", 0) - policy.get("mean_fixed_vbench", 0)
            if policy.get("mean_policy_vbench") is not None
            else None
        ),
    }
    if extra:
        row.update(extra)
    return row


def run_experiment(
    name: str,
    bundle: dict,
    X_base: np.ndarray,
    feat_names: List[str],
    impute: dict,
    *,
    output_dir: Path,
    seed: int,
    n_folds: int,
    n_boot: int,
) -> dict:
    video_ids = bundle["video_ids"]
    grid = bundle["grid_runs"]
    Y = bundle["Y_total"]
    fixed_vb = bundle["fixed_vb"]

    if name == "baseline_linear_total":
        res = run_budget_config_task(
            video_ids=video_ids,
            X=X_base,
            Y=Y,
            fixed_vb=fixed_vb,
            notta_vb=np.full(len(video_ids), np.nan),
            grid_runs=grid,
            output_dir=output_dir / "_tmp_baseline",
            seed=seed,
            n_folds=n_folds,
        )
        policy = _policy_from_budget_task(res)
    elif name.startswith("dim_"):
        dim = name.replace("dim_", "")
        if dim == "vbench_total":
            Y_use = Y
        else:
            Y_use = bundle["Y_dim"].get(dim)
            if Y_use is None:
                raise ValueError(f"unknown dim {dim}")
        res = run_budget_config_task(
            video_ids=video_ids,
            X=X_base,
            Y=Y_use,
            fixed_vb=fixed_vb,
            notta_vb=np.full(len(video_ids), np.nan),
            grid_runs=grid,
            output_dir=output_dir / f"_tmp_{name}",
            seed=seed,
            n_folds=n_folds,
        )
        policy = _policy_from_budget_task(res)
    elif name == "coarse_steps_lr":
        policy = run_coarse_steps_lr(X_base, Y, fixed_vb, grid, seed=seed, n_folds=n_folds)
    elif name == "probe_simulated":
        X_probe, _ = build_probe_features(bundle, video_ids, X_base, feat_names)
        res = run_budget_config_task(
            video_ids=video_ids,
            X=X_probe,
            Y=Y,
            fixed_vb=fixed_vb,
            notta_vb=np.full(len(video_ids), np.nan),
            grid_runs=grid,
            output_dir=output_dir / "_tmp_probe",
            seed=seed,
            n_folds=n_folds,
        )
        policy = _policy_from_budget_task(res)
    elif name == "proxy_psnr_all":
        policy = run_proxy_pick(Y, bundle["psnr"], fixed_vb, grid)
    elif name == "proxy_bestof3_psnr":
        cols = [grid.index(r) for r in BESTOF3_RUNS if r in grid]
        policy = run_proxy_pick(Y, bundle["psnr"], fixed_vb, grid, allowed_cols=cols)
    elif name == "pairwise_logistic_top4":
        policy = run_pairwise_oof(
            X_base, Y, fixed_vb, grid, seed=seed, n_folds=n_folds, use_gbm=False,
        )
    elif name == "pairwise_gbm_top4":
        if not _HAS_SKLEARN:
            skipped = {"experiment": name, "skipped": True, "reason": "sklearn not installed"}
            _write_experiment_json(output_dir, name, skipped)
            return skipped
        policy = run_pairwise_oof(
            X_base, Y, fixed_vb, grid, seed=seed, n_folds=n_folds, use_gbm=True,
        )
    elif name == "composite_psnr_ridge":
        X_psnr, _ = build_psnr_feature_block(bundle, X_base, feat_names)
        res = run_budget_config_task(
            video_ids=video_ids,
            X=X_psnr,
            Y=Y,
            fixed_vb=fixed_vb,
            notta_vb=np.full(len(video_ids), np.nan),
            grid_runs=grid,
            output_dir=output_dir / "_tmp_composite",
            seed=seed,
            n_folds=n_folds,
        )
        policy = _policy_from_budget_task(res)
    elif name == "mlp_shallow":
        policy = run_mlp_shallow(X_base, Y, fixed_vb, grid, seed=seed, n_folds=n_folds)
    else:
        raise ValueError(f"unknown experiment {name}")

    if policy.get("skipped"):
        return policy

    mask = labeled_mask(fixed_vb, Y)
    # Reconstruct OOF policy gains for bootstrap from policy means only (approx) —
    # for rigorous CI, experiments using run_budget_config_task save OOF csv.
    oof_csv = output_dir / f"{name}_oof.csv"
    if name == "baseline_linear_total":
        oof_csv = output_dir / "_tmp_baseline/budget_config_oof_predictions.csv"
    elif name in ("probe_simulated", "composite_psnr_ridge") or name.startswith("dim_"):
        pass

    cap = policy.get("fraction_oracle_captured")
    row = policy_to_row(name, policy, n_videos=int(mask.sum()))
    row["bootstrap_note"] = "point estimate; see aggregate script for pooled bootstrap"

    out_json = output_dir / f"{name}.json"
    _write_experiment_json(output_dir, name, {"policy": policy, "row": row})
    return row


def write_summary(rows: List[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "routing_experiments_summary.csv"
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lines = [
        "# Budget routing experiment suite (pilot N=200)",
        "",
        "No new videos — all methods use existing 12-config pilot VBench + PSNR/SSIM.",
        "",
        "| Experiment | N | Match % | Captured % | Policy gain | Headroom |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in sorted(rows, key=lambda x: -(x.get("captured_pct") or -1)):
        if r.get("skipped"):
            lines.append(f"| `{r['experiment']}` | — | — | — | skipped: {r.get('reason')} | — |")
            continue
        lines.append(
            f"| `{r['experiment']}` | {r.get('n_videos')} | "
            f"{100 * _num(r.get('match_rate')):.1f} | {_num(r.get('captured_pct')):.1f} | "
            f"{_num(r.get('policy_gain')):+.4f} | {_num(r.get('headroom')):.4f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- **proxy_psnr_all** / **proxy_bestof3_psnr** use PSNR as deploy-time proxy (no VBench at inference).",
        "- **probe_simulated** uses probe PSNR/SSIM from S2/S10/S20 configs already in the sweep.",
        "- For stable GBM/MLP claims, scale to **N≈999–2400** labeled videos (999v × 12 configs).",
        "",
    ]
    (output_dir / "routing_experiments_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Budget routing experiment suite")
    ap.add_argument("--series-root", type=Path, default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot")
    ap.add_argument(
        "--feature-date",
        type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-06",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--experiment", type=str, default=None, choices=ALL_EXPERIMENTS)
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=5000)
    args = ap.parse_args()

    if not args.run_all and not args.experiment:
        ap.error("Specify --experiment NAME or --run-all")

    out = args.output_dir
    if out is None:
        out = _REPO / "sweep_experiment/reports/per_video_analysis/2026-07-05/budget_routing_experiments"
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_pilot_bundle(args.series_root, args.feature_date)
    vids = bundle["video_ids"]
    impute = compute_impute(vids, bundle["features"], bundle["feat_names"])
    X_base = build_feature_matrix(vids, bundle["features"], bundle["feat_names"], impute=impute)

    exps = list(ALL_EXPERIMENTS) if args.run_all else [args.experiment]
    rows: List[dict] = []
    for name in exps:
        print(f"[run] {name} ...", file=sys.stderr)
        try:
            row = run_experiment(
                name,
                bundle,
                X_base,
                bundle["feat_names"],
                impute,
                output_dir=out,
                seed=args.seed,
                n_folds=args.n_folds,
                n_boot=args.n_boot,
            )
            rows.append(row)
            if not row.get("skipped"):
                print(
                    f"  captured={_num(row.get('captured_pct')):.1f}%  "
                    f"match={100 * _num(row.get('match_rate')):.1f}%",
                    file=sys.stderr,
                )
            else:
                print(f"  skipped: {row.get('reason')}", file=sys.stderr)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed = {"experiment": name, "skipped": True, "reason": str(e)}
            rows.append(failed)
            _write_experiment_json(out, name, failed)

    if args.run_all:
        write_summary(rows, out)
        print(f"Wrote {out}/routing_experiments_summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
