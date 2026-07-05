#!/usr/bin/env python3
"""Per-video VBench++ gain prediction experiments @ pilot N=200.

Phase 1 (CPU, existing metrics):
  exp6_knn_oracle_transfer   — kNN on Phase-0 embed → neighbor oracle config vote
  exp7_gain_predictor_probe  — ridge argmax predicted ΔVBench with probe features
  exp8_abstain_route_3way    — headroom gate → 3-way route else fixed S10
  exp9_multitask_aestech     — DOVER-weighted Aes+IQ target → route; eval total

Phase 2 (CPU, richer signals from disk):
  exp10_dover_aestech_proxy  — pick probe by 0.428·Aes+0.572·IQ on S2/S10 outputs
  exp11_tier3_probe_ridge    — Phase-0 + tier3 + probe → 3-way ridge
  exp12_trajectory_ridge     — delta_norm/final_loss from probe configs → route

Usage:
  python3 scripts/run_vbench_gain_prediction_experiments.py --run-all
  python3 scripts/run_vbench_gain_prediction_experiments.py --experiment exp6_knn_oracle_transfer
  python3 scripts/run_vbench_gain_prediction_experiments.py --aggregate-only
"""
from __future__ import annotations

import argparse
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

from scripts.analyze_adasteer_budget_oracle import discover_runs  # noqa: E402
from scripts.analyze_delta_norm_correlation import load_per_video_records  # noqa: E402
from scripts.analyze_per_video_vbench_agreement import VBENCH_DIMS  # noqa: E402
from scripts.budget_routing_common import (  # noqa: E402
    BESTOF3_RUNS,
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
DOVER_AES_W, DOVER_IQ_W = 0.428, 0.572
HEADROOM_EPS = 0.05

ALL_EXPERIMENTS = (
    "exp6_knn_oracle_transfer",
    "exp7_gain_predictor_probe",
    "exp8_abstain_route_3way",
    "exp9_multitask_aestech",
    "exp10_dover_aestech_proxy",
    "exp11_tier3_probe_ridge",
    "exp12_trajectory_ridge",
)


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
    if extra:
        row.update(extra)
    return row


def _write_json(out_dir: Path, name: str, payload: dict) -> None:
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _config_indices(grid: Sequence[str], names: Sequence[str]) -> List[int]:
    return [grid.index(r) for r in names if r in grid]


def _oof_policy_from_picks(
    picks: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    mask: np.ndarray,
) -> dict:
    valid = mask & (picks >= 0)
    pol = eval_config_pick_policy(picks[valid], Y[valid], fixed_vb[valid], grid)
    oracle_idx = np.nanargmax(Y[valid], axis=1)
    pol["oof_oracle_match_rate"] = float(np.mean(picks[valid] == oracle_idx))
    pol["n_valid"] = int(valid.sum())
    return pol


def load_trajectory_matrix(
    series_root: Path,
    grid_runs: Sequence[str],
    video_ids: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Per-video delta_norm, final_loss, mean grad norm from chunk summaries."""
    runs = discover_runs(series_root)
    n, k = len(video_ids), len(grid_runs)
    vid_idx = {v: i for i, v in enumerate(video_ids)}
    out = {
        "delta_norm": np.full((n, k), np.nan),
        "final_loss": np.full((n, k), np.nan),
        "grad_norm_mean": np.full((n, k), np.nan),
    }
    for rid in grid_runs:
        if rid not in runs:
            continue
        j = grid_runs.index(rid)
        recs = load_per_video_records(runs[rid])
        for vid, r in recs.items():
            if vid not in vid_idx:
                continue
            i = vid_idx[vid]
            if r.get("delta_norm") is not None:
                out["delta_norm"][i, j] = float(r["delta_norm"])
            if r.get("final_loss") is not None:
                out["final_loss"][i, j] = float(r["final_loss"])
            gns = r.get("raw_grad_norms") or []
            if gns:
                out["grad_norm_mean"][i, j] = float(np.mean(gns))
    return out


def build_trajectory_probe_features(
    bundle: dict,
    traj: Dict[str, np.ndarray],
    base_X: np.ndarray,
    feat_names: List[str],
    *,
    probe_runs: Sequence[str] = PROBE2,
) -> Tuple[np.ndarray, List[str]]:
    grid = bundle["grid_runs"]
    names = list(feat_names)
    cols: List[np.ndarray] = []
    fixed_j = grid.index(bundle["fixed_run"])
    for rid in probe_runs:
        if rid not in grid:
            continue
        j = grid.index(rid)
        for key in ("delta_norm", "final_loss", "grad_norm_mean"):
            v = traj[key][:, j]
            cols.append(v)
            cols.append(v - traj[key][:, fixed_j])
            names.extend([f"{rid}_{key}", f"{rid}_d{key}"])
    if cols:
        return np.column_stack([base_X] + cols), names
    return base_X, names


def run_exp6_knn(
    X: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    k: int,
    seed: int,
    n_folds: int,
) -> dict:
    mask = labeled_mask(fixed_vb, Y)
    X = X[mask]
    Y = Y[mask]
    fixed_vb = fixed_vb[mask]
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

    pol = _oof_policy_from_picks(oof_pick, Y, fixed_vb, grid, np.ones(n, dtype=bool))
    return _row("exp6_knn_oracle_transfer", pol, n=n, extra={"k": k})


def run_exp7_gain_probe(
    bundle: dict,
    X_probe: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    seed: int,
    n_folds: int,
) -> dict:
    """Ridge on ΔVBench (gain vs fixed) per config; deploy argmax predicted gain."""
    Y_gain = Y - fixed_vb[:, np.newaxis]
    with tempfile.TemporaryDirectory() as tmp:
        res = run_budget_config_task(
            video_ids=bundle["video_ids"],
            X=X_probe,
            Y=Y_gain,
            fixed_vb=np.zeros(len(fixed_vb)),
            notta_vb=np.full(len(fixed_vb), np.nan),
            grid_runs=grid,
            output_dir=Path(tmp),
            seed=seed,
            n_folds=n_folds,
        )
    pol = _policy_from_budget_task(res)
    mask = labeled_mask(fixed_vb, Y)
    n = int(mask.sum())
    return _row("exp7_gain_predictor_probe", pol, n=n, extra={"target": "delta_vbench_total"})


def run_exp8_abstain(
    X_probe: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    seed: int,
    n_folds: int,
    eps: float = HEADROOM_EPS,
) -> dict:
    mask = labeled_mask(fixed_vb, Y)
    X = X_probe[mask]
    Y = Y[mask]
    fixed_vb = fixed_vb[mask]
    n = len(X)
    headroom = np.nanmax(Y, axis=1) - fixed_vb
    y_bin = (headroom > eps).astype(float)
    full_js = _config_indices(grid, FULL3)
    if len(full_js) < 2:
        raise ValueError("need FULL3 configs in grid")
    sub_names = [grid[j] for j in full_js]
    Y_sub = Y[:, full_js]
    fixed_j = grid.index("S10_LR5e3") if "S10_LR5e3" in grid else full_js[1]

    folds = kfold_indices(n, n_folds, seed)
    oof_pick = np.full(n, fixed_j, dtype=int)

    for fi, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != fi])
        X_tr, X_te = X[train_idx], X[test_idx]
        X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)
        yb = y_bin[train_idx]
        if yb.sum() >= 10 and 0 < yb.sum() < len(yb):
            w = logistic_fit(X_tr_s, yb, lam=0.5)
            p_adapt = logistic_predict_proba(X_te_s, w)
        else:
            p_adapt = np.full(len(test_idx), float(yb.mean()))

        adapt_mask = p_adapt > 0.5
        if adapt_mask.sum() == 0:
            continue
        # Route adapters among 3 configs via ridge on Y_sub
        pred_te = np.full((len(test_idx), len(full_js)), np.nan)
        for cj, col in enumerate(full_js):
            y_col = Y[train_idx, col]
            m = np.isfinite(y_col)
            if m.sum() < 10:
                continue
            wf = ridge_fit(X_tr_s[m], y_col[m], 0.1)
            pred_te[:, cj] = ridge_predict(X_te_s, wf)
        for ti, global_i in enumerate(test_idx):
            if not adapt_mask[ti]:
                continue
            oof_pick[global_i] = full_js[int(np.nanargmax(pred_te[ti]))]

    pol = _oof_policy_from_picks(oof_pick, Y, fixed_vb, grid, np.ones(n, dtype=bool))
    return _row(
        "exp8_abstain_route_3way",
        pol,
        n=n,
        extra={"headroom_eps": eps, "apply_rate_est": float(np.mean(oof_pick != fixed_j))},
    )


def _oof_picks_from_budget_csv(csv_path: Path, grid: Sequence[str]) -> Dict[str, int]:
    """Map video_id → config index from budget_config OOF CSV."""
    import csv as csv_mod

    out: Dict[str, int] = {}
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            rid = row.get("picked_run", "")
            if rid and rid in grid:
                out[row["video_id"]] = grid.index(rid)
    return out


def run_exp9_aestech(
    bundle: dict,
    X: np.ndarray,
    Y_total: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    seed: int,
    n_folds: int,
) -> dict:
    aes = bundle["Y_dim"]["aesthetic_quality"]
    iq = bundle["Y_dim"]["imaging_quality"]
    Y_proxy = DOVER_AES_W * aes + DOVER_IQ_W * iq
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        res = run_budget_config_task(
            video_ids=bundle["video_ids"],
            X=X,
            Y=Y_proxy,
            fixed_vb=fixed_vb,
            notta_vb=np.full(len(fixed_vb), np.nan),
            grid_runs=grid,
            output_dir=tmp_path,
            seed=seed,
            n_folds=n_folds,
        )
        vid_to_pick = _oof_picks_from_budget_csv(
            tmp_path / "budget_config_oof_predictions.csv", grid,
        )
    n_vids = len(bundle["video_ids"])
    picks = np.full(n_vids, -1, dtype=int)
    for i, vid in enumerate(bundle["video_ids"]):
        if vid in vid_to_pick:
            picks[i] = vid_to_pick[vid]
    mask = labeled_mask(fixed_vb, Y_total)
    pol_total = _oof_policy_from_picks(picks, Y_total, fixed_vb, grid, mask)
    pol_proxy = _policy_from_budget_task(res)
    return _row(
        "exp9_multitask_aestech",
        pol_total,
        n=int(mask.sum()),
        extra={
            "proxy_note": "train 0.428·Aes+0.572·IQ OOF; eval VBench total",
            "captured_pct_proxy_oof": 100 * pol_proxy.get("fraction_oracle_captured", 0),
        },
    )


def run_exp10_dover_proxy(
    bundle: dict,
    Y_total: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
) -> dict:
    aes = bundle["Y_dim"]["aesthetic_quality"]
    iq = bundle["Y_dim"]["imaging_quality"]
    fused = DOVER_AES_W * aes + DOVER_IQ_W * iq
    probe_js = _config_indices(grid, PROBE2)
    full_map = {"S2_LR5e3": "S5_LR5e3", "S10_LR5e3": "S10_LR5e3"}
    n = len(fixed_vb)
    picks = np.full(n, -1, dtype=int)
    for i in range(n):
        scores = [(j, fused[i, j]) for j in probe_js if np.isfinite(fused[i, j])]
        if not scores:
            continue
        best_probe_j = max(scores, key=lambda x: x[1])[0]
        rid = grid[best_probe_j]
        target = full_map.get(rid, rid)
        if target in grid:
            picks[i] = grid.index(target)
    mask = labeled_mask(fixed_vb, Y_total)
    pol = _oof_policy_from_picks(picks, Y_total, fixed_vb, grid, mask)
    return _row(
        "exp10_dover_aestech_proxy",
        pol,
        n=int(mask.sum()),
        extra={"note": "offline sim: Aes+IQ on probe outputs (deploy needs DOVER on frames)"},
    )


def run_exp11_tier3_probe(
    bundle: dict,
    X_probe: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    seed: int,
    n_folds: int,
) -> dict:
    full_js = _config_indices(grid, FULL3)
    sub_names = [grid[j] for j in full_js]
    Y_sub = Y[:, full_js]
    with tempfile.TemporaryDirectory() as tmp:
        res = run_budget_config_task(
            video_ids=bundle["video_ids"],
            X=X_probe,
            Y=Y_sub,
            fixed_vb=fixed_vb,
            notta_vb=np.full(len(fixed_vb), np.nan),
            grid_runs=sub_names,
            output_dir=Path(tmp),
            seed=seed,
            n_folds=n_folds,
        )
    pol = _policy_from_budget_task(res)
    n = int(labeled_mask(fixed_vb, Y).sum())
    return _row("exp11_tier3_probe_ridge", pol, n=n)


def run_exp12_trajectory(
    bundle: dict,
    traj: Dict[str, np.ndarray],
    base_X: np.ndarray,
    feat_names: List[str],
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    seed: int,
    n_folds: int,
) -> dict:
    X_traj, _ = build_trajectory_probe_features(bundle, traj, base_X, feat_names, probe_runs=PROBE2)
    full_js = _config_indices(grid, FULL3)
    sub_names = [grid[j] for j in full_js]
    Y_sub = Y[:, full_js]
    with tempfile.TemporaryDirectory() as tmp:
        res = run_budget_config_task(
            video_ids=bundle["video_ids"],
            X=X_traj,
            Y=Y_sub,
            fixed_vb=fixed_vb,
            notta_vb=np.full(len(fixed_vb), np.nan),
            grid_runs=sub_names,
            output_dir=Path(tmp),
            seed=seed,
            n_folds=n_folds,
        )
    pol = _policy_from_budget_task(res)
    n = int(labeled_mask(fixed_vb, Y).sum())
    return _row("exp12_trajectory_ridge", pol, n=n)


def tail_captured(
    picks: np.ndarray,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    mask: np.ndarray,
    *,
    tail_frac: float = 0.2,
) -> Optional[float]:
    headroom = np.nanmax(Y[mask], axis=1) - fixed_vb[mask]
    pos = headroom[headroom > 1e-9]
    if len(pos) < 5:
        return None
    thr = float(np.quantile(pos, 1.0 - tail_frac))
    tail = headroom >= thr
    if tail.sum() < 3:
        return None
    idx = np.where(mask)[0][tail]
    pv = np.array([Y[i, picks[i]] for i in idx if picks[i] >= 0])
    fv = fixed_vb[idx[: len(pv)]]
    ov = np.array([np.nanmax(Y[i]) for i in idx[: len(pv)]])
    if len(pv) < 3:
        return None
    cap, _, _, _, _ = bootstrap_captured(pv, ov, fv, n_boot=2000, seed=42)
    return 100 * cap


def load_results(out_dir: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for name in ALL_EXPERIMENTS:
        p = out_dir / f"{name}.json"
        if p.is_file():
            out[name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def write_summary(results: Dict[str, dict], out_dir: Path) -> None:
    lines = [
        "# VBench++ gain prediction experiments @ N=200",
        "",
        "Goal: maximize fraction of **VBench total oracle** captured vs fixed S10.",
        "",
        "| Experiment | Captured % | Match % | Notes |",
        "|---|---:|---:|---|",
    ]
    rows = []
    for name in ALL_EXPERIMENTS:
        data = results.get(name, {})
        if data.get("skipped"):
            lines.append(f"| `{name}` | — | — | FAILED: {data.get('reason', '?')} |")
            continue
        row = data.get("row") or data
        cap = row.get("captured_pct")
        cap_s = f"{cap:.1f}" if cap is not None else "—"
        mr = row.get("match_rate")
        mr_s = f"{100 * mr:.1f}" if mr is not None else "—"
        note = row.get("note") or row.get("proxy_note") or row.get("extra", "")
        if isinstance(note, dict):
            note = str(note)
        lines.append(f"| `{name}` | {cap_s} | {mr_s} | {note} |")
        if cap is not None:
            rows.append((cap, name))

    lines += [
        "",
        "## Reference",
        "",
        "- Oracle headroom (mean): **+0.140** VBench total",
        "- Linear Phase-0 router: **~9%** captured",
        "- Best prior (probe ridge 3-way): **~12%**",
        "- Success bar: **>25%** captured with bootstrap CI excluding 0",
        "",
    ]
    if rows:
        best = max(rows, key=lambda x: x[0])
        lines.append(f"**Best this suite:** `{best[1]}` at **{best[0]:.1f}%** captured.")
        lines.append("")
    (out_dir / "vbench_gain_prediction_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-root", type=Path, default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot")
    ap.add_argument(
        "--feature-date",
        type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-06",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--experiment", choices=ALL_EXPERIMENTS, default=None)
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--knn-k", type=int, default=7)
    args = ap.parse_args()

    out = args.output_dir or (
        _REPO / "sweep_experiment/reports/per_video_analysis/2026-07-05/vbench_gain_prediction_experiments"
    )
    out.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        write_summary(load_results(out), out)
        print(f"Aggregated → {out}/vbench_gain_prediction_summary.md", file=sys.stderr)
        return 0

    if not args.run_all and not args.experiment:
        ap.error("Use --run-all, --experiment, or --aggregate-only")

    exps = list(ALL_EXPERIMENTS) if args.run_all else [args.experiment]
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
    traj = load_trajectory_matrix(args.series_root, grid, bundle["video_ids"])

    for name in exps:
        print(f"[run] {name} ...", file=sys.stderr)
        try:
            if name == "exp6_knn_oracle_transfer":
                row = run_exp6_knn(
                    X_base, Y, fixed_vb, grid, k=args.knn_k, seed=args.seed, n_folds=args.n_folds,
                )
            elif name == "exp7_gain_predictor_probe":
                row = run_exp7_gain_probe(
                    bundle, X_probe, Y, fixed_vb, grid, seed=args.seed, n_folds=args.n_folds,
                )
            elif name == "exp8_abstain_route_3way":
                row = run_exp8_abstain(
                    X_probe, Y, fixed_vb, grid, seed=args.seed, n_folds=args.n_folds,
                )
            elif name == "exp9_multitask_aestech":
                row = run_exp9_aestech(
                    bundle, X_base, Y, fixed_vb, grid, seed=args.seed, n_folds=args.n_folds,
                )
            elif name == "exp10_dover_aestech_proxy":
                row = run_exp10_dover_proxy(bundle, Y, fixed_vb, grid)
            elif name == "exp11_tier3_probe_ridge":
                row = run_exp11_tier3_probe(
                    bundle, X_probe, Y, fixed_vb, grid, seed=args.seed, n_folds=args.n_folds,
                )
            elif name == "exp12_trajectory_ridge":
                row = run_exp12_trajectory(
                    bundle, traj, X_base, bundle["feat_names"], Y, fixed_vb, grid,
                    seed=args.seed, n_folds=args.n_folds,
                )
            else:
                raise ValueError(name)
            payload = {"row": row, "policy_note": "eval on VBench total vs fixed S10_LR5e3"}
            _write_json(out, name, payload)
            print(f"  captured={row.get('captured_pct', 0):.1f}%", file=sys.stderr)
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
            _write_json(out, name, {"experiment": name, "skipped": True, "reason": str(exc)})

    if args.run_all:
        write_summary(load_results(out), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
