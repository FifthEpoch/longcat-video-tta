#!/usr/bin/env python3
"""Five deployable-routing tricks on the budget grid (offline, no new generation).

Motivation: at 1000v, absolute VBench config-routing is un-routable (see
`paper_tables/2026-07-21_router_1000v_feature_model_suite.md`). These 5 tricks
change the *action space* and *target* instead of the input dimension:

  1. skip_augmented  — add NO-TTA as a 13th action; argmax over {12 configs, NOTTA}
  2. route_for_metric — plain per-config argmax on the chosen metric (baseline)
  3. gain_target     — predict per-config gain (metric − NOTTA); skip if best gain ≤ 0
  4. adapt_gate      — binary "will TTA help?" logistic gate → config pick or NOTTA
  5. probe_route     — features augmented with cheap S2/S10/S20 probe PSNR/SSIM + Δ

All use 5-fold OOF; oracle upper bounds are reported per trick. Metric is
higher-is-better (PSNR dB or VBench raw-total = unweighted mean of 7 raw dims,
NOT normalized VBench++). PSNR runs immediately; VBench needs NOTTA VBench
backfill present under the series.

Usage (PSNR, runnable now):
    python3 scripts/run_routing_tricks.py --metric psnr \
      --series-root sweep_experiment/results/panda_ood_budget_1000v_preview \
      --feature-date sweep_experiment/reports/per_video_analysis/2026-07-12 \
      --ood-csv sweep_experiment/reports/2026-07-10/diffusion_ood_scores_segment_pool.csv \
      --output-dir sweep_experiment/reports/per_video_analysis/2026-07-21/routing_tricks_psnr_1000v

Same with `--metric vbench` after the NOTTA VBench backfill finishes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
    kfold_indices,
    logistic_fit,
    logistic_predict_proba,
    ridge_fit,
    ridge_predict,
    select_ridge_lambda,
    standardize_train_test,
)
from scripts.budget_routing_common import PROBE_RUNS, load_pilot_bundle  # noqa: E402
from scripts.run_budget_routing_experiments import build_probe_features  # noqa: E402
from scripts.analyze_adasteer_budget_vbench_oracle import (  # noqa: E402
    NOTTA_RUN_ID,
    discover_runs,
    vbench_total_score,
)
from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    load_per_video_vbench,
)
from scripts.analyze_per_video_tta_gain import load_per_video_metrics  # noqa: E402
from scripts.analyze_adasteer_budget_oracle import load_ood_quintiles  # noqa: E402
from scripts.analyze_router_auc import binary_auc  # noqa: E402

_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _nanmean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    if a.size == 0 or np.all(np.isnan(a)):
        return float("nan")
    return float(np.nanmean(a))


def _fmt(x: Optional[float], nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _fmt_d(x: Optional[float], nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}"


def _pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.1f}%"


def _oof_config_predict(
    X: np.ndarray, Y: np.ndarray, n_folds: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Per-config OOF ridge; returns (oof_pick, oof_scores n×k, lambda)."""
    n, k = Y.shape
    folds = kfold_indices(n, n_folds, seed)
    flat_y = Y.T.reshape(-1)
    fm = ~np.isnan(flat_y)
    best_lam = _LAMBDAS[0]
    if fm.sum() >= 30:
        flat_X = np.repeat(X, k, axis=0)[fm]
        fy = flat_y[fm]
        ff = kfold_indices(len(fy), min(n_folds, 5), seed)
        best_lam = select_ridge_lambda(flat_X, fy, ff, list(_LAMBDAS))

    oof_pick = np.full(n, -1, dtype=int)
    oof_scores = np.full((n, k), np.nan, dtype=float)
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        X_tr, X_te = X[train_idx], X[test_idx]
        X_tr_s, X_te_s, _, _ = standardize_train_test(X_tr, X_te)
        pred = np.full((len(test_idx), k), np.nan, dtype=float)
        for j in range(k):
            yc = Y[train_idx, j]
            m = ~np.isnan(yc)
            if m.sum() < 10:
                continue
            w = ridge_fit(X_tr_s[m], yc[m], best_lam)
            pred[:, j] = ridge_predict(X_te_s, w)
        oof_scores[test_idx] = pred
        for r, ti in enumerate(test_idx):
            row = pred[r]
            if np.all(np.isnan(row)):
                continue
            oof_pick[ti] = int(np.nanargmax(row))
    return oof_pick, oof_scores, best_lam


def _oof_logistic(X: np.ndarray, z: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """OOF logistic prob of z=1 from features."""
    n = len(z)
    folds = kfold_indices(n, n_folds, seed)
    oof_p = np.full(n, np.nan, dtype=float)
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        z_tr = z[train_idx]
        if len(np.unique(z_tr)) < 2:
            oof_p[test_idx] = float(np.mean(z_tr))
            continue
        X_tr_s, X_te_s, _, _ = standardize_train_test(X[train_idx], X[test_idx])
        w = logistic_fit(X_tr_s, z_tr, lam=0.01)
        oof_p[test_idx] = logistic_predict_proba(X_te_s, w)
    return oof_p


def _realized_from_pick(Y: np.ndarray, pick: np.ndarray) -> np.ndarray:
    n, k = Y.shape
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        j = int(pick[i])
        if 0 <= j < k and not math.isnan(Y[i, j]):
            out[i] = Y[i, j]
    return out


def _summ(
    realized: np.ndarray,
    fixed: np.ndarray,
    notta: np.ndarray,
    oracle: np.ndarray,
) -> dict:
    pol = _nanmean(realized)
    fx = _nanmean(fixed)
    nt = _nanmean(notta)
    orc = _nanmean(oracle)
    head = orc - fx if not (math.isnan(orc) or math.isnan(fx)) else float("nan")
    cap = (pol - fx) / head if abs(head) > 1e-9 else float("nan")
    return {
        "policy": pol,
        "fixed": fx,
        "notta": nt,
        "oracle": orc,
        "d_vs_fixed": pol - fx if not (math.isnan(pol) or math.isnan(fx)) else float("nan"),
        "d_vs_notta": pol - nt if not (math.isnan(pol) or math.isnan(nt)) else float("nan"),
        "oracle_head_vs_fixed": head,
        "captured": cap,
    }


def _side_effect_mean(
    other: Optional[np.ndarray], pick: np.ndarray, is_config: np.ndarray
) -> Optional[float]:
    """Mean of the *other* metric over videos where a config (not NOTTA) was picked."""
    if other is None:
        return None
    n, k = other.shape
    vals: List[float] = []
    for i in range(n):
        if not is_config[i]:
            continue
        j = int(pick[i])
        if 0 <= j < k and not math.isnan(other[i, j]):
            vals.append(other[i, j])
    return float(np.mean(vals)) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Five routing tricks (offline)")
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
    ap.add_argument("--metric", choices=("psnr", "vbench"), default="psnr")
    ap.add_argument("--ood-csv", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    metric = args.metric
    unit = "dB" if metric == "psnr" else "raw-total"
    require_vbench = metric == "vbench"
    bundle = load_pilot_bundle(
        args.series_root, args.feature_date, require_vbench=require_vbench
    )
    vids: List[str] = bundle["video_ids"]
    grid_runs: List[str] = bundle["grid_runs"]
    fixed_run: str = bundle["fixed_run"]
    n_full = len(vids)

    # ---- metric matrix + fixed column ------------------------------------
    if metric == "psnr":
        Y = np.array(bundle["psnr"], dtype=float)
        if fixed_run not in grid_runs:
            print(f"[error] fixed {fixed_run} not in grid", file=sys.stderr)
            return 2
        fixed_vb = Y[:, grid_runs.index(fixed_run)].copy()
        other = np.array(bundle["Y_total"], dtype=float)  # VBench side-effect
        other_name = "VBench-total"
    else:
        Y = np.array(bundle["Y_total"], dtype=float)
        fixed_vb = np.array(bundle["fixed_vb"], dtype=float)
        other = np.array(bundle["psnr"], dtype=float)  # PSNR side-effect
        other_name = "PSNR"
    if np.all(np.isnan(other)):
        other = None

    # ---- NOTTA per-video metric ------------------------------------------
    runs = discover_runs(args.series_root)
    notta_dir = runs.get(NOTTA_RUN_ID)
    notta = np.full(n_full, np.nan, dtype=float)
    if notta_dir is not None:
        if metric == "psnr":
            per = load_per_video_metrics(notta_dir)
            for i, v in enumerate(vids):
                val = per.get(v, {}).get("psnr")
                if val is not None:
                    notta[i] = float(val)
        else:
            vb = load_per_video_vbench(notta_dir)
            for i, v in enumerate(vids):
                tot = vbench_total_score(vb.get(v, {}), list(VBENCH_DIMS))
                if tot is not None:
                    notta[i] = float(tot)
    n_notta = int(np.sum(~np.isnan(notta)))
    print(f"[info] metric={metric} N={n_full} grid={len(grid_runs)} NOTTA-scored={n_notta}",
          file=sys.stderr)

    # ---- features ---------------------------------------------------------
    impute = compute_impute(vids, bundle["features"], bundle["feat_names"])
    X_all = build_feature_matrix(vids, bundle["features"], bundle["feat_names"], impute=impute)
    X_probe, _ = build_probe_features(bundle, vids, X_all, bundle["feat_names"])

    # ---- OOD quintiles ----------------------------------------------------
    ood_q: Dict[str, int] = {}
    if args.ood_csv and args.ood_csv.is_file():
        ood_q = load_ood_quintiles(args.ood_csv)

    # ---- common mask: needs fixed, ≥1 config, and NOTTA score -------------
    mask = (
        ~np.isnan(fixed_vb)
        & ~np.all(np.isnan(Y), axis=1)
        & ~np.isnan(notta)
    )
    n = int(mask.sum())
    if n < 30:
        print(
            f"[error] only {n} videos with fixed+config+NOTTA {metric}. "
            + ("Run the NOTTA VBench backfill first." if metric == "vbench" else ""),
            file=sys.stderr,
        )
        return 2

    Xm = X_all[mask]
    Xpm = X_probe[mask]
    Ym = Y[mask]
    fixedm = fixed_vb[mask]
    nottam = notta[mask]
    otherm = other[mask] if other is not None else None
    vids_m = [vids[i] for i in range(n_full) if mask[i]]
    q_m = np.array([ood_q.get(v, -1) for v in vids_m], dtype=int)
    config_oracle = np.nanmax(Ym, axis=1)
    aug_oracle = np.maximum(config_oracle, nottam)
    arange = np.arange(n)

    results: List[dict] = []

    # ===== Trick 2: route_for_metric (plain per-config argmax) =============
    pick2, scores2, lam2 = _oof_config_predict(Xm, Ym, args.n_folds, args.seed)
    real2 = _realized_from_pick(Ym, pick2)
    s2 = _summ(real2, fixedm, nottam, config_oracle)
    s2.update(
        trick="route_for_metric",
        desc="per-config argmax (baseline)",
        lam=lam2,
        apply_rate=1.0,
        match=float(np.mean(pick2 == np.nanargmax(Ym, axis=1))),
        side_effect=_side_effect_mean(otherm, pick2, np.ones(n, bool)),
    )
    results.append(s2)

    # ===== Trick 1: skip_augmented (configs + NOTTA as 13th action) ========
    Y_aug = np.column_stack([Ym, nottam])
    pick1, scores1, lam1 = _oof_config_predict(Xm, Y_aug, args.n_folds, args.seed)
    k = Ym.shape[1]
    real1 = _realized_from_pick(Y_aug, pick1)
    picked_notta1 = pick1 == k
    s1 = _summ(real1, fixedm, nottam, aug_oracle)
    s1.update(
        trick="skip_augmented",
        desc="argmax over {12 configs, NOTTA}",
        lam=lam1,
        apply_rate=float(np.mean(~picked_notta1)),
        skip_rate=float(np.mean(picked_notta1)),
        side_effect=_side_effect_mean(otherm, pick1, ~picked_notta1),
    )
    results.append(s1)

    # ===== Trick 3: gain_target (predict metric − NOTTA, gate at 0) ========
    G = Ym - nottam[:, None]
    pickg, scoresg, lamg = _oof_config_predict(Xm, G, args.n_folds, args.seed)
    best_pred_gain = np.array(
        [scoresg[i, pickg[i]] if pickg[i] >= 0 else np.nan for i in range(n)]
    )
    adapt3 = best_pred_gain > 0.0
    real3 = np.where(adapt3, _realized_from_pick(Ym, pickg), nottam)
    s3 = _summ(real3, fixedm, nottam, aug_oracle)
    s3.update(
        trick="gain_target",
        desc="predict per-config gain vs NOTTA; skip if best≤0",
        lam=lamg,
        apply_rate=float(np.mean(adapt3)),
        skip_rate=float(np.mean(~adapt3)),
        side_effect=_side_effect_mean(otherm, pickg, adapt3),
    )
    results.append(s3)

    # ===== Trick 4: adapt_gate (binary "will TTA help?" → pick or NOTTA) ===
    margin = config_oracle - nottam
    z = (margin > 0.0).astype(float)
    if len(np.unique(z)) == 2:
        oof_p = _oof_logistic(Xm, z, args.n_folds, args.seed)
        auc = binary_auc(oof_p, z)
    else:
        oof_p = np.full(n, float(np.mean(z)))
        auc = None
    adapt4 = oof_p > 0.5
    real4 = np.where(adapt4, real2, nottam)  # if adapt, use route_for_metric pick
    s4 = _summ(real4, fixedm, nottam, aug_oracle)
    s4.update(
        trick="adapt_gate",
        desc="logistic gate on 'best config beats NOTTA' → route_for_metric pick else NOTTA",
        lam=None,
        auc=auc,
        apply_rate=float(np.mean(adapt4)),
        skip_rate=float(np.mean(~adapt4)),
        pos_rate=float(np.mean(z)),
        side_effect=_side_effect_mean(otherm, pick2, adapt4),
    )
    # per-OOD-quintile apply-rate + realized gain vs NOTTA
    q_rows = []
    for q in sorted(set(int(x) for x in q_m if x >= 0)):
        sel = q_m == q
        if sel.sum() == 0:
            continue
        q_rows.append(
            {
                "q": q,
                "n": int(sel.sum()),
                "apply_rate": float(np.mean(adapt4[sel])),
                "d_vs_notta": _nanmean(real4[sel] - nottam[sel]),
                "oracle_d_vs_notta": _nanmean(aug_oracle[sel] - nottam[sel]),
            }
        )
    s4["quintiles"] = q_rows
    results.append(s4)

    # ===== Trick 5: probe_route (probe PSNR/SSIM features + argmax) ========
    pick5, scores5, lam5 = _oof_config_predict(Xpm, Ym, args.n_folds, args.seed)
    real5 = _realized_from_pick(Ym, pick5)
    s5 = _summ(real5, fixedm, nottam, config_oracle)
    s5.update(
        trick="probe_route",
        desc=f"features + probe {', '.join(PROBE_RUNS)} PSNR/SSIM + Δ",
        lam=lam5,
        apply_rate=1.0,
        match=float(np.mean(pick5 == np.nanargmax(Ym, axis=1))),
        side_effect=_side_effect_mean(otherm, pick5, np.ones(n, bool)),
    )
    results.append(s5)

    # ---- write outputs ----------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "metric": metric,
        "unit": unit,
        "n": n,
        "series_root": str(args.series_root),
        "feature_date": str(args.feature_date),
        "grid_runs": grid_runs,
        "fixed_run": fixed_run,
        "fixed_mean": _nanmean(fixedm),
        "notta_mean": _nanmean(nottam),
        "config_oracle_mean": _nanmean(config_oracle),
        "aug_oracle_mean": _nanmean(aug_oracle),
        "results": results,
    }
    (args.output_dir / f"routing_tricks_{metric}.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    order = ["skip_augmented", "route_for_metric", "gain_target", "adapt_gate", "probe_route"]
    by_trick = {r["trick"]: r for r in results}
    lines = [
        f"# Five routing tricks — metric = {metric} ({unit})",
        "",
        f"**Series:** `{args.series_root.name}`  ·  **N:** {n}  ·  5-fold OOF (seed {args.seed}).",
        f"**Fixed ({fixed_run}):** {_fmt(_nanmean(fixedm))}  ·  "
        f"**NOTTA:** {_fmt(_nanmean(nottam))}  ·  "
        f"**Config-oracle:** {_fmt(_nanmean(config_oracle))}  ·  "
        f"**Augmented-oracle (incl. skip):** {_fmt(_nanmean(aug_oracle))}.",
        "",
        "Higher is better. Δ vs fixed / Δ vs NOTTA are mean(policy − baseline). "
        "`Captured` = fraction of (oracle − fixed) headroom recovered "
        "(augmented oracle for skip-aware tricks, config oracle otherwise).",
        "" if metric != "vbench" else
        "> VBench-total = unweighted mean of 7 raw dims (imaging_quality/MUSIQ "
        "0–100 dominated), NOT normalized VBench++ (~0.77).",
        "",
        "| Trick | Policy | Δ vs fixed | Δ vs NOTTA | Apply% | Captured | Notes |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for t in order:
        r = by_trick.get(t)
        if not r:
            continue
        note = r["desc"]
        if r.get("auc") is not None:
            note += f" · AUC={r['auc']:.3f}"
        if r.get("skip_rate") is not None:
            note += f" · skip={_pct(r['skip_rate'])}"
        lines.append(
            f"| `{t}` | {_fmt(r['policy'])} | {_fmt_d(r['d_vs_fixed'])} | "
            f"{_fmt_d(r['d_vs_notta'])} | {_pct(r.get('apply_rate'))} | "
            f"{_pct(r.get('captured'))} | {note} |"
        )
    lines.append("")

    # side-effect + adapt-gate quintiles
    se_name = other_name
    lines += [
        f"## Cross-metric side-effect ({se_name} of picked configs)",
        "",
        f"| Trick | {se_name} on config picks |",
        "|---|---:|",
    ]
    for t in order:
        r = by_trick.get(t)
        if r and r.get("side_effect") is not None:
            lines.append(f"| `{t}` | {_fmt(r['side_effect'])} |")
    lines.append("")

    gate = by_trick.get("adapt_gate", {})
    if gate.get("quintiles"):
        lines += [
            "## adapt_gate by OOD quintile (Q1 low … Q5 high OOD)",
            "",
            "| Quintile | N | Apply% | Δ gate−NOTTA | Δ oracle−NOTTA |",
            "|---|---:|---:|---:|---:|",
        ]
        for qr in gate["quintiles"]:
            lines.append(
                f"| Q{qr['q']} | {qr['n']} | {_pct(qr['apply_rate'])} | "
                f"{_fmt_d(qr['d_vs_notta'])} | {_fmt_d(qr['oracle_d_vs_notta'])} |"
            )
        lines.append("")

    lines += [
        "## Read",
        "",
        "- **Δ vs NOTTA > 0** = the deployable policy beats no-TTA on this metric.",
        "- **skip_augmented / gain_target / adapt_gate** can choose no-TTA per video; "
        "compare their Δ vs NOTTA to `route_for_metric` (always adapts).",
        "- **probe_route** uses actual cheap-probe metrics as features (legitimate at "
        "deploy: you run the probes); it upper-bounds probe-based routing.",
        "- If skip-aware tricks pick NOTTA often AND beat `route_for_metric`, the win is "
        "an **adapt-vs-skip gate**, not fine config selection.",
        "",
    ]
    report = args.output_dir / f"routing_tricks_{metric}_summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")

    # stderr digest
    for t in order:
        r = by_trick.get(t)
        if r:
            print(
                f"  {t:16s} Δfixed={_fmt_d(r['d_vs_fixed'])} "
                f"Δnotta={_fmt_d(r['d_vs_notta'])} apply={_pct(r.get('apply_rate'))} "
                f"cap={_pct(r.get('captured'))}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
