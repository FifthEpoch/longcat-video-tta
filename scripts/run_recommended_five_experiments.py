#!/usr/bin/env python3
"""Five-experiment routing program @ pilot N=200 (maps 1:1 to recommended plan).

Experiments:
  exp1_probe_and_route   — S2+S10 probe signals → pick among {S5,S10,S20}
  exp2_dyn_delta_router  — ridge on ΔDyn vs fixed; report total + Dyn
  exp3_pairwise_ranker   — logistic + GBM pairwise among top-4 configs
  exp4_bestof3_nr_proxy  — rank S2/S10/S20 by LPIPS/SSIM (NR stand-ins)
  exp5_iq_constrained    — SKIPPED (needs TTA code change + GPU re-run)

Usage:
  python3 scripts/run_recommended_five_experiments.py --run-all
  python3 scripts/run_recommended_five_experiments.py --experiment exp2_dyn_delta_router
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.budget_routing_common import (  # noqa: E402
    BESTOF3_RUNS,
    bootstrap_captured,
    labeled_mask,
    load_metric_matrix,
    load_pilot_bundle,
    steps_bucket,
)
from scripts.run_budget_routing_experiments import (  # noqa: E402
    _HAS_SKLEARN,
    _num,
    _policy_from_budget_task,
    build_probe_features,
    run_pairwise_oof,
    run_proxy_pick,
)
from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
    eval_config_pick_policy,
    kfold_indices,
    ridge_fit,
    ridge_predict,
    run_budget_config_task,
    select_ridge_lambda,
    standardize_train_test,
)

EXP1_PROBES = ("S2_LR5e3", "S10_LR5e3")
EXP1_FULL_DEFAULTS = {"S5": "S5_LR5e3", "S10": "S10_LR5e3", "S20": "S20_LR1e3"}

ALL_FIVE = (
    "exp1_probe_and_route",
    "exp2_dyn_delta_router",
    "exp3_pairwise_ranker",
    "exp4_bestof3_nr_proxy",
    "exp5_iq_constrained",
)


def _cap_pct(policy: dict) -> Optional[float]:
    cap = policy.get("fraction_oracle_captured")
    if cap is None or (isinstance(cap, float) and math.isnan(cap)):
        return None
    return 100 * float(cap)


def _policy_row(name: str, policy: dict, *, n: int, extra: Optional[dict] = None) -> dict:
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
        "mean_policy_vbench": policy.get("mean_policy_vbench"),
        "mean_fixed_vbench": policy.get("mean_fixed_vbench"),
        "mean_oracle_vbench": policy.get("mean_oracle_vbench"),
    }
    if extra:
        row.update(extra)
    return row


def _write_json(out_dir: Path, name: str, payload: dict) -> None:
    (out_dir / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _kendall_tau(a: np.ndarray, b: np.ndarray) -> float:
    n = len(a)
    if n < 3:
        return float("nan")
    conc, discord = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if not (np.isfinite(a[i]) and np.isfinite(a[j]) and np.isfinite(b[i]) and np.isfinite(b[j])):
                continue
            sa = np.sign(a[i] - a[j])
            sb = np.sign(b[i] - b[j])
            if sa == sb and sa != 0:
                conc += 1
            elif sa != sb and sa != 0 and sb != 0:
                discord += 1
    denom = conc + discord
    return float((conc - discord) / denom) if denom else float("nan")


def run_exp1(bundle: dict, X_base: np.ndarray, feat_names: List[str], *, seed: int, n_folds: int) -> dict:
    """Probe S2+S10 (from existing pilot runs) → route among S5/S10/S20 full configs."""
    grid = bundle["grid_runs"]
    Y = bundle["Y_total"]
    Y_dyn = bundle["Y_dim"]["dynamic_degree"]
    fixed_vb = bundle["fixed_vb"]
    psnr = bundle["psnr"]
    fixed_j = grid.index(bundle["fixed_run"])

    probe_js = [grid.index(r) for r in EXP1_PROBES if r in grid]
    if len(probe_js) < 2:
        raise ValueError(f"need both probes {EXP1_PROBES}, have grid={grid}")

    full_rids = [EXP1_FULL_DEFAULTS[b] for b in ("S5", "S10", "S20") if EXP1_FULL_DEFAULTS[b] in grid]
    full_js = [grid.index(r) for r in full_rids]

    mask = labeled_mask(fixed_vb, Y)
    n = int(mask.sum())

    # --- Policy A: commit to probe winner (S2→S5, S10→S10) ---
    picks_commit = np.full(len(fixed_vb), -1, dtype=int)
    j_s2, j_s10 = probe_js[0], probe_js[1]
    for i in range(len(fixed_vb)):
        p2, p10 = psnr[i, j_s2], psnr[i, j_s10]
        if not (np.isfinite(p2) or np.isfinite(p10)):
            continue
        if np.isfinite(p10) and (not np.isfinite(p2) or p10 >= p2):
            rid = EXP1_FULL_DEFAULTS["S10"]
        else:
            rid = EXP1_FULL_DEFAULTS["S5"]
        if rid in grid:
            picks_commit[i] = grid.index(rid)

    valid = mask & (picks_commit >= 0)
    pol_commit_total = eval_config_pick_policy(
        picks_commit[valid], Y[valid], fixed_vb[valid], grid,
    )
    fixed_dyn = Y_dyn[:, fixed_j]
    pol_commit_dyn = eval_config_pick_policy(
        picks_commit[valid], Y_dyn[valid], fixed_dyn[valid], grid,
    )
    oracle_idx = np.nanargmax(Y[valid], axis=1)
    pol_commit_total["oof_oracle_match_rate"] = float(
        np.mean(picks_commit[valid] == oracle_idx)
    )

    # --- Policy B: ridge on S2+S10 probe features → pick among 3 full configs (OOF) ---
    X_probe, _ = build_probe_features(
        bundle, bundle["video_ids"], X_base, feat_names, probe_runs=EXP1_PROBES,
    )
    Y_sub = Y[:, full_js]
    sub_names = [grid[j] for j in full_js]
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
    pol_ridge_sub = _policy_from_budget_task(res)

    return {
        "commit_probe": _policy_row(
            "exp1_commit_probe",
            pol_commit_total,
            n=n,
            extra={
                "dyn_captured_pct": _cap_pct(pol_commit_dyn),
                "dyn_policy_gain": pol_commit_dyn.get("mean_policy_vbench", 0)
                - pol_commit_dyn.get("mean_fixed_vbench", 0),
                "full_default_map": EXP1_FULL_DEFAULTS,
            },
        ),
        "ridge_probe_3way": _policy_row(
            "exp1_ridge_probe_3way",
            pol_ridge_sub,
            n=n,
            extra={"candidate_configs": sub_names},
        ),
        "policies": {
            "commit_total": pol_commit_total,
            "commit_dyn": pol_commit_dyn,
            "ridge_3way": pol_ridge_sub,
        },
    }


def run_exp2(bundle: dict, X_base: np.ndarray, *, seed: int, n_folds: int) -> dict:
    """Ridge on ΔDyn vs fixed S10; evaluate picks on total VBench + Dyn."""
    grid = bundle["grid_runs"]
    Y_total = bundle["Y_total"]
    Y_dyn = bundle["Y_dim"]["dynamic_degree"]
    fixed_vb = bundle["fixed_vb"]
    fixed_j = grid.index(bundle["fixed_run"])
    fixed_dyn = Y_dyn[:, fixed_j]

    Y_delta = Y_dyn - fixed_dyn[:, np.newaxis]
    mask = labeled_mask(fixed_vb, Y_total) & np.isfinite(fixed_dyn)
    n = int(mask.sum())

    with tempfile.TemporaryDirectory() as tmp:
        res = run_budget_config_task(
            video_ids=bundle["video_ids"],
            X=X_base,
            Y=Y_delta,
            fixed_vb=fixed_dyn,
            notta_vb=np.full(len(fixed_vb), np.nan),
            grid_runs=grid,
            output_dir=Path(tmp),
            seed=seed,
            n_folds=n_folds,
        )
    pol_delta = _policy_from_budget_task(res)

    # Reconstruct OOF config picks from saved weights isn't available; use in-sample
    # argmax on predicted delta dyn for deployable eval on TOTAL vbench:
    # Train full-data ridge per config for point policy (report + note OOF from task)
    vid_m = [bundle["video_ids"][i] for i in range(len(fixed_vb)) if mask[i]]
    X = X_base[mask]
    Y_d = Y_delta[mask]
    fv_total = fixed_vb[mask]
    fv_dyn = fixed_dyn[mask]
    k = Y_d.shape[1]

    X_s = (X - X.mean(0)) / np.where(X.std(0) < 1e-8, 1.0, X.std(0))
    pred = np.full((len(X), k), np.nan)
    for j in range(k):
        y = Y_d[:, j]
        m = np.isfinite(y)
        if m.sum() < 10:
            continue
        w = ridge_fit(X_s[m], y[m], 0.1)
        pred[:, j] = ridge_predict(X_s, w)
    picks = np.nanargmax(pred, axis=1)

    pol_total_from_dyn_picks = eval_config_pick_policy(picks, Y_total[mask], fv_total, grid)
    pol_dyn_from_dyn_picks = eval_config_pick_policy(picks, Y_dyn[mask], fv_dyn, grid)
    oracle_dyn = np.nanargmax(Y_d, axis=1)
    pol_total_from_dyn_picks["oof_oracle_match_rate"] = float(np.mean(picks == np.nanargmax(Y_total[mask], axis=1)))
    pol_delta["oof_oracle_match_rate"] = float(np.mean(picks == oracle_dyn))

    return {
        "row_delta_target_oof": _policy_row("exp2_dyn_delta_oof", pol_delta, n=n),
        "row_total_vbench_from_dyn_picks": _policy_row(
            "exp2_dyn_picks_on_total_vbench",
            pol_total_from_dyn_picks,
            n=n,
        ),
        "row_dyn_from_dyn_picks": _policy_row(
            "exp2_dyn_picks_on_dyn",
            pol_dyn_from_dyn_picks,
            n=n,
            extra={"compare_to_linear_total_pct": 9.0},
        ),
        "policies": {
            "delta_oof": pol_delta,
            "total_from_dyn_picks": pol_total_from_dyn_picks,
            "dyn_from_dyn_picks": pol_dyn_from_dyn_picks,
        },
    }


def run_exp3(bundle: dict, X_base: np.ndarray, *, seed: int, n_folds: int) -> dict:
    Y = bundle["Y_total"]
    fixed_vb = bundle["fixed_vb"]
    grid = bundle["grid_runs"]
    mask = labeled_mask(fixed_vb, Y)
    n = int(mask.sum())

    pol_log = run_pairwise_oof(
        X_base, Y, fixed_vb, grid, seed=seed, n_folds=n_folds, use_gbm=False,
    )
    rows = [_policy_row("exp3_pairwise_logistic_top4", pol_log, n=n)]
    if _HAS_SKLEARN:
        pol_gbm = run_pairwise_oof(
            X_base, Y, fixed_vb, grid, seed=seed, n_folds=n_folds, use_gbm=True,
        )
        rows.append(_policy_row("exp3_pairwise_gbm_top4", pol_gbm, n=n))
    else:
        rows.append({"experiment": "exp3_pairwise_gbm_top4", "skipped": True, "reason": "no sklearn"})

    return {"rows": rows, "policies": {"logistic": pol_log, "gbm": rows[-1]}}


def run_exp4(bundle: dict, series_root: Path) -> dict:
    """Best-of-3: how well NR proxies rank configs vs VBench-total oracle."""
    grid = bundle["grid_runs"]
    Y = bundle["Y_total"]
    fixed_vb = bundle["fixed_vb"]
    runs = __import__(
        "scripts.analyze_adasteer_budget_oracle", fromlist=["discover_runs"]
    ).discover_runs(series_root)
    lpips = load_metric_matrix(runs, grid, bundle["video_ids"], "lpips")
    ssim = bundle["ssim"]
    psnr = bundle["psnr"]

    bestof_js = [grid.index(r) for r in BESTOF3_RUNS if r in grid]
    if len(bestof_js) < 2:
        raise ValueError(f"need best-of-3 configs in grid, have {grid}")

    mask = labeled_mask(fixed_vb, Y)
    n = int(mask.sum())

    proxies = {
        "lpips_nr": -lpips,  # lower LPIPS is better
        "ssim_nr": ssim,
        "psnr_ref": psnr,
        "lpips_ssim_combo": -lpips + ssim,
    }
    rows = []
    for pname, mat in proxies.items():
        picks = np.full(len(fixed_vb), -1, dtype=int)
        taus = []
        for i in range(len(fixed_vb)):
            scores = [(j, mat[i, j]) for j in bestof_js if np.isfinite(mat[i, j])]
            if not scores:
                continue
            picks[i] = max(scores, key=lambda x: x[1])[0]
            if mask[i]:
                vb_scores = [Y[i, j] for j in bestof_js if np.isfinite(Y[i, j])]
                pr_scores = [mat[i, j] for j in bestof_js if np.isfinite(mat[i, j])]
                if len(vb_scores) >= 3:
                    taus.append(_kendall_tau(np.array(pr_scores), np.array(vb_scores)))

        valid = mask & (picks >= 0)
        pol = eval_config_pick_policy(picks[valid], Y[valid], fixed_vb[valid], grid)
        oracle_valid = np.nanargmax(Y[valid], axis=1)
        pol["oof_oracle_match_rate"] = float(np.mean(picks[valid] == oracle_valid))
        rows.append(_policy_row(
            f"exp4_{pname}",
            pol,
            n=n,
            extra={
                "mean_kendall_tau_vs_vbench": float(np.nanmean(taus)) if taus else None,
                "proxy_note": "DOVER/UVQ not in pipeline; LPIPS/SSIM as NR stand-ins",
            },
        ))

    return {"rows": rows, "candidate_configs": list(BESTOF3_RUNS)}


def run_exp5_stub() -> dict:
    return {
        "experiment": "exp5_iq_constrained",
        "skipped": True,
        "reason": (
            "Requires modifying delta_experiment/scripts/run_delta_a.py to add an IQ "
            "preservation term (MUSIQ/BRISQUE on generated frames) during AdaSteer steps, "
            "then GPU re-run on 200v. Not runnable from existing metrics."
        ),
        "next_steps": [
            "Add --iq-preserve-weight and decode mid-TTA frames for NR IQ score",
            "Submit ONLY_RUNS=S10_LR5e3 with IQ term vs baseline via budget pilot sbatch",
            "Compare ΔIQ vs LoRA/retrieval frontier on pilot 200v",
        ],
    }


def load_results_from_dir(out_dir: Path) -> Dict[str, dict]:
    """Load per-task JSON outputs (for post-array aggregation)."""
    all_results: Dict[str, dict] = {}
    for name in ALL_FIVE:
        jp = out_dir / f"{name}.json"
        if jp.is_file():
            all_results[name] = json.loads(jp.read_text(encoding="utf-8"))
    return all_results


def write_summary(all_results: dict, out_dir: Path) -> None:
    lines = [
        "# Recommended five-experiment program @ N=200",
        "",
        "Maps 1:1 to the post-linear-router research plan.",
        "",
        "| Experiment | Key metric | Captured % (total VBench) | Notes |",
        "|---|---|---:|---|",
    ]

    def add_row(exp: str, metric: str, cap, note: str = "") -> None:
        cap_s = f"{cap:.1f}" if cap is not None and not (isinstance(cap, float) and math.isnan(cap)) else "—"
        lines.append(f"| {exp} | {metric} | {cap_s} | {note} |")

    e1 = all_results.get("exp1_probe_and_route", {})
    if e1.get("skipped"):
        lines.append(f"| Exp1 probe-and-route | — | — | **FAILED:** {e1.get('reason', '?')} |")
    else:
        if "commit_probe" in e1:
            r = e1["commit_probe"]
            add_row("Exp1 commit probe", "VBench total", r.get("captured_pct"), f"Dyn={r.get('dyn_captured_pct', '—')}%")
        if "ridge_probe_3way" in e1:
            r = e1["ridge_probe_3way"]
            add_row("Exp1 ridge probe 3-way", "VBench total", r.get("captured_pct"))

    e2 = all_results.get("exp2_dyn_delta_router", {})
    if e2.get("skipped"):
        lines.append(f"| Exp2 ΔDyn router | — | — | **FAILED:** {e2.get('reason', '?')} |")
    else:
        for key, label in (
            ("row_delta_target_oof", "Exp2 ΔDyn OOF"),
            ("row_total_vbench_from_dyn_picks", "Exp2 dyn picks → total"),
            ("row_dyn_from_dyn_picks", "Exp2 dyn picks → Dyn"),
        ):
            if key in e2:
                add_row(label, "see column", e2[key].get("captured_pct"))

    e3 = all_results.get("exp3_pairwise_ranker", {})
    if e3.get("skipped"):
        lines.append(f"| Exp3 pairwise | — | — | **FAILED:** {e3.get('reason', '?')} |")
    else:
        for r in e3.get("rows", []):
            if r.get("skipped"):
                lines.append(f"| {r['experiment']} | — | — | skipped |")
            else:
                add_row(r["experiment"], "VBench total", r.get("captured_pct"))

    e4 = all_results.get("exp4_bestof3_nr_proxy", {})
    if e4.get("skipped"):
        lines.append(f"| Exp4 NR proxy | — | — | **FAILED:** {e4.get('reason', '?')} |")
    else:
        for r in e4.get("rows", []):
            tau = r.get("mean_kendall_tau_vs_vbench")
            note = f"τ={tau:.3f}" if tau is not None and not math.isnan(tau) else "NR proxy rank"
            add_row(r["experiment"], "VBench total", r.get("captured_pct"), note)

    e5 = all_results.get("exp5_iq_constrained", {})
    if e5.get("skipped"):
        lines.append(f"| Exp5 IQ-constrained TTA | — | — | **skipped** (needs GPU) |")
    elif e5:
        lines.append(f"| Exp5 IQ-constrained TTA | — | — | done |")
    else:
        lines.append(f"| Exp5 IQ-constrained TTA | — | — | not run |")

    lines += [
        "",
        "## Status vs plan",
        "",
        "- **Exp1:** Uses existing S2/S10 pilot outputs (offline probe simulation). Real in-loop "
        "Δloss/DOVER needs fresh probe inference if simulation fails bar.",
        "- **Exp2:** Trains on ΔDyn; reports effect on total VBench (paper-relevant) and Dyn.",
        "- **Exp3:** Pairwise rankers (same as prior `pairwise_*_top4` jobs).",
        "- **Exp4:** LPIPS/SSIM NR proxies (DOVER not installed); measures rank quality vs oracle.",
        "- **Exp5:** Requires new TTA training runs — not included in CPU batch.",
        "",
    ]
    (out_dir / "recommended_five_experiments_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Recommended 5-experiment routing program")
    ap.add_argument("--series-root", type=Path, default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot")
    ap.add_argument(
        "--feature-date",
        type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-06",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--experiment", choices=ALL_FIVE, default=None)
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Merge existing per-task JSONs into summary (after Slurm array)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    if not args.run_all and not args.experiment and not args.aggregate_only:
        ap.error("Use --run-all, --experiment NAME, or --aggregate-only")

    out = args.output_dir or (
        _REPO / "sweep_experiment/reports/per_video_analysis/2026-07-05/recommended_five_experiments"
    )
    out.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        all_results = load_results_from_dir(out)
        if not all_results:
            print(f"[error] no JSON files under {out}", file=sys.stderr)
            return 1
        write_summary(all_results, out)
        print(f"Aggregated {len(all_results)} experiments → {out}/recommended_five_experiments_summary.md", file=sys.stderr)
        return 0

    exps = list(ALL_FIVE) if args.run_all else [args.experiment]
    all_results: Dict[str, dict] = {}

    bundle = None
    X_base = None
    feat_names = None
    if any(e != "exp5_iq_constrained" for e in exps):
        bundle = load_pilot_bundle(args.series_root, args.feature_date)
        impute = compute_impute(bundle["video_ids"], bundle["features"], bundle["feat_names"])
        X_base = build_feature_matrix(
            bundle["video_ids"], bundle["features"], bundle["feat_names"], impute=impute,
        )
        feat_names = bundle["feat_names"]

    for name in exps:
        print(f"[run] {name} ...", file=sys.stderr)
        try:
            if name == "exp1_probe_and_route":
                result = run_exp1(bundle, X_base, feat_names, seed=args.seed, n_folds=args.n_folds)
            elif name == "exp2_dyn_delta_router":
                result = run_exp2(bundle, X_base, seed=args.seed, n_folds=args.n_folds)
            elif name == "exp3_pairwise_ranker":
                result = run_exp3(bundle, X_base, seed=args.seed, n_folds=args.n_folds)
            elif name == "exp4_bestof3_nr_proxy":
                result = run_exp4(bundle, args.series_root)
            elif name == "exp5_iq_constrained":
                result = run_exp5_stub()
            else:
                raise ValueError(name)
            all_results[name] = result
            _write_json(out, name, result)
            print(f"  done {name}", file=sys.stderr)
        except Exception as exc:
            print(f"  FAILED {name}: {exc}", file=sys.stderr)
            all_results[name] = {"experiment": name, "skipped": True, "reason": str(exc)}
            _write_json(out, name, all_results[name])

    if args.run_all:
        write_summary(all_results, out)
        print(f"Wrote {out}/recommended_five_experiments_summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
