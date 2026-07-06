#!/usr/bin/env python3
"""Evaluate all four verifier routing options on the budget pilot @ N=200.

Options:
  1 — Frozen VideoScore on S2/S10 probe mp4s (deployable)
  2 — Frozen VideoReward / VideoAlign on probe mp4s (deployable)
  3 — VAE latent embeddings from probe mp4s + OOF ridge (LatSearch-style proxy)
  4 — ResNet18 frame embeddings + OOF ridge (scratch verifier proxy)

Each option reports:
  * probe_route  — rank probes, map to full config (exp14-style)
  * ridge_gain   — OOF ridge ΔVBench config picker (exp7-style)

Requires precomputed score/feature CSVs under --scores-dir / --features-dir.

Usage:
  python3 scripts/run_verifier_options_eval.py --run-all
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

from scripts.budget_routing_common import labeled_mask, load_pilot_bundle  # noqa: E402
from scripts.run_budget_routing_experiments import (  # noqa: E402
    _policy_from_budget_task,
    build_probe_features,
)
from scripts.run_vbench_gain_prediction_experiments import PROBE2, _row  # noqa: E402
from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
    run_budget_config_task,
)
from scripts.verifier_probe_common import (  # noqa: E402
    PROBE2 as VP_PROBE2,
    build_verifier_probe_features,
    eval_probe_route_policy,
    load_scores_table,
    route_from_probe_verifier,
    score_csv_columns,
    write_result_row,
)

BASELINE_BEST = 12.8
CEILING_REF = 17.5

OPTION_BACKENDS = {
    1: "videoscore",
    2: "videoreward",
}
OPTION3_RUNS = VP_PROBE2
OPTION4_RUNS = VP_PROBE2


def _load_embedding_table(features_dir: Path, run_ids: Sequence[str], mode: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Return {run_id: {video_id: emb_vector}}."""
    import csv

    table: Dict[str, Dict[str, np.ndarray]] = {rid: {} for rid in run_ids}
    for rid in run_ids:
        paths = sorted(features_dir.glob(f"{rid}_{mode}_shard*.csv"))
        if not paths:
            paths = sorted(features_dir.glob(f"{rid}_{mode}.csv"))
        for p in paths:
            with p.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    vid = row.get("video_id") or ""
                    if not vid:
                        continue
                    emb = []
                    j = 0
                    while f"emb_{j}" in row and row[f"emb_{j}"] != "":
                        emb.append(float(row[f"emb_{j}"]))
                        j += 1
                    if emb:
                        table[rid][vid] = np.array(emb, dtype=float)
    return table


def _build_embedding_probe_features(
    bundle: dict,
    video_ids: Sequence[str],
    base_X: np.ndarray,
    feat_names: List[str],
    emb_table: Dict[str, Dict[str, np.ndarray]],
    *,
    probe_runs: Sequence[str],
    prefix: str,
) -> Tuple[np.ndarray, List[str]]:
    out = base_X.copy()
    names = list(feat_names)
    cols: List[np.ndarray] = []
    fixed_rid = bundle["fixed_run"]
    fixed_emb = emb_table.get(fixed_rid, {})

    for rid in probe_runs:
        run_emb = emb_table.get(rid, {})
        dim = 0
        for v in run_emb.values():
            dim = len(v)
            break
        if dim == 0:
            continue
        mat = np.full((len(video_ids), dim), np.nan, dtype=float)
        for i, vid in enumerate(video_ids):
            e = run_emb.get(vid)
            if e is not None:
                mat[i] = e
        for j in range(dim):
            cols.append(mat[:, j])
            names.append(f"{prefix}_{rid}_e{j}")
        if rid != fixed_rid:
            dmat = np.full((len(video_ids), dim), np.nan, dtype=float)
            for i, vid in enumerate(video_ids):
                e = run_emb.get(vid)
                fe = fixed_emb.get(vid)
                if e is not None and fe is not None and len(e) == len(fe):
                    dmat[i] = e - fe
            for j in range(dim):
                cols.append(dmat[:, j])
                names.append(f"{prefix}_{rid}_de{j}")

    if cols:
        out = np.column_stack([out] + cols)
    return out, names


def _run_ridge_gain_router(
    name: str,
    X: np.ndarray,
    bundle: dict,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    seed: int,
    n_folds: int,
) -> dict:
    Y_gain = Y - fixed_vb[:, np.newaxis]
    with tempfile.TemporaryDirectory() as tmp:
        res = run_budget_config_task(
            video_ids=bundle["video_ids"],
            X=X,
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
    return _row(
        name,
        pol,
        n=int(mask.sum()),
        extra={"n_features": int(X.shape[1]), "target": "delta_vbench_total"},
    )


def _eval_frozen_option(
    opt: int,
    bundle: dict,
    scores_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    n_folds: int,
) -> Dict[str, dict]:
    backend = OPTION_BACKENDS[opt]
    grid = bundle["grid_runs"]
    Y = bundle["Y_total"]
    fixed_vb = bundle["fixed_vb"]
    mask = labeled_mask(fixed_vb, Y)
    video_ids = bundle["video_ids"]

    scores_by_run = load_scores_table(scores_dir, VP_PROBE2, backend)
    missing = [r for r in VP_PROBE2 if not scores_by_run.get(r)]
    if missing:
        raise FileNotFoundError(f"option {opt}: missing scores for {missing} under {scores_dir}")

    dims = score_csv_columns(backend)
    route_dims = ["mean5"] if backend == "videoscore" else ["overall"]
    if backend == "videoreward":
        route_dims = ["vq", "mq", "ta"]

    results: Dict[str, dict] = {}

    # probe_route
    picks = route_from_probe_verifier(
        video_ids, grid, scores_by_run, dims=route_dims,
    )
    pol = eval_probe_route_policy(picks, Y, fixed_vb, grid, mask=mask)
    row = {
        "experiment": f"opt{opt}_{backend}_probe_route",
        "option": opt,
        "backend": backend,
        "mode": "probe_route",
        "n_videos": int(mask.sum()),
        "n_scored": pol.get("n_scored"),
        "match_rate": pol.get("oof_oracle_match_rate"),
        "captured_pct": 100 * pol["fraction_oracle_captured"],
        "policy_gain": pol["mean_policy_vbench"] - pol["mean_fixed_vbench"],
        "headroom": pol["oracle_headroom"],
        "route_dims": route_dims,
    }
    results[row["experiment"]] = row
    write_result_row(out_dir, row, title=f"Option {opt} probe route ({backend})")

    # ridge_gain on verifier features
    impute = compute_impute(video_ids, bundle["features"], bundle["feat_names"])
    X_base = build_feature_matrix(video_ids, bundle["features"], bundle["feat_names"], impute=impute)
    X_ver, _ = build_verifier_probe_features(
        bundle, video_ids, X_base, bundle["feat_names"], scores_by_run,
        probe_runs=VP_PROBE2, dims=[d for d in dims if d != "mean5"],
    )
    ridge_name = f"opt{opt}_{backend}_ridge_gain"
    ridge_row = _run_ridge_gain_router(
        ridge_name, X_ver, bundle, Y, fixed_vb, grid, seed=seed, n_folds=n_folds,
    )
    ridge_row["option"] = opt
    ridge_row["backend"] = backend
    ridge_row["mode"] = "ridge_gain"
    results[ridge_name] = ridge_row
    write_result_row(out_dir, ridge_row, title=f"Option {opt} ridge gain ({backend})")
    return results


def _eval_embedding_option(
    opt: int,
    mode: str,
    bundle: dict,
    features_dir: Path,
    out_dir: Path,
    *,
    seed: int,
    n_folds: int,
) -> Dict[str, dict]:
    grid = bundle["grid_runs"]
    Y = bundle["Y_total"]
    fixed_vb = bundle["fixed_vb"]
    mask = labeled_mask(fixed_vb, Y)
    video_ids = bundle["video_ids"]

    emb_table = _load_embedding_table(features_dir, VP_PROBE2, mode)
    missing = [r for r in VP_PROBE2 if not emb_table.get(r)]
    if missing:
        raise FileNotFoundError(f"option {opt}: missing {mode} embeddings for {missing}")

    # Build pseudo-scores for probe routing: L2 norm of embedding as quality proxy
    scores_by_run: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rid in VP_PROBE2:
        scores_by_run[rid] = {}
        for vid, emb in emb_table[rid].items():
            scores_by_run[rid][vid] = {"norm": float(np.linalg.norm(emb))}

    results: Dict[str, dict] = {}
    picks = route_from_probe_verifier(
        video_ids, grid, scores_by_run, dims=["norm"],
    )
    pol = eval_probe_route_policy(picks, Y, fixed_vb, grid, mask=mask)
    pr_name = f"opt{opt}_{mode}_probe_route"
    pr_row = {
        "experiment": pr_name,
        "option": opt,
        "mode": "probe_route",
        "n_videos": int(mask.sum()),
        "n_scored": pol.get("n_scored"),
        "match_rate": pol.get("oof_oracle_match_rate"),
        "captured_pct": 100 * pol["fraction_oracle_captured"],
        "headroom": pol["oracle_headroom"],
    }
    results[pr_name] = pr_row
    write_result_row(out_dir, pr_row, title=f"Option {opt} probe route ({mode} norm)")

    impute = compute_impute(video_ids, bundle["features"], bundle["feat_names"])
    X_base = build_feature_matrix(video_ids, bundle["features"], bundle["feat_names"], impute=impute)
    prefix = "latent" if mode == "vae" else "scratch"
    X_emb, _ = _build_embedding_probe_features(
        bundle, video_ids, X_base, bundle["feat_names"], emb_table,
        probe_runs=VP_PROBE2, prefix=prefix,
    )
    rg_name = f"opt{opt}_{mode}_ridge_gain"
    rg_row = _run_ridge_gain_router(
        rg_name, X_emb, bundle, Y, fixed_vb, grid, seed=seed, n_folds=n_folds,
    )
    rg_row["option"] = opt
    rg_row["mode"] = "ridge_gain"
    results[rg_name] = rg_row
    write_result_row(out_dir, rg_row, title=f"Option {opt} ridge gain ({mode})")
    return results


def write_summary(all_results: Dict[str, dict], out_dir: Path) -> None:
    lines = [
        "# Verifier routing options @ N=200",
        "",
        "| Option | Experiment | Mode | Captured % | Match % | vs exp7 (12.8%) |",
        "|---:|---|---|---:|---:|---:|",
    ]
    best_name, best_cap = "", -1.0
    for name, row in sorted(all_results.items()):
        cap = row.get("captured_pct")
        cap_s = f"{cap:.1f}" if cap is not None else "—"
        mr = row.get("match_rate")
        mr_s = f"{100 * mr:.1f}" if mr is not None else "—"
        d = (cap - BASELINE_BEST) if cap is not None else float("nan")
        d_s = f"{d:+.1f}" if math.isfinite(d) else "—"
        opt = row.get("option", "?")
        mode = row.get("mode", "—")
        lines.append(f"| {opt} | `{name}` | {mode} | {cap_s} | {mr_s} | {d_s} |")
        if cap is not None and cap > best_cap:
            best_cap, best_name = cap, name

    lines += [
        "",
        f"**Best:** `{best_name}` at **{best_cap:.1f}%**.",
        "",
        "## Reference",
        "",
        f"- exp7 baseline: **{BASELINE_BEST}%**",
        f"- exp14 GT ceiling: **{CEILING_REF}%**",
        f"- Wave-2 GO bar: **>15%** deployable captured",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    deploy_rows = [
        r for r in all_results.values()
        if r.get("captured_pct") is not None and r.get("mode") == "ridge_gain"
    ]
    best_deploy = max(deploy_rows, key=lambda r: r["captured_pct"]) if deploy_rows else {}
    decision = {
        "best_experiment": best_name,
        "best_captured_pct": best_cap,
        "best_deployable_ridge": best_deploy.get("experiment"),
        "best_deployable_captured_pct": best_deploy.get("captured_pct"),
        "baseline_exp7_pct": BASELINE_BEST,
        "ceiling_exp14_pct": CEILING_REF,
        "wave2_go": bool(best_deploy.get("captured_pct", 0) >= 15.0),
    }
    (out_dir / "verifier_options_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--series-root",
        type=Path,
        default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot",
    )
    ap.add_argument(
        "--feature-date",
        type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-06",
    )
    ap.add_argument(
        "--scores-dir",
        type=Path,
        default=_REPO / "sweep_experiment/reports/verifier_scores",
    )
    ap.add_argument(
        "--features-dir",
        type=Path,
        default=_REPO / "sweep_experiment/reports/verifier_features",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--options", type=str, default="1,2,3,4", help="comma list or 'all'")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--run-all", action="store_true")
    args = ap.parse_args()

    if args.run_all:
        opts = [1, 2, 3, 4]
    elif args.options.lower() == "all":
        opts = [1, 2, 3, 4]
    else:
        opts = [int(x.strip()) for x in args.options.split(",") if x.strip()]

    out = args.output_dir or (args.feature_date / "verifier_options_eval")
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_pilot_bundle(args.series_root, args.feature_date)
    all_results: Dict[str, dict] = {}
    errors: List[str] = []

    for opt in opts:
        try:
            if opt in (1, 2):
                res = _eval_frozen_option(
                    opt, bundle, args.scores_dir, out, seed=args.seed, n_folds=args.n_folds,
                )
            elif opt == 3:
                res = _eval_embedding_option(
                    opt, "vae", bundle, args.features_dir, out,
                    seed=args.seed, n_folds=args.n_folds,
                )
            elif opt == 4:
                res = _eval_embedding_option(
                    opt, "resnet", bundle, args.features_dir, out,
                    seed=args.seed, n_folds=args.n_folds,
                )
            else:
                errors.append(f"unknown option {opt}")
                continue
            all_results.update(res)
            print(f"[ok] option {opt}", file=sys.stderr)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"option {opt}: {exc}")
            print(f"[FAIL] option {opt}: {exc}", file=sys.stderr)

    write_summary(all_results, out)
    (out / "all_results.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    if errors:
        (out / "errors.txt").write_text("\n".join(errors), encoding="utf-8")
        print(f"Completed with {len(errors)} errors — see {out}/errors.txt", file=sys.stderr)
    print(f"Wrote {out}/summary.md", file=sys.stderr)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
