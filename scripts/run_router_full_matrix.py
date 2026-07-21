#!/usr/bin/env python3
"""Full router ablation matrix: 7 feature blocks × {12-config, 13-config} × {PSNR, VBench}.

Fills every combination the deploy chain left partial:
  Blocks   : A (video_caption), B (diffusion_ood), C (vae_inference) and every
             non-empty subset — A, B, C, A+B, A+C, B+C, A+B+C (7 subsets).
  Actions  : 12  = argmax over the 12 AdaSteer configs only.
             13  = argmax over {12 configs, NO-TTA}  (skip is a valid action).
  Metrics  : psnr (dB) and vbench (raw-total = unweighted mean of 7 raw dims).

Baselines / ceilings (per research-partner instruction):
  * ``fixed``  = best single **population-mean** config on the paired pool for the
    stated metric (best-PSNR config for psnr, best-VBench config for vbench) — the
    strongest no-per-video-routing baseline, NOT a designated default.
  * ``oracle`` = **augmented** oracle = per-video max over {12 configs, NO-TTA};
    no-TTA is always an available action, so the correct ceiling includes it. The
    within-action config-oracle (best of 12) is also reported for reference.
  * ``captured`` = (policy − fixed) / (augmented_oracle − fixed).

All routers are leakage-free 5-fold OOF ridge (one model per config, argmax the
predicted metric). Offline: reuses cached per-config metrics + features, no new
generation. Requires the NO-TTA run present under the series for the chosen metric
(PSNR always; VBench needs the NOTTA VBench backfill).

Usage:
    python3 scripts/run_router_full_matrix.py \
      --series-root sweep_experiment/results/panda_ood_budget_1000v_preview \
      --feature-date sweep_experiment/reports/per_video_analysis/2026-07-12 \
      --output-dir sweep_experiment/reports/per_video_analysis/2026-07-21/router_full_matrix_1000v
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
)
from scripts.budget_routing_common import (  # noqa: E402
    DEPLOY_BLOCK_OOD,
    DEPLOY_BLOCK_VAE,
    DEPLOY_BLOCK_VIDEO_CORE,
    build_deploy_feature_keep,
    load_pilot_bundle,
)
from scripts.run_routing_tricks import (  # noqa: E402
    _fmt,
    _fmt_d,
    _nanmean,
    _oof_config_predict,
    _pct,
    _realized_from_pick,
)
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


# block label -> (video_core, ood, vae_latent_profile)
BLOCKS: Dict[str, Tuple[bool, bool, bool]] = {
    "A": (True, False, False),
    "B": (False, True, False),
    "C": (False, False, True),
    "A+B": (True, True, False),
    "A+C": (True, False, True),
    "B+C": (False, True, True),
    "A+B+C": (True, True, True),
}
BLOCK_ORDER = ["A", "B", "C", "A+B", "A+C", "B+C", "A+B+C"]
BLOCK_DESC = {
    DEPLOY_BLOCK_VIDEO_CORE: "A=video/caption",
    DEPLOY_BLOCK_OOD: "B=diffusion-OOD",
    DEPLOY_BLOCK_VAE: "C=VAE-profile",
}


def _load_notta(series_root: Path, vids: List[str], metric: str) -> np.ndarray:
    runs = discover_runs(series_root)
    notta = np.full(len(vids), np.nan, dtype=float)
    notta_dir = runs.get(NOTTA_RUN_ID)
    if notta_dir is None:
        return notta
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
    return notta


def _match_rate(Y: np.ndarray, pick: np.ndarray) -> float:
    n, _ = Y.shape
    good = tot = 0
    for i in range(n):
        row = Y[i]
        if np.all(np.isnan(row)):
            continue
        tot += 1
        if int(pick[i]) == int(np.nanargmax(row)):
            good += 1
    return good / tot if tot else float("nan")


def _subset_X(
    feature_date: Path,
    vids: List[str],
    bundle: dict,
    flags: Tuple[bool, bool, bool],
) -> Tuple[np.ndarray, int]:
    """Build feature matrix restricted to the requested block subset."""
    order, _ = build_deploy_feature_keep(
        feature_date,
        video_core=flags[0],
        ood=flags[1],
        vae_latent_profile=flags[2],
    )
    keep = [n for n in order if n in set(bundle["feat_names"])]
    impute = compute_impute(vids, bundle["features"], keep)
    X = build_feature_matrix(vids, bundle["features"], keep, impute=impute)
    return X, len(keep)


def _row(
    block: str,
    actions: int,
    metric: str,
    X: np.ndarray,
    Ym: np.ndarray,
    nottam: np.ndarray,
    fixedm: np.ndarray,
    config_oracle: np.ndarray,
    aug_oracle: np.ndarray,
    n_folds: int,
    seed: int,
) -> dict:
    n, k = Ym.shape
    if actions == 13:
        Y_use = np.column_stack([Ym, nottam])
        pick, _, lam = _oof_config_predict(X, Y_use, n_folds, seed)
        realized = _realized_from_pick(Y_use, pick)
        apply_rate = float(np.mean(pick != k))  # picked a config (not the NOTTA col)
        match = _match_rate(Y_use, pick)
    else:
        pick, _, lam = _oof_config_predict(X, Ym, n_folds, seed)
        realized = _realized_from_pick(Ym, pick)
        apply_rate = 1.0
        match = _match_rate(Ym, pick)

    pol = _nanmean(realized)
    fx = _nanmean(fixedm)
    nt = _nanmean(nottam)
    orc = _nanmean(aug_oracle)  # augmented ceiling (incl. NO-TTA) per instruction
    head = orc - fx if not (math.isnan(orc) or math.isnan(fx)) else float("nan")
    cap = (pol - fx) / head if abs(head) > 1e-9 else float("nan")
    return {
        "block": block,
        "actions": actions,
        "metric": metric,
        "n_feat": X.shape[1],
        "lam": lam,
        "policy": pol,
        "fixed": fx,
        "notta": nt,
        "config_oracle": _nanmean(config_oracle),
        "aug_oracle": orc,
        "d_vs_fixed": pol - fx,
        "d_vs_notta": pol - nt,
        "apply_rate": apply_rate,
        "match": match,
        "captured": cap,
    }


def run_metric(
    metric: str,
    args: argparse.Namespace,
) -> Tuple[List[dict], dict]:
    bundle = load_pilot_bundle(
        args.series_root, args.feature_date, require_vbench=(metric == "vbench")
    )
    vids: List[str] = bundle["video_ids"]
    grid_runs: List[str] = bundle["grid_runs"]
    Y = np.array(bundle["psnr" if metric == "psnr" else "Y_total"], dtype=float)
    notta = _load_notta(args.series_root, vids, metric)

    # paired pool: ≥1 config + a NO-TTA score
    pool = ~np.all(np.isnan(Y), axis=1) & ~np.isnan(notta)
    n = int(pool.sum())
    if n < 30:
        raise SystemExit(
            f"[error] metric={metric}: only {n} videos with config+NOTTA. "
            + ("Run the NOTTA VBench backfill first." if metric == "vbench" else "")
        )

    # fixed = best population-mean config on the paired pool for this metric
    with np.errstate(invalid="ignore"):
        col_means = np.array(
            [
                np.nanmean(Y[pool, j]) if np.any(~np.isnan(Y[pool, j])) else -np.inf
                for j in range(Y.shape[1])
            ]
        )
    best_idx = int(np.argmax(col_means))
    fixed_run = grid_runs[best_idx]
    fixed_vb = Y[:, best_idx]

    Ym = Y[pool]
    nottam = notta[pool]
    fixedm = fixed_vb[pool]
    config_oracle = np.nanmax(Ym, axis=1)
    aug_oracle = np.maximum(config_oracle, nottam)
    vids_pool = [vids[i] for i in range(len(vids)) if pool[i]]

    print(
        f"[info] metric={metric} N={n} grid={len(grid_runs)} "
        f"fixed=best-config {fixed_run} (mean={col_means[best_idx]:.4f})",
        file=sys.stderr,
    )

    rows: List[dict] = []
    for block in BLOCK_ORDER:
        X, nfeat = _subset_X(args.feature_date, vids_pool, bundle, BLOCKS[block])
        for actions in (12, 13):
            r = _row(
                block, actions, metric, X, Ym, nottam, fixedm,
                config_oracle, aug_oracle, args.n_folds, args.seed,
            )
            rows.append(r)
            print(
                f"  {block:5s} act={actions} feat={nfeat:3d} "
                f"Δfixed={_fmt_d(r['d_vs_fixed'])} Δnotta={_fmt_d(r['d_vs_notta'])} "
                f"apply={_pct(r['apply_rate'])} cap={_pct(r['captured'])}",
                file=sys.stderr,
            )
    meta = {
        "metric": metric,
        "n": n,
        "fixed_run": fixed_run,
        "fixed_mean": float(col_means[best_idx]),
        "notta_mean": _nanmean(nottam),
        "config_oracle_mean": _nanmean(config_oracle),
        "aug_oracle_mean": _nanmean(aug_oracle),
    }
    return rows, meta


def _metric_table(metric: str, rows: List[dict], meta: dict, unit: str) -> List[str]:
    lines = [
        f"## metric = {metric} ({unit})",
        "",
        f"**N:** {meta['n']}  ·  **Fixed = best {metric} config (`{meta['fixed_run']}`):** "
        f"{_fmt(meta['fixed_mean'])}  ·  **NOTTA:** {_fmt(meta['notta_mean'])}  ·  "
        f"**config-oracle:** {_fmt(meta['config_oracle_mean'])}  ·  "
        f"**aug-oracle (incl. NO-TTA):** {_fmt(meta['aug_oracle_mean'])}.",
        "",
        "`Captured` = (policy − fixed)/(aug-oracle − fixed). `Apply%` = fraction of "
        "videos the 13-action router adapts (does not pick NO-TTA); 100% for 12-action.",
        "",
        "| Block | Feat | Actions | Policy | Δ vs fixed | Δ vs NOTTA | Apply% | Captured | Match% |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['block']} | {r['n_feat']} | {r['actions']} | {_fmt(r['policy'])} | "
            f"{_fmt_d(r['d_vs_fixed'])} | {_fmt_d(r['d_vs_notta'])} | "
            f"{_pct(r['apply_rate'])} | {_pct(r['captured'])} | {_pct(r['match'])} |"
        )
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description="Full router ablation matrix")
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
    ap.add_argument(
        "--metrics", nargs="+", default=["psnr", "vbench"], choices=("psnr", "vbench")
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: List[dict] = []
    all_meta: Dict[str, dict] = {}
    lines = [
        "# Full router ablation matrix — blocks × {12,13 actions} × {PSNR,VBench}",
        "",
        f"**Series:** `{args.series_root.name}`  ·  **Features:** `{args.feature_date.name}`  ·  "
        f"5-fold OOF (seed {args.seed}).",
        "",
        "Blocks: **A**=video/caption (9-d), **B**=diffusion-OOD (~20-d), "
        "**C**=VAE-profile (~130-d), and subsets. **12** = argmax over 12 configs; "
        "**13** = argmax over {12 configs, NO-TTA}. Oracle = augmented (incl. NO-TTA).",
        "",
        "> VBench-total = unweighted mean of 7 raw dims (imaging_quality/MUSIQ 0–100 "
        "dominated), NOT normalized VBench++ (~0.77).",
        "",
    ]
    for metric in args.metrics:
        unit = "dB" if metric == "psnr" else "raw-total"
        rows, meta = run_metric(metric, args)
        all_rows.extend(rows)
        all_meta[metric] = meta
        lines.extend(_metric_table(metric, rows, meta, unit))

    (args.output_dir / "router_full_matrix.json").write_text(
        json.dumps({"meta": all_meta, "rows": all_rows}, indent=2), encoding="utf-8"
    )
    report = args.output_dir / "router_full_matrix_summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
