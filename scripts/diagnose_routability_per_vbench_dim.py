#!/usr/bin/env python3
"""Per-VBench-dimension routability — is any SINGLE VBench dimension routable?

VBench-total (unweighted mean of 7 raw dims) is un-routable (see diagnose_routability.py).
But recent work targets specific dimensions (TANGO/Video-T1 gain mostly in semantic /
consistency dims; imaging_quality & motion_smoothness barely move). This script repeats
the routability diagnostic PER dimension:

  for each of the 7 raw VBench dims, on the paired pool (all 12 configs + NO-TTA scored):
    - within-video config sigma       (does config choice move THIS dim per video?)
    - mean pairwise config correlation (are the 12 configs near-duplicates on this dim?)
    - corr(NO-TTA, config-mean)        (is no-TTA an independent draw on this dim?)
    - config-oracle gain over best fixed config
    - OOF (out-of-fold, 5-fold, leakage-free) ridge R^2 predicting the per-video
      oracle GAIN from features  (>0 ⇒ that dimension carries a routable signal)

Offline, no new generation.

Usage:
  python3 scripts/diagnose_routability_per_vbench_dim.py \
    --series-root sweep_experiment/results/panda_ood_budget_1000v_preview \
    --feature-date sweep_experiment/reports/per_video_analysis/2026-07-12 \
    --output-dir sweep_experiment/reports/per_video_analysis/2026-07-21/routability_per_dim_1000v
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
)
from scripts.budget_routing_common import load_pilot_bundle  # noqa: E402
from scripts.analyze_adasteer_budget_vbench_oracle import (  # noqa: E402
    NOTTA_RUN_ID,
    discover_runs,
)
from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    load_per_video_vbench,
)
from scripts.diagnose_routability import _pearson, _oof_ridge_r2, _f  # noqa: E402


def _build_dim_matrix(
    per_cfg: Dict[str, Dict[str, Dict[str, float]]],
    notta_vb: Dict[str, Dict[str, float]],
    grid_runs: List[str],
    vids: List[str],
    dim: str,
):
    n, k = len(vids), len(grid_runs)
    Y = np.full((n, k), np.nan)
    nt = np.full(n, np.nan)
    for i, v in enumerate(vids):
        for j, rid in enumerate(grid_runs):
            val = per_cfg.get(rid, {}).get(v, {}).get(dim)
            if val is not None:
                Y[i, j] = float(val)
        nval = notta_vb.get(v, {}).get(dim)
        if nval is not None:
            nt[i] = float(nval)
    return Y, nt


def main() -> int:
    ap = argparse.ArgumentParser(description="Per-VBench-dim routability")
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
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    bundle = load_pilot_bundle(args.series_root, args.feature_date, require_vbench=True)
    vids: List[str] = bundle["video_ids"]
    grid_runs: List[str] = bundle["grid_runs"]

    runs = discover_runs(args.series_root)
    per_cfg = {rid: load_per_video_vbench(runs[rid]) for rid in grid_runs if rid in runs}
    notta_vb = load_per_video_vbench(runs[NOTTA_RUN_ID]) if NOTTA_RUN_ID in runs else {}

    # features (shared across dims)
    feat_names = bundle["feat_names"]

    rows: List[dict] = []
    for dim in VBENCH_DIMS:
        Y, nt = _build_dim_matrix(per_cfg, notta_vb, grid_runs, vids, dim)
        pool = np.all(~np.isnan(Y), axis=1) & ~np.isnan(nt)
        n = int(pool.sum())
        if n < 40:
            rows.append({"dim": dim, "n": n, "note": "insufficient coverage"})
            continue
        Ym, ntm = Y[pool], nt[pool]
        vids_p = [vids[i] for i in range(len(vids)) if pool[i]]
        cfg_mean = np.nanmean(Ym, axis=1)
        within_sigma = float(np.mean(np.nanstd(Ym, axis=1)))
        corrs = [
            _pearson(Ym[:, a], Ym[:, b])
            for a in range(Ym.shape[1])
            for b in range(a + 1, Ym.shape[1])
        ]
        mean_cc = float(np.nanmean(corrs))
        corr_notta = _pearson(ntm, cfg_mean)
        col_means = np.nanmean(Ym, axis=0)
        best_j = int(np.argmax(col_means))
        fixed = Ym[:, best_j]
        gain = np.nanmax(Ym, axis=1) - fixed

        impute = compute_impute(vids_p, bundle["features"], feat_names)
        X = build_feature_matrix(vids_p, bundle["features"], feat_names, impute=impute)
        r2_gain = _oof_ridge_r2(X, gain, args.n_folds, args.seed)

        r = {
            "dim": dim,
            "n": n,
            "within_video_config_sigma": within_sigma,
            "mean_pairwise_config_corr": mean_cc,
            "corr_notta_vs_configmean": corr_notta,
            "best_config": grid_runs[best_j],
            "best_config_mean": float(col_means[best_j]),
            "notta_mean": float(np.mean(ntm)),
            "config_oracle_gain_over_fixed": float(np.mean(gain)),
            "r2_predict_oracle_gain_features": r2_gain,
        }
        rows.append(r)
        print(
            f"  {dim:24s} n={n} within_sigma={within_sigma:.4f} corr_cc={mean_cc:.3f} "
            f"corr(notta,cfg)={corr_notta:.3f} oracle_gain/fixed={r['config_oracle_gain_over_fixed']:.4f} "
            f"R2_gain={r2_gain:.4f}",
            file=sys.stderr,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "routability_per_dim.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    lines = [
        "# Per-VBench-dimension routability @ 1000v",
        "",
        f"**Series:** `{args.series_root.name}`. Paired pool per dim (all 12 configs + NO-TTA "
        "scored). OOF = out-of-fold (5-fold, leakage-free). Positive **R² (gain)** ⇒ that "
        "dimension carries a routable per-video signal; ≤0 ⇒ noise (no router can help).",
        "",
        "| VBench dim | N | within-cfg σ | config corr | corr(NO-TTA,cfg) | oracle gain/fixed | R² (gain) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("note"):
            lines.append(f"| {r['dim']} | {r['n']} | — | — | — | — | {r['note']} |")
            continue
        lines.append(
            f"| {r['dim']} | {r['n']} | {_f(r['within_video_config_sigma'])} | "
            f"{_f(r['mean_pairwise_config_corr'],3)} | {_f(r['corr_notta_vs_configmean'],3)} | "
            f"{_f(r['config_oracle_gain_over_fixed'])} | {_f(r['r2_predict_oracle_gain_features'])} |"
        )
    lines += [
        "",
        "**Read:** a dimension is routable only if R² (gain) > 0 AND corr(NO-TTA,cfg) is high "
        "(no-TTA is a stable, not independent, draw). Dimensions where corr(NO-TTA,cfg)≈0 are "
        "scoring noise (like VBench-total). imaging_quality routing is typically degenerate "
        "(monotone in 'adapt less' ⇒ collapses to ≈ no-TTA).",
        "",
    ]
    report = args.output_dir / "routability_per_dim_summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
