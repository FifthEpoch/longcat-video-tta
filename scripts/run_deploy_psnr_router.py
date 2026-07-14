#!/usr/bin/env python3
"""Deploy-strict PSNR router @ pilot N=200 — same 9-d Block A, PSNR target.

Uses the **same** handcrafted video/caption features as ``video_caption_only``
(cuts, CLIP, DINO, Laplacian, RGB entropy) but trains 12 ridge models to
predict **PSNR per config** and deploys by argmax predicted PSNR.

Contrasts with the VBench router (``run_deploy_strict_router_experiments.py``)
which predicts VBench total per config.

Usage:
  python3 scripts/run_deploy_psnr_router.py
  python3 scripts/run_deploy_psnr_router.py --output-dir sweep_experiment/reports/per_video_analysis/2026-07-06/deploy_psnr_router
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.budget_routing_common import labeled_mask, load_pilot_bundle  # noqa: E402
from scripts.run_deploy_strict_router_experiments import (  # noqa: E402
    EXPERIMENT_SPECS,
    _load_bundle,
)
from scripts.run_budget_routing_experiments import _policy_from_budget_task  # noqa: E402
from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
    eval_config_pick_policy,
    run_budget_config_task,
)


def _fixed_metric_per_video(M: np.ndarray, fixed_j: int) -> np.ndarray:
    out = np.full(M.shape[0], np.nan, dtype=float)
    for i in range(M.shape[0]):
        v = M[i, fixed_j]
        if np.isfinite(v):
            out[i] = float(v)
    return out


def _side_effect_vbench(
    picks: np.ndarray,
    Y_vbench: np.ndarray,
    fixed_vb: np.ndarray,
    grid_runs: List[str],
    valid: np.ndarray,
) -> dict:
    """VBench realized when routing by PSNR-trained picker."""
    return eval_config_pick_policy(
        picks[valid], Y_vbench[valid], fixed_vb[valid], grid_runs,
    )


def _side_effect_psnr(
    picks: np.ndarray,
    psnr: np.ndarray,
    fixed_psnr: np.ndarray,
    grid_runs: List[str],
    valid: np.ndarray,
) -> dict:
    return eval_config_pick_policy(
        picks[valid], psnr[valid], fixed_psnr[valid], grid_runs,
    )


def _load_oof_picks(csv_path: Path, video_ids: List[str], grid: List[str]) -> np.ndarray:
    import csv

    picks = np.full(len(video_ids), -1, dtype=int)
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            rid = row.get("picked_run", "")
            if vid in video_ids and rid in grid:
                picks[video_ids.index(vid)] = grid.index(rid)
    return picks


def write_summary(out: Path, report: dict) -> None:
    psnr = report["psnr_policy"]
    vb = report.get("vbench_side_effect")
    n = report["n_videos"]
    lines = [
        f"# Deploy PSNR router @ N={n} — Block A (9-d) → predict PSNR per config",
        "",
        "**Features:** same 9-d ``video_caption_only`` (cuts, CLIP, DINO, texture).",
        "**Target:** PSNR (not VBench). **Deploy:** argmax predicted PSNR → one AdaSteer.",
        "",
        "## PSNR objective (primary)",
        "",
        f"- **N:** {report['n_videos']}",
        f"- **OOF oracle-config match rate (PSNR oracle):** {100 * report['psnr_oracle_match_rate']:.1f}%",
        f"- **Mean PSNR (policy):** {psnr['mean_policy_vbench']:.4f} dB",
        f"- **Mean PSNR (fixed S10):** {psnr['mean_fixed_vbench']:.4f} dB",
        f"- **Mean PSNR (oracle):** {psnr['mean_oracle_vbench']:.4f} dB",
        f"- **Δ vs fixed:** {psnr['mean_policy_vbench'] - psnr['mean_fixed_vbench']:+.4f} dB",
        f"- **PSNR oracle headroom captured:** {100 * psnr.get('fraction_oracle_captured', float('nan')):.1f}%",
        "",
        "## VBench side effect (not optimized)",
        "",
    ]
    if vb:
        cap = 100 * vb.get("fraction_oracle_captured", float("nan"))
        lines += [
            f"- **Mean VBench total (same picks):** {vb['mean_policy_vbench']:.4f}",
            f"- **Fixed S10 VBench:** {vb['mean_fixed_vbench']:.4f}",
            f"- **VBench headroom captured (side effect):** {cap:.1f}%",
            "",
        ]
    lines += [
        "## Compare to VBench-targeted router (Block A)",
        "",
        "| Router target | PSNR Δ vs fixed | PSNR cap % | VB cap % |",
        "|---------------|------------------:|-----------:|---------:|",
        f"| **PSNR (this run)** | {report['psnr_delta_vs_fixed']:+.4f} dB | "
        f"{report['psnr_captured_pct']:.1f} | "
        f"{report.get('vbench_captured_pct_side_effect', float('nan')):.1f} |",
        f"| VBench (prior `video_caption_only`) | +0.009 dB | 1.2 | **20.8** |",
        "",
        "## Interpretation",
        "",
        "If PSNR cap % ≫ 1.2% while VB cap % drops, the **objective** was the bottleneck,",
        "not the 9-d input format. If PSNR cap % stays low, features lack PSNR signal.",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


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
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    spec = EXPERIMENT_SPECS["video_caption_only"]
    bundle, feat_names, block_map = _load_bundle(
        args.series_root, args.feature_date, spec, require_vbench=False,
    )
    video_ids = bundle["video_ids"]
    grid = bundle["grid_runs"]
    psnr = bundle["psnr"]
    Y_vb = bundle["Y_total"]
    fixed_vb = bundle["fixed_vb"]
    if bundle["fixed_run"] not in grid:
        print(
            f"[error] fixed run {bundle['fixed_run']!r} not in PSNR grid {grid}",
            file=sys.stderr,
        )
        return 2
    fixed_j = grid.index(bundle["fixed_run"])
    fixed_psnr = _fixed_metric_per_video(psnr, fixed_j)

    mask = labeled_mask(fixed_psnr, psnr)
    if mask.sum() < 30:
        print(f"[error] only {mask.sum()} labeled videos", file=sys.stderr)
        return 2

    impute = compute_impute(video_ids, bundle["features"], feat_names)
    X = build_feature_matrix(video_ids, bundle["features"], feat_names, impute=impute)

    out = args.output_dir or (args.feature_date / "deploy_psnr_router")
    out.mkdir(parents=True, exist_ok=True)

    res = run_budget_config_task(
        video_ids=video_ids,
        X=X,
        Y=psnr,
        fixed_vb=fixed_psnr,
        notta_vb=np.full(len(video_ids), np.nan),
        grid_runs=grid,
        output_dir=out,
        seed=args.seed,
        n_folds=args.n_folds,
    )

    picks = _load_oof_picks(out / "budget_config_oof_predictions.csv", video_ids, grid)
    psnr_pol = _side_effect_psnr(picks, psnr, fixed_psnr, grid, mask)
    vb_pol = _side_effect_vbench(picks, Y_vb, fixed_vb, grid, mask)

    psnr_oracle_idx = np.nanargmax(psnr[mask], axis=1)
    psnr_match = float(np.mean(picks[mask] == psnr_oracle_idx))

    report = {
        "experiment": "video_caption_psnr_target",
        "n_videos": int(mask.sum()),
        "n_features": int(X.shape[1]),
        "feature_blocks": block_map,
        "psnr_oracle_match_rate": psnr_match,
        "psnr_policy": psnr_pol,
        "vbench_side_effect": vb_pol,
        "psnr_delta_vs_fixed": psnr_pol["mean_policy_vbench"] - psnr_pol["mean_fixed_vbench"],
        "psnr_captured_pct": 100 * psnr_pol.get("fraction_oracle_captured", float("nan")),
        "vbench_captured_pct_side_effect": 100 * vb_pol.get("fraction_oracle_captured", float("nan")),
        "ridge_lambda": res.get("ridge_lambda"),
        "top_picks_psnr": psnr_pol.get("top_picks"),
    }
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary(out, report)

    print(
        f"PSNR router: Δ={report['psnr_delta_vs_fixed']:+.4f} dB, "
        f"captured={report['psnr_captured_pct']:.1f}% "
        f"(VB side effect {report['vbench_captured_pct_side_effect']:.1f}%)",
        file=sys.stderr,
    )
    print(f"Wrote {out}/summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
