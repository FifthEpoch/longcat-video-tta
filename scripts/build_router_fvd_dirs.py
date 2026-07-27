#!/usr/bin/env python3
"""Compose trained-router per-video config choices into FVD policy dirs, then FVD.

For each (metric, feature-block, action-set), reconstruct the leakage-free
5-fold **OOF** router picks (same ridge estimator as ``run_router_full_matrix``),
symlink the router-selected clip per video (a grid config, or the NOTTA clip when
a 13-action router chooses skip), restrict to the NOTTA common set so the router
FVD is matched-N against always_notta / fixed / oracle, and run ``eval_fvd.py``
against the shared preview GT cache.

The router *pick* is deployable (OOF, no per-video label leakage) — unlike the
oracle, which needs ground-truth to select. FVD itself is a pooled distribution
metric, so we simply compose the clips the router would have chosen and measure
how close that composed distribution is to the GT.

Usage:
    python3 scripts/build_router_fvd_dirs.py \
      --series-root sweep_experiment/results/panda_ood_budget_1000v_preview \
      --feature-date sweep_experiment/reports/per_video_analysis/2026-07-12 \
      --gt-cache gt_caches/panda_ood_budget_1000v_preview_longcat.npz \
      --output-root sweep_experiment/reports/budget_oracle_fvd_1000v_preview/routers \
      --metrics psnr vbench --blocks A+B+C --actions 12 13

For the VBench router to train on the corrected gen-only scores, export
``VBENCH_SUBDIR=vbench_results_geneval`` before running.
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

from scripts.analyze_adasteer_budget_oracle import NOTTA_RUN_ID  # noqa: E402
from scripts.budget_routing_common import load_pilot_bundle  # noqa: E402
from scripts.run_router_full_matrix import BLOCKS, _load_notta, _subset_X  # noqa: E402
from scripts.run_routing_tricks import _oof_config_predict  # noqa: E402
from sweep_experiment.scripts.build_budget_oracle_policy_dirs import (  # noqa: E402
    _index_grid_videos,
)
from sweep_experiment.scripts.build_oracle_policy_dirs import (  # noqa: E402
    index_method_videos,
)
from sweep_experiment.scripts.run_pilot_matched_fvd_baselines import (  # noqa: E402
    run_eval_fvd,
    symlink_policy_dir,
)


def build_one(
    *,
    series_root: Path,
    feature_date: Path,
    output_root: Path,
    metric: str,
    block: str,
    actions: int,
    n_folds: int,
    seed: int,
    gt_cache: Path,
    device: str,
    min_videos: int,
    force: bool,
    restrict_notta: bool,
    grid_index_cache: Dict[str, Dict[str, Path]],
    notta_index: Dict[str, Path],
) -> dict:
    bundle = load_pilot_bundle(
        series_root, feature_date, require_vbench=(metric == "vbench")
    )
    vids: List[str] = bundle["video_ids"]
    grid_runs: List[str] = bundle["grid_runs"]
    Y = np.array(bundle["psnr" if metric == "psnr" else "Y_total"], dtype=float)
    notta = _load_notta(series_root, vids, metric)

    pool = ~np.all(np.isnan(Y), axis=1) & ~np.isnan(notta)
    vids_pool = [vids[i] for i in range(len(vids)) if pool[i]]
    Ym = Y[pool]
    nottam = notta[pool]
    k = Ym.shape[1]

    X, nfeat = _subset_X(feature_date, vids_pool, bundle, BLOCKS[block])
    Y_use = np.column_stack([Ym, nottam]) if actions == 13 else Ym
    pick, _, lam = _oof_config_predict(X, Y_use, n_folds, seed)

    # resolve per-video source clip from the router's chosen action
    src_by_vid: Dict[str, Path] = {}
    picks: List[dict] = []
    n_apply = 0
    n_missing_src = 0
    for i, vid in enumerate(vids_pool):
        j = int(pick[i])
        if actions == 13 and j == k:
            chosen_run = NOTTA_RUN_ID
            src = notta_index.get(vid)
        elif 0 <= j < k:
            chosen_run = grid_runs[j]
            src = grid_index_cache.get(chosen_run, {}).get(vid)
        else:
            continue
        if src is None:
            n_missing_src += 1
            continue
        if restrict_notta and vid not in notta_index:
            continue
        if chosen_run != NOTTA_RUN_ID:
            n_apply += 1
        src_by_vid[vid] = src
        picks.append({"video_id": vid, "chosen_run": chosen_run})

    block_tag = block.replace("+", "")
    pol_name = f"router_{metric}_{block_tag}_{actions}act"
    ordered_ids = sorted(src_by_vid.keys())
    linked, missing = symlink_policy_dir(
        policy=pol_name,
        video_ids=ordered_ids,
        src_by_vid=src_by_vid,
        output_root=output_root,
        clean=True,
    )

    # bijectivity check: no two ids share a resolved clip
    resolved = [src_by_vid[v].resolve() for v in ordered_ids]
    n_unique = len(set(resolved))
    if n_unique != len(resolved):
        raise RuntimeError(
            f"{pol_name}: non-bijective router dir "
            f"({len(resolved)} ids -> {n_unique} unique clips)"
        )

    apply_rate = (n_apply / linked) if linked else float("nan")
    out_dir = output_root / pol_name
    (out_dir / "router_manifest.json").write_text(
        json.dumps(
            {
                "policy": pol_name,
                "metric": metric,
                "block": block,
                "actions": actions,
                "n_feat": int(nfeat),
                "lambda": float(lam),
                "linked": linked,
                "missing_src": n_missing_src,
                "apply_rate": apply_rate,
                "restrict_notta": restrict_notta,
                "picks": picks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    row = {
        "policy": pol_name,
        "metric": metric,
        "block": block,
        "actions": actions,
        "linked": linked,
        "apply_rate": apply_rate,
        "fvd": None,
        "num_valid_pairs": None,
    }

    out_json = out_dir / "fvd.json"
    rc = run_eval_fvd(
        gen_dir=out_dir / "videos",
        out_json=out_json,
        gt_cache=gt_cache,
        device=device,
        min_videos=min_videos,
        force=force,
    )
    if rc == 0 and out_json.is_file():
        blob = json.loads(out_json.read_text(encoding="utf-8"))
        row["fvd"] = blob.get("fvd")
        row["num_valid_pairs"] = blob.get("num_valid_pairs")
    else:
        row["fvd"] = f"ERROR(rc={rc})"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description="Trained-router FVD policy dirs + eval")
    ap.add_argument("--series-root", type=Path, required=True)
    ap.add_argument(
        "--feature-date", type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-12",
        help="Must match run_router_full_matrix (default 2026-07-12) so the FVD "
             "dirs reflect the same routers as the reported router matrix.",
    )
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--metrics", nargs="+", default=["psnr", "vbench"],
                    choices=("psnr", "vbench"))
    ap.add_argument("--blocks", nargs="+", default=["A+B+C"],
                    choices=list(BLOCKS.keys()))
    ap.add_argument("--actions", nargs="+", type=int, default=[12, 13],
                    choices=(12, 13))
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--min-videos", type=int, default=256)
    ap.add_argument("--force", action="store_true",
                    help="Pass --force to eval_fvd (needed when N<256).")
    ap.add_argument("--no-restrict-notta", action="store_true",
                    help="Do NOT restrict to the NOTTA common set (breaks matched-N).")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Build source indices once (bijective grid index verified by build guard).
    notta_index = index_method_videos(args.series_root, NOTTA_RUN_ID)
    print(f"[index] NOTTA: {len(notta_index)} videos", file=sys.stderr)
    # discover grid runs from one bundle so we index exactly what routing uses
    probe = load_pilot_bundle(args.series_root, args.feature_date,
                              require_vbench=False)
    grid_runs = probe["grid_runs"]
    grid_index_cache = {rid: _index_grid_videos(args.series_root, rid)
                        for rid in grid_runs}
    for rid in grid_runs:
        print(f"[index] {rid}: {len(grid_index_cache[rid])} videos", file=sys.stderr)

    rows: List[dict] = []
    for metric in args.metrics:
        for block in args.blocks:
            for actions in args.actions:
                print(f"\n=== router: metric={metric} block={block} "
                      f"actions={actions} ===", file=sys.stderr)
                row = build_one(
                    series_root=args.series_root,
                    feature_date=args.feature_date,
                    output_root=args.output_root,
                    metric=metric,
                    block=block,
                    actions=actions,
                    n_folds=args.n_folds,
                    seed=args.seed,
                    gt_cache=args.gt_cache,
                    device=args.device,
                    min_videos=args.min_videos,
                    force=args.force,
                    restrict_notta=not args.no_restrict_notta,
                    grid_index_cache=grid_index_cache,
                    notta_index=notta_index,
                )
                rows.append(row)
                print(f"  -> FVD={row['fvd']} N={row['num_valid_pairs']} "
                      f"apply={row['apply_rate']}", file=sys.stderr)

    # summary table
    lines = [
        "# Trained-router FVD (matched-N vs preview GT cache)",
        "",
        "Router picks are leakage-free 5-fold OOF ridge (argmax predicted metric); "
        "12-action = configs only, 13-action = configs + NO-TTA (skip). Restricted "
        "to the NOTTA common set for matched-N comparison to always_notta / fixed / "
        "oracle. Same `eval_fvd.py` + GT cache + `[48:62]` window.",
        "",
        "| Router | Metric | Block | Actions | N | Apply% | FVD |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        ap_str = ("—" if r["apply_rate"] is None or
                  (isinstance(r["apply_rate"], float) and np.isnan(r["apply_rate"]))
                  else f"{r['apply_rate'] * 100:.1f}%")
        fvd = r["fvd"]
        fvd_str = f"{fvd:.3f}" if isinstance(fvd, (int, float)) else str(fvd)
        lines.append(
            f"| {r['policy']} | {r['metric']} | {r['block']} | {r['actions']} | "
            f"{r.get('num_valid_pairs', '—')} | {ap_str} | {fvd_str} |"
        )
    lines.append("")
    summary = args.output_root / "router_fvd_summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    (args.output_root / "router_fvd_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
