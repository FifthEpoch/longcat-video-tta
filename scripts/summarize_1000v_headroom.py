#!/usr/bin/env python3
"""Definitive 1000v headroom narrative table — one place, all metrics.

For PSNR, VBench-total, and each of the 7 raw VBench dims, on the paired pool
(all 12 configs + NO-TTA scored), report:

  N                    videos in the paired pool
  notta                population-mean NO-TTA score           (the deployable baseline)
  best_fixed           best single config's mean              (DEPLOYABLE alternative)
  d_fixed_vs_notta     best_fixed - notta                     (the deployable gain — is any config worth it?)
  config_oracle        mean of per-video max over 12 configs  (per-video UPPER BOUND, needs GT to pick)
  d_oracle_vs_fixed    config_oracle - best_fixed             (routable headroom, if real)
  aug_oracle           mean of per-video max over 12 + NOTTA  (13-way upper bound)
  d_aug_vs_notta       aug_oracle - notta

FVD is NOT here: it is distribution-level (no per-video score), so it has no
per-video oracle. Compare FVD only at the population level via the matched-FVD job.

Offline; no generation. Usage:
  python3 scripts/summarize_1000v_headroom.py \
    --series-root sweep_experiment/results/panda_ood_budget_1000v_preview \
    --feature-date sweep_experiment/reports/per_video_analysis/2026-07-12 \
    --output-dir sweep_experiment/reports/per_video_analysis/2026-07-21/headroom_summary_1000v
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.budget_routing_common import load_pilot_bundle  # noqa: E402
from scripts.run_router_full_matrix import _load_notta  # noqa: E402
from scripts.analyze_adasteer_budget_vbench_oracle import (  # noqa: E402
    NOTTA_RUN_ID,
    discover_runs,
)
from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    load_per_video_vbench,
)
from scripts.diagnose_routability_per_vbench_dim import _build_dim_matrix  # noqa: E402


def _stats(Y: np.ndarray, notta: np.ndarray, grid_runs: List[str]) -> dict:
    """Y: (n_vid, k) per-config; notta: (n_vid,). Restrict to paired pool."""
    pool = np.all(~np.isnan(Y), axis=1) & ~np.isnan(notta)
    n = int(pool.sum())
    if n < 20:
        return {"n": n, "note": "insufficient coverage"}
    Ym, ntm = Y[pool], notta[pool]
    col_means = np.nanmean(Ym, axis=0)
    best_j = int(np.argmax(col_means))
    best_fixed = float(col_means[best_j])
    notta_mean = float(np.mean(ntm))
    config_oracle = float(np.mean(np.nanmax(Ym, axis=1)))
    aug_oracle = float(np.mean(np.maximum(np.nanmax(Ym, axis=1), ntm)))
    return {
        "n": n,
        "notta": notta_mean,
        "best_fixed": best_fixed,
        "best_config": grid_runs[best_j],
        "d_fixed_vs_notta": best_fixed - notta_mean,
        "config_oracle": config_oracle,
        "d_oracle_vs_fixed": config_oracle - best_fixed,
        "aug_oracle": aug_oracle,
        "d_aug_vs_notta": aug_oracle - notta_mean,
    }


def _fmt(x: float, nd: int = 4) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _dfmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{'+' if x >= 0 else ''}{x:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
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
    args = ap.parse_args()

    bundle = load_pilot_bundle(args.series_root, args.feature_date, require_vbench=True)
    vids: List[str] = bundle["video_ids"]
    grid_runs: List[str] = bundle["grid_runs"]

    rows: List[dict] = []

    # PSNR
    psnr = np.array(bundle["psnr"], dtype=float)
    rows.append({"metric": "PSNR (dB)", **_stats(psnr, _load_notta(args.series_root, vids, "psnr"), grid_runs)})

    # VBench-total
    ytot = np.array(bundle["Y_total"], dtype=float)
    rows.append({"metric": "VBench-total", **_stats(ytot, _load_notta(args.series_root, vids, "vbench"), grid_runs)})

    # per-dim
    runs = discover_runs(args.series_root)
    per_cfg = {rid: load_per_video_vbench(runs[rid]) for rid in grid_runs if rid in runs}
    notta_vb = load_per_video_vbench(runs[NOTTA_RUN_ID]) if NOTTA_RUN_ID in runs else {}
    for dim in VBENCH_DIMS:
        Y, nt = _build_dim_matrix(per_cfg, notta_vb, grid_runs, vids, dim)
        rows.append({"metric": f"vb:{dim}", **_stats(Y, nt, grid_runs)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "headroom_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    lines = [
        "# 1000v headroom narrative — PSNR, VBench-total, per-dim",
        "",
        f"**Series:** `{args.series_root.name}`. Paired pool per row (all 12 configs + NO-TTA scored). "
        "FVD omitted (distribution-level; no per-video oracle — use the matched-FVD job).",
        "",
        "| Metric | N | no-TTA | best fixed cfg | Δ fixed−noTTA (deployable) | config-oracle | Δ oracle−fixed (upper bnd) | Δ aug-oracle−noTTA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r.get("note"):
            lines.append(f"| {r['metric']} | {r['n']} | — | — | {r['note']} | — | — | — |")
            continue
        lines.append(
            f"| {r['metric']} | {r['n']} | {_fmt(r['notta'])} | {_fmt(r['best_fixed'])} "
            f"({r.get('best_config','')}) | {_dfmt(r['d_fixed_vs_notta'])} | {_fmt(r['config_oracle'])} "
            f"| {_dfmt(r['d_oracle_vs_fixed'])} | {_dfmt(r['d_aug_vs_notta'])} |"
        )
    lines += [
        "",
        "**Read:**",
        "- **Δ fixed−noTTA** is the only *deployable* number. If ≈0, the best single config does not beat no-TTA.",
        "- **Δ oracle−fixed** is a per-video UPPER BOUND (needs ground truth to pick) — only meaningful if it is real signal, not max-over-noise. Cross-check with `diagnose_routability*.py` (R²(gain) > 0).",
        "- **Δ aug-oracle−noTTA** adds NO-TTA as a 13th option; inflated when NO-TTA is an independent noise draw (corr(NO-TTA,cfg)≈0).",
        "",
    ]
    report = args.output_dir / "headroom_summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")
    for r in rows:
        if r.get("note"):
            print(f"  {r['metric']:24s} n={r['n']} {r['note']}", file=sys.stderr)
            continue
        print(
            f"  {r['metric']:24s} n={r['n']} notta={_fmt(r['notta'])} "
            f"fixed={_fmt(r['best_fixed'])} Δfix-notta={_dfmt(r['d_fixed_vs_notta'])} "
            f"oracle-fix={_dfmt(r['d_oracle_vs_fixed'])} aug-notta={_dfmt(r['d_aug_vs_notta'])}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
