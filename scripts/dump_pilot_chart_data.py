#!/usr/bin/env python3
"""Dump compact 200v-pilot chart data as JSON (run on the cluster, paste output).

Emits ONLY the aggregated numbers needed to render the pilot OOD/oracle charts
locally (no matplotlib dependency here, so it runs anywhere the metrics live):

  * chart1_psnr_delta            per-OOD-quintile mean/SEM/N of PSNR-oracle Δ vs
                                 NO-TTA (oracle = per-video max over 12 configs)
  * chart2_config_picks          per-quintile histogram counts of the PSNR-oracle
                                 argmax config (over the 12 configs)
  * vbench_dim_gain              per-dim config-oracle gain vs NO-TTA (raw + % of
                                 NO-TTA mean) → used to rank / pick winning dim
  * vbench_dim_delta_by_quintile per-dim, per-quintile mean/SEM/N config-oracle Δ
                                 (all 7 dims, so any dim can be charted locally)
  * winner_dim                   dim with the largest relative config-oracle gain

Usage (cluster):
    python3 scripts/dump_pilot_chart_data.py \
        --series-root sweep_experiment/results/panda_ood_budget_pilot \
        --baseline-series-root sweep_experiment/results/panda_1000v_standard \
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \
      | tee /tmp/pilot_chart_data.json

Then paste the JSON. It is small (a few hundred numbers).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    NOTTA_RUN_ID,
    OOD_COL,
    PILOT_GRID_RUN_ORDER,
    discover_runs,
)
from scripts.analyze_per_video_tta_gain import load_per_video_metrics  # noqa: E402
from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    load_per_video_vbench,
)
from scripts.caption_utils import canonical_video_id  # noqa: E402


def _canon(d: Dict[str, dict]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for k, v in d.items():
        out[canonical_video_id(k) or k] = v
    return out


def _resolve_notta(series_runs, baseline_runs) -> Optional[Path]:
    if NOTTA_RUN_ID in series_runs:
        return series_runs[NOTTA_RUN_ID]
    return baseline_runs.get(NOTTA_RUN_ID)


def _load_ood_scores(path: Path) -> Dict[str, float]:
    """Map canonical video_id -> OOD score (mean_diffusion_loss_caption)."""
    import csv

    out: Dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = canonical_video_id(row.get("video_id", ""))
            v = row.get(OOD_COL, "")
            if not vid or v in ("", None):
                continue
            try:
                x = float(v)
            except ValueError:
                continue
            if not np.isnan(x):
                out[vid] = x
    return out


def _assign_quintiles(scores: np.ndarray, n_bins: int = 5) -> np.ndarray:
    """Assign 1..n_bins quintiles over the *evaluated pool's* OOD scores.

    Quantile edges are computed only over the finite scores passed in (i.e. the
    videos actually being charted), so quintiles are balanced within the pool —
    matching the 200v-pilot semantics. NaN score -> NaN quintile.
    """
    q = np.full(scores.shape[0], np.nan, dtype=float)
    finite = np.isfinite(scores)
    vals = scores[finite]
    if vals.size == 0:
        return q
    edges = np.unique(np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1)))
    idx = np.digitize(scores[finite], edges[1:-1], right=False)
    idx = np.clip(idx, 0, max(len(edges) - 2, 0))
    q[finite] = idx + 1
    return q


def _oracle(M: np.ndarray) -> np.ndarray:
    """Per-row max ignoring NaN; NaN row -> NaN."""
    has = np.any(np.isfinite(M), axis=1)
    filled = np.where(np.isfinite(M), M, -np.inf)
    out = np.full(M.shape[0], np.nan)
    out[has] = filled[has].max(axis=1)
    return out


def _agg_by_quintile(delta: np.ndarray, quint: np.ndarray) -> Dict[str, object]:
    means, sems, ns = [], [], []
    for q in range(1, 6):
        sel = (quint == q) & np.isfinite(delta)
        vals = delta[sel]
        if vals.size:
            means.append(round(float(vals.mean()), 6))
            sems.append(round(float(vals.std(ddof=1) / np.sqrt(vals.size)), 6) if vals.size > 1 else 0.0)
            ns.append(int(vals.size))
        else:
            means.append(None)
            sems.append(None)
            ns.append(0)
    fin = delta[np.isfinite(delta)]
    pop = round(float(fin.mean()), 6) if fin.size else None
    return {"quintiles": [1, 2, 3, 4, 5], "mean": means, "sem": sems, "n": ns, "pop_mean": pop}


def main() -> int:
    ap = argparse.ArgumentParser(description="Dump compact 200v-pilot chart data as JSON")
    ap.add_argument("--series-root", type=Path,
                    default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot")
    ap.add_argument("--baseline-series-root", type=Path,
                    default=_REPO / "sweep_experiment/results/panda_1000v_standard")
    ap.add_argument("--ood-csv", type=Path,
                    default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv")
    args = ap.parse_args()

    runs = discover_runs(args.series_root)
    baseline_runs = discover_runs(args.baseline_series_root) if args.baseline_series_root.exists() else {}
    grid_runs = [r for r in PILOT_GRID_RUN_ORDER if r in runs]
    if not grid_runs:
        raise SystemExit(f"[error] no pilot grid configs under {args.series_root}")
    notta_dir = _resolve_notta(runs, baseline_runs)

    ood_scores = _load_ood_scores(args.ood_csv)

    # ---- PSNR matrices ----
    per_psnr = {r: _canon(load_per_video_metrics(runs[r])) for r in grid_runs}
    notta_psnr_map = _canon(load_per_video_metrics(notta_dir)) if notta_dir else {}
    vids = sorted({v for r in grid_runs for v, row in per_psnr[r].items() if row.get("psnr") is not None})
    K = len(grid_runs)
    P = np.full((len(vids), K), np.nan)
    npsnr = np.full(len(vids), np.nan)
    for i, v in enumerate(vids):
        for j, r in enumerate(grid_runs):
            val = per_psnr[r].get(v, {}).get("psnr")
            if val is not None:
                P[i, j] = float(val)
        nv = notta_psnr_map.get(v, {}).get("psnr")
        if nv is not None:
            npsnr[i] = float(nv)

    ood_vec = np.array([ood_scores.get(v, np.nan) for v in vids], dtype=float)
    n_ood_join = int(np.isfinite(ood_vec).sum())
    quint = _assign_quintiles(ood_vec, n_bins=5)  # balanced within evaluated pool
    have = np.any(np.isfinite(P), axis=1) & np.isfinite(npsnr) & np.isfinite(quint)
    psnr_delta = np.where(have, _oracle(P) - npsnr, np.nan)

    # chart 2: per-quintile config-pick counts
    picks = {f"Q{q}": {r: 0 for r in grid_runs} for q in range(1, 6)}
    for i in range(len(vids)):
        if not have[i]:
            continue
        q = int(quint[i])
        if 1 <= q <= 5 and np.any(np.isfinite(P[i])):
            picks[f"Q{q}"][grid_runs[int(np.nanargmax(P[i]))]] += 1

    # ---- VBench per-dim ----
    per_vb = {r: _canon(load_per_video_vbench(runs[r])) for r in grid_runs}
    notta_vb_map = _canon(load_per_video_vbench(notta_dir)) if notta_dir else {}
    dim_gain: Dict[str, dict] = {}
    dim_delta_q: Dict[str, dict] = {}
    for dim in VBENCH_DIMS:
        D = np.full((len(vids), K), np.nan)
        nd = np.full(len(vids), np.nan)
        for i, v in enumerate(vids):
            for j, r in enumerate(grid_runs):
                val = per_vb[r].get(v, {}).get(dim)
                if val is not None and np.isfinite(val):
                    D[i, j] = float(val)
            nv = notta_vb_map.get(v, {}).get(dim)
            if nv is not None and np.isfinite(nv):
                nd[i] = float(nv)
        h = np.any(np.isfinite(D), axis=1) & np.isfinite(nd)
        if int(h.sum()) < 20:
            continue
        delta = np.where(h, _oracle(D) - nd, np.nan)
        raw = float(np.nanmean(delta))
        notta_mean = float(np.nanmean(np.where(h, nd, np.nan)))
        rel = (raw / notta_mean * 100.0) if abs(notta_mean) > 1e-9 else None
        dim_gain[dim] = {
            "raw": round(raw, 6),
            "rel_pct": round(rel, 4) if rel is not None else None,
            "n": int(h.sum()),
            "notta_mean": round(notta_mean, 6),
        }
        dim_delta_q[dim] = _agg_by_quintile(delta, quint)

    winner = None
    ranked = [(d, g["rel_pct"]) for d, g in dim_gain.items() if g["rel_pct"] is not None]
    if ranked:
        winner = max(ranked, key=lambda t: t[1])[0]

    out = {
        "meta": {
            "series_root": str(args.series_root),
            "notta_dir": str(notta_dir) if notta_dir else None,
            "ood_csv": str(args.ood_csv),
            "grid_runs": grid_runs,
            "n_videos_total": len(vids),
            "n_ood_join": n_ood_join,
            "n_psnr_pool": int(have.sum()),
            "vbench_dims": list(VBENCH_DIMS),
        },
        "chart1_psnr_delta": _agg_by_quintile(psnr_delta, quint),
        "chart2_config_picks": picks,
        "vbench_dim_gain": dim_gain,
        "vbench_dim_delta_by_quintile": dim_delta_q,
        "winner_dim": winner,
    }
    print(json.dumps(out, indent=2))
    print(
        f"\n[info] grid={K} n_vids={len(vids)} ood_join={n_ood_join}/{len(vids)} "
        f"psnr_pool={int(have.sum())} vbench_dims_ok={len(dim_gain)} "
        f"winner={winner} notta={notta_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
