#!/usr/bin/env python3
"""Render 200v-pilot OOD/oracle charts as **PNG images** (run on the cluster).

Produces exactly the figures requested for the meeting:

  1. ``psnr_oracle_delta_by_ood_quintile.png``
     Bar chart — per OOD quintile, mean per-video PSNR-oracle Δ vs NO-TTA (dB),
     where oracle = per-video max PSNR over the 12 AdaSteer configs. Error bars
     = SEM. This is the "PSNR oracle Δ vs no-TTA (dB) · 200v pilot" figure.

  2. ``config_picks_quintile_Q{1..5}.png``  (5 separate PNGs)
     For every OOD quintile, a histogram over the 12 configs counting how many
     videos in that quintile were won by each config under the PSNR oracle
     (per-video argmax PSNR). A combined 5-panel ``config_picks_all_quintiles.png``
     is also written for convenience.

  3a. ``vbench_dim_oracle_gain.png``
     Bar chart ranking the 7 VBench dims by config-oracle gain vs NO-TTA
     (per-video max over 12 configs − NO-TTA), shown as **relative %** of the
     NO-TTA mean so dims on different scales (MUSIQ 0–100 vs consistency 0–1)
     are comparable. Identifies the "winning" dimension.
  3b. ``vbench_<winner>_oracle_delta_by_ood_quintile.png``
     The chart-1 treatment applied to the winning VBench dim: per OOD quintile,
     mean config-oracle Δ vs NO-TTA (raw dim units).

All charts are per-video oracle **upper bounds** (max over configs), i.e. not
deployable policies — they show headroom, not a learned router's realized gain.

Offline: reuses cached per-config PSNR/VBench + the OOD CSV. No generation.

Usage (cluster):
    conda activate longcat   # or the env with matplotlib/numpy
    python3 scripts/plot_pilot_ood_charts.py \
        --series-root sweep_experiment/results/panda_ood_budget_pilot \
        --baseline-series-root sweep_experiment/results/panda_1000v_standard \
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \
        --out-dir sweep_experiment/reports/per_video_analysis/2026-07-21/pilot_ood_charts
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    NOTTA_RUN_ID,
    PILOT_GRID_RUN_ORDER,
    discover_runs,
    load_ood_quintiles,
)
from scripts.analyze_per_video_tta_gain import load_per_video_metrics  # noqa: E402
from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    load_per_video_vbench,
)
from scripts.caption_utils import canonical_video_id  # noqa: E402

DIM_SHORT = {
    "subject_consistency": "Subject consistency",
    "background_consistency": "Background consistency",
    "aesthetic_quality": "Aesthetic quality",
    "motion_smoothness": "Motion smoothness",
    "dynamic_degree": "Dynamic degree",
    "imaging_quality": "Imaging quality",
    "temporal_flickering": "Temporal flickering",
}
QUINTILE_LABELS = {
    1: "Q1\n(most in-dist)",
    2: "Q2",
    3: "Q3",
    4: "Q4",
    5: "Q5\n(most OOD)",
}
BLUE = "#2c6fbb"
GREEN = "#2e8b57"
ORANGE = "#d9822b"


def _canon_dict(d: Dict[str, dict]) -> Dict[str, dict]:
    """Re-key a {video_name: ...} dict by canonical video id."""
    out: Dict[str, dict] = {}
    for k, v in d.items():
        cid = canonical_video_id(k) or k
        out[cid] = v
    return out


def _resolve_notta_dir(
    series_runs: Dict[str, Path], baseline_runs: Dict[str, Path]
) -> Optional[Path]:
    if NOTTA_RUN_ID in series_runs:
        return series_runs[NOTTA_RUN_ID]
    return baseline_runs.get(NOTTA_RUN_ID)


# --------------------------------------------------------------------------- data
def build_psnr(
    runs: Dict[str, Path], grid_runs: List[str], notta_dir: Optional[Path]
) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Return (video_ids, P[N,K] config PSNR, notta[N] PSNR)."""
    per = {r: _canon_dict(load_per_video_metrics(runs[r])) for r in grid_runs}
    notta_per = _canon_dict(load_per_video_metrics(notta_dir)) if notta_dir else {}

    vids = sorted(
        {v for r in grid_runs for v, row in per[r].items() if row.get("psnr") is not None}
    )
    K = len(grid_runs)
    P = np.full((len(vids), K), np.nan, dtype=float)
    notta = np.full(len(vids), np.nan, dtype=float)
    for i, v in enumerate(vids):
        for j, r in enumerate(grid_runs):
            val = per[r].get(v, {}).get("psnr")
            if val is not None:
                P[i, j] = float(val)
        nv = notta_per.get(v, {}).get("psnr")
        if nv is not None:
            notta[i] = float(nv)
    return vids, P, notta


def build_vbench_dim(
    runs: Dict[str, Path],
    grid_runs: List[str],
    notta_dir: Optional[Path],
    vids: List[str],
    dim: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (D[N,K] config dim-score, notta[N] dim-score) aligned to *vids*."""
    per = {r: _canon_dict(load_per_video_vbench(runs[r])) for r in grid_runs}
    notta_per = _canon_dict(load_per_video_vbench(notta_dir)) if notta_dir else {}
    K = len(grid_runs)
    D = np.full((len(vids), K), np.nan, dtype=float)
    notta = np.full(len(vids), np.nan, dtype=float)
    for i, v in enumerate(vids):
        for j, r in enumerate(grid_runs):
            val = per[r].get(v, {}).get(dim)
            if val is not None and np.isfinite(val):
                D[i, j] = float(val)
        nv = notta_per.get(v, {}).get(dim)
        if nv is not None and np.isfinite(nv):
            notta[i] = float(nv)
    return D, notta


def _quintile_stats(
    delta: np.ndarray, quint: np.ndarray
) -> Tuple[List[int], List[float], List[float], List[int]]:
    """Mean + SEM of *delta* per quintile (1..5), ignoring NaN."""
    qs, means, sems, ns = [], [], [], []
    for q in range(1, 6):
        sel = (quint == q) & np.isfinite(delta)
        vals = delta[sel]
        qs.append(q)
        if vals.size:
            means.append(float(vals.mean()))
            sems.append(float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0)
            ns.append(int(vals.size))
        else:
            means.append(np.nan)
            sems.append(0.0)
            ns.append(0)
    return qs, means, sems, ns


# --------------------------------------------------------------------------- charts
def chart_quintile_delta(
    delta: np.ndarray,
    quint: np.ndarray,
    *,
    title: str,
    ylabel: str,
    color: str,
    out_path: Path,
) -> Dict[str, object]:
    qs, means, sems, ns = _quintile_stats(delta, quint)
    pop = float(np.nanmean(delta[np.isfinite(delta)])) if np.any(np.isfinite(delta)) else np.nan

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(qs))
    bars = ax.bar(x, means, yerr=sems, capsize=4, color=color, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    if np.isfinite(pop):
        ax.axhline(pop, color="gray", linestyle="--", linewidth=1.0, label=f"pop. mean = {pop:+.3f}")
        ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([QUINTILE_LABELS[q] for q in qs], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    for xi, m, n in zip(x, means, ns):
        if np.isfinite(m):
            off = 0.02 * (abs(max(means, default=1)) or 1)
            ax.text(xi, m + (off if m >= 0 else -off), f"{m:+.3f}\n(n={n})",
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return {"quintiles": qs, "mean_delta": means, "sem": sems, "n": ns, "pop_mean": pop}


def chart_config_picks(
    P: np.ndarray,
    quint: np.ndarray,
    grid_runs: List[str],
    out_dir: Path,
) -> Dict[str, object]:
    """One histogram per quintile of PSNR-oracle argmax config counts."""
    K = len(grid_runs)
    counts_by_q: Dict[int, np.ndarray] = {q: np.zeros(K, dtype=int) for q in range(1, 6)}
    for i in range(P.shape[0]):
        q = int(quint[i]) if np.isfinite(quint[i]) else 0
        if q < 1 or q > 5:
            continue
        row = P[i]
        if not np.any(np.isfinite(row)):
            continue
        counts_by_q[q][int(np.nanargmax(row))] += 1

    x = np.arange(K)
    ymax = max((c.max() for c in counts_by_q.values() if c.size), default=1)
    for q in range(1, 6):
        counts = counts_by_q[q]
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(x, counts, color=BLUE, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(grid_runs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("videos won (PSNR oracle)")
        ax.set_ylim(0, ymax * 1.18 + 1)
        ax.set_title(
            f"200v pilot — config picks, OOD {QUINTILE_LABELS[q].splitlines()[0]} "
            f"(n={int(counts.sum())})",
            fontsize=11,
        )
        for xi, c in zip(x, counts):
            if c:
                ax.text(xi, c + 0.02 * ymax + 0.05, str(int(c)), ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        out_path = out_dir / f"config_picks_quintile_Q{q}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_path}")

    # combined 5-panel
    fig, axes = plt.subplots(5, 1, figsize=(9, 14), sharex=True)
    for q, ax in zip(range(1, 6), axes):
        counts = counts_by_q[q]
        ax.bar(x, counts, color=BLUE, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(f"{QUINTILE_LABELS[q].splitlines()[0]}\n(n={int(counts.sum())})", fontsize=9)
        ax.set_ylim(0, ymax * 1.18 + 1)
        for xi, c in zip(x, counts):
            if c:
                ax.text(xi, c + 0.02 * ymax + 0.05, str(int(c)), ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(grid_runs, rotation=45, ha="right", fontsize=8)
    axes[0].set_title("200v pilot — PSNR-oracle config picks by OOD quintile", fontsize=12)
    fig.tight_layout()
    out_path = out_dir / "config_picks_all_quintiles.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return {grid_runs[j]: {f"Q{q}": int(counts_by_q[q][j]) for q in range(1, 6)} for j in range(K)}


def chart_vbench_dim_gain(
    dim_gain: Dict[str, Dict[str, float]],
    out_path: Path,
) -> Optional[str]:
    """Rank dims by relative config-oracle gain (%) vs NO-TTA. Returns winner."""
    dims = [d for d in VBENCH_DIMS if d in dim_gain and np.isfinite(dim_gain[d]["rel_pct"])]
    if not dims:
        print("  [warn] no VBench dims with finite gain — skipping dim-gain chart")
        return None
    dims = sorted(dims, key=lambda d: dim_gain[d]["rel_pct"], reverse=True)
    rel = [dim_gain[d]["rel_pct"] for d in dims]
    raw = [dim_gain[d]["raw"] for d in dims]
    ns = [int(dim_gain[d]["n"]) for d in dims]
    colors = [GREEN if r > 0 else ORANGE for r in rel]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(dims))
    ax.bar(x, rel, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DIM_SHORT.get(d, d) for d in dims], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("config-oracle gain vs NO-TTA (% of NO-TTA mean)")
    ax.set_title("200v pilot — VBench per-dimension oracle headroom (max over 12 configs)", fontsize=11)
    for xi, r, rw, n in zip(x, rel, raw, ns):
        ax.text(xi, r + (0.02 * max(map(abs, rel)) if r >= 0 else -0.02 * max(map(abs, rel))),
                f"{r:+.2f}%\nraw {rw:+.3f}\n(n={n})",
                ha="center", va="bottom" if r >= 0 else "top", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return dims[0]


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="200v-pilot OOD/oracle PNG charts")
    ap.add_argument(
        "--series-root", type=Path,
        default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot",
    )
    ap.add_argument(
        "--baseline-series-root", type=Path,
        default=_REPO / "sweep_experiment/results/panda_1000v_standard",
        help="Where NO-TTA lives when it is not under --series-root (pilot joins "
             "NOTTA from panda_1000v_standard).",
    )
    ap.add_argument(
        "--ood-csv", type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv",
    )
    ap.add_argument(
        "--out-dir", type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-21/pilot_ood_charts",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(args.series_root)
    baseline_runs = discover_runs(args.baseline_series_root) if args.baseline_series_root.exists() else {}
    grid_runs = [r for r in PILOT_GRID_RUN_ORDER if r in runs]
    if not grid_runs:
        raise SystemExit(f"[error] no pilot grid configs under {args.series_root}")
    notta_dir = _resolve_notta_dir(runs, baseline_runs)
    print(f"[info] grid configs: {len(grid_runs)}  NOTTA dir: {notta_dir}")

    quint_map = load_ood_quintiles(args.ood_csv)
    print(f"[info] OOD quintiles loaded for {len(quint_map)} videos from {args.ood_csv}")

    # ---- PSNR ------------------------------------------------------------
    vids, P, notta_psnr = build_psnr(runs, grid_runs, notta_dir)
    quint = np.array([quint_map.get(v, 0) for v in vids], dtype=float)
    quint[quint == 0] = np.nan
    have_psnr = np.any(np.isfinite(P), axis=1) & np.isfinite(notta_psnr) & np.isfinite(quint)
    psnr_oracle = np.where(np.any(np.isfinite(P), axis=1), np.nanmax(np.where(np.isfinite(P), P, -np.inf), axis=1), np.nan)
    psnr_delta = np.where(have_psnr, psnr_oracle - notta_psnr, np.nan)
    print(f"[info] PSNR pool: {int(have_psnr.sum())} videos with config+NOTTA+quintile")

    summary: Dict[str, object] = {
        "n_videos_total": len(vids),
        "n_psnr_pool": int(have_psnr.sum()),
        "grid_runs": grid_runs,
    }

    # Chart 1
    summary["chart1_psnr_oracle_delta_by_quintile"] = chart_quintile_delta(
        psnr_delta, quint,
        title="PSNR oracle Δ vs NO-TTA (dB) · 200v pilot\n(oracle = per-video max over 12 configs)",
        ylabel="mean ΔPSNR (dB)", color=BLUE,
        out_path=args.out_dir / "psnr_oracle_delta_by_ood_quintile.png",
    )

    # Chart 2 (5 PNGs + combined)
    summary["chart2_config_picks"] = chart_config_picks(
        np.where(have_psnr[:, None], P, np.nan), quint, grid_runs, args.out_dir,
    )

    # ---- VBench per dim --------------------------------------------------
    dim_gain: Dict[str, Dict[str, float]] = {}
    dim_delta_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for dim in VBENCH_DIMS:
        D, notta_d = build_vbench_dim(runs, grid_runs, notta_dir, vids, dim)
        have = np.any(np.isfinite(D), axis=1) & np.isfinite(notta_d)
        if int(have.sum()) < 20:
            print(f"  [warn] dim {dim}: only {int(have.sum())} paired videos — skipping")
            continue
        oracle = np.where(np.any(np.isfinite(D), axis=1),
                          np.nanmax(np.where(np.isfinite(D), D, -np.inf), axis=1), np.nan)
        delta = np.where(have, oracle - notta_d, np.nan)
        raw = float(np.nanmean(delta))
        notta_mean = float(np.nanmean(np.where(have, notta_d, np.nan)))
        rel = (raw / notta_mean * 100.0) if abs(notta_mean) > 1e-9 else np.nan
        dim_gain[dim] = {"raw": raw, "rel_pct": rel, "n": int(have.sum()), "notta_mean": notta_mean}
        dim_delta_cache[dim] = (delta, quint)
        print(f"  dim {dim:24s} n={int(have.sum()):4d} raw Δ={raw:+.4f} ({rel:+.2f}% of NO-TTA)")

    summary["vbench_dim_gain"] = dim_gain

    # Chart 3a
    winner = chart_vbench_dim_gain(dim_gain, args.out_dir / "vbench_dim_oracle_gain.png")
    summary["vbench_winner_dim"] = winner

    # Chart 3b
    if winner and winner in dim_delta_cache:
        delta, q = dim_delta_cache[winner]
        summary["chart3b_winner_delta_by_quintile"] = chart_quintile_delta(
            delta, q,
            title=f"{DIM_SHORT.get(winner, winner)} oracle Δ vs NO-TTA · 200v pilot\n"
                  f"(oracle = per-video max over 12 configs)",
            ylabel=f"mean Δ {DIM_SHORT.get(winner, winner)}", color=GREEN,
            out_path=args.out_dir / f"vbench_{winner}_oracle_delta_by_ood_quintile.png",
        )

    (args.out_dir / "chart_data.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll charts + chart_data.json written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
