#!/usr/bin/env python3
"""Render 200v-pilot OOD/oracle charts from the dumped JSON (run LOCALLY).

Pairs with ``dump_pilot_chart_data.py`` (cluster). Reads the pasted JSON and
writes PNGs — no cluster data access required.

Usage (local):
    python3 scripts/render_pilot_charts_from_json.py \
        --json /path/to/pilot_chart_data.json \
        --out-dir charts_out
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIM_SHORT = {
    "subject_consistency": "Subject consistency",
    "background_consistency": "Background consistency",
    "aesthetic_quality": "Aesthetic quality",
    "motion_smoothness": "Motion smoothness",
    "dynamic_degree": "Dynamic degree",
    "imaging_quality": "Imaging quality",
    "temporal_flickering": "Temporal flickering",
}
QUINTILE_LABELS = {1: "Q1\n(most in-dist)", 2: "Q2", 3: "Q3", 4: "Q4", 5: "Q5\n(most OOD)"}
BLUE, GREEN, ORANGE = "#2c6fbb", "#2e8b57", "#d9822b"


def _bar_quintile(stats: dict, *, title: str, ylabel: str, color: str, out_path: Path) -> None:
    qs = stats["quintiles"]
    means = [np.nan if m is None else m for m in stats["mean"]]
    sems = [0.0 if s is None else s for s in stats["sem"]]
    ns = stats["n"]
    pop = stats.get("pop_mean")

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(len(qs))
    ax.bar(x, means, yerr=sems, capsize=4, color=color, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    if pop is not None:
        ax.axhline(pop, color="gray", linestyle="--", linewidth=1.0, label=f"pop. mean = {pop:+.3f}")
        ax.legend(frameon=False, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([QUINTILE_LABELS[q] for q in qs], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    finite = [abs(m) for m in means if np.isfinite(m)]
    off = 0.02 * (max(finite) if finite else 1.0)
    for xi, m, n in zip(x, means, ns):
        if np.isfinite(m):
            ax.text(xi, m + (off if m >= 0 else -off), f"{m:+.3f}\n(n={n})",
                    ha="center", va="bottom" if m >= 0 else "top", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _config_picks(picks: Dict[str, Dict[str, int]], grid_runs: List[str], out_dir: Path,
                  label: str = "200v pilot") -> None:
    x = np.arange(len(grid_runs))
    per_q = {q: [picks.get(f"Q{q}", {}).get(r, 0) for r in grid_runs] for q in range(1, 6)}
    ymax = max((max(v) if v else 0 for v in per_q.values()), default=1)

    for q in range(1, 6):
        counts = per_q[q]
        fig, ax = plt.subplots(figsize=(9, 4.6))
        ax.bar(x, counts, color=BLUE, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(grid_runs, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("videos won (PSNR oracle)")
        ax.set_ylim(0, ymax * 1.18 + 1)
        ax.set_title(f"{label} — config picks, OOD {QUINTILE_LABELS[q].splitlines()[0]} "
                     f"(n={int(sum(counts))})", fontsize=11)
        for xi, c in zip(x, counts):
            if c:
                ax.text(xi, c + 0.02 * ymax + 0.05, str(int(c)), ha="center", va="bottom", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"config_picks_quintile_Q{q}.png", dpi=150)
        plt.close(fig)
        print(f"  wrote {out_dir / f'config_picks_quintile_Q{q}.png'}")

    fig, axes = plt.subplots(5, 1, figsize=(9, 14), sharex=True)
    for q, ax in zip(range(1, 6), axes):
        counts = per_q[q]
        ax.bar(x, counts, color=BLUE, edgecolor="black", linewidth=0.6)
        ax.set_ylabel(f"{QUINTILE_LABELS[q].splitlines()[0]}\n(n={int(sum(counts))})", fontsize=9)
        ax.set_ylim(0, ymax * 1.18 + 1)
        for xi, c in zip(x, counts):
            if c:
                ax.text(xi, c + 0.02 * ymax + 0.05, str(int(c)), ha="center", va="bottom", fontsize=7)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(grid_runs, rotation=45, ha="right", fontsize=8)
    axes[0].set_title(f"{label} — PSNR-oracle config picks by OOD quintile", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "config_picks_all_quintiles.png", dpi=150)
    plt.close(fig)
    print(f"  wrote {out_dir / 'config_picks_all_quintiles.png'}")


def _dim_gain(dim_gain: Dict[str, dict], out_path: Path, label: str = "200v pilot") -> None:
    dims = [d for d, g in dim_gain.items() if g.get("rel_pct") is not None]
    if not dims:
        print("  [warn] no VBench dims with finite gain — skipping dim-gain chart")
        return
    dims = sorted(dims, key=lambda d: dim_gain[d]["rel_pct"], reverse=True)
    rel = [dim_gain[d]["rel_pct"] for d in dims]
    raw = [dim_gain[d]["raw"] for d in dims]
    ns = [dim_gain[d]["n"] for d in dims]
    colors = [GREEN if r > 0 else ORANGE for r in rel]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(dims))
    ax.bar(x, rel, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([DIM_SHORT.get(d, d) for d in dims], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("config-oracle gain vs NO-TTA (% of NO-TTA mean)")
    ax.set_title(f"{label} — VBench per-dimension oracle headroom (max over 12 configs)", fontsize=11)
    span = max((abs(r) for r in rel), default=1.0)
    lo = min(rel + [0.0])
    ax.set_ylim(lo - 0.12 * span, max(rel + [0.0]) + 0.30 * span)  # headroom so labels clear title
    for xi, r, rw, n in zip(x, rel, raw, ns):
        ax.text(xi, r + (0.02 * span if r >= 0 else -0.02 * span),
                f"{r:+.2f}%\nraw {rw:+.3f}\n(n={n})",
                ha="center", va="bottom" if r >= 0 else "top", fontsize=7)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Render 200v-pilot charts from dumped JSON")
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("charts_out"))
    ap.add_argument("--label", type=str, default="200v pilot",
                    help="Label used in chart titles (e.g. '1000v preview (seed-clean)').")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(args.json.read_text(encoding="utf-8"))
    grid_runs = data["meta"]["grid_runs"]
    winner = data.get("winner_dim")
    label = args.label
    print(f"[info] grid={len(grid_runs)} psnr_pool={data['meta'].get('n_psnr_pool')} winner={winner}")

    # Chart 1
    _bar_quintile(
        data["chart1_psnr_delta"],
        title=f"PSNR oracle Δ vs NO-TTA (dB) · {label}\n(oracle = per-video max over 12 configs)",
        ylabel="mean ΔPSNR (dB)", color=BLUE,
        out_path=args.out_dir / "psnr_oracle_delta_by_ood_quintile.png",
    )

    # Chart 2
    _config_picks(data["chart2_config_picks"], grid_runs, args.out_dir, label=label)

    # Chart 3a
    _dim_gain(data["vbench_dim_gain"], args.out_dir / "vbench_dim_oracle_gain.png", label=label)

    # Chart 3b
    if winner and winner in data.get("vbench_dim_delta_by_quintile", {}):
        _bar_quintile(
            data["vbench_dim_delta_by_quintile"][winner],
            title=f"{DIM_SHORT.get(winner, winner)} oracle Δ vs NO-TTA · {label}\n"
                  f"(oracle = per-video max over 12 configs)",
            ylabel=f"mean Δ {DIM_SHORT.get(winner, winner)}", color=GREEN,
            out_path=args.out_dir / f"vbench_{winner}_oracle_delta_by_ood_quintile.png",
        )
    print(f"\nAll charts written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
