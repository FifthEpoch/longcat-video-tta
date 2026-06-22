#!/usr/bin/env python3
"""Generate standalone PNG slides for failed TTA gating hypotheses (H1–H9).

Reads correlation / gains CSVs when present; falls back to headline numbers
from ``sweep_experiment/reports/hypothesis_outcomes_2026-06-15.md``.

Outputs one PNG per hypothesis under:
  sweep_experiment/reports/figures/hypothesis_failures/H{N}_*.png

Cluster usage (login node, after criteria_correlation job):

    python scripts/plot_hypothesis_failures.py \\
        --gains-csv sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --correlation-dir sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation \\
        --budget-series-root sweep_experiment/results/panda_ood_budget_pilot \\
        --output-dir sweep_experiment/reports/figures/hypothesis_failures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    DEFAULT_OOD,
    DEFAULT_SERIES,
    FIXED_ADA_RUN_ID,
    OOD_COL,
    discover_runs,
    load_ood_quintiles,
    load_run_psnr,
    oracle_winner,
    parse_run_hparams,
)
from scripts.caption_utils import canonical_video_id

DEFAULT_GAINS = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv"
)
DEFAULT_CORR = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation"
)
DEFAULT_OUT = _REPO_ROOT / "sweep_experiment/reports/figures/hypothesis_failures"

# Fallback Spearman ρ when correlation_table.csv is unavailable (from hypothesis_outcomes md).
FALLBACK_RHO: Dict[str, Dict[str, float]] = {
    "H1": {"ADA": -0.069, "LORA_R8_TTA": 0.073, "feature": "mean_flow"},
    "H2": {"ADA": 0.013, "LORA_R8_TTA": -0.088, "feature": "NOTTA PSNR"},
    "H3_words": {"ADA": 0.013, "LORA_R8_TTA": -0.023, "feature": "caption words"},
    "H3_chars": {"ADA": 0.020, "LORA_R8_TTA": -0.046, "feature": "caption chars"},
    "H5": {"ADA": -0.162, "LORA_R8_TTA": -0.130, "feature": "mean_diffusion_loss_caption"},
    "H6": {"ADA": -0.136, "LORA_R8_TTA": -0.141, "feature": "mean_grad_norm_lora"},
    "H7": {"ADA": 0.178, "LORA_R8_TTA": 0.094, "feature": "bpp_png_avg"},
    "H8_l1": {"ADA": 0.142, "LORA_R8_TTA": 0.0, "feature": "rec_err_l1"},
    "H8_lpips": {"ADA": 0.0, "LORA_R8_TTA": 0.143, "feature": "rec_err_lpips"},
}

# H5 quintile ΔPSNR fallback (ADA, from hypothesis_outcomes narrative).
H5_QUINTILE_DPSNR = [0.11, 0.05, 0.0, -0.05, -0.12]

# H9 budget pilot highlights (population PSNR / modal oracle steps).
H9_CONFIG_PSNR = {
    "S2_LR1e2": 18.126,
    "S10_LR5e3": 17.929,
    "S10_LR1e2": 18.0,
}


def _finite_values(values: Sequence[object]) -> np.ndarray:
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(f):
            out.append(f)
    return np.asarray(out, dtype=float)


def _metric_lim(
    values: Sequence[object],
    *,
    pad_frac: float = 0.12,
    min_pad: float = 0.05,
) -> Tuple[float, float]:
    arr = _finite_values(values)
    if arr.size == 0:
        return 0.0, 1.0
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo
    pad = max(min_pad, span * pad_frac) if span > 1e-9 else min_pad
    return lo - pad, hi + pad


def _add_baseline_hline(ax: plt.Axes, value: float, *, label: str = "NOTTA baseline") -> None:
    ax.axhline(
        float(value),
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        zorder=0,
        label=label,
    )


def _add_baseline_vline(ax: plt.Axes, value: float, *, label: str = "Fixed AdaSteer baseline") -> None:
    ax.axvline(
        float(value),
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        zorder=0,
        label=label,
    )


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def load_correlation_csv(corr_dir: Path) -> Dict[Tuple[str, str], float]:
    """Return {(method, feature): rho} from correlation_table.csv."""
    path = corr_dir / "correlation_table.csv"
    out: Dict[Tuple[str, str], float] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        methods = [c for c in reader.fieldnames if c not in ("feature", "tier", "N")]
        for row in reader:
            feat = row.get("feature", "")
            for m in methods:
                v = row.get(m, "")
                if v in ("", None):
                    continue
                try:
                    out[(m, feat)] = float(v)
                except ValueError:
                    pass
    return out


def plot_rho_bars(
    out_dir: Path,
    hyp_id: str,
    title: str,
    feature: str,
    rho_ada: float,
    rho_lora: float,
    *,
    predicted_sign: Optional[str] = None,
    fname: Optional[str] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    methods = ["AdaSteer", "LoRA R8"]
    rhos = [rho_ada, rho_lora]
    colors = ["#4C72B0", "#DD8452"]
    bars = ax.bar(methods, rhos, color=colors, edgecolor="#333333", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.axhline(0.2, color="#55A868", linestyle="--", linewidth=1, label="|ρ|≥0.2 bar")
    ax.axhline(-0.2, color="#55A868", linestyle="--", linewidth=1)
    ax.set_ylabel("Spearman ρ(ΔPSNR, feature)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylim(-0.35, 0.35)
    ax.legend(loc="upper right", fontsize=8)
    subtitle = f"Feature: {feature}"
    if predicted_sign:
        subtitle += f"  |  Predicted: {predicted_sign}"
    ax.text(0.5, -0.22, subtitle, transform=ax.transAxes, ha="center", fontsize=9)
    for bar, r in zip(bars, rhos):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            r + (0.02 if r >= 0 else -0.05),
            f"{r:+.3f}",
            ha="center",
            va="bottom" if r >= 0 else "top",
            fontsize=10,
        )
    fig.text(0.02, 0.02, "Verdict: FAIL (|ρ| < 0.2 on both methods)", fontsize=9, color="#C44E52")
    return _save(fig, out_dir, fname or f"{hyp_id}_rho_bars.png")


def load_gains_quintile_dpsnr(
    gains_csv: Path,
    ood_csv: Path,
    method: str = "ADA",
) -> Tuple[List[int], List[float]]:
    """Mean ΔPSNR by OOD quintile for one method."""
    ood_q = load_ood_quintiles(ood_csv)
    dpsnr_by_vid: Dict[str, float] = {}
    with gains_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = canonical_video_id(row.get("video_id", ""))
            col = f"{method}_dpsnr"
            if not vid or col not in row:
                continue
            try:
                dpsnr_by_vid[vid] = float(row[col])
            except ValueError:
                pass
    buckets: Dict[int, List[float]] = {q: [] for q in range(1, 6)}
    for vid, d in dpsnr_by_vid.items():
        q = ood_q.get(vid)
        if q is not None:
            buckets[q].append(d)
    qs = list(range(1, 6))
    means = [
        float(np.mean(buckets[q])) if buckets[q] else float("nan")
        for q in qs
    ]
    return qs, means


def plot_quintile_dpsnr(
    out_dir: Path,
    hyp_id: str,
    title: str,
    quintile_means: Sequence[float],
    *,
    ylabel: str = "Mean ΔPSNR vs NOTTA (dB)",
    predicted: str = "higher OOD → more gain",
    fname: Optional[str] = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    qs = np.arange(1, 6)
    ax.plot(qs, quintile_means, "o-", color="#C44E52", linewidth=2, markersize=8)
    ax.axhline(0, color="#888888", linestyle=":", linewidth=1)
    ax.set_xticks(qs)
    ax.set_xticklabels([f"Q{q}\n({'low' if q == 1 else 'high' if q == 5 else 'mid'} OOD)" for q in qs])
    ax.set_xlabel("Diffusion OOD quintile")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.text(
        0.5, -0.18,
        f"Predicted: {predicted}  |  Observed: inverted / flat",
        transform=ax.transAxes, ha="center", fontsize=9,
    )
    fig.text(0.02, 0.02, "Verdict: FAIL / Falsified", fontsize=9, color="#C44E52")
    return _save(fig, out_dir, fname or f"{hyp_id}_ood_quintile_dpsnr.png")


def plot_h4_noprompt(out_dir: Path, gains_csv: Path) -> Optional[Path]:
    """H4 inconclusive: population mean Δ with vs without caption."""
    if not gains_csv.exists():
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(
            ["AdaSteer\nw/ caption", "AdaSteer\nno prompt", "LoRA\nw/ caption", "LoRA\nno prompt"],
            [0.008, 0.002, -0.076, -0.065],
            color=["#4C72B0", "#8DA0CB", "#DD8452", "#F5A673"],
            edgecolor="#333",
        )
        _add_baseline_hline(ax, 0.0, label="NOTTA baseline (Δ=0)")
        vals = [0.008, 0.002, -0.076, -0.065]
        ylo, yhi = _metric_lim(vals, min_pad=0.02)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel("Mean ΔPSNR vs NOTTA (dB)")
        ax.set_title("H4 — Caption ablation (fallback numbers)", fontweight="bold")
        fig.text(0.02, 0.02, "Verdict: INCONCLUSIVE", fontsize=9, color="#CCB974")
        return _save(fig, out_dir, "H4_caption_ablation.png")

    stats: Dict[str, List[float]] = {}
    with gains_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for m in ("ADA", "ADA_NOPROMPT", "LORA_R8_TTA", "LORA_R8_TTA_NOPROMPT"):
                col = f"{m}_dpsnr"
                if col in row and row[col] not in ("", None):
                    try:
                        stats.setdefault(m, []).append(float(row[col]))
                    except ValueError:
                        pass
    if not stats:
        return None
    labels, vals = [], []
    mapping = [
        ("ADA", "AdaSteer w/ caption"),
        ("ADA_NOPROMPT", "AdaSteer no prompt"),
        ("LORA_R8_TTA", "LoRA w/ caption"),
        ("LORA_R8_TTA_NOPROMPT", "LoRA no prompt"),
    ]
    for key, lab in mapping:
        if key in stats:
            labels.append(lab)
            vals.append(float(np.mean(stats[key])))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, vals, color=["#4C72B0", "#8DA0CB", "#DD8452", "#F5A673"][: len(vals)], edgecolor="#333")
    _add_baseline_hline(ax, 0.0, label="NOTTA baseline (Δ=0)")
    ylo, yhi = _metric_lim(vals, min_pad=0.02)
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("Mean ΔPSNR vs NOTTA (dB)")
    ax.set_title("H4 — No-caption TTA (mixed, tiny effects)", fontweight="bold")
    fig.text(0.02, 0.02, "Verdict: INCONCLUSIVE", fontsize=9, color="#CCB974")
    return _save(fig, out_dir, "H4_caption_ablation.png")


def plot_h9_budget(
    out_dir: Path,
    series_root: Path,
    ood_csv: Path,
) -> List[Path]:
    paths: List[Path] = []
    runs = discover_runs(series_root)
    grid_runs = sorted(r for r in runs if r.startswith("S"))
    if len(grid_runs) < 2:
        # Fallback schematic
        fig, ax = plt.subplots(figsize=(7, 4))
        configs = list(H9_CONFIG_PSNR.keys())
        config_vals = [H9_CONFIG_PSNR[c] for c in configs]
        ax.bar(configs, config_vals, color="#8172B3", edgecolor="#333")
        fixed_psnr = H9_CONFIG_PSNR.get(FIXED_ADA_RUN_ID)
        if fixed_psnr is not None:
            _add_baseline_hline(ax, fixed_psnr, label=f"Fixed AdaSteer ({FIXED_ADA_RUN_ID})")
        ylo, yhi = _metric_lim(config_vals + ([fixed_psnr] if fixed_psnr is not None else []), min_pad=0.05)
        ax.set_ylim(ylo, yhi)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_ylabel("Mean PSNR (dB)")
        ax.set_title("H9 — Population best ≠ high-OOD rule (pilot fallback)", fontweight="bold")
        fig.text(0.02, 0.02, "Verdict: FAIL — S2_LR1e2 wins population PSNR", fontsize=9, color="#C44E52")
        paths.append(_save(fig, out_dir, "H9_population_best_config.png"))
        return paths

    psnr_by_run = {rid: load_run_psnr(runs[rid]) for rid in grid_runs}
    all_vids = sorted(set().union(*[set(d.keys()) for d in psnr_by_run.values()]))
    ood_q = load_ood_quintiles(ood_csv) if ood_csv.exists() else {}

    # Population mean PSNR per config
    pop_means = {
        rid: float(np.mean(list(d.values()))) if d else float("nan")
        for rid, d in psnr_by_run.items()
    }
    top = sorted(pop_means.items(), key=lambda x: -x[1])[:6]
    top_vals = [t[1] for t in top]
    fixed_psnr = pop_means.get(FIXED_ADA_RUN_ID)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh([t[0] for t in top], top_vals, color="#8172B3", edgecolor="#333")
    if fixed_psnr is not None:
        _add_baseline_vline(ax, fixed_psnr, label=f"Fixed AdaSteer ({FIXED_ADA_RUN_ID})")
    ref_vals = list(top_vals)
    if fixed_psnr is not None:
        ref_vals.append(fixed_psnr)
    xlo, xhi = _metric_lim(ref_vals, pad_frac=0.08, min_pad=0.03)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("Mean PSNR (dB)")
    ax.set_title("H9 — Top grid configs by population PSNR (pilot)", fontweight="bold")
    ax.invert_yaxis()
    if fixed_psnr is not None:
        ax.legend(fontsize=8, loc="lower right")
    paths.append(_save(fig, out_dir, "H9_population_best_config.png"))

    if not ood_q:
        return paths

    # Mean PSNR by OOD quintile for fixed vs oracle vs quintile-adaptive
    quintile_best: Dict[int, str] = {}
    for q in range(1, 6):
        vids_q = [v for v in all_vids if ood_q.get(v) == q]
        if not vids_q:
            continue
        best_counts: Dict[str, int] = {}
        for v in vids_q:
            row = {rid: psnr_by_run[rid].get(v) for rid in grid_runs}
            w = oracle_winner(row, grid_runs)
            if w:
                best_counts[w] = best_counts.get(w, 0) + 1
        if best_counts:
            quintile_best[q] = max(best_counts, key=best_counts.get)

    fixed = FIXED_ADA_RUN_ID if FIXED_ADA_RUN_ID in grid_runs else grid_runs[0]
    series: Dict[str, List[float]] = {
        "Fixed AdaSteer": [],
        "Oracle-best": [],
        "Quintile-adaptive": [],
    }
    qs = list(range(1, 6))
    for q in qs:
        vids_q = [v for v in all_vids if ood_q.get(v) == q]
        if not vids_q:
            for s in series:
                series[s].append(float("nan"))
            continue
        fixed_vals = [psnr_by_run[fixed][v] for v in vids_q if v in psnr_by_run[fixed]]
        oracle_vals = []
        adapt_vals = []
        rid_adapt = quintile_best.get(q)
        for v in vids_q:
            row = {rid: psnr_by_run[rid].get(v) for rid in grid_runs}
            w = oracle_winner(row, grid_runs)
            if w and v in psnr_by_run.get(w, {}):
                oracle_vals.append(psnr_by_run[w][v])
            if rid_adapt and v in psnr_by_run.get(rid_adapt, {}):
                adapt_vals.append(psnr_by_run[rid_adapt][v])
        series["Fixed AdaSteer"].append(float(np.mean(fixed_vals)) if fixed_vals else float("nan"))
        series["Oracle-best"].append(float(np.mean(oracle_vals)) if oracle_vals else float("nan"))
        series["Quintile-adaptive"].append(float(np.mean(adapt_vals)) if adapt_vals else float("nan"))

    fig, ax = plt.subplots(figsize=(7, 4))
    for label, vals, c in zip(
        series.keys(),
        series.values(),
        ["#4C72B0", "#55A868", "#C44E52"],
    ):
        ax.plot(qs, vals, "o-", label=label, linewidth=2, color=c)
    fixed_vals = series.get("Fixed AdaSteer", [])
    fixed_mean = float(np.nanmean(fixed_vals)) if fixed_vals else float("nan")
    if np.isfinite(fixed_mean):
        _add_baseline_hline(ax, fixed_mean, label=f"Mean fixed ({FIXED_ADA_RUN_ID})")
    all_vals = [v for vals in series.values() for v in vals if np.isfinite(v)]
    if np.isfinite(fixed_mean):
        all_vals.append(fixed_mean)
    if all_vals:
        ylo, yhi = _metric_lim(all_vals, pad_frac=0.08, min_pad=0.15)
        ax.set_ylim(ylo, yhi)
    ax.set_xticks(qs)
    ax.set_xlabel("OOD quintile")
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("H9 — No monotonic OOD→budget rule", fontweight="bold")
    ax.legend(fontsize=8)
    fig.text(
        0.02, 0.02,
        "Verdict: FAIL — quintile-adaptive ≪ oracle; modal configs vary",
        fontsize=9, color="#C44E52",
    )
    paths.append(_save(fig, out_dir, "H9_ood_quintile_policies.png"))

    # Modal oracle steps by quintile (show non-monotonicity)
    steps_by_q = []
    for q in qs:
        rid = quintile_best.get(q, "")
        st, _lr = parse_run_hparams(rid)
        steps_by_q.append(st if st is not None else 0)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(qs, steps_by_q, color="#8172B3", edgecolor="#333")
    ax.set_xlabel("OOD quintile")
    ax.set_ylabel("Modal oracle steps")
    ax.set_title("H9 — Modal oracle steps by OOD quintile", fontweight="bold")
    ax.text(0.5, -0.15, "H9 predicts high OOD → MORE steps — not observed", transform=ax.transAxes, ha="center", fontsize=9)
    paths.append(_save(fig, out_dir, "H9_modal_oracle_steps_by_quintile.png"))

    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot failed hypothesis figures for slides")
    ap.add_argument("--gains-csv", type=Path, default=DEFAULT_GAINS)
    ap.add_argument("--ood-csv", type=Path, default=DEFAULT_OOD)
    ap.add_argument("--correlation-dir", type=Path, default=DEFAULT_CORR)
    ap.add_argument("--budget-series-root", type=Path, default=DEFAULT_SERIES)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out = args.output_dir
    corr = load_correlation_csv(args.correlation_dir)
    written: List[Path] = []

    def _rho(method: str, feature: str, fallback_key: str) -> Tuple[float, float]:
        ada = corr.get((method.replace("LoRA R8", "LORA_R8_TTA"), feature))
        if ada is None and method == "AdaSteer":
            ada = corr.get(("ADA", feature))
        lora = corr.get(("LORA_R8_TTA", feature))
        fb = FALLBACK_RHO.get(fallback_key, FALLBACK_RHO.get(feature, {}))
        if ada is None:
            ada = fb.get("ADA", fb.get("AdaSteer", 0.0))
        if lora is None:
            lora = fb.get("LORA_R8_TTA", fb.get("LoRA", 0.0))
        return float(ada or 0.0), float(lora or 0.0)

    # H1 motion
    r_ada, r_lora = _rho("AdaSteer", "mean_flow", "H1")
    written.append(plot_rho_bars(
        out, "H1",
        "H1 — Motion (RAFT mean-flow) does not predict TTA gain",
        "mean_flow", r_ada, r_lora, predicted_sign="+",
    ))

    # H2 baseline PSNR
    r_ada, r_lora = _rho("AdaSteer", "NOTTA_psnr", "H2")
    if (r_ada, r_lora) == (0.0, 0.0):
        r_ada, r_lora = FALLBACK_RHO["H2"]["ADA"], FALLBACK_RHO["H2"]["LORA_R8_TTA"]
    written.append(plot_rho_bars(
        out, "H2",
        "H2 — Baseline PSNR does not predict TTA headroom",
        "NOTTA PSNR", r_ada, r_lora,
    ))

    # H3 caption length (words)
    r_ada, r_lora = FALLBACK_RHO["H3_words"]["ADA"], FALLBACK_RHO["H3_words"]["LORA_R8_TTA"]
    written.append(plot_rho_bars(
        out, "H3",
        "H3 — Caption length uncorrelated with ΔPSNR",
        "caption word count", r_ada, r_lora, fname="H3_caption_length_rho.png",
    ))

    # H4
    p = plot_h4_noprompt(out, args.gains_csv)
    if p:
        written.append(p)

    # H5 OOD quintile
    if args.gains_csv.exists() and args.ood_csv.exists():
        _, qmeans = load_gains_quintile_dpsnr(args.gains_csv, args.ood_csv, "ADA")
    else:
        qmeans = H5_QUINTILE_DPSNR
    written.append(plot_quintile_dpsnr(
        out, "H5",
        "H5 — Higher OOD → LESS ΔPSNR (falsified)",
        qmeans,
        predicted="higher OOD → more TTA benefit",
    ))
    r_ada, r_lora = _rho("AdaSteer", "mean_diffusion_loss_caption", "H5")
    written.append(plot_rho_bars(
        out, "H5",
        "H5 — Diffusion OOD ρ(ΔPSNR) wrong sign",
        "mean_diffusion_loss_caption", r_ada, r_lora,
        predicted_sign="+ (higher loss → more gain)",
        fname="H5_ood_rho_bars.png",
    ))

    # H6 grad norm
    r_ada, r_lora = _rho("AdaSteer", "mean_grad_norm_lora", "H6")
    written.append(plot_rho_bars(
        out, "H6",
        "H6 — LoRA grad-norm probe fails gate bar",
        "mean_grad_norm_lora", r_ada, r_lora, predicted_sign="+",
    ))

    # H7 bpp
    r_ada, r_lora = _rho("AdaSteer", "bpp_png_avg", "H7")
    written.append(plot_rho_bars(
        out, "H7",
        "H7 — PNG bpp weak / single-method signal",
        "bpp_png_avg", r_ada, r_lora,
    ))

    # H8 VAE rec error — two features
    r_ada, _ = FALLBACK_RHO["H8_l1"]["ADA"], 0
    _, r_lora = 0, FALLBACK_RHO["H8_lpips"]["LORA_R8_TTA"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["AdaSteer\n(rec_err_l1)", "LoRA\n(rec_err_lpips)"], [0.142, 0.143], color=["#4C72B0", "#DD8452"], edgecolor="#333")
    ax.axhline(0.2, color="#55A868", linestyle="--", label="|ρ|≥0.2")
    ax.set_ylabel("|Spearman ρ| (approx)")
    ax.set_title("H8 — VAE rec-error proxies below bar", fontweight="bold")
    ax.legend(fontsize=8)
    fig.text(0.02, 0.02, "Verdict: FAIL — split across methods", fontsize=9, color="#C44E52")
    written.append(_save(fig, out, "H8_vae_recerr_rho.png"))

    # H9 budget grid
    written.extend(plot_h9_budget(out, args.budget_series_root, args.ood_csv))

    print(f"\nGenerated {len(written)} figures under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
