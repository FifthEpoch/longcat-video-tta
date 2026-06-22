#!/usr/bin/env python3
"""Generate slide-ready PNGs for TTA gating presentation (4 gates + oracle + H9).

Outputs under:
  sweep_experiment/reports/figures/presentation/

Data sources (prefer real CSVs; fallbacks labeled in FALLBACK_* constants):
  - per_video_gains.csv  → oracle method routing (3-way / 2-way)
  - criteria_correlation → Spearman ρ bars
  - diffusion_ood_scores.csv + gains → H5 quintile ΔPSNR  [NEEDS CLUSTER SCP]
  - panda_ood_budget_pilot merged summaries → H9 live plots  [NEEDS CLUSTER SCP]
  - phase1_oracle_fvd/fvd_summary.json → oracle FVD bars

Usage:
    python scripts/plot_presentation_figures.py

With cluster data synced locally:
    python scripts/plot_presentation_figures.py \\
        --gains-csv sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --budget-series-root sweep_experiment/results/panda_ood_budget_pilot
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    DEFAULT_OOD,
    DEFAULT_SERIES,
    FIXED_ADA_RUN_ID,
    PILOT_GRID_RUN_ORDER,
    discover_runs,
    load_ood_quintiles,
    load_run_psnr,
    oracle_winner,
    parse_run_hparams,
)

DEFAULT_GAINS = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv"
)
DEFAULT_CORR = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation"
)
DEFAULT_CORR_FULL = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/criteria_correlation_full.csv"
)
DEFAULT_FVD = _REPO_ROOT / "sweep_experiment/reports/phase1_oracle_fvd/fvd_summary.json"
DEFAULT_OUT = _REPO_ROOT / "sweep_experiment/reports/figures/presentation"

# ---------------------------------------------------------------------------
# Fallback numbers (hypothesis_outcomes_2026-06-15.md + pilot paste 2026-06-22)
# Replace with live CSVs where noted.
# ---------------------------------------------------------------------------

# H5 — NEEDS diffusion_ood_scores.csv for quintile plot from real data
H5_QUINTILE_DPSNR_FALLBACK = [0.11, 0.05, 0.0, -0.05, -0.12]
H5_RHO = {"ADA": -0.162, "LORA": -0.130, "feature": "mean_diffusion_loss_uncond"}

# H6 — grad norm from bootstrap job 11135260; loss_drop from tier3 (weak)
H6_FEATURES = [
    ("mean_grad_norm_lora", "LoRA grad norm", {"ADA": -0.136, "LORA": -0.141}),
    ("mean_loss_drop_pct", "1-step loss drop", {"ADA": 0.028, "LORA": 0.031}),
    ("loss_var_caption", "Loss variance", {"ADA": 0.024, "LORA": 0.018}),
]

# H1/H7 motion battery (job 11135260 bootstrap point ρ)
MOTION_FEATURES = [
    ("mean_flow", "RAFT mean-flow", {"ADA": -0.061, "LORA": 0.086}),
    ("latent_temporal_l2_mean", "Latent temporal L2", {"ADA": -0.042, "LORA": 0.114}),
    ("dino_temporal_l2_mean", "DINO temporal L2", {"ADA": -0.033, "LORA": 0.109}),
]

# H8 — split across methods
H8_FEATURES = [
    ("rec_err_l1", "VAE rec L1", {"ADA": 0.142, "LORA": 0.0}),
    ("rec_err_lpips", "VAE rec LPIPS", {"ADA": 0.0, "LORA": 0.143}),
]

# Method oracle — verified from per_video_gains.csv N=999 (local 2026-06-22)
ORACLE_METHOD = {
    "NOTTA": {"psnr": 17.930, "fvd": 155.94},
    "ADA": {"psnr": 17.938, "fvd": 156.22},
    "LORA": {"psnr": 17.855, "fvd": 158.85},
    "oracle_3way": {"psnr": 18.156, "fvd": 149.57, "uplift": 0.226, "ci": (0.186, 0.271)},
    "oracle_2way": {"psnr": 18.124, "fvd": None, "uplift": 0.193, "ci": (0.159, 0.232)},
    "winners_3way": {"NOTTA": 345, "ADA": 446, "LORA": 208},
}

# H9 pilot — from cluster adasteer_budget_oracle_pilot.md (2026-06-22 paste)
H9_CONFIG_TABLE: List[Dict[str, object]] = [
    {"run_id": "S2_LR1e3", "steps": 2, "lr": "1e-3", "psnr": 18.052, "ssim": 0.6369, "lpips": 0.3461, "fvd": 325.9, "fid": 62.7},
    {"run_id": "S2_LR5e3", "steps": 2, "lr": "5e-3", "psnr": 18.113, "ssim": 0.6372, "lpips": 0.3452, "fvd": 320.9, "fid": 61.8},
    {"run_id": "S2_LR1e2", "steps": 2, "lr": "1e-2", "psnr": 18.126, "ssim": 0.6390, "lpips": 0.3448, "fvd": 335.6, "fid": 61.6},
    {"run_id": "S5_LR1e3", "steps": 5, "lr": "1e-3", "psnr": 18.105, "ssim": 0.6370, "lpips": 0.3454, "fvd": 317.5, "fid": 62.3},
    {"run_id": "S5_LR5e3", "steps": 5, "lr": "5e-3", "psnr": 18.053, "ssim": 0.6370, "lpips": 0.3466, "fvd": 316.7, "fid": 63.1},
    {"run_id": "S5_LR1e2", "steps": 5, "lr": "1e-2", "psnr": 17.900, "ssim": 0.6330, "lpips": 0.3513, "fvd": 319.7, "fid": 63.6},
    {"run_id": "S10_LR1e3", "steps": 10, "lr": "1e-3", "psnr": 18.086, "ssim": 0.6370, "lpips": 0.3454, "fvd": 316.5, "fid": 61.5},
    {"run_id": "S10_LR5e3", "steps": 10, "lr": "5e-3", "psnr": 17.929, "ssim": 0.6328, "lpips": 0.3506, "fvd": 331.2, "fid": 63.4},
    {"run_id": "S10_LR1e2", "steps": 10, "lr": "1e-2", "psnr": 17.991, "ssim": 0.6332, "lpips": 0.3477, "fvd": 331.4, "fid": 63.8},
    {"run_id": "S20_LR1e3", "steps": 20, "lr": "1e-3", "psnr": 18.022, "ssim": 0.6366, "lpips": 0.3460, "fvd": 318.6, "fid": 62.0},
    {"run_id": "S20_LR5e3", "steps": 20, "lr": "5e-3", "psnr": 17.908, "ssim": 0.6330, "lpips": 0.3506, "fvd": 318.9, "fid": 64.0},
    {"run_id": "S20_LR1e2", "steps": 20, "lr": "1e-2", "psnr": 17.877, "ssim": 0.6262, "lpips": 0.3538, "fvd": 334.3, "fid": 65.3},
    {"run_id": "ORACLE", "steps": "—", "lr": "—", "psnr": 18.779, "ssim": 0.6497, "lpips": 0.3281, "fvd": None, "fid": None},
]

H9_ORACLE_PICKS: List[Tuple[str, int, float]] = [
    ("S20_LR1e2", 54, 5.4),
    ("S10_LR1e2", 42, 4.2),
    ("S2_LR5e3", 13, 1.3),
    ("S20_LR1e3", 13, 1.3),
    ("S20_LR5e3", 12, 1.2),
    ("S2_LR1e3", 12, 1.2),
    ("S5_LR1e2", 11, 1.1),
    ("S5_LR1e3", 9, 0.9),
]

H9_QUINTILE_POLICIES = {
    "Q1": {"fixed": 18.249, "oracle": 19.122, "modal": "S20_LR1e2"},
    "Q2": {"fixed": 18.956, "oracle": 19.466, "modal": "S10_LR1e2"},
    "Q3": {"fixed": 19.242, "oracle": 20.170, "modal": "S20_LR1e2"},
    "Q4": {"fixed": 17.893, "oracle": 18.734, "modal": "S20_LR1e2"},
    "Q5": {"fixed": 15.308, "oracle": 16.404, "modal": "S10_LR1e2"},
}

H9_FIXED_BASELINE_RUN = FIXED_ADA_RUN_ID  # S10_LR5e3 — pilot fixed AdaSteer config


def _finite_values(values: Iterable[object]) -> np.ndarray:
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
    values: Iterable[object],
    *,
    pad_frac: float = 0.12,
    min_pad: float = 0.05,
) -> Tuple[float, float]:
    """Tight y/x limits around data (not from zero) so small metric gaps are visible."""
    arr = _finite_values(values)
    if arr.size == 0:
        return 0.0, 1.0
    lo, hi = float(arr.min()), float(arr.max())
    span = hi - lo
    pad = max(min_pad, span * pad_frac) if span > 1e-9 else min_pad
    return lo - pad, hi + pad


def _add_baseline_hline(
    ax: plt.Axes,
    value: float,
    *,
    label: str = "NOTTA baseline",
) -> None:
    ax.axhline(
        float(value),
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        zorder=0,
        label=label,
    )


def _add_baseline_vline(
    ax: plt.Axes,
    value: float,
    *,
    label: str = "Fixed AdaSteer baseline",
) -> None:
    ax.axvline(
        float(value),
        color="#888888",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        zorder=0,
        label=label,
    )


def _h9_fixed_baseline_psnr(configs: List[Dict[str, object]]) -> Optional[float]:
    for c in configs:
        if c.get("run_id") == H9_FIXED_BASELINE_RUN:
            return float(c["psnr"])
    return None


def _save(fig: plt.Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}")
    return path


def load_correlation_csv(corr_dir: Path) -> Dict[Tuple[str, str], float]:
    path = corr_dir / "correlation_table.csv"
    out: Dict[Tuple[str, str], float] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return out
        methods = [c for c in reader.fieldnames if c.endswith("_rho")]
        for row in reader:
            feat = row.get("feature", "")
            for col in methods:
                m = col.replace("_rho", "")
                v = row.get(col, "")
                if v in ("", None):
                    continue
                try:
                    out[(m, feat)] = float(v)
                except ValueError:
                    pass
    return out


def _rho(corr: Dict[Tuple[str, str], float], feature: str, fb: Dict[str, float]) -> Tuple[float, float]:
    ada = corr.get(("ADA", feature), fb.get("ADA", 0.0))
    lora = corr.get(("LORA_R8_TTA", feature), fb.get("LORA", 0.0))
    return float(ada), float(lora)


def plot_gate_h5_quintile(
    out_dir: Path,
    gains_csv: Path,
    ood_csv: Path,
) -> Path:
    if gains_csv.exists() and ood_csv.exists():
        from scripts.caption_utils import canonical_video_id

        ood_q = load_ood_quintiles(ood_csv)
        buckets: Dict[int, List[float]] = {q: [] for q in range(1, 6)}
        with gains_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vid = canonical_video_id(row.get("video_id", ""))
                q = ood_q.get(vid)
                col = "ADA_dpsnr"
                if vid and q and col in row and row[col]:
                    try:
                        buckets[q].append(float(row[col]))
                    except ValueError:
                        pass
        qmeans = [
            float(np.mean(buckets[q])) if buckets[q] else float("nan")
            for q in range(1, 6)
        ]
        subtitle = "Source: per_video_gains + diffusion_ood_scores"
    else:
        qmeans = H5_QUINTILE_DPSNR_FALLBACK
        subtitle = "FALLBACK — SCP diffusion_ood_scores.csv for live quintiles"

    fig, ax = plt.subplots(figsize=(7, 4.5))
    qs = np.arange(1, 6)
    ax.plot(qs, qmeans, "o-", color="#C44E52", linewidth=2.5, markersize=9)
    ax.axhline(0, color="#888888", linestyle=":", linewidth=1)
    ax.set_xticks(qs)
    ax.set_xticklabels([f"Q{q}\n({'low' if q == 1 else 'high' if q == 5 else 'mid'} OOD)" for q in qs])
    ax.set_xlabel("Diffusion OOD quintile (mean_diffusion_loss_caption)")
    ax.set_ylabel("Mean ΔPSNR vs NOTTA (dB) — AdaSteer")
    ax.set_title("Gate 1 — Model-perceived difficulty (H5)\nHigher OOD → LESS TTA benefit", fontweight="bold")
    ax.text(0.5, -0.20, subtitle, transform=ax.transAxes, ha="center", fontsize=8, color="#666")
    return _save(fig, out_dir, "gate_h5_ood_quintile_dpsnr.png")


def plot_multi_rho_bars(
    out_dir: Path,
    fname: str,
    title: str,
    features: Sequence[Tuple[str, str, Dict[str, float]]],
    corr: Dict[Tuple[str, str], float],
    *,
    predicted_sign: str = "+",
) -> Path:
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4.2), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, (feat_key, label, fb) in zip(axes, features):
        r_ada, r_lora = _rho(corr, feat_key, fb)
        bars = ax.bar(["AdaSteer", "LoRA R8"], [r_ada, r_lora], color=["#4C72B0", "#DD8452"], edgecolor="#333")
        ax.axhline(0, color="#333", linewidth=0.8)
        ax.axhline(0.2, color="#55A868", linestyle="--", linewidth=1)
        ax.axhline(-0.2, color="#55A868", linestyle="--", linewidth=1)
        ax.set_title(label, fontsize=10)
        ax.set_ylim(-0.35, 0.35)
        for bar, r in zip(bars, [r_ada, r_lora]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                r + (0.03 if r >= 0 else -0.06),
                f"{r:+.3f}",
                ha="center",
                va="bottom" if r >= 0 else "top",
                fontsize=9,
            )
    axes[0].set_ylabel("Spearman ρ(ΔPSNR, feature)")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.02)
    if predicted_sign:
        fig.text(0.5, -0.02, f"Predicted: {predicted_sign}", ha="center", fontsize=9, color="#666")
    fig.tight_layout()
    return _save(fig, out_dir, fname)


def compute_oracle_from_gains(gains_csv: Path) -> Dict[str, object]:
    if not gains_csv.exists():
        return ORACLE_METHOD
    rows = list(csv.DictReader(gains_csv.open(newline="", encoding="utf-8")))

    def f(r, k):
        v = r.get(k, "")
        return float(v) if v not in ("", None) else float("nan")

    notta = [f(r, "NOTTA_psnr") for r in rows]
    ada = [f(r, "ADA_psnr") for r in rows]
    lora = [f(r, "LORA_R8_TTA_psnr") for r in rows]
    o3, o2, g3, g2 = [], [], [], []
    winners = {"NOTTA": 0, "ADA": 0, "LORA": 0}
    for r in rows:
        ps3 = {"NOTTA": f(r, "NOTTA_psnr"), "ADA": f(r, "ADA_psnr"), "LORA": f(r, "LORA_R8_TTA_psnr")}
        w3 = max(ps3, key=ps3.get)
        winners[w3] += 1
        o3.append(ps3[w3])
        g3.append(ps3[w3] - ps3["NOTTA"])
        ps2 = {"NOTTA": ps3["NOTTA"], "ADA": ps3["ADA"]}
        o2.append(max(ps2.values()))
        g2.append(max(ps2.values()) - ps3["NOTTA"])

    rng = np.random.default_rng(42)
    n = len(rows)

    def boot_ci(gains):
        if not gains:
            return None, None
        arr = np.asarray(gains, dtype=float)
        boots = [float(rng.choice(arr, size=n, replace=True).mean()) for _ in range(5000)]
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    ci3 = boot_ci(g3)
    ci2 = boot_ci(g2)
    return {
        "NOTTA": {"psnr": float(np.mean(notta))},
        "ADA": {"psnr": float(np.mean(ada))},
        "LORA": {"psnr": float(np.mean(lora))},
        "oracle_3way": {
            "psnr": float(np.mean(o3)),
            "uplift": float(np.mean(o3) - np.mean(notta)),
            "ci": ci3,
        },
        "oracle_2way": {
            "psnr": float(np.mean(o2)),
            "uplift": float(np.mean(o2) - np.mean(notta)),
            "ci": ci2,
        },
        "winners_3way": winners,
    }


def plot_oracle_method(
    out_dir: Path,
    gains_csv: Path,
    fvd_json: Path,
    *,
    two_way: bool = False,
) -> List[Path]:
    paths: List[Path] = []
    stats = compute_oracle_from_gains(gains_csv)

    fvd_data = {}
    if fvd_json.exists():
        with fvd_json.open(encoding="utf-8") as f:
            fvd_data = json.load(f)

    if two_way:
        labels = ["Always\nNOTTA", "Always\nAdaSteer", "Oracle\n(NOTTA|ADA)"]
        psnr_vals = [
            stats["NOTTA"]["psnr"],
            stats["ADA"]["psnr"],
            stats["oracle_2way"]["psnr"],
        ]
        colors = ["#999999", "#4C72B0", "#55A868"]
        tag = "2way"
        uplift = stats["oracle_2way"]["uplift"]
        ci = stats["oracle_2way"].get("ci")
    else:
        labels = ["Always\nNOTTA", "Always\nAdaSteer", "Always\nLoRA", "Oracle\n(3-way)"]
        psnr_vals = [
            stats["NOTTA"]["psnr"],
            stats["ADA"]["psnr"],
            stats["LORA"]["psnr"],
            stats["oracle_3way"]["psnr"],
        ]
        colors = ["#999999", "#4C72B0", "#DD8452", "#55A868"]
        tag = "3way"
        uplift = stats["oracle_3way"]["uplift"]
        ci = stats["oracle_3way"].get("ci")

    notta_psnr = float(stats["NOTTA"]["psnr"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(labels, psnr_vals, color=colors, edgecolor="#333")
    _add_baseline_hline(ax, notta_psnr, label="NOTTA baseline")
    ylo, yhi = _metric_lim(psnr_vals)
    ax.set_ylim(ylo, yhi)
    ax.set_ylabel("Mean PSNR (dB)")
    title = f"Method oracle — {'NOTTA vs AdaSteer' if two_way else 'NOTTA / AdaSteer / LoRA'}"
    ax.set_title(title, fontweight="bold")
    ci_txt = ""
    if ci and ci[0] is not None:
        ci_txt = f"  |  Bootstrap Δ vs NOTTA: {uplift:+.3f} dB [{ci[0]:+.3f}, {ci[1]:+.3f}]"
    ax.text(0.5, -0.18, f"Oracle uplift vs NOTTA: {uplift:+.3f} dB{ci_txt}", transform=ax.transAxes, ha="center", fontsize=9)
    for bar, v in zip(bars, psnr_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, va="bottom")
    fig.subplots_adjust(bottom=0.22)
    paths.append(_save(fig, out_dir, f"oracle_method_{tag}_psnr.png"))

    if not two_way and fvd_data:
        fvd_labels = ["NOTTA", "AdaSteer", "LoRA R8", "Oracle\n(best PSNR)"]
        fvd_vals = [
            fvd_data.get("always_notta", ORACLE_METHOD["NOTTA"]["fvd"]),
            fvd_data.get("always_ada", ORACLE_METHOD["ADA"]["fvd"]),
            fvd_data.get("always_lora", ORACLE_METHOD["LORA"]["fvd"]),
            fvd_data.get("oracle_best_psnr", ORACLE_METHOD["oracle_3way"]["fvd"]),
        ]
        notta_fvd = float(fvd_vals[0])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(fvd_labels, fvd_vals, color=["#999999", "#4C72B0", "#DD8452", "#55A868"], edgecolor="#333")
        _add_baseline_hline(ax, notta_fvd, label="NOTTA baseline")
        ylo, yhi = _metric_lim(fvd_vals)
        ax.set_ylim(ylo, yhi)
        ax.set_ylabel("FVD ↓ better")
        ax.set_title("Method oracle FVD (job 11061632, 14 cond + 14 gen frames)", fontweight="bold")
        ax.text(
            0.5, -0.18,
            f"Oracle FVD {fvd_vals[3]:.1f} vs NOTTA {fvd_vals[0]:.1f} (Δ {fvd_vals[3]-fvd_vals[0]:+.1f})",
            transform=ax.transAxes, ha="center", fontsize=9,
        )
        fig.subplots_adjust(bottom=0.22)
        paths.append(_save(fig, out_dir, "oracle_method_3way_fvd.png"))

    return paths


def plot_h9_config_psnr(out_dir: Path, configs: List[Dict[str, object]]) -> Path:
    grid = [c for c in configs if c["run_id"] != "ORACLE"]
    oracle = next(c for c in configs if c["run_id"] == "ORACLE")
    order = sorted(grid, key=lambda c: -float(c["psnr"]))
    labels = [str(c["run_id"]) for c in order] + ["ORACLE"]
    vals = [float(c["psnr"]) for c in order] + [float(oracle["psnr"])]
    colors = ["#8172B3"] * len(order) + ["#55A868"]
    fixed_psnr = _h9_fixed_baseline_psnr(configs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(labels, vals, color=colors, edgecolor="#333")
    if fixed_psnr is not None:
        _add_baseline_vline(
            ax,
            fixed_psnr,
            label=f"Fixed AdaSteer ({H9_FIXED_BASELINE_RUN})",
        )
    ax.axvline(float(oracle["psnr"]), color="#55A868", linestyle=":", alpha=0.5, label="Per-video oracle")
    ref_vals = list(vals)
    if fixed_psnr is not None:
        ref_vals.append(fixed_psnr)
    xlo, xhi = _metric_lim(ref_vals, pad_frac=0.08, min_pad=0.03)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("Mean PSNR (dB)")
    ax.set_title("H9 — 12-config budget grid + per-video oracle (N=200 pilot)", fontweight="bold")
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    return _save(fig, out_dir, "h9_config_psnr_bar.png")


def plot_h9_pick_frequency(out_dir: Path, picks: List[Tuple[str, int, float]]) -> Path:
    labels = [p[0] for p in picks]
    pcts = [p[2] for p in picks]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, pcts, color="#8172B3", edgecolor="#333")
    ax.set_ylabel("Oracle pick frequency (%)")
    ax.set_xlabel("Grid config (per-video best PSNR)")
    ax.set_title("H9 — Oracle budget picks are sparse (no dominant config)", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.text(
        0.5,
        -0.42,
        "Top: S20_LR1e2 5.4%, S10_LR1e2 4.2% — remainder spread (~80%+ other configs)",
        transform=ax.transAxes,
        ha="center",
        fontsize=8,
        color="#666",
    )
    fig.subplots_adjust(bottom=0.32)
    return _save(fig, out_dir, "h9_oracle_pick_frequency.png")


def plot_h9_quintile_policies(out_dir: Path, quintiles: Dict[str, Dict[str, object]]) -> Path:
    qs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    fixed = [float(quintiles[q]["fixed"]) for q in qs]
    oracle = [float(quintiles[q]["oracle"]) for q in qs]
    x = np.arange(len(qs))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - w / 2, fixed, w, label=f"Fixed AdaSteer ({H9_FIXED_BASELINE_RUN})", color="#4C72B0", edgecolor="#333")
    ax.bar(x + w / 2, oracle, w, label="Oracle-best", color="#55A868", edgecolor="#333")
    fixed_mean = float(np.mean(fixed))
    _add_baseline_hline(
        ax,
        fixed_mean,
        label=f"Mean fixed ({H9_FIXED_BASELINE_RUN})",
    )
    ylo, yhi = _metric_lim(fixed + oracle + [fixed_mean], pad_frac=0.08, min_pad=0.15)
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{q}\n{quintiles[q]['modal']}" for q in qs],
        rotation=35,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Mean PSNR (dB)")
    ax.set_title("H9 — OOD quintile: fixed vs oracle (Q5 rescue +1.10 dB)", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    fig.subplots_adjust(bottom=0.22)
    return _save(fig, out_dir, "h9_ood_quintile_policies.png")


def plot_h9_psnr_fvd_tradeoff(out_dir: Path, configs: List[Dict[str, object]]) -> Path:
    grid = [c for c in configs if c["run_id"] != "ORACLE" and c.get("fvd") is not None]
    fvds = [float(c["fvd"]) for c in grid]
    psnrs = [float(c["psnr"]) for c in grid]
    fixed = next((c for c in grid if c["run_id"] == H9_FIXED_BASELINE_RUN), None)

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(fvds, psnrs, s=80, color="#8172B3", edgecolor="#333", label="Grid configs", zorder=3)
    for c in grid:
        ax.annotate(
            str(c["run_id"]),
            (float(c["fvd"]), float(c["psnr"])),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    if fixed is not None:
        fx, fy = float(fixed["fvd"]), float(fixed["psnr"])
        ax.scatter(
            [fx],
            [fy],
            s=140,
            marker="*",
            color="#888888",
            edgecolor="#333",
            linewidth=0.8,
            label=f"Fixed AdaSteer ({H9_FIXED_BASELINE_RUN})",
            zorder=4,
        )
        _add_baseline_hline(ax, fy, label=f"Fixed PSNR ({H9_FIXED_BASELINE_RUN})")
        _add_baseline_vline(ax, fx, label=f"Fixed FVD ({H9_FIXED_BASELINE_RUN})")
    xlo, xhi = _metric_lim(fvds, pad_frac=0.08, min_pad=2.0)
    ylo, yhi = _metric_lim(psnrs, pad_frac=0.08, min_pad=0.03)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel("FVD ↓ better")
    ax.set_ylabel("PSNR (dB) ↑ better")
    ax.set_title("H9 — PSNR vs FVD tradeoff (12 pilot configs)", fontweight="bold")
    ax.text(
        0.02,
        0.02,
        "Best PSNR (S2_LR1e2) = worst FVD; best FVD ≈ S10_LR1e3",
        transform=ax.transAxes,
        fontsize=9,
        color="#666",
        va="bottom",
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    return _save(fig, out_dir, "h9_psnr_fvd_scatter.png")


def try_live_h9_from_series(series_root: Path, ood_csv: Path) -> Optional[Tuple[List, List, Dict]]:
    """If pilot merged summaries exist locally, recompute pick frequency + quintiles."""
    runs = discover_runs(series_root)
    grid_runs = [r for r in PILOT_GRID_RUN_ORDER if r in runs]
    if len(grid_runs) < 6:
        return None
    psnr_by_run = {rid: load_run_psnr(runs[rid]) for rid in grid_runs}
    all_vids = sorted(set().union(*[set(d.keys()) for d in psnr_by_run.values()]))
    winners: Dict[str, int] = {}
    for vid in all_vids:
        row = {rid: psnr_by_run[rid].get(vid) for rid in grid_runs}
        w = oracle_winner(row, grid_runs)
        if w:
            winners[w] = winners.get(w, 0) + 1
    n = len(all_vids) or 200
    picks = sorted(winners.items(), key=lambda x: -x[1])[:8]
    pick_rows = [(k, v, 100.0 * v / n) for k, v in picks]
    return pick_rows, grid_runs, psnr_by_run


def main() -> int:
    ap = argparse.ArgumentParser(description="Presentation figures for TTA gating deck")
    ap.add_argument("--gains-csv", type=Path, default=DEFAULT_GAINS)
    ap.add_argument("--ood-csv", type=Path, default=DEFAULT_OOD)
    ap.add_argument("--correlation-dir", type=Path, default=DEFAULT_CORR)
    ap.add_argument("--fvd-json", type=Path, default=DEFAULT_FVD)
    ap.add_argument("--budget-series-root", type=Path, default=DEFAULT_SERIES)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out = args.output_dir
    corr = load_correlation_csv(args.correlation_dir)
    written: List[Path] = []

    # Gate slides
    written.append(plot_gate_h5_quintile(out, args.gains_csv, args.ood_csv))
    written.append(plot_multi_rho_bars(
        out, "gate_h5_ood_rho_bars.png",
        "Gate 1 — Diffusion OOD ρ(ΔPSNR) wrong sign (H5)",
        [("mean_diffusion_loss_uncond", "Diffusion loss (uncond)", H5_RHO)],
        corr, predicted_sign="+ (high loss → more gain)",
    ))
    written.append(plot_multi_rho_bars(
        out, "gate_h6_loss_norm_rho_bars.png",
        "Gate 2 — Loss-norm / steep-surface probes (H6)",
        H6_FEATURES, corr, predicted_sign="+",
    ))
    written.append(plot_multi_rho_bars(
        out, "gate_motion_complexity_rho_bars.png",
        "Gate 3 — Visual/temporal complexity (H1 motion + H7)",
        MOTION_FEATURES, corr, predicted_sign="+ (unclear)",
    ))
    written.append(plot_multi_rho_bars(
        out, "gate_h8_vae_recerr_rho_bars.png",
        "Gate 4 — VAE reconstruction observability (H8)",
        H8_FEATURES, corr, predicted_sign="+ (high rec err caps gain?)",
    ))

    # Oracle slides
    written.extend(plot_oracle_method(out, args.gains_csv, args.fvd_json, two_way=False))
    written.extend(plot_oracle_method(out, args.gains_csv, args.fvd_json, two_way=True))

    # H9 slides
    h9_picks = H9_ORACLE_PICKS
    live = try_live_h9_from_series(args.budget_series_root, args.ood_csv)
    if live:
        h9_picks = live[0]
        print("[info] Using live H9 oracle pick frequency from budget pilot series")

    written.append(plot_h9_config_psnr(out, H9_CONFIG_TABLE))
    written.append(plot_h9_pick_frequency(out, h9_picks))
    written.append(plot_h9_quintile_policies(out, H9_QUINTILE_POLICIES))
    written.append(plot_h9_psnr_fvd_tradeoff(out, H9_CONFIG_TABLE))

    print(f"\nGenerated {len(written)} presentation figures under {out}")
    missing = []
    if not args.ood_csv.exists():
        missing.append("diffusion_ood_scores.csv → gate_h5 quintile uses FALLBACK")
    if not args.budget_series_root.exists():
        missing.append("panda_ood_budget_pilot/ → H9 uses pasted fallback tables")
    if missing:
        print("\n[data gaps — SCP from cluster for live plots]")
        for m in missing:
            print(f"  - {m}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
