#!/usr/bin/env python3
"""Pairwise Spearman correlation heatmaps + key scatter plots.

Joins per-video tables (PSNR/SSIM/LPIPS/VBench Δ, OOD) and optional method-level
FVD/FID from ``merged_summary.json`` (FVD is population-only per method).

Outputs under ``--output-dir``:
  * ``correlation_summary.md``
  * ``heatmap_<method>.png`` — Spearman ρ among per-video outcomes + OOD
  * ``scatter_<x>_vs_<y>_<method>.png`` — key pairs
  * ``method_level_fvd_vs_dpsnr.png`` — ΔPSNR vs ΔFVD across methods (N=methods)

Example:
    python3 scripts/plot_cross_metric_correlations.py \\
        --vbench-gains-csv sweep_experiment/reports/per_video_analysis/2026-06-28/vbench_agreement/per_video_vbench_gains.csv \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --method-dirs NOTTA:sweep_experiment/results/panda_1000v_standard/NOTTA \\
        --method-dirs ADA:sweep_experiment/results/panda_1000v_standard/ADA \\
        --output-dir sweep_experiment/reports/per_video_analysis/2026-06-28/cross_metric_corr
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_adasteer_budget_oracle import load_merged_summary  # noqa: E402
from scripts.analyze_per_video_vbench_agreement import VBENCH_DIMS  # noqa: E402
from scripts.per_video_metric_store import (  # noqa: E402
    OOD_DEFAULT_COL,
    load_or_build_wide_table,
    spearman_rho,
    vbench_total,
)

DIM_SHORT = {
    "subject_consistency": "Subj",
    "background_consistency": "BG",
    "aesthetic_quality": "Aes",
    "motion_smoothness": "Motn",
    "dynamic_degree": "Dyn",
    "imaging_quality": "IQ",
    "temporal_flickering": "Flick",
}

METHODS_DEFAULT = ["ADA", "LORA_R8_TTA", "K5_SIM"]


def _parse_method_dir(s: str) -> Tuple[str, Path]:
    parts = s.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"expected NAME:PATH, got {s!r}")
    return parts[0], Path(parts[1])


def _outcome_columns(method: str) -> List[Tuple[str, str]]:
    """Return (column_key, short_label) for correlation matrix."""
    cols: List[Tuple[str, str]] = [
        (f"ood_{OOD_DEFAULT_COL}", "OOD"),
        (f"{method}_dpsnr", "ΔPSNR"),
        (f"{method}_dssim", "ΔSSIM"),
        (f"{method}_dlpips", "ΔLPIPS"),
    ]
    for d in VBENCH_DIMS:
        cols.append((f"{method}_d{d}", f"Δ{DIM_SHORT[d]}"))
    cols.append((f"{method}_dvbench_total", "ΔVBench"))
    return cols


def _add_vbench_total_columns(wide_rows: Dict[str, Dict[str, float]], methods: Sequence[str]) -> None:
    for vid, row in wide_rows.items():
        base = vbench_total(row, "NOTTA")
        for m in methods:
            vt = vbench_total(row, m)
            if not math.isnan(vt) and not math.isnan(base):
                row[f"{m}_dvbench_total"] = vt - base
            else:
                row[f"{m}_dvbench_total"] = float("nan")


def correlation_matrix(
    wide_rows: Dict[str, Dict[str, float]],
    video_ids: Sequence[str],
    col_specs: Sequence[Tuple[str, str]],
) -> Tuple[List[str], np.ndarray]:
    labels = [lbl for _, lbl in col_specs]
    n = len(col_specs)
    mat = np.full((n, n), np.nan)
    arrays = [wide_rows[vid].get(k, float("nan")) for k, _ in col_specs]
    vecs = [
        np.array([wide_rows[vid].get(k, float("nan")) for vid in video_ids], dtype=float)
        for k, _ in col_specs
    ]
    for i in range(n):
        for j in range(n):
            if i == j:
                mat[i, j] = 1.0
                continue
            rho = spearman_rho(vecs[i], vecs[j])
            mat[i, j] = rho if rho is not None else float("nan")
    return labels, mat


def plot_heatmap(
    path: Path,
    labels: Sequence[str],
    mat: np.ndarray,
    *,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.65), max(5, n * 0.55)))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if not math.isnan(v):
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.5 else "black")
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman ρ")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = ~(np.isnan(xs) | np.isnan(ys))
    fig, ax = plt.subplots(figsize=(5, 4))
    if mask.sum() >= 2:
        ax.scatter(xs[mask], ys[mask], s=8, alpha=0.35, linewidths=0)
        rho = spearman_rho(xs, ys)
        rho_s = f"{rho:+.3f}" if rho is not None else "n/a"
        ax.set_title(f"{title}\nSpearman ρ = {rho_s}", fontsize=10)
    else:
        ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="#999", lw=0.5)
    ax.axvline(0, color="#999", lw=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def method_level_fvd_plot(
    path: Path,
    method_dirs: Sequence[Tuple[str, Path]],
    *,
    baseline: str = "NOTTA",
) -> Optional[List[dict]]:
    """Scatter ΔPSNR vs ΔFVD across methods (not per-video)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: List[dict] = []
    base_m = load_merged_summary(next(p for n, p in method_dirs if n == baseline))
    base_psnr = float(base_m.get("psnr") or float("nan"))
    base_fvd = float(base_m.get("fvd") or float("nan"))

    for name, p in method_dirs:
        m = load_merged_summary(p)
        if not m:
            continue
        psnr = float(m.get("psnr") or float("nan"))
        fvd = float(m.get("fvd") or float("nan"))
        rows.append({
            "method": name,
            "dpsnr": psnr - base_psnr if not math.isnan(psnr) else float("nan"),
            "dfvd": fvd - base_fvd if not math.isnan(fvd) else float("nan"),
            "psnr": psnr,
            "fvd": fvd,
        })

    if len(rows) < 2:
        return None

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for r in rows:
        if math.isnan(r["dpsnr"]) or math.isnan(r["dfvd"]):
            continue
        ax.scatter(r["dpsnr"], r["dfvd"], s=60)
        ax.annotate(r["method"], (r["dpsnr"], r["dfvd"]), fontsize=8, xytext=(4, 4),
                    textcoords="offset points")
    ax.set_xlabel("Population ΔPSNR vs NOTTA (dB)")
    ax.set_ylabel("Population ΔFVD vs NOTTA")
    ax.set_title("Method-level: ΔPSNR vs ΔFVD\n(each point = one run config, not per-video)")
    ax.axhline(0, color="#999", lw=0.5)
    ax.axvline(0, color="#999", lw=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return rows


def build_markdown(
    method_results: Dict[str, Tuple[List[str], np.ndarray]],
    method_fvd_rows: Optional[List[dict]],
    n_videos: int,
) -> str:
    lines = [
        "# Cross-metric Spearman correlations",
        "",
        f"- **Per-video N:** {n_videos}",
        "- **FVD:** population-only; see ``method_level_fvd_vs_dpsnr.png`` for method-level ΔPSNR vs ΔFVD.",
        "",
    ]
    for method, (labels, mat) in method_results.items():
        lines.append(f"## `{method}` heatmap")
        lines.append("")
        lines.append(f"See ``heatmap_{method}.png``.")
        lines.append("")
        lines.append("### Key pairs")
        lines.append("")
        idx = {lbl: i for i, lbl in enumerate(labels)}

        def _rho(a: str, b: str) -> str:
            if a not in idx or b not in idx:
                return "n/a"
            v = mat[idx[a], idx[b]]
            return f"{v:+.3f}" if not math.isnan(v) else "n/a"

        lines.append("| Pair | ρ |")
        lines.append("|---|---:|")
        pairs = [
            ("OOD", "ΔPSNR"),
            ("OOD", "ΔAes"),
            ("OOD", "ΔIQ"),
            ("OOD", "ΔVBench"),
            ("ΔPSNR", "ΔAes"),
            ("ΔPSNR", "ΔIQ"),
            ("ΔPSNR", "ΔVBench"),
            ("ΔAes", "ΔIQ"),
        ]
        for a, b in pairs:
            lines.append(f"| {a} vs {b} | {_rho(a, b)} |")
        lines.append("")

    if method_fvd_rows:
        lines.append("## Method-level ΔFVD vs ΔPSNR")
        lines.append("")
        lines.append("| Method | ΔPSNR | ΔFVD |")
        lines.append("|---|---:|---:|")
        for r in method_fvd_rows:
            lines.append(
                f"| {r['method']} | {r['dpsnr']:+.3f} | {r['dfvd']:+.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vbench-gains-csv", type=Path, required=True)
    ap.add_argument("--ood-csv", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--methods", nargs="*", default=METHODS_DEFAULT)
    ap.add_argument("--method-dirs", action="append", type=_parse_method_dir, default=[])
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    wide = load_or_build_wide_table(
        gains_csv=args.vbench_gains_csv,
        ood_csv=args.ood_csv,
        cache_dir=args.cache_dir,
    )
    _add_vbench_total_columns(wide.rows, args.methods)

    method_results: Dict[str, Tuple[List[str], np.ndarray]] = {}
    scatter_pairs = [
        ("ood_" + OOD_DEFAULT_COL, "ΔPSNR", "OOD caption loss", "ΔPSNR (dB)"),
        ("ood_" + OOD_DEFAULT_COL, "ΔAes", "OOD caption loss", "ΔAesthetic"),
        ("ood_" + OOD_DEFAULT_COL, "ΔIQ", "OOD caption loss", "ΔImaging quality"),
        ("ΔPSNR", "ΔAes", "ΔPSNR (dB)", "ΔAesthetic"),
        ("ΔPSNR", "ΔVBench", "ΔPSNR (dB)", "ΔVBench total"),
        ("ΔAes", "ΔIQ", "ΔAesthetic", "ΔImaging quality"),
    ]

    for method in args.methods:
        if method not in wide.methods:
            continue
        specs = _outcome_columns(method)
        labels, mat = correlation_matrix(wide.rows, wide.video_ids, specs)
        method_results[method] = (labels, mat)
        plot_heatmap(
            args.output_dir / f"heatmap_{method}.png",
            labels,
            mat,
            title=f"Spearman ρ — {method} vs NOTTA (N={len(wide.video_ids)})",
        )
        idx = {lbl: i for i, lbl in enumerate(labels)}
        col_by_label = {lbl: key for key, lbl in specs}
        for xl, yl, xname, yname in scatter_pairs:
            if xl not in idx or yl not in idx:
                continue
            xkey = col_by_label[xl]
            ykey = col_by_label[yl]
            xs = wide.column(xkey)
            ys = wide.column(ykey)
            safe = f"{xl}_vs_{yl}_{method}".replace("Δ", "d").replace(" ", "_")
            plot_scatter(
                args.output_dir / f"scatter_{safe}.png",
                xs, ys,
                xlabel=xname,
                ylabel=yname,
                title=f"{method}: {xl} vs {yl}",
            )

    fvd_rows = None
    if args.method_dirs:
        fvd_rows = method_level_fvd_plot(
            args.output_dir / "method_level_fvd_vs_dpsnr.png",
            args.method_dirs,
        )

    md = build_markdown(method_results, fvd_rows, len(wide.video_ids))
    out_md = args.output_dir / "correlation_summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"Wrote {args.output_dir} ({len(method_results)} heatmaps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
