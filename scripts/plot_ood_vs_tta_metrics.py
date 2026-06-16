#!/usr/bin/env python3
"""Plot diffusion OOD score vs TTA gain and baseline PSNR (presentation slides).

Joins ``per_video_gains.csv`` (from ``analyze_per_video_tta_gain.py``) with
``diffusion_ood_scores.csv`` (from ``compute_diffusion_ood_score.py``) on
canonical ``video_id`` (``panda_XXXX``). Uses ``mean_diffusion_loss_caption``
as the OOD axis — same column as ``analyze_oracle_winner_characteristics.py``.

Emits under ``--output-dir``:
  * ``ood_vs_delta_psnr_scatter.png``   — OOD vs per-video ΔPSNR (AdaSteer, LoRA)
  * ``ood_vs_delta_psnr_quintile.png``  — quintile-binned mean ΔPSNR + 95% CI
  * ``ood_vs_baseline_psnr_scatter.png`` — OOD vs NOTTA baseline PSNR
  * ``ood_vs_tta_metrics_summary.md``   — Spearman ρ + sample sizes

Regenerate ``per_video_gains.csv`` at N=999 (NOTTA ∩ ADA ∩ LoRA only):

    python3 scripts/analyze_per_video_tta_gain.py \\
        --series-path sweep_experiment/results/panda_1000v_standard \\
        --methods NOTTA ADA LORA_R8_TTA \\
        --dynamicness-json datasets/panda_1000_480p/dynamic_degree.json \\
        --captions-csv datasets/panda_1000_480p/metadata.csv \\
        --output-dir sweep_experiment/reports/per_video_analysis/2026-06-09

If ``per_video_gains.csv`` was overwritten with extra methods (e.g. TinyLoRA
variants), the intersection shrinks; pass ``--methods`` as above to restore
N=999 for the oracle trio.

Cluster usage (login node; requires OOD CSV from Stage 1b):

    python3 scripts/plot_ood_vs_tta_metrics.py \\
        --gains-csv sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --output-dir sweep_experiment/reports/per_video_analysis/2026-06-09

Dependencies: numpy, matplotlib (no scipy / pandas).
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_GAINS = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv"
)
DEFAULT_OOD = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv"
)
DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09"
)

OOD_COL = "mean_diffusion_loss_caption"
BASELINE = "NOTTA"
DEFAULT_TTA_METHODS = ("ADA", "LORA_R8_TTA")
METHOD_LABELS = {
    "ADA": "AdaSteer",
    "LORA_R8_TTA": "LoRA",
}


def _coerce(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    return x


def load_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def pearson_r(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3:
        return None
    x = xs[mask].astype(np.float64)
    y = ys[mask].astype(np.float64)
    sx, sy = x - x.mean(), y - y.mean()
    den = math.sqrt(float((sx * sx).sum()) * float((sy * sy).sum()))
    if den <= 0:
        return None
    return float((sx * sy).sum() / den)


def spearman_rho(xs: np.ndarray, ys: np.ndarray) -> Optional[float]:
    mask = ~(np.isnan(xs) | np.isnan(ys))
    if mask.sum() < 3:
        return None

    def _ranks(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(a.size, dtype=np.float64)
        uniq, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            sum_ranks = np.zeros(uniq.size, dtype=np.float64)
            np.add.at(sum_ranks, inv, ranks)
            avg_ranks = sum_ranks / counts
            ranks = avg_ranks[inv]
        return ranks

    return pearson_r(_ranks(xs[mask]), _ranks(ys[mask]))


def linear_fit(x: np.ndarray, y: np.ndarray) -> Optional[Tuple[float, float]]:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 2:
        return None
    xs = x[mask].astype(np.float64)
    ys = y[mask].astype(np.float64)
    x_mean, y_mean = xs.mean(), ys.mean()
    num = float(((xs - x_mean) * (ys - y_mean)).sum())
    den = float(((xs - x_mean) ** 2).sum())
    if den <= 0:
        return None
    slope = num / den
    return slope, y_mean - slope * x_mean


def quantile_bin(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    bin_idx = np.full(values.shape, -1, dtype=int)
    mask = ~np.isnan(values)
    if not mask.any():
        return bin_idx, np.array([])
    v = values[mask]
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        bin_idx[mask] = 0
        return bin_idx, edges
    idx = np.clip(np.searchsorted(edges[1:-1], v, side="right"), 0, edges.size - 2)
    bin_idx[mask] = idx
    return bin_idx, edges


def join_gains_ood(
    gains_rows: List[dict],
    ood_rows: List[dict],
    ood_col: str,
) -> Tuple[List[str], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    ood_map: Dict[str, float] = {}
    for r in ood_rows:
        vid = (r.get("video_id") or "").strip()
        if vid:
            ood_map[vid] = _coerce(r.get(ood_col))

    video_ids: List[str] = []
    ood_vals: List[float] = []
    baseline_psnr: List[float] = []
    delta_by_method: Dict[str, List[float]] = {}

    for g in gains_rows:
        vid = (g.get("video_id") or "").strip()
        if not vid or vid not in ood_map:
            continue
        ood_v = ood_map[vid]
        if math.isnan(ood_v):
            continue
        video_ids.append(vid)
        ood_vals.append(ood_v)
        baseline_psnr.append(_coerce(g.get(f"{BASELINE}_psnr")))
        for col in g:
            if col.endswith("_dpsnr"):
                m = col[: -len("_dpsnr")]
                delta_by_method.setdefault(m, []).append(_coerce(g.get(col)))

    return (
        video_ids,
        np.asarray(ood_vals, dtype=float),
        np.asarray(baseline_psnr, dtype=float),
        {m: np.asarray(v, dtype=float) for m, v in delta_by_method.items()},
    )


def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    return plt


def _method_label(name: str) -> str:
    return METHOD_LABELS.get(name, name)


def plot_ood_vs_delta_scatter(
    plt,
    out_path: Path,
    ood: np.ndarray,
    methods: Sequence[str],
    delta_by_method: Dict[str, np.ndarray],
    ood_col: str,
    title: str = "",
):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    cmap = plt.get_cmap("tab10")

    for i, m in enumerate(methods):
        d = delta_by_method[m]
        mask = ~(np.isnan(ood) | np.isnan(d))
        x, y = ood[mask], d[mask]
        color = cmap(i % 10)
        label = _method_label(m)
        ax.scatter(x, y, s=10, alpha=0.35, color=color, edgecolor="none", label=label)
        fit = linear_fit(x, y)
        rho = spearman_rho(ood, d)
        r = pearson_r(ood, d)
        if fit is not None and x.size >= 2:
            slope, intercept = fit
            xs_line = np.linspace(float(x.min()), float(x.max()), 64)
            stat = (
                f"ρ={rho:+.3f}  r={r:+.3f}"
                if rho is not None and r is not None
                else f"slope={slope:+.3f}"
            )
            ax.plot(
                xs_line,
                slope * xs_line + intercept,
                color=color,
                linewidth=1.4,
                label=f"{label} fit ({stat})",
            )

    ax.set_xlabel(f"diffusion OOD ({ood_col})")
    ax.set_ylabel(r"per-video $\Delta$PSNR vs NOTTA (dB)")
    ax.set_title(title or "ΔPSNR vs diffusion OOD (scatter)")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ood_vs_delta_quintile(
    plt,
    out_path: Path,
    ood: np.ndarray,
    methods: Sequence[str],
    delta_by_method: Dict[str, np.ndarray],
    ood_col: str,
    n_bins: int = 5,
    title: str = "",
):
    bin_idx, _ = quantile_bin(ood, n_bins)
    n_bins_eff = max(int(bin_idx.max()) + 1 if (bin_idx >= 0).any() else 0, 1)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.axhline(0.0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    cmap = plt.get_cmap("tab10")
    per_bin_n = [0] * n_bins_eff
    per_bin_center = [float("nan")] * n_bins_eff

    for i, m in enumerate(methods):
        d = delta_by_method[m]
        means = np.full(n_bins_eff, np.nan)
        sems = np.full(n_bins_eff, np.nan)
        centers = np.full(n_bins_eff, np.nan)
        for b in range(n_bins_eff):
            mask = (bin_idx == b) & ~np.isnan(d) & ~np.isnan(ood)
            n = int(mask.sum())
            if n == 0:
                continue
            ys = d[mask]
            means[b] = float(ys.mean())
            sems[b] = float(ys.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0
            centers[b] = float(ood[mask].mean())
            per_bin_n[b] = max(per_bin_n[b], n)
            per_bin_center[b] = centers[b]
        ax.errorbar(
            centers,
            means,
            yerr=1.96 * sems,
            marker="o",
            capsize=2,
            linewidth=1.6,
            label=_method_label(m),
            color=cmap(i % 10),
        )

    annot_y = ax.get_ylim()[0]
    for b in range(n_bins_eff):
        if per_bin_n[b] > 0 and not math.isnan(per_bin_center[b]):
            ax.annotate(
                f"n={per_bin_n[b]}",
                (per_bin_center[b], annot_y),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=7,
                color="grey",
            )

    avg_n = int(np.mean([n for n in per_bin_n if n > 0])) if any(per_bin_n) else 0
    ax.set_xlabel(f"diffusion OOD ({ood_col}; quintile-binned, n≈{avg_n}/bin)")
    ax.set_ylabel(r"mean $\Delta$PSNR vs NOTTA (dB; 95% CI)")
    ax.set_title(title or "ΔPSNR vs diffusion OOD (quintiles)")
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_ood_vs_baseline_psnr(
    plt,
    out_path: Path,
    ood: np.ndarray,
    baseline_psnr: np.ndarray,
    ood_col: str,
    title: str = "",
):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    mask = ~(np.isnan(ood) | np.isnan(baseline_psnr))
    x, y = ood[mask], baseline_psnr[mask]
    ax.scatter(x, y, s=10, alpha=0.35, color="tab:blue", edgecolor="none")
    fit = linear_fit(x, y)
    rho = spearman_rho(ood, baseline_psnr)
    r = pearson_r(ood, baseline_psnr)
    if fit is not None and x.size >= 2:
        slope, intercept = fit
        xs_line = np.linspace(float(x.min()), float(x.max()), 64)
        stat = (
            f"ρ={rho:+.3f}  r={r:+.3f}"
            if rho is not None and r is not None
            else f"slope={slope:+.3f}"
        )
        ax.plot(
            xs_line,
            slope * xs_line + intercept,
            color="tab:red",
            linewidth=1.4,
            label=f"LS fit ({stat})",
        )
        ax.legend(loc="best")
    ax.set_xlabel(f"diffusion OOD ({ood_col})")
    ax.set_ylabel(f"{BASELINE} baseline PSNR (dB)")
    ax.set_title(title or f"Baseline PSNR vs diffusion OOD (N={int(mask.sum())})")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary_md(
    out_path: Path,
    n_joined: int,
    n_gains: int,
    n_ood: int,
    ood_col: str,
    methods: Sequence[str],
    ood: np.ndarray,
    baseline_psnr: np.ndarray,
    delta_by_method: Dict[str, np.ndarray],
    gains_path: Path,
    ood_path: Path,
) -> None:
    lines = [
        "# OOD vs TTA metrics (presentation summary)",
        "",
        f"- Gains CSV: `{gains_path}` ({n_gains} rows)",
        f"- OOD CSV: `{ood_path}` ({n_ood} rows)",
        f"- Joined on `video_id` with finite `{ood_col}`: **N={n_joined}**",
        f"- OOD column: `{ood_col}` (caption-conditioned flow-matching MSE)",
        "",
        "## Spearman ρ (OOD vs metric)",
        "",
        "| metric | N | Spearman ρ | Pearson r |",
        "|---|---:|---:|---:|",
    ]

    rho_b = spearman_rho(ood, baseline_psnr)
    r_b = pearson_r(ood, baseline_psnr)
    n_b = int((~(np.isnan(ood) | np.isnan(baseline_psnr))).sum())
    lines.append(
        f"| `{BASELINE}_psnr` (baseline) | {n_b} | "
        f"{rho_b:+.3f} | {r_b:+.3f} |"
        if rho_b is not None and r_b is not None
        else f"| `{BASELINE}_psnr` (baseline) | {n_b} | n/a | n/a |"
    )

    for m in methods:
        d = delta_by_method[m]
        rho = spearman_rho(ood, d)
        r = pearson_r(ood, d)
        n = int((~(np.isnan(ood) | np.isnan(d))).sum())
        label = _method_label(m)
        if rho is not None and r is not None:
            lines.append(f"| `{m}` ΔPSNR ({label}) | {n} | {rho:+.3f} | {r:+.3f} |")
        else:
            lines.append(f"| `{m}` ΔPSNR ({label}) | {n} | n/a | n/a |")

    lines += [
        "",
        "## Interpretation (exploratory)",
        "",
        "Higher `mean_diffusion_loss_caption` ⇒ the base LongCat-Video model is "
        "more 'surprised' by the visible frames (out-of-distribution). Positive "
        "ρ(ΔPSNR, OOD) would support the hypothesis that TTA helps more on OOD "
        "videos; ρ(baseline PSNR, OOD) captures whether OOD correlates with "
        "absolute reconstruction difficulty under NOTTA.",
        "",
        "## Output figures",
        "",
        "- `ood_vs_delta_psnr_scatter.png` — per-video scatter + LS fit",
        "- `ood_vs_delta_psnr_quintile.png` — OOD quintile means with 95% CI",
        "- `ood_vs_baseline_psnr_scatter.png` — OOD vs NOTTA PSNR",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot diffusion OOD vs TTA ΔPSNR and baseline PSNR"
    )
    ap.add_argument("--gains-csv", type=Path, default=DEFAULT_GAINS)
    ap.add_argument("--ood-csv", type=Path, default=DEFAULT_OOD)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--ood-col",
        default=OOD_COL,
        help=f"OOD score column (default: {OOD_COL})",
    )
    ap.add_argument(
        "--tta-methods",
        nargs="+",
        default=list(DEFAULT_TTA_METHODS),
        help="TTA methods to plot (must have <METHOD>_dpsnr in gains CSV)",
    )
    ap.add_argument("--n-bins", type=int, default=5, help="OOD quintile bins")
    ap.add_argument(
        "--title-suffix",
        default="Panda 1000v",
        help="Suffix appended to plot titles",
    )
    args = ap.parse_args()

    if not args.gains_csv.exists():
        print(f"[error] gains CSV not found: {args.gains_csv}", file=sys.stderr)
        return 2
    if not args.ood_csv.exists():
        print(f"[error] OOD CSV not found: {args.ood_csv}", file=sys.stderr)
        print(
            "  Generate via scripts/compute_diffusion_ood_score.py (Stage 1b) or "
            "scripts/sbatch/submit_per_video_feature_pipeline.sh",
            file=sys.stderr,
        )
        return 2

    gains_rows = load_csv_rows(args.gains_csv)
    ood_rows = load_csv_rows(args.ood_csv)

    if args.ood_col not in (ood_rows[0] if ood_rows else {}):
        print(
            f"[error] OOD column {args.ood_col!r} not in {args.ood_csv}",
            file=sys.stderr,
        )
        return 2

    video_ids, ood, baseline_psnr, delta_all = join_gains_ood(
        gains_rows, ood_rows, args.ood_col,
    )
    if not video_ids:
        print("[error] empty join — check video_id overlap and OOD column", file=sys.stderr)
        return 2

    methods: List[str] = []
    delta_by_method: Dict[str, np.ndarray] = {}
    for m in args.tta_methods:
        col = f"{m}_dpsnr"
        if m not in delta_all:
            print(f"[warn] {col} not in gains CSV; skipping {m}", file=sys.stderr)
            continue
        methods.append(m)
        delta_by_method[m] = delta_all[m]

    if not methods:
        print("[error] no TTA methods with ΔPSNR columns found", file=sys.stderr)
        return 2

    n = len(video_ids)
    suffix = f" — {args.title_suffix} (N={n})"
    plt = _setup_matplotlib()
    out_dir = args.output_dir

    scatter_path = out_dir / "ood_vs_delta_psnr_scatter.png"
    quintile_path = out_dir / "ood_vs_delta_psnr_quintile.png"
    baseline_path = out_dir / "ood_vs_baseline_psnr_scatter.png"
    summary_path = out_dir / "ood_vs_tta_metrics_summary.md"

    plot_ood_vs_delta_scatter(
        plt,
        scatter_path,
        ood,
        methods,
        delta_by_method,
        args.ood_col,
        title=f"ΔPSNR vs diffusion OOD (scatter){suffix}",
    )
    plot_ood_vs_delta_quintile(
        plt,
        quintile_path,
        ood,
        methods,
        delta_by_method,
        args.ood_col,
        n_bins=args.n_bins,
        title=f"ΔPSNR vs diffusion OOD (quintiles){suffix}",
    )
    plot_ood_vs_baseline_psnr(
        plt,
        baseline_path,
        ood,
        baseline_psnr,
        args.ood_col,
        title=f"NOTTA PSNR vs diffusion OOD{suffix}",
    )
    write_summary_md(
        summary_path,
        n,
        len(gains_rows),
        len(ood_rows),
        args.ood_col,
        methods,
        ood,
        baseline_psnr,
        delta_by_method,
        args.gains_csv,
        args.ood_csv,
    )

    print(f"Joined N={n} videos ({args.ood_col})")
    for m in methods:
        rho = spearman_rho(ood, delta_by_method[m])
        print(f"  ρ(OOD, {m} ΔPSNR) = {rho:+.3f}" if rho is not None else f"  {m}: n/a")
    rho_b = spearman_rho(ood, baseline_psnr)
    print(f"  ρ(OOD, {BASELINE} PSNR) = {rho_b:+.3f}" if rho_b is not None else "  baseline: n/a")
    print(f"\nWrote {scatter_path}")
    print(f"Wrote {quintile_path}")
    print(f"Wrote {baseline_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
