#!/usr/bin/env python3
"""Evaluate OOD skip-gate policies vs always-NOTTA / always-frontier (VBench++ + PSNR).

Policy: if OOD > threshold → NOTTA; else → frontier method (LoRA or retrieval).

Uses existing ``per_video_vbench_gains.csv`` + ``diffusion_ood_scores.csv`` — no GPU.

Example (cluster login or sbatch):
    python3 scripts/eval_ood_skip_gate_policy.py \\
        --gains-csv sweep_experiment/reports/per_video_analysis/2026-06-30/vbench_agreement/per_video_vbench_gains.csv \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --output-dir sweep_experiment/reports/per_video_analysis/2026-06-30/ood_skip_gate
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
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_per_video_vbench_agreement import VBENCH_DIMS  # noqa: E402
from scripts.per_video_metric_store import (  # noqa: E402
    OOD_DEFAULT_COL,
    load_gains_csv,
    load_ood_column,
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

BASELINE = "NOTTA"
FRONTIER_METHODS = ("LORA_R8_TTA", "K5_SIM", "K10_SIM")


def _coerce(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _mean(vals: Sequence[float]) -> Optional[float]:
    arr = np.asarray(vals, dtype=float)
    mask = ~np.isnan(arr)
    if mask.sum() == 0:
        return None
    return float(np.mean(arr[mask]))


def abs_vbench_total(row: Dict[str, float], method: str) -> float:
    vals = [_coerce(row.get(f"{method}_{d}")) for d in VBENCH_DIMS]
    if any(math.isnan(v) for v in vals):
        return float("nan")
    return float(np.mean(vals))


def abs_metric(row: Dict[str, float], method: str, suffix: str) -> float:
    return _coerce(row.get(f"{method}_{suffix}"))


def eval_policy(
    video_ids: Sequence[str],
    rows: Dict[str, Dict[str, float]],
    ood: Dict[str, float],
    *,
    frontier: str,
    threshold: float,
) -> dict:
    psnr: List[float] = []
    vb_tot: List[float] = []
    vb_dims: Dict[str, List[float]] = {d: [] for d in VBENCH_DIMS}
    n_skip = 0
    n_apply = 0

    for vid in video_ids:
        row = rows[vid]
        o = ood.get(vid)
        if o is None or math.isnan(o):
            continue
        use_notta = o > threshold
        m = BASELINE if use_notta else frontier
        if use_notta:
            n_skip += 1
        else:
            n_apply += 1

        p = abs_metric(row, m, "psnr")
        vt = abs_vbench_total(row, m)
        if not math.isnan(p):
            psnr.append(p)
        if not math.isnan(vt):
            vb_tot.append(vt)
        for d in VBENCH_DIMS:
            v = abs_metric(row, m, d)
            if not math.isnan(v):
                vb_dims[d].append(v)

    n = n_skip + n_apply
    return {
        "frontier": frontier,
        "threshold": threshold,
        "n": n,
        "n_skip_notta": n_skip,
        "n_apply_frontier": n_apply,
        "skip_pct": 100.0 * n_skip / max(n, 1),
        "psnr": _mean(psnr),
        "vbench_total": _mean(vb_tot),
        "vbench_dims": {d: _mean(vb_dims[d]) for d in VBENCH_DIMS},
    }


def fixed_always(video_ids: Sequence[str], rows: Dict[str, Dict[str, float]], method: str) -> dict:
    psnr, vb_tot = [], []
    vb_dims: Dict[str, List[float]] = {d: [] for d in VBENCH_DIMS}
    for vid in video_ids:
        row = rows[vid]
        p = abs_metric(row, method, "psnr")
        vt = abs_vbench_total(row, method)
        if not math.isnan(p):
            psnr.append(p)
        if not math.isnan(vt):
            vb_tot.append(vt)
        for d in VBENCH_DIMS:
            v = abs_metric(row, method, d)
            if not math.isnan(v):
                vb_dims[d].append(v)
    return {
        "policy": f"always_{method}",
        "frontier": method,
        "threshold": None,
        "n": len(video_ids),
        "n_skip_notta": len(video_ids) if method == BASELINE else 0,
        "n_apply_frontier": 0 if method == BASELINE else len(video_ids),
        "skip_pct": 100.0 if method == BASELINE else 0.0,
        "psnr": _mean(psnr),
        "vbench_total": _mean(vb_tot),
        "vbench_dims": {d: _mean(vb_dims[d]) for d in VBENCH_DIMS},
    }


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _fmt_delta(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}"


def build_report(results: List[dict], *, ood_col: str) -> str:
    baselines = {r["policy"]: r for r in results if r["policy"].startswith("always_")}
    notta_vb = baselines.get("always_NOTTA", {}).get("vbench_total")
    notta_psnr = baselines.get("always_NOTTA", {}).get("psnr")

    lines: List[str] = []
    lines.append("# OOD skip-gate policy evaluation")
    lines.append("")
    lines.append(
        f"- **Rule:** if `{ood_col}` > τ → NOTTA; else → frontier method"
    )
    lines.append("- **Metrics:** mean absolute PSNR / VBench++ on routed output (N=999 intersection)")
    lines.append(
        "- **Note:** VBench total = mean of 7 dims (IQ on raw 0–100 scale, same as other reports)"
    )
    lines.append("")

    for frontier in FRONTIER_METHODS:
        lines.append(f"## Frontier: `{frontier}`")
        lines.append("")
        lines.append(
            "| Policy | τ (OOD) | skip% | PSNR | VBench total | ΔPSNR vs NOTTA | ΔVBench vs NOTTA | Aes | IQ |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

        section_rows: List[dict] = [baselines["always_NOTTA"]]
        if f"always_{frontier}" in baselines:
            section_rows.append(baselines[f"always_{frontier}"])
        section_rows.extend(
            r for r in results
            if r.get("policy") == f"skip_gate_{frontier}"
        )

        for r in section_rows:
            if r["policy"] == "always_NOTTA":
                pol = "always NOTTA"
                tau = "—"
            elif r["policy"] == f"always_{frontier}":
                pol = f"always `{frontier}`"
                tau = "—"
            else:
                pol = f"skip-gate τ={_fmt(r.get('threshold'), 4)}"
                tau = _fmt(r.get("threshold"), 4)
            dpsnr = None
            dvb = None
            if notta_psnr is not None and r.get("psnr") is not None:
                dpsnr = r["psnr"] - notta_psnr
            if notta_vb is not None and r.get("vbench_total") is not None:
                dvb = r["vbench_total"] - notta_vb
            lines.append(
                f"| {pol} | {tau} | {_fmt(r.get('skip_pct'), 1)}% | "
                f"{_fmt(r.get('psnr'))} | {_fmt(r.get('vbench_total'))} | "
                f"{_fmt_delta(dpsnr)} | {_fmt_delta(dvb)} | "
                f"{_fmt(r.get('vbench_dims', {}).get('aesthetic_quality'))} | "
                f"{_fmt(r.get('vbench_dims', {}).get('imaging_quality'), 1)} |"
            )

        always_f = baselines.get(f"always_{frontier}")
        if always_f and notta_vb is not None and always_f.get("vbench_total") is not None:
            dvb = always_f["vbench_total"] - notta_vb
            dpsnr = (always_f.get("psnr") or 0) - (notta_psnr or 0)
            lines.append("")
            lines.append(
                f"**Always `{frontier}` vs NOTTA:** ΔVBench total {_fmt_delta(dvb)}, "
                f"ΔPSNR {_fmt_delta(dpsnr)}"
            )
        lines.append("")

    lines.append("## Reading guide")
    lines.append("")
    lines.append(
        "- **Good skip-gate:** beats `always {frontier}` on VBench total (or IQ) while matching NOTTA on PSNR/FVD-sensitive videos."
    )
    lines.append(
        "- **τ at high OOD quantile** skips the hardest ~10–30% where frontier gains are smallest (ρ(OOD,ΔAes) ≈ −0.27)."
    )
    lines.append("- FVD for routed policies requires a symlink + `eval_fvd` pass (not computed here).")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="OOD skip-gate policy eval (VBench++ + PSNR)")
    ap.add_argument("--gains-csv", type=Path, required=True)
    ap.add_argument("--ood-csv", type=Path, required=True)
    ap.add_argument("--ood-col", default=OOD_DEFAULT_COL)
    ap.add_argument(
        "--quantiles",
        nargs="*",
        type=float,
        default=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        help="OOD quantiles for threshold sweep",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    gains_rows, _ = load_gains_csv(args.gains_csv)
    ood = load_ood_column(args.ood_csv, args.ood_col)
    video_ids = sorted(set(gains_rows.keys()) & set(ood.keys()))
    if len(video_ids) < 10:
        print(f"ERROR: too few videos ({len(video_ids)})", file=sys.stderr)
        return 2

    ood_vals = np.array([ood[v] for v in video_ids], dtype=float)
    thresholds = sorted({float(np.quantile(ood_vals, q)) for q in args.quantiles})

    results: List[dict] = []
    results.append(fixed_always(video_ids, gains_rows, BASELINE))
    for fm in FRONTIER_METHODS:
        results.append(fixed_always(video_ids, gains_rows, fm))

    for frontier in FRONTIER_METHODS:
        for tau in thresholds:
            r = eval_policy(video_ids, gains_rows, ood, frontier=frontier, threshold=tau)
            r["policy"] = f"skip_gate_{frontier}"
            results.append(r)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ood_skip_gate_policies.csv"
    fieldnames = [
        "policy", "frontier", "threshold", "n", "n_skip_notta", "n_apply_frontier",
        "skip_pct", "psnr", "vbench_total",
        *[f"vbench_{d}" for d in VBENCH_DIMS],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            row = {k: r.get(k) for k in fieldnames if k in r}
            for d in VBENCH_DIMS:
                row[f"vbench_{d}"] = r.get("vbench_dims", {}).get(d)
            w.writerow(row)

    md_path = args.output_dir / "ood_skip_gate_summary.md"
    md_path.write_text(build_report(results, ood_col=args.ood_col), encoding="utf-8")
    print(f"Wrote {csv_path} ({len(video_ids)} videos, {len(results)} policies)")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
