#!/usr/bin/env python3
"""Step 1: Spearman ρ between NOTTA baseline scores and per-video TTA Δ outcomes.

Cheap CPU analysis — no new GPU jobs. Uses ``per_video_vbench_gains.csv``.

Example:
    python3 scripts/analyze_baseline_outcome_predictors.py \\
        --gains-csv sweep_experiment/reports/per_video_analysis/2026-06-30/vbench_agreement/per_video_vbench_gains.csv \\
        --output-dir sweep_experiment/reports/per_video_analysis/2026-06-30/predictor_transfer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.predictor_analysis_common import (  # noqa: E402
    GATE_METHODS,
    PASS_MIN_METHODS,
    PASS_RHO,
    compute_rho_grid,
    format_rho,
    load_vbench_gains,
    notta_baseline_predictors,
    outcome_specs,
    passes_gate,
    predictor_interp,
    write_rho_csv,
)


def _headline_outcomes():
    return ["psnr", "ssim", "lpips", "aesthetic_quality", "imaging_quality", "vbench_total"]


def build_report(
    video_ids: Sequence[str],
    gains: Dict,
    methods: Sequence[str],
    rho_rows: List[Dict],
) -> str:
    specs = {s.key: s for s in outcome_specs()}
    headline = _headline_outcomes()
    lines: List[str] = []
    lines.append("# NOTTA baseline → TTA Δ outcomes (Spearman ρ)")
    lines.append("")
    lines.append(f"- **Videos:** {len(video_ids)}")
    lines.append(f"- **Pass bar:** |ρ| ≥ {PASS_RHO} on ≥ {PASS_MIN_METHODS} of {', '.join(GATE_METHODS)}")
    lines.append("- **Predictors:** NOTTA PSNR/SSIM/LPIPS + NOTTA VBench++ dims (pre-TTA output quality)")
    lines.append("")

    passes: List[str] = []
    for pred_col, pred_label in notta_baseline_predictors():
        lines.append(f"## {pred_label} (`{pred_col}`)")
        lines.append("")
        lines.append(f"*{predictor_interp(pred_col)}*")
        lines.append("")
        hdr = "| Method | " + " | ".join(
            specs[k].label if k in specs else k for k in headline
        ) + " |"
        lines.append(hdr)
        lines.append("|---|" + "|".join(["---:"] * len(headline)) + "|")
        for method in methods:
            cells = []
            for out_key in headline:
                rho = next(
                    (
                        r["rho"]
                        for r in rho_rows
                        if r["predictor"] == pred_col
                        and r["method"] == method
                        and r["outcome"] == out_key
                    ),
                    None,
                )
                cells.append(format_rho(rho))
            lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
        lines.append("")

    for pred_col, _ in notta_baseline_predictors():
        for out_key in headline:
            ok, hits = passes_gate(rho_rows, pred_col, out_key, methods=GATE_METHODS)
            if ok:
                passes.append(f"{pred_col} → {out_key} ({', '.join(hits)})")

    lines.append("## Pass / fail summary")
    lines.append("")
    if passes:
        lines.append("**Passed pass bar:**")
        for p in sorted(set(passes)):
            lines.append(f"- {p}")
    else:
        lines.append("**No NOTTA baseline predictor clears the deployable pass bar.**")
    lines.append("")
    lines.append("## Reading guide")
    lines.append("")
    lines.append("- Strong |ρ| on **ΔIQ** or **ΔAes** would support routing on baseline perceptual quality.")
    lines.append("- Compare **ADA** vs **LoRA** columns — a gate must generalize across methods.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains-csv", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="*", default=list(GATE_METHODS))
    args = ap.parse_args()

    video_ids, gains, all_methods = load_vbench_gains(args.gains_csv)
    methods = [m for m in args.methods if m in all_methods]
    if not methods:
        methods = list(GATE_METHODS)

    predictors = notta_baseline_predictors()
    pred_arrays = {
        col: np.array([float(gains[vid].get(col, float("nan"))) for vid in video_ids], dtype=float)
        for col, _ in predictors
    }

    rho_rows = compute_rho_grid(
        video_ids, pred_arrays, gains, methods, outcome_specs()
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rho_csv(out_dir / "baseline_outcome_rho.csv", rho_rows)
    report = build_report(video_ids, gains, methods, rho_rows)
    (out_dir / "baseline_outcome_predictors.md").write_text(report, encoding="utf-8")
    print(f"Wrote {out_dir / 'baseline_outcome_predictors.md'} ({len(video_ids)} videos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
