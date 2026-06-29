#!/usr/bin/env python3
"""Spearman ρ between per-video VBench++ Δ (vs NOTTA) and Phase-0 predictors.

Joins ``per_video_vbench_gains.csv`` (from ``analyze_per_video_vbench_agreement.py``)
with the feature / OOD / Tier-3 probe CSVs used by ``correlate_tta_gain_with_features.py``.

Cluster example (after agreement + feature pipeline at 2026-06-09):

    python3 scripts/correlate_vbench_gain_with_features.py \\
        --gains-csv sweep_experiment/reports/per_video_analysis/2026-06-28/vbench_agreement/per_video_vbench_gains.csv \\
        --features-csv sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --output-dir sweep_experiment/reports/per_video_analysis/2026-06-28/vbench_predictors

Outputs:
  * ``vbench_correlation_summary.md``  — headline ρ for OOD + top Tier-1 features
  * ``vbench_correlation_table.csv``   — full ρ grid (method × vbench_dim × feature)
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
from scripts.correlate_tta_gain_with_features import (  # noqa: E402
    FEATURE_INTERPRETATIONS,
    OOD_SUMMARY_COLUMNS,
    TIER1_FEATURES,
    bootstrap_spearman_ci,
    load_features_csv,
    load_ood_csv,
    spearman_rho,
)
from scripts.summarize_vbench_population_per_video import DIM_SHORT  # noqa: E402

HEADLINE_FEATURES: Tuple[str, ...] = (
    "mean_diffusion_loss_caption",
    "mean_diffusion_loss_uncond",
    "latent_norm_mean",
    "mean_flow",
    "dino_temporal_l2_mean",
    "rgb_histogram_entropy_mean",
    "rec_err_lpips",
    "mean_grad_norm_lora",
    "clip_text_image_sim_mean",
)

METHODS_DEFAULT: Tuple[str, ...] = (
    "ADA",
    "LORA_R8_TTA",
    "K5_SIM",
    "K5_RAND",
    "K10_SIM",
    "K10_RAND",
)

VBENCH_HEADLINE: Tuple[str, ...] = (
    "aesthetic_quality",
    "imaging_quality",
    "subject_consistency",
    "dynamic_degree",
)


def _coerce(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return x


def load_vbench_gains(path: Path) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Return (methods, {video_id -> {col -> float}})."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        methods: List[str] = []
        for fn in fieldnames:
            for dim in VBENCH_DIMS:
                suffix = f"_d{dim}"
                if fn.endswith(suffix):
                    methods.append(fn[: -len(suffix)])
                    break
            if fn.endswith("_dpsnr"):
                methods.append(fn[: -len("_dpsnr")])
        methods = sorted(set(methods))
        rows: Dict[str, Dict[str, float]] = {}
        for r in reader:
            vid = (r.get("video_id") or "").strip()
            if not vid:
                continue
            rows[vid] = {k: _coerce(v) for k, v in r.items()}
    return methods, rows


def _join_videos(
    gains: Dict[str, Dict[str, float]],
    features: Dict[str, Dict[str, float]],
    ood: Optional[Dict[str, Dict[str, float]]],
) -> List[str]:
    common = set(gains.keys()) & set(features.keys())
    if ood:
        common &= set(ood.keys())
    return sorted(common)


def build_report(
    video_ids: Sequence[str],
    methods: Sequence[str],
    gains: Dict[str, Dict[str, float]],
    features: Dict[str, Dict[str, float]],
    ood: Optional[Dict[str, Dict[str, float]]],
    feature_names: Sequence[str],
    vbench_dims: Sequence[str],
    *,
    rho_threshold: float = 0.2,
    bootstrap: bool = False,
) -> Tuple[str, List[List[str]]]:
    csv_rows: List[List[str]] = []
    lines: List[str] = []
    lines.append("# VBench++ Δ vs Phase-0 predictors (Spearman ρ)")
    lines.append("")
    lines.append(f"- **Videos (intersection):** {len(video_ids)}")
    lines.append(f"- **Pass bar (same as PSNR gating):** |ρ| ≥ {rho_threshold} on ≥ 2 methods")
    lines.append("")
    lines.append(
        "Tests whether OOD / motion / complexity features that **failed** to predict "
        "ΔPSNR (H1–H8) nonetheless predict **per-video VBench++ shifts** under TTA."
    )
    lines.append("")

    for feat in feature_names:
        if ood:
            feat_vals = np.array(
                [_coerce(ood.get(vid, {}).get(feat) or features.get(vid, {}).get(feat))
                 for vid in video_ids],
                dtype=float,
            )
        else:
            feat_vals = np.array(
                [_coerce(features.get(vid, {}).get(feat)) for vid in video_ids],
                dtype=float,
            )

        interp = FEATURE_INTERPRETATIONS.get(feat, feat)
        lines.append(f"## `{feat}`")
        lines.append("")
        lines.append(f"*{interp}*")
        lines.append("")
        hdr = "| Method | " + " | ".join(DIM_SHORT.get(d, d) for d in vbench_dims) + " | PSNR Δ |"
        lines.append(hdr)
        lines.append("|---|" + "|".join(["---:"] * (len(vbench_dims) + 1)) + "|")

        pass_count = 0
        for method in methods:
            cells = []
            for dim in vbench_dims:
                col = f"{method}_d{dim}"
                y = np.array([gains.get(vid, {}).get(col, float("nan")) for vid in video_ids], dtype=float)
                rho = spearman_rho(feat_vals, y)
                if rho is not None and abs(rho) >= rho_threshold:
                    pass_count += 1
                cells.append(f"{rho:+.3f}" if rho is not None else "n/a")
                csv_rows.append([feat, method, dim, f"{rho:.6f}" if rho is not None else "", str(len(video_ids))])
            y_psnr = np.array(
                [gains.get(vid, {}).get(f"{method}_dpsnr", float("nan")) for vid in video_ids],
                dtype=float,
            )
            rho_p = spearman_rho(feat_vals, y_psnr)
            cells.append(f"{rho_p:+.3f}" if rho_p is not None else "n/a")
            csv_rows.append([feat, method, "psnr", f"{rho_p:.6f}" if rho_p is not None else "", str(len(video_ids))])
            lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Reading guide")
    lines.append("")
    lines.append("- Compare each row to **H5** (OOD vs ΔPSNR): if OOD ρ is similar for VBench IQ/Aes, "
                 "predictors do not become useful just because we change the metric.")
    lines.append("- **|ρ| < 0.15** everywhere → no deployable offline gate for VBench frontier.")
    lines.append("- Strong **Aes-only** ρ with flat PSNR ρ would suggest perceptual-specific routing "
                 "(worth a dedicated follow-up).")
    lines.append("")

    return "\n".join(lines), csv_rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains-csv", type=Path, required=True)
    ap.add_argument("--features-csv", type=Path, required=True)
    ap.add_argument("--ood-csv", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="*", default=list(METHODS_DEFAULT))
    ap.add_argument("--rho-threshold", type=float, default=0.2)
    args = ap.parse_args()

    methods_found, gains = load_vbench_gains(args.gains_csv)
    methods = [m for m in args.methods if m in methods_found]
    if not methods:
        methods = [m for m in methods_found if m.upper() != "NOTTA"]

    features = load_features_csv(args.features_csv)
    ood: Optional[Dict[str, Dict[str, str]]] = None
    if args.ood_csv:
        ood, _ood_cols = load_ood_csv(args.ood_csv)
    video_ids = _join_videos(gains, features, ood)
    if len(video_ids) < 50:
        print(f"[error] intersection too small: {len(video_ids)} videos", file=sys.stderr)
        return 2

    # Build feature list: headline + any OOD summary cols present
    feature_names = list(HEADLINE_FEATURES)
    for c in OOD_SUMMARY_COLUMNS:
        if c not in feature_names:
            feature_names.append(c)

    report, csv_rows = build_report(
        video_ids, methods, gains, features, ood, feature_names, VBENCH_HEADLINE,
        rho_threshold=args.rho_threshold,
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vbench_correlation_summary.md").write_text(report, encoding="utf-8")
    with (out_dir / "vbench_correlation_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["feature", "method", "outcome", "spearman_rho", "n"])
        w.writerows(csv_rows)
    print(f"Wrote {out_dir / 'vbench_correlation_summary.md'} ({len(video_ids)} videos)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
