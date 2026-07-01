#!/usr/bin/env python3
"""Step 3: Rank-based AUC for win/loss classification (cheap router screen).

Labels each video win=1 if Δoutcome > threshold else loss=0; scores each predictor
by Mann-Whitney AUC. No sklearn/xgboost required.

Use after Steps 1–2: if best single-feature AUC ≈ 0.52, hand features are dead;
if AUC ≈ 0.60+, worth a learned router or deployable threshold.

Example:
    python3 scripts/analyze_router_auc.py \\
        --gains-csv .../per_video_vbench_gains.csv \\
        --features-csv .../video_features.csv \\
        --ood-csv .../diffusion_ood_scores.csv \\
        --output-dir .../predictor_transfer
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

from scripts.predictor_analysis_common import (  # noqa: E402
    GATE_METHODS,
    METHODS_DEFAULT,
    intersect_videos,
    join_feature_tables,
    load_vbench_gains,
    notta_baseline_predictors,
    outcome_column,
    outcome_specs,
    predictor_interp,
)
from scripts.summarize_vbench_population_per_video import DIM_SHORT  # noqa: E402

DEFAULT_OUTCOMES = [
    "psnr", "ssim", "lpips", "aesthetic_quality", "imaging_quality", "vbench_total"
]

DEFAULT_THRESHOLDS = {
    "psnr": 0.1,
    "ssim": 0.01,
    "lpips": 0.01,
    "aesthetic_quality": 0.01,
    "imaging_quality": 0.01,
    "subject_consistency": 0.01,
    "dynamic_degree": 0.01,
    "motion_smoothness": 0.01,
    "background_consistency": 0.01,
    "temporal_flickering": 0.01,
    "vbench_total": 0.01,
}


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """AUC via rank sum (Mann-Whitney U), labels in {0,1}."""
    mask = ~(np.isnan(scores) | np.isnan(labels))
    s = scores[mask]
    y = labels[mask]
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < 5 or n_neg < 5:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(s, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    rank_sum_pos = float(ranks[y == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def build_predictor_arrays(
    video_ids: Sequence[str],
    gains: Dict[str, Dict[str, float]],
    features: Dict[str, Dict[str, float]],
) -> Dict[str, np.ndarray]:
    preds: Dict[str, np.ndarray] = {}
    for col, _ in notta_baseline_predictors():
        preds[col] = np.array(
            [float(gains[vid].get(col, float("nan"))) for vid in video_ids], dtype=float
        )
    for name in sorted({k for d in features.values() for k in d}):
        preds[name] = np.array(
            [float(features.get(vid, {}).get(name, float("nan"))) for vid in video_ids],
            dtype=float,
        )
    return preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gains-csv", type=Path, required=True)
    ap.add_argument("--features-csv", type=Path, required=True)
    ap.add_argument("--ood-csv", type=Path, default=None)
    ap.add_argument("--tier3-csv", type=Path, default=None)
    ap.add_argument("--flow-csv", type=Path, default=None)
    ap.add_argument("--bpp-csv", type=Path, default=None)
    ap.add_argument("--fft-csv", type=Path, default=None)
    ap.add_argument("--vae-recerr-csv", type=Path, default=None)
    ap.add_argument("--motion-csv", type=Path, default=None)
    ap.add_argument("--loss-var-csv", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="*", default=list(GATE_METHODS))
    ap.add_argument("--outcomes", nargs="*", default=DEFAULT_OUTCOMES)
    ap.add_argument("--min-auc", type=float, default=0.55,
                    help="Flag predictors with AUC >= this or <= 1-this")
    args = ap.parse_args()

    video_ids_g, gains, all_methods = load_vbench_gains(args.gains_csv)
    methods = [m for m in args.methods if m in all_methods] or list(GATE_METHODS)
    specs = {s.key: s for s in outcome_specs()}

    features, _, tiers = join_feature_tables(
        features_csv=args.features_csv,
        ood_csv=args.ood_csv,
        tier3_csv=args.tier3_csv,
        flow_csv=args.flow_csv,
        bpp_csv=args.bpp_csv,
        fft_csv=args.fft_csv,
        vae_recerr_csv=args.vae_recerr_csv,
        motion_csv=args.motion_csv,
        loss_var_csv=args.loss_var_csv,
    )
    video_ids = intersect_videos(video_ids_g, features.keys())
    if len(video_ids) < 50:
        print(f"[error] intersection too small: {len(video_ids)}", file=sys.stderr)
        return 2

    predictors = build_predictor_arrays(video_ids, gains, features)
    auc_rows: List[Dict[str, object]] = []

    for method in methods:
        for out_key in args.outcomes:
            if out_key not in specs:
                continue
            spec = specs[out_key]
            col = outcome_column(method, spec)
            thr = DEFAULT_THRESHOLDS.get(out_key, 0.01)
            deltas = np.array(
                [float(gains[vid].get(col, float("nan"))) for vid in video_ids], dtype=float
            )
            labels = np.where(deltas > thr, 1.0, 0.0)
            labels[np.isnan(deltas)] = np.nan
            for pred_name, scores in predictors.items():
                auc = binary_auc(scores, labels)
                auc_rows.append({
                    "predictor": pred_name,
                    "method": method,
                    "outcome": out_key,
                    "threshold": thr,
                    "auc": auc,
                    "n": int(np.sum(~np.isnan(scores) & ~np.isnan(labels))),
                    "win_rate": float(np.nanmean(labels == 1)),
                })

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "router_auc_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["predictor", "method", "outcome", "threshold", "auc", "n", "win_rate"],
        )
        w.writeheader()
        for r in auc_rows:
            row = dict(r)
            auc = row.get("auc")
            row["auc"] = f"{auc:.4f}" if auc is not None else ""
            row["win_rate"] = f"{row['win_rate']:.3f}"
            w.writerow(row)

    lines: List[str] = []
    lines.append("# Router AUC screen (win/loss classification)")
    lines.append("")
    lines.append(f"- **Videos:** {len(video_ids)}")
    lines.append(f"- **Methods:** {', '.join(methods)}")
    lines.append(f"- **Flag threshold:** AUC ≥ {args.min_auc:.2f} or ≤ {1 - args.min_auc:.2f}")
    lines.append("- **Label:** win if Δoutcome > threshold (see CSV)")
    lines.append("")

    for method in methods:
        lines.append(f"## `{method}` — top predictors by |AUC−0.5|")
        lines.append("")
        for out_key in args.outcomes:
            subset = [
                r for r in auc_rows
                if r["method"] == method and r["outcome"] == out_key and r["auc"] is not None
            ]
            if not subset:
                continue
            ranked = sorted(
                subset,
                key=lambda r: abs(float(r["auc"]) - 0.5),
                reverse=True,
            )[:5]
            lines.append(f"### {out_key} (win rate {ranked[0]['win_rate']:.1%})")
            lines.append("")
            lines.append("| Predictor | AUC | n |")
            lines.append("|---|---:|---:|")
            for r in ranked:
                auc = float(r["auc"])
                flag = " **" if auc >= args.min_auc or auc <= 1 - args.min_auc else ""
                lines.append(
                    f"| `{r['predictor']}`{flag} | {auc:.3f} | {r['n']} |"
                )
            lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- AUC ≈ 0.50 → no better than chance for routing wins.")
    lines.append("- AUC ≥ 0.55 on **both** ADA and LoRA for same outcome → candidate gate (verify with quintile policy).")
    lines.append("- AUC < 0.45 → predictor identifies **losses** (useful for skip-gate).")
    lines.append("")

    report_path = out_dir / "router_auc_summary.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
