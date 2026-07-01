#!/usr/bin/env python3
"""Step 2: Full Phase-0 feature battery vs all TTA Δ outcomes (PSNR/SSIM/LPIPS/VBench++).

Extends the headline-only ``correlate_vbench_gain_with_features.py`` run.

Example:
    python3 scripts/analyze_feature_outcome_battery.py \\
        --gains-csv .../per_video_vbench_gains.csv \\
        --features-csv .../2026-06-09/video_features.csv \\
        --ood-csv .../2026-06-09/diffusion_ood_scores.csv \\
        --output-dir .../predictor_transfer
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
    METHODS_DEFAULT,
    PASS_MIN_METHODS,
    PASS_RHO,
    compute_rho_grid,
    format_rho,
    intersect_videos,
    join_feature_tables,
    load_vbench_gains,
    outcome_specs,
    passes_gate,
    predictor_interp,
    write_rho_csv,
)
from scripts.summarize_vbench_population_per_video import DIM_SHORT  # noqa: E402

HEADLINE_OUTCOMES = [
    "psnr", "ssim", "lpips",
    "aesthetic_quality", "imaging_quality", "subject_consistency",
    "dynamic_degree", "vbench_total",
]


def build_report(
    video_ids: Sequence[str],
    feature_names: List[str],
    methods: Sequence[str],
    rho_rows: List[Dict],
    tiers: Dict[str, str],
) -> str:
    lines: List[str] = []
    lines.append("# Phase-0 features → TTA Δ outcomes (full battery)")
    lines.append("")
    lines.append(f"- **Videos (intersection):** {len(video_ids)}")
    lines.append(f"- **Features:** {len(feature_names)}")
    lines.append(f"- **Pass bar:** |ρ| ≥ {PASS_RHO} on ≥ {PASS_MIN_METHODS} of {', '.join(GATE_METHODS)}")
    lines.append("")

    # Collect passes
    pass_list: List[str] = []
    for feat in feature_names:
        for out in HEADLINE_OUTCOMES:
            ok, hits = passes_gate(rho_rows, feat, out, methods=GATE_METHODS)
            if ok:
                pass_list.append(f"`{feat}` → {out} ({', '.join(hits)})")

    lines.append("## Pass / fail headline")
    lines.append("")
    if pass_list:
        lines.append(f"**{len(pass_list)} predictor→outcome pairs pass** (headline outcomes only):")
        for p in sorted(pass_list)[:40]:
            lines.append(f"- {p}")
        if len(pass_list) > 40:
            lines.append(f"- … and {len(pass_list) - 40} more (see CSV)")
    else:
        lines.append("**No feature clears the deployable pass bar on headline outcomes.**")
    lines.append("")

    # Top |rho| per outcome for ADA
    lines.append("## Strongest |ρ| per outcome (`ADA`, headline features only)")
    lines.append("")
    lines.append("| Outcome | Feature | ρ | Tier | Also ≥0.2 on LoRA? |")
    lines.append("|---|---|---:|---|---|")
    for out in HEADLINE_OUTCOMES:
        candidates = [
            r for r in rho_rows
            if r["method"] == "ADA" and r["outcome"] == out and r["rho"] is not None
        ]
        if not candidates:
            continue
        best = max(candidates, key=lambda r: abs(float(r["rho"])))
        feat = str(best["predictor"])
        rho_ada = float(best["rho"])
        rho_lora = next(
            (
                float(r["rho"])
                for r in rho_rows
                if r["predictor"] == feat and r["method"] == "LORA_R8_TTA"
                and r["outcome"] == out and r["rho"] is not None
            ),
            float("nan"),
        )
        lora_ok = "yes" if abs(rho_lora) >= PASS_RHO else "no"
        lines.append(
            f"| {out} | `{feat}` | {rho_ada:+.3f} | {tiers.get(feat, '?')} | {lora_ok} |"
        )
    lines.append("")

    # Detailed tables for headline Phase-0 features (same as old vbench_correlation_summary)
    headline_feats = [
        "mean_diffusion_loss_caption",
        "latent_norm_mean",
        "dino_temporal_l2_mean",
        "rgb_histogram_entropy_mean",
        "latent_kurtosis",
        "clip_text_image_sim_mean",
        "delta_caption_minus_uncond",
        "mean_grad_norm_lora",
        "rec_err_lpips",
        "mean_flow",
    ]
    vbench_cols = ["aesthetic_quality", "imaging_quality", "subject_consistency", "dynamic_degree"]

    lines.append("## Headline feature tables (matches prior VBench predictor report)")
    lines.append("")
    for feat in headline_feats:
        if feat not in feature_names:
            continue
        lines.append(f"### `{feat}`")
        lines.append("")
        lines.append(f"*{predictor_interp(feat)}*")
        lines.append("")
        hdr = "| Method | " + " | ".join(DIM_SHORT.get(d, d) for d in vbench_cols) + " | PSNR | SSIM | IQ |"
        lines.append(hdr)
        lines.append("|---|" + "|".join(["---:"] * (len(vbench_cols) + 3)) + "|")
        for method in methods:
            cells = []
            for out in vbench_cols + ["psnr", "ssim", "imaging_quality"]:
                rho = next(
                    (
                        r["rho"]
                        for r in rho_rows
                        if r["predictor"] == feat and r["method"] == method
                        and r["outcome"] == out
                    ),
                    None,
                )
                cells.append(format_rho(rho))
            lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Reading guide")
    lines.append("")
    lines.append("- Full grid: ``feature_outcome_rho.csv``")
    lines.append("- LoRA-only ρ on ΔIQ/Aes with flat ADA ρ → skip-gate, not apply-gate")
    lines.append("")
    return "\n".join(lines)


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
    ap.add_argument("--methods", nargs="*", default=list(METHODS_DEFAULT))
    args = ap.parse_args()

    video_ids_g, gains, all_methods = load_vbench_gains(args.gains_csv)
    methods = [m for m in args.methods if m in all_methods] or list(METHODS_DEFAULT)

    features, feature_names, tiers = join_feature_tables(
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

    pred_arrays = {
        name: np.array(
            [float(features.get(vid, {}).get(name, float("nan"))) for vid in video_ids],
            dtype=float,
        )
        for name in feature_names
    }

    rho_rows = compute_rho_grid(
        video_ids, pred_arrays, gains, methods, outcome_specs()
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rho_csv(out_dir / "feature_outcome_rho.csv", rho_rows)
    report = build_report(video_ids, feature_names, methods, rho_rows, tiers)
    (out_dir / "feature_outcome_battery.md").write_text(report, encoding="utf-8")
    print(f"Wrote {out_dir / 'feature_outcome_battery.md'} ({len(video_ids)} videos, {len(feature_names)} features)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
