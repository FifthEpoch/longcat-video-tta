#!/usr/bin/env python3
"""
Analyze CLIP-gate calibration runs and recommend a threshold.

Typical usage:
  python sweep_experiment/scripts/analyze_clip_gate_calibration.py \
    --run /scratch/wc3013/longcat-video-tta/sweep_experiment/results_clip_gate_calib/es_ablation_disable/ES_D1/summary.json \
    --run /scratch/wc3013/longcat-video-tta/sweep_experiment/results_clip_gate_calib/lora_constrained_sweep/LA2/summary.json \
    --run /scratch/wc3013/longcat-video-tta/sweep_experiment/results_clip_gate_calib/delta_b_low_lr/DBL4/summary.json \
    --baseline-csv /scratch/wc3013/longcat-video-tta/baseline_experiment/results/panda_no_tta_continuation/NOTTA/per_video_metrics.csv \
    --output-json /scratch/wc3013/longcat-video-tta/sweep_experiment/results_clip_gate_calib/clip_gate_calibration_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _stem_from_path(path: str) -> str:
    return Path(path).stem


def load_baseline_csv(path: str) -> Dict[str, Dict[str, float]]:
    """Load baseline per-video metrics from per_video_metrics.csv."""
    out: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("filename", "")
            key = _stem_from_path(name) if name else ""
            if not key:
                continue
            out[key] = {
                "psnr": _safe_float(row.get("psnr")),
                "ssim": _safe_float(row.get("ssim")),
                "lpips": _safe_float(row.get("lpips")),
            }
    return out


def load_baseline_summary(path: str) -> Dict[str, Dict[str, float]]:
    """Load baseline per-video metrics from summary.json if it has results[]."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = data.get("results", [])
    out: Dict[str, Dict[str, float]] = {}
    for r in rows:
        key = r.get("video_name") or _stem_from_path(r.get("filename", "") or r.get("video_path", ""))
        if not key:
            continue
        out[str(key)] = {
            "psnr": _safe_float(r.get("psnr")),
            "ssim": _safe_float(r.get("ssim")),
            "lpips": _safe_float(r.get("lpips")),
        }
    return out


def load_tta_rows(summary_path: str) -> Tuple[str, List[dict]]:
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for r in data.get("results", []):
        if not r.get("success", False):
            continue
        score = _safe_float(r.get("clip_alignment_score"))
        if score is None:
            continue
        key = r.get("video_name") or _stem_from_path(r.get("video_path", ""))
        if not key:
            continue
        rows.append({
            "video_key": str(key),
            "score": score,
            "psnr": _safe_float(r.get("psnr")),
            "ssim": _safe_float(r.get("ssim")),
            "lpips": _safe_float(r.get("lpips")),
            "series": data.get("series_name", ""),
            "method": data.get("method", ""),
            "run_name": f"{Path(summary_path).parent.parent.name}/{Path(summary_path).parent.name}",
        })
    return f"{Path(summary_path).parent.parent.name}/{Path(summary_path).parent.name}", rows


def percentile(values: List[float], p: float) -> float:
    if not values:
        raise ValueError("No values for percentile")
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    i = int(k)
    j = min(i + 1, len(xs) - 1)
    frac = k - i
    return xs[i] * (1.0 - frac) + xs[j] * frac


def simulate_threshold(
    rows: List[dict],
    baseline: Dict[str, Dict[str, float]],
    threshold: float,
    harmful_delta_psnr: float,
) -> dict:
    n = len(rows)
    skipped = [r for r in rows if r["score"] < threshold]
    skip_keys = {r["video_key"] for r in skipped}

    # With gate: skipped videos use baseline metric, others use TTA metric.
    sim_psnr = []
    sim_ssim = []
    sim_lpips = []
    comparable = 0
    harmful_total = 0
    harmful_skipped = 0
    helpful_total = 0
    helpful_skipped = 0

    for r in rows:
        key = r["video_key"]
        b = baseline.get(key)
        tta_psnr = r.get("psnr")
        tta_ssim = r.get("ssim")
        tta_lpips = r.get("lpips")
        is_skipped = key in skip_keys

        if b and b.get("psnr") is not None and tta_psnr is not None:
            comparable += 1
            delta = tta_psnr - b["psnr"]
            if delta < -abs(harmful_delta_psnr):
                harmful_total += 1
                if is_skipped:
                    harmful_skipped += 1
            if delta > abs(harmful_delta_psnr):
                helpful_total += 1
                if is_skipped:
                    helpful_skipped += 1

        # PSNR
        if is_skipped:
            if b and b.get("psnr") is not None:
                sim_psnr.append(b["psnr"])
        elif tta_psnr is not None:
            sim_psnr.append(tta_psnr)

        # SSIM
        if is_skipped:
            if b and b.get("ssim") is not None:
                sim_ssim.append(b["ssim"])
        elif tta_ssim is not None:
            sim_ssim.append(tta_ssim)

        # LPIPS (lower is better)
        if is_skipped:
            if b and b.get("lpips") is not None:
                sim_lpips.append(b["lpips"])
        elif tta_lpips is not None:
            sim_lpips.append(tta_lpips)

    return {
        "threshold": threshold,
        "num_rows": n,
        "num_skipped": len(skipped),
        "skip_rate": (len(skipped) / n) if n else 0.0,
        "sim_psnr_mean": statistics.mean(sim_psnr) if sim_psnr else None,
        "sim_ssim_mean": statistics.mean(sim_ssim) if sim_ssim else None,
        "sim_lpips_mean": statistics.mean(sim_lpips) if sim_lpips else None,
        "comparable_count": comparable,
        "harmful_total": harmful_total,
        "harmful_skipped": harmful_skipped,
        "harmful_skip_recall": (harmful_skipped / harmful_total) if harmful_total else None,
        "helpful_total": helpful_total,
        "helpful_skipped": helpful_skipped,
        "helpful_skip_rate": (helpful_skipped / helpful_total) if helpful_total else None,
    }


def main():
    p = argparse.ArgumentParser(description="Analyze CLIP-gate calibration outcomes.")
    p.add_argument("--run", action="append", required=True,
                   help="Path to calibration summary.json. Repeat per run.")
    p.add_argument("--baseline-csv", type=str, default=None,
                   help="Path to baseline per_video_metrics.csv.")
    p.add_argument("--baseline-summary", type=str, default=None,
                   help="Path to baseline summary.json with results[].")
    p.add_argument("--percentiles", type=str, default="10,20,25,30,35,40,50",
                   help="Comma-separated threshold percentiles over CLIP scores.")
    p.add_argument("--harmful-delta-psnr", type=float, default=0.1,
                   help="Define harmful as delta_psnr < -value.")
    p.add_argument("--max-skip-rate", type=float, default=0.25,
                   help="Constraint for recommended threshold.")
    p.add_argument("--max-helpful-skip-rate", type=float, default=0.15,
                   help="Constraint for recommended threshold.")
    p.add_argument("--output-json", type=str, default=None,
                   help="Optional output path for machine-readable report.")
    args = p.parse_args()

    baseline: Dict[str, Dict[str, float]] = {}
    if args.baseline_csv:
        baseline.update(load_baseline_csv(args.baseline_csv))
    if args.baseline_summary:
        baseline.update(load_baseline_summary(args.baseline_summary))

    all_rows: List[dict] = []
    run_summaries = []
    for rp in args.run:
        run_name, rows = load_tta_rows(rp)
        all_rows.extend(rows)
        run_summaries.append({"run": run_name, "num_rows": len(rows)})

    if not all_rows:
        raise SystemExit("No CLIP-scored rows found in provided runs.")

    scores = [r["score"] for r in all_rows]
    percs = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]
    thresholds = [percentile(scores, pp) for pp in percs]

    eval_rows = [
        simulate_threshold(
            rows=all_rows,
            baseline=baseline,
            threshold=t,
            harmful_delta_psnr=args.harmful_delta_psnr,
        )
        for t in thresholds
    ]

    # Recommendation: maximize simulated PSNR under constraints.
    constrained = []
    for r in eval_rows:
        helpful_skip = r["helpful_skip_rate"]
        if helpful_skip is None:
            helpful_skip = 0.0
        if r["skip_rate"] <= args.max_skip_rate and helpful_skip <= args.max_helpful_skip_rate:
            constrained.append(r)
    pool = constrained if constrained else eval_rows
    # Prefer rows with simulated psnr; fallback to lowest skip-rate if unavailable.
    with_psnr = [r for r in pool if r["sim_psnr_mean"] is not None]
    if with_psnr:
        recommended = max(with_psnr, key=lambda r: r["sim_psnr_mean"])
    else:
        recommended = min(pool, key=lambda r: r["skip_rate"])

    # Console report
    print("=" * 80)
    print("CLIP-GATE CALIBRATION REPORT")
    print("=" * 80)
    print(f"Runs analyzed: {len(run_summaries)}")
    for rs in run_summaries:
        print(f"  - {rs['run']}: {rs['num_rows']} samples")
    print(f"Total CLIP-scored samples: {len(all_rows)}")
    print(f"Score range: min={min(scores):.4f}, max={max(scores):.4f}, mean={statistics.mean(scores):.4f}")
    print(f"Baseline rows available: {len(baseline)}")
    print("-" * 80)
    print("Threshold sweep:")
    print("thr      skip%   sim_psnr  sim_ssim  sim_lpips  harm_recall  helpful_skip")
    for r in eval_rows:
        harm = "n/a" if r["harmful_skip_recall"] is None else f"{100*r['harmful_skip_recall']:.1f}%"
        hskip = "n/a" if r["helpful_skip_rate"] is None else f"{100*r['helpful_skip_rate']:.1f}%"
        psnr = "n/a" if r["sim_psnr_mean"] is None else f"{r['sim_psnr_mean']:.4f}"
        ssim = "n/a" if r["sim_ssim_mean"] is None else f"{r['sim_ssim_mean']:.4f}"
        lpips = "n/a" if r["sim_lpips_mean"] is None else f"{r['sim_lpips_mean']:.4f}"
        print(f"{r['threshold']:.4f}  {100*r['skip_rate']:5.1f}%  {psnr:>8}  {ssim:>8}  {lpips:>9}  {harm:>11}  {hskip:>12}")
    print("-" * 80)
    print("Recommended threshold:")
    print(f"  {recommended['threshold']:.4f}")
    print(f"  skip_rate={100*recommended['skip_rate']:.1f}%")
    if recommended["sim_psnr_mean"] is not None:
        print(f"  simulated_psnr={recommended['sim_psnr_mean']:.4f}")
    if recommended["harmful_skip_recall"] is not None:
        print(f"  harmful_skip_recall={100*recommended['harmful_skip_recall']:.1f}%")
    if recommended["helpful_skip_rate"] is not None:
        print(f"  helpful_skip_rate={100*recommended['helpful_skip_rate']:.1f}%")
    print("=" * 80)

    if args.output_json:
        payload = {
            "runs": run_summaries,
            "num_samples": len(all_rows),
            "score_stats": {
                "min": min(scores),
                "max": max(scores),
                "mean": statistics.mean(scores),
            },
            "baseline_count": len(baseline),
            "threshold_candidates": eval_rows,
            "recommended": recommended,
            "constraints": {
                "max_skip_rate": args.max_skip_rate,
                "max_helpful_skip_rate": args.max_helpful_skip_rate,
                "harmful_delta_psnr": args.harmful_delta_psnr,
            },
        }
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved JSON report to: {out_path}")


if __name__ == "__main__":
    main()
