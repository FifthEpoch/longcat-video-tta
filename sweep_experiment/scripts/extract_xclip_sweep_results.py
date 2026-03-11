#!/usr/bin/env python3
"""
Extract threshold-sweep metrics for X-CLIP-gated runs.

Default behavior scans:
  - results_xclip_gate_thr_*/series_lora_constrained/LA2/summary.json
  - results_xclip_gate_thr_*/series_delta_b_low_lr/DBL4/summary.json

Prints one CSV line per summary with:
series,run,thr,backend,n_ok,psnr,ssim,lpips,skip_rate,num_skipped,num_scored
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, List, Optional


DEFAULT_PATTERNS = [
    "results_xclip_gate_thr_*/series_lora_constrained/LA2/summary.json",
    "results_xclip_gate_thr_*/series_delta_b_low_lr/DBL4/summary.json",
]


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    filtered = [v for v in values if v is not None]
    return statistics.mean(filtered) if filtered else None


def fmt(value: Optional[float], ndigits: int = 6) -> str:
    if value is None:
        return "nan"
    return f"{value:.{ndigits}f}"


def load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_rows(root: Path, patterns: List[str]) -> List[dict]:
    rows: List[dict] = []
    for pat in patterns:
        for summary_path in sorted(root.glob(pat)):
            data = load_summary(summary_path)
            ok = [r for r in data.get("results", []) if r.get("success")]
            psnr = mean_or_none(r.get("psnr") for r in ok)
            ssim = mean_or_none(r.get("ssim") for r in ok)
            lpips = mean_or_none(r.get("lpips") for r in ok)

            gate_stats = data.get("clip_gate_stats", {})
            rows.append(
                {
                    "series": summary_path.parent.parent.name,
                    "run": summary_path.parent.name,
                    "thr": data.get("clip_gate_threshold"),
                    "backend": data.get("clip_gate_backend"),
                    "n_ok": data.get("num_successful"),
                    "psnr": psnr,
                    "ssim": ssim,
                    "lpips": lpips,
                    "skip": gate_stats.get("skip_rate"),
                    "num_skipped": gate_stats.get("num_skipped"),
                    "num_scored": gate_stats.get("num_scored"),
                    "summary_path": str(summary_path),
                }
            )
    rows.sort(key=lambda r: (r["series"], float(r["thr"]) if r.get("thr") is not None else 999))
    return rows


def print_csv(rows: List[dict], include_path: bool) -> None:
    header = [
        "series",
        "run",
        "thr",
        "backend",
        "n_ok",
        "psnr",
        "ssim",
        "lpips",
        "skip_rate",
        "num_skipped",
        "num_scored",
    ]
    if include_path:
        header.append("summary_path")
    print(",".join(header))

    for r in rows:
        fields = [
            r.get("series"),
            r.get("run"),
            r.get("thr"),
            r.get("backend"),
            r.get("n_ok"),
            fmt(r.get("psnr")),
            fmt(r.get("ssim")),
            fmt(r.get("lpips")),
            fmt(r.get("skip")),
            r.get("num_skipped"),
            r.get("num_scored"),
        ]
        if include_path:
            fields.append(r.get("summary_path"))
        print(",".join(str(x) for x in fields))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract X-CLIP threshold sweep metrics as CSV.")
    parser.add_argument(
        "--root",
        type=str,
        default="/scratch/wc3013/longcat-video-tta/sweep_experiment",
        help="Root directory that contains results_xclip_gate_thr_* directories.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Glob pattern under --root to summary.json (repeatable).",
    )
    parser.add_argument(
        "--include-path",
        action="store_true",
        help="Include summary_path column in output.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    patterns = args.pattern if args.pattern else DEFAULT_PATTERNS
    rows = collect_rows(root, patterns)
    print_csv(rows, include_path=args.include_path)


if __name__ == "__main__":
    main()
