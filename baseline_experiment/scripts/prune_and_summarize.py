#!/usr/bin/env python3
"""
Post-process baseline results: summarize metrics, prune saved videos,
and write a markdown report.

Two modes:
  1) --create-keep-list  (run 1 only)
     Sort all videos by PSNR, keep top-N and bottom-N, write keep_videos.txt.

  2) --keep-list PATH    (runs 2-5)
     Load an existing keep_videos.txt and keep only those videos.

In both modes the script deletes non-kept mp4s from generated_videos/
and writes a RESULTS.md summary.

Usage:
  # After run 1 (creates the keep list):
  python prune_and_summarize.py --results-dir .../cond2_gen14 --create-keep-list

  # After runs 2-5 (reuses the keep list from run 1):
  python prune_and_summarize.py --results-dir .../cond14_gen14 \
      --keep-list .../cond2_gen14/keep_videos.txt
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_metrics(csv_path: Path) -> list[dict]:
    """Read per_video_metrics.csv and return rows with numeric fields cast."""
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["index"] = int(row["index"])
                row["psnr"] = float(row["psnr"]) if row.get("psnr") else None
                row["ssim"] = float(row["ssim"]) if row.get("ssim") else None
                row["lpips"] = float(row["lpips"]) if row.get("lpips") else None
            except (ValueError, TypeError):
                row["psnr"] = row["ssim"] = row["lpips"] = None
            rows.append(row)
    return rows


def load_summary(results_dir: Path) -> dict:
    """Load summary.json if it exists."""
    p = results_dir / "summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def valid_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r["psnr"] is not None]


def stats(values: list[float]) -> dict:
    if not values:
        return {}
    return {
        "mean": round(np.mean(values), 4),
        "std": round(np.std(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def extract_index_from_filename(name: str) -> int | None:
    """Extract leading integer index from filenames like '032_slug_PSNR-...mp4'."""
    m = re.match(r"^(\d+)_", name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Pruning
# ---------------------------------------------------------------------------

def prune_videos(gen_dir: Path, keep_indices: set[int]) -> tuple[int, int]:
    """Delete mp4s whose index is not in keep_indices. Returns (kept, deleted)."""
    if not gen_dir.exists():
        return 0, 0
    kept = deleted = 0
    for mp4 in sorted(gen_dir.glob("*.mp4")):
        idx = extract_index_from_filename(mp4.name)
        if idx is not None and idx not in keep_indices:
            mp4.unlink()
            deleted += 1
        else:
            kept += 1
    return kept, deleted


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(
    results_dir: Path,
    rows: list[dict],
    summary: dict,
    keep_indices: set[int],
    top_rows: list[dict],
    bottom_rows: list[dict],
):
    vrows = valid_rows(rows)
    psnrs = [r["psnr"] for r in vrows]
    ssims = [r["ssim"] for r in vrows]
    lpipss = [r["lpips"] for r in vrows]

    # Derive config label from directory name or summary
    config_label = results_dir.name
    num_cond = summary.get("num_cond_frames", "?")
    num_gen = summary.get("num_gen_frames", "?")

    lines = []
    lines.append(f"# Baseline Results: {num_cond} cond + {num_gen} gen frames")
    lines.append("")
    lines.append(f"**Directory**: `{results_dir}`  ")
    lines.append(f"**Videos evaluated**: {len(vrows)} / {len(rows)}  ")
    if summary.get("resolution"):
        lines.append(f"**Resolution**: {summary['resolution']}  ")
    if summary.get("timing"):
        t = summary["timing"]
        lines.append(f"**Model load**: {t.get('model_load_s', '?')}s  ")
        lines.append(f"**Total inference**: {t.get('total_inference_s', '?')}s  ")
        pv = t.get("per_video_inference_s", {})
        if pv:
            lines.append(f"**Per-video inference**: {pv.get('mean', '?')}s "
                         f"\u00b1 {pv.get('std', '?')}s  ")

    # Aggregate metrics table
    lines.append("")
    lines.append("## Aggregate Metrics (all 100 videos)")
    lines.append("")
    lines.append("| Metric | Mean | Std | Min | Max |")
    lines.append("|--------|------|-----|-----|-----|")
    for name, vals in [("PSNR", psnrs), ("SSIM", ssims), ("LPIPS", lpipss)]:
        s = stats(vals)
        lines.append(f"| {name} | {s.get('mean','-')} | {s.get('std','-')} "
                     f"| {s.get('min','-')} | {s.get('max','-')} |")

    # Top N table
    lines.append("")
    lines.append(f"## Top {len(top_rows)} Videos by PSNR (best)")
    lines.append("")
    lines.append("| Rank | Index | Filename | Caption | PSNR | SSIM | LPIPS |")
    lines.append("|------|-------|----------|---------|------|------|-------|")
    for rank, r in enumerate(reversed(top_rows), 1):  # best first
        caption_short = (r.get("caption", "") or "")[:60]
        lines.append(
            f"| {rank} | {r['index']:03d} | {r.get('filename','')} "
            f"| {caption_short} | {r['psnr']:.4f} | {r['ssim']:.4f} | {r['lpips']:.4f} |"
        )

    # Bottom N table
    lines.append("")
    lines.append(f"## Bottom {len(bottom_rows)} Videos by PSNR (worst)")
    lines.append("")
    lines.append("| Rank | Index | Filename | Caption | PSNR | SSIM | LPIPS |")
    lines.append("|------|-------|----------|---------|------|------|-------|")
    for rank, r in enumerate(bottom_rows, 1):  # worst first
        caption_short = (r.get("caption", "") or "")[:60]
        lines.append(
            f"| {rank} | {r['index']:03d} | {r.get('filename','')} "
            f"| {caption_short} | {r['psnr']:.4f} | {r['ssim']:.4f} | {r['lpips']:.4f} |"
        )

    # Kept videos list
    lines.append("")
    lines.append(f"## Kept Videos ({len(keep_indices)} total)")
    lines.append("")
    lines.append("Video indices kept for visual inspection: "
                 + ", ".join(f"{i:03d}" for i in sorted(keep_indices)))
    lines.append("")

    md_path = results_dir / "RESULTS.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Prune baseline videos & write summary")
    p.add_argument("--results-dir", type=str, required=True,
                   help="Path to a single run's results directory")
    p.add_argument("--create-keep-list", action="store_true",
                   help="(Run 1) Create keep_videos.txt from top/bottom PSNR")
    p.add_argument("--keep-list", type=str, default=None,
                   help="(Runs 2-5) Path to existing keep_videos.txt")
    p.add_argument("--top-n", type=int, default=10,
                   help="Number of top-PSNR videos to keep")
    p.add_argument("--bottom-n", type=int, default=10,
                   help="Number of bottom-PSNR videos to keep")
    args = p.parse_args()

    if not args.create_keep_list and args.keep_list is None:
        print("ERROR: Specify --create-keep-list or --keep-list PATH")
        sys.exit(1)

    results_dir = Path(args.results_dir)
    csv_path = results_dir / "per_video_metrics.csv"
    gen_dir = results_dir / "generated_videos"

    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    # ── Load metrics ──────────────────────────────────────────────────
    rows = load_metrics(csv_path)
    vrows = valid_rows(rows)
    print(f"Loaded {len(vrows)} valid rows from {csv_path}")

    if not vrows:
        print("ERROR: No valid metric rows found")
        sys.exit(1)

    summary = load_summary(results_dir)

    # Sort by PSNR ascending
    sorted_by_psnr = sorted(vrows, key=lambda r: r["psnr"])
    bottom_rows = sorted_by_psnr[:args.bottom_n]
    top_rows = sorted_by_psnr[-args.top_n:]

    # ── Determine keep set ────────────────────────────────────────────
    if args.create_keep_list:
        keep_indices = set(r["index"] for r in bottom_rows + top_rows)
        keep_path = results_dir / "keep_videos.txt"
        with open(keep_path, "w") as f:
            for idx in sorted(keep_indices):
                f.write(f"{idx}\n")
        print(f"Created keep list: {keep_path} ({len(keep_indices)} videos)")
    else:
        with open(args.keep_list) as f:
            keep_indices = set(int(line.strip()) for line in f if line.strip())
        print(f"Loaded keep list: {args.keep_list} ({len(keep_indices)} videos)")
        # Recompute top/bottom from the full set for the report
        # (they still reflect all 100 videos, not just the kept ones)

    # ── Prune videos ──────────────────────────────────────────────────
    kept, deleted = prune_videos(gen_dir, keep_indices)
    print(f"Pruned generated_videos/: kept {kept}, deleted {deleted}")

    # ── Write markdown ────────────────────────────────────────────────
    write_markdown(results_dir, rows, summary, keep_indices, top_rows, bottom_rows)

    print("Done.")


if __name__ == "__main__":
    main()
