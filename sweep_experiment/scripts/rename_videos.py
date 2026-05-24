#!/usr/bin/env python3
"""
Rename saved experiment videos to include per-video metrics in the filename.

Naming convention:
  <idx>_<caption-with-dashes>_FVD-<run_fvd>_PSNR-<psnr>_SSIM-<ssim>_LPIPS-<lpips>_FID-<run_fid>_<method>.mp4

Example:
  023_cricket-shot_FVD-555.7_PSNR-9.096_SSIM-0.335_LPIPS-0.656_FID-78.6_adasteer.mp4

Usage:
  python rename_videos.py --run-dir /path/to/results/sanity_100/panda_adasteer_ablation/AS_BARE
  python rename_videos.py --run-dir /path/to/results/sanity_100/panda_adasteer_ablation/NOTTA_G4 --method no-TTA

  # Process multiple run directories at once:
  python rename_videos.py \
      --run-dir /path/to/results/sanity_100/panda_adasteer_ablation/NOTTA_G4 \
      --run-dir /path/to/results/sanity_100/panda_adasteer_ablation/AS_BARE \
      --run-dir /path/to/results/sanity_100/panda_adasteer_ablation/AS_ES1
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path


METHOD_MAP = {
    "delta_a": "adasteer",
    "delta_b": "adasteer-B",
    "delta_c": "adasteer-C",
    "lora_tta": "lora",
    "full_tta": "no-TTA",
    "tinylora": "tinylora",
}


def sanitize_caption(caption: str, max_len: int = 80) -> str:
    """Convert caption to a filesystem-safe slug."""
    s = caption.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s


def extract_index(video_name: str) -> str:
    """Extract zero-padded index from video_name like 'panda_0023' -> '023'."""
    m = re.search(r"(\d+)$", video_name)
    if m:
        return m.group(1).lstrip("0") or "0"
    return video_name


def determine_method(summary: dict, override: str | None = None) -> str:
    if override:
        return override
    method_raw = summary.get("method", "unknown")
    steps = summary.get("delta_steps", None)
    if steps is not None and int(steps) == 0:
        return "no-TTA"
    return METHOD_MAP.get(method_raw, method_raw)


def build_new_filename(
    video_name: str,
    caption: str,
    psnr: float,
    ssim: float,
    lpips: float,
    run_fvd: float | None,
    run_fid: float | None,
    method: str,
) -> str:
    idx = extract_index(video_name)
    slug = sanitize_caption(caption)
    fvd_str = f"FVD-{run_fvd:.1f}" if run_fvd is not None else "FVD-NA"
    fid_str = f"FID-{run_fid:.1f}" if run_fid is not None else "FID-NA"
    return (
        f"{idx}_{slug}_{fvd_str}"
        f"_PSNR-{psnr:.3f}_SSIM-{ssim:.3f}_LPIPS-{lpips:.3f}"
        f"_{fid_str}_{method}.mp4"
    )


def process_run_dir(run_dir: str, method_override: str | None, dry_run: bool):
    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    videos_dir = run_path / "videos"

    if not summary_path.exists():
        print(f"  SKIP: no summary.json in {run_dir}")
        return
    if not videos_dir.exists():
        print(f"  SKIP: no videos/ dir in {run_dir}")
        return

    with open(summary_path) as f:
        summary = json.load(f)

    method = determine_method(summary, method_override)
    run_fvd = summary.get("fvd") or summary.get("online_fvd")
    run_fid = summary.get("fid") or summary.get("online_fid")
    results = summary.get("results", [])

    existing_mp4s = {p.name: p for p in videos_dir.glob("*.mp4")}

    renamed = 0
    for r in results:
        if not r.get("success", False):
            continue
        vname = r.get("video_name", "")
        caption = r.get("caption", "unknown")
        psnr = r.get("psnr")
        ssim = r.get("ssim")
        lpips_val = r.get("lpips")
        if psnr is None or ssim is None or lpips_val is None:
            continue

        old_candidates = [
            f"{vname}_delta_a.mp4",
            f"{vname}_lora.mp4",
            f"{vname}_full.mp4",
            f"{vname}_tinylora.mp4",
            f"{vname}.mp4",
        ]
        old_path = None
        for cand in old_candidates:
            if cand in existing_mp4s:
                old_path = existing_mp4s[cand]
                break

        if old_path is None:
            continue

        new_name = build_new_filename(
            vname, caption, psnr, ssim, lpips_val, run_fvd, run_fid, method)
        new_path = videos_dir / new_name

        if old_path == new_path:
            continue

        if dry_run:
            print(f"    {old_path.name}  ->  {new_name}")
        else:
            old_path.rename(new_path)
        renamed += 1

    action = "would rename" if dry_run else "renamed"
    print(f"  {action} {renamed}/{len(results)} videos "
          f"(method={method}, FVD={run_fvd}, FID={run_fid})")


def main():
    p = argparse.ArgumentParser(
        description="Rename experiment videos to include metrics in filename.")
    p.add_argument("--run-dir", action="append", required=True,
                   help="Path(s) to run result directory (can repeat)")
    p.add_argument("--method", type=str, default=None,
                   help="Override method name in filename (e.g. 'no-TTA', 'adasteer')")
    p.add_argument("--dry-run", action="store_true",
                   help="Print renames without executing")
    args = p.parse_args()

    for rd in args.run_dir:
        print(f"\n=== {rd} ===")
        process_run_dir(rd, args.method, args.dry_run)

    if args.dry_run:
        print("\n(dry run — no files were modified)")


if __name__ == "__main__":
    main()
