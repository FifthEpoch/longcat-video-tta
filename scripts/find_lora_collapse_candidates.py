#!/usr/bin/env python3
"""Find UCF-101 videos with the largest LoRA single-clip collapse.

Joins a No-TTA run and a LoRA run on the same video pool, computes the
per-video PSNR gap (NOTTA_psnr - LORA_psnr), sorts descending, and writes
the top-K candidates to a JSON retain-list + a CSV with full details.

The K candidates are diversified by UCF-101 action class so the eventual
cover panel isn't four clips of the same activity. The action class is
inferred from the UCF naming convention ``v_<ActionClass>_gNN_cNN``; we
fall back to ``other`` if a filename does not match.

Outputs (under --out-dir):
    retain.json          {"all": [video_name, ...]}  -- consumed by the
                                                       filmstrip builder
    collapse_gains.csv   per-video deltas + action class + caption
    summary.txt          per-action-class breakdown + top-K printout

Stdlib-only; no numpy/pandas required.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# UCF-101 filename convention: v_<ActionClass>_gNN_cNN[...].mp4
UCF_NAME_RE = re.compile(r"^v_([A-Za-z]+)_g\d+_c\d+", re.IGNORECASE)


def load_per_video(run_dir: Path) -> Dict[str, Dict]:
    """Return {video_name: record} from {run_dir}/summary.json or chunks."""
    candidates: List[Path] = []
    sj = run_dir / "summary.json"
    if sj.exists():
        candidates.append(sj)
    candidates.extend(sorted(run_dir.glob("chunk_*/summary.json")))
    if not candidates:
        raise SystemExit(f"No summary.json under {run_dir}")
    records: Dict[str, Dict] = {}
    for path in candidates:
        with path.open() as f:
            blob = json.load(f)
        pv = blob.get("per_video_results") or blob.get("results") or []
        for r in pv:
            name = r.get("video_name") or r.get("video_id")
            if not name:
                continue
            records[name] = r
    return records


def action_class(video_name: str) -> str:
    """Best-effort UCF action class extraction."""
    stem = Path(video_name).stem
    m = UCF_NAME_RE.match(stem)
    if m:
        return m.group(1)
    return "other"


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return v


def diversify(rows: List[Dict], top_k: int, per_class_cap: int) -> List[Dict]:
    """Greedy diversification: walk rows in collapse-gap order, keep at most
    ``per_class_cap`` from any single action class until we have ``top_k``.

    Falls through to a second pass without the per-class cap if the diverse
    set is smaller than top_k (so we always return top_k entries if
    available).
    """
    chosen: List[Dict] = []
    per_class: Counter = Counter()
    seen = set()
    for r in rows:
        if r["video"] in seen:
            continue
        cls = r["action_class"]
        if per_class[cls] >= per_class_cap:
            continue
        chosen.append(r)
        seen.add(r["video"])
        per_class[cls] += 1
        if len(chosen) >= top_k:
            break
    if len(chosen) < top_k:
        for r in rows:
            if r["video"] in seen:
                continue
            chosen.append(r)
            seen.add(r["video"])
            if len(chosen) >= top_k:
                break
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--notta-dir", required=True, type=Path,
                        help="Directory of the No-TTA run (contains summary.json or chunks).")
    parser.add_argument("--lora-dir", required=True, type=Path,
                        help="Directory of the collapsed LoRA run.")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Optional dataset dir for metadata.csv (caption / class).")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=30,
                        help="How many cover-image candidates to retain.")
    parser.add_argument("--per-class-cap", type=int, default=2,
                        help="Max candidates per UCF action class.")
    parser.add_argument("--min-notta-psnr", type=float, default=15.0,
                        help="Drop candidates whose No-TTA PSNR is below this, "
                             "so the contrast is 'NoTTA fine -> LoRA broken' "
                             "rather than 'both bad'.")
    args = parser.parse_args()

    notta = load_per_video(args.notta_dir)
    lora = load_per_video(args.lora_dir)
    shared = sorted(set(notta) & set(lora))
    if not shared:
        raise SystemExit("No overlap between NOTTA and LORA per-video records.")

    captions: Dict[str, str] = {}
    if args.dataset_dir is not None:
        meta = args.dataset_dir / "metadata.csv"
        if meta.exists():
            with meta.open(newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    fname = row.get("filename") or row.get("video_path") or row.get("path")
                    if not fname:
                        continue
                    cap = row.get("caption") or row.get("class") or row.get("action") or ""
                    captions[Path(fname).stem] = cap
                    captions[fname] = cap

    rows: List[Dict] = []
    for name in shared:
        n_psnr = safe_float(notta[name].get("psnr"))
        l_psnr = safe_float(lora[name].get("psnr"))
        n_ssim = safe_float(notta[name].get("ssim"))
        l_ssim = safe_float(lora[name].get("ssim"))
        n_lpips = safe_float(notta[name].get("lpips"))
        l_lpips = safe_float(lora[name].get("lpips"))
        if n_psnr is None or l_psnr is None:
            continue
        if n_psnr < args.min_notta_psnr:
            continue
        rows.append({
            "video": name,
            "action_class": action_class(name),
            "psnr_notta": n_psnr,
            "psnr_lora": l_psnr,
            "collapse_psnr": n_psnr - l_psnr,
            "ssim_notta": n_ssim,
            "ssim_lora": l_ssim,
            "collapse_ssim": (None if n_ssim is None or l_ssim is None
                              else n_ssim - l_ssim),
            "lpips_notta": n_lpips,
            "lpips_lora": l_lpips,
            "collapse_lpips": (None if n_lpips is None or l_lpips is None
                               else l_lpips - n_lpips),  # higher LPIPS is worse
            "caption": captions.get(name) or captions.get(Path(name).stem, ""),
        })

    # Sort by largest PSNR collapse first.
    rows.sort(key=lambda r: r["collapse_psnr"], reverse=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "collapse_gains.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    chosen = diversify(rows, args.top_k, args.per_class_cap)
    retain_path = args.out_dir / "retain.json"
    retain_path.write_text(json.dumps({"all": [r["video"] for r in chosen]}, indent=2))

    # Human-readable summary.
    out_lines: List[str] = []
    out_lines.append(f"Joined NOTTA <-> LoRA on {len(shared)} videos "
                     f"({len(rows)} after NoTTA-PSNR >= {args.min_notta_psnr} filter).")
    psnrs_n = [r["psnr_notta"] for r in rows]
    psnrs_l = [r["psnr_lora"] for r in rows]
    gaps = [r["collapse_psnr"] for r in rows]
    out_lines.append(
        f"  NOTTA   mean PSNR = {statistics.fmean(psnrs_n):.2f}  "
        f"min = {min(psnrs_n):.2f}  max = {max(psnrs_n):.2f}"
    )
    out_lines.append(
        f"  LoRA    mean PSNR = {statistics.fmean(psnrs_l):.2f}  "
        f"min = {min(psnrs_l):.2f}  max = {max(psnrs_l):.2f}"
    )
    out_lines.append(
        f"  Collapse mean = {statistics.fmean(gaps):+.2f}  "
        f"median = {statistics.median(gaps):+.2f}  "
        f"max = {max(gaps):+.2f}"
    )
    out_lines.append("")
    out_lines.append("Per-action-class candidate count (full pool):")
    cls_counter = Counter(r["action_class"] for r in rows)
    for cls, n in cls_counter.most_common():
        out_lines.append(f"  {cls:<24s} n={n}")
    out_lines.append("")
    out_lines.append(f"Selected top {len(chosen)} (per-class cap = {args.per_class_cap}):")
    out_lines.append(
        f"  {'video':<32s} {'class':<22s} {'NoTTA':>7s} {'LoRA':>7s} {'gap':>7s}"
    )
    out_lines.append("  " + "-" * 80)
    for r in chosen:
        out_lines.append(
            f"  {r['video']:<32s} {r['action_class']:<22s} "
            f"{r['psnr_notta']:>7.2f} {r['psnr_lora']:>7.2f} {r['collapse_psnr']:>+7.2f}"
        )
    summary_path = args.out_dir / "summary.txt"
    summary_path.write_text("\n".join(out_lines) + "\n")

    print("\n".join(out_lines))
    print(f"\nWrote:\n  {csv_path}\n  {retain_path}\n  {summary_path}")


if __name__ == "__main__":
    main()
