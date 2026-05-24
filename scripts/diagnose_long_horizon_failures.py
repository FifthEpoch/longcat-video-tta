#!/usr/bin/env python3
"""Diagnose long-horizon AdaSteer regression on a per-video basis.

Compares per-video PSNR/SSIM/LPIPS between a baseline run (No-TTA) and a
treatment run (e.g. AdaSteer S10), joins to a Panda-70M-style metadata.csv
(filename, caption, [category]), and reports per-theme mean deltas plus the
top-K worst videos.

Inputs:
    --notta-dir   Directory containing chunk_N/summary.json for No-TTA.
    --treat-dir   Directory containing chunk_N/summary.json for the treatment
                  (e.g. AdaSteer S10).
    --dataset-dir Dataset root containing metadata.csv (filename, caption).
    --out-csv     Output CSV path (per-video deltas + caption).
    --top-k       Number of worst-performers to print (default 25).

Stdout:
    Summary table with overall means, per-theme means (motion/static/sport/
    cooking/nature/animal/vehicle/talking_head/other), and quintile buckets
    of No-TTA PSNR (does AdaSteer hurt high-quality or low-quality videos
    more?). The summary is plain text and safe to paste into chat.

The script is stdlib-only (no numpy/pandas dependency) so it runs on any
cluster node without an environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional


# Coarse caption-keyword taxonomy. Order matters: a video falls into the first
# matching bucket. Tuned to common Panda-70M caption phrasing.
THEME_PATTERNS = [
    ("sport",        r"\b(sport|soccer|football|basketball|baseball|tennis|golf|skiing|surfing|skateboard|race|racing|running|sprint|swim|swimming|boxing|wrestling|martial|fight|kick)\b"),
    ("dance_music",  r"\b(dance|dancing|dancer|music|concert|sing|singing|band|orchestra|guitar|piano|drum|drumming|perform)\b"),
    ("cooking",      r"\b(cook|cooking|kitchen|recipe|bake|baking|chef|food|meal|fry|frying|grill|grilling|chop|chopping|stir|mix|mixing)\b"),
    ("nature",       r"\b(nature|forest|tree|ocean|sea|beach|wave|river|mountain|sky|cloud|sunset|sunrise|landscape|wildlife|outdoor)\b"),
    ("animal",       r"\b(dog|cat|bird|horse|cow|sheep|lion|tiger|bear|elephant|fish|monkey|animal|pet|puppy|kitten|squirrel|deer)\b"),
    ("vehicle",      r"\b(car|truck|motorcycle|bike|bicycle|driving|train|airplane|plane|boat|ship|vehicle|drone|skateboard|scooter)\b"),
    ("talking_head", r"\b(interview|talk|talking|speak|speaking|presenter|presentation|host|tutorial|lecture|explain|explaining|news|anchor)\b"),
    ("crowd",        r"\b(crowd|audience|people gathered|protest|rally|parade|festival|wedding|ceremony)\b"),
    ("indoor_misc",  r"\b(room|indoor|office|home|living room|bedroom|bathroom|sit|sitting|chair|couch|table)\b"),
]

THEME_NAMES = [name for name, _ in THEME_PATTERNS] + ["other"]


def load_per_video_records(run_dir: Path) -> Dict[str, Dict]:
    """Return {video_name: record} from all chunk_N/summary.json under run_dir.

    Accepts either run_dir directly containing chunk_N/ subdirs or a parent
    that contains a single nested run directory.
    """
    chunks = sorted(run_dir.glob("chunk_*/summary.json"))
    if not chunks:
        chunks = sorted(run_dir.glob("*/chunk_*/summary.json"))
    if not chunks:
        single = run_dir / "summary.json"
        if single.exists():
            chunks = [single]
    if not chunks:
        raise SystemExit(f"No chunk_N/summary.json under {run_dir}")

    records: Dict[str, Dict] = {}
    for cpath in chunks:
        with cpath.open() as f:
            blob = json.load(f)
        per_video = blob.get("per_video_results", blob.get("results", []))
        for r in per_video:
            name = r.get("video_name") or r.get("video_id")
            if not name:
                continue
            records[name] = r
    return records


def load_captions(dataset_dir: Path) -> Dict[str, Dict[str, str]]:
    meta_path = dataset_dir / "metadata.csv"
    if not meta_path.exists():
        return {}
    out: Dict[str, Dict[str, str]] = {}
    with meta_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            fname = row.get("filename") or row.get("video_path") or row.get("path")
            if not fname:
                continue
            stem = Path(fname).stem
            out[fname] = row
            out[stem] = row
    return out


def classify_theme(caption: str) -> str:
    text = (caption or "").lower()
    for name, pat in THEME_PATTERNS:
        if re.search(pat, text):
            return name
    return "other"


def safe_delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return a - b


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return statistics.fmean(clean)


def quintile_buckets(values: List[float]) -> List[float]:
    s = sorted(values)
    n = len(s)
    if n < 5:
        return [float("-inf"), float("inf")]
    return [s[int(n * q)] for q in (0.2, 0.4, 0.6, 0.8)]


def bucket_of(value: float, edges: List[float]) -> int:
    for i, edge in enumerate(edges):
        if value <= edge:
            return i
    return len(edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notta-dir", required=True, type=Path)
    parser.add_argument("--treat-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path,
                        help="Panda-style dataset root with metadata.csv")
    parser.add_argument("--out-csv", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--treat-label", default="treat",
                        help="Short label for the treatment column (default 'treat').")
    args = parser.parse_args()

    notta = load_per_video_records(args.notta_dir)
    treat = load_per_video_records(args.treat_dir)
    captions = load_captions(args.dataset_dir)

    shared = sorted(set(notta) & set(treat))
    print(f"No-TTA records: {len(notta)}  |  {args.treat_label} records: {len(treat)}  |  joined: {len(shared)}")
    if not shared:
        raise SystemExit("No overlapping video_name keys between the two runs.")

    rows: List[Dict[str, Optional[float]]] = []
    for name in shared:
        n_rec = notta[name]
        t_rec = treat[name]
        cap_row = captions.get(name) or captions.get(Path(name).stem) or {}
        caption = cap_row.get("caption", "")
        category = cap_row.get("category", "")
        theme = classify_theme(caption)
        rows.append({
            "video": name,
            "category": category,
            "theme": theme,
            "psnr_notta": n_rec.get("psnr"),
            "psnr_treat": t_rec.get("psnr"),
            "dpsnr": safe_delta(t_rec.get("psnr"), n_rec.get("psnr")),
            "ssim_notta": n_rec.get("ssim"),
            "ssim_treat": t_rec.get("ssim"),
            "dssim": safe_delta(t_rec.get("ssim"), n_rec.get("ssim")),
            "lpips_notta": n_rec.get("lpips"),
            "lpips_treat": t_rec.get("lpips"),
            "dlpips": safe_delta(t_rec.get("lpips"), n_rec.get("lpips")),
            "caption": caption,
        })

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote per-video deltas to {args.out_csv}")

    # ---------------- Overall summary ----------------
    print("\n=== Overall mean deltas (treat - notta; positive PSNR/SSIM = better; negative LPIPS = better) ===")
    for key, sign in (("dpsnr", "+"), ("dssim", "+"), ("dlpips", "-")):
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            continue
        m = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        wins = sum(1 for v in vals if (v > 0 if sign == "+" else v < 0))
        ties = sum(1 for v in vals if v == 0)
        losses = len(vals) - wins - ties
        print(f"  {key:7s}  mean={m:+.4f}  std={sd:.4f}  N={len(vals):4d}  wins={wins}  ties={ties}  losses={losses}")

    # ---------------- Per-theme summary ----------------
    print("\n=== Per-theme mean dPSNR / dSSIM / dLPIPS  (N = videos per theme) ===")
    print(f"  {'theme':14s}  {'N':>4s}  {'dPSNR':>10s}  {'dSSIM':>10s}  {'dLPIPS':>10s}")
    for theme in THEME_NAMES:
        subset = [r for r in rows if r["theme"] == theme]
        if not subset:
            continue
        m_psnr = mean(r["dpsnr"] for r in subset)
        m_ssim = mean(r["dssim"] for r in subset)
        m_lpi = mean(r["dlpips"] for r in subset)
        fp = lambda v: f"{v:+.4f}" if v is not None else "  n/a   "
        print(f"  {theme:14s}  {len(subset):4d}  {fp(m_psnr):>10s}  {fp(m_ssim):>10s}  {fp(m_lpi):>10s}")

    # ---------------- Quintile buckets on No-TTA PSNR ----------------
    print("\n=== Quintile buckets on No-TTA PSNR (does AdaSteer hurt good or bad videos more?) ===")
    psnr_vals = [r["psnr_notta"] for r in rows if r["psnr_notta"] is not None]
    if len(psnr_vals) >= 5:
        edges = quintile_buckets(psnr_vals)
        buckets: Dict[int, List[Dict]] = defaultdict(list)
        for r in rows:
            if r["psnr_notta"] is None:
                continue
            buckets[bucket_of(r["psnr_notta"], edges)].append(r)
        print(f"  {'bucket':8s}  {'PSNR range':>20s}  {'N':>4s}  {'dPSNR':>10s}  {'dSSIM':>10s}  {'dLPIPS':>10s}")
        edges_full = [min(psnr_vals)] + edges + [max(psnr_vals)]
        for i in range(5):
            subset = buckets.get(i, [])
            if not subset:
                continue
            rng = f"[{edges_full[i]:.2f}, {edges_full[i+1]:.2f}]"
            m_psnr = mean(r["dpsnr"] for r in subset)
            m_ssim = mean(r["dssim"] for r in subset)
            m_lpi = mean(r["dlpips"] for r in subset)
            fp = lambda v: f"{v:+.4f}" if v is not None else "  n/a   "
            print(f"  Q{i+1:<6d}  {rng:>20s}  {len(subset):4d}  {fp(m_psnr):>10s}  {fp(m_ssim):>10s}  {fp(m_lpi):>10s}")

    # ---------------- Worst videos by dPSNR ----------------
    valid = [r for r in rows if r["dpsnr"] is not None]
    worst = sorted(valid, key=lambda r: r["dpsnr"])[: args.top_k]
    print(f"\n=== Top {args.top_k} worst videos by dPSNR (AdaSteer regressions) ===")
    print(f"  {'#':3s}  {'theme':14s}  {'dPSNR':>8s}  {'dSSIM':>8s}  {'dLPIPS':>8s}  video  caption")
    for i, r in enumerate(worst, 1):
        caption_snippet = (r["caption"] or "")[:80]
        ds = f"{r['dssim']:+.4f}" if r["dssim"] is not None else "  n/a "
        dl = f"{r['dlpips']:+.4f}" if r["dlpips"] is not None else "  n/a "
        print(f"  {i:<3d}  {r['theme']:14s}  {r['dpsnr']:+.4f}  {ds:>8s}  {dl:>8s}  {r['video']}  {caption_snippet}")

    # ---------------- Best videos by dPSNR ----------------
    best = sorted(valid, key=lambda r: r["dpsnr"], reverse=True)[: args.top_k]
    print(f"\n=== Top {args.top_k} best videos by dPSNR (AdaSteer gains) ===")
    print(f"  {'#':3s}  {'theme':14s}  {'dPSNR':>8s}  {'dSSIM':>8s}  {'dLPIPS':>8s}  video  caption")
    for i, r in enumerate(best, 1):
        caption_snippet = (r["caption"] or "")[:80]
        ds = f"{r['dssim']:+.4f}" if r["dssim"] is not None else "  n/a "
        dl = f"{r['dlpips']:+.4f}" if r["dlpips"] is not None else "  n/a "
        print(f"  {i:<3d}  {r['theme']:14s}  {r['dpsnr']:+.4f}  {ds:>8s}  {dl:>8s}  {r['video']}  {caption_snippet}")


if __name__ == "__main__":
    main()
