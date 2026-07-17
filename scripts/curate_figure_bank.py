#!/usr/bin/env python3
"""Curate a small, protected 'figure bank' of example videos before cleanup.

Paper qualitative panels need only a handful of examples per method, not the
full 1000xN generated set. This copies a matched set of example clips — the
SAME canonical video ids across every method in a series, so they line up in a
side-by-side figure — into a protected directory that the bulk-delete step
skips.

Selection:
  * candidate ids = intersection of canonical ids present across all selected
    methods (so every panel row exists for every method)
  * if --ood-csv is given, pick evenly across the 5 OOD quintiles (easy->hard);
    otherwise pick evenly spaced across the sorted id list
  * --per-series N controls how many ids to keep (default 10)

Standard cluster usage (user runs this; no slurm):

    python3 scripts/curate_figure_bank.py \\
        --series-root sweep_experiment/results/panda_1000v_standard \\
        --dest sweep_experiment/figure_bank \\
        --per-series 10 \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv

Copies to <dest>/<series_name>/<run>/<original_filename>. Re-runnable
(skips files already copied). Stdlib only.
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional

_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def _canonical_video_id(s: Optional[str]) -> str:
    if not s:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


def discover_runs(series_root: Path) -> List[Path]:
    return [p for p in sorted(series_root.iterdir())
            if p.is_dir() and any(p.rglob("*.mp4"))]


def run_id_to_files(run_dir: Path) -> Dict[str, Path]:
    """canonical id -> first mp4 path for that id under the run."""
    out: Dict[str, Path] = {}
    for mp4 in sorted(run_dir.rglob("*.mp4")):
        cid = _canonical_video_id(mp4.name)
        if cid and cid not in out:
            out[cid] = mp4
    return out


def load_ood_quintiles(path: Path) -> Dict[str, int]:
    """canonical id -> quintile 1..5 (1=most in-dist). Tolerant of schema."""
    if not path or not path.is_file():
        return {}
    rows = []
    with path.open(newline="") as f:
        for r in csv.DictReader(f):
            vid = _canonical_video_id(r.get("video_id") or r.get("video_name") or "")
            score = r.get("diffusion_ood_score") or r.get("ood_score") or r.get("ood")
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            if vid:
                rows.append((vid, score))
    if not rows:
        return {}
    rows.sort(key=lambda x: x[1])
    n = len(rows)
    out: Dict[str, int] = {}
    for i, (vid, _) in enumerate(rows):
        out[vid] = min(5, 1 + (i * 5) // n)
    return out


def pick_ids(candidates: List[str], n: int,
             quintiles: Dict[str, int]) -> List[str]:
    candidates = sorted(candidates)
    if n >= len(candidates):
        return candidates
    if quintiles:
        by_q: Dict[int, List[str]] = {}
        for cid in candidates:
            by_q.setdefault(quintiles.get(cid, 0), []).append(cid)
        picked: List[str] = []
        per_q = max(1, n // max(1, len(by_q)))
        for q in sorted(by_q):
            bucket = sorted(by_q[q])
            step = max(1, len(bucket) // per_q)
            picked.extend(bucket[::step][:per_q])
        return sorted(set(picked))[:n]
    step = max(1, len(candidates) // n)
    return candidates[::step][:n]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series-root", type=Path, required=True)
    ap.add_argument("--dest", type=Path, default=Path("sweep_experiment/figure_bank"))
    ap.add_argument("--per-series", type=int, default=10)
    ap.add_argument("--ood-csv", type=Path, default=None)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="restrict to these run/method subdir names")
    ap.add_argument("--ids-file", type=Path, default=None,
                    help="newline-separated canonical ids to pin (e.g. a "
                         "pool_ids/<fp>.txt from build_run_manifest.py); "
                         "guarantees the SAME ids across series sharing a pool")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.series_root.is_dir():
        print(f"[error] series root not found: {args.series_root}", file=sys.stderr)
        return 2

    runs = discover_runs(args.series_root)
    if args.methods:
        runs = [r for r in runs if r.name in set(args.methods)]
    if not runs:
        print(f"[error] no runs with mp4s under {args.series_root}", file=sys.stderr)
        return 2

    run_files = {r.name: run_id_to_files(r) for r in runs}
    # matched candidate ids = present in EVERY selected run
    common = set.intersection(*[set(f.keys()) for f in run_files.values()]) \
        if run_files else set()
    if not common:
        print("[warn] no canonical id common to all runs; "
              "falling back to union (panels may be incomplete)", file=sys.stderr)
        common = set().union(*[set(f.keys()) for f in run_files.values()])

    if args.ids_file and args.ids_file.is_file():
        pinned = {ln.strip() for ln in args.ids_file.read_text().splitlines() if ln.strip()}
        missing = pinned - common
        common = common & pinned
        print(f"[info] pinned to {len(pinned)} ids from {args.ids_file}; "
              f"{len(common)} present here, {len(missing)} absent", file=sys.stderr)
        if not common:
            print("[error] none of the pinned ids exist in this series", file=sys.stderr)
            return 2

    quintiles = load_ood_quintiles(args.ood_csv) if args.ood_csv else {}
    picked = pick_ids(sorted(common), args.per_series, quintiles)
    print(f"[info] {args.series_root.name}: {len(runs)} runs, "
          f"{len(common)} matched ids, keeping {len(picked)}", file=sys.stderr)
    if quintiles:
        dist = {}
        for cid in picked:
            dist[quintiles.get(cid, 0)] = dist.get(quintiles.get(cid, 0), 0) + 1
        print(f"[info] quintile spread of picks: {dict(sorted(dist.items()))}",
              file=sys.stderr)

    series_name = args.series_root.name
    n_copied = 0
    for run in runs:
        files = run_files[run.name]
        for cid in picked:
            src = files.get(cid)
            if src is None:
                continue
            dst = args.dest / series_name / run.name / src.name
            if dst.exists():
                continue
            if args.dry_run:
                print(f"[dry] cp {src} -> {dst}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            n_copied += 1
    verb = "would copy" if args.dry_run else "copied"
    print(f"[info] {verb} {n_copied} clips into {args.dest / series_name}",
          file=sys.stderr)
    # write a small provenance note per series
    if not args.dry_run and n_copied:
        note = args.dest / series_name / "FIGURE_BANK.txt"
        note.write_text(
            f"source series: {args.series_root}\n"
            f"kept ids ({len(picked)}): {', '.join(picked)}\n"
            f"runs: {', '.join(sorted(run_files))}\n"
            f"ood-stratified: {bool(quintiles)}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
