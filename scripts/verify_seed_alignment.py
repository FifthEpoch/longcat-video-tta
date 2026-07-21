#!/usr/bin/env python3
"""Verify per-video generation-seed alignment between two runs (cluster).

The harnesses seed each generation as ``base_seed + local_chunk_index + step_i``
where ``local_chunk_index`` is the video's position within its chunk's
``eval_videos`` slice (i.e. the ``enumerate`` order, which is exactly the order
``per_video_results`` are written to ``summary.json``). So for the evaluated
first rollout (step_i = 0) the seed a run used for a given video is:

    seed(video) = base_seed + position_within_its_chunk

This script reconstructs that per-video seed for two runs from their chunk
``summary.json`` files and reports how many shared videos got the SAME seed.
Use it to confirm that configs and NO-TTA (the arms we compare per-video) were
generated with matched seeds — and to quantify any mismatch before deciding to
re-generate.

Usage:
    python3 scripts/verify_seed_alignment.py \
        --run-a sweep_experiment/results/panda_ood_budget_1000v_preview/S10_LR5e3 \
        --run-b sweep_experiment/results/panda_ood_budget_1000v_preview/NOTTA

    # pilot config vs the cross-series NO-TTA it was joined against:
    python3 scripts/verify_seed_alignment.py \
        --run-a sweep_experiment/results/panda_ood_budget_pilot/S10_LR5e3 \
        --run-b sweep_experiment/results/panda_1000v_standard/NOTTA
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.caption_utils import canonical_video_id  # noqa: E402


def _reconstruct_seeds(run_dir: Path, base_seed: int) -> Dict[str, int]:
    """Map canonical video_id -> reconstructed generation seed (step_i = 0).

    Position within each chunk's per-video list == the local index used to seed
    generation, so seed = base_seed + position.
    """
    chunk_dirs = sorted(run_dir.glob("chunk_*/"))
    if not chunk_dirs:
        chunk_dirs = [run_dir]
    seeds: Dict[str, int] = {}
    for chunk_dir in chunk_dirs:
        summ = chunk_dir / "summary.json"
        if not summ.exists():
            continue
        with summ.open(encoding="utf-8") as f:
            data = json.load(f)
        recs = data.get("per_video_results", data.get("results", []))
        for pos, rec in enumerate(recs):
            vname = rec.get("video_name") or rec.get("video_id") or ""
            vid = canonical_video_id(vname)
            if not vid:
                continue
            seed = base_seed + pos  # step_i = 0 (evaluated first rollout)
            seeds.setdefault(vid, seed)
    return seeds


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify per-video seed alignment between two runs")
    ap.add_argument("--run-a", type=Path, required=True, help="Run dir (e.g. .../S10_LR5e3)")
    ap.add_argument("--run-b", type=Path, required=True, help="Run dir (e.g. .../NOTTA)")
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=10, help="Sample mismatches to print")
    args = ap.parse_args()

    sa = _reconstruct_seeds(args.run_a, args.base_seed)
    sb = _reconstruct_seeds(args.run_b, args.base_seed)
    print(f"run-a: {args.run_a}  ({len(sa)} videos)")
    print(f"run-b: {args.run_b}  ({len(sb)} videos)")

    common = sorted(set(sa) & set(sb))
    if not common:
        print("\n[!] NO shared video ids — cannot compare (check paths).")
        return 1

    match = [v for v in common if sa[v] == sb[v]]
    mism = [v for v in common if sa[v] != sb[v]]
    rate = len(match) / len(common) * 100.0
    print(f"\nshared videos : {len(common)}")
    print(f"seed MATCH    : {len(match)} ({rate:.1f}%)")
    print(f"seed MISMATCH : {len(mism)} ({100 - rate:.1f}%)")

    if mism:
        print(f"\nsample mismatches (video_id: seed_a vs seed_b), up to {args.show}:")
        for v in mism[: args.show]:
            print(f"  {v}: {sa[v]} vs {sb[v]}")

    if rate >= 99.9:
        print("\nVERDICT: seeds are aligned — per-video comparison is seed-matched. No regen needed.")
    elif rate <= 5.0:
        print("\nVERDICT: seeds are essentially UNMATCHED — these arms used different initial "
              "noise per video. Re-generate run-b on run-a's pool + chunking to align.")
    else:
        print("\nVERDICT: PARTIAL alignment — investigate pool ordering / chunking differences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
