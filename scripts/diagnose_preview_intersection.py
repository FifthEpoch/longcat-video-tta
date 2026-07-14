#!/usr/bin/env python3
"""Explain why the 12-config PSNR intersection is small and scope the rerun.

Each config's per-video metrics are keyed by a canonical video id derived from
the video filename. If configs launched against an unstable / shifting dataset
(missing symlinks during the first sweep), they end up scoring *different*
physical videos even though each looks ~complete. This tool quantifies that so
we only resubmit the misaligned configs (and, per config, which chunks) instead
of blindly rerunning the whole grid.

Usage:
  python3 scripts/diagnose_preview_intersection.py \
      --series-root sweep_experiment/results/panda_ood_budget_1000v_preview \
      --retain-json sweep_experiment/lists/panda_ood_budget_1000v_preview_videos.json \
      --reference S10_LR1e3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.analyze_adasteer_budget_oracle import PILOT_GRID_RUN_ORDER  # noqa: E402
from scripts.analyze_per_video_tta_gain import (  # noqa: E402
    _canonical_video_id,
    _records_from_blob,
    load_per_video_metrics,
)


def _finite_ids(run_dir: Path) -> Set[str]:
    ids: Set[str] = set()
    for vid, m in load_per_video_metrics(run_dir).items():
        p = m.get("psnr")
        if p is not None and not math.isnan(p):
            ids.add(vid)
    return ids


def _chunk_ids(run_dir: Path) -> Dict[int, Set[str]]:
    """Per-chunk finite-PSNR id sets so we can pinpoint bad chunks."""
    out: Dict[int, Set[str]] = {}
    for cf in sorted(run_dir.glob("chunk_*/summary.json")):
        try:
            idx = int(cf.parent.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        try:
            with cf.open() as f:
                blob = json.load(f)
        except OSError:
            continue
        ids: Set[str] = set()
        for r in _records_from_blob(blob):
            raw = (r.get("video_name") or r.get("video_id") or r.get("video")
                   or r.get("video_path") or r.get("path"))
            vid = _canonical_video_id(raw if raw is not None else "")
            psnr = r.get("psnr", r.get("avg_psnr"))
            try:
                ok = psnr is not None and not math.isnan(float(psnr))
            except (TypeError, ValueError):
                ok = False
            if vid and ok:
                ids.add(vid)
        out[idx] = ids
    return out


def _load_retain_ids(retain_json: Path) -> Set[str]:
    if not retain_json.exists():
        return set()
    with retain_json.open() as f:
        blob = json.load(f)
    if isinstance(blob, dict):
        items = blob.get("videos") or blob.get("video_ids") or blob.get("ids") or []
    else:
        items = blob
    out: Set[str] = set()
    for it in items:
        if isinstance(it, dict):
            it = it.get("video_id") or it.get("video_name") or it.get("path") or ""
        vid = _canonical_video_id(str(it))
        if vid:
            out.add(vid)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--series-root",
        type=Path,
        default=_REPO / "sweep_experiment/results/panda_ood_budget_1000v_preview",
    )
    ap.add_argument(
        "--retain-json",
        type=Path,
        default=_REPO / "sweep_experiment/lists/panda_ood_budget_1000v_preview_videos.json",
    )
    ap.add_argument("--reference", default="S10_LR1e3",
                    help="Config known to be run against the current dataset.")
    ap.add_argument("--overlap-threshold", type=float, default=0.95,
                    help="Configs below this overlap-with-reference are flagged for rerun.")
    ap.add_argument("--per-chunk", action="store_true",
                    help="Also print per-chunk overlap for flagged configs.")
    args = ap.parse_args()

    present = [r for r in PILOT_GRID_RUN_ORDER if (args.series_root / r).is_dir()]
    if not present:
        print(f"No configs under {args.series_root}", file=sys.stderr)
        return 2

    id_sets = {r: _finite_ids(args.series_root / r) for r in present}
    retain = _load_retain_ids(args.retain_json)

    ref = args.reference
    if ref not in id_sets:
        ref = max(id_sets, key=lambda r: len(id_sets[r]))
        print(f"[warn] reference not found; using largest config {ref}", file=sys.stderr)
    ref_ids = id_sets[ref]

    print(f"Series     : {args.series_root}")
    print(f"Reference  : {ref}  ({len(ref_ids)} finite ids)")
    if retain:
        print(f"Retain list: {args.retain_json.name}  ({len(retain)} ids)")
    print()

    intersection: Optional[Set[str]] = None
    for r in present:
        intersection = id_sets[r] if intersection is None else intersection & id_sets[r]

    hdr = f"{'config':<12} {'finite':>7} {'∩ref':>7} {'ovl%':>6}"
    if retain:
        hdr += f" {'∈retain':>8}"
    hdr += "  flag"
    print(hdr)
    print("-" * len(hdr))

    rerun: List[str] = []
    for r in present:
        ids = id_sets[r]
        inter_ref = len(ids & ref_ids)
        ovl = inter_ref / len(ref_ids) if ref_ids else float("nan")
        in_retain = len(ids & retain) if retain else 0
        flag = "" if (r == ref or ovl >= args.overlap_threshold) else "RERUN"
        if flag:
            rerun.append(r)
        line = f"{r:<12} {len(ids):>7} {inter_ref:>7} {100 * ovl:>5.1f}%"
        if retain:
            line += f" {in_retain:>8}"
        line += f"  {flag}"
        print(line)

    print()
    print(f"videos finite in ALL {len(present)} configs: {len(intersection or set())}")
    aligned = [r for r in present if r not in rerun]
    if aligned:
        aligned_inter: Optional[Set[str]] = None
        for r in aligned:
            aligned_inter = id_sets[r] if aligned_inter is None else aligned_inter & id_sets[r]
        print(f"videos finite in the {len(aligned)} ALIGNED configs: "
              f"{len(aligned_inter or set())}")

    if rerun:
        print()
        print(f"CONFIGS TO RERUN ({len(rerun)}, "
              f"{len(rerun) * 10} chunk-jobs @ 10 chunks):")
        print("  " + " ".join(rerun))
        print()
        print("Rerun only these (not the whole grid):")
        print(f'  ONLY_RUNS="{" ".join(rerun)}" CONFIRM=1 bash scripts/wipe_preview_1000v_sweep.sh')
        print(f'  ONLY_RUNS="{" ".join(rerun)}" bash sweep_experiment/sbatch/submit_adasteer_budget_1000v_preview.sh')

        if args.per_chunk:
            print()
            print("Per-chunk overlap with reference (flagged configs):")
            for r in rerun:
                ch = _chunk_ids(args.series_root / r)
                bad = []
                for idx in sorted(ch):
                    c_ovl = len(ch[idx] & ref_ids) / len(ref_ids) if ref_ids else 0.0
                    tag = "" if c_ovl >= args.overlap_threshold else " <-"
                    if tag:
                        bad.append(idx)
                    print(f"  {r} chunk_{idx:<2} n={len(ch[idx]):>3} ovl={100 * c_ovl:5.1f}%{tag}")
                if bad:
                    print(f"    -> bad chunks in {r}: {bad}")
    else:
        print()
        print("All present configs align with the reference — no rerun needed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
