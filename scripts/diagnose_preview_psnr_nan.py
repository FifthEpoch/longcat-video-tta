#!/usr/bin/env python3
"""Classify why preview-budget sweep rows lack finite PSNR.

Failure modes (per ``run_delta_a.py`` result dict):
  * ``failed``       — ``success=False`` (exception; no generation metrics)
  * ``no_psnr_key``  — ``success=True`` but generation/metrics block skipped
  * ``psnr_nan``     — ``success=True``, ``psnr`` present but NaN (metric eval failed)
  * ``finite``       — ``success=True`` with finite PSNR (same path as pilot)

Also compares a reference run (default S2_LR1e3) to see whether NaN rows are
video-specific instability at S10 LR vs a global pipeline bug.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.caption_utils import canonical_video_id  # noqa: E402


def _classify(row: dict) -> str:
    if not row.get("success", False):
        return "failed"
    psnr = row.get("psnr")
    if psnr is None:
        return "no_psnr_key"
    try:
        v = float(psnr)
    except (TypeError, ValueError):
        return "no_psnr_key"
    if math.isnan(v):
        return "psnr_nan"
    if math.isinf(v):
        return "psnr_inf"
    return "finite"


def _load_all_results(run_dir: Path) -> List[dict]:
    rows: List[dict] = []
    for summary in sorted(run_dir.glob("chunk_*/summary.json")):
        blob = json.loads(summary.read_text(encoding="utf-8"))
        rows.extend(blob.get("results", []))
    return rows


def _vid(row: dict) -> str:
    raw = row.get("video_name") or row.get("video_id") or row.get("video_path") or ""
    return canonical_video_id(str(raw))


def _summarize_run(run_dir: Path) -> Tuple[Counter, Dict[str, dict]]:
    by_class: Counter = Counter()
    by_vid: Dict[str, dict] = {}
    for row in _load_all_results(run_dir):
        vid = _vid(row)
        if not vid:
            continue
        cls = _classify(row)
        by_class[cls] += 1
        by_vid[vid] = row
    return by_class, by_vid


def _top_errors(rows: List[dict], n: int = 8) -> List[Tuple[str, int]]:
    errs = Counter()
    for row in rows:
        if row.get("success", False):
            continue
        msg = (row.get("error") or "unknown")[:120]
        errs[msg] += 1
    return errs.most_common(n)


def _sample_rows(rows: List[dict], cls: str, n: int = 3) -> List[dict]:
    out = []
    for row in rows:
        if _classify(row) != cls:
            continue
        out.append(row)
        if len(out) >= n:
            break
    return out


def _fmt_row(row: dict) -> str:
    vid = _vid(row)
    parts = [
        f"vid={vid}",
        f"success={row.get('success')}",
        f"psnr={row.get('psnr', '—')}",
        f"gen_time={row.get('gen_time', '—')}",
        f"train_time={row.get('train_time', '—')}",
    ]
    if row.get("error"):
        parts.append(f"error={str(row.get('error'))[:80]}")
    if "tta_skipped" in row:
        parts.append(f"tta_skipped={row.get('tta_skipped')}")
    if row.get("anchor_gate_decision"):
        parts.append(f"anchor={row.get('anchor_gate_decision')}")
    if row.get("clip_gate_decision"):
        parts.append(f"clip={row.get('clip_gate_decision')}")
    return "  " + " | ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--series-root",
        type=Path,
        default=_REPO / "sweep_experiment/results/panda_ood_budget_1000v_preview",
    )
    ap.add_argument(
        "--target-runs",
        nargs="+",
        default=["S10_LR1e3", "S10_LR5e3"],
    )
    ap.add_argument(
        "--reference-run",
        default="S2_LR1e3",
        help="Run where most videos have finite PSNR (sanity baseline).",
    )
    args = ap.parse_args()

    ref_dir = args.series_root / args.reference_run
    if not ref_dir.is_dir():
        print(f"[error] reference run missing: {ref_dir}", file=sys.stderr)
        return 2

    ref_counts, ref_by_vid = _summarize_run(ref_dir)
    ref_finite = {v for v, r in ref_by_vid.items() if _classify(r) == "finite"}

    print(f"Series: {args.series_root}")
    print(f"Reference: {args.reference_run}")
    print(f"  classes: {dict(ref_counts)}")
    print()

    for run_id in args.target_runs:
        run_dir = args.series_root / run_id
        if not run_dir.is_dir():
            print(f"{run_id}: MISSING")
            continue

        rows = _load_all_results(run_dir)
        counts, by_vid = _summarize_run(run_dir)
        finite = {v for v, r in by_vid.items() if _classify(r) == "finite"}
        bad = ref_finite - finite
        good_both = ref_finite & finite

        print("=" * 72)
        print(run_id)
        print(f"  classes: {dict(counts)}")
        print(f"  finite PSNR: {len(finite)}")
        print(f"  ref-finite but not here: {len(bad)}")
        print(f"  finite in BOTH ref+target: {len(good_both)}")

        if counts.get("failed"):
            print("  top errors (failed rows):")
            for msg, c in _top_errors(rows):
                print(f"    [{c}] {msg}")

        for cls in ("psnr_nan", "no_psnr_key", "failed", "finite"):
            samples = _sample_rows(rows, cls, n=2)
            if samples:
                print(f"  sample {cls}:")
                for s in samples:
                    print(_fmt_row(s))

        # PSNR agreement on overlap
        diffs = []
        for vid in sorted(good_both)[:2000]:
            p_ref = float(ref_by_vid[vid]["psnr"])
            p_tgt = float(by_vid[vid]["psnr"])
            diffs.append(p_tgt - p_ref)
        if diffs:
            import numpy as np

            arr = np.asarray(diffs, dtype=float)
            print(
                f"  PSNR(ref) - PSNR({run_id}) on shared finite vids: "
                f"mean={arr.mean():.3f}  std={arr.std():.3f}  "
                f"min={arr.min():.3f}  max={arr.max():.3f}"
            )
            print(
                "  (Large spread is expected across LR/steps; identical values would be suspicious.)"
            )

        if bad:
            print(f"  sample ref-finite / target-bad vids: {sorted(bad)[:5]}")
            for vid in sorted(bad)[:2]:
                r = by_vid.get(vid, {})
                print(f"    {vid}: class={_classify(r)} {_fmt_row(r).strip()}")
        print()

    print("Interpretation guide:")
    print("  - finite rows: same metric path as pilot; trustworthy if success=True.")
    print("  - failed: exception during train/gen — check 'top errors'.")
    print("  - psnr_nan: generation ran but GT/gen frame compare failed.")
    print("  - no_psnr_key: generation block skipped (rare with default sweep flags).")
    print("  - merge population PSNR=nan can happen when a few NaN rows poison np.mean")
    print("    (display bug); per-video chunk data may still be mostly fine.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
