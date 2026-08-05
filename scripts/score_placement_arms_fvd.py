#!/usr/bin/env python3
"""Matched-N FVD for the EXP2 placement arms (ADA_ADALN, ADA_RESID) vs NO-TTA.

FVD is a *distribution* metric, so all policies MUST be scored on the SAME video
set (comparing an 80-video FVD to the 900-video NOTTA 157.05 would repeat the old
N-mismatch confound). This script:

  1. indexes the clips for NOTTA (from the preview series) and each placement arm,
  2. restricts to the COMMON video ids (intersection),
  3. symlinks each policy's clips into its own dir on that common set,
  4. runs eval_fvd.py against the shared preview GT cache with the SAME 14/14
     window + --force (N<256), identically for every policy.

Because the slicing/window is identical across policies, the COMPARISON is fair
even if the absolute FVD differs slightly from a full-N run.

Usage (cluster, GPU):
  python3 scripts/score_placement_arms_fvd.py \
    --placement-series sweep_experiment/results/placement_ablation_panda \
    --preview-series   sweep_experiment/results/panda_ood_budget_1000v_preview \
    --gt-cache gt_caches/panda_ood_budget_1000v_preview_longcat.npz \
    --arms ADA_ADALN ADA_RESID --notta-run NOTTA \
    --output-root sweep_experiment/reports/budget_oracle_fvd_1000v_preview/placement_arms
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from sweep_experiment.scripts.build_oracle_policy_dirs import (  # noqa: E402
    index_method_videos,
)
from sweep_experiment.scripts.run_pilot_matched_fvd_baselines import (  # noqa: E402
    run_eval_fvd,
    symlink_policy_dir,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Matched-N FVD for EXP2 placement arms")
    ap.add_argument("--placement-series", type=Path, required=True)
    ap.add_argument("--preview-series", type=Path, required=True,
                    help="Series holding the paired NO-TTA run (same GT-cache pool).")
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["ADA_ADALN", "ADA_RESID"])
    ap.add_argument("--notta-run", default="NOTTA")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-linked", type=int, default=40)
    args = ap.parse_args()

    if not args.gt_cache.is_file():
        print(f"ERROR: GT cache missing: {args.gt_cache}", file=sys.stderr)
        return 2

    # Index every policy's clips (vid -> mp4 path).
    idx: Dict[str, Dict[str, Path]] = {}
    idx[args.notta_run] = index_method_videos(args.preview_series, args.notta_run)
    print(f"[index] {args.notta_run}: {len(idx[args.notta_run])} videos", file=sys.stderr)
    for arm in args.arms:
        idx[arm] = index_method_videos(args.placement_series, arm)
        print(f"[index] {arm}: {len(idx[arm])} videos", file=sys.stderr)

    # Common set across ALL policies -> matched-N.
    common = set(idx[args.notta_run])
    for arm in args.arms:
        common &= set(idx[arm])
    common_ids: List[str] = sorted(common)
    print(f"[common] matched-N set: N={len(common_ids)}")
    if len(common_ids) < args.min_linked:
        print(f"ERROR: common set N={len(common_ids)} < min_linked={args.min_linked}",
              file=sys.stderr)
        return 1

    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for policy in [args.notta_run, *args.arms]:
        src_by_vid = {v: idx[policy][v] for v in common_ids}
        # bijectivity guard: no two ids share a resolved clip
        resolved = [src_by_vid[v].resolve() for v in common_ids]
        if len(set(resolved)) != len(resolved):
            print(f"ERROR: {policy}: non-bijective clip resolution "
                  f"({len(resolved)} ids -> {len(set(resolved))} unique)", file=sys.stderr)
            return 1
        linked, missing = symlink_policy_dir(
            policy=policy, video_ids=common_ids, src_by_vid=src_by_vid,
            output_root=args.output_root, clean=True,
        )
        out_dir = args.output_root / policy
        out_json = out_dir / "fvd.json"
        rc = run_eval_fvd(
            gen_dir=out_dir / "videos", out_json=out_json, gt_cache=args.gt_cache,
            device=args.device, min_videos=args.min_linked, force=True,
        )
        fvd = None
        if rc == 0 and out_json.is_file():
            blob = json.loads(out_json.read_text(encoding="utf-8"))
            fvd = blob.get("fvd")
            nvp = blob.get("num_valid_pairs")
        else:
            nvp = None
            fvd = f"ERROR(rc={rc})"
        rows.append({"policy": policy, "linked": linked, "fvd": fvd,
                     "num_valid_pairs": nvp})
        print(f"  -> {policy}: FVD={fvd} N={nvp}")

    # summary (deltas vs NO-TTA)
    fvd_by = {r["policy"]: r["fvd"] for r in rows}
    base = fvd_by.get(args.notta_run)
    lines = [
        f"# EXP2 placement arms — matched-N FVD (N={len(common_ids)})",
        "",
        f"Common video set across {args.notta_run} + {', '.join(args.arms)}; same "
        "`eval_fvd.py` + preview GT cache + 14/14 window + --force. Δ vs NO-TTA "
        "(negative = better).",
        "",
        "| Policy | N | FVD | Δ vs NO-TTA |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        f = r["fvd"]
        d = ""
        if isinstance(f, (int, float)) and isinstance(base, (int, float)):
            d = f"{f - base:+.3f}"
        fs = f"{f:.3f}" if isinstance(f, (int, float)) else str(f)
        lines.append(f"| {r['policy']} | {r['num_valid_pairs']} | {fs} | {d} |")
    lines.append("")
    (args.output_root / "placement_arms_fvd_summary.md").write_text(
        "\n".join(lines), encoding="utf-8")
    (args.output_root / "placement_arms_fvd_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nWrote {args.output_root/'placement_arms_fvd_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
