#!/usr/bin/env python3
"""Compose single-config and random-mixture FVD policy dirs to DECOMPOSE the
oracle FVD headroom.

The PSNR-oracle composition lowers pooled FVD vs NO-TTA with a CI that excludes
0 (ΔFVD ≈ -10, 2026-07-31). But per-video PSNR selection is max-over-noise /
unroutable and all trained routers were FVD-null, so we do NOT yet know the
MECHANISM of that -10. Three candidates:

  (A) one config is simply more in-distribution   -> deployable as a FIXED config
  (B) mixture diversity across configs            -> deployable as RANDOM assignment
  (C) genuine per-video targeting                 -> needs a GT-free probe

This script builds the policy dirs needed to tell them apart, then
sweep_experiment/scripts/fvd_bootstrap_ci.py scores them under the SAME
matched-GT-cache paired bootstrap as always_notta / fixed / oracle:

  * always_<RUN> for every grid config  -> tests (A): does any single config
    hit the oracle's FVD on its own?
  * random_mixture (uniformly random config per video, seeded) -> tests (B):
    does mere config diversity reproduce the oracle's FVD?

All dirs are restricted to the NOTTA common set (matched-N) and resolve each
video's clip via the metric-fingerprint index (same bijective resolution as the
duplication-bug fix), so FVD is directly comparable to the existing rows.

Usage:
    python3 scripts/build_fvd_decomposition_dirs.py \
      --series-root sweep_experiment/results/panda_ood_budget_1000v_preview \
      --feature-date sweep_experiment/reports/per_video_analysis/2026-07-12 \
      --output-root sweep_experiment/reports/budget_oracle_fvd_1000v_preview/decomp \
      --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.analyze_adasteer_budget_oracle import NOTTA_RUN_ID  # noqa: E402
from scripts.budget_routing_common import load_pilot_bundle  # noqa: E402
from sweep_experiment.scripts.build_budget_oracle_policy_dirs import (  # noqa: E402
    _index_grid_videos,
)
from sweep_experiment.scripts.build_oracle_policy_dirs import (  # noqa: E402
    index_method_videos,
)
from sweep_experiment.scripts.run_pilot_matched_fvd_baselines import (  # noqa: E402
    symlink_policy_dir,
)


def _build_policy(
    *,
    pol_name: str,
    src_by_vid: Dict[str, Path],
    output_root: Path,
) -> dict:
    ordered_ids = sorted(src_by_vid.keys())
    resolved = [src_by_vid[v].resolve() for v in ordered_ids]
    n_unique = len(set(resolved))
    if n_unique != len(resolved):
        raise RuntimeError(
            f"{pol_name}: non-bijective policy dir "
            f"({len(resolved)} ids -> {n_unique} unique clips)"
        )
    linked, missing = symlink_policy_dir(
        policy=pol_name,
        video_ids=ordered_ids,
        src_by_vid=src_by_vid,
        output_root=output_root,
        clean=True,
    )
    return {"policy": pol_name, "linked": linked, "missing": missing}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build single-config + random-mixture FVD dirs")
    ap.add_argument("--series-root", type=Path, required=True)
    ap.add_argument(
        "--feature-date", type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-12",
    )
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-mixture", type=int, default=1,
                    help="How many independent random-mixture policies to build "
                         "(different seeds) to gauge mixture-FVD variance.")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    # Video pool + grid run list (same bundle the routers/oracle used).
    bundle = load_pilot_bundle(args.series_root, args.feature_date,
                               require_vbench=False)
    grid_runs: List[str] = bundle["grid_runs"]

    notta_index = index_method_videos(args.series_root, NOTTA_RUN_ID)
    print(f"[index] NOTTA common set: {len(notta_index)} videos", file=sys.stderr)
    notta_ids = set(notta_index.keys())

    grid_index_cache: Dict[str, Dict[str, Path]] = {
        rid: _index_grid_videos(args.series_root, rid) for rid in grid_runs
    }
    for rid in grid_runs:
        print(f"[index] {rid}: {len(grid_index_cache[rid])} videos", file=sys.stderr)

    rows: List[dict] = []

    # (A) one policy dir per single config, restricted to the NOTTA common set.
    for rid in grid_runs:
        src_by_vid = {
            vid: p for vid, p in grid_index_cache[rid].items() if vid in notta_ids
        }
        row = _build_policy(
            pol_name=f"always_{rid}",
            src_by_vid=src_by_vid,
            output_root=args.output_root,
        )
        rows.append(row)
        print(f"  always_{rid}: linked {row['linked']}", file=sys.stderr)

    # (B) random-mixture: uniformly random config per video (seeded).
    common_ids = sorted(
        v for v in notta_ids
        if any(v in grid_index_cache[rid] for rid in grid_runs)
    )
    for m in range(args.n_mixture):
        rng = np.random.default_rng(args.seed + m)
        src_by_vid: Dict[str, Path] = {}
        picks: List[dict] = []
        for vid in common_ids:
            avail = [rid for rid in grid_runs if vid in grid_index_cache[rid]]
            if not avail:
                continue
            rid = avail[int(rng.integers(0, len(avail)))]
            src_by_vid[vid] = grid_index_cache[rid][vid]
            picks.append({"video_id": vid, "chosen_run": rid})
        pol_name = "random_mixture" if args.n_mixture == 1 else f"random_mixture_{m}"
        row = _build_policy(
            pol_name=pol_name, src_by_vid=src_by_vid, output_root=args.output_root,
        )
        (args.output_root / pol_name / "mixture_manifest.json").write_text(
            json.dumps({"policy": pol_name, "seed": args.seed + m, "picks": picks},
                       indent=2),
            encoding="utf-8",
        )
        rows.append(row)
        print(f"  {pol_name}: linked {row['linked']}", file=sys.stderr)

    (args.output_root / "decomp_build_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {args.output_root}/decomp_build_summary.json "
          f"({len(rows)} policy dirs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
