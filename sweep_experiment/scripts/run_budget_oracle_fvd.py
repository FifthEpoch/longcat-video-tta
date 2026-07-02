#!/usr/bin/env python3
"""Compute FVD/FID for budget-grid oracle (per-video best PSNR config).

Wraps ``build_budget_oracle_policy_dirs.py`` + ``eval_fvd.py`` using the
precomputed GT I3D cache (same protocol as ``run_phase1_oracle_fvd.py``).

Usage:
    # After budget sweep with saved mp4s (NO_SAVE_VIDEOS=0):
    python sweep_experiment/scripts/run_budget_oracle_fvd.py \\
        --series-root sweep_experiment/results/panda_ood_budget_pilot \\
        --gt-cache gt_caches/panda_1000_longcat.npz

    # Full 1000v best-config series (S2, S10_LR1e2, S20_LR1e2):
    python sweep_experiment/scripts/run_budget_oracle_fvd.py \\
        --series-root sweep_experiment/results/panda_ood_budget_1000v \\
        --gt-cache gt_caches/panda_1000_longcat.npz

    # Re-eval only (policy dir already built):
    python sweep_experiment/scripts/run_budget_oracle_fvd.py --skip-build
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_NUM_COND_FRAMES = 14
DEFAULT_NUM_GEN_FRAMES = 14
MIN_LINKED_DEFAULT = 50


def main() -> int:
    ap = argparse.ArgumentParser(description="Budget-grid oracle FVD eval")
    ap.add_argument(
        "--series-root",
        type=Path,
        default=Path("sweep_experiment/results/panda_ood_budget_pilot"),
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("sweep_experiment/reports/budget_oracle_fvd"),
    )
    ap.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("gt_caches/panda_1000_longcat.npz"),
    )
    ap.add_argument("--grid-runs", nargs="*", default=None)
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-linked", type=int, default=MIN_LINKED_DEFAULT)
    args = ap.parse_args()

    policy = "oracle_best_psnr"
    gen_dir = args.output_root / policy / "videos"
    out_json = args.output_root / policy / "fvd.json"

    if not args.skip_build:
        build_cmd = [
            sys.executable,
            str(_REPO_ROOT / "sweep_experiment/scripts/build_budget_oracle_policy_dirs.py"),
            "--series-root", str(args.series_root),
            "--output-root", str(args.output_root),
            "--clean",
        ]
        if args.grid_runs:
            build_cmd.extend(["--grid-runs", *args.grid_runs])
        print("Building budget oracle policy dir...")
        rc = subprocess.call(build_cmd, cwd=str(_REPO_ROOT))
        if rc != 0:
            return rc

    manifest_path = args.output_root / policy / "manifest.json"
    linked = 0
    if manifest_path.exists():
        blob = json.loads(manifest_path.read_text(encoding="utf-8"))
        linked = int(blob.get("linked_videos") or 0)

    if linked < args.min_linked:
        print(
            f"ERROR: only {linked} symlinks (need >= {args.min_linked}). "
            "Budget sweep needs saved mp4s (NO_SAVE_VIDEOS=0).",
            file=sys.stderr,
        )
        return 1

    if not args.gt_cache.exists():
        print(f"ERROR: GT cache missing: {args.gt_cache}", file=sys.stderr)
        return 2

    eval_script = _REPO_ROOT / "sweep_experiment/scripts/eval_fvd.py"
    cmd = [
        sys.executable, str(eval_script),
        "--gen-dir", str(gen_dir),
        "--gt-cache", str(args.gt_cache),
        "--num-cond-frames", str(DEFAULT_NUM_COND_FRAMES),
        "--num-gen-frames", str(DEFAULT_NUM_GEN_FRAMES),
        "--min-videos", str(min(args.min_linked, linked)),
        "--output", str(out_json),
        "--device", args.device,
    ]
    if args.force:
        cmd.append("--force")
    elif linked < 256:
        # Pilot has ~200 videos; eval_fvd defaults to min 256 without --force.
        cmd.append("--force")

    print("Running eval_fvd on budget oracle dir...")
    print(" ", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(_REPO_ROOT))
    if rc != 0:
        return rc

    summary_path = args.output_root / "fvd_summary.json"
    if out_json.exists():
        blob = json.loads(out_json.read_text(encoding="utf-8"))
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({policy: blob.get("fvd"), "fid": blob.get("fid")}, f, indent=2)
        print(f"\nOracle FVD={blob.get('fvd')} FID={blob.get('fid')}")
        print(f"Wrote {out_json}")
        print(f"Re-run analyzer with: --oracle-fvd-json {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
