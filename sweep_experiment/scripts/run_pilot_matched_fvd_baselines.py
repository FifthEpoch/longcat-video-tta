#!/usr/bin/env python3
"""Apples-to-apples FVD on the budget pilot video set (same N, same eval_fvd protocol).

Builds symlink dirs for policies on the **same** video IDs as the budget PSNR
oracle manifest (typically 200 pilot clips), then runs ``eval_fvd.py`` with the
GT I3D cache and ``--force`` (pilot N < 256).

Policies (default):
  * ``always_notta``       — NOTTA mp4s from ``panda_1000v_standard``
  * ``fixed_S10_LR5e3``    — fixed deployable AdaSteer config on pilot grid
  * ``oracle_best_psnr``   — skip build if manifest exists; re-eval optional

Usage:
    python sweep_experiment/scripts/run_pilot_matched_fvd_baselines.py

    python sweep_experiment/scripts/run_pilot_matched_fvd_baselines.py \\
        --skip-eval --policies always_notta fixed_S10_LR5e3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    FIXED_ADA_RUN_ID,
    NOTTA_RUN_ID,
    _infer_baseline_series_root,
)
from sweep_experiment.scripts.build_budget_oracle_policy_dirs import (  # noqa: E402
    _index_grid_videos,
)
from sweep_experiment.scripts.build_oracle_policy_dirs import (  # noqa: E402
    index_method_videos,
)

DEFAULT_NUM_COND_FRAMES = 14
DEFAULT_NUM_GEN_FRAMES = 14
ORACLE_POLICY = "oracle_best_psnr"


def load_pilot_video_ids(manifest_path: Path) -> List[str]:
    blob = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = blob.get("entries") or []
    ids = sorted({str(e["video_id"]) for e in entries if e.get("video_id")})
    if not ids:
        raise ValueError(f"no video_id entries in {manifest_path}")
    return ids


def symlink_policy_dir(
    *,
    policy: str,
    video_ids: Sequence[str],
    src_by_vid: Dict[str, Path],
    output_root: Path,
    clean: bool,
) -> Tuple[int, int]:
    out_dir = output_root / policy
    videos_dir = out_dir / "videos"
    if clean and out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    missing = 0
    manifest: List[dict] = []
    for vid in video_ids:
        src = src_by_vid.get(vid)
        if src is None:
            missing += 1
            continue
        dst = videos_dir / f"{vid}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        linked += 1
        manifest.append({"video_id": vid, "source_mp4": str(src)})

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": policy,
                "linked_videos": linked,
                "missing_videos": missing,
                "entries": manifest,
            },
            f,
            indent=2,
        )
    print(f"Built {policy}: linked={linked} missing={missing} -> {videos_dir}")
    return linked, missing


def run_eval_fvd(
    *,
    gen_dir: Path,
    out_json: Path,
    gt_cache: Path,
    device: str,
    min_videos: int,
    force: bool,
) -> int:
    n_mp4 = len(list(gen_dir.glob("*.mp4")))
    if n_mp4 < min_videos:
        print(f"ERROR: {gen_dir} has only {n_mp4} mp4s", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        str(_REPO_ROOT / "sweep_experiment/scripts/eval_fvd.py"),
        "--gen-dir",
        str(gen_dir),
        "--gt-cache",
        str(gt_cache),
        "--num-cond-frames",
        str(DEFAULT_NUM_COND_FRAMES),
        "--num-gen-frames",
        str(DEFAULT_NUM_GEN_FRAMES),
        "--min-videos",
        str(min(min_videos, n_mp4)),
        "--output",
        str(out_json),
        "--device",
        device,
    ]
    if force:
        cmd.append("--force")
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(_REPO_ROOT))


def write_summary(output_root: Path, rows: List[dict]) -> None:
    lines = [
        "# Pilot matched FVD baselines (same video IDs, same protocol)",
        "",
        "All policies use the budget PSNR-oracle manifest video set, "
        "`eval_fvd.py` + GT cache, `--force` (N≈200 < 256).",
        "",
        "| Policy | N linked | FVD | num_valid_pairs | sample warning |",
        "|---|---:|---:|---:|---|",
    ]
    for r in rows:
        warn = r.get("sample_size_warning") or "—"
        if warn and len(str(warn)) > 40:
            warn = "yes (N<256)"
        lines.append(
            f"| {r['policy']} | {r.get('linked', '—')} | "
            f"{r.get('fvd', '—')} | {r.get('num_valid_pairs', '—')} | {warn} |"
        )
    lines.append("")
    path = output_root / "pilot_matched_fvd_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Matched pilot FVD baselines")
    ap.add_argument(
        "--series-root",
        type=Path,
        default=Path("sweep_experiment/results/panda_ood_budget_pilot"),
    )
    ap.add_argument(
        "--baseline-series-root",
        type=Path,
        default=None,
        help="NOTTA mp4 source (default: panda_1000v_standard)",
    )
    ap.add_argument(
        "--oracle-manifest",
        type=Path,
        default=Path(
            "sweep_experiment/reports/budget_oracle_fvd/oracle_best_psnr/manifest.json"
        ),
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        default=Path("sweep_experiment/reports/pilot_matched_fvd"),
    )
    ap.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("gt_caches/panda_1000_longcat.npz"),
    )
    ap.add_argument(
        "--fixed-run",
        type=str,
        default=FIXED_ADA_RUN_ID,
    )
    ap.add_argument(
        "--policies",
        nargs="*",
        default=["always_notta", "fixed_S10_LR5e3", "oracle_best_psnr"],
    )
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-linked", type=int, default=50)
    ap.add_argument("--force", action="store_true", default=True)
    args = ap.parse_args()

    if not args.oracle_manifest.is_file():
        print(f"ERROR: oracle manifest missing: {args.oracle_manifest}", file=sys.stderr)
        print("Run budget oracle FVD build first.", file=sys.stderr)
        return 2

    video_ids = load_pilot_video_ids(args.oracle_manifest)
    print(f"Pilot video set: N={len(video_ids)}")

    baseline_root = args.baseline_series_root or _infer_baseline_series_root(
        args.series_root
    )
    if not args.gt_cache.is_file():
        print(f"ERROR: GT cache missing: {args.gt_cache}", file=sys.stderr)
        return 2

    summary_rows: List[dict] = []

    for policy in args.policies:
        if policy == ORACLE_POLICY:
            gen_dir = (
                args.output_root.parent / "budget_oracle_fvd" / ORACLE_POLICY / "videos"
            )
            if not gen_dir.is_dir():
                gen_dir = args.output_root / ORACLE_POLICY / "videos"
            out_json = gen_dir.parent / "fvd.json"
            linked = len(list(gen_dir.glob("*.mp4"))) if gen_dir.is_dir() else 0
            if linked == 0 and args.oracle_manifest.parent.joinpath("videos").is_dir():
                gen_dir = args.oracle_manifest.parent / "videos"
                out_json = args.oracle_manifest.parent / "fvd.json"
                linked = len(list(gen_dir.glob("*.mp4")))
            print(f"[oracle] using existing dir {gen_dir} (N={linked})")
        elif policy == "always_notta":
            if not args.skip_build:
                notta_index = index_method_videos(baseline_root, NOTTA_RUN_ID)
                linked, missing = symlink_policy_dir(
                    policy=policy,
                    video_ids=video_ids,
                    src_by_vid=notta_index,
                    output_root=args.output_root,
                    clean=args.clean,
                )
                if linked < args.min_linked:
                    print(
                        f"ERROR: {policy} linked={linked} < {args.min_linked}",
                        file=sys.stderr,
                    )
                    return 1
            gen_dir = args.output_root / policy / "videos"
            out_json = args.output_root / policy / "fvd.json"
            linked = len(list(gen_dir.glob("*.mp4")))
        elif policy.startswith("fixed_") or policy == args.fixed_run:
            run_id = args.fixed_run if policy.startswith("fixed_") else policy
            if not args.skip_build:
                grid_index = _index_grid_videos(args.series_root, run_id)
                pol_name = f"fixed_{run_id}"
                linked, missing = symlink_policy_dir(
                    policy=pol_name,
                    video_ids=video_ids,
                    src_by_vid=grid_index,
                    output_root=args.output_root,
                    clean=args.clean,
                )
                if linked < args.min_linked:
                    print(
                        f"ERROR: {pol_name} linked={linked} < {args.min_linked}",
                        file=sys.stderr,
                    )
                    return 1
                policy = pol_name
            gen_dir = args.output_root / policy / "videos"
            out_json = args.output_root / policy / "fvd.json"
            linked = len(list(gen_dir.glob("*.mp4")))
        else:
            print(f"WARN: unknown policy {policy}, skip", file=sys.stderr)
            continue

        row = {"policy": policy, "linked": linked}
        if args.skip_eval:
            summary_rows.append(row)
            continue

        if out_json.is_file() and not args.clean:
            blob = json.loads(out_json.read_text(encoding="utf-8"))
            print(f"  skip eval — exists {out_json} FVD={blob.get('fvd')}")
            row.update(
                {
                    "fvd": blob.get("fvd"),
                    "num_valid_pairs": blob.get("num_valid_pairs"),
                    "sample_size_warning": blob.get("sample_size_warning"),
                }
            )
            summary_rows.append(row)
            continue

        rc = run_eval_fvd(
            gen_dir=gen_dir,
            out_json=out_json,
            gt_cache=args.gt_cache,
            device=args.device,
            min_videos=args.min_linked,
            force=args.force,
        )
        if rc != 0:
            return rc
        if out_json.is_file():
            blob = json.loads(out_json.read_text(encoding="utf-8"))
            row.update(
                {
                    "fvd": blob.get("fvd"),
                    "num_valid_pairs": blob.get("num_valid_pairs"),
                    "sample_size_warning": blob.get("sample_size_warning"),
                }
            )
        summary_rows.append(row)

    write_summary(args.output_root, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
