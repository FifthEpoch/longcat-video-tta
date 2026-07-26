#!/usr/bin/env python3
"""Build symlinked video dirs for budget-grid oracle FVD (H9).

Per eval video, pick the grid config with highest PSNR (same rule as
``scripts/analyze_adasteer_budget_oracle.py``), then symlink that config's
generated mp4 into ``<output-root>/oracle_best_psnr/videos/``.

Requires saved mp4s (``NO_SAVE_VIDEOS=0`` during the budget sweep). Pilot
runs with metrics-only mode cannot populate this dir — re-run best configs
with video retention or use the 1000v incremental jobs.

Usage:
    python sweep_experiment/scripts/build_budget_oracle_policy_dirs.py \\
        --series-root sweep_experiment/results/panda_ood_budget_pilot \\
        --output-root sweep_experiment/reports/budget_oracle_fvd
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    discover_runs,
    load_run_psnr,
    oracle_winner,
)
from scripts.caption_utils import canonical_video_id
from sweep_experiment.scripts.build_oracle_policy_dirs import (  # noqa: E402
    _mp4_readable,
)


_METRIC_RE = {
    "psnr": re.compile(r"PSNR-(-?\d+\.?\d*)"),
    "ssim": re.compile(r"SSIM-(-?\d+\.?\d*)"),
    "lpips": re.compile(r"LPIPS-(-?\d+\.?\d*)"),
}


def _parse_filename_metrics(name: str) -> Dict[str, float]:
    """Extract per-video PSNR/SSIM/LPIPS embedded in a renamed config mp4."""
    out: Dict[str, float] = {}
    for key, rx in _METRIC_RE.items():
        m = rx.search(name)
        if m:
            try:
                out[key] = float(m.group(1))
            except ValueError:
                pass
    return out


def _record_metrics(rec: dict) -> Dict[str, Optional[float]]:
    """Pull per-video psnr/ssim/lpips from a summary record (flat or nested)."""
    src = rec.get("metrics") if isinstance(rec.get("metrics"), dict) else rec
    out: Dict[str, Optional[float]] = {}
    for key in ("psnr", "ssim", "lpips"):
        val = src.get(key)
        try:
            out[key] = float(val) if val is not None else None
        except (TypeError, ValueError):
            out[key] = None
    return out


def _index_grid_videos(
    series_root: Path, run_id: str, *, tol: float = 5e-4
) -> Dict[str, Path]:
    """Map canonical video_id -> generated mp4 for one grid config.

    Grid configs rename outputs to
    ``{idx}_{caption}_..._PSNR-{p}_SSIM-{s}_LPIPS-{l}_..._adasteer.mp4``.  The
    leading ``{idx}`` is NOT a usable key (prefixes repeat and are sparse), and
    the summary's ``output_path`` is the stale pre-rename path.  The only
    reliable join is the per-video metric fingerprint embedded in the filename
    matched against the record's ``psnr/ssim/lpips``.  A bijectivity guard
    aborts the build if two ids resolve to the same file (which would corrupt
    FVD by silently duplicating videos, as happened before this fix).
    """
    run_dir = series_root / run_id
    if not run_dir.is_dir():
        return {}

    out: Dict[str, Path] = {}
    used_files: set = set()
    ambiguous = 0
    unresolved = 0
    chunk_dirs = sorted(run_dir.glob("chunk_*/"))
    if not chunk_dirs:
        chunk_dirs = [run_dir]

    for chunk_dir in chunk_dirs:
        videos_dir = chunk_dir / "videos"
        summary_path = chunk_dir / "summary.json"
        if not summary_path.exists() or not videos_dir.is_dir():
            continue
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)

        file_metrics = [
            (p, _parse_filename_metrics(p.name))
            for p in sorted(videos_dir.glob("*.mp4"))
        ]

        for rec in summary.get("per_video_results", summary.get("results", [])):
            if not rec.get("success", False):
                continue
            vname = rec.get("video_name", "")
            vid = canonical_video_id(vname)
            if not vid:
                continue
            rm = _record_metrics(rec)
            if rm.get("psnr") is None:
                unresolved += 1
                continue

            cands: List[Path] = []
            for p, fm in file_metrics:
                if "psnr" not in fm or abs(fm["psnr"] - rm["psnr"]) > tol:
                    continue
                if (rm.get("ssim") is not None and "ssim" in fm
                        and abs(fm["ssim"] - rm["ssim"]) > tol):
                    continue
                if (rm.get("lpips") is not None and "lpips" in fm
                        and abs(fm["lpips"] - rm["lpips"]) > tol):
                    continue
                cands.append(p)

            if len(cands) != 1:
                if len(cands) > 1:
                    ambiguous += 1
                else:
                    unresolved += 1
                continue
            mp4 = cands[0]
            if not _mp4_readable(mp4):
                unresolved += 1
                continue
            rp = mp4.resolve()
            if rp in used_files:
                # Never map two ids to one file — drop rather than duplicate.
                ambiguous += 1
                continue
            used_files.add(rp)
            out[vid] = rp

    n_unique = len(set(out.values()))
    if n_unique != len(out):
        raise RuntimeError(
            f"_index_grid_videos({run_id}): non-bijective mapping "
            f"({len(out)} ids -> {n_unique} unique files). Refusing to build a "
            "duplicated policy dir (would corrupt FVD)."
        )
    if ambiguous or unresolved:
        print(
            f"  [index] {run_id}: mapped {len(out)} ids "
            f"(ambiguous={ambiguous}, unresolved={unresolved})",
            file=sys.stderr,
        )
    return out


def build_oracle_dir(
    *,
    series_root: Path,
    output_root: Path,
    grid_runs: Optional[List[str]] = None,
    policy_name: str = "oracle_best_psnr",
) -> Tuple[int, int]:
    runs = discover_runs(series_root)
    if grid_runs is None:
        grid_runs = sorted(r for r in runs if r.startswith("S"))
    else:
        grid_runs = [r for r in grid_runs if r in runs]

    psnr_by_run: Dict[str, Dict[str, float]] = {
        rid: load_run_psnr(runs[rid]) for rid in grid_runs
    }
    all_vids = sorted(set().union(*[set(d.keys()) for d in psnr_by_run.values()]))

    video_index: Dict[str, Dict[str, Path]] = {}
    for rid in grid_runs:
        video_index[rid] = _index_grid_videos(series_root, rid)

    out_dir = output_root / policy_name
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    skipped = 0
    manifest: List[dict] = []

    for vid in all_vids:
        row = {rid: psnr_by_run[rid].get(vid) for rid in grid_runs}
        winner = oracle_winner(row, grid_runs)
        if winner is None:
            skipped += 1
            continue
        src = video_index.get(winner, {}).get(vid)
        if src is None:
            skipped += 1
            continue
        dst = videos_dir / f"{vid}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        linked += 1
        manifest.append({
            "video_id": vid,
            "winner_run": winner,
            "psnr": row.get(winner),
            "source_mp4": str(src),
        })

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": policy_name,
                "series_root": str(series_root),
                "grid_runs": grid_runs,
                "linked_videos": linked,
                "skipped_videos": skipped,
                "entries": manifest,
            },
            f,
            indent=2,
        )
    print(f"Built {policy_name}: linked={linked} skipped={skipped} -> {videos_dir}")
    return linked, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description="Build budget-grid oracle policy video dirs")
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
        "--grid-runs", nargs="*", default=None,
        help="Subset of run IDs (default: all S* dirs with PSNR under series-root)",
    )
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if not args.series_root.is_dir():
        print(f"ERROR: series root not found: {args.series_root}", file=sys.stderr)
        return 2

    if args.clean and args.output_root.exists():
        import shutil
        shutil.rmtree(args.output_root)

    linked, skipped = build_oracle_dir(
        series_root=args.series_root,
        output_root=args.output_root,
        grid_runs=args.grid_runs,
    )
    if linked == 0:
        print(
            "ERROR: no videos linked — budget sweep likely used NO_SAVE_VIDEOS=1. "
            "Re-run with saved mp4s or use run_budget_oracle_fvd --skip-build after "
            "populating videos.",
            file=sys.stderr,
        )
        return 1
    if skipped:
        print(f"WARN: skipped {skipped} videos (missing PSNR winner mp4)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
