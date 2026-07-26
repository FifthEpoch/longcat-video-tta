#!/usr/bin/env python3
"""Fold backfilled VBench dimensions into a method dir's
``merged_summary.json["vbench"]`` dict.

Reads every ``chunk_*/vbench_results/vbench_<dim>_eval_results.json`` file
and computes a weighted mean of the per-video scalar across chunks. Updates
in-place: only adds dimensions that aren't already present (or replaces all
if --force).

VBench's ``vbench_<dim>_eval_results.json`` schema in 0.1.5 is one of:

  Form A:  {"<dim>": [overall_score, [{"video_path": ..., "video_results": ...}, ...]]}
  Form B:  {"<dim>": [overall_score, {"video_path1": score1, ...}]}

We try both. We trust the `overall_score` for averaging across chunks,
weighted by per-chunk ``num_videos``.

Run after VBench backfill jobs complete:

    python3 scripts/update_merged_with_vbench.py \
        --method-dir sweep_experiment/results/panda_1000v_standard/NOTTA
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ALL_VBENCH_DIMS = [
    "subject_consistency", "background_consistency", "aesthetic_quality",
    "motion_smoothness", "dynamic_degree", "imaging_quality",
    "temporal_flickering",
]


def _extract_overall_score(parsed: dict, dim: str) -> Optional[float]:
    """Return the headline scalar from a vbench_<dim>_eval_results.json."""
    if not isinstance(parsed, dict):
        return None

    body = parsed.get(dim)
    if body is None and len(parsed) == 1:
        body = next(iter(parsed.values()))
    if body is None:
        return None

    # body may be [score, ...] or just a list/dict
    if isinstance(body, list) and body:
        first = body[0]
        if isinstance(first, (int, float)):
            return float(first)
        # fall through to averaging per-video below
        per_video = first if isinstance(first, (list, dict)) else None
    else:
        per_video = body

    # If we have a list of per-video dicts, average them
    if isinstance(per_video, list):
        scores = []
        for item in per_video:
            if isinstance(item, dict):
                v = item.get("video_results", item.get("video_score",
                    item.get("score")))
                if isinstance(v, (int, float)):
                    scores.append(float(v))
        if scores:
            return float(sum(scores) / len(scores))

    if isinstance(per_video, dict):
        scores = [float(v) for v in per_video.values()
                  if isinstance(v, (int, float))]
        if scores:
            return float(sum(scores) / len(scores))

    return None


def _count_videos_in_chunk(chunk_dir: Path, videos_subdir: str = "videos") -> int:
    vd = chunk_dir / videos_subdir
    if not vd.is_dir():
        return 0
    return len(list(vd.glob("*.mp4")))


def collect_dim_scores(method_dir: Path,
                       dim: str,
                       vbench_subdir: str = "vbench_results",
                       videos_subdir: str = "videos",
                       ) -> Tuple[Optional[float], int, List[Path]]:
    """Returns (weighted_mean, total_videos, list_of_files_used)."""
    files = []
    for cd in sorted(method_dir.glob("chunk_*")):
        f = cd / vbench_subdir / f"vbench_{dim}_eval_results.json"
        if f.exists():
            files.append((cd, f))
    if not files:
        # older single-job layout
        f = method_dir / vbench_subdir / f"vbench_{dim}_eval_results.json"
        if f.exists():
            files.append((method_dir, f))

    if not files:
        return None, 0, []

    total_w = 0
    weighted = 0.0
    used = []
    for chunk_dir, f in files:
        try:
            parsed = json.loads(f.read_text())
        except Exception as exc:
            print(f"  [warn] failed to parse {f}: {exc}", file=sys.stderr)
            continue
        s = _extract_overall_score(parsed, dim)
        if s is None:
            print(f"  [warn] no usable score in {f}", file=sys.stderr)
            continue
        w = _count_videos_in_chunk(chunk_dir, videos_subdir) or 1
        weighted += s * w
        total_w += w
        used.append(f)

    if total_w == 0:
        return None, 0, used
    return weighted / total_w, total_w, used


def update_method_dir(method_dir: Path, force: bool = False,
                      vbench_subdir: str = "vbench_results",
                      videos_subdir: str = "videos",
                      deprecate_existing: bool = False) -> int:
    if not method_dir.exists():
        print(f"[error] {method_dir} does not exist", file=sys.stderr)
        return 2

    summary_path = method_dir / "merged_summary.json"
    if not summary_path.exists():
        # fall back to summary.json (older single-job layout)
        alt = method_dir / "summary.json"
        if alt.exists():
            summary_path = alt
        else:
            print(f"[error] no merged_summary.json or summary.json in {method_dir}",
                  file=sys.stderr)
            return 2

    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:
        print(f"[error] failed to parse {summary_path}: {exc}", file=sys.stderr)
        return 2

    vbench = summary.get("vbench") or {}
    if not isinstance(vbench, dict):
        print(f"[warn] existing 'vbench' field is not a dict; resetting",
              file=sys.stderr)
        vbench = {}

    # Preserve the old (full-clip / contaminated) VBench for audit before we
    # overwrite it with generated-only scores. Note: the merged_summary may not
    # have held any VBench previously (e.g. the preview series kept per-video
    # scores only under vbench_results/), in which case there is nothing to
    # stash — that is expected, not an error.
    if deprecate_existing:
        if vbench and "vbench_fullclip_deprecated" not in summary:
            summary["vbench_fullclip_deprecated"] = dict(vbench)
            print("  [audit] stashed old full-clip vbench -> 'vbench_fullclip_deprecated'")
            vbench = {}  # rebuild from gen-only results
        else:
            print("  [audit] no pre-existing merged vbench to stash (expected for preview)")
        summary["vbench_window_note"] = (
            "vbench = GENERATED-ONLY (cond frames trimmed via make_geneval_clips.py); "
            "vbench_fullclip_deprecated (if present) = old cond+gen full-clip scores "
            "(do not cite)."
        )

    print(f"Method dir    : {method_dir}")
    print(f"Summary       : {summary_path.name}")
    print(f"VBench subdir : {vbench_subdir}")
    print(f"Existing vbench dims : {sorted(vbench.keys()) or '-'}")
    print()

    n_added = 0
    n_skipped = 0
    n_missing = 0
    chunk_counts: Dict[str, int] = {}

    for dim in ALL_VBENCH_DIMS:
        if dim in vbench and not force:
            print(f"  {dim:<25} SKIP (existing = {vbench[dim]:.4f})")
            n_skipped += 1
            continue
        score, n_vid, files = collect_dim_scores(
            method_dir, dim, vbench_subdir=vbench_subdir, videos_subdir=videos_subdir)
        if score is None:
            print(f"  {dim:<25} MISSING (no per-chunk results found)")
            n_missing += 1
            continue
        action = "REPLACE" if dim in vbench else "ADD    "
        print(f"  {dim:<25} {action} = {score:.4f}  "
              f"(n_chunks={len(files)}, n_videos={n_vid})")
        vbench[dim] = float(score)
        chunk_counts[dim] = len(files)
        n_added += 1

    summary["vbench"] = vbench
    if "vbench_num_chunks" in summary or chunk_counts:
        existing_nc = summary.get("vbench_num_chunks") or {}
        if not isinstance(existing_nc, dict):
            existing_nc = {}
        existing_nc.update(chunk_counts)
        summary["vbench_num_chunks"] = existing_nc

    backup_path = summary_path.with_suffix(summary_path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(json.dumps(json.loads(summary_path.read_text()),
                                         indent=2))
        print(f"\n  Backup written: {backup_path.name}")

    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Updated: {summary_path}")
    print()
    print(f"  added={n_added}  skipped={n_skipped}  missing={n_missing}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-dir", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="Replace existing vbench entries instead of skipping.")
    ap.add_argument("--vbench-subdir", default="vbench_results",
                    help="Per-chunk results dir to read (default 'vbench_results'; "
                         "use 'vbench_results_geneval' for gen-only scores).")
    ap.add_argument("--videos-subdir", default="videos",
                    help="Per-chunk clip dir used for per-chunk video-count weights "
                         "(default 'videos'; use 'videos_geneval').")
    ap.add_argument("--deprecate-existing", action="store_true",
                    help="Stash the current summary['vbench'] under "
                         "'vbench_fullclip_deprecated' and rebuild 'vbench' from "
                         "--vbench-subdir (use when switching to gen-only).")
    args = ap.parse_args()
    return update_method_dir(args.method_dir, args.force,
                             vbench_subdir=args.vbench_subdir,
                             videos_subdir=args.videos_subdir,
                             deprecate_existing=args.deprecate_existing)


if __name__ == "__main__":
    sys.exit(main())
