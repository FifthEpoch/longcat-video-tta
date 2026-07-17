#!/usr/bin/env python3
"""Inventory + document generated-video runs *before* deleting their frames.

Motivation
----------
The saved ``*.mp4`` files are the only asset from which frame-based metrics
(VBench++, FID, FVD, any future perceptual metric) can be (re)computed. Once
they are deleted, each run is frozen at whatever metrics already live in its
``merged_summary.json``. Before any cleanup we therefore capture, for every
run, a durable manifest so the run stays fully described in the paper record:

  * provenance      : cluster path, mp4 count, N videos, num chunks
  * metrics         : every population metric present in merged_summary.json
  * eval pool       : the sorted set of canonical eval video ids + a sha1
                      "pool fingerprint" so runs evaluated on the SAME test
                      pool share a fingerprint and are provably comparable
                      (the "apples-to-apples" guarantee the paper needs)

Outputs (under --out-dir, default sweep_experiment/reports/cleanup_manifests/<date>):
  manifest.json          one JSON entry per (series, run)
  MANIFEST.md            human table + shared-pool groupings
  pool_fingerprints.csv  run -> pool hash + N (comparability audit)
  pool_ids/<fp>.txt      the sorted canonical id list for each distinct pool

Standard cluster usage (the user runs this; no slurm needed):

    python3 scripts/build_run_manifest.py \\
        --results-root sweep_experiment/results \\
        --results-root delta_experiment/results \\
        --results-root comparison_methods/data \\
        --results-root t2v_experiment/results \\
        --results-root baseline_experiment/results

Only depends on the Python stdlib.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Canonical video-id extraction (mirrors analyze_per_video_tta_gain.py so the
# fingerprints join with the rest of the analysis tooling).
# ---------------------------------------------------------------------------
_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def _canonical_video_id(s: Optional[str]) -> str:
    if not s:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


# Population metric keys we care about (superset of the merged_summary schema).
POP_METRICS = ("psnr", "ssim", "lpips", "fvd", "fid", "vbench")
# VBench is frequently stored as a per-dimension DICT (or under an alternate
# key), not a scalar. Detect presence across all known spellings so we never
# false-flag a fully-scored run as "VBench-missing".
_VBENCH_KEYS = (
    "vbench", "vbench_total_score", "vbench_total", "vbench_score",
    "vbench_quality_score", "vbench_num_chunks",
)
_PROVENANCE_KEYS = (
    "num_videos", "num_successful", "num_chunks",
    "fvd_num_videos", "fid_num_frames_gen", "vbench_num_chunks",
    "avg_train_time", "avg_gen_time", "avg_total_time",
)


def _coerce_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def _records_from_blob(blob) -> List[dict]:
    if isinstance(blob, list):
        return [r for r in blob if isinstance(r, dict)]
    if not isinstance(blob, dict):
        return []
    for key in ("results", "per_video_results", "per_video"):
        v = blob.get(key)
        if isinstance(v, list):
            return [r for r in v if isinstance(r, dict)]
    return []


def _record_vid(r: dict) -> str:
    raw = (r.get("video_name") or r.get("video_id") or r.get("video")
           or r.get("video_path") or r.get("path"))
    return _canonical_video_id(raw if raw is not None else "")


def _load_json(path: Path) -> Optional[dict]:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] {path}: {e}", file=sys.stderr)
        return None


def _git_commit(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# Per-run scan
# ---------------------------------------------------------------------------
def _metric_presence(d: dict, m: str) -> Tuple[bool, Optional[float]]:
    """(present?, scalar-if-any). VBench may be a dict -> present but scalar=None."""
    if m == "vbench":
        for k in _VBENCH_KEYS:
            if d.get(k) is not None:
                return True, _coerce_float(d.get(k))
        return False, None
    v = d.get(m)
    return (v is not None), _coerce_float(v)


def _run_pop_metrics(
    run_dir: Path,
) -> Tuple[Dict[str, Optional[float]], Dict[str, bool], dict]:
    """Return (scalar metrics, presence flags, provenance) for a run.

    Unions across EVERY merged_summary.json under the run (some series split
    FVD/FID and PSNR/VBench across nested merged files), so presence matches
    the authoritative glob-based view rather than only the run-level file.
    """
    pop: Dict[str, Optional[float]] = {m: None for m in POP_METRICS}
    has: Dict[str, bool] = {m: False for m in POP_METRICS}
    prov: dict = {}
    for p in sorted(run_dir.rglob("merged_summary.json")):
        d = _load_json(p) or {}
        for m in POP_METRICS:
            present, val = _metric_presence(d, m)
            if present:
                has[m] = True
                if pop[m] is None and val is not None:
                    pop[m] = val
        for k in _PROVENANCE_KEYS:
            if d.get(k) is not None and k not in prov:
                prov[k] = d.get(k)
    return pop, has, prov


def _run_pool(run_dir: Path) -> Tuple[List[str], Dict[str, bool]]:
    """Return (sorted canonical eval ids, per-metric per-video availability)."""
    candidates = sorted(run_dir.glob("chunk_*/summary.json"))
    if not candidates:
        candidates = sorted(run_dir.glob("chunk_*/results.json"))
    if not candidates:
        for flat in ("merged_summary.json", "summary.json"):
            if (run_dir / flat).exists():
                candidates = [run_dir / flat]
                break

    ids: set = set()
    have_pv = {m: False for m in ("psnr", "ssim", "lpips", "vbench")}
    for cf in candidates:
        blob = _load_json(cf)
        if blob is None:
            continue
        for r in _records_from_blob(blob):
            vid = _record_vid(r)
            if not vid:
                continue
            ids.add(vid)
            for m in have_pv:
                if _coerce_float(r.get(m)) is not None:
                    have_pv[m] = True
    return sorted(ids), have_pv


def _count_mp4(run_dir: Path) -> int:
    return sum(1 for _ in run_dir.rglob("*.mp4"))


def _pool_fingerprint(ids: List[str]) -> str:
    if not ids:
        return "EMPTY"
    h = hashlib.sha1("\n".join(ids).encode("utf-8")).hexdigest()
    return h[:12]


def scan_run(series: str, run_dir: Path) -> dict:
    pop, has, prov = _run_pop_metrics(run_dir)
    ids, have_pv = _run_pool(run_dir)
    fp = _pool_fingerprint(ids)
    return {
        "series": series,
        "run": run_dir.name,
        "path": str(run_dir),
        "mp4_count": _count_mp4(run_dir),
        "n_eval_videos": len(ids),
        "pool_fingerprint": fp,
        "pop_metrics": pop,
        "has_metrics": has,
        "per_video_available": have_pv,
        "provenance": prov,
        "eval_ids": ids,
    }


def is_run_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    return (
        (p / "merged_summary.json").exists()
        or (p / "summary.json").exists()
        or any(p.glob("chunk_*/summary.json"))
        or any(p.glob("chunk_*/results.json"))
    )


def scan_results_root(root: Path) -> List[dict]:
    out: List[dict] = []
    if not root.is_dir():
        return out
    for series_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        series = f"{root.name}/{series_dir.name}"
        run_dirs = [p for p in sorted(series_dir.iterdir()) if is_run_dir(p)]
        if not run_dirs and is_run_dir(series_dir):
            # series dir is itself a single run (flat layout)
            run_dirs = [series_dir]
        for rd in run_dirs:
            out.append(scan_run(series, rd))
    return out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _fmt(x: Optional[float], nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def render_markdown(entries: List[dict], repo_root: Path) -> str:
    date = _dt.date.today().isoformat()
    commit = _git_commit(repo_root)
    lines = [
        f"# Run manifest — {date}",
        "",
        f"Git commit: `{commit}`",
        "",
        "Durable record of every generated-video run BEFORE cleanup. Each run's "
        "population metrics come from `merged_summary.json`; the pool "
        "fingerprint is a sha1 over the sorted canonical eval video ids, so runs "
        "with the **same fingerprint were evaluated on the same test pool** and "
        "are directly comparable.",
        "",
        "PSNR/SSIM higher=better; LPIPS/FVD/FID lower=better. `✓/—` in the VBench "
        "column marks whether a VBench score is already computed (missing = must "
        "be backfilled before deleting frames).",
        "",
        "| series | run | mp4 | N | pool | PSNR | SSIM | LPIPS | FVD | FID | VBench |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for e in sorted(entries, key=lambda x: (x["series"], x["run"])):
        m = e["pop_metrics"]
        vb = "✓" if e.get("has_metrics", {}).get("vbench") else "—"
        lines.append(
            f"| {e['series']} | {e['run']} | {e['mp4_count']} | {e['n_eval_videos']} | "
            f"`{e['pool_fingerprint']}` | {_fmt(m.get('psnr'))} | {_fmt(m.get('ssim'),4)} | "
            f"{_fmt(m.get('lpips'),4)} | {_fmt(m.get('fvd'),1)} | {_fmt(m.get('fid'),1)} | {vb} |"
        )
    lines.append("")

    # Shared-pool groupings
    groups: Dict[str, List[dict]] = {}
    for e in entries:
        groups.setdefault(e["pool_fingerprint"], []).append(e)
    multi = {fp: es for fp, es in groups.items()
             if fp != "EMPTY" and len({(x["series"], x["run"]) for x in es}) > 1}
    lines += [
        "## Shared test-pool groups (comparable runs)",
        "",
        "Runs sharing a fingerprint were evaluated on the identical set of eval "
        "video ids — safe to place side-by-side in one comparison table.",
        "",
    ]
    if not multi:
        lines.append("_(no fingerprint shared across >1 run)_")
    for fp, es in sorted(multi.items(), key=lambda kv: -len(kv[1])):
        n = es[0]["n_eval_videos"]
        lines.append(f"- **`{fp}`** (N={n}, {len(es)} runs): "
                     + ", ".join(sorted(f"{x['series']}/{x['run']}" for x in es)))
    lines.append("")

    # VBench-missing but has frames -> backfill candidates
    lines += [
        "## VBench-missing runs that still have frames (backfill BEFORE delete)",
        "",
        "| series | run | mp4 | N | has PSNR? |",
        "|---|---|---:|---:|:--:|",
    ]
    any_bf = False
    for e in sorted(entries, key=lambda x: (x["series"], x["run"])):
        if e["mp4_count"] > 0 and not e.get("has_metrics", {}).get("vbench"):
            any_bf = True
            has_psnr = "✓" if e.get("has_metrics", {}).get("psnr") else "—"
            lines.append(f"| {e['series']} | {e['run']} | {e['mp4_count']} | "
                         f"{e['n_eval_videos']} | {has_psnr} |")
    if not any_bf:
        lines.append("| _(none)_ | | | | |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", action="append", default=[], type=Path,
                    help="results tree to scan (repeatable)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = ap.parse_args()

    roots = args.results_root or [
        Path("sweep_experiment/results"),
        Path("delta_experiment/results"),
        Path("comparison_methods/data"),
        Path("t2v_experiment/results"),
        Path("baseline_experiment/results"),
    ]
    out_dir = args.out_dir or (
        Path("sweep_experiment/reports/cleanup_manifests") / _dt.date.today().isoformat()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pool_ids").mkdir(exist_ok=True)

    entries: List[dict] = []
    for root in roots:
        got = scan_results_root(root)
        print(f"[info] {root}: {len(got)} runs", file=sys.stderr)
        entries.extend(got)

    # JSON (drop the big id lists from the main file; write them separately)
    slim = []
    seen_fp: set = set()
    for e in entries:
        ids = e.pop("eval_ids")
        fp = e["pool_fingerprint"]
        if fp not in ("EMPTY",) and fp not in seen_fp and ids:
            (out_dir / "pool_ids" / f"{fp}.txt").write_text("\n".join(ids) + "\n")
            seen_fp.add(fp)
        slim.append(e)
    (out_dir / "manifest.json").write_text(json.dumps(slim, indent=2))

    # pool fingerprints csv
    with (out_dir / "pool_fingerprints.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["series", "run", "pool_fingerprint", "n_eval_videos", "mp4_count",
                    "has_vbench"])
        for e in sorted(slim, key=lambda x: (x["series"], x["run"])):
            w.writerow([e["series"], e["run"], e["pool_fingerprint"],
                        e["n_eval_videos"], e["mp4_count"],
                        int(bool(e.get("has_metrics", {}).get("vbench")))])

    (out_dir / "MANIFEST.md").write_text(render_markdown(slim, args.repo_root))
    print(f"[info] wrote manifest for {len(slim)} runs -> {out_dir}", file=sys.stderr)
    print(out_dir / "MANIFEST.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
