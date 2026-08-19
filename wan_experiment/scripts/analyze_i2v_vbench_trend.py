#!/usr/bin/env python3
"""VBench++ trend over successive 5 s windows.

    python wan_experiment/scripts/analyze_i2v_vbench_trend.py \
        --series-dir wan_experiment/results/i2v_bon_32v_hybrid \
        --out sweep_experiment/reports/paper_tables/2026-08-18_wan_i2v_bon32_vbench_trend.md
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from analyze_i2v_vbench import LABEL, METHODS, _fmt  # noqa: E402

WINDOW_DIR_RE = re.compile(r"^vbench_w(\d+)_(\d+)$")


def _discover_windows(series_dir: Path, horizon_s: float) -> list[tuple[int, int, str]]:
    found = set()
    for p in series_dir.glob(f"*_h{int(horizon_s)}s_shard*/vbench_w*_*/joined.json"):
        m = WINDOW_DIR_RE.match(p.parent.name)
        if m:
            found.add((int(m.group(1)), int(m.group(2)), p.parent.name[len("vbench_"):]))
    return sorted(found)


def _load(series_dir: Path, method: str, clip: str, horizon_s: float) -> dict:
    hits = sorted(series_dir.glob(
        f"{method}_h{int(horizon_s)}s_shard*/vbench_{clip}/joined.json"
    ))
    if not hits:
        raise FileNotFoundError(
            f"no vbench_{clip}/joined.json for {method} under {series_dir}"
        )
    return json.loads(hits[0].read_text())


def _keys(joined: dict) -> dict[str, dict]:
    out = {}
    for rec in joined.get("per_video") or []:
        key = rec.get("file_name") or rec.get("stem")
        if key:
            out[str(key)] = rec
    return out


def analyze(series_dir: Path, horizon_s: float) -> dict:
    windows = _discover_windows(series_dir, horizon_s)
    if not windows:
        raise FileNotFoundError(
            f"no vbench_w*_*/joined.json under {series_dir}. "
            "Score with --clip windows first."
        )
    dims = []
    pop = {m: {} for m in METHODS}
    paired_keys = None
    for start, end, clip in windows:
        loaded = {}
        by = {}
        for m in METHODS:
            loaded[m] = _load(series_dir, m, clip, horizon_s)
            by[m] = _keys(loaded[m])
        keys = set.intersection(*(set(by[m]) for m in METHODS))
        paired_keys = keys if paired_keys is None else (paired_keys & keys)
        for m in METHODS:
            for rec in by[m].values():
                for d in rec.get("vbench") or {}:
                    if d not in dims:
                        dims.append(d)
            pop[m][clip] = {}
            for d in dims:
                xs = [
                    by[m][k]["vbench"][d]
                    for k in keys
                    if d in (by[m][k].get("vbench") or {})
                ]
                pop[m][clip][d] = {
                    "n": len(xs),
                    "mean": statistics.fmean(xs) if xs else None,
                    "median": statistics.median(xs) if xs else None,
                }
    return {
        "series_dir": str(series_dir),
        "n_paired": len(paired_keys or []),
        "windows": [
            {"start_s": s, "end_s": e, "clip": c, "label": f"{s}–{e} s"}
            for s, e, c in windows
        ],
        "dimensions": dims,
        "population": pop,
    }


def render(result: dict) -> str:
    dims = result["dimensions"]
    wins = result["windows"]
    pop = result["population"]
    header = "| Window | " + " | ".join(LABEL[m] for m in METHODS) + " |"
    sep = "|---|" + "---:|" * len(METHODS)
    lines = [
        f"# Wan VBench++ 5 s window trend — `{Path(result['series_dir']).name}`",
        "",
        f"Paired videos: **{result['n_paired']}**. "
        "Each cell is median / mean. Higher is better. "
        "These windows are diagnostics; the official comparable number "
        "is still the **full clip**.",
        "",
    ]
    for d in dims:
        lines += [f"## {d}", "", header, sep]
        for w in wins:
            cells = []
            for m in METHODS:
                cell = pop[m].get(w["clip"], {}).get(d) or {}
                cells.append(f"{_fmt(cell.get('median'))} / {_fmt(cell.get('mean'))}")
            lines.append(f"| {w['label']} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## How to read",
        "",
        "- A falling `dynamic_degree` **mean** (fraction of clips RAFT "
        "calls dynamic) is the freeze trend. Median 0 in every window "
        "means most clips were already still.",
        "- `imaging_quality` / `aesthetic_quality` drifting after piece 0 "
        "is where search actually acts (piece 0 is shared).",
        "- Do not replace the full-clip VBench++ table with any one window.",
        "",
    ]
    return "\n".join(lines)


def chart_payload(result: dict) -> dict:
    """JSON the canvas can paste: one series per method, per dimension."""
    dims = {}
    cats = [w["label"] for w in result["windows"]]
    for d in result["dimensions"]:
        series = []
        for m in METHODS:
            series.append({
                "name": LABEL[m],
                "data": [
                    (result["population"][m].get(w["clip"], {}).get(d) or {}).get("median")
                    for w in result["windows"]
                ],
            })
        dims[d] = {"categories": cats, "series": series}
    return {
        "series_dir": result["series_dir"],
        "n_paired": result["n_paired"],
        "windows": result["windows"],
        "dimensions": dims,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", required=True, type=Path)
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    result = analyze(args.series_dir, args.horizon_s)
    text = render(result)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    json_path = args.json_out
    if json_path is None and args.out:
        json_path = args.out.with_suffix(".json")
    if json_path:
        json_path.write_text(json.dumps(chart_payload(result), indent=2))
        print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
