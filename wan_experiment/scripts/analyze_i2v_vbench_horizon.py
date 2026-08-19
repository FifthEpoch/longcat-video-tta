#!/usr/bin/env python3
"""5 s vs 30 s VBench++, including first-16 / last-16 like the drift table.

    python wan_experiment/scripts/analyze_i2v_vbench_horizon.py \
        --series-dir wan_experiment/results/i2v_notta_16v \
        --out sweep_experiment/reports/paper_tables/2026-08-19_wan_i2v_notta16_vbench_headtail.md

Reads vbench_{full,first5,first1,last1}/joined.json under h5s_shard* and
h30s_shard*. first1/last1 skip cond frame 0 and take 16 frames — same
window as score_i2v_drift.py. Those 1 s clips are diagnostics; cite
full (5 s) and first5 (30 s opening) for a same-duration VBench++ pair.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from analyze_i2v_vbench import _fmt  # noqa: E402

HORIZONS = (5, 30)
CLIPS = ("first1", "last1", "first5", "full")
CLIP_LABEL = {
    "first1": "first 16 fr (skip f0)",
    "last1": "last 16 fr",
    "first5": "first 5 s",
    "full": "full clip",
}


def _load(series_dir: Path, horizon_s: int, clip: str) -> dict | None:
    h = int(horizon_s)
    hits = sorted(series_dir.glob(f"h{h}s_shard*/vbench_{clip}/joined.json"))
    if not hits:
        hits = sorted(series_dir.glob(f"*_h{h}s_shard*/vbench_{clip}/joined.json"))
    if not hits:
        return None
    return json.loads(hits[0].read_text())


def _keys(joined: dict) -> dict[str, dict]:
    out = {}
    for rec in joined.get("per_video") or []:
        key = rec.get("file_name") or rec.get("stem")
        if key:
            out[str(key)] = rec
    return out


def _pop(joined: dict, dim: str) -> dict:
    xs = []
    for rec in joined.get("per_video") or []:
        v = (rec.get("vbench") or {}).get(dim)
        if v is not None:
            xs.append(float(v))
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs) if xs else None,
        "median": statistics.median(xs) if xs else None,
    }


def _rel(head: float | None, tail: float | None):
    if head is None or tail is None:
        return None
    if abs(head) < 1e-12:
        return None
    return (tail - head) / head


def analyze(series_dir: Path) -> dict:
    loaded = {}
    missing = []
    dims: list[str] = []
    paired: set[str] | None = None
    for h in HORIZONS:
        loaded[h] = {}
        for clip in CLIPS:
            j = _load(series_dir, h, clip)
            if j is None:
                missing.append(f"h{h}s/{clip}")
                continue
            loaded[h][clip] = j
            keys = set(_keys(j))
            paired = keys if paired is None else (paired & keys)
            for rec in j.get("per_video") or []:
                for d in rec.get("vbench") or {}:
                    if d not in dims:
                        dims.append(d)
    pop = {}
    for h in HORIZONS:
        pop[h] = {}
        for clip in CLIPS:
            if clip not in loaded[h]:
                continue
            pop[h][clip] = {d: _pop(loaded[h][clip], d) for d in dims}
    return {
        "series_dir": str(series_dir),
        "n_paired": len(paired or []),
        "dimensions": dims,
        "population": pop,
        "missing": missing,
        "clips_present": {
            h: sorted(loaded[h]) for h in HORIZONS
        },
    }


def render(result: dict) -> str:
    dims = result["dimensions"]
    pop = result["population"]
    lines = [
        f"# Wan I2V 5 s vs 30 s VBench++ — `{Path(result['series_dir']).name}`",
        "",
        f"Paired videos: **{result['n_paired']}**. Higher is better. "
        "`first1` / `last1` are **16-frame diagnostics** (skip cond frame 0), "
        "the same windows as `score_i2v_drift.py`. They are not official "
        "VBench++. Cite **5 s full** vs **30 s first5** for a same-duration "
        "quality pair. 5 s and 30 s were **separate generates** (same "
        "images/seed); first1 is not a shared prefix.",
        "",
    ]
    if result["missing"]:
        lines += [
            "Missing (score these first): " + ", ".join(result["missing"]),
            "",
        ]
    lines += [
        "## Head vs tail (16 frames)",
        "",
        "| Dimension | 5 s first16 | 5 s last16 | 5 s Δrel | "
        "30 s first16 | 30 s last16 | 30 s Δrel |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for d in dims:
        cells = []
        rels = {}
        for h in HORIZONS:
            head = (pop.get(h, {}).get("first1") or {}).get(d) or {}
            tail = (pop.get(h, {}).get("last1") or {}).get(d) or {}
            cells.append(f"{_fmt(head.get('median'))} / {_fmt(head.get('mean'))}")
            cells.append(f"{_fmt(tail.get('median'))} / {_fmt(tail.get('mean'))}")
            rels[h] = _rel(head.get("median"), tail.get("median"))
            cells.append(_fmt(rels[h]))
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Δrel = (last16 − first16) / first16 on **medians**. Same formula "
        "as the handpicked drift table. Sign meaning differs: VBench "
        "higher is better, so negative Δrel is a quality drop. Handpicked "
        "sharpness *up* is oversharpening (bad), not a VBench win.",
        "",
        "## Same-duration 5 s window",
        "",
        "| Dimension | 5 s full | 30 s first5 | 30 s − 5 s (med) |",
        "|---|---:|---:|---:|",
    ]
    for d in dims:
        a = (pop.get(5, {}).get("full") or {}).get(d) or {}
        b = (pop.get(30, {}).get("first5") or {}).get(d) or {}
        delta = None
        if a.get("median") is not None and b.get("median") is not None:
            delta = b["median"] - a["median"]
        lines.append(
            f"| {d} | {_fmt(a.get('median'))} / {_fmt(a.get('mean'))} | "
            f"{_fmt(b.get('median'))} / {_fmt(b.get('mean'))} | {_fmt(delta)} |"
        )
    lines += [
        "",
        "## Full clip (unequal length; not a duration-matched pair)",
        "",
        "| Dimension | 5 s full (~85 fr) | 30 s full (~481 fr) |",
        "|---|---:|---:|",
    ]
    for d in dims:
        a = (pop.get(5, {}).get("full") or {}).get(d) or {}
        b = (pop.get(30, {}).get("full") or {}).get(d) or {}
        lines.append(
            f"| {d} | {_fmt(a.get('median'))} / {_fmt(a.get('mean'))} | "
            f"{_fmt(b.get('median'))} / {_fmt(b.get('mean'))} |"
        )
    lines += [
        "",
        "## How to read",
        "",
        "- Official comparable VBench++ on these 16 is **5 s full**. "
        "30 s full is a long-clip diagnostic, not VBench-I2V's 5 s recipe.",
        "- If 16-frame Δrel at 30 s is much worse than at 5 s, VBench "
        "sees the same horizon stress the handpicked table reported "
        "(sharp +167% / motion −60%).",
        "- `dynamic_degree` is 0/1 RAFT. Means are the fraction of clips "
        "called dynamic; medians are often 0.",
        "- Do not invent PSNR. These stills have no paired 30 s GT.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = analyze(args.series_dir)
    text = render(result)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    if result["missing"]:
        print("missing: " + ", ".join(result["missing"]), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
