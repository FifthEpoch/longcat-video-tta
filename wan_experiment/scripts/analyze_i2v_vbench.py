#!/usr/bin/env python3
"""Compare official VBench outcomes across do-nothing / always / gated.

Also correlates each VBench dimension (higher=better) with the handcrafted
last-chunk composite (lower=better). A working verifier should show a
*negative* Spearman rho. Near-zero rho means the handcrafted score is not
a useful proxy for standard quality.

    python wan_experiment/scripts/analyze_i2v_vbench.py \
        --series-dir wan_experiment/results/i2v_bon_32v_hybrid \
        --clip last5
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


METHODS = ("notta", "always_bon", "gated_bon")
LABEL = {
    "notta": "do-nothing",
    "always_bon": "always-search",
    "gated_bon": "gated-search",
}


def _ranks(xs: list[float]) -> list[float]:
    n = len(xs)
    order = sorted(range(n), key=lambda i: xs[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: list[float], ys: list[float]):
    if len(xs) < 3:
        return None
    mx = statistics.fmean(xs)
    my = statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(xs: list[float], ys: list[float]):
    return _pearson(_ranks(xs), _ranks(ys))


def _load_joined(series_dir: Path, method: str, clip: str, horizon_s: float) -> dict:
    h = int(horizon_s)
    hits = sorted(series_dir.glob(f"{method}_h{h}s_shard*/vbench_{clip}/joined.json"))
    if not hits:
        raise FileNotFoundError(
            f"no vbench_{clip}/joined.json for {method} under {series_dir}"
        )
    return json.loads(hits[0].read_text())


def _by_key(joined: dict) -> dict[str, dict]:
    out = {}
    for rec in joined.get("per_video") or []:
        key = rec.get("file_name") or rec.get("stem")
        if key:
            out[str(key)] = rec
    return out


def _fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def analyze(series_dir: Path, clip: str, horizon_s: float) -> dict:
    loaded = {m: _load_joined(series_dir, m, clip, horizon_s) for m in METHODS}
    by = {m: _by_key(loaded[m]) for m in METHODS}
    keys = sorted(set.intersection(*(set(by[m]) for m in METHODS)))
    dims = []
    for m in METHODS:
        for rec in by[m].values():
            for d in (rec.get("vbench") or {}):
                if d not in dims:
                    dims.append(d)

    pop = {}
    for m in METHODS:
        pop[m] = {}
        for d in dims:
            xs = [by[m][k]["vbench"][d] for k in keys if d in (by[m][k].get("vbench") or {})]
            pop[m][d] = {
                "n": len(xs),
                "mean": statistics.fmean(xs) if xs else None,
                "median": statistics.median(xs) if xs else None,
            }

    paired = []
    for k in keys:
        rec = {"key": k}
        for m in METHODS:
            rec[m] = dict(by[m][k].get("vbench") or {})
            rec[f"{m}_last"] = by[m][k].get("last_chunk_score")
        paired.append(rec)

    wins = {d: {"gated>always": 0, "tie": 0, "gated<always": 0} for d in dims}
    for rec in paired:
        for d in dims:
            g = rec["gated_bon"].get(d)
            a = rec["always_bon"].get(d)
            if g is None or a is None:
                continue
            if g > a + 1e-9:
                wins[d]["gated>always"] += 1
            elif a > g + 1e-9:
                wins[d]["gated<always"] += 1
            else:
                wins[d]["tie"] += 1

    corr = {}
    for m in METHODS:
        corr[m] = {}
        for d in dims:
            xs, ys = [], []
            for rec in paired:
                last = rec.get(f"{m}_last")
                v = rec[m].get(d)
                if last is None or v is None:
                    continue
                xs.append(float(last))
                ys.append(float(v))
            corr[m][d] = {
                "n": len(xs),
                "spearman": spearman(xs, ys) if len(xs) >= 3 else None,
            }

    return {
        "series_dir": str(series_dir),
        "clip": clip,
        "n_paired": len(keys),
        "dimensions": dims,
        "population": pop,
        "wins": wins,
        "correlation": corr,
        "keys": keys,
    }


def render(result: dict) -> str:
    dims = result["dimensions"]
    pop = result["population"]
    lines = [
        f"# Wan I2V official VBench — `{Path(result['series_dir']).name}` / {result['clip']}",
        "",
        f"Paired videos: **{result['n_paired']}**. "
        "VBench higher is better. Handcrafted last-chunk lower is better.",
        "",
        "## Population (median / mean)",
        "",
        "| Dimension | do-nothing | always-search | gated-search | gated−always (med) |",
        "|---|---:|---:|---:|---:|",
    ]
    for d in dims:
        cells = []
        meds = {}
        for m in METHODS:
            cell = pop[m][d]
            meds[m] = cell["median"]
            cells.append(f"{_fmt(cell['median'])} / {_fmt(cell['mean'])}")
        delta = None
        if meds["gated_bon"] is not None and meds["always_bon"] is not None:
            delta = meds["gated_bon"] - meds["always_bon"]
        lines.append(
            f"| {d} | {cells[0]} | {cells[1]} | {cells[2]} | {_fmt(delta)} |"
        )
    lines += [
        "",
        "## Gated vs always (per-video, higher VBench wins)",
        "",
        "| Dimension | gated>always | tie | gated<always |",
        "|---|---:|---:|---:|",
    ]
    for d in dims:
        w = result["wins"][d]
        lines.append(
            f"| {d} | {w['gated>always']} | {w['tie']} | {w['gated<always']} |"
        )
    lines += [
        "",
        "## Does the handcrafted last-chunk score track VBench?",
        "",
        "Spearman rho(last-chunk, VBench dim). Expected sign if the verifier "
        "is a useful quality proxy: **negative**. Cite this before claiming "
        "the composite detects drift that helps performance.",
        "",
        "| Dimension | do-nothing ρ | always-search ρ | gated-search ρ |",
        "|---|---:|---:|---:|",
    ]
    for d in dims:
        cells = [_fmt(result["correlation"][m][d]["spearman"]) for m in METHODS]
        lines.append(f"| {d} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines += [
        "",
        "## How to read this",
        "",
        "- Quality win for gating: gated median ≥ always on most dims, "
        "and gated>always counts beat gated<always.",
        "- Efficiency-only: VBench tie, gated cheaper (already known on the "
        "hybrid 32v wall clock).",
        "- Verifier broken / misaligned: rho near 0, or positive rho on "
        "`imaging_quality` / `dynamic_degree` (our score punishes sharp/"
        "freeze deviation; official dims may reward raw sharpness or motion).",
        "- These 32 clips have **no 30 s GT video**. Do not invent PSNR.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", required=True, type=Path)
    ap.add_argument("--clip", default="last5", choices=["full", "last5", "first5"])
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = analyze(args.series_dir, args.clip, args.horizon_s)
    text = render(result)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
