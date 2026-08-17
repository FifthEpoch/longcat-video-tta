#!/usr/bin/env python3
"""Paired last-chunk + per-step gate analysis for Wan I2V chunked BoN.

    python wan_experiment/scripts/analyze_i2v_bon.py \
        --series-dir wan_experiment/results/i2v_bon_32v_hybrid
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def _load_rows(series_dir: Path, method: str, horizon_s: float) -> list[dict]:
    h = int(horizon_s)
    rows = []
    for p in sorted(series_dir.glob(f"{method}_h{h}s_shard*/summary.json")):
        data = json.loads(p.read_text())
        rows.extend(r for r in data.get("rows") or [] if r.get("ok"))
    if not rows:
        raise FileNotFoundError(f"no ok rows for {method} under {series_dir}")
    return rows


def _key(row: dict) -> str:
    return row.get("file_name") or row.get("stem") or row.get("mp4")


def _last_score(row: dict):
    if row.get("last_chunk_score") is not None:
        return float(row["last_chunk_score"])
    chunks = row.get("chunks") or []
    if not chunks:
        return None
    last = chunks[-1]
    if last.get("chosen_score") is not None:
        return float(last["chosen_score"])
    cands = last.get("candidates") or []
    chosen = next((c for c in cands if c.get("chosen")), None)
    if chosen and chosen.get("score") is not None:
        return float(chosen["score"])
    return None


def _mean_med(xs: list[float]) -> tuple[float, float]:
    return (statistics.fmean(xs), statistics.median(xs))


def _fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def analyze(series_dir: Path, horizon_s: float) -> dict:
    methods = {
        "notta": _load_rows(series_dir, "notta", horizon_s),
        "always_bon": _load_rows(series_dir, "always_bon", horizon_s),
        "gated_bon": _load_rows(series_dir, "gated_bon", horizon_s),
    }
    by = {m: {_key(r): r for r in rows} for m, rows in methods.items()}
    keys = sorted(set(by["notta"]) & set(by["always_bon"]) & set(by["gated_bon"]))
    paired = []
    for k in keys:
        rec = {"key": k}
        for m in methods:
            rec[m] = _last_score(by[m][k])
            rec[f"{m}_row"] = by[m][k]
        if None in (rec["notta"], rec["always_bon"], rec["gated_bon"]):
            continue
        rec["always_minus_notta"] = rec["always_bon"] - rec["notta"]
        rec["gated_minus_notta"] = rec["gated_bon"] - rec["notta"]
        rec["gated_minus_always"] = rec["gated_bon"] - rec["always_bon"]
        paired.append(rec)

    last = {m: [p[m] for p in paired] for m in methods}
    stats = {
        m: {"mean": _mean_med(last[m])[0], "median": _mean_med(last[m])[1]}
        for m in methods
    }
    deltas = {
        "always_minus_notta": [p["always_minus_notta"] for p in paired],
        "gated_minus_notta": [p["gated_minus_notta"] for p in paired],
        "gated_minus_always": [p["gated_minus_always"] for p in paired],
    }
    delta_stats = {
        name: {
            "mean": statistics.fmean(xs),
            "median": statistics.median(xs),
            "n_better": sum(1 for x in xs if x < 0),
            "n_tie": sum(1 for x in xs if x == 0),
            "n_worse": sum(1 for x in xs if x > 0),
            "n_better_or_tie": sum(1 for x in xs if x <= 0),
        }
        for name, xs in deltas.items()
    }

    reason_ctr = Counter()
    fire_chunks = 0
    searchable = 0
    walls = {m: [] for m in methods}
    for p in paired:
        grow = p["gated_bon_row"]
        walls["notta"].append(p["notta_row"].get("seconds"))
        walls["always_bon"].append(p["always_bon_row"].get("seconds"))
        walls["gated_bon"].append(grow.get("seconds"))
        for ch in grow.get("chunks") or []:
            if ch.get("chunk", 0) < int(grow.get("search_from_chunk") or 1):
                continue
            searchable += 1
            reason_ctr[ch.get("gate_reason") or "unknown"] += 1
            if ch.get("gated_fired"):
                fire_chunks += 1

    wall_stats = {}
    for m, xs in walls.items():
        xs = [x for x in xs if x is not None]
        wall_stats[m] = {
            "mean": statistics.fmean(xs) if xs else None,
            "n": len(xs),
        }

    return {
        "n_paired": len(paired),
        "stats": stats,
        "delta_stats": delta_stats,
        "reason_counts": dict(reason_ctr),
        "n_searchable_chunks": searchable,
        "n_gated_fired": fire_chunks,
        "wall": wall_stats,
        "paired": paired,
    }


def _md_table(report: dict) -> str:
    s = report["stats"]
    d = report["delta_stats"]
    lines = [
        "# Wan I2V hybrid-gate last-chunk (auto)",
        "",
        f"N paired = {report['n_paired']}. Lower composite is better.",
        "",
        "| Method | Mean | Median | Mean wall (s) |",
        "|---|---|---|---|",
    ]
    for m, label in (
        ("notta", "NOTTA"),
        ("always_bon", "always-BoN k=4"),
        ("gated_bon", "gated-BoN hybrid"),
    ):
        w = report["wall"][m]["mean"]
        lines.append(
            f"| {label} | {_fmt(s[m]['mean'])} | {_fmt(s[m]['median'])} | {_fmt(w, 1)} |"
        )
    lines += [
        "",
        "| Contrast | Mean Δ | Median Δ | better | tie | worse | better-or-tie |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, label in (
        ("always_minus_notta", "always − NOTTA"),
        ("gated_minus_notta", "gated − NOTTA"),
        ("gated_minus_always", "gated − always"),
    ):
        x = d[name]
        lines.append(
            f"| {label} | {_fmt(x['mean'])} | {_fmt(x['median'])} | "
            f"{x['n_better']} | {x['n_tie']} | {x['n_worse']} | "
            f"{x['n_better_or_tie']}/{report['n_paired']} |"
        )
    lines += [
        "",
        f"Gated fired {report['n_gated_fired']}/{report['n_searchable_chunks']} "
        f"searchable chunks. Reasons: `{report['reason_counts']}`.",
        "",
        "## Per-video last-chunk",
        "",
        "| i | key | NOTTA | always | gated | always−N | gated−N | gated−A |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for i, p in enumerate(report["paired"]):
        key = str(p["key"])[:40]
        lines.append(
            f"| {i:02d} | {key} | {_fmt(p['notta'])} | {_fmt(p['always_bon'])} | "
            f"{_fmt(p['gated_bon'])} | {_fmt(p['always_minus_notta'])} | "
            f"{_fmt(p['gated_minus_notta'])} | {_fmt(p['gated_minus_always'])} |"
        )
    lines += [
        "",
        "## Per-video gate trace (gated method)",
        "",
        "| i | reasons | incoming | Δincoming | chosen−cand0 | last-1s out |",
        "|---|---|---|---|---|---|",
    ]
    for i, p in enumerate(report["paired"]):
        grow = p["gated_bon_row"]
        reasons = grow.get("gate_reasons") or [
            (ch.get("gate_reason") or "?") for ch in (grow.get("chunks") or [])
        ]
        inc = grow.get("incoming_series") or [
            ch.get("incoming_drift") for ch in (grow.get("chunks") or [])
        ]
        dlt = [
            ch.get("incoming_delta") for ch in (grow.get("chunks") or [])
        ]
        vs0 = grow.get("chosen_minus_cand0_series") or [
            ch.get("chosen_minus_cand0") for ch in (grow.get("chunks") or [])
        ]
        out = grow.get("outgoing_series") or [
            ch.get("outgoing_drift") for ch in (grow.get("chunks") or [])
        ]
        lines.append(
            f"| {i:02d} | {reasons} | {[_fmt(x) for x in inc]} | "
            f"{[_fmt(x) for x in dlt]} | {[_fmt(x) for x in vs0]} | "
            f"{[_fmt(x) for x in out]} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", required=True)
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--out-md", default="")
    args = ap.parse_args()
    series_dir = Path(args.series_dir).resolve()
    report = analyze(series_dir, args.horizon_s)
    slim = {k: report[k] for k in report if k != "paired"}
    print(json.dumps(slim, indent=2))
    print()
    md = _md_table(report)
    print(md)
    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
