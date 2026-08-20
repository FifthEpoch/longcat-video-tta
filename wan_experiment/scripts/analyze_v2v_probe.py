#!/usr/bin/env python3
"""Read knob_probe summary and decide whether shift / CFG are live.

    python wan_experiment/scripts/analyze_v2v_probe.py \
        --series-dir wan_experiment/results/v2v_panda_probe
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load(series_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(series_dir.glob("knob_probe_h*s_shard*/summary.json")):
        data = json.loads(p.read_text())
        rows.extend(r for r in data.get("rows") or [] if r.get("ok") and r.get("probe"))
    if not rows:
        raise FileNotFoundError(f"no knob_probe rows under {series_dir}")
    return rows


def analyze(series_dir: Path) -> dict:
    clips = _load(series_dir)
    shift_votes = [bool(c.get("shift_live")) for c in clips]
    cfg_votes = [bool(c.get("cfg_live")) for c in clips]
    recs = [c.get("recommendation") for c in clips]
    by_knob: dict[tuple, list[float]] = {}
    for c in clips:
        for row in (c.get("probe") or {}).get("rows") or []:
            key = (row.get("shift"), row.get("cfg"))
            mot = row.get("temporal_motion")
            if mot is not None:
                by_knob.setdefault(key, []).append(float(mot))
    lines = [
        "# V2V knob probe",
        "",
        f"clips={len(clips)}  shift_live={sum(shift_votes)}/{len(shift_votes)}  "
        f"cfg_live={sum(cfg_votes)}/{len(cfg_votes)}",
        "",
        "| shift | cfg | mean motion |",
        "|---:|---:|---:|",
    ]
    for (shift, cfg), xs in sorted(by_knob.items()):
        lines.append(f"| {shift} | {cfg} | {statistics.fmean(xs):.5f} |")
    keep_shift = sum(shift_votes) > 0
    keep_cfg = sum(cfg_votes) > 0
    lines += [
        "",
        f"**Keep shift_search:** {keep_shift}",
        f"**Keep CFG search:** {keep_cfg}",
        "",
        "If shift_live is false, submit the N=8 bake-off with `SKIP_SHIFT=1`.",
        "CFG stays wave-2 unless cfg_live is true (unexpected on DMD).",
        "",
        "Per-clip recommendations:",
    ]
    for c, rec in zip(clips, recs):
        lines.append(f"- {c.get('file_name')}: {rec}")
    text = "\n".join(lines) + "\n"
    return {
        "shift_live": keep_shift,
        "cfg_live": keep_cfg,
        "n": len(clips),
        "markdown": text,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    result = analyze(args.series_dir)
    print(result["markdown"])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(result["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
