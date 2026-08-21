#!/usr/bin/env python3
"""Paired per-video tail motion from summary.json (no fat sidecars)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rows(path: Path) -> dict:
    data = json.loads(path.read_text())
    out = {}
    for rec in data.get("rows") or []:
        if not rec.get("ok") or rec.get("skipped"):
            continue
        key = rec.get("file_name") or rec.get("stem") or rec.get("mp4")
        if key:
            out[str(key)] = rec
    return out


def _chunk0(rec: dict) -> dict:
    chunks = rec.get("chunks") or []
    return chunks[0] if chunks else {}


def _fmt(x) -> str:
    try:
        if x is None or x != x:
            return "     nan"
    except Exception:
        return "     nan"
    return f"{float(x):8.5f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path,
                    default=Path("wan_experiment/results/v2v_panda_bakeoff_8v"))
    ap.add_argument("--series-dir", type=Path,
                    default=Path("wan_experiment/results/v2v_panda_lineage_8v"))
    args = ap.parse_args()

    notta = _rows(args.baseline_dir / "notta_h30s_shard0/summary.json")
    seed = _rows(args.baseline_dir / "seed_bon_h30s_shard0/summary.json")
    methods = []
    for name in (
        "live_bon", "live_hist", "longlive_notta",
        "longlive_sink", "longlive_prefix_sink", "longlive_live_bon",
        "rolling_notta",
    ):
        p = args.series_dir / f"{name}_h30s_shard0/summary.json"
        if p.is_file():
            methods.append((name, _rows(p)))

    hdr = f"{'video':<16} {'notta':>8} {'seed':>8}"
    for name, _ in methods:
        hdr += f" {name[:8]:>8}"
    hdr += "  prefix  n_div  c0_gate"
    print(hdr)
    for key in sorted(notta):
        n = notta[key]
        s = seed.get(key) or {}
        line = f"{key:<16} {_fmt(n.get('tail_motion'))} {_fmt(s.get('tail_motion'))}"
        first = methods[0][1].get(key) if methods else {}
        c0 = _chunk0(first)
        for _, mapping in methods:
            rec = mapping.get(key) or {}
            line += f" {_fmt(rec.get('tail_motion'))}"
        pm = first.get("prefix_motion")
        if pm is None:
            pm = c0.get("prefix_motion")
        nd = first.get("n_divergent_chunks")
        print(
            f"{line}  {_fmt(pm).strip():>7}  {nd!s:>5}  "
            f"{c0.get('gate_reason')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
