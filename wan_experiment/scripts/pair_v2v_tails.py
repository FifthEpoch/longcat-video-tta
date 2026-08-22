#!/usr/bin/env python3
"""Paired per-video tail motion from summary.json (no fat sidecars)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _usable(rec: dict) -> bool:
    return bool(
        rec.get("ok")
        and not rec.get("skipped")
        and rec.get("tail_motion") is not None
    )


def _rows(path: Path) -> dict:
    """Prefer per-video sidecars. confirm_32v summary.json is full of skip stubs."""
    out = {}
    d = path.parent if path.name == "summary.json" else path
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            if p.name in {"summary.json", "joined.json"} or "vbench" in p.name:
                continue
            try:
                rec = json.loads(p.read_text())
            except Exception:
                continue
            if not _usable(rec):
                continue
            key = rec.get("file_name") or rec.get("stem") or p.stem
            out[str(key)] = rec
    if path.is_file() and path.name == "summary.json":
        try:
            data = json.loads(path.read_text())
        except Exception:
            data = {}
        for rec in data.get("rows") or []:
            if not _usable(rec):
                continue
            key = rec.get("file_name") or rec.get("stem") or rec.get("mp4")
            if key and str(key) not in out:
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
    ap.add_argument("--series-dir", type=Path, action="append", dest="series_dirs")
    args = ap.parse_args()
    series_dirs = args.series_dirs or [
        Path("wan_experiment/results/v2v_panda_lineage_8v")
    ]

    notta = _rows(args.baseline_dir / "notta_h30s_shard0/summary.json")
    seed = _rows(args.baseline_dir / "seed_bon_h30s_shard0/summary.json")
    methods = []
    seen = set()
    for series_dir in series_dirs:
        for name in (
            "live_bon", "live_hist", "longlive_notta",
            "longlive_sink", "longlive_prefix_sink", "longlive_live_bon",
            "rolling_notta", "rolling_rho_lo", "rolling_rho_hi",
            "rolling_adapt", "rolling_look", "appear_bon",
        ):
            if name in seen:
                continue
            p = series_dir / f"{name}_h30s_shard0/summary.json"
            if p.is_file():
                methods.append((name, _rows(p)))
                seen.add(name)
    for name in ("rolling_notta",):
        if name in seen:
            continue
        p = args.baseline_dir / f"{name}_h30s_shard0/summary.json"
        if p.is_file():
            methods.append((name, _rows(p)))
            seen.add(name)

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
