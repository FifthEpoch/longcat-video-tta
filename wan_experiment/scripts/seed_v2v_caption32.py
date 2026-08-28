#!/usr/bin/env python3
"""Hardlink caption N=32 mp4 + sidecars into an N=128 method dir.

The runner skips existing mp4s, so N=128 only generates indices 32..127
of the same sorted Panda pool. Does not copy summary.json or vbench_*
(those must be rebuilt on the full 128).

    python3 -u wan_experiment/scripts/seed_v2v_caption32.py \
        --src wan_experiment/results/v2v_panda_caption_32v \
        --dst wan_experiment/results/v2v_panda_caption_128v \
        --method notta --keep 32
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

_IDX = re.compile(r"^(\d{3})_")


def _index(name: str) -> int | None:
    m = _IDX.match(name)
    return int(m.group(1)) if m else None


def seed_method(src_root: Path, dst_root: Path, method: str, keep: int) -> int:
    src = src_root / f"{method}_h30s_shard0"
    dst = dst_root / f"{method}_h30s_shard0"
    if not src.is_dir():
        raise FileNotFoundError(f"N=32 dir missing: {src}")
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    stem_bad = 0
    for mp4 in sorted(src.glob("*.mp4")):
        idx = _index(mp4.name)
        if idx is None or idx >= keep:
            continue
        if mp4.stat().st_size <= 10_000:
            print(f"  skip tiny {mp4.name}", flush=True)
            continue
        sidecar = src / f"{mp4.stem}.json"
        if sidecar.is_file():
            rec = json.loads(sidecar.read_text())
            src_prompt = rec.get("prompt_source")
            if src_prompt and src_prompt != "metadata_csv":
                stem_bad += 1
                print(f"  REFUSE {mp4.name} prompt_source={src_prompt}", flush=True)
                continue
        dest_mp4 = dst / mp4.name
        if dest_mp4.exists():
            n += 1
            continue
        try:
            os.link(mp4, dest_mp4)
        except OSError:
            dest_mp4.write_bytes(mp4.read_bytes())
        if sidecar.is_file():
            dest_js = dst / sidecar.name
            if not dest_js.exists():
                try:
                    os.link(sidecar, dest_js)
                except OSError:
                    dest_js.write_text(sidecar.read_text())
        n += 1
    if stem_bad:
        raise SystemExit(f"{method}: {stem_bad} stem-prompt sidecars; not seeded")
    if n < keep:
        print(f"WARN {method}: seeded {n}/{keep} (N=32 may be short)", flush=True)
    else:
        print(f"seeded {method}: {n} mp4+json  {src} -> {dst}", flush=True)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--method", action="append", dest="methods", required=True)
    ap.add_argument("--keep", type=int, default=32)
    args = ap.parse_args()
    total = 0
    for method in args.methods:
        total += seed_method(args.src, args.dst, method, args.keep)
    print(f"total seeded={total} keep={args.keep}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
