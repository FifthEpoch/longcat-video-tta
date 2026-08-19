#!/usr/bin/env python3
"""Resolve the 128 MovieGen prompts for t2v_bon_128v_vbenchlong.

Preference:
  1. --prompt-file if given
  2. Self-Forcing MovieGenVideoBench_extended.txt (Qwen-refined, field default)
  3. Vendored official first 128 (datasets/moviegen_128.txt)

Writes a one-prompt-per-line file. Does not call Qwen.

    python wan_experiment/scripts/prepare_t2v_prompts.py \
        --sf-root /scratch/wc3013/third_party/Self-Forcing \
        --out datasets/moviegen_128_resolved.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path


N = 128


def _read_prompts(path: Path, n: int) -> list[str]:
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < n:
        raise SystemExit(f"{path} has {len(lines)} prompts; need {n}")
    return lines[:n]


def resolve_prompt_file(sf_root: Path | None, explicit: Path | None, vendor: Path) -> tuple[Path, str]:
    if explicit is not None:
        p = explicit.expanduser().resolve()
        if not p.is_file():
            raise SystemExit(f"--prompt-file not found: {p}")
        return p, "explicit"
    if sf_root is not None:
        ext = (sf_root / "prompts" / "MovieGenVideoBench_extended.txt").resolve()
        if ext.is_file():
            return ext, "self_forcing_extended"
        raw = (sf_root / "prompts" / "MovieGenVideoBench.txt").resolve()
        if raw.is_file():
            return raw, "self_forcing_raw"
    vendor = vendor.resolve()
    if vendor.is_file():
        return vendor, "vendor_moviegen_128"
    raise SystemExit(
        "no MovieGen prompt file found. Pass --prompt-file or vendor "
        f"{vendor}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf-root", default="")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--vendor", default="datasets/moviegen_128.txt")
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src, kind = resolve_prompt_file(
        Path(args.sf_root) if args.sf_root else None,
        Path(args.prompt_file) if args.prompt_file else None,
        Path(args.vendor),
    )
    prompts = _read_prompts(src, args.n)
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(prompts) + "\n")
    print(f"source={kind} path={src} n={len(prompts)} out={out}")
    print(f"first={prompts[0][:80]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
