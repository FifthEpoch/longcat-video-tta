#!/usr/bin/env python3
"""Derive across-timestep diffusion-loss variance from OOD CSV (H-T2-5).

Reads ``diffusion_ood_scores.csv`` produced by ``compute_diffusion_ood_score.py``
and emits per-video variance of caption-conditioned and unconditional losses
across the per-timestep columns.

Output CSV:
    video_id, loss_var_caption, loss_var_uncond, n_timesteps

Run:
    python3 scripts/derive_loss_variance.py \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-09/loss_var_features.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

_CAP_RE = re.compile(r"^diffusion_loss_(caption|uncond)_t(\d+)$")


def _parse_timestep_cols(fieldnames: List[str], mode: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for fn in fieldnames:
        m = _CAP_RE.match(fn)
        if m and m.group(1) == mode:
            out.append((int(m.group(2)), fn))
    return sorted(out, key=lambda x: x[0])


def _coerce(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or math.isinf(x):
        return None
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ood-csv", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if not args.ood_csv.exists():
        print(f"[error] --ood-csv not found: {args.ood_csv}", file=sys.stderr)
        return 2

    with args.ood_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"[error] {args.ood_csv} has no header", file=sys.stderr)
            return 2
        cap_cols = _parse_timestep_cols(reader.fieldnames, "caption")
        unc_cols = _parse_timestep_cols(reader.fieldnames, "uncond")
        rows_in = list(reader)

    if not cap_cols:
        print("[error] no diffusion_loss_caption_t* columns found", file=sys.stderr)
        return 2

    fieldnames = ["video_id", "loss_var_caption", "loss_var_uncond", "n_timesteps"]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows_in:
            vid = (r.get("video_id") or "").strip()
            if not vid:
                continue
            cap_vals = [_coerce(r.get(c)) for _, c in cap_cols]
            cap_vals = [x for x in cap_vals if x is not None]
            unc_vals = [_coerce(r.get(c)) for _, c in unc_cols]
            unc_vals = [x for x in unc_vals if x is not None]
            var_cap = float(np.var(cap_vals, ddof=0)) if len(cap_vals) >= 2 else ""
            var_unc = float(np.var(unc_vals, ddof=0)) if len(unc_vals) >= 2 else ""
            writer.writerow({
                "video_id": vid,
                "loss_var_caption": f"{var_cap:.6f}" if var_cap != "" else "",
                "loss_var_uncond": f"{var_unc:.6f}" if var_unc != "" else "",
                "n_timesteps": len(cap_cols),
            })

    print(f"Wrote {args.output}  ({len(rows_in)} rows, {len(cap_cols)} timesteps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
