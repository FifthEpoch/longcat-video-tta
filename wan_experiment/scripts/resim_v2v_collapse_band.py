#!/usr/bin/env python3
"""Offline collapse-gate + motion-band picker on existing V2V cand logs.

No new generate. Per-chunk picks only — not a counterfactual 30 s tail.

    python3 -u wan_experiment/scripts/resim_v2v_collapse_band.py \
      --method-dir wan_experiment/results/v2v_panda_confirm_32v/seed_bon_h30s_shard0 \
      --method-dir wan_experiment/results/v2v_panda_live_32v/live_bon_h30s_shard0 \
      --method-dir wan_experiment/results/v2v_panda_bakeoff_8v/seed_bon_h30s_shard0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


COLLAPSE_FRAC = 0.7
BAND = (0.85, 1.15)
STILL_PREFIX = 1e-3


def _motion(c: dict) -> float:
    if c.get("temporal_motion") is not None:
        try:
            return float(c["temporal_motion"])
        except (TypeError, ValueError):
            pass
    sig = c.get("signals") or {}
    m = sig.get("temporal_motion")
    try:
        return float(m)
    except (TypeError, ValueError):
        return float("nan")


def _appear(c: dict) -> float:
    for k in ("appear_score", "hinge_score", "score"):
        v = c.get(k)
        if v is not None:
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if x == x:
                return x
    return 0.0


def _cand0(cands: list[dict]) -> dict | None:
    for c in cands:
        if int(c.get("cand", -1)) == 0:
            return c
    return cands[0] if cands else None


def _pick(cands: list[dict], prefix: float | None) -> tuple[int, str]:
    """Return (cand_id, reason)."""
    if not cands:
        return 0, "empty"
    c0 = _cand0(cands)
    m0 = _motion(c0) if c0 else float("nan")
    if prefix is None or prefix != prefix or prefix < STILL_PREFIX:
        return 0, "still_prefix"
    if not (m0 == m0) or m0 >= COLLAPSE_FRAC * prefix:
        return 0, "no_collapse"
    lo, hi = BAND[0] * prefix, BAND[1] * prefix
    feasible = []
    for c in cands:
        m = _motion(c)
        if m == m and lo <= m <= hi:
            feasible.append(c)
    if feasible:
        best = min(feasible, key=_appear)
        return int(best.get("cand", 0)), "band_appear"
    best = min(
        cands,
        key=lambda c: (
            abs(_motion(c) - prefix)
            if _motion(c) == _motion(c)
            else 1e9
        ),
    )
    return int(best.get("cand", 0)), "nearest_prefix"


def _load_videos(method_dir: Path) -> list[dict]:
    rows = []
    for p in sorted(method_dir.glob("*.json")):
        if p.name in {"summary.json", "joined.json"} or "vbench" in p.name:
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if not rec.get("ok") or rec.get("skipped"):
            continue
        chunks = rec.get("chunks") or []
        if not chunks:
            continue
        key = rec.get("file_name") or rec.get("stem") or p.stem
        rows.append({"key": str(key), "tail": rec.get("tail_motion"), "chunks": chunks})
    return rows


def _orig_cand(ch: dict) -> int:
    if ch.get("chosen_cand") is not None:
        return int(ch["chosen_cand"])
    for c in ch.get("candidates") or []:
        if c.get("chosen"):
            return int(c.get("cand", 0))
    return 0


def analyze(method_dir: Path) -> None:
    rows = _load_videos(method_dir)
    print(f"\n===== {method_dir}  n={len(rows)} =====")
    print(
        f"{'video':<16} {'tail':>8} {'ch':>3} {'col':>3} {'≠o':>3} "
        f"{'rec':>3} {'dmp':>3}  first_reason"
    )
    n_ch = n_k = n_col = n_diff = n_rec = n_dmp = 0
    hot_keep = []
    for row in rows:
        col = diff = recov = damp = 0
        first_reason = ""
        for i, ch in enumerate(row["chunks"]):
            cands = ch.get("candidates") or []
            n_ch += 1
            if len(cands) > 1:
                n_k += 1
            prefix = ch.get("prefix_motion")
            try:
                prefix = float(prefix) if prefix is not None else None
            except (TypeError, ValueError):
                prefix = None
            pick, reason = _pick(cands, prefix)
            if i == 0:
                first_reason = reason
            orig = _orig_cand(ch)
            c0 = _cand0(cands)
            m0 = _motion(c0) if c0 else float("nan")
            mp = None
            for c in cands:
                if int(c.get("cand", -1)) == pick:
                    mp = _motion(c)
                    break
            if reason in ("band_appear", "nearest_prefix"):
                col += 1
                n_col += 1
            if pick != orig:
                diff += 1
                n_diff += 1
            if mp == mp and m0 == m0 and mp > m0 + 1e-5:
                recov += 1
                n_rec += 1
            if mp == mp and m0 == m0 and mp < m0 - 1e-5:
                damp += 1
                n_dmp += 1
            stem = Path(row["key"]).stem
            if stem in {"panda_0022", "panda_0027", "panda_0028"} and i == 0:
                hot_keep.append(
                    f"  {stem} ch0 prefix={prefix} c0={m0:.5f} "
                    f"orig={orig} resim={pick} ({reason}) m_pick={mp}"
                )
        tail = row["tail"]
        tstr = f"{float(tail):8.5f}" if tail is not None else "     —"
        print(
            f"{row['key']:<16} {tstr} {len(row['chunks']):>3} {col:>3} "
            f"{diff:>3} {recov:>3} {damp:>3}  {first_reason}"
        )
    print(
        f"  chunks={n_ch} searched_k>1={n_k} collapse_fire={n_col} "
        f"resim≠orig={n_diff} recover={n_rec} damp={n_dmp}"
    )
    print(
        "  Policy: still prefix or no collapse → cand0. "
        "Else band[0.85,1.15]×prefix + min appear; else nearest |m-prefix|."
    )
    print("  Not a 30 s tail. Later chunks condition on the original commits.")
    for line in hot_keep:
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method-dir", type=Path, action="append", required=True)
    args = ap.parse_args()
    for d in args.method_dir:
        if not d.is_dir():
            print(f"(missing) {d}")
            continue
        analyze(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
