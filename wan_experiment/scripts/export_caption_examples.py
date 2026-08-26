#!/usr/bin/env python3
"""Copy matched caption clips + print per-video wall time.

Login CPU. Does not delete anything.

    python3 -u wan_experiment/scripts/export_caption_examples.py
"""
from __future__ import annotations

import json
import shutil
import statistics
from pathlib import Path

ROOT = Path("/scratch/wc3013/longcat-video-tta/wan_experiment/results")
CAP = ROOT / "v2v_panda_caption_32v"
PREFIX = ROOT / "v2v_panda_caption_prefix_32v"
CROSS = ROOT / "v2v_panda_caption_cross_32v"
DEST = ROOT / "v2v_panda_caption_examples"

SHOW = ("notta", "sf_always_search", "sf_pseudo")
SHOW_LABEL = {
    "notta": "self_forcing",
    "sf_always_search": "always_search",
    "sf_pseudo": "pseudo",
}
TIME_JOBS = (
    (CAP, (
        "notta", "rolling_notta",
        "sf_rewind", "sf_sick_search", "sf_pseudo", "sf_sink", "sf_always_search",
        "rf_always_search", "rf_rewind", "rf_sick_search", "rf_pseudo", "rf_sink",
    )),
    (PREFIX, ("seed_bon", "live_bon", "appear_bon")),
    (CROSS, ("sf_roll", "rf_chunk")),
)


def _dir(series: Path, method: str) -> Path:
    hits = sorted(series.glob(f"{method}_h*s_shard*"))
    return hits[0] if hits else series / f"{method}_h30s_shard0"


def _rows(d: Path) -> dict[str, dict]:
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        if p.name in {"summary.json", "joined.json"} or "vbench" in p.name:
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        key = str(rec.get("file_name") or rec.get("stem") or p.stem)
        out[key] = rec
    return out


def _dyn_by_key(d: Path) -> dict[str, float]:
    p = d / "vbench_full" / "joined.json"
    if not p.is_file():
        return {}
    data = json.loads(p.read_text())
    out = {}
    for rec in data.get("per_video") or []:
        vb = rec.get("vbench") or {}
        if "dynamic_degree" not in vb:
            continue
        key = str(rec.get("file_name") or rec.get("stem") or "")
        if key:
            out[key] = float(vb["dynamic_degree"])
    return out


def _pick_keys(loaded: dict[str, dict[str, dict]]) -> list[str]:
    keys = sorted(set.intersection(*(set(loaded[m]) for m in SHOW)))
    chosen = []
    for pref in ("panda_0000.mp4", "panda_0001.mp4", "panda_0002.mp4", "panda_0003.mp4"):
        if pref in keys:
            chosen.append(pref)
    dyn = {m: _dyn_by_key(_dir(CAP, m)) for m in SHOW}
    tails = {m: {k: float(loaded[m][k]["tail_motion"]) for k in keys
                 if loaded[m][k].get("tail_motion") is not None} for m in SHOW}
    extras = []
    for k in keys:
        if k in chosen:
            continue
        if dyn["notta"].get(k, 0) < 0.5 and (
            dyn["sf_always_search"].get(k, 0) >= 0.5
            or dyn["sf_pseudo"].get(k, 0) >= 0.5
        ):
            extras.append(("dyn_wake", k))
        elif tails["notta"].get(k) and tails["sf_always_search"].get(k, 0) > 1.3 * tails["notta"][k]:
            extras.append(("tail_up", k))
    for _, k in extras:
        if k not in chosen:
            chosen.append(k)
        if len(chosen) >= 6:
            break
    for k in keys:
        if k not in chosen:
            chosen.append(k)
        if len(chosen) >= 6:
            break
    return chosen


def main() -> None:
    print("== per-video wall seconds (sidecar `seconds`) ==")
    all_rows = {}
    methods = []
    for series, names in TIME_JOBS:
        for m in names:
            methods.append(m)
            all_rows[m] = _rows(_dir(series, m))
    keys = sorted(all_rows.get("notta") or {})
    hdr = ["video"] + methods
    print("\t".join(hdr))
    for k in keys:
        cells = [k]
        for m in methods:
            rec = all_rows[m].get(k) or {}
            sec = rec.get("seconds")
            cells.append("" if sec is None else f"{float(sec):.1f}")
        print("\t".join(cells))
    print()
    print("== method wall seconds (median / mean / n) ==")
    print(f"{'method':22} {'n':>3} {'median_s':>9} {'mean_s':>9}")
    for m in methods:
        xs = [float(r["seconds"]) for r in all_rows[m].values() if r.get("seconds") is not None]
        if not xs:
            print(f"{m:22}   0         —         —")
            continue
        print(f"{m:22} {len(xs):3} {statistics.median(xs):9.1f} {statistics.fmean(xs):9.1f}")

    DEST.mkdir(parents=True, exist_ok=True)
    show_loaded = {m: all_rows[m] for m in SHOW}
    picks = _pick_keys(show_loaded)
    print()
    print(f"== copy matched clips → {DEST} ==")
    for k in picks:
        stem = Path(k).stem
        for m in SHOW:
            src = Path((show_loaded[m][k].get("mp4") or "") or (_dir(CAP, m) / f"{stem}.mp4"))
            if not src.is_file():
                src = _dir(CAP, m) / f"{stem}.mp4"
            dest = DEST / f"{stem}__{SHOW_LABEL[m]}.mp4"
            if src.is_file():
                shutil.copy2(src, dest)
                print(f"  {dest.name}  {src.stat().st_size}B")
            else:
                print(f"  MISSING {src}")
    print("scp from login:")
    print(
        "  scp 'wc3013@torch-login-a-2:"
        f"{DEST}/*.mp4' ~/Desktop/caption_examples/"
    )


if __name__ == "__main__":
    main()
