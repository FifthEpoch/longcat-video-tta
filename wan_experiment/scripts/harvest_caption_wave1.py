#!/usr/bin/env python3
"""Login-CPU harvest: caption WAVE=1 tails + optional VBench, AdaSteer 8v.

    python3 -u wan_experiment/scripts/harvest_caption_wave1.py

Cite vs caption SF notta. Do not mix stem-prompt dirs.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path("/scratch/wc3013/longcat-video-tta/wan_experiment/results")
CAP = ROOT / "v2v_panda_caption_32v"
ADA = ROOT / "v2v_panda_adasteer_8v"
PREFIX = ROOT / "v2v_panda_caption_prefix_32v"
CROSS = ROOT / "v2v_panda_caption_cross_32v"

CAP_METHODS = (
    "notta",
    "rolling_notta",
    "sf_rewind",
    "sf_sick_search",
    "sf_pseudo",
    "sf_sink",
    "sf_always_search",
    "rf_always_search",
    "rf_rewind",
    "rf_sick_search",
    "rf_pseudo",
    "rf_sink",
)
ADA_METHODS = ("ada_fixed", "ada_stream", "ada_resid")
PREFIX_METHODS = ("seed_bon", "live_bon", "appear_bon")
CROSS_METHODS = ("sf_roll", "rf_chunk")
VB_DIMS = (
    "subject_consistency",
    "imaging_quality",
    "dynamic_degree",
    "temporal_flickering",
)


def _usable(rec: dict) -> bool:
    return bool(
        rec.get("ok")
        and not rec.get("skipped")
        and rec.get("tail_motion") is not None
    )


def _dir(series: Path, method: str) -> Path:
    hits = sorted(series.glob(f"{method}_h*s_shard*"))
    return hits[0] if hits else series / f"{method}_h30s_shard0"


def _rows(d: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        if p.name in {"summary.json", "joined.json"} or "vbench" in p.name:
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if not _usable(rec):
            continue
        key = str(rec.get("file_name") or rec.get("stem") or p.stem)
        out[key] = rec
    return out


def _median(xs):
    xs = [float(x) for x in xs if x is not None and x == x]
    return statistics.median(xs) if xs else None


def _wl(keys, method_rows, base_rows):
    w = l = t = 0
    for k in keys:
        a = method_rows[k].get("tail_motion")
        b = base_rows[k].get("tail_motion")
        if a is None or b is None:
            continue
        if abs(float(a) - float(b)) < 1e-8:
            t += 1
        elif float(a) > float(b):
            w += 1
        else:
            l += 1
    return w, l, t


def _sources(rows: dict[str, dict]) -> list[str]:
    return sorted({str(r.get("prompt_source") or "?") for r in rows.values()})


def _vbench(d: Path) -> dict | None:
    for name in ("vbench_full/joined.json", "vbench_full/summary.json"):
        p = d / name
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        pop = data.get("population") or {}
        out = {}
        for dim in VB_DIMS:
            cell = pop.get(dim)
            if isinstance(cell, dict) and cell.get("median") is not None:
                out[dim] = float(cell["median"])
        if out:
            n = None
            for dim in VB_DIMS:
                cell = pop.get(dim)
                if isinstance(cell, dict):
                    n = cell.get("n") or cell.get("count") or cell.get("num_videos")
                    if n is not None:
                        break
            out["_n"] = n
            return out
    return None


def _fmt(x, nd=5):
    return "—" if x is None else f"{x:.{nd}f}"


def _pct(cur, base):
    if cur is None or base is None or base == 0:
        return "—"
    return f"{100.0 * (cur / base - 1.0):+.0f}%"


def _print_vbench(label: str, d: Path, vb: dict | None) -> None:
    if not vb:
        print(f"  {label:18} VBench —")
        return
    n = vb.get("_n")
    n_s = "" if n is None else f" n={n}"
    print(
        f"  {label:18} VBench{n_s} subj={_fmt(vb.get('subject_consistency'), 3)} "
        f"IQ={_fmt(vb.get('imaging_quality'), 2)} "
        f"Dyn={_fmt(vb.get('dynamic_degree'), 2)} "
        f"flick={_fmt(vb.get('temporal_flickering'), 3)}"
    )


def _ada_delta(rec: dict) -> str:
    chunks = rec.get("chunks") or rec.get("chunk_logs") or []
    if not chunks:
        return "no-chunks"
    ada = chunks[0].get("adasteer") or {}
    fits = ada.get("fits") or []
    if not fits:
        return "no-fits"
    norms = []
    for fit in fits:
        n = fit.get("delta_norm_after_blend", fit.get("delta_norm"))
        if n is not None:
            norms.append(f"{float(n):.4g}")
    return "|δ|=" + ",".join(norms) if norms else "fits-no-norm"


def harvest_series(series: Path, methods: tuple[str, ...], base_name: str, first_n: int | None = None) -> None:
    print(f"\n== {series.name} ==")
    print(f"path {series} exists={series.is_dir()}")
    loaded = {m: _rows(_dir(series, m)) for m in methods}
    base = loaded.get(base_name) or {}
    keys_all = sorted(base)
    if first_n:
        keys_all = keys_all[:first_n]
    print(f"baseline {base_name} n={len(base)} first_n={first_n or 'all'}")
    print(
        f"{'method':18} {'n':>3} {'mp4':>3} {'src':16} "
        f"{'tail':>8} {'vsBase':>7} {'W/L/tie':>9}"
    )
    for m in methods:
        d = _dir(series, m)
        rows = loaded[m]
        keys = [k for k in keys_all if k in rows] if keys_all else sorted(rows)
        if first_n:
            keys = keys[:first_n]
        tails = [float(rows[k]["tail_motion"]) for k in keys]
        med = _median(tails)
        bmed = _median([float(base[k]["tail_motion"]) for k in keys if k in base]) if base else None
        wlt = _wl(keys, rows, base) if base and m != base_name else (0, 0, len(keys))
        mp4 = len(list(d.glob("*.mp4"))) if d.is_dir() else 0
        src = ",".join(_sources(rows)) or "?"
        vs = "—" if m == base_name else _pct(med, bmed)
        w, l, t = wlt
        print(
            f"{m:18} {len(rows):3d} {mp4:3d} {src:16} "
            f"{_fmt(med)} {vs:>7} {w}/{l}/{t}"
        )
        _print_vbench(m, d, _vbench(d))
        if m.startswith("ada_") and rows:
            sample = next(iter(rows.values()))
            print(f"  {m:18} first sidecar {_ada_delta(sample)} "
                  f"prompt_source={sample.get('prompt_source')}")
            err = sample.get("error")
            if err:
                print(f"  {m:18} error {str(err)[:240]}")


def peek_errors(series: Path, methods: tuple[str, ...]) -> None:
    print(f"\n== errors / empty {series.name} ==")
    for m in methods:
        d = _dir(series, m)
        if not d.is_dir():
            print(f"{m:18} MISSING {d}")
            continue
        js = [p for p in d.glob("*.json") if p.name not in {"summary.json", "joined.json"}]
        mp4 = list(d.glob("*.mp4"))
        print(f"{m:18} json={len(js)} mp4={len(mp4)}")
        for p in js[:3]:
            try:
                rec = json.loads(p.read_text())
            except Exception as e:
                print(f"  {p.name} unreadable {e}")
                continue
            err = rec.get("error") or rec.get("traceback")
            if err or not rec.get("ok"):
                print(f"  {p.name} ok={rec.get('ok')} {str(err)[:300]}")


def main() -> int:
    harvest_series(CAP, CAP_METHODS, "notta")
    rf = _rows(_dir(CAP, "rolling_notta"))
    print("\n== vs caption RF host (rolling_notta) ==")
    for m in (
        "rf_always_search",
        "rf_rewind",
        "rf_sick_search",
        "rf_pseudo",
        "rf_sink",
    ):
        rows = _rows(_dir(CAP, m))
        keys = sorted(set(rows) & set(rf))
        med = _median([float(rows[k]["tail_motion"]) for k in keys])
        bmed = _median([float(rf[k]["tail_motion"]) for k in keys])
        w, l, t = _wl(keys, rows, rf)
        print(f"{m:18} n={len(keys)} tail={_fmt(med)} vsRF={_pct(med, bmed)} {w}/{l}/{t}")

    print("\n== AdaSteer vs caption notta first 8 ==")
    harvest_series(ADA, ADA_METHODS, "notta", first_n=None)
    notta8 = _rows(_dir(CAP, "notta"))
    keys8 = sorted(notta8)[:8]
    print(f"caption notta first8 n={len(keys8)} "
          f"tail={_fmt(_median([float(notta8[k]['tail_motion']) for k in keys8]))}")
    for m in ADA_METHODS:
        rows = _rows(_dir(ADA, m))
        keys = [k for k in keys8 if k in rows]
        med = _median([float(rows[k]["tail_motion"]) for k in keys])
        bmed = _median([float(notta8[k]["tail_motion"]) for k in keys])
        w, l, t = _wl(keys, rows, notta8)
        print(f"{m:18} paired={len(keys)} tail={_fmt(med)} vsSF8={_pct(med, bmed)} {w}/{l}/{t}")
    peek_errors(ADA, ADA_METHODS)

    notta = _rows(_dir(CAP, "notta"))
    print("\n== Prefix-match vs caption notta ==")
    print(f"path {PREFIX} exists={PREFIX.is_dir()}")
    for m in PREFIX_METHODS:
        d = _dir(PREFIX, m)
        rows = _rows(d)
        keys = sorted(set(rows) & set(notta))
        med = _median([float(rows[k]["tail_motion"]) for k in keys])
        bmed = _median([float(notta[k]["tail_motion"]) for k in keys])
        w, l, t = _wl(keys, rows, notta)
        mp4 = len(list(d.glob("*.mp4"))) if d.is_dir() else 0
        src = ",".join(_sources(rows)) or "?"
        print(
            f"{m:18} n={len(rows):2d} mp4={mp4:2d} {src:16} "
            f"tail={_fmt(med)} vsSF={_pct(med, bmed)} {w}/{l}/{t}"
        )
        _print_vbench(m, d, _vbench(d))
        if rows:
            sample = next(iter(rows.values()))
            print(f"  {m:18} prompt_source={sample.get('prompt_source')} "
                  f"prompt={(sample.get('prompt') or '')[:80]}")

    print("\n== Crossed host vs caption notta ==")
    print(f"path {CROSS} exists={CROSS.is_dir()}")
    for m in CROSS_METHODS:
        d = _dir(CROSS, m)
        rows = _rows(d)
        keys = sorted(set(rows) & set(notta))
        med = _median([float(rows[k]["tail_motion"]) for k in keys])
        bmed = _median([float(notta[k]["tail_motion"]) for k in keys])
        w, l, t = _wl(keys, rows, notta)
        mp4 = len(list(d.glob("*.mp4"))) if d.is_dir() else 0
        src = ",".join(_sources(rows)) or "?"
        print(
            f"{m:18} n={len(rows):2d} mp4={mp4:2d} {src:16} "
            f"tail={_fmt(med)} vsSF={_pct(med, bmed)} {w}/{l}/{t}"
        )
        _print_vbench(m, d, _vbench(d))
    peek_errors(PREFIX, PREFIX_METHODS)
    peek_errors(CROSS, CROSS_METHODS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
