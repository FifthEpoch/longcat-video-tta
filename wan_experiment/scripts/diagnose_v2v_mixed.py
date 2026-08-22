#!/usr/bin/env python3
"""N=8 mixed-result audit: is a tail gain flicker, is a quality gain freeze?

    python3 -u wan_experiment/scripts/diagnose_v2v_mixed.py \
      --baseline-dir wan_experiment/results/v2v_panda_bakeoff_8v \
      --series-dir wan_experiment/results/v2v_panda_lineage_8v \
      --series-dir wan_experiment/results/v2v_panda_ideas_8v
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


DIMS = (
    "imaging_quality",
    "subject_consistency",
    "dynamic_degree",
    "temporal_flickering",
    "motion_smoothness",
    "aesthetic_quality",
)


def _norm(key: str | None) -> str:
    if not key:
        return ""
    p = Path(str(key))
    return p.name if p.suffix else (p.name + ".mp4")


def _usable(rec: dict) -> bool:
    return bool(
        rec.get("ok")
        and not rec.get("skipped")
        and rec.get("tail_motion") is not None
    )


def _load_tails(method_dir: Path) -> dict[str, dict]:
    out = {}
    if not method_dir.is_dir():
        return out
    for p in sorted(method_dir.glob("*.json")):
        if p.name in {"summary.json", "joined.json"} or "vbench" in p.name:
            continue
        try:
            rec = json.loads(p.read_text())
        except Exception:
            continue
        if not _usable(rec):
            continue
        key = _norm(rec.get("file_name") or rec.get("stem") or p.stem)
        out[key] = rec
    summary = method_dir / "summary.json"
    if summary.is_file():
        try:
            data = json.loads(summary.read_text())
        except Exception:
            data = {}
        for rec in data.get("rows") or []:
            if not _usable(rec):
                continue
            key = _norm(rec.get("file_name") or rec.get("stem") or rec.get("mp4"))
            if key and key not in out:
                out[key] = rec
    return out


def _load_vbench(method_dir: Path) -> dict[str, dict]:
    for name in ("vbench_full/joined.json", "vbench_full/summary.json"):
        p = method_dir / name
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        out = {}
        for rec in data.get("per_video") or []:
            key = _norm(rec.get("file_name") or rec.get("stem") or rec.get("mp4"))
            vb = rec.get("vbench") or {}
            if key:
                out[key] = {d: vb.get(d) for d in DIMS}
        if out:
            return out
    return {}


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx < 1e-12 or dy < 1e-12:
        return None
    return num / (dx * dy)


def _fmt(x, nd=5):
    if x is None:
        return "     —"
    try:
        if x != x:
            return "     —"
    except Exception:
        return "     —"
    return f"{float(x):.{nd}f}"


def _find_method(series_dirs: list[Path], name: str) -> Path | None:
    for s in series_dirs:
        hits = sorted(s.glob(f"{name}_h*s_shard0"))
        if hits:
            return hits[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, required=True)
    ap.add_argument("--series-dir", type=Path, action="append", required=True)
    ap.add_argument(
        "--methods",
        nargs="+",
        default=(
            "longlive_prefix_sink",
            "longlive_notta",
            "rolling_notta",
            "appear_bon",
            "live_hist",
        ),
    )
    args = ap.parse_args()

    notta_dir = next(iter(sorted(args.baseline_dir.glob("notta_h*s_shard0"))), None)
    seed_dir = next(iter(sorted(args.baseline_dir.glob("seed_bon_h*s_shard0"))), None)
    if notta_dir is None:
        raise SystemExit(f"no notta under {args.baseline_dir}")
    notta_t = _load_tails(notta_dir)
    notta_v = _load_vbench(notta_dir)
    seed_t = _load_tails(seed_dir) if seed_dir else {}

    print("Mixed-result audit. Tail↑ with flicker↓ / IQ↓ = junk motion.")
    print("Tail↓ with IQ↑ / Dyn=0 = identity freeze. Cite paired clips.\n")

    for name in args.methods:
        d = _find_method(args.series_dir, name)
        if d is None:
            print(f"## {name}\n(missing)\n")
            continue
        tails = _load_tails(d)
        vbench = _load_vbench(d)
        keys = sorted(k for k in notta_t if k in tails)
        print(f"## {name}  paired n={len(keys)}  dir={d.name}")
        print(
            f"{'video':<16} {'d_tail':>8} {'d_IQ':>7} {'d_flick':>8} "
            f"{'d_subj':>7} {'dyn_n':>5} {'dyn_m':>5}"
        )
        d_tail, d_iq, d_fl, d_subj = [], [], [], []
        tail_win = tail_loss = iq_loss = 0
        for k in keys:
            nt = float(notta_t[k]["tail_motion"])
            mt = float(tails[k]["tail_motion"])
            dt = mt - nt
            nv = notta_v.get(k) or {}
            mv = vbench.get(k) or {}
            diq = None
            dfl = None
            ds = None
            if nv.get("imaging_quality") is not None and mv.get("imaging_quality") is not None:
                diq = float(mv["imaging_quality"]) - float(nv["imaging_quality"])
            if nv.get("temporal_flickering") is not None and mv.get("temporal_flickering") is not None:
                dfl = float(mv["temporal_flickering"]) - float(nv["temporal_flickering"])
            if nv.get("subject_consistency") is not None and mv.get("subject_consistency") is not None:
                ds = float(mv["subject_consistency"]) - float(nv["subject_consistency"])
            print(
                f"{k:<16} {_fmt(dt)} {_fmt(diq, 3)} {_fmt(dfl, 4)} "
                f"{_fmt(ds, 4)} {_fmt(nv.get('dynamic_degree'), 2)} "
                f"{_fmt(mv.get('dynamic_degree'), 2)}"
            )
            if dt > 1e-5:
                tail_win += 1
            elif dt < -1e-5:
                tail_loss += 1
            if diq is not None and diq < -1.0:
                iq_loss += 1
            d_tail.append(dt)
            if diq is not None:
                d_iq.append(diq)
            if dfl is not None:
                d_fl.append(dfl)
            if ds is not None:
                d_subj.append(ds)
        same_seed = 0
        if seed_t:
            for k in keys:
                if k in seed_t and abs(
                    float(tails[k]["tail_motion"]) - float(seed_t[k]["tail_motion"])
                ) < 1e-5:
                    same_seed += 1
        r_fl = _pearson(d_tail, d_fl) if len(d_fl) == len(d_tail) else None
        r_iq = _pearson(d_tail, d_iq) if len(d_iq) == len(d_tail) else None
        print(
            f"  tail win/loss/tie={tail_win}/{tail_loss}/{len(keys)-tail_win-tail_loss}  "
            f"median d_tail={_fmt(statistics.median(d_tail) if d_tail else None)}  "
            f"IQ drops>1.0: {iq_loss}/{len(keys)}"
        )
        print(
            f"  pearson(d_tail, d_flicker)={_fmt(r_fl, 3)}  "
            f"pearson(d_tail, d_IQ)={_fmt(r_iq, 3)}  "
            f"bitmatch seed_bon={same_seed}/{len(keys)}"
        )
        if r_fl is not None and r_fl < -0.4:
            print("  READ: tail gain tracks flicker drop → junk motion, do not fix IQ.")
        elif r_iq is not None and r_iq < -0.4:
            print("  READ: tail gain trades IQ → only fix if d_flicker is flat.")
        elif tail_win == 0 and d_iq and statistics.median(d_iq) > 0.5:
            print("  READ: quality up, no tail wins → freeze/identity, not a Dyn fix.")
        elif tail_win > tail_loss and iq_loss == 0:
            print("  READ: clean N=8 leftover. Scaling is the test, not more N=8.")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
