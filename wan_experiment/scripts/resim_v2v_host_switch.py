#!/usr/bin/env python3
"""H2/H3 offline: does first-chunk motion pick the better 30 s host?

No GPU. Both SF notta and RF rolling_notta 30 s videos already exist.
Score chunk-0 motion from the mp4 (skip prefix_pix, next 81 frames =
21 latents). Counterfactual tail = the already-generated video whose
chunk-0 motion won.

  bake = argmax chunk-0 motion (H2)
  veto = RF unless RF chunk-0 < 0.8 × SF chunk-0 (H3)

This is not the cheap online method (generate both chunk 0, continue
with the winner). It answers whether chunk-0 rank predicts the 30 s
tail. Prefix-motion gate already lost (+9% vs always-RF +31%).

    python3 -u wan_experiment/scripts/resim_v2v_host_switch.py
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


TRUST = 0.8
CHUNK_LATENTS = 21
ROOT = Path("wan_experiment/results")

PAIRS = (
    (
        "n8",
        ROOT / "v2v_panda_bakeoff_8v/notta_h30s_shard0",
        ROOT / "v2v_panda_lineage_8v/rolling_notta_h30s_shard0",
    ),
    (
        "n32",
        ROOT / "v2v_panda_confirm_32v/notta_h30s_shard0",
        ROOT / "v2v_panda_forward_32v/rolling_notta_h30s_shard0",
    ),
    (
        "n128",
        ROOT / "v2v_panda_rolling_128v/notta_h30s_shard0",
        ROOT / "v2v_panda_rolling_128v/rolling_notta_h30s_shard0",
    ),
)


def _usable(rec: dict) -> bool:
    return bool(
        rec.get("ok")
        and not rec.get("skipped")
        and rec.get("tail_motion") is not None
    )


def _key(rec: dict, fallback: str = "") -> str:
    k = rec.get("file_name") or rec.get("stem") or fallback
    return Path(str(k)).name if k else fallback


def _load(d: Path) -> dict[str, dict]:
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
        if not _usable(rec):
            continue
        rec = dict(rec)
        rec["_sidecar"] = str(p)
        out[_key(rec, p.stem)] = rec
    return out


def _prefix_pix(rec: dict) -> int:
    v = rec.get("prefix_pix")
    if v is not None:
        return int(v)
    return 1 + 4 * (9 - 1)


def _chunk_pix(n_lat: int = CHUNK_LATENTS) -> int:
    return 1 + 4 * (int(n_lat) - 1)


def _read_span(mp4: Path, start: int, count: int):
    frames = []
    try:
        import imageio.v2 as imageio

        r = imageio.get_reader(str(mp4))
        try:
            for i, im in enumerate(r):
                if i < start:
                    continue
                if i >= start + count:
                    break
                frames.append(np_rgb(im))
        finally:
            try:
                r.close()
            except Exception:
                pass
        if frames:
            return frames
    except Exception:
        pass
    try:
        import cv2

        cap = cv2.VideoCapture(str(mp4))
        if not cap.isOpened():
            return []
        idx = 0
        while len(frames) < count:
            ok, im = cap.read()
            if not ok:
                break
            if idx >= start:
                frames.append(np_rgb(im[:, :, ::-1] if im.ndim == 3 else im))
            idx += 1
        cap.release()
    except Exception:
        return []
    return frames


def np_rgb(im):
    import numpy as np

    arr = np.asarray(im)[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr.astype(np.float32) * 255.0), 0, 255).astype(
            np.uint8
        )
    return arr.astype(np.float32) / 255.0


def _motion(frames) -> float:
    import numpy as np

    if len(frames) < 2:
        return float("nan")
    diffs = [float(np.mean(np.abs(frames[i] - frames[i - 1]))) for i in range(1, len(frames))]
    return float(sum(diffs) / len(diffs))


def _spearman(xs, ys) -> float:
    n = len(xs)
    if n < 3:
        return float("nan")
    rx = _ranks(xs)
    ry = _ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _summary(vals):
    vals = [float(v) for v in vals if v == v]
    if not vals:
        return float("nan"), float("nan")
    return statistics.median(vals), sum(vals) / len(vals)


def _pct(new, old) -> str:
    if old is None or old != old or old == 0 or new != new:
        return "   n/a"
    return f"{100.0 * (new - old) / old:+6.1f}%"


def _wl(picked, sf, rf):
    w = l = t = 0
    for p, a, b in zip(picked, sf, rf):
        if p != p or a != a or b != b:
            continue
        best = max(a, b)
        worst = min(a, b)
        if abs(p - best) < 1e-12 and abs(p - worst) < 1e-12:
            t += 1
        elif abs(p - best) < 1e-12:
            w += 1
        else:
            l += 1
    return w, l, t


def run_pair(label: str, sf_dir: Path, rf_dir: Path, trust: float) -> dict | None:
    sf_map = _load(sf_dir)
    rf_map = _load(rf_dir)
    keys = sorted(set(sf_map) & set(rf_map))
    print(f"\n=== {label}  n={len(keys)}  {sf_dir}  vs  {rf_dir}")
    if not keys:
        print("  no paired sidecars")
        return None

    rows = []
    chunk_pix = _chunk_pix()
    for key in keys:
        sf = sf_map[key]
        rf = rf_map[key]
        p = _prefix_pix(sf)
        sf_mp4 = Path(sf.get("mp4") or "")
        rf_mp4 = Path(rf.get("mp4") or "")
        if not sf_mp4.is_file():
            cand = next(sf_dir.glob(f"*{Path(key).stem}*.mp4"), None)
            sf_mp4 = cand if cand else sf_mp4
        if not rf_mp4.is_file():
            cand = next(rf_dir.glob(f"*{Path(key).stem}*.mp4"), None)
            rf_mp4 = cand if cand else rf_mp4
        sf_c0 = _motion(_read_span(sf_mp4, p, chunk_pix)) if sf_mp4.is_file() else float("nan")
        rf_c0 = _motion(_read_span(rf_mp4, p, chunk_pix)) if rf_mp4.is_file() else float("nan")
        sf_tail = float(sf["tail_motion"])
        rf_tail = float(rf["tail_motion"])
        bake = "sf" if sf_c0 > rf_c0 else "rf"
        veto = "sf" if (
            sf_c0 == sf_c0 and rf_c0 == rf_c0 and rf_c0 < trust * sf_c0
        ) else "rf"
        bake_tail = sf_tail if bake == "sf" else rf_tail
        veto_tail = sf_tail if veto == "sf" else rf_tail
        rows.append({
            "key": key,
            "sf_c0": sf_c0,
            "rf_c0": rf_c0,
            "sf_tail": sf_tail,
            "rf_tail": rf_tail,
            "bake": bake,
            "veto": veto,
            "bake_tail": bake_tail,
            "veto_tail": veto_tail,
        })

    sf_t = [r["sf_tail"] for r in rows]
    rf_t = [r["rf_tail"] for r in rows]
    bake_t = [r["bake_tail"] for r in rows]
    veto_t = [r["veto_tail"] for r in rows]
    sf_med, sf_mean = _summary(sf_t)
    rf_med, rf_mean = _summary(rf_t)
    bake_med, bake_mean = _summary(bake_t)
    veto_med, veto_mean = _summary(veto_t)
    c0_d = [r["rf_c0"] - r["sf_c0"] for r in rows if r["rf_c0"] == r["rf_c0"] and r["sf_c0"] == r["sf_c0"]]
    t_d = [r["rf_tail"] - r["sf_tail"] for r in rows if r["rf_c0"] == r["rf_c0"] and r["sf_c0"] == r["sf_c0"]]
    rho = _spearman(c0_d, t_d) if len(c0_d) == len(t_d) else float("nan")
    n_bake_sf = sum(1 for r in rows if r["bake"] == "sf")
    n_veto_sf = sum(1 for r in rows if r["veto"] == "sf")
    n_c0 = sum(1 for r in rows if r["sf_c0"] == r["sf_c0"] and r["rf_c0"] == r["rf_c0"])

    print(f"  chunk0 readable {n_c0}/{len(rows)}")
    print(f"  {'arm':<12} {'med':>8} {'mean':>8}  vs SF          vs RF          W/L/T vs best")
    for name, med, mean, tails in (
        ("sf_notta", sf_med, sf_mean, sf_t),
        ("rf_rolling", rf_med, rf_mean, rf_t),
        ("h2_bake", bake_med, bake_mean, bake_t),
        ("h3_veto", veto_med, veto_mean, veto_t),
    ):
        w, l, t = _wl(tails, sf_t, rf_t)
        print(
            f"  {name:<12} {med:8.5f} {mean:8.5f}  "
            f"{_pct(med, sf_med)} / {_pct(mean, sf_mean)}  "
            f"{_pct(med, rf_med)} / {_pct(mean, rf_mean)}  "
            f"{w}/{l}/{t}"
        )
    print(
        f"  bake picks SF on {n_bake_sf}/{len(rows)}; "
        f"veto switches to SF on {n_veto_sf}/{len(rows)}"
    )
    print(f"  Spearman(Δc0, Δtail) = {rho:.3f}  (does chunk-0 rank predict 30 s?)")
    print("  per-video: key  sf_c0  rf_c0  sf_tail  rf_tail  bake  veto")
    for r in rows:
        print(
            f"    {r['key']:<16} {r['sf_c0']:8.5f} {r['rf_c0']:8.5f} "
            f"{r['sf_tail']:8.5f} {r['rf_tail']:8.5f}  "
            f"{r['bake']:>2} {r['veto']:>2}"
        )
    return {
        "label": label,
        "n": len(rows),
        "n_c0": n_c0,
        "sf_med": sf_med,
        "rf_med": rf_med,
        "bake_med": bake_med,
        "veto_med": veto_med,
        "rho": rho,
        "bake_sf": n_bake_sf,
        "veto_sf": n_veto_sf,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trust", type=float, default=TRUST)
    ap.add_argument("--only", choices=("n8", "n32", "n128", "all"), default="all")
    args = ap.parse_args()
    print(
        "H2 bake = argmax chunk-0 motion; "
        f"H3 veto = RF unless RF c0 < {args.trust:g}× SF c0. "
        "Tails from existing sidecars. No GPU."
    )
    for label, sf_dir, rf_dir in PAIRS:
        if args.only != "all" and label != args.only:
            continue
        run_pair(label, sf_dir, rf_dir, args.trust)
    print(
        "\nCall: H2/H3 YES only if bake or veto beats always-RF on "
        "median tail without needing a new generate. "
        "If no, chunk-0 does not pick the 30 s winner — do not GPU a router."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
