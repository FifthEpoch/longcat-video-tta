#!/usr/bin/env python3
"""Family A offline: do RF losses die mid-rollout, or were they quieter from chunk 0?

No GPU. Reads existing SF notta + RF rolling mp4s. Same 81-frame windows as
H2 (skip prefix_pix, then 6 × 21-latent chunks).

Rewind only has a job if RF *losses vs SF tail* are enriched for a late
drop (chunk 0 healthy, last chunks die) and wins are not. If losses are
quieter from chunk 0, salvage is the wrong story — SF may have been
hallucinating motion.

    python3 -u wan_experiment/scripts/resim_v2v_rf_chunk_trace.py --only n128

Cite medians. Paper baseline stays SF notta. This script does not promote
a method; it only kill/go Family A.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from resim_v2v_host_switch import (  # noqa: E402
    PAIRS,
    _chunk_pix,
    _load,
    _motion,
    _prefix_pix,
    _spearman,
    _summary,
    np_rgb,
)

N_CHUNKS = 6
DROP = 0.8
NAMED = ("panda_0004.mp4", "panda_0027.mp4", "panda_0035.mp4",
         "panda_0044.mp4", "panda_0087.mp4")


def _read_all(mp4: Path):
    frames = []
    try:
        import imageio.v2 as imageio

        r = imageio.get_reader(str(mp4))
        try:
            for im in r:
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
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(np_rgb(im[:, :, ::-1] if im.ndim == 3 else im))
        cap.release()
    except Exception:
        return []
    return frames


def _find_mp4(rec: dict, d: Path, key: str) -> Path:
    p = Path(rec.get("mp4") or "")
    if p.is_file():
        return p
    cand = next(d.glob(f"*{Path(key).stem}*.mp4"), None)
    return cand if cand else p


def _chunk_mots(frames, prefix_pix: int, chunk_pix: int) -> list[float]:
    out = []
    for i in range(N_CHUNKS):
        a = prefix_pix + i * chunk_pix
        b = a + chunk_pix
        out.append(_motion(frames[a:b]) if frames and b <= len(frames) else float("nan"))
    return out


def _late_drop(mots: list[float], drop: float) -> bool:
    c0, last = mots[0], mots[-1]
    if c0 != c0 or last != last or c0 <= 0:
        return False
    return last < drop * c0


def _fmt(xs) -> str:
    med, mean = _summary(xs)
    if med != med:
        return "   n/a"
    return f"{med:.5f} / {mean:.5f}"


def run_pair(label: str, sf_dir: Path, rf_dir: Path, drop: float) -> None:
    sf_map = _load(sf_dir)
    rf_map = _load(rf_dir)
    keys = sorted(set(sf_map) & set(rf_map))
    print(f"\n=== {label}  n={len(keys)}  {sf_dir.name}  vs  {rf_dir.name}")
    if not keys:
        print("  no paired sidecars")
        return

    chunk_pix = _chunk_pix()
    rows = []
    for key in keys:
        sf, rf = sf_map[key], rf_map[key]
        p = _prefix_pix(sf)
        sf_fr = _read_all(_find_mp4(sf, sf_dir, key))
        rf_fr = _read_all(_find_mp4(rf, rf_dir, key))
        sf_c = _chunk_mots(sf_fr, p, chunk_pix)
        rf_c = _chunk_mots(rf_fr, p, chunk_pix)
        sf_tail = float(sf["tail_motion"])
        rf_tail = float(rf["tail_motion"])
        win = rf_tail > sf_tail
        rows.append({
            "key": key,
            "sf_c": sf_c,
            "rf_c": rf_c,
            "sf_tail": sf_tail,
            "rf_tail": rf_tail,
            "win": win,
            "rf_drop": _late_drop(rf_c, drop),
            "sf_drop": _late_drop(sf_c, drop),
        })

    wins = [r for r in rows if r["win"]]
    losses = [r for r in rows if not r["win"]]
    print(f"  RF tail > SF: {len(wins)}/{len(rows)}   RF tail ≤ SF: {len(losses)}/{len(rows)}")
    print(f"  late drop = last chunk < {drop:g} × chunk 0")
    print(f"  {'split':<10} {'n':>4}  {'RF drop':>8}  {'SF drop':>8}  "
          f"{'RF c0 med/mean':>18}  {'RF c5 med/mean':>18}  {'tail med/mean':>18}")
    for name, sub in ("all", rows), ("wins", wins), ("losses", losses):
        if not sub:
            continue
        n_rd = sum(1 for r in sub if r["rf_drop"])
        n_sd = sum(1 for r in sub if r["sf_drop"])
        c0 = [r["rf_c"][0] for r in sub if r["rf_c"][0] == r["rf_c"][0]]
        c5 = [r["rf_c"][-1] for r in sub if r["rf_c"][-1] == r["rf_c"][-1]]
        print(
            f"  {name:<10} {len(sub):>4}  {n_rd:>3}/{len(sub):<3}  {n_sd:>3}/{len(sub):<3}  "
            f"{_fmt(c0):>18}  {_fmt(c5):>18}  {_fmt([r['rf_tail'] for r in sub]):>18}"
        )

    print("  RF median motion by chunk")
    print(f"  {'split':<10}", end="")
    for i in range(N_CHUNKS):
        print(f"  c{i:>1}", end="")
    print()
    for name, sub in ("wins", wins), ("losses", losses):
        print(f"  {name:<10}", end="")
        for i in range(N_CHUNKS):
            vals = [r["rf_c"][i] for r in sub if r["rf_c"][i] == r["rf_c"][i]]
            med, _ = _summary(vals)
            print(f"  {med:6.4f}" if med == med else "     n/a", end="")
        print()

    rho_c0 = _spearman(
        [r["rf_c"][0] for r in rows if r["rf_c"][0] == r["rf_c"][0]],
        [r["rf_tail"] for r in rows if r["rf_c"][0] == r["rf_c"][0]],
    )
    print(f"  Spearman(RF c0, RF tail) = {rho_c0:.3f}")

    print("  named clips (key  win  rf_drop  sf_c0  rf_c0  rf_c5  sf_tail  rf_tail)")
    shown = 0
    for r in rows:
        stem = Path(r["key"]).name
        if stem not in NAMED and Path(r["key"]).stem + ".mp4" not in NAMED:
            continue
        shown += 1
        print(
            f"    {r['key']:<16}  {'W' if r['win'] else 'L'}  "
            f"{'drop' if r['rf_drop'] else 'flat':<4}  "
            f"{r['sf_c'][0]:7.5f}  {r['rf_c'][0]:7.5f}  {r['rf_c'][-1]:7.5f}  "
            f"{r['sf_tail']:7.5f}  {r['rf_tail']:7.5f}"
        )
    if shown == 0:
        print("    (none of the named clips in this split)")

    n_loss_drop = sum(1 for r in losses if r["rf_drop"])
    n_win_drop = sum(1 for r in wins if r["rf_drop"])
    print("\n  Call for Family A (rewind):")
    if not losses:
        print("    no RF losses — nothing to salvage.")
        return
    loss_rate = n_loss_drop / len(losses)
    win_rate = n_win_drop / len(wins) if wins else 0.0
    print(
        f"    late-drop rate  losses {n_loss_drop}/{len(losses)} ({loss_rate:.0%})  "
        f"wins {n_win_drop}/{len(wins)} ({win_rate:.0%})"
    )
    if loss_rate >= 0.5 and loss_rate >= win_rate + 0.2:
        print(
            "    GO — losses are enriched for a late drop. "
            "Rewind N=32 vs rolling_notta is allowed."
        )
    elif loss_rate <= 0.25:
        print(
            "    NO GPU — losses are not a late collapse. "
            "They were quieter from chunk 0 (or never dropped). "
            "Salvage/rewind is the wrong story."
        )
    else:
        print(
            "    HOLD — mixed. Read the per-chunk medians and named clips "
            "before a GPU. Do not retune DROP after seeing this."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("n8", "n32", "n128", "all"), default="n128")
    ap.add_argument("--drop", type=float, default=DROP)
    args = ap.parse_args()
    print(
        "Family A offline: RF per-chunk motion on existing mp4s. "
        f"late drop = c5 < {args.drop:g}×c0. No GPU. "
        "Paper baseline stays SF notta; this only asks whether rewind "
        "has a signal on RF losses."
    )
    for label, sf_dir, rf_dir in PAIRS:
        if args.only != "all" and label != args.only:
            continue
        run_pair(label, sf_dir, rf_dir, args.drop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
