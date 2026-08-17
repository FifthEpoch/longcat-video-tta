#!/usr/bin/env python3
"""Offline GT-free drift on already-written Wan I2V mp4s (CPU, login-node OK).

Same four signals as the LongCat verifier: sharpness (Laplacian var),
colorfulness (Hasler-Süsstrunk), contrast (luma std), temporal_motion
(|Δframe|). Compares the first 1 s window to the last 1 s window.

    /scratch/wc3013/conda-envs/self_forcing/bin/python \
        wan_experiment/scripts/score_i2v_drift.py \
        --dir wan_experiment/results/i2v_notta_16v/h5s_shard0 \
        --dir wan_experiment/results/i2v_notta_16v/h30s_shard0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FPS = 16
WIN = 16  # 1 s


def _to_gray(frames: np.ndarray) -> np.ndarray:
    w = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return frames[..., :3] @ w


def _laplacian_var(gray: np.ndarray) -> np.ndarray:
    lap = (
        -4.0 * gray
        + np.roll(gray, 1, axis=1) + np.roll(gray, -1, axis=1)
        + np.roll(gray, 1, axis=2) + np.roll(gray, -1, axis=2)
    )
    lap = lap[:, 1:-1, 1:-1]
    return lap.reshape(lap.shape[0], -1).var(axis=1)


def _colorfulness(frames: np.ndarray) -> np.ndarray:
    r, g, b = frames[..., 0], frames[..., 1], frames[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    rg2 = rg.reshape(rg.shape[0], -1)
    yb2 = yb.reshape(yb.shape[0], -1)
    std = np.sqrt(rg2.var(axis=1) + yb2.var(axis=1))
    mean = np.sqrt(rg2.mean(axis=1) ** 2 + yb2.mean(axis=1) ** 2)
    return std + 0.3 * mean


def _contrast(gray: np.ndarray) -> np.ndarray:
    return gray.reshape(gray.shape[0], -1).std(axis=1)


def _motion(frames: np.ndarray) -> np.ndarray:
    d = np.abs(np.diff(frames[..., :3], axis=0)).reshape(
        frames.shape[0] - 1, -1
    ).mean(axis=1)
    return np.concatenate([[np.nan], d])


def _read_head_tail(path: Path, win: int = WIN, skip_first: int = 1):
    """Stream the mp4; keep only skip_first:skip_first+win and the last win.

    Login-node OOM (2026-08-17): loading 481×480×832 float32 is ~2.3 GB/clip.
    Frame 0 is the I2V cond image — include it in the head and motion
    'collapses' because the still→video jump is in the reference window.
    """
    import imageio.v2 as imageio

    r = imageio.get_reader(str(path))
    head_u8: list[np.ndarray] = []
    tail_u8: list[np.ndarray] = []
    n = 0
    for im in r:
        im = np.asarray(im)[..., :3]
        if skip_first <= n < skip_first + win:
            head_u8.append(im)
        tail_u8.append(im)
        if len(tail_u8) > win:
            tail_u8.pop(0)
        n += 1
    r.close()
    if n < skip_first + 2 * win:
        raise RuntimeError(f"{path} has {n} frames; need >{skip_first + 2 * win}")
    head = np.stack(head_u8, axis=0).astype(np.float32) / 255.0
    tail = np.stack(tail_u8, axis=0).astype(np.float32) / 255.0
    return n, head, tail


def _window_stats(frames: np.ndarray) -> dict:
    gray = _to_gray(frames)
    motion = _motion(frames)
    return {
        "sharpness": float(np.mean(_laplacian_var(gray))),
        "colorfulness": float(np.mean(_colorfulness(frames))),
        "contrast": float(np.mean(_contrast(gray))),
        "temporal_motion": float(np.nanmean(motion)),
    }


def _head_tail_rel(n: int, head: np.ndarray, tail: np.ndarray) -> dict:
    hs = _window_stats(head)
    ts = _window_stats(tail)
    out = {"n_frames": int(n), "win": int(head.shape[0]), "skip_first": 1}
    for k in hs:
        h, t = hs[k], ts[k]
        rel = (t - h) / h if abs(h) > 1e-12 else float("nan")
        out[k] = {"head": h, "tail": t, "rel": rel}
    return out


def score_dir(d: Path) -> dict:
    mp4s = sorted(d.glob("*.mp4"))
    rows = []
    for p in mp4s:
        n, head, tail = _read_head_tail(p)
        rec = _head_tail_rel(n, head, tail)
        rec["mp4"] = str(p)
        rec["stem"] = p.stem
        rows.append(rec)
        print(
            f"  {p.name}: T={rec['n_frames']}  "
            f"sharp {rec['sharpness']['rel']:+.3f}  "
            f"color {rec['colorfulness']['rel']:+.3f}  "
            f"contr {rec['contrast']['rel']:+.3f}  "
            f"motion {rec['temporal_motion']['rel']:+.3f}",
            flush=True,
        )
    summary = {"dir": str(d), "n": len(rows), "rows": rows}
    if rows:
        keys = ("sharpness", "colorfulness", "contrast", "temporal_motion")
        summary["mean_rel"] = {
            k: float(np.nanmean([r[k]["rel"] for r in rows])) for k in keys
        }
        summary["median_rel"] = {
            k: float(np.nanmedian([r[k]["rel"] for r in rows])) for k in keys
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", action="append", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    reports = []
    for d in args.dir:
        d = Path(d)
        print(f"=== {d} ===", flush=True)
        reports.append(score_dir(d))
    print("\ntail/head relative change (1 s windows, skip cond frame 0):")
    for rep in reports:
        mr = rep.get("mean_rel", {})
        md = rep.get("median_rel", {})
        print(
            f"  {Path(rep['dir']).name}  n={rep['n']}  "
            f"mean  sharp={mr.get('sharpness', float('nan')):+.3f}  "
            f"color={mr.get('colorfulness', float('nan')):+.3f}  "
            f"contr={mr.get('contrast', float('nan')):+.3f}  "
            f"motion={mr.get('temporal_motion', float('nan')):+.3f}"
        )
        print(
            f"  {Path(rep['dir']).name}  n={rep['n']}  "
            f"median sharp={md.get('sharpness', float('nan')):+.3f}  "
            f"color={md.get('colorfulness', float('nan')):+.3f}  "
            f"contr={md.get('contrast', float('nan')):+.3f}  "
            f"motion={md.get('temporal_motion', float('nan')):+.3f}"
        )
    out = args.out
    if out is None and args.dir:
        out = Path(args.dir[-1]) / "drift_head_tail.json"
        if len(args.dir) > 1:
            out = Path(args.dir[0]).parent / "drift_head_tail.json"
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reports, indent=2))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
