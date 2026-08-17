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


def _read_mp4(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    r = imageio.get_reader(str(path))
    frames = [np.asarray(im)[..., :3] for im in r]
    r.close()
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    return arr


def _window_means(frames: np.ndarray) -> dict:
    gray = _to_gray(frames)
    sigs = {
        "sharpness": _laplacian_var(gray),
        "colorfulness": _colorfulness(frames),
        "contrast": _contrast(gray),
        "temporal_motion": _motion(frames),
    }
    t = frames.shape[0]
    w = min(WIN, max(1, t // 3))
    out = {"n_frames": int(t), "win": int(w)}
    for k, v in sigs.items():
        v = np.asarray(v, dtype=np.float64)
        head = float(np.nanmean(v[:w]))
        tail = float(np.nanmean(v[-w:]))
        rel = (tail - head) / head if abs(head) > 1e-12 else float("nan")
        out[k] = {"head": head, "tail": tail, "rel": rel}
    return out


def score_dir(d: Path) -> dict:
    mp4s = sorted(d.glob("*.mp4"))
    rows = []
    for p in mp4s:
        frames = _read_mp4(p)
        rec = _window_means(frames)
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
        summary["mean_rel"] = {
            k: float(np.nanmean([r[k]["rel"] for r in rows]))
            for k in ("sharpness", "colorfulness", "contrast", "temporal_motion")
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
    print("\nmean tail/head relative change (1 s windows):")
    for rep in reports:
        mr = rep.get("mean_rel", {})
        print(
            f"  {Path(rep['dir']).name}  n={rep['n']}  "
            f"sharp={mr.get('sharpness', float('nan')):+.3f}  "
            f"color={mr.get('colorfulness', float('nan')):+.3f}  "
            f"contr={mr.get('contrast', float('nan')):+.3f}  "
            f"motion={mr.get('temporal_motion', float('nan')):+.3f}"
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
