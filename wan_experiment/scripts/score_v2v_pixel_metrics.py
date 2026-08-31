#!/usr/bin/env python3
"""Paired 30 s pixel metrics for caption V2V. No new generate.

Source leftover is real (128/128 >= 55 s). Compare the invented tail
to the true future after the 33-frame opening, resampled to 16 fps.

    python3 -u wan_experiment/scripts/score_v2v_pixel_metrics.py \
        --series-dir wan_experiment/results/v2v_panda_caption_128v \
        --methods notta rolling_notta sf_pseudo sf_always_search
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

GEN_FPS = 16.0
PREFIX_LATENTS = 9
SKIP_SRC = 1 + 4 * (PREFIX_LATENTS - 1)  # 33; frames actually encoded


def _sidecars(method_dir: Path) -> list[Path]:
    out = []
    for p in sorted(method_dir.glob("*.json")):
        if p.name in {"summary.json", "joined.json"} or "vbench" in p.name:
            continue
        if "pixel" in p.name:
            continue
        out.append(p)
    return out


def _read_mp4(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    r = imageio.get_reader(str(path))
    try:
        frames = [np.asarray(im)[..., :3] for im in r]
    finally:
        r.close()
    if not frames:
        raise RuntimeError(f"empty mp4 {path}")
    return np.stack(frames, axis=0)


def _source_tail(path: Path, skip: int, tail_s: float, n_out: int) -> np.ndarray:
    import imageio.v2 as imageio

    r = imageio.get_reader(str(path))
    try:
        meta = r.get_meta_data() or {}
        src_fps = float(meta.get("fps") or 30.0)
        n_src = max(int(round(tail_s * src_fps)), 1)
        grabbed = []
        for i, im in enumerate(r):
            if i < skip:
                continue
            grabbed.append(np.asarray(im)[..., :3])
            if len(grabbed) >= n_src:
                break
    finally:
        r.close()
    if len(grabbed) < 2:
        raise RuntimeError(f"short source tail {path} got {len(grabbed)}")
    idx = np.linspace(0, len(grabbed) - 1, n_out).round().astype(int)
    return np.stack([grabbed[int(i)] for i in idx], axis=0)


def _resize(frames: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    h, w = hw
    if frames.shape[1] == h and frames.shape[2] == w:
        return frames
    import cv2

    return np.stack(
        [cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA) for f in frames],
        axis=0,
    )


def _psnr_ssim(gen: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import peak_signal_noise_ratio as ski_psnr
    from skimage.metrics import structural_similarity as ski_ssim

    n = min(gen.shape[0], gt.shape[0])
    ps, ss = [], []
    for i in range(n):
        g = gen[i].astype(np.float64) / 255.0
        t = gt[i].astype(np.float64) / 255.0
        ps.append(float(ski_psnr(t, g, data_range=1.0)))
        ss.append(float(ski_ssim(t, g, data_range=1.0, channel_axis=2)))
    return float(np.mean(ps)), float(np.mean(ss))


def _lpips_mean(gen: np.ndarray, gt: np.ndarray, device: str, step: int) -> float | None:
    try:
        import lpips
        import torch
    except ImportError:
        return None
    n = min(gen.shape[0], gt.shape[0])
    idxs = list(range(0, n, max(1, int(step))))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    loss = lpips.LPIPS(net="alex").to(device).eval()
    vals = []
    with torch.no_grad():
        for i in idxs:
            a = torch.from_numpy(gen[i]).permute(2, 0, 1).float().div(127.5).sub(1)
            b = torch.from_numpy(gt[i]).permute(2, 0, 1).float().div(127.5).sub(1)
            vals.append(float(loss(a[None].to(device), b[None].to(device)).item()))
    return float(np.mean(vals)) if vals else None


def _median(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None and x == x]
    return float(statistics.median(xs)) if xs else None


def score_method(series: Path, method: str, device: str, lpips_step: int, force: bool) -> dict:
    method_dir = series / f"{method}_h30s_shard0"
    out_dir = method_dir / "pixel_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for js in _sidecars(method_dir):
        rec = json.loads(js.read_text())
        if not rec.get("ok"):
            continue
        key = rec.get("file_name") or rec.get("stem") or js.stem
        dest = out_dir / f"{js.stem}.json"
        if dest.is_file() and not force:
            rows.append(json.loads(dest.read_text()))
            continue
        mp4 = Path(rec["mp4"]) if rec.get("mp4") else method_dir / f"{js.stem}.mp4"
        src = Path(rec["video_path"])
        prefix_pix = int(rec.get("prefix_pix") or SKIP_SRC)
        gen = _read_mp4(mp4)
        tail = gen[prefix_pix:]
        if tail.shape[0] < 8:
            raise RuntimeError(f"{mp4.name}: tail {tail.shape[0]} after prefix {prefix_pix}")
        tail_s = float(tail.shape[0]) / GEN_FPS
        gt = _source_tail(src, SKIP_SRC, tail_s, tail.shape[0])
        gt = _resize(gt, (tail.shape[1], tail.shape[2]))
        n = min(tail.shape[0], gt.shape[0])
        tail, gt = tail[:n], gt[:n]
        psnr, ssim = _psnr_ssim(tail, gt)
        lp = _lpips_mean(tail, gt, device, lpips_step)
        row = {
            "file_name": key,
            "stem": rec.get("stem") or js.stem,
            "n_frames": int(n),
            "tail_s": tail_s,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lp,
        }
        dest.write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(
            f"  {method} {mp4.name} n={n} psnr={psnr:.3f} ssim={ssim:.4f} "
            f"lpips={lp if lp is not None else 'na'}",
            flush=True,
        )
    summary = {
        "method": method,
        "n": len(rows),
        "psnr": _median([r["psnr"] for r in rows]),
        "ssim": _median([r["ssim"] for r in rows]),
        "lpips": _median([r["lpips"] for r in rows if r.get("lpips") is not None]),
        "protocol": (
            f"gen tail after prefix_pix @ {GEN_FPS} fps vs source after "
            f"{SKIP_SRC} frames, time-resampled to 16 fps"
        ),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=[
        "notta", "rolling_notta", "sf_pseudo", "sf_always_search",
    ])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lpips-step", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    series = args.series_dir
    if not series.is_absolute():
        series = Path.cwd() / series
    print(f"pixel metrics series={series} methods={args.methods}", flush=True)
    summaries = []
    for m in args.methods:
        print(f"===== {m} =====", flush=True)
        summaries.append(score_method(
            series, m, args.device, args.lpips_step, args.force,
        ))
    print("\n# Caption 128 paired 30 s pixels (medians)")
    print("| Method | n | PSNR ↑ | SSIM ↑ | LPIPS ↓ |")
    print("|---|---:|---:|---:|---:|")
    for s in summaries:
        def fmt(x, nd=3):
            return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"
        print(
            f"| {s['method']} | {s['n']} | {fmt(s['psnr'])} | "
            f"{fmt(s['ssim'], 4)} | {fmt(s['lpips'], 4)} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
