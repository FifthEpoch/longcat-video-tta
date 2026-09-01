#!/usr/bin/env python3
"""I3D FVD on aligned 30 s V2V tails. No new generate.

Do not score the full mp4 — that includes the real 2 s prefix.
Same pairing as PSNR: gen after prefix_pix @ 16 fps vs source after
33 frames, time-resampled. I3D sees 16 consecutive frames.

Primary: all non-overlapping 16-frame windows on the tail
(~31 clips × 128 videos). Last-window FVD is also written (n=128).

    python3 -u wan_experiment/scripts/score_v2v_aligned_fvd.py \
        --series-dir wan_experiment/results/v2v_panda_caption_128v \
        --methods notta rolling_notta sf_pseudo sf_always_search
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import functional as TF

_REPO = Path(__file__).resolve().parents[2]
_SWEEP = _REPO / "sweep_experiment" / "scripts"
_WAN = Path(__file__).resolve().parent
for p in (_REPO, _SWEEP, _WAN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from eval_fvd import (  # noqa: E402
    compute_frechet_distance,
    extract_i3d_features,
    _load_i3d,
    _i3d_file_hash,
)
from score_v2v_pixel_metrics import (  # noqa: E402
    GEN_FPS,
    SKIP_SRC,
    _read_mp4,
    _resize,
    _sidecars,
    _source_tail,
)

CLIP = 16
I3D_SIZE = 224


def _to_i3d(clip: np.ndarray) -> torch.Tensor:
    """uint8 [T,H,W,C] → [1,T,C,224,224] float [0,1]."""
    tensors = []
    for fr in clip:
        img = torch.from_numpy(fr).permute(2, 0, 1).float().div(255.0)
        img = TF.resize(img, I3D_SIZE, interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.center_crop(img, I3D_SIZE)
        tensors.append(img)
    return torch.stack(tensors, dim=0).unsqueeze(0)


def _windows(frames: np.ndarray) -> list[np.ndarray]:
    n = (frames.shape[0] // CLIP) * CLIP
    return [frames[i:i + CLIP] for i in range(0, n, CLIP)]


def _cache_path(series: Path, method: str, stem: str) -> Path:
    d = series / f"{method}_h30s_shard0" / "pixel_full" / "fvd_i3d"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}.npz"


def _gt_cache_path(series: Path, stem: str) -> Path:
    d = series / "fvd_gt_i3d"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{stem}.npz"


def _features_for_video(
    rec: dict,
    js: Path,
    method_dir: Path,
    series: Path,
    method: str,
    i3d,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    stem = rec.get("stem") or js.stem
    cache = _cache_path(series, method, stem)
    if cache.is_file():
        z = np.load(cache)
        return z["gen"], z["ref"]
    mp4 = Path(rec["mp4"]) if rec.get("mp4") else method_dir / f"{js.stem}.mp4"
    src = Path(rec["video_path"])
    if not mp4.is_file() or not src.is_file():
        print(f"  SKIP missing {mp4} or {src}", flush=True)
        return None
    prefix_pix = int(rec.get("prefix_pix") or SKIP_SRC)
    gen = _read_mp4(mp4)[prefix_pix:]
    if gen.shape[0] < CLIP:
        print(f"  SKIP short tail {mp4.name} n={gen.shape[0]}", flush=True)
        return None
    tail_s = float(gen.shape[0]) / GEN_FPS
    gt = _source_tail(src, SKIP_SRC, tail_s, gen.shape[0])
    gt = _resize(gt, (gen.shape[1], gen.shape[2]))
    n = min(gen.shape[0], gt.shape[0])
    gen, gt = gen[:n], gt[:n]
    g_wins = _windows(gen)
    r_wins = _windows(gt)
    n_w = min(len(g_wins), len(r_wins))
    if n_w < 1:
        print(f"  SKIP no 16-frame window {mp4.name}", flush=True)
        return None
    g_wins, r_wins = g_wins[:n_w], r_wins[:n_w]
    gt_npz = _gt_cache_path(series, stem)
    if gt_npz.is_file():
        feats_ref = np.load(gt_npz)["ref"]
        if feats_ref.shape[0] != n_w:
            feats_ref = extract_i3d_features(
                [_to_i3d(w) for w in r_wins], i3d, device, batch_size,
            )
            np.savez(gt_npz, ref=feats_ref)
    else:
        feats_ref = extract_i3d_features(
            [_to_i3d(w) for w in r_wins], i3d, device, batch_size,
        )
        np.savez(gt_npz, ref=feats_ref)
    feats_gen = extract_i3d_features(
        [_to_i3d(w) for w in g_wins], i3d, device, batch_size,
    )
    np.savez(cache, gen=feats_gen, ref=feats_ref)
    print(
        f"  {method} {mp4.name} windows={n_w} n={n}",
        flush=True,
    )
    return feats_gen, feats_ref


def score_method(
    series: Path,
    method: str,
    i3d,
    i3d_hash: str,
    device: str,
    batch_size: int,
) -> dict:
    method_dir = series / f"{method}_h30s_shard0"
    gen_all = []
    ref_all = []
    last_gen = []
    last_ref = []
    n_vid = 0
    for js in _sidecars(method_dir):
        rec = json.loads(js.read_text())
        if not rec.get("ok"):
            continue
        pair = _features_for_video(
            rec, js, method_dir, series, method, i3d, device, batch_size,
        )
        if pair is None:
            continue
        g, r = pair
        gen_all.append(g)
        ref_all.append(r)
        last_gen.append(g[-1])
        last_ref.append(r[-1])
        n_vid += 1
    if n_vid < 1:
        raise RuntimeError(f"{method}: no FVD clips")
    feats_g = np.concatenate(gen_all, axis=0)
    feats_r = np.concatenate(ref_all, axis=0)
    fvd = compute_frechet_distance(feats_g, feats_r)
    last_g = np.stack(last_gen, axis=0)
    last_r = np.stack(last_ref, axis=0)
    fvd_last = compute_frechet_distance(last_g, last_r)
    out = {
        "method": method,
        "fvd": float(fvd),
        "fvd_last16": float(fvd_last),
        "n_videos": n_vid,
        "n_clips": int(feats_g.shape[0]),
        "clip_frames": CLIP,
        "protocol": (
            "aligned 30 s tails; I3D 16-frame non-overlapping windows; "
            "fvd_last16 = last window only (n=videos)"
        ),
        "feature_extractor": "i3d_kinetics400_torchscript",
        "i3d_weights_sha256": i3d_hash,
        "sample_size_warning": (
            None if feats_g.shape[0] >= 256
            else f"n_clips={feats_g.shape[0]} < 256"
        ),
    }
    dest = method_dir / "pixel_full" / "fvd.json"
    dest.write_text(json.dumps(out, indent=2))
    print(
        f"  {method} FVD={fvd:.3f} last16={fvd_last:.3f} "
        f"n_vid={n_vid} n_clips={feats_g.shape[0]}",
        flush=True,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series-dir", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=[
        "notta", "rolling_notta", "sf_pseudo", "sf_always_search",
    ])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()
    series = args.series_dir
    if not series.is_absolute():
        series = Path.cwd() / series
    print(f"aligned FVD series={series} methods={args.methods}", flush=True)
    print("Loading I3D (Kinetics-400 TorchScript)...", flush=True)
    i3d = _load_i3d(args.device)
    i3d_hash = _i3d_file_hash()
    print(f"  I3D sha {i3d_hash[:16]}...", flush=True)
    rows = []
    for m in args.methods:
        print(f"===== {m} =====", flush=True)
        rows.append(score_method(
            series, m, i3d, i3d_hash, args.device, args.batch_size,
        ))
    print("\n# Caption 128 aligned-tail FVD (I3D, 16-frame windows)")
    print("| Method | n_videos | n_clips | FVD ↓ | last-16 FVD ↓ |")
    print("|---|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['method']} | {r['n_videos']} | {r['n_clips']} | "
            f"{r['fvd']:.2f} | {r['fvd_last16']:.2f} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
