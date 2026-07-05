#!/usr/bin/env python3
"""Rich LongCat-VAE latent profiles on the TTA-visible window (Tier 1).

Encodes frames [0, 48) with the same ``encode_video`` path as TTA, then
pools latents [B, C, T, H, W] into a fixed ~130-dim vector per video:

  * Per region (full / context / target): per-channel mean+std, token-norm
    stats, temporal-delta stats (mean, std, p90).
  * Context-vs-target per-channel energy ratios.

Context/target split mirrors ``compute_diffusion_ood_score.py`` /
``run_delta_a.py`` (``tta_context_frames=14`` → 4 context latents).

Output: ``vae_latent_profile_features.csv`` (one row per video).

Run:
    python3 scripts/extract_vae_latent_profile_features.py \\
        --checkpoint-dir /scratch/$USER/longcat-video-checkpoints \\
        --videos-dir datasets/panda_ood_budget_pilot_480p \\
        --output sweep_experiment/reports/per_video_analysis/2026-07-06/vae_latent_profile_features.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "delta_experiment" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT))

TTA_TOTAL_FRAMES: int = 48
GEN_START_FRAME: int = 48
TTA_CONTEXT_FRAMES: int = 14
VAE_TEMPORAL_SCALE: int = 4
AUTO_TTA_VISIBLE_RANGE: Tuple[int, int] = (
    max(0, GEN_START_FRAME - TTA_TOTAL_FRAMES),
    GEN_START_FRAME,
)

_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def _canonical_video_id(s: Optional[str]) -> str:
    if not s:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


def _parse_frame_range_arg(arg: str, default: Tuple[int, int]) -> Tuple[int, int]:
    if not arg or arg.lower() == "auto":
        return default
    a, b = arg.split(":", 1)
    return int(a), int(b)


def list_video_paths(videos_dir: Path) -> List[Path]:
    candidates: List[Path] = []
    subdir = videos_dir / "videos"
    if subdir.is_dir():
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(subdir.glob(ext))
    if not candidates:
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(videos_dir.rglob(ext))
    return sorted(candidates, key=lambda p: _canonical_video_id(p.name))


def _num_context_latents(tta_context_frames: int) -> int:
    return max(1, 1 + (max(1, tta_context_frames) - 1) // VAE_TEMPORAL_SCALE)


def _region_feature_names(prefix: str, n_channels: int) -> List[str]:
    names: List[str] = []
    for c in range(n_channels):
        names.append(f"vae_{prefix}_ch{c}_mean")
        names.append(f"vae_{prefix}_ch{c}_std")
    names.extend([
        f"vae_{prefix}_toknorm_mean",
        f"vae_{prefix}_toknorm_std",
        f"vae_{prefix}_toknorm_p90",
        f"vae_{prefix}_dt_mean",
        f"vae_{prefix}_dt_std",
        f"vae_{prefix}_dt_p90",
    ])
    return names


def profile_feature_names(n_channels: int = 16) -> List[str]:
    names: List[str] = []
    for prefix in ("full", "ctx", "tgt"):
        names.extend(_region_feature_names(prefix, n_channels))
    for c in range(n_channels):
        names.append(f"vae_ctx_tgt_ch{c}_ratio")
    return names


def compute_latent_profile(
    latents: "torch.Tensor",
    *,
    num_ctx_lat: int,
    n_channels: int = 16,
) -> Dict[str, float]:
    """latents: [1, C, T, H, W] on CPU or GPU."""
    import torch

    out: Dict[str, float] = {n: float("nan") for n in profile_feature_names(n_channels)}
    if latents.ndim != 5 or latents.shape[1] < n_channels:
        return out

    z = latents[:, :n_channels].detach().float()
    t_lat = z.shape[2]
    n_ctx = max(1, min(num_ctx_lat, t_lat - 1))
    regions = {
        "full": z,
        "ctx": z[:, :, :n_ctx],
        "tgt": z[:, :, n_ctx:],
    }

    for prefix, reg in regions.items():
        if reg.numel() == 0:
            continue
        for c in range(n_channels):
            ch = reg[:, c]
            out[f"vae_{prefix}_ch{c}_mean"] = float(ch.mean().item())
            out[f"vae_{prefix}_ch{c}_std"] = float(ch.std(unbiased=False).item())
        toknorm = reg.norm(dim=1)
        flat = toknorm.reshape(-1)
        out[f"vae_{prefix}_toknorm_mean"] = float(flat.mean().item())
        out[f"vae_{prefix}_toknorm_std"] = float(flat.std(unbiased=False).item())
        out[f"vae_{prefix}_toknorm_p90"] = float(torch.quantile(flat, 0.9).item())
        if reg.shape[2] >= 2:
            diff = reg[:, :, 1:] - reg[:, :, :-1]
            dnorm = diff.reshape(diff.shape[0], diff.shape[1], -1).norm(dim=-1)
            dflat = dnorm.reshape(-1)
            out[f"vae_{prefix}_dt_mean"] = float(dflat.mean().item())
            out[f"vae_{prefix}_dt_std"] = float(dflat.std(unbiased=False).item())
            out[f"vae_{prefix}_dt_p90"] = float(torch.quantile(dflat, 0.9).item())

    ctx, tgt = regions["ctx"], regions["tgt"]
    if ctx.numel() > 0 and tgt.numel() > 0:
        for c in range(n_channels):
            ctx_e = ctx[:, c].abs().mean()
            tgt_e = tgt[:, c].abs().mean()
            out[f"vae_ctx_tgt_ch{c}_ratio"] = float((ctx_e / (tgt_e + 1e-8)).item())

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint-dir", type=str, required=True)
    ap.add_argument("--videos-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--tta-visible-frames", type=str, default="auto")
    ap.add_argument("--tta-context-frames", type=int, default=TTA_CONTEXT_FRAMES)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    visible = _parse_frame_range_arg(args.tta_visible_frames, AUTO_TTA_VISIBLE_RANGE)
    n_visible = visible[1] - visible[0]
    num_ctx_lat = _num_context_latents(args.tta_context_frames)
    feat_names = profile_feature_names()
    meta_cols = [
        "video_id", "n_visible_frames", "tta_visible_range",
        "n_latent_frames", "n_context_latents", "tta_context_frames",
    ]
    fieldnames = meta_cols + feat_names

    existing: Dict[str, dict] = {}
    if args.resume and args.output.exists():
        with args.output.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                vid = (r.get("video_id") or "").strip()
                if vid:
                    existing[vid] = r

    paths = list_video_paths(args.videos_dir)
    if args.max_videos > 0:
        paths = paths[: args.max_videos]
    todo = [
        vp for vp in paths
        if _canonical_video_id(vp.name) not in existing or not args.resume
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading LongCat VAE...")
    try:
        from common import encode_video, load_longcat_components, load_video_frames
    except ImportError as exc:
        print(f"[error] common import failed: {exc}", file=sys.stderr)
        return 2
    import torch

    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16,
    )
    vae = components["vae"]
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    rows = dict(existing)
    n_done, n_err = 0, 0
    t0 = time.time()
    tta_start = max(0, visible[0])

    for i, vp in enumerate(todo):
        vid = _canonical_video_id(vp.name)
        try:
            pixel = load_video_frames(
                str(vp),
                num_frames=n_visible,
                height=480,
                width=832,
                start_frame=tta_start,
            ).to(args.device, torch.bfloat16)
            with torch.inference_mode():
                latents = encode_video(vae, pixel, normalize=True)
            prof = compute_latent_profile(
                latents, num_ctx_lat=num_ctx_lat, n_channels=latents.shape[1],
            )
            row = {
                "video_id": vid,
                "n_visible_frames": n_visible,
                "tta_visible_range": f"{visible[0]}:{visible[1]}",
                "n_latent_frames": int(latents.shape[2]),
                "n_context_latents": num_ctx_lat,
                "tta_context_frames": args.tta_context_frames,
            }
            row.update(prof)
            rows[vid] = row
            n_done += 1
            del pixel, latents
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001
            print(f"[error] {vp.name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            n_err += 1

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(todo)}] done={n_done} err={n_err}", flush=True)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for vid in sorted(rows):
            row = rows[vid]
            out: Dict[str, str] = {}
            for k in fieldnames:
                v = row.get(k)
                if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    out[k] = ""
                elif isinstance(v, float):
                    out[k] = f"{v:.6f}"
                else:
                    out[k] = "" if v is None else str(v)
            w.writerow(out)

    print(
        f"Wrote {args.output}  features={len(feat_names)} "
        f"new={n_done} err={n_err} elapsed={time.time()-t0:.1f}s",
    )
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
