#!/usr/bin/env python3
"""Extract learned features from budget-pilot **probe output** mp4s (Options 3–4).

Modes:
  vae     — LongCat VAE latent embedding (~72-d per probe run, Option 3)
  resnet  — Frozen ResNet18 frame embedding (~512-d per probe run, Option 4)

Output CSV: one row per (video_id, run_id) with emb_* columns.

Usage:
  python3 scripts/extract_probe_learned_features.py \\
      --mode vae \\
      --series-root sweep_experiment/results/panda_ood_budget_pilot \\
      --run-id S2_LR5e3 \\
      --checkpoint-dir /scratch/$USER/longcat-video-checkpoints \\
      --output sweep_experiment/reports/verifier_features/S2_LR5e3_vae_shard0.csv \\
      --shard-id 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "delta_experiment" / "scripts"))
sys.path.insert(0, str(_REPO))

from scripts.verifier_probe_common import iter_probe_mp4s  # noqa: E402

TTA_VISIBLE = (0, 48)
N_FRAMES_RESNET = 8
VAE_EMB_DIM = 72
RESNET_EMB_DIM = 512


def _vae_embedding(latents) -> np.ndarray:
    """Pool [1,C,T,H,W] latents to fixed 72-d vector."""
    import torch

    z = latents[:, :16].detach().float()
    parts: List[float] = []
    for c in range(min(16, z.shape[1])):
        ch = z[:, c]
        parts.append(float(ch.mean().item()))
        parts.append(float(ch.std(unbiased=False).item()))
        parts.append(float(ch.abs().mean().item()))
        parts.append(float(torch.quantile(ch.reshape(-1), 0.9).item()))
    tok = z.norm(dim=1).reshape(-1)
    parts.extend([
        float(tok.mean().item()),
        float(tok.std(unbiased=False).item()),
        float(torch.quantile(tok, 0.9).item()),
        float(tok.max().item()),
    ])
    while len(parts) < VAE_EMB_DIM:
        parts.append(0.0)
    return np.array(parts[:VAE_EMB_DIM], dtype=np.float32)


def _resnet_embedding(model, preprocess, torch, mp4: Path, device: str) -> np.ndarray:
    import av

    container = av.open(str(mp4))
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    if not frames:
        raise ValueError(f"no frames in {mp4}")
    idx = np.linspace(0, len(frames) - 1, N_FRAMES_RESNET, dtype=int)
    sampled = [frames[i] for i in idx]
    from PIL import Image

    tensors = [preprocess(Image.fromarray(fr)).unsqueeze(0) for fr in sampled]
    batch = torch.cat(tensors, dim=0).to(device)
    with torch.no_grad():
        feats = model(batch)
    return feats.mean(dim=0).cpu().numpy().astype(np.float32)


def _emb_fieldnames(mode: str) -> List[str]:
    dim = VAE_EMB_DIM if mode == "vae" else RESNET_EMB_DIM
    return ["video_id", "run_id", "mp4_path", "mode"] + [f"emb_{i}" for i in range(dim)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("vae", "resnet"), required=True)
    ap.add_argument("--series-root", type=Path, required=True)
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--checkpoint-dir", type=str, default="")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mp4s = list(iter_probe_mp4s(args.series_root, args.run_id))
    if args.num_shards > 1:
        mp4s = [t for i, t in enumerate(mp4s) if i % args.num_shards == args.shard_id]
    if args.limit > 0:
        mp4s = mp4s[: args.limit]
    if not mp4s:
        print(f"WARN: no probe mp4s for {args.run_id}", file=sys.stderr)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = _emb_fieldnames(args.mode)

    vae = None
    resnet_model = preprocess = torch = None
    if args.mode == "vae":
        if not args.checkpoint_dir:
            print("ERROR: --checkpoint-dir required for vae mode", file=sys.stderr)
            return 2
        from common import encode_video, load_longcat_components, load_video_frames

        import torch as _torch

        torch = _torch
        components = load_longcat_components(
            args.checkpoint_dir, device=args.device, dtype=torch.bfloat16,
        )
        vae = components["vae"]
        vae.eval()
        for p in vae.parameters():
            p.requires_grad_(False)
    else:
        import torch as _torch
        import torchvision.models as models
        from torchvision.models import ResNet18_Weights

        torch = _torch
        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet_model = models.resnet18(weights=weights)
        resnet_model.fc = torch.nn.Identity()
        resnet_model.eval().to(args.device)
        preprocess = weights.transforms()

    n_ok, n_err = 0, 0
    t0 = time.time()
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (vid, mp4) in enumerate(mp4s):
            try:
                if args.mode == "vae":
                    from common import load_video_frames, encode_video

                    n_vis = TTA_VISIBLE[1] - TTA_VISIBLE[0]
                    pixel = load_video_frames(
                        str(mp4), num_frames=n_vis, height=480, width=832,
                        start_frame=TTA_VISIBLE[0],
                    ).to(args.device, torch.bfloat16)
                    with torch.inference_mode():
                        latents = encode_video(vae, pixel, normalize=True)
                    emb = _vae_embedding(latents)
                    del pixel, latents
                    torch.cuda.empty_cache()
                else:
                    emb = _resnet_embedding(resnet_model, preprocess, torch, mp4, args.device)

                row = {
                    "video_id": vid,
                    "run_id": args.run_id,
                    "mp4_path": str(mp4),
                    "mode": args.mode,
                }
                for j, val in enumerate(emb):
                    row[f"emb_{j}"] = f"{float(val):.6f}"
                w.writerow(row)
                n_ok += 1
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {mp4}: {exc}", file=sys.stderr)
                traceback.print_exc()
                n_err += 1
            if (i + 1) % 10 == 0:
                print(f"  {args.mode} {args.run_id} shard {args.shard_id}: {i+1}/{len(mp4s)}", file=sys.stderr)

    print(
        f"Wrote {args.output} ok={n_ok} err={n_err} elapsed={time.time()-t0:.1f}s",
        file=sys.stderr,
    )
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
