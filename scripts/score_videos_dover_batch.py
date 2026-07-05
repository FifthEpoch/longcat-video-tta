#!/usr/bin/env python3
"""Batch-score generated mp4s with DOVER (aesthetic + technical branches).

Requires one-time setup: bash scripts/setup_dover_env.sh

Usage:
  python3 scripts/score_videos_dover_batch.py \\
      --input-dir sweep_experiment/results/panda_ood_budget_pilot/S2_LR5e3 \\
      --run-id S2_LR5e3 \\
      --output sweep_experiment/reports/dover_scores/S2_LR5e3_shard0.csv \\
      --shard-id 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

DOVER_AES_W, DOVER_IQ_W = 0.428, 0.572

_MEAN = None
_STD = None


def _iter_mp4s(root: Path) -> Iterator[Path]:
    for p in sorted(root.rglob("*.mp4")):
        if "videos" in p.parts:
            yield p


def _load_dover(dover_root: Path, weights: Path, device: str):
    import torch
    from dover.datasets import UnifiedFrameSampler, spatial_temporal_view_decomposition
    from dover.models import DOVER

    sys.path.insert(0, str(dover_root))
    opt_path = dover_root / "dover.yml"
    with opt_path.open(encoding="utf-8") as f:
        opt = yaml.safe_load(f)
    opt["test_load_path"] = str(weights)

    evaluator = DOVER(**opt["model"]["args"]).to(device)
    evaluator.load_state_dict(torch.load(str(weights), map_location=device))
    evaluator.eval()

    dopt = opt["data"]["val-l1080p"]["args"]
    temporal_samplers = {}
    for stype, sopt in dopt["sample_types"].items():
        if "t_frag" not in sopt:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"],
            )
        else:
            temporal_samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )
    mean = torch.FloatTensor([123.675, 116.28, 103.53])
    std = torch.FloatTensor([58.395, 57.12, 57.375])
    return evaluator, dopt, temporal_samplers, mean, std, torch


def _score_one(
    evaluator,
    dopt,
    temporal_samplers,
    mean,
    std,
    torch,
    video_path: Path,
    device: str,
) -> Tuple[float, float, float]:
    from dover.datasets import spatial_temporal_view_decomposition

    views, _ = spatial_temporal_view_decomposition(
        str(video_path), dopt["sample_types"], temporal_samplers,
    )
    for k, v in views.items():
        num_clips = dopt["sample_types"][k].get("num_clips", 1)
        views[k] = (
            ((v.permute(1, 2, 3, 0) - mean) / std)
            .permute(3, 0, 1, 2)
            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
            .transpose(0, 1)
            .to(device)
        )
    with torch.no_grad():
        results = [r.mean().item() for r in evaluator(views)]
    technical, aesthetic = float(results[0]), float(results[1])
    # Same fusion as DOVER evaluate_one_video.py -f (for cross-probe ranking).
    x = (technical - 0.1107) / 0.07355 * 0.6104 + (aesthetic + 0.08285) / 0.03774 * 0.3896
    fused = float(1 / (1 + np.exp(-x)))
    return aesthetic, technical, fused


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--dover-root", type=Path, default=Path(os.environ.get("DOVER_ROOT", "")))
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not args.dover_root.is_dir():
        print("ERROR: set DOVER_ROOT or run bash scripts/setup_dover_env.sh", file=sys.stderr)
        return 2

    weights = args.weights or (args.dover_root / "pretrained_weights" / "DOVER.pth")
    if not weights.is_file():
        print(f"ERROR: missing weights: {weights}", file=sys.stderr)
        return 2

    mp4s = list(_iter_mp4s(args.input_dir))
    if args.num_shards > 1:
        mp4s = [p for i, p in enumerate(mp4s) if i % args.num_shards == args.shard_id]
    if args.limit > 0:
        mp4s = mp4s[: args.limit]

    if not mp4s:
        print(f"WARN: no mp4s under {args.input_dir}", file=sys.stderr)
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluator, dopt, samplers, mean, std, torch = _load_dover(args.dover_root, weights, args.device)

    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "run_id", "mp4_path", "aesthetic", "technical", "fused"])
        for i, mp4 in enumerate(mp4s):
            vid = mp4.stem
            try:
                aes, tech, fused = _score_one(
                    evaluator, dopt, samplers, mean, std, torch, mp4, args.device,
                )
                w.writerow([vid, args.run_id, str(mp4), f"{aes:.6f}", f"{tech:.6f}", f"{fused:.6f}"])
            except Exception as exc:
                print(f"FAIL {mp4}: {exc}", file=sys.stderr)
            if (i + 1) % 10 == 0:
                print(f"  {args.run_id} shard {args.shard_id}: {i + 1}/{len(mp4s)}", file=sys.stderr)

    print(f"Wrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
