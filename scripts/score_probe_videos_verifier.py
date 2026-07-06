#!/usr/bin/env python3
"""Batch-score budget-pilot probe mp4s with frozen learned verifiers.

Backends (Options 1–2):
  videoscore   — TIGER-Lab/VideoScore (5-dim regression, Option 1)
  videoreward  — Kling VideoReward / VideoAlign (VQ/MQ/TA, Option 2)
  visionreward — THUDM VisionReward-Video scalar (optional, slow)

Requires one-time setup:
  bash scripts/setup_verifier_models.sh

Usage:
  python3 scripts/score_probe_videos_verifier.py \\
      --backend videoscore \\
      --input-dir sweep_experiment/results/panda_ood_budget_pilot/S2_LR5e3 \\
      --run-id S2_LR5e3 \\
      --output sweep_experiment/reports/verifier_scores/S2_LR5e3_videoscore_shard0.csv \\
      --shard-id 0 --num-shards 4
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.caption_utils import load_resolved_captions_csv  # noqa: E402
from scripts.verifier_probe_common import canonical_video_id, iter_probe_mp4s  # noqa: E402

DEFAULT_PROMPT = (
    "A natural video clip with smooth motion, consistent appearance, "
    "and good visual quality."
)
MAX_NUM_FRAMES = 16
VISIONREWARD_MAX_QUESTIONS = 29


def _load_videoscore(device: str):
    import torch
    from transformers import AutoProcessor

    model_name = os.environ.get("VIDEOSCORE_MODEL", "TIGER-Lab/VideoScore")
    try:
        from mantis.models.idefics2 import Idefics2ForSequenceClassification
    except ImportError as exc:
        raise ImportError(
            "Install mantis-llava for VideoScore: pip install mantis-llava"
        ) from exc

    processor = AutoProcessor.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model = Idefics2ForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.bfloat16,
    ).eval().to(device)
    return model, processor, torch


def _score_videoscore(
    model,
    processor,
    torch,
    mp4: Path,
    prompt: str,
    device: str,
) -> Dict[str, float]:
    import av

    container = av.open(str(mp4))
    stream = container.streams.video[0]
    n_frames = stream.frames or 0
    if n_frames <= 0:
        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
        n_frames = len(frames)
    else:
        indices = np.linspace(0, max(n_frames - 1, 0), MAX_NUM_FRAMES, dtype=int)
        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in set(indices.tolist()):
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= MAX_NUM_FRAMES:
                break
    container.close()
    if not frames:
        raise ValueError(f"no frames decoded from {mp4}")

    eval_prompt = (
        "Suppose you are an expert in judging AI-generated video quality. "
        f"The text prompt is: {prompt}\n"
        "Score visual quality, temporal consistency, dynamic degree, "
        "text-to-video alignment, and factual consistency from 1.0 to 4.0."
    )
    images = [[img] for img in frames]
    inputs = processor(text=eval_prompt, images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits[0]
    names = ["vq", "tc", "dd", "tva", "fc"]
    out = {names[i]: float(logits[i].item()) for i in range(min(len(names), logits.shape[0]))}
    out["mean5"] = float(np.mean(list(out.values())))
    return out


def _load_videoreward(device: str, dtype_str: str = "bfloat16"):
    root = Path(os.environ.get("VIDEOALIGN_ROOT", ""))
    if not root.is_dir():
        raise FileNotFoundError(
            "Set VIDEOALIGN_ROOT to cloned VideoAlign repo "
            "(bash scripts/setup_verifier_models.sh)"
        )
    sys.path.insert(0, str(root))
    import torch

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    from inference import VideoVLMRewardInference  # type: ignore

    ckpt = os.environ.get("VIDEOREWARD_CKPT", str(root / "checkpoints"))
    inferencer = VideoVLMRewardInference(ckpt, device=device, dtype=dtype)
    return inferencer


def _score_videoreward(inferencer, mp4: Path, prompt: str) -> Dict[str, float]:
    rewards = inferencer.reward([str(mp4)], [prompt], use_norm=True)
    r = rewards[0]
    return {
        "vq": float(r["VQ"]),
        "mq": float(r["MQ"]),
        "ta": float(r["TA"]),
        "overall": float(r.get("Overall", r["VQ"] + r["MQ"] + r["TA"])),
    }


def _load_visionreward(device: str):
    root = Path(os.environ.get("VISIONREWARD_ROOT", ""))
    if not root.is_dir():
        raise FileNotFoundError(
            "Set VISIONREWARD_ROOT (bash scripts/setup_verifier_models.sh)"
        )
    sys.path.insert(0, str(root))
    # VisionReward inference loads CogVLM2-Video internally on first call.
    qa_path = root / "VisionReward_Video" / "VisionReward_video_qa.txt"
    weight_path = root / "VisionReward_Video" / "weight.json"
    if not qa_path.is_file():
        raise FileNotFoundError(f"missing {qa_path}")
    import json

    with qa_path.open(encoding="utf-8") as f:
        questions = [ln.strip() for ln in f if ln.strip()]
    with weight_path.open(encoding="utf-8") as f:
        weight_obj = json.load(f)
    weights = np.array(weight_obj if isinstance(weight_obj, list) else list(weight_obj.values()))
    questions = questions[:VISIONREWARD_MAX_QUESTIONS]
    weights = weights[: len(questions)]
    return root, questions, weights


def _score_visionreward(root: Path, questions: List[str], weights: np.ndarray, mp4: Path, prompt: str) -> Dict[str, float]:
    from inference_video import inference, score as vr_score  # type: ignore

    # inference_video.score runs all questions; reuse if available.
    s = vr_score(str(mp4), prompt)
    return {"score": float(s)}


def _collect_mp4s(
    input_dir: Path,
    run_id: str,
    shard_id: int,
    num_shards: int,
    limit: int,
) -> List[Tuple[str, Path]]:
    series_root = input_dir.parent if input_dir.name == run_id else input_dir
    mp4s = list(iter_probe_mp4s(series_root, run_id))
    if not mp4s:
        mp4s = [
            (canonical_video_id(p.name), p)
            for p in sorted((series_root / run_id).rglob("*.mp4"))
            if "videos" in p.parts
        ]
    if num_shards > 1:
        mp4s = [t for i, t in enumerate(mp4s) if i % num_shards == shard_id]
    if limit > 0:
        mp4s = mp4s[:limit]
    return mp4s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("videoscore", "videoreward", "visionreward"), required=True)
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--run-id", type=str, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--captions-csv", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mp4s = _collect_mp4s(args.input_dir, args.run_id, args.shard_id, args.num_shards, args.limit)
    if not mp4s:
        print(f"WARN: no mp4s under {args.input_dir}", file=sys.stderr)
        return 0

    captions: Dict[str, str] = {}
    if args.captions_csv and args.captions_csv.is_file():
        captions = load_resolved_captions_csv(args.captions_csv)
    elif (_REPO / "datasets/panda_ood_budget_pilot_480p/metadata.csv").is_file():
        captions = load_resolved_captions_csv(_REPO / "datasets/panda_ood_budget_pilot_480p/metadata.csv")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.backend == "videoscore":
        model, processor, torch = _load_videoscore(args.device)
        score_fn = lambda mp4, prompt: _score_videoscore(model, processor, torch, mp4, prompt, args.device)
        fieldnames = ["video_id", "run_id", "mp4_path", "prompt", "vq", "tc", "dd", "tva", "fc", "mean5"]
    elif args.backend == "videoreward":
        inferencer = _load_videoreward(args.device)
        score_fn = lambda mp4, prompt: _score_videoreward(inferencer, mp4, prompt)
        fieldnames = ["video_id", "run_id", "mp4_path", "prompt", "vq", "mq", "ta", "overall"]
    else:
        root, questions, weights = _load_visionreward(args.device)
        score_fn = lambda mp4, prompt: _score_visionreward(root, questions, weights, mp4, prompt)
        fieldnames = ["video_id", "run_id", "mp4_path", "prompt", "score"]

    n_ok = 0
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, (vid, mp4) in enumerate(mp4s):
            prompt = captions.get(vid) or DEFAULT_PROMPT
            try:
                scores = score_fn(mp4, prompt)
                row = {
                    "video_id": vid,
                    "run_id": args.run_id,
                    "mp4_path": str(mp4),
                    "prompt": prompt[:200],
                    **{k: f"{v:.6f}" for k, v in scores.items()},
                }
                w.writerow(row)
                n_ok += 1
            except Exception as exc:
                print(f"FAIL {mp4}: {exc}", file=sys.stderr)
            if (i + 1) % 5 == 0:
                print(f"  {args.backend} {args.run_id} shard {args.shard_id}: {i + 1}/{len(mp4s)}", file=sys.stderr)

    print(f"Wrote {args.output} ({n_ok}/{len(mp4s)} ok)", file=sys.stderr)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
