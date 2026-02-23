#!/usr/bin/env python3
"""
LongCat-Video Text-to-Video Baseline

Generates videos from text prompts only (0 conditioning frames) using the
LongCat-Video pipeline.  Reads prompts from a Panda-70M metadata CSV and
saves one MP4 per prompt.

Usage:
    python t2v_experiment/scripts/run_longcat_t2v.py \
        --checkpoint-dir /path/to/longcat-checkpoints \
        --meta-csv datasets/panda_100/metadata.csv \
        --output-dir t2v_experiment/results/longcat \
        --num-frames 29 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_LONGCAT_DIR = _REPO_ROOT / "LongCat-Video"
sys.path.insert(0, str(_LONGCAT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from torchvision.io import write_video


def parse_caption(raw: str) -> str:
    """Extract the first caption from a Panda-70M list-string."""
    if raw.startswith("[") and "'" in raw:
        try:
            captions = eval(raw)
            return captions[0] if captions else "A video clip"
        except Exception:
            pass
    return raw


def load_prompts(meta_csv: str, max_videos: int = 100) -> list[dict]:
    entries = []
    with open(meta_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append({
                "filename": row.get("filename", f"video_{len(entries):04d}.mp4"),
                "prompt": parse_caption(row.get("caption", "A video clip")),
            })
            if len(entries) >= max_videos:
                break
    return entries


def main():
    parser = argparse.ArgumentParser(description="LongCat T2V baseline")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--meta-csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=29,
                        help="Total frames to generate (must be 4k+1)")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────
    print("=" * 70)
    print("LongCat-Video T2V Baseline")
    print("=" * 70)
    t0 = time.time()

    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

    dtype = torch.bfloat16
    ckpt = args.checkpoint_dir

    tokenizer = AutoTokenizer.from_pretrained(ckpt, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder = UMT5EncoderModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKLWan.from_pretrained(ckpt, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ckpt, subfolder="scheduler", torch_dtype=dtype)
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        ckpt, subfolder="dit", cp_split_hw=[1, 1],
        enable_flashattn2=True, torch_dtype=dtype,
    )

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder,
        vae=vae, scheduler=scheduler, dit=dit,
    )
    pipe.to("cuda")

    negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, "
        "style, works, paintings, images, static, overall gray, worst quality, "
        "low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, "
        "misshapen limbs, fused fingers, still picture, messy background"
    )

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print()

    # ── Load prompts ──────────────────────────────────────────────────
    prompts = load_prompts(args.meta_csv, max_videos=args.max_videos)
    print(f"Loaded {len(prompts)} prompts from {args.meta_csv}")
    print()

    # ── Generate ──────────────────────────────────────────────────────
    results = []
    generator = torch.Generator(device="cuda")

    for i, entry in enumerate(prompts):
        prompt = entry["prompt"]
        name = Path(entry["filename"]).stem
        print(f"[{i+1}/{len(prompts)}] {name}: {prompt[:80]}...")

        generator.manual_seed(args.seed)
        t1 = time.time()

        try:
            output = pipe.generate_t2v(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
            )[0]

            gen_time = time.time() - t1

            out_path = os.path.join(video_dir, f"{name}_t2v.mp4")
            out_tensor = torch.from_numpy(np.array(output))
            out_tensor = (out_tensor * 255).clamp(0, 255).to(torch.uint8)
            write_video(out_path, out_tensor, fps=args.fps, video_codec="libx264",
                        options={"crf": "18"})

            results.append({
                "name": name,
                "prompt": prompt,
                "video_path": out_path,
                "gen_time": gen_time,
                "num_frames": output.shape[0],
                "status": "ok",
            })
            print(f"  {gen_time:.1f}s, {output.shape[0]} frames, saved: {out_path}")

        except Exception as e:
            results.append({
                "name": name,
                "prompt": prompt,
                "status": "error",
                "error": str(e),
            })
            print(f"  ERROR: {e}")

        del output
        torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────
    ok = [r for r in results if r["status"] == "ok"]
    summary = {
        "model": "longcat-video",
        "mode": "t2v",
        "num_frames": args.num_frames,
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "total_videos": len(results),
        "successful": len(ok),
        "avg_gen_time": float(np.mean([r["gen_time"] for r in ok])) if ok else 0.0,
        "results": results,
    }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print(f"LongCat T2V Complete: {len(ok)}/{len(results)} successful")
    if ok:
        print(f"Avg gen time: {summary['avg_gen_time']:.1f}s")
    print(f"Results: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
