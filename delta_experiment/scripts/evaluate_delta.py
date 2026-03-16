#!/usr/bin/env python3
"""
Evaluation script for δ-TTA experiments on LongCat-Video.

Given a results directory (from run_delta_a/b/c.py) and the dataset,
this script:
1. Loads the pre-trained model
2. For each video in the results:
   a. Encodes conditioning frames → latents
   b. Generates a video continuation (baseline or with δ applied)
   c. Computes PSNR, SSIM, LPIPS against ground-truth frames
3. Writes per-video metrics and a summary to disk.

Usage:
    python evaluate_delta.py \\
        --checkpoint-dir /path/to/longcat-video-checkpoints \\
        --data-dir /path/to/dataset \\
        --results-dir results/delta_a \\
        --output-dir results/delta_a/eval \\
        --mode baseline  # or "adapted"

When mode=baseline, generation uses the unmodified model.
When mode=adapted, the script is a placeholder – actual adaptation
must be done in the generation pipeline. For now, the evaluation
focuses on baseline to establish a comparison point.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    decode_latents,
    generate_video_continuation,
    compute_psnr,
    compute_ssim_batch,
    compute_lpips_batch,
    save_results,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
)


def evaluate_single_video(
    components: Dict,
    video_path: str,
    caption: str,
    num_cond_frames: int = 13,
    num_frames: int = 93,
    gen_start_frame: int = 32,
    resolution: str = "480p",
    seed: int = 42,
    device: str = "cuda",
) -> Dict:
    """Evaluate a single video: generate continuation and compute metrics.

    Uses anchor-based indexing:
      Conditioning = video[anchor - num_cond : anchor]
      GT           = video[anchor : anchor + num_gen]

    Returns a dict with PSNR, SSIM, LPIPS, and timing info.
    """
    vae = components["vae"]
    pipe = components["pipe"]

    num_gen = num_frames - num_cond_frames
    anchor = gen_start_frame

    # Load conditioning frames from [anchor - num_cond : anchor]
    cond_start = anchor - num_cond_frames
    cond_pixel = load_video_frames(
        video_path, num_cond_frames, height=480, width=832,
        start_frame=cond_start,
    ).to(device, torch.bfloat16)

    # Load GT future frames from [anchor : anchor + num_gen]
    gt_future_pixel = load_video_frames(
        video_path, num_gen, height=480, width=832,
        start_frame=anchor,
    ).to(device, torch.bfloat16)

    if gt_future_pixel.shape[2] == 0:
        return {"error": "No GT future frames available"}

    # Convert conditioning frames to PIL for pipeline
    from PIL import Image
    cond_np = ((cond_pixel[0].permute(1, 2, 3, 0).float().cpu().numpy() + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    cond_pil = [Image.fromarray(cond_np[i]) for i in range(cond_np.shape[0])]

    # Generate
    t0 = time.time()
    try:
        gen_output = generate_video_continuation(
            pipe=pipe,
            video_frames=cond_pil,
            prompt=caption,
            num_cond_frames=num_cond_frames,
            num_frames=num_frames,
            seed=seed,
            resolution=resolution,
            device=device,
        )
        gen_time = time.time() - t0
    except Exception as e:
        return {"error": f"Generation failed: {e}", "gen_time": time.time() - t0}

    # gen_output is np array [N, H, W, 3] in [0, 1]
    gen_frames = gen_output[num_cond_frames:]  # skip conditioning
    if len(gen_frames) == 0:
        return {"error": "No generated future frames", "gen_time": gen_time}

    # Convert GT future to [0, 1] for metrics
    gt_future_01 = ((gt_future_pixel[0].float() + 1.0) / 2.0).clamp(0, 1)  # [C, T, H, W]
    gt_future_01 = gt_future_01.permute(1, 0, 2, 3)  # [T, C, H, W]

    # Convert generated to tensor [T, C, H, W]
    gen_t = torch.from_numpy(gen_frames).permute(0, 3, 1, 2).float().to(device)  # [T, C, H, W]

    # Match temporal length
    min_t = min(gt_future_01.shape[0], gen_t.shape[0])
    gt_future_01 = gt_future_01[:min_t]
    gen_t = gen_t[:min_t]

    # Resize if needed
    if gt_future_01.shape[2:] != gen_t.shape[2:]:
        gen_t = torch.nn.functional.interpolate(
            gen_t, size=gt_future_01.shape[2:], mode="bilinear", align_corners=False
        )

    # Compute metrics
    psnr = compute_psnr(gen_t, gt_future_01)
    ssim = compute_ssim_batch(gen_t, gt_future_01)
    lpips_val = compute_lpips_batch(gen_t, gt_future_01)

    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips_val,
        "gen_time": gen_time,
        "num_gt_frames": min_t,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate δ-TTA results")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to LongCat-Video checkpoints")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to video dataset")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Path to TTA results dir (with summary.json)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to write evaluation results")
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline"],
                        help="Evaluation mode")
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--num-cond-frames", type=int, default=13)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--gen-start-frame", type=int, default=32,
                        help="Fixed anchor frame where generation starts. "
                             "Cond = video[anchor-cond : anchor], "
                             "GT = video[anchor : anchor+gen].")
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support
    ckpt_path = os.path.join(args.output_dir, "eval_checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0

    print("=" * 70)
    print(f"δ-TTA Evaluation ({args.mode})")
    print("=" * 70)

    # Load model
    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16
    )

    # Get video list
    if args.results_dir and os.path.exists(os.path.join(args.results_dir, "summary.json")):
        with open(os.path.join(args.results_dir, "summary.json")) as f:
            tta_summary = json.load(f)
        videos = [
            {
                "video_path": r["video_path"],
                "caption": r["caption"],
                "video_name": r["video_name"],
            }
            for r in tta_summary.get("results", [])
            if "error" not in r
        ]
    else:
        from common import load_ucf101_video_list
        videos = load_ucf101_video_list(
            args.data_dir, max_videos=args.max_videos, seed=args.seed
        )
        for v in videos:
            v["video_name"] = Path(v["video_path"]).stem

    videos = videos[:args.max_videos]
    print(f"Evaluating {len(videos)} videos...")

    all_metrics = []

    for idx, entry in enumerate(videos):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry.get("caption", "")
        video_name = entry.get("video_name", Path(video_path).stem)

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}")

        try:
            metrics = evaluate_single_video(
                components=components,
                video_path=video_path,
                caption=caption,
                num_cond_frames=args.num_cond_frames,
                num_frames=args.num_frames,
                gen_start_frame=args.gen_start_frame,
                resolution=args.resolution,
                seed=args.seed,
                device=args.device,
            )
            metrics["video_name"] = video_name
            metrics["video_path"] = video_path

            if "psnr" in metrics:
                print(f"  PSNR: {metrics['psnr']:.2f}, "
                      f"SSIM: {metrics['ssim']:.4f}, "
                      f"LPIPS: {metrics['lpips']:.4f}, "
                      f"Gen time: {metrics['gen_time']:.1f}s")
            else:
                print(f"  {metrics.get('error', 'Unknown error')}")

            all_metrics.append(metrics)
            torch_gc()

        except Exception as e:
            print(f"  ERROR: {e}")
            all_metrics.append({
                "video_name": video_name,
                "video_path": video_path,
                "error": str(e),
            })

        save_checkpoint({"next_idx": idx + 1}, ckpt_path)

    # Compute summary statistics
    valid = [m for m in all_metrics if "psnr" in m]
    summary = {
        "mode": args.mode,
        "num_videos": len(all_metrics),
        "num_valid": len(valid),
        "avg_psnr": float(np.mean([m["psnr"] for m in valid])) if valid else 0,
        "std_psnr": float(np.std([m["psnr"] for m in valid])) if valid else 0,
        "avg_ssim": float(np.mean([m["ssim"] for m in valid])) if valid else 0,
        "std_ssim": float(np.std([m["ssim"] for m in valid])) if valid else 0,
        "avg_lpips": float(np.mean([m["lpips"] for m in valid])) if valid else 0,
        "std_lpips": float(np.std([m["lpips"] for m in valid])) if valid else 0,
        "avg_gen_time": float(np.mean([m["gen_time"] for m in valid])) if valid else 0,
        "per_video": all_metrics,
    }

    save_results(summary, os.path.join(args.output_dir, "eval_summary.json"))

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"  Valid videos : {summary['num_valid']}/{summary['num_videos']}")
    print(f"  Avg PSNR     : {summary['avg_psnr']:.2f} ± {summary['std_psnr']:.2f}")
    print(f"  Avg SSIM     : {summary['avg_ssim']:.4f} ± {summary['std_ssim']:.4f}")
    print(f"  Avg LPIPS    : {summary['avg_lpips']:.4f} ± {summary['std_lpips']:.4f}")
    print(f"  Avg Gen time : {summary['avg_gen_time']:.1f}s")
    print(f"\nSaved to: {args.output_dir}/eval_summary.json")


if __name__ == "__main__":
    main()
