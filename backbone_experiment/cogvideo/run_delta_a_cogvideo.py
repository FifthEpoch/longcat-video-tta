#!/usr/bin/env python3
"""
Delta-A TTA for CogVideoX-5B-I2V.

Adds a single learnable delta vector to the timestep embedding of the
CogVideoX transformer. Only the delta (512 parameters) is trained per video;
all transformer weights remain frozen.

CogVideoX uses time_embed_dim=512. The delta is injected via a forward hook
on the time_embedding module so the full pipeline sees the adaptation during
generation.

Usage:
    python run_delta_a_cogvideo.py \\
        --model-path THUDM/CogVideoX-5b-I2V \\
        --data-dir /path/to/dataset \\
        --output-dir results/cogvideo_delta_a \\
        --delta-steps 20 --delta-lr 1e-3
"""

import argparse
import copy
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_cogvideo import (
    DeltaAWrapperCogVideo,
    load_cogvideo_components,
    load_video_frames,
    encode_video_cogvideo,
    encode_prompt_cogvideo,
    compute_flow_matching_loss_cogvideo,
    generate_video_cogvideo,
    save_results,
    save_video_from_numpy,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    load_panda70m_video_list,
    evaluate_generation_metrics,
)


# ============================================================================
# Optimization loop
# ============================================================================

def optimize_delta_a_cogvideo(
    wrapper: DeltaAWrapperCogVideo,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_steps: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict:
    """Optimize the delta vector using flow-matching loss.

    Only the 512-dimensional delta vector is updated. The transformer
    weights remain frozen.
    """
    optimizer = AdamW(
        [wrapper.delta], lr=lr, betas=(0.9, 0.999), eps=1e-15
    )

    wrapper.train()
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        loss = compute_flow_matching_loss_cogvideo(
            transformer=wrapper,
            latents=latents,
            prompt_embeds=prompt_embeds,
            device=device,
            dtype=dtype,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([wrapper.delta], 1.0)
        optimizer.step()

        losses.append(loss.item())

    return {
        "losses": losses,
        "delta_norm": wrapper.delta.detach().norm().item(),
    }


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Delta-A TTA for CogVideoX-5B-I2V"
    )

    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--meta-path", type=str, default=None,
                        help="CSV metadata for Panda-70M videos")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--restart", action="store_true")

    parser.add_argument("--delta-steps", type=int, default=20)
    parser.add_argument("--delta-lr", type=float, default=1e-3)

    parser.add_argument("--num-cond-frames", type=int, default=1,
                        help="Number of conditioning frames (CogVideoX I2V uses 1)")
    parser.add_argument("--num-train-frames", type=int, default=49,
                        help="Number of frames to load for TTA training")
    parser.add_argument("--num-gen-frames", type=int, default=49,
                        help="Number of frames to generate")
    parser.add_argument("--gen-start-frame", type=int, default=49,
                        help="Anchor frame for evaluation; generation starts here")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--skip-generation", action="store_true")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = 0
    all_results = []
    if ckpt and not args.restart:
        start_idx = ckpt.get("next_idx", 0)
        all_results = ckpt.get("results", [])

    print("=" * 70)
    print("Delta-A TTA for CogVideoX-5B-I2V")
    print("=" * 70)
    print(f"Model path     : {args.model_path}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Delta steps    : {args.delta_steps}")
    print(f"Delta LR       : {args.delta_lr}")
    print(f"Resume from idx: {start_idx}")
    print("=" * 70)

    # Load model
    print("\nLoading CogVideoX-5B-I2V components...")
    components = load_cogvideo_components(
        args.model_path, device=args.device, dtype=torch.bfloat16
    )
    transformer = components["transformer"]
    vae = components["vae"]
    pipe = components["pipe"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    # Enable gradient checkpointing
    transformer.enable_gradient_checkpointing()
    print("Gradient checkpointing: ENABLED")

    # Determine time_embed_dim from model config
    time_embed_dim = getattr(transformer.config, "time_embed_dim", 512)
    print(f"time_embed_dim : {time_embed_dim}")

    total_params = sum(p.numel() for p in transformer.parameters())
    print(f"Total transformer params : {total_params:,}")
    print(f"Delta-A trainable params : {time_embed_dim}")

    # Save experiment config
    exp_config = {
        "method": "delta_a",
        "backbone": "CogVideoX-5B-I2V",
        "training": {
            "delta_steps": args.delta_steps,
            "delta_lr": args.delta_lr,
            "time_embed_dim": time_embed_dim,
            "trainable_params": time_embed_dim,
            "total_params": total_params,
        },
        "generation": {
            "num_cond_frames": args.num_cond_frames,
            "num_gen_frames": args.num_gen_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "resolution": "720x480",
        },
        "seed": args.seed,
        "max_videos": args.max_videos,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    # Load video list
    videos = load_panda70m_video_list(
        args.data_dir, meta_path=args.meta_path,
        max_videos=args.max_videos, seed=args.seed,
    )
    print(f"\nTotal videos: {len(videos)}")

    # Move transformer to GPU
    transformer = transformer.to(args.device)

    videos_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    print(f"\nProcessing {len(videos) - start_idx} videos...\n")

    for idx, entry in enumerate(tqdm(videos, desc="Delta-A (CogVideo)")):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            # Load frames (720x480 for CogVideoX)
            pixel_frames = load_video_frames(
                video_path, args.num_train_frames,
                height=480, width=720,
                start_frame=0,
            ).to(args.device, torch.bfloat16)

            # Encode to latents
            latents = encode_video_cogvideo(vae, pixel_frames, device=args.device)
            print(f"  Latent shape: {list(latents.shape)}")

            # Encode text
            prompt_embeds, prompt_mask = encode_prompt_cogvideo(
                tokenizer, text_encoder, caption,
                device=args.device, dtype=torch.bfloat16,
            )

            # Offload VAE + text encoder to CPU during training
            vae.to("cpu")
            text_encoder.to("cpu")
            torch.cuda.empty_cache()

            # Create fresh wrapper with zeroed delta
            wrapper = DeltaAWrapperCogVideo(
                transformer, time_embed_dim=time_embed_dim
            ).to(args.device)

            # Optimize
            t0 = time.time()
            opt_result = optimize_delta_a_cogvideo(
                wrapper=wrapper,
                latents=latents,
                prompt_embeds=prompt_embeds,
                num_steps=args.delta_steps,
                lr=args.delta_lr,
                device=args.device,
                dtype=torch.bfloat16,
            )
            train_time = time.time() - t0

            result = {
                "idx": idx,
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "train_time": train_time,
                "final_loss": (
                    opt_result["losses"][-1] if opt_result["losses"] else None
                ),
                "delta_norm": opt_result["delta_norm"],
                "success": True,
            }

            print(f"  Train: {train_time:.1f}s, "
                  f"Loss: {result['final_loss']:.4f}, "
                  f"Delta norm: {result['delta_norm']:.4f}")

            # Bring VAE + text encoder back for generation
            vae.to(args.device)
            text_encoder.to(args.device)

            gen_time = 0.0
            if not args.skip_generation:
                from PIL import Image

                # Extract conditioning frame
                pf = pixel_frames.squeeze(0)  # [C, T, H, W]
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_frame_np = (
                    pf[:, 0].permute(1, 2, 0).float().cpu().numpy() * 255
                ).astype(np.uint8)
                cond_image = Image.fromarray(cond_frame_np)

                # Install delta hook for generation
                wrapper.apply_hook()
                try:
                    gen_start = time.time()
                    gen_frames = generate_video_cogvideo(
                        pipe=pipe,
                        image=cond_image,
                        prompt=caption,
                        num_frames=args.num_gen_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed + idx,
                        device=args.device,
                    )
                    gen_time = time.time() - gen_start

                    output_path = os.path.join(
                        videos_dir, f"{video_name}_delta_a.mp4"
                    )
                    save_video_from_numpy(gen_frames, output_path, fps=8)
                    result["output_path"] = output_path
                    result["gen_time"] = gen_time

                    num_gen = args.num_gen_frames - args.num_cond_frames
                    metrics = evaluate_generation_metrics(
                        gen_output=gen_frames,
                        video_path=video_path,
                        num_cond_frames=args.num_cond_frames,
                        num_gen_frames=num_gen,
                        gen_start_frame=args.gen_start_frame,
                        device=args.device,
                    )
                    result.update(metrics)
                    print(f"  Gen: {gen_time:.1f}s, "
                          f"PSNR={metrics['psnr']:.2f}, "
                          f"SSIM={metrics['ssim']:.4f}, "
                          f"LPIPS={metrics['lpips']:.4f}")
                finally:
                    wrapper.remove_hook()

            result["total_time"] = train_time + gen_time
            all_results.append(result)

            del wrapper, latents, pixel_frames, prompt_embeds, prompt_mask
            torch_gc()

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "idx": idx,
                "video_name": video_name,
                "video_path": video_path,
                "error": str(e),
                "success": False,
            })
            vae.to(args.device)
            text_encoder.to(args.device)
            torch_gc()

        save_checkpoint(
            {"next_idx": idx + 1, "results": all_results}, ckpt_path
        )

    # Final summary
    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "delta_a",
        "backbone": "CogVideoX-5B-I2V",
        "delta_steps": args.delta_steps,
        "delta_lr": args.delta_lr,
        "trainable_params": time_embed_dim,
        "total_params": total_params,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "num_failed": len(all_results) - len(successful),
        "avg_train_time": (
            np.mean([r["train_time"] for r in successful]) if successful else 0
        ),
        "avg_final_loss": (
            np.mean([r["final_loss"] for r in successful
                     if r.get("final_loss") is not None])
            if successful else 0
        ),
        "avg_delta_norm": (
            np.mean([r["delta_norm"] for r in successful
                     if r.get("delta_norm") is not None])
            if successful else 0
        ),
        "results": all_results,
    }

    if successful:
        psnr_vals = [r["psnr"] for r in successful if "psnr" in r]
        ssim_vals = [r["ssim"] for r in successful if "ssim" in r]
        lpips_vals = [r["lpips"] for r in successful if "lpips" in r]
        if psnr_vals:
            summary["avg_psnr"] = float(np.mean(psnr_vals))
        if ssim_vals:
            summary["avg_ssim"] = float(np.mean(ssim_vals))
        if lpips_vals:
            summary["avg_lpips"] = float(np.mean(lpips_vals))

    save_results(summary, os.path.join(args.output_dir, "summary.json"))

    print("\n" + "=" * 70)
    print("Delta-A TTA for CogVideoX-5B-I2V Complete!")
    print("=" * 70)
    print(f"Successful: {len(successful)}/{len(all_results)}")
    if successful:
        print(f"Avg train time : {summary['avg_train_time']:.1f}s")
        print(f"Avg final loss : {summary['avg_final_loss']:.4f}")
        print(f"Avg delta norm : {summary['avg_delta_norm']:.4f}")
        if "avg_psnr" in summary:
            print(f"Avg PSNR       : {summary['avg_psnr']:.2f}")
        if "avg_ssim" in summary:
            print(f"Avg SSIM       : {summary['avg_ssim']:.4f}")
        if "avg_lpips" in summary:
            print(f"Avg LPIPS      : {summary['avg_lpips']:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
