#!/usr/bin/env python3
"""
Delta-A TTA for Open-Sora v2.0: add a single learnable δ vector to the
timestep embedding.

In Open-Sora v2.0 the timestep embedding `vec` has shape [B, hidden_size]
(default hidden_size=3072). Delta A adds a learnable vector δ ∈ R^{hidden_size}
to this embedding via a forward hook on ``time_in``:

    vec' = time_in(timestep_embedding(t, 256)) + δ

This is the simplest δ-TTA method — one vector per video, discarded after.

Usage:
    python run_delta_a_opensora.py \\
        --config-path configs/diffusion/inference/256px.py \\
        --data-dir /path/to/dataset \\
        --output-dir results/delta_a_opensora \\
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

from common_opensora import (
    DeltaAWrapperOpenSora,
    load_opensora_components,
    load_video_frames,
    encode_video_opensora,
    decode_latents_opensora,
    encode_prompt_opensora,
    patchify_latents,
    compute_flow_matching_loss_opensora,
    compute_flow_matching_loss_conditioned_opensora,
    evaluate_generation_metrics,
    save_video_from_numpy,
    save_results,
    load_checkpoint,
    save_checkpoint,
    load_ucf101_video_list,
    torch_gc,
)


# ============================================================================
# Delta-A optimization loop
# ============================================================================

def optimize_delta_a_opensora(
    wrapper: DeltaAWrapperOpenSora,
    cond_latents: torch.Tensor,
    train_latents: torch.Tensor,
    text_inputs: Dict[str, torch.Tensor],
    num_steps: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    guidance: float = 7.5,
    patch_size: int = 2,
) -> Dict:
    """Optimize the delta vector using conditioning-aware flow-matching loss.

    Only the single delta vector (hidden_size parameters) is trainable.
    """
    optimizer = AdamW([wrapper.delta], lr=lr, betas=(0.9, 0.999), eps=1e-15)

    wrapper.train()
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        loss = compute_flow_matching_loss_conditioned_opensora(
            dit=wrapper,
            cond_latents=cond_latents,
            target_latents=train_latents,
            text_inputs=text_inputs,
            device=device,
            dtype=dtype,
            guidance=guidance,
            patch_size=patch_size,
            forward_fn=lambda **kw: wrapper(**kw),
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
# Latent split helper
# ============================================================================

def split_tta_latents(
    latents: torch.Tensor,
    num_context_latents: int,
) -> tuple:
    """Split latents into context and train along the temporal axis."""
    T_total = latents.shape[2]
    T_cond = min(num_context_latents, T_total - 1)

    cond = latents[:, :, :T_cond].contiguous()
    train = latents[:, :, T_cond:].contiguous()
    return cond, train


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Delta-A TTA for Open-Sora v2.0")

    parser.add_argument("--config-path", type=str, required=True,
                        help="Open-Sora v2.0 inference config")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--delta-steps", type=int, default=20)
    parser.add_argument("--delta-lr", type=float, default=1e-3)

    parser.add_argument("--num-cond-frames", type=int, default=5)
    parser.add_argument("--tta-total-frames", type=int, default=33)
    parser.add_argument("--tta-context-frames", type=int, default=5)
    parser.add_argument("--gen-start-frame", type=int, default=33)
    parser.add_argument("--num-gen-frames", type=int, default=129)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--guidance-img", type=float, default=3.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=455)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--patch-size", type=int, default=2)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0

    print("=" * 70)
    print("Delta-A TTA for Open-Sora v2.0")
    print("=" * 70)
    print(f"Config path    : {args.config_path}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Delta steps    : {args.delta_steps}")
    print(f"Delta LR       : {args.delta_lr}")
    print(f"Resume from idx: {start_idx}")
    print("=" * 70)

    print("\nLoading Open-Sora v2.0 model components...")
    components = load_opensora_components(
        args.config_path, device=args.device, dtype=torch.bfloat16,
    )
    dit = components["dit"]
    vae = components["vae"]
    t5 = components["t5"]
    clip = components["clip"]
    cfg = components["config"]

    hidden_size = dit.hidden_size
    print(f"MMDiT hidden_size: {hidden_size}")
    print(f"Delta-A trainable params: {hidden_size:,}")

    videos = load_ucf101_video_list(args.data_dir, max_videos=args.max_videos, seed=args.seed)
    print(f"\nTotal videos: {len(videos)}")

    all_results = []
    videos_dir = os.path.join(args.output_dir, "videos")
    if not args.skip_generation:
        os.makedirs(videos_dir, exist_ok=True)

    vae_time_ratio = getattr(vae, "time_compression_ratio", 4)

    for idx, entry in enumerate(videos):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            tta_start_frame = max(0, args.gen_start_frame - args.tta_total_frames)
            pixel_frames = load_video_frames(
                video_path, args.tta_total_frames,
                height=args.height, width=args.width,
                start_frame=tta_start_frame,
            ).to(args.device, torch.bfloat16)

            all_latents = encode_video_opensora(vae, pixel_frames)

            num_ctx_lat = max(1, (args.tta_context_frames - 1) // vae_time_ratio + 1)
            cond_latents, train_latents = split_tta_latents(all_latents, num_ctx_lat)
            print(f"  Latent split: cond={cond_latents.shape[2]}, train={train_latents.shape[2]}")

            lat_T = all_latents.shape[2]
            lat_H = all_latents.shape[3]
            lat_W = all_latents.shape[4]
            num_img_tokens = lat_T * (lat_H // args.patch_size) * (lat_W // args.patch_size)

            text_inputs = encode_prompt_opensora(
                t5, clip, caption,
                device=args.device, dtype=torch.bfloat16,
                num_img_tokens=num_img_tokens,
            )

            wrapper = DeltaAWrapperOpenSora(dit, hidden_size=hidden_size).to(args.device)

            # Offload VAE + text encoders to CPU during training
            vae.to("cpu")
            t5.to("cpu")
            clip.to("cpu")
            torch.cuda.empty_cache()

            t0 = time.time()
            opt_result = optimize_delta_a_opensora(
                wrapper=wrapper,
                cond_latents=cond_latents,
                train_latents=train_latents,
                text_inputs=text_inputs,
                num_steps=args.delta_steps,
                lr=args.delta_lr,
                device=args.device,
                dtype=torch.bfloat16,
                guidance=args.guidance_scale,
                patch_size=args.patch_size,
            )
            train_time = time.time() - t0

            result = {
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "train_time": train_time,
                "final_loss": opt_result["losses"][-1] if opt_result["losses"] else None,
                "delta_norm": opt_result["delta_norm"],
                "success": True,
            }

            print(f"  Train time: {train_time:.1f}s, "
                  f"Final loss: {result['final_loss']:.4f}, "
                  f"Delta norm: {result['delta_norm']:.4f}")

            # Bring VAE + text encoders back
            vae.to(args.device)
            t5.to(args.device)
            clip.to(args.device)

            gen_time = 0.0
            if not args.skip_generation:
                from run_full_tta_opensora import generate_video_opensora

                gen_cond_start = max(0, args.gen_start_frame - args.num_cond_frames)
                gen_pixel = load_video_frames(
                    video_path, args.num_cond_frames,
                    height=args.height, width=args.width,
                    start_frame=gen_cond_start,
                ).to(args.device, torch.bfloat16)

                ref_latents = encode_video_opensora(vae, gen_pixel)

                wrapper.apply_hook()
                try:
                    gen_start = time.time()
                    gen_frames = generate_video_opensora(
                        dit=dit, vae=vae, t5=t5, clip=clip,
                        prompt=caption,
                        ref_latents=ref_latents,
                        num_frames=args.num_gen_frames,
                        num_steps=args.num_inference_steps,
                        guidance=args.guidance_scale,
                        guidance_img=args.guidance_img,
                        height=args.height, width=args.width,
                        seed=args.seed + idx,
                        device=args.device, dtype=torch.bfloat16,
                        patch_size=args.patch_size,
                    )
                    gen_time = time.time() - gen_start

                    output_path = os.path.join(videos_dir, f"{video_name}_delta_a.mp4")
                    save_video_from_numpy(gen_frames, output_path, fps=24)
                    result["output_path"] = output_path
                    result["gen_time"] = gen_time

                    num_gen = min(gen_frames.shape[0] - args.num_cond_frames, 80)
                    metrics = evaluate_generation_metrics(
                        gen_frames=gen_frames,
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

                del gen_pixel, ref_latents

            result["total_time"] = train_time + gen_time
            all_results.append(result)

            del wrapper, all_latents, cond_latents, train_latents
            del pixel_frames, text_inputs
            torch_gc()

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "video_name": video_name,
                "video_path": video_path,
                "error": str(e),
                "success": False,
            })
            # Ensure models are back on GPU for the next video
            vae.to(args.device)
            t5.to(args.device)
            clip.to(args.device)
            torch_gc()

        save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)

    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "delta_a_opensora",
        "delta_steps": args.delta_steps,
        "delta_lr": args.delta_lr,
        "hidden_size": hidden_size,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "avg_train_time": float(np.mean([r.get("train_time", 0) for r in successful])) if successful else 0,
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))

    print(f"\nResults saved to {args.output_dir}/summary.json")
    if successful:
        print(f"Avg train time: {summary['avg_train_time']:.1f}s")


if __name__ == "__main__":
    main()
