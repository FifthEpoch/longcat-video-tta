#!/usr/bin/env python3
"""
Full-model Test-Time Adaptation (TTA) for CogVideoX-5B-I2V.

Fine-tunes ALL transformer parameters on conditioning frames for each video,
then generates continuations. Weights are reset between videos.

Key differences from LongCat-Video full TTA:
  - CogVideoX uses CogVideoXTransformer3DModel (not a custom DiT)
  - Resolution: 720x480 (not 832x480); Panda-70M videos are resized
  - VAE: AutoencoderKLCogVideoX (temporal 4x, spatial 8x)
  - Text encoder: T5 (max_length=226)
  - I2V pipeline: single conditioning frame (PIL Image)
  - Uses SGD optimizer (AdamW states too large for single GPU)
  - Gradient checkpointing via transformer.enable_gradient_checkpointing()

Usage:
    python run_full_tta_cogvideo.py \\
        --model-path THUDM/CogVideoX-5b-I2V \\
        --data-dir /path/to/dataset \\
        --output-dir results/cogvideo_full_tta \\
        --learning-rate 1e-5 --num-steps 10
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
from torch.optim import SGD, AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_cogvideo import (
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
# Full-model TTA training loop
# ============================================================================

def finetune_full_cogvideo(
    transformer: nn.Module,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    num_steps: int = 10,
    lr: float = 1e-5,
    warmup_steps: int = 2,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    optimizer_type: str = "sgd",
) -> Dict:
    """Fine-tune all CogVideoX transformer parameters using flow-matching loss.

    Parameters
    ----------
    optimizer_type : 'sgd' (default, no state — fits on single GPU) or 'adamw'
    """
    params = [p for p in transformer.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found. Did you unfreeze the model?")

    if optimizer_type == "adamw":
        optimizer = AdamW(params, lr=lr, betas=(0.9, 0.999),
                          weight_decay=weight_decay, eps=1e-8)
    else:
        optimizer = SGD(params, lr=lr, momentum=0.0, weight_decay=weight_decay)

    transformer.train()
    losses = []
    train_start = time.time()

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        if step < warmup_steps and warmup_steps > 0:
            warmup_lr = lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        loss = compute_flow_matching_loss_cogvideo(
            transformer=transformer,
            latents=latents,
            prompt_embeds=prompt_embeds,
            device=device,
            dtype=dtype,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()

        losses.append(loss.item())
        del loss

        if step % 5 == 0:
            torch.cuda.empty_cache()

    train_time = time.time() - train_start
    transformer.eval()
    torch.cuda.empty_cache()

    return {"losses": losses, "train_time": train_time}


def reset_transformer_weights(
    transformer: nn.Module,
    base_state: Dict[str, torch.Tensor],
):
    """Reset transformer to its pretrained weights."""
    with torch.no_grad():
        for name, param in transformer.named_parameters():
            if name in base_state:
                param.data.copy_(base_state[name].to(param.device))


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full-model TTA for CogVideoX-5B-I2V"
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

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adamw"])

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
    print("Full-Model TTA for CogVideoX-5B-I2V")
    print("=" * 70)
    print(f"Model path     : {args.model_path}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Learning rate  : {args.learning_rate}")
    print(f"Num steps      : {args.num_steps}")
    print(f"Optimizer      : {args.optimizer}")
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

    # Unfreeze all transformer parameters
    for p in transformer.parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in transformer.parameters())
    trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Total transformer params : {total_params:,}")
    print(f"Trainable params         : {trainable_params:,}")

    # Save base state for per-video reset (CPU to save GPU mem)
    print("Saving base state for per-video reset...")
    base_state = {
        k: v.detach().cpu().clone()
        for k, v in transformer.state_dict().items()
    }
    base_mem_gb = sum(v.numel() * v.element_size() for v in base_state.values()) / 1e9
    print(f"Base state size: {base_mem_gb:.2f} GB")

    # Save experiment config
    exp_config = {
        "method": "full_tta",
        "backbone": "CogVideoX-5B-I2V",
        "training": {
            "learning_rate": args.learning_rate,
            "num_steps": args.num_steps,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "optimizer": args.optimizer,
            "total_params": total_params,
            "trainable_params": trainable_params,
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

    for idx, entry in enumerate(tqdm(videos, desc="Full TTA (CogVideo)")):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            # Reset transformer to base weights
            reset_transformer_weights(transformer, base_state)

            # Load frames for TTA training (720x480 for CogVideoX)
            pixel_frames = load_video_frames(
                video_path, args.num_train_frames,
                height=480, width=720,
                start_frame=0,
            ).to(args.device, torch.bfloat16)

            # Encode video to latents
            latents = encode_video_cogvideo(vae, pixel_frames, device=args.device)
            print(f"  Latent shape: {list(latents.shape)}")

            # Encode text prompt
            prompt_embeds, prompt_mask = encode_prompt_cogvideo(
                tokenizer, text_encoder, caption,
                device=args.device, dtype=torch.bfloat16,
            )

            # Offload VAE + text encoder to CPU during training
            vae.to("cpu")
            text_encoder.to("cpu")
            torch.cuda.empty_cache()

            # Fine-tune
            train_result = finetune_full_cogvideo(
                transformer=transformer,
                latents=latents,
                prompt_embeds=prompt_embeds,
                num_steps=args.num_steps,
                lr=args.learning_rate,
                warmup_steps=args.warmup_steps,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                device=args.device,
                dtype=torch.bfloat16,
                optimizer_type=args.optimizer,
            )

            result = {
                "idx": idx,
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "train_time": train_result["train_time"],
                "final_loss": (
                    train_result["losses"][-1] if train_result["losses"] else None
                ),
                "avg_loss": (
                    sum(train_result["losses"]) / len(train_result["losses"])
                    if train_result["losses"] else None
                ),
                "num_train_steps": len(train_result["losses"]),
                "success": True,
            }

            print(f"  Train: {train_result['train_time']:.1f}s, "
                  f"Loss: {result['final_loss']:.4f}")

            # Bring VAE + text encoder back for generation
            vae.to(args.device)
            text_encoder.to(args.device)

            gen_time = 0.0
            if not args.skip_generation:
                from PIL import Image

                # Extract conditioning frame (first frame, 720x480)
                pf = pixel_frames.squeeze(0)  # [C, T, H, W]
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_frame_np = (
                    pf[:, 0].permute(1, 2, 0).float().cpu().numpy() * 255
                ).astype(np.uint8)
                cond_image = Image.fromarray(cond_frame_np)

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

                output_path = os.path.join(videos_dir, f"{video_name}_full.mp4")
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

            result["total_time"] = train_result["train_time"] + gen_time
            all_results.append(result)

            del latents, pixel_frames, prompt_embeds, prompt_mask
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
            # Ensure components are back on the right device after error
            vae.to(args.device)
            text_encoder.to(args.device)
            torch_gc()

        save_checkpoint(
            {"next_idx": idx + 1, "results": all_results}, ckpt_path
        )

    # Final summary
    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "full_tta",
        "backbone": "CogVideoX-5B-I2V",
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "optimizer": args.optimizer,
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
    print("Full-Model TTA for CogVideoX-5B-I2V Complete!")
    print("=" * 70)
    print(f"Successful: {len(successful)}/{len(all_results)}")
    if successful:
        print(f"Avg train time : {summary['avg_train_time']:.1f}s")
        print(f"Avg final loss : {summary['avg_final_loss']:.4f}")
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
