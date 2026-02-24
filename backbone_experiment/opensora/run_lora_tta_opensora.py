#!/usr/bin/env python3
"""
LoRA Test-Time Adaptation (TTA) for Open-Sora v2.0.

Injects low-rank adapters into the MMDiT and fine-tunes only the LoRA
parameters on conditioning frames for each video, then generates
continuations using Open-Sora's denoising pipeline.
LoRA weights are re-initialized between videos (no base state save needed).

Key features:
- Injects LoRA into MMDiT attention (qkv + proj by default)
- Only trains LoRA parameters (~0.1-1% of total)
- Uses AdamW optimizer (LoRA params are small enough)
- Between videos, resets LoRA weights instead of restoring full state dict
- Flow-matching loss on conditioning latents
- Gradient checkpointing via auto_grad_checkpoint
- VAE / text encoder CPU offloading during training
- Checkpoints progress for resumability

Usage:
    python run_lora_tta_opensora.py \\
        --config-path configs/diffusion/inference/256px.py \\
        --data-dir /path/to/dataset \\
        --output-dir results/lora_tta_opensora \\
        --learning-rate 1e-4 --num-steps 10 --lora-rank 4
"""

import argparse
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

from lora_layers import (
    inject_lora_into_mmdit,
    get_lora_parameters,
    count_lora_parameters,
    reset_lora_weights,
)


# ============================================================================
# LoRA TTA training loop
# ============================================================================

def finetune_lora_opensora(
    dit: nn.Module,
    cond_latents: torch.Tensor,
    train_latents: torch.Tensor,
    text_inputs: Dict[str, torch.Tensor],
    num_steps: int = 10,
    lr: float = 1e-4,
    warmup_steps: int = 2,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    guidance: float = 7.5,
    patch_size: int = 2,
) -> Dict:
    """Fine-tune LoRA parameters using conditioning-aware flow-matching loss.

    Returns dict with keys: losses, train_time.
    """
    params = get_lora_parameters(dit)
    if not params:
        raise ValueError("No LoRA parameters found. Did you inject LoRA?")

    optimizer = AdamW(params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    dit.train()
    losses = []
    train_start = time.time()

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        if step < warmup_steps and warmup_steps > 0:
            warmup_lr = lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        loss = compute_flow_matching_loss_conditioned_opensora(
            dit=dit,
            cond_latents=cond_latents,
            target_latents=train_latents,
            text_inputs=text_inputs,
            device=device,
            dtype=dtype,
            guidance=guidance,
            patch_size=patch_size,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()

        losses.append(loss.item())
        del loss

        if step % 5 == 0:
            torch.cuda.empty_cache()

    train_time = time.time() - train_start
    dit.eval()
    torch.cuda.empty_cache()

    return {"losses": losses, "train_time": train_time}


# ============================================================================
# Denoising-based generation (standalone, no distributed)
# ============================================================================

def generate_video_opensora(
    dit: nn.Module,
    vae: nn.Module,
    t5,
    clip,
    prompt: str,
    ref_latents: torch.Tensor,
    num_frames: int = 129,
    num_steps: int = 50,
    guidance: float = 7.5,
    guidance_img: float = 3.0,
    height: int = 256,
    width: int = 455,
    seed: int = 42,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    patch_size: int = 2,
) -> np.ndarray:
    """Generate video using the Open-Sora v2.0 denoising loop.

    LoRA is injected in-place, so the model's forward path automatically
    uses LoRA-adapted weights.

    Returns np.ndarray [N, H, W, 3] in [0, 1].
    """
    from opensora.utils.sampling import (
        get_noise, get_schedule, prepare, pack, unpack,
    )
    from opensora.utils.inference import prepare_inference_condition

    temporal_reduction = 4
    num_lat_frames = (num_frames - 1) // temporal_reduction + 1

    z = get_noise(
        1, height, width, num_lat_frames, device, dtype, seed,
        patch_size=patch_size,
        channel=dit.config.in_channels // (patch_size ** 2),
    )

    timesteps = get_schedule(
        num_steps,
        (z.shape[-1] * z.shape[-2]) // patch_size ** 2,
        num_lat_frames,
        shift=True,
    )

    cond_type = "t2v"
    ref_list = [None]

    if ref_latents is not None:
        T_ref = ref_latents.shape[2]
        cond_type = "v2v_head" if T_ref > 1 else "i2v_head"
        ref_list = [[ref_latents.squeeze(0)]]

    masks, masked_ref = prepare_inference_condition(
        z, cond_type, ref_list=ref_list, causal=True,
    )

    neg_prompt = ""
    text_list = [prompt, neg_prompt, neg_prompt]

    inp = prepare(t5, clip, z, prompt=text_list, patch_size=patch_size)

    cond_packed = pack(torch.cat((masks, masked_ref), dim=1), patch_size=patch_size)
    inp["cond"] = torch.cat([cond_packed, cond_packed, torch.zeros_like(cond_packed)], dim=0)

    guidance_vec = torch.full((z.shape[0] * 3,), guidance, device=device, dtype=dtype)
    img = inp["img"]

    dit.eval()
    with torch.no_grad():
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=dtype, device=device)

            cond_x = img[: len(img) // 3]
            img_in = torch.cat([cond_x, cond_x, cond_x], dim=0)

            pred = dit(
                img=img_in,
                img_ids=inp["img_ids"],
                txt=inp["txt"],
                txt_ids=inp["txt_ids"],
                timesteps=t_vec,
                y_vec=inp["y_vec"],
                guidance=guidance_vec,
                cond=inp["cond"],
            )

            cond_pred, uncond_pred, uncond2_pred = pred.chunk(3, dim=0)
            pred_combined = uncond2_pred + guidance_img * (uncond_pred - uncond2_pred) + guidance * (cond_pred - uncond_pred)
            pred_combined = torch.cat([pred_combined] * 3, dim=0)

            img = img + (t_prev - t_curr) * pred_combined

    img = img[: len(img) // 3]
    x = unpack(img, height, width, num_lat_frames, patch_size=patch_size)

    if ref_latents is not None and cond_type == "i2v_head":
        x[0, :, :1] = ref_list[0][0][:, :1]

    video = vae.decode(x)
    video = video[:, :, :num_frames]
    video = (video.clamp(-1, 1) + 1.0) / 2.0
    video = video.squeeze(0).permute(1, 2, 3, 0).float().cpu().numpy()

    return video


# ============================================================================
# Latent split helper
# ============================================================================

def split_tta_latents(
    latents: torch.Tensor,
    num_context_latents: int,
) -> tuple:
    """Split latents into context and train along the temporal axis.

    Returns (cond_latents, train_latents).
    """
    T_total = latents.shape[2]
    T_cond = min(num_context_latents, T_total - 1)

    cond = latents[:, :, :T_cond].contiguous()
    train = latents[:, :, T_cond:].contiguous()
    return cond, train


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LoRA TTA for Open-Sora v2.0")

    parser.add_argument("--config-path", type=str, required=True,
                        help="Open-Sora v2.0 inference config (e.g. configs/diffusion/inference/256px.py)")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--restart", action="store_true")

    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=4.0)

    parser.add_argument("--num-cond-frames", type=int, default=5,
                        help="Pixel frames used as conditioning for generation")
    parser.add_argument("--tta-total-frames", type=int, default=33,
                        help="Total pixel frames before anchor to load for TTA")
    parser.add_argument("--tta-context-frames", type=int, default=5,
                        help="Leading frames treated as clean context (timestep=0)")
    parser.add_argument("--gen-start-frame", type=int, default=33,
                        help="Anchor frame index in source video")
    parser.add_argument("--num-gen-frames", type=int, default=129,
                        help="Number of frames to generate")
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
    start_idx = 0
    all_results = []
    if ckpt and not args.restart:
        start_idx = ckpt.get("next_idx", 0)
        all_results = ckpt.get("results", [])

    print("=" * 70)
    print("LoRA TTA for Open-Sora v2.0")
    print("=" * 70)
    print(f"Config path    : {args.config_path}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Learning rate  : {args.learning_rate}")
    print(f"Num steps      : {args.num_steps}")
    print(f"LoRA rank      : {args.lora_rank}")
    print(f"LoRA alpha     : {args.lora_alpha}")
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

    print(f"Injecting LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    inject_lora_into_mmdit(
        dit, rank=args.lora_rank, alpha=args.lora_alpha,
        dropout=0.0, target_modules=["qkv", "proj"],
    )

    param_info = count_lora_parameters(dit)
    total_params = param_info["total"]
    trainable_params = param_info["trainable"]
    lora_params = param_info["lora"]
    print(f"Total DiT params : {total_params:,}")
    print(f"LoRA params      : {lora_params:,}")
    print(f"Trainable params : {trainable_params:,} ({param_info['trainable_pct']:.2f}%)")

    exp_config = {
        "method": "lora_tta_opensora",
        "training": {
            "learning_rate": args.learning_rate,
            "num_steps": args.num_steps,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "optimizer": "adamw",
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "lora_params": lora_params,
        },
        "generation": {
            "num_cond_frames": args.num_cond_frames,
            "num_gen_frames": args.num_gen_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "height": args.height,
            "width": args.width,
        },
        "seed": args.seed,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    videos = load_ucf101_video_list(args.data_dir, max_videos=args.max_videos, seed=args.seed)
    print(f"\nTotal videos: {len(videos)}")

    videos_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    vae_time_ratio = getattr(vae, "time_compression_ratio", 4)
    vae_space_ratio = getattr(vae, "spatial_compression_ratio", 8)

    for idx, entry in enumerate(tqdm(videos, desc="LoRA TTA (Open-Sora)")):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            reset_lora_weights(dit)

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

            lat_H = all_latents.shape[3]
            lat_W = all_latents.shape[4]
            lat_T = all_latents.shape[2]
            num_img_tokens = lat_T * (lat_H // args.patch_size) * (lat_W // args.patch_size)

            text_inputs = encode_prompt_opensora(
                t5, clip, caption,
                device=args.device, dtype=torch.bfloat16,
                num_img_tokens=num_img_tokens,
            )

            vae.to("cpu")
            t5.to("cpu")
            clip.to("cpu")
            torch.cuda.empty_cache()

            train_result = finetune_lora_opensora(
                dit=dit,
                cond_latents=cond_latents,
                train_latents=train_latents,
                text_inputs=text_inputs,
                num_steps=args.num_steps,
                lr=args.learning_rate,
                warmup_steps=args.warmup_steps,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                device=args.device,
                dtype=torch.bfloat16,
                guidance=args.guidance_scale,
                patch_size=args.patch_size,
            )

            result = {
                "idx": idx,
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "train_time": train_result["train_time"],
                "final_loss": train_result["losses"][-1] if train_result["losses"] else None,
                "num_train_steps": len(train_result["losses"]),
                "success": True,
            }

            vae.to(args.device)
            t5.to(args.device)
            clip.to(args.device)

            gen_time = 0.0
            if not args.skip_generation:
                gen_cond_start = max(0, args.gen_start_frame - args.num_cond_frames)
                gen_pixel = load_video_frames(
                    video_path, args.num_cond_frames,
                    height=args.height, width=args.width,
                    start_frame=gen_cond_start,
                ).to(args.device, torch.bfloat16)

                ref_latents = encode_video_opensora(vae, gen_pixel)

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

                output_path = os.path.join(videos_dir, f"{video_name}_lora.mp4")
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

                del gen_pixel, ref_latents

            result["total_time"] = train_result["train_time"] + gen_time

            print(f"  Train: {train_result['train_time']:.1f}s, "
                  f"Loss: {result['final_loss']:.4f}"
                  + (f", Gen: {gen_time:.1f}s" if not args.skip_generation else "")
                  + (f", PSNR={result.get('psnr', 'N/A')}" if "psnr" in result else ""))

            all_results.append(result)

            del all_latents, cond_latents, train_latents
            del pixel_frames, text_inputs
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
            torch_gc()

        save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)

    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "lora_tta_opensora",
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "optimizer": "adamw",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "lora_params": lora_params,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "avg_train_time": float(np.mean([r["train_time"] for r in successful])) if successful else 0,
        "avg_final_loss": float(np.mean([r["final_loss"] for r in successful if r.get("final_loss") is not None])) if successful else 0,
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))

    print("\n" + "=" * 70)
    print("LoRA TTA (Open-Sora v2.0) Complete!")
    print("=" * 70)
    print(f"Successful: {len(successful)}/{len(all_results)}")
    if successful:
        print(f"Avg train time: {summary['avg_train_time']:.1f}s")
        print(f"Avg final loss: {summary['avg_final_loss']:.4f}")
    print(f"LoRA rank={args.lora_rank}, alpha={args.lora_alpha}")
    print(f"LoRA params: {lora_params:,} / {total_params:,} ({param_info['trainable_pct']:.2f}%)")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
