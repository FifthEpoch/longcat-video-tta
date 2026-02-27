#!/usr/bin/env python3
"""
Norm-Tuning TTA: Fine-tune only the affine parameters of normalization layers.

Inspired by TENT (Wang et al., ICLR 2021), this method freezes the entire
DiT and only unfreezes the gamma/beta of selected normalization layers.

Supported modes (--norm-target):
  "cross_attn_norm"  -- pre_crs_attn_norm weight+bias (393K params)
  "qk_norm"          -- RMSNorm weights in attn.q/k_norm (24K params)
  "all_norm"          -- both of the above (418K params)

No wrapper needed: the DiT's own norm params are unfrozen directly.
"""

import argparse
import copy
import gc
import json
import math
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    compute_flow_matching_loss_conditioned,
    generate_video_continuation,
    save_results,
    save_video_from_numpy,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    add_augmentation_args,
    add_tta_frame_args,
    parse_speed_factors,
    split_tta_latents,
    evaluate_generation_metrics,
)
from early_stopping import (
    AnchoredEarlyStopper,
    add_early_stopping_args,
    build_early_stopper_from_args,
)


# ============================================================================
# Norm parameter collection
# ============================================================================

def collect_norm_params(dit: nn.Module, norm_target: str) -> List[nn.Parameter]:
    """Collect norm affine parameters to unfreeze based on target mode."""
    params = []

    for block in dit.blocks:
        if norm_target in ("cross_attn_norm", "all_norm"):
            norm = block.pre_crs_attn_norm
            if hasattr(norm, "weight") and norm.weight is not None:
                params.append(norm.weight)
            if hasattr(norm, "bias") and norm.bias is not None:
                params.append(norm.bias)

        if norm_target in ("qk_norm", "all_norm"):
            for sub_name in ("attn.q_norm", "attn.k_norm",
                             "cross_attn.q_norm", "cross_attn.k_norm"):
                parts = sub_name.split(".")
                mod = block
                for p in parts:
                    mod = getattr(mod, p, None)
                    if mod is None:
                        break
                if mod is not None and hasattr(mod, "weight") and mod.weight is not None:
                    params.append(mod.weight)

    return params


def snapshot_params(params: List[nn.Parameter]) -> List[torch.Tensor]:
    """Deep-copy parameter data for save/restore."""
    return [p.data.clone() for p in params]


def restore_params(params: List[nn.Parameter], snapshot: List[torch.Tensor]):
    """Restore parameter data from snapshot."""
    for p, s in zip(params, snapshot):
        p.data.copy_(s)


# ============================================================================
# Training wrapper (thin wrapper for forward with gradient checkpointing)
# ============================================================================

class NormTuneForward(nn.Module):
    """Thin wrapper that runs the DiT forward with gradient checkpointing.

    No delta injection -- the DiT itself has unfrozen norm params.
    """

    def __init__(self, dit: nn.Module):
        super().__init__()
        self.dit = dit

    @property
    def config(self):
        return self.dit.config

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        **kwargs,
    ):
        dit = self.dit

        B, _, T, H, W = hidden_states.shape
        N_t = T // dit.patch_size[0]
        N_h = H // dit.patch_size[1]
        N_w = W // dit.patch_size[2]

        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)

        dtype = dit.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = dit.x_embedder(hidden_states)

        import torch.amp as amp
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            t = dit.t_embedder(
                timestep.float().flatten(), dtype=torch.float32
            ).reshape(B, N_t, -1)

        encoder_hidden_states = dit.y_embedder(encoder_hidden_states)

        if dit.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = (
                encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            )
            encoder_attention_mask = (
                encoder_attention_mask * 0 + 1
            ).to(encoder_attention_mask.dtype)

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = (
                encoder_hidden_states.squeeze(1)
                .masked_select(encoder_attention_mask.unsqueeze(-1) != 0)
                .view(1, -1, hidden_states.shape[-1])
            )
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(
                1, -1, hidden_states.shape[-1]
            )

        import functools as _ft
        from torch.utils.checkpoint import checkpoint as _ckpt_fn
        _ckpt = _ft.partial(_ckpt_fn, use_reentrant=False)

        for block in dit.blocks:
            if torch.is_grad_enabled():
                hidden_states = _ckpt(
                    block, hidden_states, encoder_hidden_states, t,
                    y_seqlens, (N_t, N_h, N_w),
                    num_cond_latents=num_cond_latents,
                )
            else:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, t,
                    y_seqlens, (N_t, N_h, N_w),
                    num_cond_latents=num_cond_latents,
                )

        hidden_states = dit.final_layer(hidden_states, t, (N_t, N_h, N_w))
        hidden_states = dit.unpatchify(hidden_states, N_t, N_h, N_w)
        hidden_states = hidden_states.to(torch.float32)

        return hidden_states


# ============================================================================
# Optimization loop
# ============================================================================

def optimize_norm_params(
    wrapper: NormTuneForward,
    norm_params: List[nn.Parameter],
    cond_latents: torch.Tensor,
    train_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_steps: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    early_stopper: Optional[AnchoredEarlyStopper] = None,
    train_latents_variants: Optional[List[Dict]] = None,
) -> Dict:
    """Optimize norm affine parameters using conditioning-aware loss."""
    optimizer = AdamW(norm_params, lr=lr, betas=(0.9, 0.999), eps=1e-15)

    if train_latents_variants is None:
        train_latents_variants = [{"latents": train_latents, "name": "orig"}]

    def _save_fn():
        return snapshot_params(norm_params)

    wrapper.dit.train()
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        vi = torch.randint(0, len(train_latents_variants), (1,)).item()
        step_train = train_latents_variants[vi]["latents"]

        loss = compute_flow_matching_loss_conditioned(
            dit=wrapper,
            cond_latents=cond_latents,
            target_latents=step_train,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            device=device,
            dtype=dtype,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(norm_params, 1.0)
        optimizer.step()

        losses.append(loss.item())

        if early_stopper is not None:
            should_stop, es_info = early_stopper.step(
                step + 1, save_fn=_save_fn,
            )
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    wrapper.dit.eval()

    es_state = None
    if early_stopper is not None:
        early_stopper.restore(
            restore_fn=lambda s: restore_params(norm_params, s)
        )
        es_state = early_stopper.state

    return {
        "losses": losses,
        "early_stopping_info": es_state,
    }


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Norm-Tuning TTA for LongCat-Video")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--norm-steps", type=int, default=20)
    parser.add_argument("--norm-lr", type=float, default=1e-3)
    parser.add_argument("--norm-target", type=str, default="all_norm",
                        choices=["cross_attn_norm", "qk_norm", "all_norm"],
                        help="Which norm layers to tune")
    parser.add_argument("--num-cond-frames", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--gen-start-frame", type=int, default=32)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--no-save-videos", action="store_true")
    parser.add_argument("--also-tune-delta", action="store_true",
                        help="Also learn a Delta-A vector alongside norm params")
    add_early_stopping_args(parser)
    add_augmentation_args(parser)
    add_tta_frame_args(parser)
    args = parser.parse_args()

    if args.tta_total_frames is None:
        args.tta_total_frames = args.gen_start_frame
    if args.tta_context_frames is None or args.tta_context_frames > args.tta_total_frames:
        args.tta_context_frames = args.num_cond_frames

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0

    print("=" * 70)
    print("Norm-Tuning TTA for LongCat-Video")
    print("=" * 70)
    print(f"Norm target    : {args.norm_target}")
    print(f"Norm steps     : {args.norm_steps}")
    print(f"Norm LR        : {args.norm_lr}")
    print(f"Resume from idx: {start_idx}")
    print("=" * 70)

    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16
    )
    dit = components["dit"]
    vae = components["vae"]
    pipe = components["pipe"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    import functools
    from torch.utils.checkpoint import checkpoint as _ckpt_fn
    dit.gradient_checkpointing = True
    dit._gradient_checkpointing_func = functools.partial(_ckpt_fn, use_reentrant=False)
    print("Gradient checkpointing: ENABLED")

    # Freeze everything
    for p in dit.parameters():
        p.requires_grad = False

    # Collect and unfreeze norm params
    norm_params = collect_norm_params(dit, args.norm_target)
    for p in norm_params:
        p.requires_grad = True

    # Optionally add a Delta-A vector (combined tuning)
    delta_hook_handle = None
    if args.also_tune_delta:
        adaln_dim = dit.config.adaln_tembed_dim
        delta_param = nn.Parameter(torch.zeros(adaln_dim, device=args.device))
        norm_params.append(delta_param)

        def _delta_hook(_module, _input, output):
            return output + delta_param.unsqueeze(0).to(output.dtype)

        delta_hook_handle = dit.t_embedder.register_forward_hook(_delta_hook)
        print(f"Also tuning Delta-A vector: {adaln_dim} params")

    trainable = sum(p.numel() for p in norm_params)
    total = sum(p.numel() for p in dit.parameters())
    print(f"Total params to tune: {len(norm_params)} tensors, {trainable:,} params "
          f"({100 * trainable / total:.4f}%)")

    # Save original values for per-video reset
    original_norm_snapshot = snapshot_params(norm_params)

    wrapper = NormTuneForward(dit)

    from common import load_ucf101_video_list
    videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed
    )
    print(f"\nTotal videos: {len(videos)}")

    early_stopper = build_early_stopper_from_args(args)
    all_results = []
    videos_dir = os.path.join(args.output_dir, "videos")
    if not args.skip_generation and not args.no_save_videos:
        os.makedirs(videos_dir, exist_ok=True)

    for idx, entry in enumerate(videos):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            tta_start = args.gen_start_frame - args.tta_total_frames
            pixel_frames = load_video_frames(
                video_path, args.tta_total_frames, height=480, width=832,
                start_frame=max(0, tta_start),
            ).to(args.device, torch.bfloat16)

            all_latents = encode_video(vae, pixel_frames, normalize=True)

            vae_t_scale = 4
            num_ctx_lat = 1 + (args.tta_context_frames - 1) // vae_t_scale
            cond_latents, train_latents, val_latents = split_tta_latents(
                all_latents, num_ctx_lat,
                holdout_fraction=getattr(args, "es_holdout_fraction", 0.25),
            )
            print(f"  Latent split: cond={cond_latents.shape[2]}, "
                  f"train={train_latents.shape[2]}, "
                  f"val={val_latents.shape[2] if val_latents is not None else 0}")

            gen_cond_start = args.gen_start_frame - args.num_cond_frames
            gen_pixel_frames = load_video_frames(
                video_path, args.num_cond_frames, height=480, width=832,
                start_frame=max(0, gen_cond_start),
            ).to(args.device, torch.bfloat16)

            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder, caption,
                device=args.device, dtype=torch.bfloat16,
            )

            train_latents_variants = None
            if args.aug_enabled:
                from common import build_augmented_pixel_variants
                pix_variants = build_augmented_pixel_variants(
                    pixel_frames,
                    enable_flip=args.aug_flip,
                    rotate_deg=args.aug_rotate_deg,
                    rotate_random_min=args.aug_rotate_random_min,
                    rotate_random_max=args.aug_rotate_random_max,
                    rotate_random_count=args.aug_rotate_random_count,
                    rotate_random_step=args.aug_rotate_random_step,
                    rotate_zoom=args.aug_rotate_zoom,
                    speed_factors=parse_speed_factors(args.aug_speed_factors),
                )
                train_latents_variants = []
                for pv in pix_variants:
                    if pv["name"] == "orig":
                        train_latents_variants.append({"latents": train_latents, "name": "orig"})
                    else:
                        aug_lat = encode_video(vae, pv["pixel_frames"], normalize=True)
                        t_start = cond_latents.shape[2]
                        t_end = t_start + train_latents.shape[2]
                        train_latents_variants.append({
                            "latents": aug_lat[:, :, t_start:t_end],
                            "name": pv["name"],
                        })

            # Reset norm params to original values for this video
            restore_params(norm_params, original_norm_snapshot)

            if early_stopper is not None and val_latents is not None:
                def _es_forward_fn(hs, ts, ncl):
                    return wrapper(
                        hidden_states=hs, timestep=ts,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_mask,
                        num_cond_latents=ncl,
                    )
                early_stopper.setup(
                    model=wrapper,
                    cond_latents=cond_latents,
                    val_latents=val_latents,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    device=args.device,
                    dtype=torch.bfloat16,
                    forward_fn=_es_forward_fn,
                    video_id=video_name,
                    save_fn=lambda: snapshot_params(norm_params),
                )

            vae.to("cpu")
            text_encoder.to("cpu")
            torch.cuda.empty_cache()

            t0 = time.time()
            opt_result = optimize_norm_params(
                wrapper=wrapper,
                norm_params=norm_params,
                cond_latents=cond_latents,
                train_latents=train_latents,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                num_steps=args.norm_steps,
                lr=args.norm_lr,
                device=args.device,
                dtype=torch.bfloat16,
                early_stopper=early_stopper if val_latents is not None else None,
                train_latents_variants=train_latents_variants,
            )
            train_time = time.time() - t0

            result = {
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "train_time": train_time,
                "final_loss": opt_result["losses"][-1] if opt_result["losses"] else None,
                "early_stopping_info": opt_result.get("early_stopping_info"),
                "success": True,
            }

            loss_str = f"{result['final_loss']:.4f}" if result["final_loss"] is not None else "N/A"
            print(f"  Train time: {train_time:.1f}s, Final loss: {loss_str}")

            vae.to(args.device)
            text_encoder.to(args.device)

            gen_time = 0.0
            if not args.skip_generation:
                from PIL import Image
                pf = gen_pixel_frames.squeeze(0)
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_images = []
                for t_idx in range(pf.shape[1]):
                    frame_np = (pf[:, t_idx].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                    cond_images.append(Image.fromarray(frame_np))

                gen_start = time.time()
                gen_frames = generate_video_continuation(
                    pipe=pipe,
                    video_frames=cond_images,
                    prompt=caption,
                    num_cond_frames=args.num_cond_frames,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed + idx,
                    resolution=args.resolution,
                    device=args.device,
                )
                gen_time = time.time() - gen_start

                result["gen_time"] = gen_time
                output_path = os.path.join(videos_dir, f"{video_name}_norm_tune.mp4")
                if not args.no_save_videos:
                    save_video_from_numpy(gen_frames, output_path, fps=24)
                    result["output_path"] = output_path

                num_gen = args.num_frames - args.num_cond_frames
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

            result["total_time"] = train_time + gen_time
            all_results.append(result)

            del all_latents, cond_latents, train_latents, val_latents
            del pixel_frames, gen_pixel_frames, prompt_embeds, prompt_mask
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

        save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)

    if delta_hook_handle is not None:
        delta_hook_handle.remove()

    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "norm_tune",
        "norm_target": args.norm_target,
        "norm_steps": args.norm_steps,
        "norm_lr": args.norm_lr,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "gen_start_frame": args.gen_start_frame,
        "trainable_params": trainable,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "avg_train_time": np.mean([r.get("train_time", 0) for r in successful]) if successful else 0,
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))
    print(f"\nResults saved to {args.output_dir}/summary.json")


if __name__ == "__main__":
    main()
