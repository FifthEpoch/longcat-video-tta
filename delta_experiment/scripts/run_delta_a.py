#!/usr/bin/env python3
"""
Delta-A TTA: Add a single learnable δ vector to the timestep embedding.

In LongCat-Video, the timestep embedding `t` has shape [B, T, C_t] where
C_t = adaln_tembed_dim (default 512). Delta A adds a learnable vector
δ ∈ R^{C_t} to this embedding before it enters each transformer block's
AdaLN modulation:

    t' = t + δ

This is the simplest δ-TTA method — one vector per video, discarded after.

Usage:
    python run_delta_a.py \\
        --checkpoint-dir /path/to/longcat-video-checkpoints \\
        --data-dir /path/to/dataset \\
        --output-dir results/delta_a \\
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

# Ensure common.py is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    compute_flow_matching_loss,
    compute_flow_matching_loss_conditioned,
    generate_video_continuation,
    save_results,
    save_video_from_numpy,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    build_augmented_latent_variants,
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
# Delta-A wrapper: hooks the timestep embedding
# ============================================================================

class DeltaAWrapper(nn.Module):
    """Wraps a LongCatVideoTransformer3DModel to inject δ into the
    timestep embedding before it reaches the transformer blocks.

    The delta vector is added to the timestep embedding `t` which has
    shape [B, T, C_t] (output of t_embedder).
    """

    def __init__(self, dit: nn.Module, adaln_tembed_dim: int = 512):
        super().__init__()
        self.dit = dit
        # Freeze all DiT parameters
        for p in self.dit.parameters():
            p.requires_grad = False

        # Learnable delta vector
        self.delta = nn.Parameter(torch.zeros(adaln_tembed_dim))

        # Generation hooks (installed/removed around pipeline calls)
        self._gen_hook = None

    @property
    def config(self):
        """Proxy config to the inner DiT so callers like compute_flow_matching_loss work."""
        return self.dit.config

    # ------------------------------------------------------------------
    # Hook-based injection for pipeline generation
    # ------------------------------------------------------------------
    def apply_to_dit(self):
        """Install a forward hook on t_embedder so the pipeline's full
        forward path (KV-cache, BSA, etc.) sees the delta."""
        delta = self.delta

        def _hook(_module, _input, output):
            # t_embedder output: [B*T, C_t] — add delta broadcast
            return output + delta.unsqueeze(0).to(output.dtype)

        self._gen_hook = self.dit.t_embedder.register_forward_hook(_hook)

    def remove_from_dit(self):
        """Remove the generation hook."""
        if self._gen_hook is not None:
            self._gen_hook.remove()
            self._gen_hook = None

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        **kwargs,
    ):
        """Forward with delta injected into timestep embedding."""
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

        hidden_states = dit.x_embedder(hidden_states)  # [B, N, C]

        import torch.amp as amp
        with amp.autocast(device_type="cuda", dtype=torch.float32):
            t = dit.t_embedder(
                timestep.float().flatten(), dtype=torch.float32
            ).reshape(B, N_t, -1)  # [B, T, C_t]

        # ── Delta injection ──
        t = t + self.delta.unsqueeze(0).unsqueeze(0)  # broadcast [1, 1, C_t]

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

        # Through transformer blocks (with gradient checkpointing)
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

def optimize_delta_a(
    wrapper: DeltaAWrapper,
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
    """Optimize the delta vector using conditioning-aware loss.

    Parameters
    ----------
    cond_latents  : clean context latents [B, C, T_cond, H, W]
    train_latents : target latents to noise and compute loss on [B, C, T_train, H, W]
    train_latents_variants : optional augmented variants of train_latents
    """
    optimizer = AdamW([wrapper.delta], lr=lr, betas=(0.9, 0.999), eps=1e-15)

    # Build variant list (original only if no augmentation)
    if train_latents_variants is None:
        train_latents_variants = [{"latents": train_latents, "name": "orig"}]

    def _save_fn():
        return copy.deepcopy(wrapper.delta.data)

    wrapper.train()
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # Randomly pick a variant
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
        torch.nn.utils.clip_grad_norm_([wrapper.delta], 1.0)
        optimizer.step()

        losses.append(loss.item())

        # Early stopping check
        if early_stopper is not None:
            should_stop, es_info = early_stopper.step(
                step + 1, save_fn=_save_fn,
            )
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    # Restore best if early stopping was used
    es_state = None
    if early_stopper is not None:
        early_stopper.restore(
            restore_fn=lambda s: wrapper.delta.data.copy_(s)
        )
        es_state = early_stopper.state

    return {
        "losses": losses,
        "delta_norm": wrapper.delta.detach().norm().item(),
        "early_stopping_info": es_state,
    }


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Delta-A TTA for LongCat-Video")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--delta-steps", type=int, default=20)
    parser.add_argument("--delta-lr", type=float, default=1e-3)
    parser.add_argument("--num-cond-frames", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--gen-start-frame", type=int, default=32,
                        help="Fixed anchor frame where generation starts. "
                             "Cond = video[anchor-cond : anchor]. "
                             "Ensures fair comparison across configs.")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip video generation (only train delta)")
    add_early_stopping_args(parser)
    add_augmentation_args(parser)
    add_tta_frame_args(parser)
    args = parser.parse_args()

    # Default tta_total_frames to gen_start_frame (all pre-anchor frames)
    if args.tta_total_frames is None:
        args.tta_total_frames = args.gen_start_frame
    # Default tta_context_frames to match generation conditioning
    if args.tta_context_frames is None or args.tta_context_frames > args.tta_total_frames:
        args.tta_context_frames = args.num_cond_frames

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support
    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0

    print("=" * 70)
    print("Delta-A TTA for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Delta steps    : {args.delta_steps}")
    print(f"Delta LR       : {args.delta_lr}")
    print(f"Augmentation   : {args.aug_enabled}")
    print(f"Resume from idx: {start_idx}")
    print("=" * 70)

    # Load model
    print("\nLoading model components...")
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
    print("Gradient checkpointing: ENABLED (use_reentrant=False)")

    adaln_dim = dit.config.adaln_tembed_dim

    # Load video list
    from common import load_ucf101_video_list
    videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed
    )
    print(f"\nTotal videos: {len(videos)}")

    # Build early stopper
    early_stopper = build_early_stopper_from_args(args)

    # Results
    all_results = []
    videos_dir = os.path.join(args.output_dir, "videos")
    if not args.skip_generation:
        os.makedirs(videos_dir, exist_ok=True)

    for idx, entry in enumerate(videos):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            # ── Frame loading ─────────────────────────────────────────
            # Load tta_total_frames ending at anchor for TTA
            tta_start = args.gen_start_frame - args.tta_total_frames
            pixel_frames = load_video_frames(
                video_path, args.tta_total_frames, height=480, width=832,
                start_frame=max(0, tta_start),
            ).to(args.device, torch.bfloat16)

            all_latents = encode_video(vae, pixel_frames, normalize=True)

            # Split into context / train / val
            vae_t_scale = 4  # vae_scale_factor_temporal
            num_ctx_lat = 1 + (args.tta_context_frames - 1) // vae_t_scale
            cond_latents, train_latents, val_latents = split_tta_latents(
                all_latents, num_ctx_lat,
                holdout_fraction=getattr(args, "es_holdout_fraction", 0.25),
            )
            print(f"  Latent split: cond={cond_latents.shape[2]}, "
                  f"train={train_latents.shape[2]}, "
                  f"val={val_latents.shape[2] if val_latents is not None else 0}")

            # Also keep the original num_cond_frames pixel clip for generation
            gen_cond_start = args.gen_start_frame - args.num_cond_frames
            gen_pixel_frames = load_video_frames(
                video_path, args.num_cond_frames, height=480, width=832,
                start_frame=max(0, gen_cond_start),
            ).to(args.device, torch.bfloat16)

            # Encode text
            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder, caption,
                device=args.device, dtype=torch.bfloat16,
            )

            # Build augmented train latent variants if enabled
            train_latents_variants = None
            if args.aug_enabled:
                # Augment only the train portion at pixel level
                # For simplicity, augment the full pixel clip and re-encode,
                # then slice out the train portion
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
                        # Slice out the train portion (same offsets)
                        t_start = cond_latents.shape[2]
                        t_end = t_start + train_latents.shape[2]
                        train_latents_variants.append({
                            "latents": aug_lat[:, :, t_start:t_end],
                            "name": pv["name"],
                        })
                print(f"  Augmentation: {len(train_latents_variants)} variants "
                      f"({', '.join(v['name'] for v in train_latents_variants)})")

            # Create wrapper with fresh delta
            wrapper = DeltaAWrapper(dit, adaln_tembed_dim=adaln_dim).to(args.device)

            # Setup early stopper
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
                    save_fn=lambda: copy.deepcopy(wrapper.delta.data),
                )

            # Offload VAE + text encoder to CPU during training
            vae.to("cpu")
            text_encoder.to("cpu")
            torch.cuda.empty_cache()

            # Optimize
            t0 = time.time()
            opt_result = optimize_delta_a(
                wrapper=wrapper,
                cond_latents=cond_latents,
                train_latents=train_latents,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                num_steps=args.delta_steps,
                lr=args.delta_lr,
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
                "delta_norm": opt_result["delta_norm"],
                "early_stopping_info": opt_result.get("early_stopping_info"),
                "success": True,
            }

            print(f"  Train time: {train_time:.1f}s, "
                  f"Final loss: {result['final_loss']:.4f}, "
                  f"Delta norm: {result['delta_norm']:.4f}")

            # ── Generation ──────────────────────────────────────────
            # Bring VAE + text encoder back to GPU for generation
            vae.to(args.device)
            text_encoder.to(args.device)

            gen_time = 0.0
            if not args.skip_generation:
                from PIL import Image

                # Convert generation conditioning frames to PIL images
                pf = gen_pixel_frames.squeeze(0)  # [C, T, H, W]
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_images = []
                for t_idx in range(pf.shape[1]):
                    frame_np = (pf[:, t_idx].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                    cond_images.append(Image.fromarray(frame_np))

                # Install delta hook so the pipeline sees the adapted model
                wrapper.apply_to_dit()
                try:
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

                    output_path = os.path.join(videos_dir, f"{video_name}_delta_a.mp4")
                    save_video_from_numpy(gen_frames, output_path, fps=24)
                    result["output_path"] = output_path
                    result["gen_time"] = gen_time

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
                finally:
                    wrapper.remove_from_dit()

            result["total_time"] = train_time + gen_time
            all_results.append(result)

            # Cleanup per-video
            del wrapper, all_latents, cond_latents, train_latents, val_latents
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

        # Save checkpoint
        save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)

    # Save final results
    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "delta_a",
        "delta_steps": args.delta_steps,
        "delta_lr": args.delta_lr,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "avg_train_time": np.mean([r.get("train_time", 0) for r in successful]) if successful else 0,
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))
    print(f"\nResults saved to {args.output_dir}/summary.json")
    if successful:
        print(f"Avg train time: {summary['avg_train_time']:.1f}s")


if __name__ == "__main__":
    main()
