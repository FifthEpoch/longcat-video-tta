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
    generate_video_continuation,
    save_results,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
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

        # Through transformer blocks
        for block in dit.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                t,
                y_seqlens,
                (N_t, N_h, N_w),
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
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_steps: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    early_stopper: Optional[AnchoredEarlyStopper] = None,
) -> Dict:
    """Optimize the delta vector on conditioning latents."""
    optimizer = AdamW([wrapper.delta], lr=lr, betas=(0.9, 0.999), eps=1e-15)

    wrapper.train()
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        loss = compute_flow_matching_loss(
            dit=wrapper,
            latents=latents,
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
            should_stop, es_info = early_stopper.step(step + 1)
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    # Restore best if early stopping was used
    es_state = None
    if early_stopper is not None:
        early_stopper.restore(
            restore_fn=lambda sd: wrapper.load_state_dict(sd, strict=False)
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
    parser.add_argument("--num-cond-frames", type=int, default=13)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-generation", action="store_true")
    add_early_stopping_args(parser)
    args = parser.parse_args()

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

    for idx, entry in enumerate(videos):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            # Load and encode video
            pixel_frames = load_video_frames(
                video_path, args.num_cond_frames, height=480, width=832
            ).to(args.device, torch.bfloat16)

            latents = encode_video(vae, pixel_frames, normalize=True)

            # Encode text
            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder, caption,
                device=args.device, dtype=torch.bfloat16,
            )

            # Create wrapper with fresh delta
            wrapper = DeltaAWrapper(dit, adaln_tembed_dim=adaln_dim).to(args.device)

            # Setup early stopper
            if early_stopper is not None:
                early_stopper.setup(
                    model=wrapper,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    device=args.device,
                    dtype=torch.bfloat16,
                    forward_fn=lambda nl, ts: wrapper(
                        hidden_states=nl, timestep=ts,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_mask,
                    ),
                )

            # Optimize
            t0 = time.time()
            opt_result = optimize_delta_a(
                wrapper=wrapper,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                num_steps=args.delta_steps,
                lr=args.delta_lr,
                device=args.device,
                dtype=torch.bfloat16,
                early_stopper=early_stopper,
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
            }

            print(f"  Train time: {train_time:.1f}s, "
                  f"Final loss: {result['final_loss']:.4f}, "
                  f"Delta norm: {result['delta_norm']:.4f}")

            all_results.append(result)

            # Cleanup per-video
            del wrapper, latents, pixel_frames, prompt_embeds, prompt_mask
            torch_gc()

        except Exception as e:
            print(f"  ERROR: {e}")
            all_results.append({
                "video_name": video_name,
                "error": str(e),
            })

        # Save checkpoint
        save_checkpoint({"next_idx": idx + 1}, ckpt_path)

    # Save final results
    summary = {
        "method": "delta_a",
        "delta_steps": args.delta_steps,
        "delta_lr": args.delta_lr,
        "num_videos": len(all_results),
        "avg_train_time": np.mean([r.get("train_time", 0) for r in all_results if "train_time" in r]),
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))
    print(f"\nResults saved to {args.output_dir}/summary.json")
    print(f"Avg train time: {summary['avg_train_time']:.1f}s")


if __name__ == "__main__":
    main()
