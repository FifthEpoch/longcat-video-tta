#!/usr/bin/env python3
"""
Delta-B TTA: Per-layer modulation offsets.

Each LongCatSingleStreamBlock produces 6 modulation vectors from the
timestep embedding via adaLN_modulation:
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

Delta B learns grouped δ vectors that are added to the timestep embedding
*before* it enters each block's adaLN_modulation. Blocks are divided into
groups to keep parameter count small.

Architecture:
    - 48 blocks, grouped into `num_groups` groups (default 4)
    - Each group has one δ_group ∈ R^{adaln_tembed_dim}
    - δ_group is added to `t` before it enters each block in the group

Usage:
    python run_delta_b.py \\
        --checkpoint-dir /path/to/longcat-video-checkpoints \\
        --data-dir /path/to/dataset \\
        --output-dir results/delta_b \\
        --delta-steps 20 --delta-lr 1e-3 --num-groups 4
"""

import argparse
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
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    compute_flow_matching_loss,
    generate_video_continuation,
    save_results,
    save_video_from_numpy,
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
# Delta-B wrapper: per-group modulation offsets
# ============================================================================

class DeltaBWrapper(nn.Module):
    """Wraps LongCatVideoTransformer3DModel to inject per-group δ vectors
    into the timestep embedding before each block's adaLN modulation.

    Parameters
    ----------
    dit : the frozen DiT model
    num_groups : number of delta groups (blocks are split evenly)
    adaln_tembed_dim : dimension of the timestep embedding
    """

    def __init__(
        self,
        dit: nn.Module,
        num_groups: int = 4,
        adaln_tembed_dim: int = 512,
    ):
        super().__init__()
        self.dit = dit
        self.num_groups = num_groups
        self.num_blocks = len(dit.blocks)

        # Freeze all DiT parameters
        for p in self.dit.parameters():
            p.requires_grad = False

        # Create per-group delta vectors
        self.deltas = nn.ParameterList([
            nn.Parameter(torch.zeros(adaln_tembed_dim))
            for _ in range(num_groups)
        ])

        # Assign blocks to groups
        blocks_per_group = math.ceil(self.num_blocks / num_groups)
        self.block_to_group = [
            min(i // blocks_per_group, num_groups - 1)
            for i in range(self.num_blocks)
        ]

        # Generation hooks (installed/removed around pipeline calls)
        self._gen_hooks: list = []

    @property
    def config(self):
        """Proxy config to the inner DiT."""
        return self.dit.config

    # ------------------------------------------------------------------
    # Hook-based injection for pipeline generation
    # ------------------------------------------------------------------
    def apply_to_dit(self):
        """Install per-block forward pre-hooks so the pipeline's full
        forward path sees per-group delta offsets on the timestep embedding."""
        self._gen_hooks = []
        for i, block in enumerate(self.dit.blocks):
            group_idx = self.block_to_group[i]
            delta_vec = self.deltas[group_idx]

            def _make_hook(dv):
                def hook(_module, args):
                    args = list(args)
                    # args[2] is `t` with shape [B, T, C_t]
                    args[2] = args[2] + dv.unsqueeze(0).unsqueeze(0).to(args[2].dtype)
                    return tuple(args)
                return hook

            h = block.register_forward_pre_hook(_make_hook(delta_vec))
            self._gen_hooks.append(h)

    def remove_from_dit(self):
        """Remove all generation hooks."""
        for h in self._gen_hooks:
            h.remove()
        self._gen_hooks = []

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        **kwargs,
    ):
        """Forward with per-group delta offsets on timestep embedding."""
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
            t_base = dit.t_embedder(
                timestep.float().flatten(), dtype=torch.float32
            ).reshape(B, N_t, -1)  # [B, T, C_t]

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

        # Through transformer blocks with per-group delta injection
        for i, block in enumerate(dit.blocks):
            group_idx = self.block_to_group[i]
            delta = self.deltas[group_idx]
            # Inject delta into timestep embedding for this block
            t_modified = t_base + delta.unsqueeze(0).unsqueeze(0)

            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                t_modified,
                y_seqlens,
                (N_t, N_h, N_w),
                num_cond_latents=num_cond_latents,
            )

        hidden_states = dit.final_layer(hidden_states, t_base, (N_t, N_h, N_w))
        hidden_states = dit.unpatchify(hidden_states, N_t, N_h, N_w)
        hidden_states = hidden_states.to(torch.float32)

        return hidden_states


# ============================================================================
# Optimization loop
# ============================================================================

def optimize_delta_b(
    wrapper: DeltaBWrapper,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_steps: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    early_stopper: Optional[AnchoredEarlyStopper] = None,
) -> Dict:
    """Optimize the per-group delta vectors on conditioning latents."""
    delta_params = list(wrapper.deltas.parameters())
    optimizer = AdamW(delta_params, lr=lr, betas=(0.9, 0.999), eps=1e-15)

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
        for p in delta_params:
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_([p], 1.0)
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

    delta_norms = [d.detach().norm().item() for d in wrapper.deltas]
    return {
        "losses": losses,
        "delta_norms": delta_norms,
        "early_stopping_info": es_state,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Delta-B TTA for LongCat-Video")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--delta-steps", type=int, default=20)
    parser.add_argument("--delta-lr", type=float, default=1e-3)
    parser.add_argument("--num-groups", type=int, default=4,
                        help="Number of delta groups across blocks")
    parser.add_argument("--num-cond-frames", type=int, default=13)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip video generation (only train delta)")
    add_early_stopping_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0

    print("=" * 70)
    print("Delta-B TTA for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Num groups     : {args.num_groups}")
    print(f"Delta steps    : {args.delta_steps}")
    print(f"Delta LR       : {args.delta_lr}")
    print(f"Resume from idx: {start_idx}")
    print("=" * 70)

    # Load model
    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16
    )
    dit = components["dit"]
    vae = components["vae"]
    pipe = components["pipe"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    adaln_dim = dit.config.adaln_tembed_dim

    from common import load_ucf101_video_list
    videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed
    )
    print(f"\nTotal videos: {len(videos)}")

    early_stopper = build_early_stopper_from_args(args)
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
            pixel_frames = load_video_frames(
                video_path, args.num_cond_frames, height=480, width=832
            ).to(args.device, torch.bfloat16)

            latents = encode_video(vae, pixel_frames, normalize=True)

            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder, caption,
                device=args.device, dtype=torch.bfloat16,
            )

            # Fresh wrapper per video
            wrapper = DeltaBWrapper(
                dit, num_groups=args.num_groups, adaln_tembed_dim=adaln_dim
            ).to(args.device)

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

            t0 = time.time()
            opt_result = optimize_delta_b(
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
                "delta_norms": opt_result["delta_norms"],
                "num_groups": args.num_groups,
                "early_stopping_info": opt_result.get("early_stopping_info"),
                "success": True,
            }

            print(f"  Train time: {train_time:.1f}s, "
                  f"Final loss: {result['final_loss']:.4f}, "
                  f"Norms: {opt_result['delta_norms']}")

            # ── Generation ──────────────────────────────────────────
            gen_time = 0.0
            if not args.skip_generation:
                from PIL import Image

                pf = pixel_frames.squeeze(0)  # [C, T, H, W]
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_images = []
                for t_idx in range(pf.shape[1]):
                    frame_np = (pf[:, t_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    cond_images.append(Image.fromarray(frame_np))

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

                    output_path = os.path.join(videos_dir, f"{video_name}_delta_b.mp4")
                    save_video_from_numpy(gen_frames, output_path, fps=24)
                    result["output_path"] = output_path
                    result["gen_time"] = gen_time
                    print(f"  Gen: {gen_time:.1f}s → {output_path}")
                finally:
                    wrapper.remove_from_dit()

            result["total_time"] = train_time + gen_time
            all_results.append(result)

            del wrapper, latents, pixel_frames, prompt_embeds, prompt_mask
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

    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "delta_b",
        "num_groups": args.num_groups,
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
