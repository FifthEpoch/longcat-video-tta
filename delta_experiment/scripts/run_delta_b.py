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
import copy
import gc
import json
import math
import os
import re
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
    add_caption_guard_args,
    add_caption_override_args,
    add_clip_gate_args,
    parse_speed_factors,
    split_tta_latents,
    evaluate_generation_metrics,
    evaluate_clip_gate,
    summarize_clip_gate_stats,
    validate_caption_quality,
    apply_fixed_caption,
)
from early_stopping import (
    AnchoredEarlyStopper,
    add_early_stopping_args,
    build_early_stopper_from_args,
)


def _slugify_text(text: str, max_len: int = 64) -> str:
    """Create a filesystem-safe prompt slug for output filenames."""
    s = (text or "").lower().strip()
    s = re.sub(r"[^a-z0-9\\s-]", "", s)
    s = re.sub(r"[\\s]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:max_len] if s else "no_prompt"


# ============================================================================
# Delta-B wrapper: per-group modulation offsets
# ============================================================================

class DeltaBWrapper(nn.Module):
    """Wraps LongCatVideoTransformer3DModel to inject per-group δ vectors.

    Supports two injection targets controlled by ``delta_target``:
      - "timestep": adds δ to the timestep embedding before adaLN_modulation
        (original implementation, 512-dim per group)
      - "hidden":  adds δ as a residual to hidden states after each block
        (new implementation, 4096-dim per group, mirrors Open-Sora's approach
        where δ modifies the representation stream rather than the denoising
        schedule)

    Parameters
    ----------
    dit : the frozen DiT model
    num_groups : number of delta groups (blocks are split evenly)
    adaln_tembed_dim : dimension of the timestep embedding (for "timestep" mode)
    hidden_size : dimension of hidden states (for "hidden" mode)
    delta_target : "timestep" or "hidden"
    """

    def __init__(
        self,
        dit: nn.Module,
        num_groups: int = 4,
        adaln_tembed_dim: int = 512,
        hidden_size: int = 4096,
        delta_target: str = "timestep",
        delta_dim: int = None,
        target_blocks: str = "all",
    ):
        super().__init__()
        self.dit = dit
        self.num_groups = num_groups
        self.num_blocks = len(dit.blocks)
        self.delta_target = delta_target
        self.target_block_indices = _parse_target_blocks(target_blocks, self.num_blocks)

        for p in self.dit.parameters():
            p.requires_grad = False

        full_dim = adaln_tembed_dim if delta_target == "timestep" else hidden_size
        self._full_dim = full_dim
        self._partial_dim = delta_dim if delta_dim is not None else full_dim
        self.deltas = nn.ParameterList([
            nn.Parameter(torch.zeros(self._partial_dim))
            for _ in range(num_groups)
        ])

        # Optional final-layer delta (used in "hidden" mode to match Open-Sora)
        if delta_target == "hidden":
            self.delta_final = nn.Parameter(torch.zeros(delta_dim))
        else:
            self.delta_final = None

        blocks_per_group = math.ceil(self.num_blocks / num_groups)
        self.block_to_group = [
            min(i // blocks_per_group, num_groups - 1)
            for i in range(self.num_blocks)
        ]

        self._gen_hooks: list = []

    def _pad_delta(self, dv: torch.Tensor) -> torch.Tensor:
        """Zero-pad a partial-dimension delta to the full target dimension."""
        if dv.shape[0] >= self._full_dim:
            return dv
        return F.pad(dv, (0, self._full_dim - dv.shape[0]))

    @property
    def config(self):
        """Proxy config to the inner DiT."""
        return self.dit.config

    # ------------------------------------------------------------------
    # Hook-based injection for pipeline generation
    # ------------------------------------------------------------------
    def apply_to_dit(self):
        """Install per-block hooks for the pipeline's forward path."""
        self._gen_hooks = []
        pad = self._pad_delta

        if self.delta_target == "timestep":
            for i, block in enumerate(self.dit.blocks):
                if self.target_block_indices is not None and i not in self.target_block_indices:
                    continue
                group_idx = self.block_to_group[i]
                delta_vec = self.deltas[group_idx]

                def _make_pre_hook(dv):
                    def hook(_module, args):
                        args = list(args)
                        args[2] = args[2] + pad(dv).unsqueeze(0).unsqueeze(0).to(args[2].dtype)
                        return tuple(args)
                    return hook

                h = block.register_forward_pre_hook(_make_pre_hook(delta_vec))
                self._gen_hooks.append(h)
        else:
            for i, block in enumerate(self.dit.blocks):
                if self.target_block_indices is not None and i not in self.target_block_indices:
                    continue
                group_idx = self.block_to_group[i]
                delta_vec = self.deltas[group_idx]

                def _make_post_hook(dv):
                    def hook(_module, _args, output):
                        expanded = pad(dv).unsqueeze(0).unsqueeze(0)
                        if isinstance(output, tuple):
                            return (output[0] + expanded.to(output[0].dtype),) + output[1:]
                        return output + expanded.to(output.dtype)
                    return hook

                h = block.register_forward_hook(_make_post_hook(delta_vec))
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
        """Forward with per-group delta injection."""
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

        import functools as _ft
        from torch.utils.checkpoint import checkpoint as _ckpt_fn
        _ckpt = _ft.partial(_ckpt_fn, use_reentrant=False)

        for i, block in enumerate(dit.blocks):
            active_block = (
                self.target_block_indices is None or i in self.target_block_indices
            )
            group_idx = self.block_to_group[i]
            delta = self._pad_delta(self.deltas[group_idx])

            if self.delta_target == "timestep":
                t_modified = (
                    t_base + delta.unsqueeze(0).unsqueeze(0)
                    if active_block else t_base
                )
                if torch.is_grad_enabled():
                    hidden_states = _ckpt(
                        block, hidden_states, encoder_hidden_states, t_modified,
                        y_seqlens, (N_t, N_h, N_w),
                        num_cond_latents=num_cond_latents,
                    )
                else:
                    hidden_states = block(
                        hidden_states, encoder_hidden_states, t_modified,
                        y_seqlens, (N_t, N_h, N_w),
                        num_cond_latents=num_cond_latents,
                    )
            else:
                if torch.is_grad_enabled():
                    hidden_states = _ckpt(
                        block, hidden_states, encoder_hidden_states, t_base,
                        y_seqlens, (N_t, N_h, N_w),
                        num_cond_latents=num_cond_latents,
                    )
                else:
                    hidden_states = block(
                        hidden_states, encoder_hidden_states, t_base,
                        y_seqlens, (N_t, N_h, N_w),
                        num_cond_latents=num_cond_latents,
                    )
                if active_block:
                    hidden_states = hidden_states + delta.unsqueeze(0).unsqueeze(0).to(hidden_states.dtype)

        # Final layer — apply delta_final in "hidden" mode
        if self.delta_target == "hidden" and self.delta_final is not None:
            d_final = self._pad_delta(self.delta_final)
            hidden_states = hidden_states + d_final.unsqueeze(0).unsqueeze(0).to(hidden_states.dtype)

        hidden_states = dit.final_layer(hidden_states, t_base, (N_t, N_h, N_w))
        hidden_states = dit.unpatchify(hidden_states, N_t, N_h, N_w)
        hidden_states = hidden_states.to(torch.float32)

        return hidden_states


# ============================================================================
# Optimization loop
# ============================================================================

def optimize_delta_b(
    wrapper: DeltaBWrapper,
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
    """Optimize the per-group delta vectors using conditioning-aware loss."""
    delta_params = list(wrapper.deltas.parameters())
    if wrapper.delta_final is not None:
        delta_params.append(wrapper.delta_final)
    optimizer = AdamW(delta_params, lr=lr, betas=(0.9, 0.999), eps=1e-15)

    if train_latents_variants is None:
        train_latents_variants = [{"latents": train_latents, "name": "orig"}]

    def _save_fn():
        state = [copy.deepcopy(d.data) for d in wrapper.deltas]
        if wrapper.delta_final is not None:
            state.append(copy.deepcopy(wrapper.delta_final.data))
        return state

    wrapper.train()
    losses = []

    es_check_time = 0.0
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
        for p in delta_params:
            if p.grad is not None:
                torch.nn.utils.clip_grad_norm_([p], 1.0)
        optimizer.step()

        losses.append(loss.item())

        if early_stopper is not None:
            es_t0 = time.time()
            should_stop, es_info = early_stopper.step(
                step + 1, save_fn=_save_fn,
            )
            es_check_time += time.time() - es_t0
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    es_state = None
    if early_stopper is not None:
        def _restore_fn(saved):
            for d, s in zip(wrapper.deltas, saved[:len(wrapper.deltas)]):
                d.data.copy_(s)
            if wrapper.delta_final is not None and len(saved) > len(wrapper.deltas):
                wrapper.delta_final.data.copy_(saved[-1])
        early_stopper.restore(restore_fn=_restore_fn)
        es_state = early_stopper.state

    delta_norms = [d.detach().norm().item() for d in wrapper.deltas]
    if wrapper.delta_final is not None:
        delta_norms.append(wrapper.delta_final.detach().norm().item())
    return {
        "losses": losses,
        "delta_norms": delta_norms,
        "es_check_time": es_check_time,
        "early_stopping_info": es_state,
    }


def _parse_target_blocks(target_blocks: str, num_blocks: int) -> Optional[set]:
    """Parse --delta-target-blocks into a set of block indices.

    Accepts:
      "all"    -> None (every block)
      "last_N" -> last N block indices
      "0,5,10" -> explicit comma-separated indices
    """
    target_blocks = target_blocks.strip().lower()
    if target_blocks == "all":
        return None
    if target_blocks.startswith("last_"):
        n = int(target_blocks.split("_", 1)[1])
        if n <= 0 or n > num_blocks:
            raise ValueError(f"last_{n} invalid for {num_blocks} blocks")
        return set(range(num_blocks - n, num_blocks))
    indices = set(int(x.strip()) for x in target_blocks.split(","))
    for idx in indices:
        if idx < 0 or idx >= num_blocks:
            raise ValueError(f"Block index {idx} out of range [0, {num_blocks})")
    return indices


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
    parser.add_argument("--delta-target", type=str, default="timestep",
                        choices=["timestep", "hidden"],
                        help="Where to inject delta: 'timestep' adds to t "
                             "(adaLN input, 512-dim), 'hidden' adds to block "
                             "output (hidden state residual, 4096-dim)")
    parser.add_argument("--delta-dim", type=int, default=None,
                        help="Learn only the first k dimensions of the delta "
                             "vector, zero-padding the rest. Defaults to full "
                             "dim (512 for timestep, 4096 for hidden).")
    parser.add_argument("--delta-target-blocks", type=str, default="all",
                        help="Which DiT blocks to apply Delta-B to. "
                             "'all' = every block, 'last_N' = last N blocks "
                             "(e.g. 'last_4'), or comma-separated block indices.")
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
    parser.add_argument("--no-save-videos", action="store_true",
                        help="Delete generated videos after evaluation to save disk space")
    add_early_stopping_args(parser)
    add_augmentation_args(parser)
    add_tta_frame_args(parser)
    add_caption_guard_args(parser)
    add_caption_override_args(parser)
    add_clip_gate_args(parser)
    args = parser.parse_args()

    # Default tta_total_frames to num_cond_frames (historical behavior)
    if args.tta_total_frames is None:
        args.tta_total_frames = args.num_cond_frames
    # Default tta_context_frames to match generation conditioning
    if args.tta_context_frames is None or args.tta_context_frames > args.tta_total_frames:
        args.tta_context_frames = args.num_cond_frames
    # Safety: never let TTA include anchor/future GT frames.
    if args.tta_total_frames > args.gen_start_frame:
        print(
            f"[WARN] tta_total_frames ({args.tta_total_frames}) exceeds "
            f"gen_start_frame ({args.gen_start_frame}); clamping to avoid GT leakage."
        )
        args.tta_total_frames = args.gen_start_frame
    if args.tta_context_frames > args.tta_total_frames:
        args.tta_context_frames = args.tta_total_frames

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
    print(f"Delta target   : {args.delta_target}")
    print(f"Target blocks  : {args.delta_target_blocks}")
    print(f"Delta steps    : {args.delta_steps}")
    print(f"Delta LR       : {args.delta_lr}")
    print(f"Augmentation   : {args.aug_enabled}")
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

    import functools
    from torch.utils.checkpoint import checkpoint as _ckpt_fn
    dit.gradient_checkpointing = True
    dit._gradient_checkpointing_func = functools.partial(_ckpt_fn, use_reentrant=False)
    print("Gradient checkpointing: ENABLED (use_reentrant=False)")

    adaln_dim = dit.config.adaln_tembed_dim

    from common import load_ucf101_video_list
    videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed, validate_decodable=True
    )
    videos = apply_fixed_caption(videos, args.fixed_caption, context="eval")
    validate_caption_quality(
        videos,
        mode=args.caption_guard_mode,
        min_nonempty_ratio=args.caption_guard_min_nonempty_ratio,
        min_unique_ratio=args.caption_guard_min_unique_ratio,
        max_top1_ratio=args.caption_guard_max_top1_ratio,
        max_generic_top1_ratio=args.caption_guard_max_generic_top1_ratio,
        top_k=args.caption_guard_topk,
        context="eval",
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

        clip_gate_info = evaluate_clip_gate(
            video_path=video_path,
            caption=caption,
            gen_start_frame=args.gen_start_frame,
            tta_total_frames=args.tta_total_frames,
            device=args.device,
            enabled=args.clip_gate_enabled,
            threshold=args.clip_gate_threshold,
            backend=args.clip_gate_backend,
            model_name=args.clip_gate_model,
            sample_frames=args.clip_gate_sample_frames,
            aggregation=args.clip_gate_aggregation,
            sampling_mode=args.clip_gate_sampling_mode,
            late_fraction=args.clip_gate_late_fraction,
            late_only=args.clip_gate_late_only,
            fail_open=args.clip_gate_fail_open,
            log_only=args.clip_gate_log_only,
        )
        if clip_gate_info.get("clip_alignment_score") is not None:
            print(
                "  CLIP gate: "
                f"score={clip_gate_info['clip_alignment_score']:.4f}, "
                f"decision={clip_gate_info['clip_gate_decision']}, "
                f"mode={clip_gate_info['clip_gate_sampling_mode']}"
            )
        elif clip_gate_info.get("clip_gate_enabled"):
            print(f"  CLIP gate: decision={clip_gate_info['clip_gate_decision']} "
                  f"({clip_gate_info.get('clip_gate_reason', 'n/a')})")

        try:
            wrapper = None
            if clip_gate_info.get("tta_skipped", False):
                gen_cond_start = args.gen_start_frame - args.num_cond_frames
                gen_pixel_frames = load_video_frames(
                    video_path, args.num_cond_frames, height=480, width=832,
                    start_frame=max(0, gen_cond_start),
                ).to(args.device, torch.bfloat16)
                opt_result = {
                    "losses": [],
                    "delta_norms": [0.0 for _ in range(args.num_groups)],
                    "es_check_time": 0.0,
                    "early_stopping_info": None,
                }
                train_time = 0.0
            else:
                # ── Frame loading ─────────────────────────────────────────
                tta_start = args.gen_start_frame - args.tta_total_frames
                pixel_frames = load_video_frames(
                    video_path, args.tta_total_frames, height=480, width=832,
                    start_frame=max(0, tta_start),
                ).to(args.device, torch.bfloat16)

                all_latents = encode_video(vae, pixel_frames, normalize=True)

                # Split into context / train / val
                vae_t_scale = 4
                num_ctx_lat = 1 + (args.tta_context_frames - 1) // vae_t_scale
                cond_latents, train_latents, val_latents = split_tta_latents(
                    all_latents, num_ctx_lat,
                    holdout_fraction=getattr(args, "es_holdout_fraction", 0.25),
                )
                print(f"  Latent split: cond={cond_latents.shape[2]}, "
                      f"train={train_latents.shape[2]}, "
                      f"val={val_latents.shape[2] if val_latents is not None else 0}")

                # Generation conditioning frames
                gen_cond_start = args.gen_start_frame - args.num_cond_frames
                gen_pixel_frames = load_video_frames(
                    video_path, args.num_cond_frames, height=480, width=832,
                    start_frame=max(0, gen_cond_start),
                ).to(args.device, torch.bfloat16)

                prompt_embeds, prompt_mask = encode_prompt(
                    tokenizer, text_encoder, caption,
                    device=args.device, dtype=torch.bfloat16,
                )

                # Build augmented train latent variants if enabled
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
                    print(f"  Augmentation: {len(train_latents_variants)} variants "
                          f"({', '.join(v['name'] for v in train_latents_variants)})")

                # Fresh wrapper per video
                wrapper = DeltaBWrapper(
                    dit, num_groups=args.num_groups, adaln_tembed_dim=adaln_dim,
                    hidden_size=dit.config.hidden_size,
                    delta_target=args.delta_target,
                    delta_dim=getattr(args, 'delta_dim', None),
                target_blocks=args.delta_target_blocks,
                ).to(args.device)

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
                        save_fn=lambda: (
                            [copy.deepcopy(d.data) for d in wrapper.deltas]
                            + ([copy.deepcopy(wrapper.delta_final.data)]
                               if wrapper.delta_final is not None else [])
                        ),
                    )

                # Offload VAE + text encoder to CPU during training
                vae.to("cpu")
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

                t0 = time.time()
                opt_result = optimize_delta_b(
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
                "es_check_time": opt_result.get("es_check_time", 0.0),
                "final_loss": opt_result["losses"][-1] if opt_result["losses"] else None,
                "delta_norms": opt_result["delta_norms"],
                "num_groups": args.num_groups,
                "early_stopping_info": opt_result.get("early_stopping_info"),
                "success": True,
            }
            result.update(clip_gate_info)

            print(
                  f"  CLIP: {clip_gate_info.get('clip_gate_eval_time', 0.0):.2f}s, "
                  f"ES: {opt_result.get('es_check_time', 0.0):.2f}s, "
                  f"Train time: {train_time:.1f}s, "
                  + (f"Final loss: {result['final_loss']:.4f}, "
                     if result.get("final_loss") is not None else "Final loss: N/A (skipped), ")
                  + f"Norms: {opt_result['delta_norms']}")

            # ── Generation ──────────────────────────────────────────
            # Bring VAE + text encoder back to GPU for generation
            if not clip_gate_info.get("tta_skipped", False):
                vae.to(args.device)
                text_encoder.to(args.device)

            gen_time = 0.0
            if not args.skip_generation:
                from PIL import Image

                pf = gen_pixel_frames.squeeze(0)  # [C, T, H, W]
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_images = []
                for t_idx in range(pf.shape[1]):
                    frame_np = (pf[:, t_idx].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                    cond_images.append(Image.fromarray(frame_np))

                if clip_gate_info.get("tta_skipped", False):
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
                    if not args.no_save_videos:
                        prompt_slug = _slugify_text(caption)
                        clip_tag = "clip-skip" if clip_gate_info.get("tta_skipped", False) else "clip-tta"
                        output_name = (
                            f"{idx+1:03d}_{clip_tag}_{prompt_slug}_"
                            f"psnr-{metrics['psnr']:.3f}_"
                            f"ssim-{metrics['ssim']:.3f}_"
                            f"lpips-{metrics['lpips']:.3f}.mp4"
                        )
                        output_path = os.path.join(videos_dir, output_name)
                        save_video_from_numpy(gen_frames, output_path, fps=24)
                        result["output_path"] = output_path
                    print(f"  Gen: {gen_time:.1f}s, "
                          f"PSNR={metrics['psnr']:.2f}, "
                          f"SSIM={metrics['ssim']:.4f}, "
                          f"LPIPS={metrics['lpips']:.4f}")
                else:
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
                        if not args.no_save_videos:
                            prompt_slug = _slugify_text(caption)
                            clip_tag = "clip-skip" if clip_gate_info.get("tta_skipped", False) else "clip-tta"
                            output_name = (
                                f"{idx+1:03d}_{clip_tag}_{prompt_slug}_"
                                f"psnr-{metrics['psnr']:.3f}_"
                                f"ssim-{metrics['ssim']:.3f}_"
                                f"lpips-{metrics['lpips']:.3f}.mp4"
                            )
                            output_path = os.path.join(videos_dir, output_name)
                            save_video_from_numpy(gen_frames, output_path, fps=24)
                            result["output_path"] = output_path
                        print(f"  Gen: {gen_time:.1f}s, "
                              f"PSNR={metrics['psnr']:.2f}, "
                              f"SSIM={metrics['ssim']:.4f}, "
                              f"LPIPS={metrics['lpips']:.4f}")
                    finally:
                        wrapper.remove_from_dit()

            result["total_time"] = (
                float(clip_gate_info.get("clip_gate_eval_time", 0.0))
                + train_time
                + gen_time
            )
            all_results.append(result)

            # Variables are re-created each loop; rely on explicit GC for cleanup.
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
        "delta_target": args.delta_target,
        "delta_target_blocks": args.delta_target_blocks,
        "num_groups": args.num_groups,
        "delta_steps": args.delta_steps,
        "delta_lr": args.delta_lr,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "gen_start_frame": args.gen_start_frame,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "avg_train_time": np.mean([r.get("train_time", 0) for r in successful]) if successful else 0,
        "avg_clip_gate_eval_time": (
            np.mean([r.get("clip_gate_eval_time", 0.0) for r in successful]) if successful else 0
        ),
        "avg_es_check_time": (
            np.mean([r.get("es_check_time", 0.0) for r in successful]) if successful else 0
        ),
        "avg_gen_time": (
            np.mean([r.get("gen_time", 0.0) for r in successful]) if successful else 0
        ),
        "avg_total_time": (
            np.mean([r.get("total_time", 0.0) for r in successful]) if successful else 0
        ),
        "clip_gate_enabled": args.clip_gate_enabled,
        "clip_gate_threshold": args.clip_gate_threshold,
        "clip_gate_backend": args.clip_gate_backend,
        "clip_gate_model": args.clip_gate_model,
        "clip_gate_sample_frames": args.clip_gate_sample_frames,
        "clip_gate_aggregation": args.clip_gate_aggregation,
        "clip_gate_sampling_mode": "late_only" if args.clip_gate_late_only else args.clip_gate_sampling_mode,
        "clip_gate_late_fraction": args.clip_gate_late_fraction,
        "clip_gate_log_only": args.clip_gate_log_only,
        "clip_gate_fail_open": args.clip_gate_fail_open,
        "clip_gate_stats": summarize_clip_gate_stats(successful),
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))
    print(f"\nResults saved to {args.output_dir}/summary.json")
    if successful:
        print(f"Avg CLIP gate time: {summary['avg_clip_gate_eval_time']:.2f}s")
        print(f"Avg ES check time : {summary['avg_es_check_time']:.2f}s")
        print(f"Avg train time: {summary['avg_train_time']:.1f}s")
        print(f"Avg gen time: {summary['avg_gen_time']:.1f}s")
        print(f"Avg total time: {summary['avg_total_time']:.1f}s")


if __name__ == "__main__":
    main()
