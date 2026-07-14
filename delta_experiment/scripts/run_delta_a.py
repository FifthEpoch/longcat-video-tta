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
import hashlib
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
    compute_flow_matching_loss_conditioned_fixed_grad,
    generate_video_continuation,
    save_results,
    save_video_from_numpy,
    rename_videos_with_metrics,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    build_augmented_latent_variants,
    add_augmentation_args,
    add_tta_frame_args,
    add_caption_guard_args,
    add_caption_override_args,
    add_tta_disable_caption_args,
    tta_caption_for,
    add_feature_frame_guard_args,
    add_clip_gate_args,
    parse_speed_factors,
    split_tta_latents,
    evaluate_generation_metrics,
    build_retrieval_pool,
    retrieve_neighbors,
    evaluate_clip_gate,
    summarize_clip_gate_stats,
    validate_caption_quality,
    apply_fixed_caption,
    validate_tta_feature_budget,
    add_online_eval_args,
    OnlineFrechetAccumulator,
    finalize_online_eval,
    aggregate_quality_metrics,
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
    grad_accum: int = 1,
    anchor_reg_latents: Optional[torch.Tensor] = None,
    anchor_reg_weight: float = 0.0,
    anchor_reg_sigmas: Optional[List[float]] = None,
    anchor_reg_noise_draws: int = 1,
    anchor_reg_video_id: str = "",
    anchor_x0_weight: float = 0.0,
) -> Dict:
    """Optimize the delta vector using conditioning-aware loss.

    Parameters
    ----------
    cond_latents  : clean context latents [B, C, T_cond, H, W]
    train_latents : target latents to noise and compute loss on [B, C, T_train, H, W]
    train_latents_variants : optional augmented variants of train_latents
    grad_accum : number of forward/backward passes per optimizer step.
                 Each pass uses a fresh noise sample, giving a better
                 gradient estimate without increasing peak GPU memory.
    """
    optimizer = AdamW([wrapper.delta], lr=lr, betas=(0.9, 0.999), eps=1e-15)

    if train_latents_variants is None:
        train_latents_variants = [{"latents": train_latents, "name": "orig"}]
    anchor_reg_sigmas = anchor_reg_sigmas or [0.25, 0.5, 0.75]
    anchor_reg_enabled = (
        anchor_reg_weight > 0.0
        and anchor_reg_latents is not None
        and anchor_reg_latents.shape[2] > 0
    )
    anchor_reg_noises = []
    if anchor_reg_enabled:
        seed_base = int(hashlib.md5(anchor_reg_video_id.encode()).hexdigest()[:8], 16) % (2**31)
        for draw_idx in range(anchor_reg_noise_draws):
            gen = torch.Generator(device=device)
            gen.manual_seed(seed_base + 1009 + draw_idx)
            anchor_reg_noises.append(torch.randn(
                anchor_reg_latents.shape, generator=gen,
                device=device, dtype=anchor_reg_latents.dtype,
            ))

    def _save_fn():
        return copy.deepcopy(wrapper.delta.data)

    wrapper.train()
    losses = []
    base_losses = []
    anchor_reg_losses = []
    raw_grad_norms = []

    es_check_time = 0.0
    for step in range(num_steps):
        optimizer.zero_grad()

        step_loss_sum = 0.0
        for _ga in range(grad_accum):
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
                anchor_x0_weight=anchor_x0_weight,
            )
            base_loss = loss
            anchor_loss = None
            if anchor_reg_enabled:
                anchor_loss = compute_flow_matching_loss_conditioned_fixed_grad(
                    dit=wrapper,
                    cond_latents=cond_latents,
                    target_latents=anchor_reg_latents,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    fixed_sigmas=anchor_reg_sigmas,
                    fixed_noises=anchor_reg_noises,
                    device=device,
                    dtype=dtype,
                )
                loss = base_loss + anchor_reg_weight * anchor_loss
                anchor_reg_losses.append(anchor_loss.item())
            (loss / grad_accum).backward()
            step_loss_sum += loss.item()
            base_losses.append(base_loss.item())

        raw_norm = torch.nn.utils.clip_grad_norm_([wrapper.delta], float("inf")).item()
        raw_grad_norms.append(raw_norm)
        if raw_norm > 1.0:
            scale = 1.0 / (raw_norm + 1e-6)
            wrapper.delta.grad.mul_(scale)
        optimizer.step()

        losses.append(step_loss_sum / grad_accum)

        if early_stopper is not None:
            es_t0 = time.time()
            should_stop, es_info = early_stopper.step(
                step + 1, save_fn=_save_fn,
            )
            es_check_time += time.time() - es_t0
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    if raw_grad_norms:
        clipped_count = sum(1 for n in raw_grad_norms if n > 1.0)
        print(f"  Grad norms: min={min(raw_grad_norms):.2f} "
              f"max={max(raw_grad_norms):.2f} "
              f"mean={sum(raw_grad_norms)/len(raw_grad_norms):.2f} "
              f"clipped={clipped_count}/{len(raw_grad_norms)}")

    es_state = None
    if early_stopper is not None:
        early_stopper.restore(
            restore_fn=lambda s: wrapper.delta.data.copy_(s)
        )
        es_state = early_stopper.state

    return {
        "losses": losses,
        "base_losses": base_losses,
        "anchor_reg_losses": anchor_reg_losses,
        "delta_norm": wrapper.delta.detach().norm().item(),
        "raw_grad_norms": raw_grad_norms,
        "es_check_time": es_check_time,
        "early_stopping_info": es_state,
    }


def compute_anchor_gate_info(
    es_state: Optional[Dict],
    mode: str,
    threshold: float,
    soft_scale: float,
) -> Dict:
    """Turn anchor-loss early-stopping state into a generation-time gate."""
    mode = (mode or "off").lower()
    info = {
        "anchor_gate_mode": mode,
        "anchor_gate_threshold": threshold,
        "anchor_gate_soft_scale": soft_scale,
        "anchor_gate_enabled": mode != "off",
        "anchor_gate_decision": "use",
        "anchor_gate_scale": 1.0,
        "anchor_gate_reason": "disabled" if mode == "off" else "",
        "anchor_gate_initial_loss": None,
        "anchor_gate_best_loss": None,
        "anchor_gate_improvement": None,
        "anchor_gate_relative_improvement": None,
        "anchor_gate_best_step": None,
    }
    if mode == "off":
        return info

    if not es_state:
        info.update({
            "anchor_gate_decision": "use",
            "anchor_gate_reason": "missing_anchor_state",
        })
        return info

    initial = es_state.get("initial_loss")
    best = es_state.get("best_loss")
    best_step = int(es_state.get("best_step", 0) or 0)
    info["anchor_gate_initial_loss"] = initial
    info["anchor_gate_best_loss"] = best
    info["anchor_gate_best_step"] = best_step
    if initial is None or best is None or initial <= 0:
        info.update({
            "anchor_gate_decision": "use",
            "anchor_gate_reason": "invalid_anchor_losses",
        })
        return info

    improvement = float(initial) - float(best)
    rel = improvement / max(abs(float(initial)), 1e-12)
    info["anchor_gate_improvement"] = improvement
    info["anchor_gate_relative_improvement"] = rel

    passes = best_step > 0 and rel >= threshold
    if mode == "log_only":
        info.update({
            "anchor_gate_decision": "use",
            "anchor_gate_reason": "log_only_pass" if passes else "log_only_fail",
        })
    elif mode == "binary":
        info.update({
            "anchor_gate_decision": "use" if passes else "skip",
            "anchor_gate_scale": 1.0 if passes else 0.0,
            "anchor_gate_reason": "anchor_pass" if passes else "anchor_fail",
        })
    elif mode == "soft":
        scale = min(1.0, max(0.0, rel / max(soft_scale, 1e-12)))
        if rel < threshold or best_step == 0:
            scale = 0.0
        info.update({
            "anchor_gate_decision": "use" if scale > 0 else "skip",
            "anchor_gate_scale": scale,
            "anchor_gate_reason": "anchor_soft_scale" if scale > 0 else "anchor_fail",
        })
    else:
        raise ValueError(f"Unknown anchor gate mode: {mode}")
    return info


def _optimize_delta_a_batch(
    wrapper: DeltaAWrapper,
    batch_data: List[Dict],
    num_steps: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    anchor_x0_weight: float = 0.0,
) -> Dict:
    """Optimize a shared delta vector across multiple videos.

    At each training step, one video is randomly sampled from the batch.
    Its conditioning and training latents are loaded to GPU, the flow-matching
    loss is computed, and the shared delta is updated. This acts as natural
    regularization: the delta must improve denoising across diverse content.
    """
    optimizer = AdamW([wrapper.delta], lr=lr, betas=(0.9, 0.999), eps=1e-15)

    wrapper.train()
    losses = []
    raw_grad_norms = []
    n_vids = len(batch_data)

    for step in range(num_steps):
        optimizer.zero_grad()

        vi = step % n_vids
        bd = batch_data[vi]

        cond_lat = bd["cond_latents"].to(device)
        train_lat = bd["train_latents"].to(device)
        pe = bd["prompt_embeds"].to(device)
        pm = bd["prompt_mask"].to(device) if bd["prompt_mask"] is not None else None

        loss = compute_flow_matching_loss_conditioned(
            dit=wrapper,
            cond_latents=cond_lat,
            target_latents=train_lat,
            prompt_embeds=pe,
            prompt_mask=pm,
            device=device,
            dtype=dtype,
            anchor_x0_weight=anchor_x0_weight,
        )

        loss.backward()
        raw_norm = torch.nn.utils.clip_grad_norm_([wrapper.delta], float("inf")).item()
        raw_grad_norms.append(raw_norm)
        if raw_norm > 1.0:
            scale = 1.0 / (raw_norm + 1e-6)
            wrapper.delta.grad.mul_(scale)
        optimizer.step()

        losses.append(loss.item())

        del cond_lat, train_lat, pe, pm

    if raw_grad_norms:
        clipped_count = sum(1 for n in raw_grad_norms if n > 1.0)
        print(f"  Grad norms: min={min(raw_grad_norms):.2f} "
              f"max={max(raw_grad_norms):.2f} "
              f"mean={sum(raw_grad_norms)/len(raw_grad_norms):.2f} "
              f"clipped={clipped_count}/{len(raw_grad_norms)}")

    return {
        "losses": losses,
        "delta_norm": wrapper.delta.detach().norm().item(),
        "raw_grad_norms": raw_grad_norms,
        "es_check_time": 0.0,
        "early_stopping_info": None,
    }


def _summarize_grad_norms(results):
    """Aggregate raw gradient norm statistics across all videos."""
    all_norms = []
    for r in results:
        all_norms.extend(r.get("raw_grad_norms", []))
    if not all_norms:
        return {}
    clipped = sum(1 for n in all_norms if n > 1.0)
    return {
        "total_steps": len(all_norms),
        "clipped_steps": clipped,
        "clip_rate": clipped / len(all_norms),
        "min": min(all_norms),
        "max": max(all_norms),
        "mean": sum(all_norms) / len(all_norms),
        "median": sorted(all_norms)[len(all_norms) // 2],
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
    parser.add_argument("--start-video-idx", type=int, default=0,
                        help="Start processing from this index in the video list (for chunked runs)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Number of videos to process from start-video-idx (0 = all remaining)")
    parser.add_argument("--delta-steps", type=int, default=20)
    parser.add_argument("--delta-lr", type=float, default=1e-3)
    parser.add_argument(
        "--initial-delta-dir", type=str, default=None,
        help="Directory of per-video .pt delta tensors (from --save-delta-dir) "
             "to continue TTA without re-training from scratch. Loads "
             "<video_name>.pt before the optimizer loop.",
    )
    parser.add_argument(
        "--save-delta-dir", type=str, default=None,
        help="After TTA, save wrapper.delta to <dir>/<video_name>.pt before "
             "generation. Used for incremental S10→S20 workflows.",
    )
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
    parser.add_argument("--batch-videos", type=int, default=1,
                        help="Number of videos per TTA batch. 1=instance-level (default), "
                             "K>1=retrieval-augmented batch-level (train on eval video + "
                             "K-1 nearest neighbours from the retrieval pool).")
    parser.add_argument("--tta-grad-accum", type=int, default=1,
                        help="Gradient accumulation steps per optimizer update. "
                             "Each accumulation draws a fresh noise sample, giving "
                             "a better gradient estimate without extra GPU memory. "
                             "Effective batch = tta_grad_accum noise draws per step.")
    parser.add_argument("--batch-method", type=str, default="similarity",
                        choices=["sequential", "similarity"],
                        help="(Legacy) Retrieval is always by text-prompt similarity.")
    parser.add_argument("--rollout-steps", type=int, default=1,
                        help="Number of autoregressive rollout steps. After generating "
                             "one chunk, the last num_cond_frames of output become "
                             "conditioning for the next chunk. The learned delta is held "
                             "fixed across all rollout steps.")
    parser.add_argument("--retrieval-pool-dir", type=str, default=None,
                        help="Directory containing the larger retrieval pool dataset "
                             "(e.g. 1000 videos). Required when --batch-videos > 1. "
                             "Eval videos come from --data-dir; neighbours come from here.")
    parser.add_argument("--anchor-gate-mode", type=str, default="off",
                        choices=["off", "log_only", "binary", "soft"],
                        help="Use anchor-loss early-stopping state as a generation-time gate.")
    parser.add_argument("--anchor-gate-threshold", type=float, default=0.0,
                        help="Minimum relative anchor-loss improvement required to apply TTA.")
    parser.add_argument("--anchor-gate-soft-scale", type=float, default=0.01,
                        help="Relative anchor improvement that maps to scale=1.0 in soft mode.")
    parser.add_argument("--anchor-reg-weight", type=float, default=0.0,
                        help="Weight for differentiable fixed-sigma anchor regularization on held-out latents.")
    parser.add_argument("--anchor-reg-sigmas", type=str, default="0.25,0.5,0.75",
                        help="Comma-separated fixed sigma values for anchor regularization.")
    parser.add_argument("--anchor-reg-noise-draws", type=int, default=1,
                        help="Number of deterministic noise draws per sigma for anchor regularization.")
    parser.add_argument("--anchor-x0-weight", type=float, default=0.0,
                        help="Weight for the anchor-frame x0 consistency auxiliary loss "
                             "(rectified-flow recovery; Modification 1 of "
                             "sweep_experiment/reports/LITERATURE_tta_recipe_modifications_2026-06-12.md). "
                             "Default 0.0 = byte-identical to pre-patch behaviour.")
    add_early_stopping_args(parser)
    add_augmentation_args(parser)
    add_tta_frame_args(parser)
    add_caption_guard_args(parser)
    add_caption_override_args(parser)
    add_tta_disable_caption_args(parser)
    add_feature_frame_guard_args(parser)
    add_online_eval_args(parser)
    add_clip_gate_args(parser)
    args = parser.parse_args()

    # Default tta_total_frames to gen_start_frame (use all pre-anchor frames)
    if args.tta_total_frames is None:
        args.tta_total_frames = args.gen_start_frame
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
    validate_tta_feature_budget(args, context="delta_a")
    args.anchor_reg_sigmas_parsed = [float(x) for x in args.anchor_reg_sigmas.split(",") if x]

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support
    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = 0
    _ckpt_results = []
    if ckpt:
        start_idx = ckpt.get("next_idx", 0)
        _ckpt_results = ckpt.get("results", [])

    method_label = "Delta-A (AdaSteer)"
    if args.anchor_x0_weight > 0.0:
        method_label = f"Delta-A (AdaSteer) + x0 (λ={args.anchor_x0_weight:g})"

    print("=" * 70)
    print(f"{method_label} TTA for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"Delta steps    : {args.delta_steps}")
    print(f"Delta LR       : {args.delta_lr}")
    print(f"Anchor-x0 wt   : {args.anchor_x0_weight}")
    print(f"Augmentation   : {args.aug_enabled}")
    print(f"Rollout steps  : {args.rollout_steps}")
    print(f"Grad accum     : {args.tta_grad_accum}")
    print(f"Batch videos   : {args.batch_videos}")
    print(f"Batch method   : {args.batch_method}")
    print(f"TTA no-caption : {args.tta_disable_caption}")
    if args.retrieval_pool_dir:
        print(f"Retrieval pool : {args.retrieval_pool_dir}")
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

    # Load evaluation videos (always from --data-dir, same 100 as usual)
    from common import load_ucf101_video_list
    eval_videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed, validate_decodable=True
    )
    eval_videos = apply_fixed_caption(eval_videos, args.fixed_caption, context="eval")
    validate_caption_quality(
        eval_videos,
        mode=args.caption_guard_mode,
        min_nonempty_ratio=args.caption_guard_min_nonempty_ratio,
        min_unique_ratio=args.caption_guard_min_unique_ratio,
        max_top1_ratio=args.caption_guard_max_top1_ratio,
        max_generic_top1_ratio=args.caption_guard_max_generic_top1_ratio,
        top_k=args.caption_guard_topk,
        context="eval",
    )
    if args.start_video_idx > 0 or args.chunk_size > 0:
        end = len(eval_videos)
        if args.chunk_size > 0:
            end = min(args.start_video_idx + args.chunk_size, end)
        eval_videos = eval_videos[args.start_video_idx:end]
        print(f"Chunk: videos [{args.start_video_idx}:{end}] → {len(eval_videos)} videos")

    print(f"\nEvaluation videos: {len(eval_videos)}")

    # Build retrieval pool for batch-level TTA
    batch_level = args.batch_videos > 1
    pool_entries = None
    pool_embeddings = None
    st_model = None

    if batch_level:
        pool_dir = args.retrieval_pool_dir or args.data_dir
        if pool_dir == args.data_dir:
            print(f"\nWARNING: --retrieval-pool-dir not set; using --data-dir as pool. "
                  f"For proper retrieval-augmented TTA, provide a larger pool dataset.",
                  file=sys.stderr)

        pool_entries = load_ucf101_video_list(
            pool_dir, max_videos=999999, seed=args.seed
        )
        pool_entries = apply_fixed_caption(pool_entries, args.fixed_caption, context="retrieval_pool")
        validate_caption_quality(
            pool_entries,
            mode=args.caption_guard_mode,
            min_nonempty_ratio=args.caption_guard_min_nonempty_ratio,
            min_unique_ratio=args.caption_guard_min_unique_ratio,
            max_top1_ratio=args.caption_guard_max_top1_ratio,
            max_generic_top1_ratio=args.caption_guard_max_generic_top1_ratio,
            top_k=args.caption_guard_topk,
            context="retrieval_pool",
        )
        print(f"Retrieval pool: {len(pool_entries)} videos from {pool_dir}")
        pool_embeddings, st_model = build_retrieval_pool(pool_entries)

    # Build early stopper
    early_stopper = build_early_stopper_from_args(args)

    # Results (restore from checkpoint if resuming)
    all_results = list(_ckpt_results)
    videos_dir = os.path.join(args.output_dir, "videos")
    fvd_accumulator = OnlineFrechetAccumulator(
        device=args.device, compute_fid=args.compute_fid,
        min_videos=args.min_fvd_videos,
        gt_cache_path=getattr(args, "gt_features_cache", None),
    ) if args.compute_fvd else None
    fvd_ckpt_path = os.path.join(args.output_dir, "fvd_checkpoint.npz")
    if fvd_accumulator is not None and start_idx > 0:
        fvd_accumulator.load_stats(fvd_ckpt_path)
    if not args.skip_generation and not args.no_save_videos:
        os.makedirs(videos_dir, exist_ok=True)

    # ── Per-video loop ──
    for v_idx, eval_entry in enumerate(eval_videos):
        if v_idx < start_idx:
            continue

        eval_name = Path(eval_entry["video_path"]).stem
        print(f"\n{'='*70}")
        print(f"[{v_idx + 1}/{len(eval_videos)}] {eval_name}")

        clip_gate_info = evaluate_clip_gate(
            video_path=eval_entry["video_path"],
            caption=eval_entry["caption"],
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

        # Build training batch: eval video + K-1 nearest neighbours
        if batch_level and not clip_gate_info.get("tta_skipped", False):
            neighbors = retrieve_neighbors(
                eval_entry, pool_entries, pool_embeddings, st_model,
                k=args.batch_videos,
            )
            training_entries = [eval_entry] + neighbors
            print(f"  Batch: 1 eval + {len(neighbors)} retrieved neighbours "
                  f"(total {len(training_entries)})")
            for ni, ne in enumerate(neighbors[:5]):
                print(f"    neighbour {ni+1}: {Path(ne['video_path']).stem} "
                      f"-- \"{ne['caption'][:60]}\"")
            if len(neighbors) > 5:
                print(f"    ... and {len(neighbors) - 5} more")
        else:
            training_entries = [eval_entry]
            if batch_level and clip_gate_info.get("tta_skipped", False):
                print("  CLIP gate triggered: skip TTA for this sample, neighbors ignored.")

        try:
            wrapper = None
            if clip_gate_info.get("tta_skipped", False):
                gen_cond_start = args.gen_start_frame - args.num_cond_frames
                gen_pixel_frames = load_video_frames(
                    eval_entry["video_path"], args.num_cond_frames,
                    height=480, width=832, start_frame=max(0, gen_cond_start),
                ).to(args.device, torch.bfloat16).cpu()
                opt_result = {
                    "losses": [],
                    "delta_norm": 0.0,
                    "es_check_time": 0.0,
                    "early_stopping_info": None,
                }
                train_time = 0.0
                timing = {k: 0.0 for k in [
                    "load_frames", "encode_latents", "encode_prompt",
                    "aug_build", "aug_encode", "tta_train", "es_setup",
                    "es_check", "tta_train_net", "train_total",
                ]}
            else:
                # ── Detailed per-module timing ──
                timing = {
                    "load_frames": 0.0,
                    "encode_latents": 0.0,
                    "encode_prompt": 0.0,
                    "aug_build": 0.0,
                    "aug_encode": 0.0,
                    "tta_train": 0.0,
                    "es_setup": 0.0,
                }

                # ── Pre-encode all videos in the training batch ──
                _cached_gen_cond_frames = None
                _cached_tta_pixels = None
                # Only hold out a val split if it will actually be consumed
                # (early stopping or anchor regularization). Otherwise adapt on
                # all observed frames — carving an unused val split silently
                # discards ~25% of the available TTA adaptation signal.
                _use_val = early_stopper is not None or float(
                    getattr(args, "anchor_reg_weight", 0.0) or 0.0
                ) > 0.0
                _tta_holdout = (
                    getattr(args, "es_holdout_fraction", 0.25) if _use_val else 0.0
                )
                batch_data = []
                for entry in training_entries:
                    video_path = entry["video_path"]
                    caption = entry["caption"]

                    tta_start = args.gen_start_frame - args.tta_total_frames
                    _t = time.time()
                    pixel_frames = load_video_frames(
                        video_path, args.tta_total_frames, height=480, width=832,
                        start_frame=max(0, tta_start),
                    ).to(args.device, torch.bfloat16)
                    timing["load_frames"] += time.time() - _t

                    _t = time.time()
                    all_latents = encode_video(vae, pixel_frames, normalize=True)
                    timing["encode_latents"] += time.time() - _t

                    vae_t_scale = 4
                    num_ctx_lat = 1 + (args.tta_context_frames - 1) // vae_t_scale
                    cond_latents, train_latents, val_latents = split_tta_latents(
                        all_latents, num_ctx_lat,
                        holdout_fraction=_tta_holdout,
                    )

                    _t = time.time()
                    prompt_embeds, prompt_mask = encode_prompt(
                        tokenizer, text_encoder,
                        tta_caption_for(args, caption),
                        device=args.device, dtype=torch.bfloat16,
                    )
                    timing["encode_prompt"] += time.time() - _t

                    batch_data.append({
                        "video_path": video_path,
                        "video_name": Path(video_path).stem,
                        "caption": caption,
                        "cond_latents": cond_latents.cpu(),
                        "train_latents": train_latents.cpu(),
                        "val_latents": val_latents.cpu() if val_latents is not None else None,
                        "prompt_embeds": prompt_embeds.cpu(),
                        "prompt_mask": prompt_mask.cpu() if prompt_mask is not None else None,
                    })

                    if _cached_gen_cond_frames is None and args.tta_total_frames >= args.num_cond_frames:
                        _cached_gen_cond_frames = pixel_frames[:, :, -args.num_cond_frames:].clone()
                    if _cached_tta_pixels is None and args.aug_enabled and not batch_level:
                        # First entry is the eval video; keep its TTA-window pixels
                        # on CPU so the augmentation path can reuse them instead of
                        # re-decoding the same clip from disk.
                        _cached_tta_pixels = pixel_frames.detach().to("cpu")
                    del all_latents, pixel_frames
                    torch_gc()

                if _cached_gen_cond_frames is not None:
                    gen_pixel_frames = _cached_gen_cond_frames
                    _cached_gen_cond_frames = None
                else:
                    gen_cond_start = args.gen_start_frame - args.num_cond_frames
                    gen_pixel_frames = load_video_frames(
                        eval_entry["video_path"], args.num_cond_frames,
                        height=480, width=832, start_frame=max(0, gen_cond_start),
                    ).to(args.device, torch.bfloat16).cpu()

                # ── Create fresh delta ──
                wrapper = DeltaAWrapper(dit, adaln_tembed_dim=adaln_dim).to(args.device)

                if args.initial_delta_dir:
                    init_path = os.path.join(
                        args.initial_delta_dir, f"{eval_name}.pt",
                    )
                    if os.path.isfile(init_path):
                        load_kw = {"map_location": args.device}
                        try:
                            init_delta = torch.load(
                                init_path, weights_only=True, **load_kw,
                            )
                        except TypeError:
                            init_delta = torch.load(init_path, **load_kw)
                        wrapper.delta.data.copy_(init_delta.to(wrapper.delta.dtype))
                        print(
                            f"  Loaded initial delta from {init_path} "
                            f"(norm={wrapper.delta.detach().norm().item():.4f})"
                        )
                    else:
                        print(
                            f"  [WARN] initial delta missing: {init_path} "
                            "(training from scratch for this video)"
                        )

                # Offload VAE + text encoder to CPU during training
                vae.to("cpu")
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

                # ── Train ──
                _t_train_start = time.time()
                if batch_level:
                    opt_result = _optimize_delta_a_batch(
                        wrapper=wrapper,
                        batch_data=batch_data,
                        num_steps=args.delta_steps,
                        lr=args.delta_lr,
                        device=args.device,
                        dtype=torch.bfloat16,
                        anchor_x0_weight=args.anchor_x0_weight,
                    )
                else:
                    bd = batch_data[0]
                    cond_lat = bd["cond_latents"].to(args.device)
                    train_lat = bd["train_latents"].to(args.device)
                    val_lat = bd["val_latents"].to(args.device) if bd["val_latents"] is not None else None
                    pe = bd["prompt_embeds"].to(args.device)
                    pm = bd["prompt_mask"].to(args.device) if bd["prompt_mask"] is not None else None

                    if early_stopper is not None and val_lat is not None:
                        def _es_forward_fn(hs, ts, ncl):
                            return wrapper(
                                hidden_states=hs, timestep=ts,
                                encoder_hidden_states=pe,
                                encoder_attention_mask=pm,
                                num_cond_latents=ncl,
                            )

                        _t = time.time()
                        early_stopper.setup(
                            model=wrapper,
                            cond_latents=cond_lat,
                            val_latents=val_lat,
                            prompt_embeds=pe,
                            prompt_mask=pm,
                            device=args.device,
                            dtype=torch.bfloat16,
                            forward_fn=_es_forward_fn,
                            video_id=bd["video_name"],
                            save_fn=lambda: copy.deepcopy(wrapper.delta.data),
                        )
                        timing["es_setup"] = time.time() - _t

                    train_latents_variants = None
                    if args.aug_enabled:
                        from common import build_augmented_pixel_variants
                        _tta_start = args.gen_start_frame - args.tta_total_frames
                        _t = time.time()
                        if _cached_tta_pixels is not None:
                            # Reuse pixels decoded during pre-encode (no disk re-read).
                            _pf = _cached_tta_pixels.to(args.device, torch.bfloat16)
                        else:
                            _pf = load_video_frames(
                                bd["video_path"], args.tta_total_frames,
                                height=480, width=832, start_frame=max(0, _tta_start),
                            ).to(args.device, torch.bfloat16)
                        pix_variants = build_augmented_pixel_variants(
                            _pf,
                            enable_flip=args.aug_flip,
                            rotate_deg=args.aug_rotate_deg,
                            rotate_random_min=args.aug_rotate_random_min,
                            rotate_random_max=args.aug_rotate_random_max,
                            rotate_random_count=args.aug_rotate_random_count,
                            rotate_random_step=args.aug_rotate_random_step,
                            rotate_zoom=args.aug_rotate_zoom,
                            speed_factors=parse_speed_factors(args.aug_speed_factors),
                        )
                        timing["aug_build"] = time.time() - _t

                        _t = time.time()
                        vae.to(args.device)
                        train_latents_variants = []
                        for pv in pix_variants:
                            if pv["name"] == "orig":
                                train_latents_variants.append({"latents": train_lat, "name": "orig"})
                            else:
                                aug_lat = encode_video(vae, pv["pixel_frames"], normalize=True)
                                t_start = cond_lat.shape[2]
                                t_end = t_start + train_lat.shape[2]
                                train_latents_variants.append({
                                    "latents": aug_lat[:, :, t_start:t_end],
                                    "name": pv["name"],
                                })
                        vae.to("cpu")
                        torch.cuda.empty_cache()
                        timing["aug_encode"] = time.time() - _t
                        del _pf

                    _t = time.time()
                    opt_result = optimize_delta_a(
                        wrapper=wrapper,
                        cond_latents=cond_lat,
                        train_latents=train_lat,
                        prompt_embeds=pe,
                        prompt_mask=pm,
                        num_steps=args.delta_steps,
                        lr=args.delta_lr,
                        device=args.device,
                        dtype=torch.bfloat16,
                        early_stopper=early_stopper if val_lat is not None else None,
                        train_latents_variants=train_latents_variants,
                        grad_accum=args.tta_grad_accum,
                        anchor_reg_latents=val_lat,
                        anchor_reg_weight=args.anchor_reg_weight,
                        anchor_reg_sigmas=args.anchor_reg_sigmas_parsed,
                        anchor_reg_noise_draws=args.anchor_reg_noise_draws,
                        anchor_reg_video_id=bd["video_name"],
                        anchor_x0_weight=args.anchor_x0_weight,
                    )
                    timing["tta_train"] = time.time() - _t

                train_time = time.time() - _t_train_start
                timing["es_check"] = opt_result.get("es_check_time", 0.0)
                timing["tta_train_net"] = timing["tta_train"] - timing["es_check"]
                timing["train_total"] = train_time
                print(f"  Train time: {train_time:.1f}s, "
                      f"Delta norm: {opt_result['delta_norm']:.4f}")

                if args.save_delta_dir:
                    os.makedirs(args.save_delta_dir, exist_ok=True)
                    delta_out = os.path.join(args.save_delta_dir, f"{eval_name}.pt")
                    torch.save(wrapper.delta.detach().cpu(), delta_out)
                    print(f"  Saved delta checkpoint: {delta_out}")

                print(f"  Timing breakdown: "
                      f"load={timing['load_frames']:.1f}s, "
                      f"encode={timing['encode_latents']:.1f}s, "
                      f"prompt={timing['encode_prompt']:.1f}s, "
                      f"aug_build={timing['aug_build']:.1f}s, "
                      f"aug_encode={timing['aug_encode']:.1f}s, "
                      f"es_setup={timing['es_setup']:.1f}s, "
                      f"train={timing['tta_train_net']:.1f}s, "
                      f"es_check={timing['es_check']:.1f}s")

            # ── Generate ONLY for the eval video ──
            if not clip_gate_info.get("tta_skipped", False):
                vae.to(args.device)
                text_encoder.to(args.device)

            result = {
                "video_name": eval_name,
                "video_path": eval_entry["video_path"],
                "caption": eval_entry["caption"],
                "train_time": train_time,
                "es_check_time": opt_result.get("es_check_time", 0.0),
                "final_loss": opt_result["losses"][-1] if opt_result["losses"] else None,
                "final_base_loss": opt_result["base_losses"][-1] if opt_result.get("base_losses") else None,
                "final_anchor_reg_loss": opt_result["anchor_reg_losses"][-1] if opt_result.get("anchor_reg_losses") else None,
                "delta_norm": opt_result["delta_norm"],
                "raw_grad_norms": opt_result.get("raw_grad_norms", []),
                "batch_size": len(training_entries),
                "num_neighbors": len(training_entries) - 1,
                "early_stopping_info": opt_result.get("early_stopping_info"),
                "timing": timing,
                "success": True,
            }
            result.update(clip_gate_info)
            anchor_gate_info = compute_anchor_gate_info(
                opt_result.get("early_stopping_info"),
                mode=args.anchor_gate_mode,
                threshold=args.anchor_gate_threshold,
                soft_scale=args.anchor_gate_soft_scale,
            )
            result.update(anchor_gate_info)
            if anchor_gate_info["anchor_gate_enabled"]:
                print(
                    "  Anchor gate: "
                    f"mode={anchor_gate_info['anchor_gate_mode']}, "
                    f"decision={anchor_gate_info['anchor_gate_decision']}, "
                    f"scale={anchor_gate_info['anchor_gate_scale']:.3f}, "
                    f"rel_impr={anchor_gate_info['anchor_gate_relative_improvement']}, "
                    f"reason={anchor_gate_info['anchor_gate_reason']}"
                )

            gen_time = 0.0
            if not args.skip_generation:
                from PIL import Image

                num_gen = args.num_frames - args.num_cond_frames
                rollout_steps = args.rollout_steps

                gen_pf = gen_pixel_frames.to(args.device)
                pf = gen_pf.squeeze(0)
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_images = []
                for t_idx in range(pf.shape[1]):
                    frame_np = (pf[:, t_idx].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                    cond_images.append(Image.fromarray(frame_np))

                all_step_metrics = []
                prev_gen_frames = None

                anchor_scale = float(anchor_gate_info.get("anchor_gate_scale", 1.0) or 0.0)
                apply_delta = (
                    not clip_gate_info.get("tta_skipped", False)
                    and anchor_gate_info.get("anchor_gate_decision") != "skip"
                    and anchor_scale > 0.0
                )
                original_delta = None
                if apply_delta:
                    if abs(anchor_scale - 1.0) > 1e-6:
                        original_delta = wrapper.delta.data.clone()
                        wrapper.delta.data.mul_(anchor_scale)
                    wrapper.apply_to_dit()

                try:
                    for step_i in range(rollout_steps):
                        step_gen_start_frame = args.gen_start_frame + step_i * num_gen

                        if step_i > 0:
                            tail = prev_gen_frames[num_gen:]
                            cond_images = []
                            for t_idx in range(tail.shape[0]):
                                frame_np = (np.clip(tail[t_idx], 0, 1) * 255).astype(np.uint8)
                                cond_images.append(Image.fromarray(frame_np))

                        gen_start = time.time()
                        gen_frames = generate_video_continuation(
                            pipe=pipe,
                            video_frames=cond_images,
                            prompt=eval_entry["caption"],
                            num_cond_frames=args.num_cond_frames,
                            num_frames=args.num_frames,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            seed=args.seed + v_idx + step_i,
                            resolution=args.resolution,
                            device=args.device,
                        )
                        step_gen_time = time.time() - gen_start
                        gen_time += step_gen_time

                        step_metrics = evaluate_generation_metrics(
                            gen_output=gen_frames,
                            video_path=eval_entry["video_path"],
                            num_cond_frames=args.num_cond_frames,
                            num_gen_frames=num_gen,
                            gen_start_frame=step_gen_start_frame,
                            device=args.device,
                            return_gt_frames=(step_i == 0 and fvd_accumulator is not None),
                        )
                        _gt_for_fvd = step_metrics.pop("gt_frames_hwc", None)
                        all_step_metrics.append(step_metrics)

                        if step_i == 0 and fvd_accumulator is not None:
                            fvd_accumulator.update(gen_frames, eval_entry["video_path"],
                                                   args.num_cond_frames, num_gen, args.gen_start_frame,
                                                   gt_frames_hwc=_gt_for_fvd)

                        if step_i == 0:
                            output_path = os.path.join(videos_dir, f"{eval_name}_delta_a.mp4")
                            if not args.no_save_videos:
                                # Save raw pixels (no cond/gen borders) so disk
                                # FVD via eval_fvd matches online I3D features.
                                save_video_from_numpy(
                                    gen_frames, output_path, fps=24,
                                )
                                result["output_path"] = output_path

                        prev_gen_frames = gen_frames
                finally:
                    if apply_delta:
                        wrapper.remove_from_dit()
                    if original_delta is not None:
                        wrapper.delta.data.copy_(original_delta)

                result["gen_time"] = gen_time
                result["rollout_steps"] = rollout_steps

                for si, sm in enumerate(all_step_metrics):
                    for mk in ("psnr", "ssim", "lpips"):
                        result["step_%d_%s" % (si + 1, mk)] = sm.get(mk)

                avg_metrics = {}
                for mk in ("psnr", "ssim", "lpips"):
                    vals = [sm[mk] for sm in all_step_metrics if sm.get(mk) is not None and sm[mk] == sm[mk]]
                    avg_metrics[mk] = float(np.mean(vals)) if vals else float("nan")
                result.update(avg_metrics)

                print("    Metrics: PSNR=%.2f, SSIM=%.4f, LPIPS=%.4f" % (
                    avg_metrics["psnr"], avg_metrics["ssim"], avg_metrics["lpips"]))
                if rollout_steps > 1:
                    print("    Rollout: " + ", ".join(
                        "step%d PSNR=%.2f" % (si + 1, sm.get("psnr", float("nan")))
                        for si, sm in enumerate(all_step_metrics)))

                del gen_pf
                torch_gc()

            result["total_time"] = (
                float(clip_gate_info.get("clip_gate_eval_time", 0.0))
                + train_time
                + gen_time
            )
            all_results.append(result)

            save_checkpoint({"next_idx": v_idx + 1, "results": all_results}, ckpt_path)
            if fvd_accumulator is not None:
                fvd_accumulator.save_stats(fvd_ckpt_path)

            # Cleanup per-video
            wrapper = None
            batch_data = None
            gen_pixel_frames = None
            torch_gc()

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "video_name": eval_name,
                "video_path": eval_entry["video_path"],
                "error": str(e),
                "success": False,
            })
            save_checkpoint({"next_idx": v_idx + 1, "results": all_results}, ckpt_path)
            if fvd_accumulator is not None:
                fvd_accumulator.save_stats(fvd_ckpt_path)
            torch_gc()

    # Save final results
    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "delta_a",
        "delta_steps": args.delta_steps,
        "delta_lr": args.delta_lr,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "gen_start_frame": args.gen_start_frame,
        "rollout_steps": args.rollout_steps,
        "tta_grad_accum": args.tta_grad_accum,
        "anchor_gate_mode": args.anchor_gate_mode,
        "anchor_gate_threshold": args.anchor_gate_threshold,
        "anchor_gate_soft_scale": args.anchor_gate_soft_scale,
        "anchor_reg_weight": args.anchor_reg_weight,
        "anchor_reg_sigmas": args.anchor_reg_sigmas,
        "anchor_reg_noise_draws": args.anchor_reg_noise_draws,
        "batch_videos": args.batch_videos,
        "retrieval_pool_dir": args.retrieval_pool_dir,
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
        "avg_timing": {
            k: float(np.mean([r.get("timing", {}).get(k, 0.0) for r in successful]))
            for k in [
                "load_frames", "encode_latents", "encode_prompt",
                "aug_build", "aug_encode", "tta_train", "tta_train_net",
                "es_setup", "es_check", "train_total",
            ]
        } if successful else {},
        "aug_enabled": args.aug_enabled,
        "aug_flip": getattr(args, "aug_flip", False),
        "es_disable": getattr(args, "es_disable", False),
        "es_check_every": getattr(args, "es_check_every", 5),
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
        "anchor_gate_stats": {
            "enabled": args.anchor_gate_mode != "off",
            "mode": args.anchor_gate_mode,
            "num_use": sum(1 for r in successful if r.get("anchor_gate_decision") == "use"),
            "num_skip": sum(1 for r in successful if r.get("anchor_gate_decision") == "skip"),
            "avg_scale": (
                float(np.mean([r.get("anchor_gate_scale", 1.0) for r in successful]))
                if successful else 0.0
            ),
            "avg_relative_improvement": (
                float(np.mean([
                    r["anchor_gate_relative_improvement"]
                    for r in successful
                    if r.get("anchor_gate_relative_improvement") is not None
                ]))
                if any(r.get("anchor_gate_relative_improvement") is not None for r in successful)
                else None
            ),
        },
        "grad_norm_stats": _summarize_grad_norms(successful),
        "results": all_results,
    }
    aggregate_quality_metrics(summary)
    finalize_online_eval(fvd_accumulator, summary, videos_dir, args)
    save_results(summary, os.path.join(args.output_dir, "summary.json"))
    if not args.no_save_videos:
        rename_videos_with_metrics(summary, videos_dir)
    print(f"\nResults saved to {args.output_dir}/summary.json")
    if successful:
        print(f"Avg CLIP gate time: {summary['avg_clip_gate_eval_time']:.2f}s")
        print(f"Avg ES check time : {summary['avg_es_check_time']:.2f}s")
        print(f"Avg train time: {summary['avg_train_time']:.1f}s")
        print(f"Avg gen time: {summary['avg_gen_time']:.1f}s")
        print(f"Avg total time: {summary['avg_total_time']:.1f}s")
        gns = summary.get("grad_norm_stats", {})
        if gns:
            print(f"\nGradient norm stats (clip threshold=1.0):")
            print(f"  Clipped: {gns['clipped_steps']}/{gns['total_steps']} "
                  f"({gns['clip_rate']:.1%})")
            print(f"  Raw norms: min={gns['min']:.2f} median={gns['median']:.2f} "
                  f"mean={gns['mean']:.2f} max={gns['max']:.2f}")
        if summary.get("avg_timing"):
            at = summary["avg_timing"]
            print(f"\nDetailed avg timing per video:")
            print(f"  Load frames  : {at.get('load_frames', 0):.2f}s")
            print(f"  Encode latent: {at.get('encode_latents', 0):.2f}s")
            print(f"  Encode prompt: {at.get('encode_prompt', 0):.2f}s")
            print(f"  Aug build    : {at.get('aug_build', 0):.2f}s")
            print(f"  Aug encode   : {at.get('aug_encode', 0):.2f}s")
            print(f"  ES setup     : {at.get('es_setup', 0):.2f}s")
            print(f"  TTA train    : {at.get('tta_train_net', 0):.2f}s")
            print(f"  ES checking  : {at.get('es_check', 0):.2f}s")
            print(f"  Train total  : {at.get('train_total', 0):.2f}s")


if __name__ == "__main__":
    main()
