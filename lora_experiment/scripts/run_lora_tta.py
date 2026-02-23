#!/usr/bin/env python3
"""
LoRA Test-Time Adaptation (TTA) for LongCat-Video.

Fine-tunes lightweight LoRA adapters on conditioning frames for each video,
then generates continuations using LongCat-Video's video continuation pipeline.

Key features:
- Injects LoRA adapters into the DiT's Linear layers (qkv, proj, ffn)
- Fine-tunes ONLY on conditioning frames (no ground truth in training)
- Uses LongCat-Video's native video continuation pipeline for generation
- Resets LoRA weights between videos
- Checkpoints progress for resumability
- Optional early stopping via anchor loss on held-out frames

Usage:
    python run_lora_tta.py \\
        --checkpoint-dir /path/to/longcat-video-checkpoints \\
        --data-dir /path/to/dataset \\
        --output-dir results/lora_tta_r8_lr2e-4 \\
        --lora-rank 8 --learning-rate 2e-4 --num-steps 20
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
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

# Ensure common.py and early_stopping.py are importable from delta_experiment
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_DELTA_SCRIPTS = _REPO_ROOT / "delta_experiment" / "scripts"
sys.path.insert(0, str(_DELTA_SCRIPTS))
sys.path.insert(0, str(_REPO_ROOT))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    compute_flow_matching_loss,
    compute_flow_matching_loss_conditioned,
    generate_video_continuation,
    save_results,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    load_ucf101_video_list,
    decode_latents,
    compute_psnr,
    compute_ssim_batch,
    compute_lpips_batch,
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
# Built-in LoRA support (uses LongCat-Video's native LoRAModule)
# ============================================================================

_LONGCAT_ROOT = _REPO_ROOT / "LongCat-Video"
sys.path.insert(0, str(_LONGCAT_ROOT))
from longcat_video.modules.lora_utils import LoRAModule


def inject_builtin_lora_into_dit(
    dit: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    target_modules: List[str] = ("qkv", "proj"),
    target_ffn: bool = False,
    target_blocks: str = "all",
) -> List[LoRAModule]:
    """Inject LoRA via LongCat's native LoRAModule + forward-hook mechanism.

    Instead of replacing nn.Linear modules (like our custom LoRALinear),
    this creates standalone LoRAModule instances and patches the original
    module's forward to add the LoRA contribution -- exactly matching the
    official LongCat-Video LoRA workflow.
    """
    device = next(dit.parameters()).device
    dtype = next(dit.parameters()).dtype

    block_indices = _parse_target_blocks(target_blocks, len(dit.blocks))
    label = (f"{sorted(block_indices)} ({len(block_indices)}/{len(dit.blocks)})"
             if block_indices is not None else f"all ({len(dit.blocks)})")
    print(f"  [builtin] LoRA target blocks: {label}")

    lora_modules: List[LoRAModule] = []

    def _maybe_add(module: nn.Module, name: str, n_sep: int = 1):
        if not isinstance(module, nn.Linear):
            return
        lora = LoRAModule(name, module, multiplier=1.0,
                          lora_dim=rank, alpha=alpha, n_seperate=n_sep)
        lora = lora.to(device=device, dtype=dtype)
        lora_modules.append(lora)

        if not hasattr(module, "org_forward"):
            module.org_forward = module.forward
        hooked_fwd = _build_hooked_forward(module, lora)
        module.forward = hooked_fwd

    for block_idx, block in enumerate(dit.blocks):
        if block_indices is not None and block_idx not in block_indices:
            continue

        if hasattr(block, "attn"):
            attn = block.attn
            if "qkv" in target_modules and hasattr(attn, "qkv"):
                _maybe_add(attn.qkv, f"blocks.{block_idx}.attn.qkv", n_sep=3)
            if "proj" in target_modules and hasattr(attn, "proj"):
                _maybe_add(attn.proj, f"blocks.{block_idx}.attn.proj")

        if hasattr(block, "cross_attn"):
            xattn = block.cross_attn
            if "qkv" in target_modules:
                if hasattr(xattn, "q_linear"):
                    _maybe_add(xattn.q_linear, f"blocks.{block_idx}.cross_attn.q_linear")
                if hasattr(xattn, "kv_linear"):
                    _maybe_add(xattn.kv_linear, f"blocks.{block_idx}.cross_attn.kv_linear", n_sep=2)
            if "proj" in target_modules and hasattr(xattn, "proj"):
                _maybe_add(xattn.proj, f"blocks.{block_idx}.cross_attn.proj")

        if target_ffn and hasattr(block, "ffn"):
            ffn = block.ffn
            for layer_name in ("w1", "w2", "w3"):
                if hasattr(ffn, layer_name):
                    _maybe_add(getattr(ffn, layer_name),
                               f"blocks.{block_idx}.ffn.{layer_name}")

    return lora_modules


def _build_hooked_forward(module: nn.Module, lora: LoRAModule):
    """Build a patched forward that adds the LoRA contribution."""
    def hooked_forward(x, *args, **kwargs):
        org_output = module.org_forward(x, *args, **kwargs)
        if lora.use_lora:
            lx = lora.lora_down(x.to(lora.lora_down.weight.dtype))
            lx = lora.lora_up(lx)
            org_output = org_output + lx.to(org_output.dtype) * lora.multiplier * lora.alpha_scale
        return org_output
    return hooked_forward


def get_builtin_lora_parameters(lora_modules: List[LoRAModule]) -> List[nn.Parameter]:
    """Collect trainable parameters from builtin LoRA modules."""
    params = []
    for lora in lora_modules:
        params.extend(lora.parameters())
    return [p for p in params if not isinstance(p, torch.Tensor) or p.requires_grad]


def count_builtin_lora_parameters(lora_modules: List[LoRAModule]) -> Dict[str, int]:
    """Count trainable parameters in builtin LoRA modules."""
    trainable = sum(p.numel() for lora in lora_modules
                    for p in lora.parameters() if p.requires_grad)
    total = sum(p.numel() for lora in lora_modules for p in lora.parameters())
    return {"total_lora": total, "trainable": trainable}


def reset_builtin_lora_weights(lora_modules: List[LoRAModule]):
    """Re-initialize builtin LoRA weights to zero-output state."""
    for lora in lora_modules:
        nn.init.kaiming_uniform_(lora.lora_down.weight, a=math.sqrt(5))
        if hasattr(lora.lora_up, "blocks"):
            for blk in lora.lora_up.blocks:
                nn.init.zeros_(blk.weight)
        else:
            nn.init.zeros_(lora.lora_up.weight)


def unhook_builtin_lora(dit: nn.Module):
    """Restore all forward methods patched by inject_builtin_lora_into_dit."""
    for _, module in dit.named_modules():
        if hasattr(module, "org_forward"):
            module.forward = module.org_forward
            delattr(module, "org_forward")


# ============================================================================
# Custom LoRA implementation (our original approach)
# ============================================================================

class LoRALinear(nn.Module):
    """Low-rank adapter that wraps an existing nn.Linear.

    The original weight stays frozen; only lora_down and lora_up are trained.
    Output = original_forward(x) + (x @ lora_down^T @ lora_up^T) * (alpha / rank)
    """

    def __init__(
        self,
        original: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Kaiming init for down, zero init for up (standard LoRA init)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x, *args, **kwargs):
        # Original computation (frozen)
        orig_out = self.original(x, *args, **kwargs)
        # LoRA delta
        lora_out = self.lora_up(self.lora_down(self.dropout(x.to(self.lora_down.weight.dtype))))
        return orig_out + lora_out * self.scaling


def _parse_target_blocks(target_blocks: str, num_blocks: int) -> Optional[set]:
    """Parse --lora-target-blocks into a set of block indices.

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


def inject_lora_into_dit(
    dit: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: List[str] = ("qkv", "proj"),
    target_ffn: bool = False,
    target_blocks: str = "all",
) -> List[LoRALinear]:
    """Inject LoRA adapters into the DiT's transformer blocks.

    Parameters
    ----------
    dit : LongCatVideoTransformer3DModel
    rank : LoRA rank
    alpha : LoRA alpha (scaling = alpha / rank)
    dropout : Dropout rate for LoRA
    target_modules : Which attention modules to target. Options: "qkv", "proj"
    target_ffn : If True, also target FFN (w1, w2, w3) layers
    target_blocks : Which blocks to inject LoRA into.
        "all" = every block, "last_N" = last N blocks, or comma-separated indices.

    Returns
    -------
    List of all LoRALinear modules created.
    """
    lora_modules = []
    device = next(dit.parameters()).device
    dtype = next(dit.parameters()).dtype

    block_indices = _parse_target_blocks(target_blocks, len(dit.blocks))
    if block_indices is not None:
        print(f"  LoRA target blocks: {sorted(block_indices)} "
              f"({len(block_indices)}/{len(dit.blocks)})")
    else:
        print(f"  LoRA target blocks: all ({len(dit.blocks)})")

    for block_idx, block in enumerate(dit.blocks):
        if block_indices is not None and block_idx not in block_indices:
            continue
        # Self-attention layers
        if hasattr(block, "attn"):
            attn = block.attn
            if "qkv" in target_modules and hasattr(attn, "qkv"):
                orig = attn.qkv
                lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                lora = lora.to(device=device, dtype=dtype)
                attn.qkv = lora
                lora_modules.append(lora)

            if "proj" in target_modules and hasattr(attn, "proj"):
                orig = attn.proj
                if isinstance(orig, nn.Linear):
                    lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                    lora = lora.to(device=device, dtype=dtype)
                    attn.proj = lora
                    lora_modules.append(lora)

        # Cross-attention layers
        if hasattr(block, "cross_attn"):
            xattn = block.cross_attn
            if "qkv" in target_modules:
                if hasattr(xattn, "q_linear") and isinstance(xattn.q_linear, nn.Linear):
                    orig = xattn.q_linear
                    lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                    lora = lora.to(device=device, dtype=dtype)
                    xattn.q_linear = lora
                    lora_modules.append(lora)

                if hasattr(xattn, "kv_linear") and isinstance(xattn.kv_linear, nn.Linear):
                    orig = xattn.kv_linear
                    lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                    lora = lora.to(device=device, dtype=dtype)
                    xattn.kv_linear = lora
                    lora_modules.append(lora)

            if "proj" in target_modules and hasattr(xattn, "proj"):
                orig = xattn.proj
                if isinstance(orig, nn.Linear):
                    lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                    lora = lora.to(device=device, dtype=dtype)
                    xattn.proj = lora
                    lora_modules.append(lora)

        # FFN layers
        if target_ffn and hasattr(block, "ffn"):
            ffn = block.ffn
            for layer_name in ("w1", "w2", "w3"):
                if hasattr(ffn, layer_name):
                    orig = getattr(ffn, layer_name)
                    if isinstance(orig, nn.Linear):
                        lora = LoRALinear(orig, rank=rank, alpha=alpha, dropout=dropout)
                        lora = lora.to(device=device, dtype=dtype)
                        setattr(ffn, layer_name, lora)
                        lora_modules.append(lora)

    return lora_modules


def get_lora_parameters(lora_modules: List[LoRALinear]) -> List[nn.Parameter]:
    """Collect all trainable LoRA parameters."""
    params = []
    for lora in lora_modules:
        params.extend(lora.lora_down.parameters())
        params.extend(lora.lora_up.parameters())
    return params


def count_lora_parameters(lora_modules: List[LoRALinear]) -> Dict[str, int]:
    """Count total / trainable LoRA parameters."""
    total = sum(p.numel() for lora in lora_modules for p in lora.parameters())
    trainable = sum(
        p.numel()
        for lora in lora_modules
        for p in [*lora.lora_down.parameters(), *lora.lora_up.parameters()]
    )
    return {"total_lora": total, "trainable": trainable}


def reset_lora_weights(lora_modules: List[LoRALinear]):
    """Re-initialize all LoRA weights to their initial state (zero output)."""
    for lora in lora_modules:
        nn.init.kaiming_uniform_(lora.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(lora.lora_up.weight)


def save_lora_weights(lora_modules: List[LoRALinear], path: str):
    """Save LoRA weights to a file."""
    state = {}
    for i, lora in enumerate(lora_modules):
        state[f"lora_{i}.down"] = lora.lora_down.weight.detach().cpu()
        state[f"lora_{i}.up"] = lora.lora_up.weight.detach().cpu()
    torch.save(state, path)


# ============================================================================
# LoRA TTA training loop
# ============================================================================

def finetune_lora_on_conditioning(
    dit: nn.Module,
    lora_modules,
    cond_latents: torch.Tensor,
    train_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_steps: int = 20,
    lr: float = 2e-4,
    warmup_steps: int = 3,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    early_stopper: Optional[AnchoredEarlyStopper] = None,
    lora_param_fn=None,
    train_latents_variants: Optional[List[Dict]] = None,
) -> Dict:
    """Fine-tune LoRA adapters using conditioning-aware loss.

    Parameters
    ----------
    cond_latents  : clean context latents [B, C, T_cond, H, W]
    train_latents : target latents to noise and compute loss on [B, C, T_train, H, W]
    train_latents_variants : optional augmented variants of train_latents

    Returns
    -------
    dict with keys: losses, train_time, early_stopping_info
    """
    if lora_param_fn is not None:
        lora_params = lora_param_fn()
    else:
        lora_params = get_lora_parameters(lora_modules)
    if not lora_params:
        raise ValueError("No LoRA parameters found.")

    optimizer = AdamW(
        lora_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
        eps=1e-8,
    )

    if train_latents_variants is None:
        train_latents_variants = [{"latents": train_latents, "name": "orig"}]

    def _save_fn():
        return [p.data.clone() for p in lora_params]

    def _restore_from_snapshot(snapshot):
        for p, saved in zip(lora_params, snapshot):
            p.data.copy_(saved)

    dit.train()
    losses = []
    train_start = time.time()

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        # LR warmup
        if step < warmup_steps and warmup_steps > 0:
            warmup_lr = lr * (step + 1) / warmup_steps
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        vi = torch.randint(0, len(train_latents_variants), (1,)).item()
        step_train = train_latents_variants[vi]["latents"]

        loss = compute_flow_matching_loss_conditioned(
            dit=dit,
            cond_latents=cond_latents,
            target_latents=step_train,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            device=device,
            dtype=dtype,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(lora_params, max_grad_norm)
        optimizer.step()

        losses.append(loss.item())
        del loss

        if step % 10 == 0:
            torch.cuda.empty_cache()

        if early_stopper is not None:
            should_stop, es_info = early_stopper.step(
                step + 1, save_fn=_save_fn,
            )
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    train_time = time.time() - train_start
    dit.eval()

    es_state = None
    if early_stopper is not None:
        early_stopper.restore(restore_fn=_restore_from_snapshot)
        es_state = early_stopper.state

    torch.cuda.empty_cache()

    return {
        "losses": losses,
        "train_time": train_time,
        "early_stopping_info": es_state,
    }


def _restore_lora_from_state(model: nn.Module, state_dict: dict):
    """Restore LoRA parameters from a snapshot state dict."""
    current = model.state_dict()
    for k, v in state_dict.items():
        if k in current:
            current[k].copy_(v)


# ============================================================================
# Evaluation helpers
# ============================================================================

def save_video_from_numpy(frames: np.ndarray, output_path: str, fps: int = 24):
    """Save video from numpy array [N, H, W, 3] in [0, 1]."""
    import imageio

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames_u8 = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    imageio.mimwrite(output_path, frames_u8, fps=fps, codec="libx264", quality=9)


# ============================================================================
# Main experiment
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="LoRA TTA for LongCat-Video")

    # Data / output
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Path to LongCat-Video checkpoint directory")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to video dataset directory")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--restart", action="store_true",
                        help="Restart from beginning (ignore checkpoint)")

    # LoRA arguments
    parser.add_argument("--lora-rank", type=int, default=8,
                        help="LoRA rank (default: 8)")
    parser.add_argument("--lora-alpha", type=float, default=16.0,
                        help="LoRA alpha for scaling (default: 16)")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="LoRA dropout rate")
    parser.add_argument("--target-ffn", action="store_true",
                        help="Also apply LoRA to FFN (w1, w2, w3) layers")
    parser.add_argument("--target-modules", type=str, default="qkv,proj",
                        help="Comma-separated attention modules to target (qkv, proj)")
    parser.add_argument("--lora-target-blocks", type=str, default="all",
                        help="Which DiT blocks to inject LoRA into. "
                             "'all' = every block, 'last_N' = last N blocks "
                             "(e.g. 'last_4', 'last_8'), or comma-separated "
                             "block indices (e.g. '44,45,46,47')")
    parser.add_argument("--use-builtin-lora", action="store_true",
                        help="Use LongCat-Video's native LoRAModule instead of "
                             "our custom LoRALinear wrapper. Enables n_separate "
                             "for fused QKV (3-way) and KV (2-way) projections.")
    parser.add_argument("--save-lora-weights", action="store_true",
                        help="Save per-video LoRA weights")

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Video continuation arguments
    parser.add_argument("--num-cond-frames", type=int, default=2,
                        help="Number of conditioning frames from the input video")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Total number of frames to generate (including cond)")
    parser.add_argument("--gen-start-frame", type=int, default=32,
                        help="Fixed anchor frame where generation starts. "
                             "Cond = video[anchor-cond : anchor]. "
                             "Ensures fair comparison across configs.")
    parser.add_argument("--num-inference-steps", type=int, default=50,
                        help="Number of diffusion denoising steps for generation")
    parser.add_argument("--guidance-scale", type=float, default=4.0,
                        help="Guidance scale for generation")
    parser.add_argument("--resolution", type=str, default="480p",
                        help="Target resolution for generation")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip video generation (only train)")
    parser.add_argument("--no-save-videos", action="store_true",
                        help="Skip saving generated videos to disk (metrics still computed in-memory)")

    # Early stopping
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
    start_idx = 0
    all_results = []
    if ckpt and not args.restart:
        start_idx = ckpt.get("next_idx", 0)
        all_results = ckpt.get("results", [])

    target_modules = [m.strip() for m in args.target_modules.split(",")]

    print("=" * 70)
    print("LoRA Test-Time Adaptation for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output dir     : {args.output_dir}")
    use_builtin = getattr(args, "use_builtin_lora", False)
    print(f"LoRA impl      : {'builtin (LongCat native)' if use_builtin else 'custom (LoRALinear)'}")
    print(f"LoRA rank      : {args.lora_rank}")
    print(f"LoRA alpha     : {args.lora_alpha}")
    print(f"Target modules : {target_modules}")
    print(f"Target blocks  : {args.lora_target_blocks}")
    print(f"Target FFN     : {args.target_ffn}")
    print(f"Learning rate  : {args.learning_rate}")
    print(f"Num steps      : {args.num_steps}")
    print(f"Resume from idx: {start_idx}")
    print("=" * 70)

    # Load model components
    print("\nLoading LongCat-Video model components...")
    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16
    )
    dit = components["dit"]
    vae = components["vae"]
    pipe = components["pipe"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    # Enable gradient checkpointing to reduce activation memory during backprop
    import functools
    from torch.utils.checkpoint import checkpoint as _ckpt_fn
    dit.gradient_checkpointing = True
    dit._gradient_checkpointing_func = functools.partial(_ckpt_fn, use_reentrant=False)
    print("Gradient checkpointing: ENABLED (use_reentrant=False)")

    # Freeze all DiT parameters
    for p in dit.parameters():
        p.requires_grad = False

    # Inject LoRA -- builtin or custom path
    lora_impl = "builtin" if use_builtin else "custom"
    print(f"\nInjecting LoRA adapters [{lora_impl}] (rank={args.lora_rank}, "
          f"alpha={args.lora_alpha}, blocks={args.lora_target_blocks})...")

    if use_builtin:
        lora_modules = inject_builtin_lora_into_dit(
            dit,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=target_modules,
            target_ffn=args.target_ffn,
            target_blocks=args.lora_target_blocks,
        )
        param_counts = count_builtin_lora_parameters(lora_modules)
        _get_lora_params = lambda: get_builtin_lora_parameters(lora_modules)
        _reset_lora = lambda: reset_builtin_lora_weights(lora_modules)
    else:
        lora_modules = inject_lora_into_dit(
            dit,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=target_modules,
            target_ffn=args.target_ffn,
            target_blocks=args.lora_target_blocks,
        )
        param_counts = count_lora_parameters(lora_modules)
        _get_lora_params = lambda: get_lora_parameters(lora_modules)
        _reset_lora = lambda: reset_lora_weights(lora_modules)

    total_dit_params = sum(p.numel() for p in dit.parameters())
    print(f"LoRA modules created : {len(lora_modules)}")
    print(f"LoRA trainable params: {param_counts['trainable']:,}")
    print(f"Total DiT params     : {total_dit_params:,}")
    print(f"Trainable %          : {100 * param_counts['trainable'] / total_dit_params:.4f}%")

    # Save experiment config
    exp_config = {
        "method": f"lora_tta_{lora_impl}",
        "lora": {
            "implementation": lora_impl,
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": getattr(args, "lora_dropout", 0.0),
            "target_modules": target_modules,
            "target_blocks": args.lora_target_blocks,
            "target_ffn": args.target_ffn,
            "num_modules": len(lora_modules),
            "trainable_params": param_counts["trainable"],
        },
        "training": {
            "learning_rate": args.learning_rate,
            "num_steps": args.num_steps,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
        },
        "generation": {
            "num_cond_frames": args.num_cond_frames,
            "num_frames": args.num_frames,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "resolution": args.resolution,
        },
        "seed": args.seed,
        "max_videos": args.max_videos,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    # Load video list
    videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed
    )
    print(f"\nTotal videos: {len(videos)}")

    # Build early stopper
    early_stopper = build_early_stopper_from_args(args)
    if early_stopper is not None:
        print(f"[EarlyStopper] Enabled – check_every={early_stopper.check_every}, "
              f"patience={early_stopper.patience}")
    else:
        print("[EarlyStopper] Disabled")

    # Process videos
    print(f"\nProcessing {len(videos) - start_idx} videos...\n")
    videos_dir = os.path.join(args.output_dir, "videos")
    lora_dir = os.path.join(args.output_dir, "lora_weights")
    if not args.no_save_videos:
        os.makedirs(videos_dir, exist_ok=True)
    if args.save_lora_weights:
        os.makedirs(lora_dir, exist_ok=True)

    for idx, entry in enumerate(tqdm(videos, desc="LoRA TTA")):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            # ── Frame loading ─────────────────────────────────────────
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

            # Encode text
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

            # Reset LoRA weights before each video
            _reset_lora()

            # Setup early stopper for this video
            if early_stopper is not None and val_latents is not None:
                def _es_forward_fn(hs, ts, ncl):
                    return dit(
                        hidden_states=hs, timestep=ts,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_mask,
                        num_cond_latents=ncl,
                    )

                def _save_fn():
                    return {k: v.clone() for k, v in dit.state_dict().items() if "lora" in k}

                early_stopper.setup(
                    model=dit,
                    cond_latents=cond_latents,
                    val_latents=val_latents,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    device=args.device,
                    dtype=torch.bfloat16,
                    forward_fn=_es_forward_fn,
                    video_id=video_name,
                    save_fn=_save_fn,
                )

            # Fine-tune LoRA
            train_result = finetune_lora_on_conditioning(
                dit=dit,
                lora_modules=lora_modules,
                cond_latents=cond_latents,
                train_latents=train_latents,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                num_steps=args.num_steps,
                lr=args.learning_rate,
                warmup_steps=args.warmup_steps,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                device=args.device,
                dtype=torch.bfloat16,
                early_stopper=early_stopper if val_latents is not None else None,
                train_latents_variants=train_latents_variants,
                lora_param_fn=_get_lora_params,
            )

            result = {
                "idx": idx,
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "train_time": train_result["train_time"],
                "final_loss": train_result["losses"][-1] if train_result["losses"] else None,
                "avg_loss": (
                    sum(train_result["losses"]) / len(train_result["losses"])
                    if train_result["losses"]
                    else None
                ),
                "num_train_steps": len(train_result["losses"]),
                "early_stopping_info": train_result.get("early_stopping_info"),
                "success": True,
            }

            # Generate video continuation
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

                output_path = os.path.join(videos_dir, f"{video_name}_lora.mp4")
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
                print(f"    Metrics: PSNR={metrics['psnr']:.2f}, "
                      f"SSIM={metrics['ssim']:.4f}, "
                      f"LPIPS={metrics['lpips']:.4f}")

            result["total_time"] = train_result["train_time"] + gen_time

            # Optionally save LoRA weights
            if args.save_lora_weights:
                lora_path = os.path.join(lora_dir, f"{video_name}_lora.pt")
                save_lora_weights(lora_modules, lora_path)

            print(f"  Train: {train_result['train_time']:.1f}s, "
                  f"Loss: {result['final_loss']:.4f}"
                  + (f", Gen: {gen_time:.1f}s" if not args.skip_generation else ""))

            all_results.append(result)

            # Cleanup
            del all_latents, cond_latents, train_latents, val_latents
            del pixel_frames, gen_pixel_frames, prompt_embeds, prompt_mask
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

        # Save checkpoint after each video
        save_checkpoint(
            {"next_idx": idx + 1, "results": all_results},
            ckpt_path,
        )

    # Save final results
    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "lora_tta",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "num_failed": len(all_results) - len(successful),
        "avg_train_time": (
            np.mean([r["train_time"] for r in successful])
            if successful
            else 0
        ),
        "avg_final_loss": (
            np.mean([r["final_loss"] for r in successful if r.get("final_loss") is not None])
            if successful
            else 0
        ),
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))

    print("\n" + "=" * 70)
    print("LoRA TTA Complete!")
    print("=" * 70)
    print(f"Successful: {len(successful)}/{len(all_results)}")
    if successful:
        print(f"Avg train time: {summary['avg_train_time']:.1f}s")
        print(f"Avg final loss: {summary['avg_final_loss']:.4f}")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
