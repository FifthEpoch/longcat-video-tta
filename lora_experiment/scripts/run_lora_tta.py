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
    parse_speed_factors,
)
from early_stopping import (
    AnchoredEarlyStopper,
    add_early_stopping_args,
    build_early_stopper_from_args,
)


# ============================================================================
# LoRA implementation for LongCat-Video DiT
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


def inject_lora_into_dit(
    dit: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: List[str] = ("qkv", "proj"),
    target_ffn: bool = False,
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

    Returns
    -------
    List of all LoRALinear modules created.
    """
    lora_modules = []
    device = next(dit.parameters()).device
    dtype = next(dit.parameters()).dtype

    for block_idx, block in enumerate(dit.blocks):
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
    lora_modules: List[LoRALinear],
    latents: torch.Tensor,
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
    latents_variants: Optional[List[Dict]] = None,
) -> Dict:
    """Fine-tune LoRA adapters on conditioning latents.

    Returns
    -------
    dict with keys: losses, train_time, early_stopping_info
    """
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

    # Build variant list (original only if no augmentation)
    if latents_variants is None:
        latents_variants = [{"latents": latents, "name": "orig"}]

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

        # Randomly pick a variant
        vi = torch.randint(0, len(latents_variants), (1,)).item()
        step_latents = latents_variants[vi]["latents"]

        loss = compute_flow_matching_loss(
            dit=dit,
            latents=step_latents,
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

        # Early stopping check
        if early_stopper is not None:
            should_stop, es_info = early_stopper.step(step + 1)
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    train_time = time.time() - train_start
    dit.eval()

    # Restore best if early stopping was used
    es_state = None
    if early_stopper is not None:
        early_stopper.restore(
            restore_fn=lambda sd: _restore_lora_from_state(dit, sd)
        )
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
    parser.add_argument("--save-lora-weights", action="store_true",
                        help="Save per-video LoRA weights")

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Video continuation arguments
    parser.add_argument("--num-cond-frames", type=int, default=13,
                        help="Number of conditioning frames from the input video")
    parser.add_argument("--num-frames", type=int, default=93,
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

    # Early stopping
    add_early_stopping_args(parser)
    add_augmentation_args(parser)

    args = parser.parse_args()

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
    print(f"LoRA rank      : {args.lora_rank}")
    print(f"LoRA alpha     : {args.lora_alpha}")
    print(f"Target modules : {target_modules}")
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

    # Freeze all DiT parameters
    for p in dit.parameters():
        p.requires_grad = False

    # Inject LoRA
    print(f"\nInjecting LoRA adapters (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    lora_modules = inject_lora_into_dit(
        dit,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_modules=target_modules,
        target_ffn=args.target_ffn,
    )

    param_counts = count_lora_parameters(lora_modules)
    total_dit_params = sum(p.numel() for p in dit.parameters())
    print(f"LoRA modules created : {len(lora_modules)}")
    print(f"LoRA trainable params: {param_counts['trainable']:,}")
    print(f"Total DiT params     : {total_dit_params:,}")
    print(f"Trainable %          : {100 * param_counts['trainable'] / total_dit_params:.4f}%")

    # Save experiment config
    exp_config = {
        "method": "lora_tta",
        "lora": {
            "rank": args.lora_rank,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": target_modules,
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
            # Load conditioning frames using anchor-based indexing
            cond_start = args.gen_start_frame - args.num_cond_frames
            pixel_frames = load_video_frames(
                video_path, args.num_cond_frames, height=480, width=832,
                start_frame=cond_start,
            ).to(args.device, torch.bfloat16)

            latents = encode_video(vae, pixel_frames, normalize=True)

            # Encode text
            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder, caption,
                device=args.device, dtype=torch.bfloat16,
            )

            # Build augmented latent variants if enabled
            latents_variants = None
            if args.aug_enabled:
                latents_variants = build_augmented_latent_variants(
                    pixel_frames=pixel_frames,
                    base_latents=latents,
                    vae=vae,
                    enable_flip=args.aug_flip,
                    rotate_deg=args.aug_rotate_deg,
                    rotate_random_min=args.aug_rotate_random_min,
                    rotate_random_max=args.aug_rotate_random_max,
                    rotate_random_count=args.aug_rotate_random_count,
                    rotate_random_step=args.aug_rotate_random_step,
                    rotate_zoom=args.aug_rotate_zoom,
                    speed_factors=parse_speed_factors(args.aug_speed_factors),
                )
                print(f"  Augmentation: {len(latents_variants)} variants "
                      f"({', '.join(v['name'] for v in latents_variants)})")

            # Reset LoRA weights before each video
            reset_lora_weights(lora_modules)

            # Setup early stopper for this video
            if early_stopper is not None:
                early_stopper.setup(
                    model=dit,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    device=args.device,
                    dtype=torch.bfloat16,
                )

            # Fine-tune LoRA
            train_result = finetune_lora_on_conditioning(
                dit=dit,
                lora_modules=lora_modules,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                num_steps=args.num_steps,
                lr=args.learning_rate,
                warmup_steps=args.warmup_steps,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                device=args.device,
                dtype=torch.bfloat16,
                early_stopper=early_stopper,
                latents_variants=latents_variants,
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

                # Convert pixel frames to list of PIL images for the pipeline
                pf = pixel_frames.squeeze(0)  # [C, T, H, W]
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_images = []
                for t_idx in range(pf.shape[1]):
                    frame_np = (pf[:, t_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
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

                # Save generated video
                output_path = os.path.join(videos_dir, f"{video_name}_lora.mp4")
                save_video_from_numpy(gen_frames, output_path, fps=24)
                result["output_path"] = output_path
                result["gen_time"] = gen_time

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
