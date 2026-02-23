#!/usr/bin/env python3
"""
Full-model Test-Time Adaptation (TTA) for LongCat-Video.

Fine-tunes ALL DiT parameters on conditioning frames for each video,
then generates continuations. Weights are reset between videos to measure
an upper-bound for single-video TTA.

Key features:
- Unfreezes all DiT parameters for maximum expressivity
- Saves base state for per-video reset
- Uses flow-matching loss on conditioning latents only
- Uses LongCat-Video's native video continuation pipeline
- Checkpoints progress for resumability
- Optional early stopping via anchor loss on held-out frames

Usage:
    python run_full_tta.py \\
        --checkpoint-dir /path/to/longcat-video-checkpoints \\
        --data-dir /path/to/dataset \\
        --output-dir results/full_tta_lr1e-5 \\
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
from torch.optim import AdamW, SGD
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
# Full-model TTA training loop
# ============================================================================

def finetune_full_on_conditioning(
    dit: nn.Module,
    cond_latents: torch.Tensor,
    train_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_steps: int = 10,
    lr: float = 1e-5,
    warmup_steps: int = 2,
    weight_decay: float = 0.01,
    max_grad_norm: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    early_stopper: Optional[AnchoredEarlyStopper] = None,
    train_latents_variants: Optional[List[Dict]] = None,
    optimizer_type: str = "sgd",
) -> Dict:
    """Fine-tune all DiT parameters using conditioning-aware loss.

    Parameters
    ----------
    cond_latents  : clean context latents [B, C, T_cond, H, W]
    train_latents : target latents to noise and compute loss on [B, C, T_train, H, W]
    train_latents_variants : optional augmented variants of train_latents
    optimizer_type : 'sgd' (default, no state — fits on single GPU) or 'adamw'

    Returns
    -------
    dict with keys: losses, train_time, early_stopping_info
    """
    params = [p for p in dit.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found. Did you unfreeze the model?")

    if optimizer_type == "adamw":
        optimizer = AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
            eps=1e-8,
        )
    else:
        optimizer = SGD(
            params,
            lr=lr,
            momentum=0.0,
            weight_decay=weight_decay,
        )

    if train_latents_variants is None:
        train_latents_variants = [{"latents": train_latents, "name": "orig"}]

    # Note: For full TTA, snapshotting entire model is expensive.
    # We rely on the early stopper's internal snapshotting if needed.

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
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        optimizer.step()

        losses.append(loss.item())
        del loss

        if step % 5 == 0:
            torch.cuda.empty_cache()

        if early_stopper is not None:
            should_stop, es_info = early_stopper.step(step + 1)
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    train_time = time.time() - train_start
    dit.eval()

    es_state = None
    if early_stopper is not None:
        def _restore_full(state_dict):
            for k, v in state_dict.items():
                parts = k.split(".")
                mod = dit
                for part in parts[:-1]:
                    mod = getattr(mod, part)
                getattr(mod, parts[-1]).data.copy_(v)

        early_stopper.restore(restore_fn=_restore_full)
        es_state = early_stopper.state

    torch.cuda.empty_cache()

    return {
        "losses": losses,
        "train_time": train_time,
        "early_stopping_info": es_state,
    }


def reset_dit_weights(dit: nn.Module, base_state: Dict[str, torch.Tensor]):
    """Reset DiT to its pretrained weights."""
    with torch.no_grad():
        for name, param in dit.named_parameters():
            if name in base_state:
                param.data.copy_(base_state[name].to(param.device))


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
    parser = argparse.ArgumentParser(description="Full-model TTA for LongCat-Video")

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

    # Training arguments
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate (lower than LoRA to avoid instability)")
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Number of fine-tuning steps per video")
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["sgd", "adamw"],
                        help="Optimizer for full-model TTA. SGD uses no state "
                             "(fits on single GPU); AdamW may OOM.")

    # Video continuation arguments
    parser.add_argument("--num-cond-frames", type=int, default=2)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--gen-start-frame", type=int, default=32,
                        help="Fixed anchor frame where generation starts. "
                             "Cond = video[anchor-cond : anchor]. "
                             "Ensures fair comparison across configs.")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--skip-generation", action="store_true")
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

    print("=" * 70)
    print("Full-Model Test-Time Adaptation for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output dir     : {args.output_dir}")
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

    # Enable gradient checkpointing to fit full backprop in GPU memory
    import functools
    from torch.utils.checkpoint import checkpoint as _ckpt_fn
    dit.gradient_checkpointing = True
    dit._gradient_checkpointing_func = functools.partial(_ckpt_fn, use_reentrant=False)
    print("Gradient checkpointing: ENABLED (use_reentrant=False)")

    # Unfreeze all DiT parameters
    for p in dit.parameters():
        p.requires_grad = True

    total_params = sum(p.numel() for p in dit.parameters())
    trainable_params = sum(p.numel() for p in dit.parameters() if p.requires_grad)
    print(f"Total DiT params    : {total_params:,}")
    print(f"Trainable params    : {trainable_params:,}")

    # Save base state for per-video reset (CPU to save GPU memory)
    print("Saving base state for per-video reset...")
    base_state = {k: v.detach().cpu().clone() for k, v in dit.state_dict().items()}
    print(f"Base state size: {sum(v.numel() * v.element_size() for v in base_state.values()) / 1e9:.2f} GB")

    # Save experiment config
    exp_config = {
        "method": "full_tta",
        "training": {
            "learning_rate": args.learning_rate,
            "num_steps": args.num_steps,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "total_params": total_params,
            "trainable_params": trainable_params,
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
    if not args.no_save_videos:
        os.makedirs(videos_dir, exist_ok=True)

    for idx, entry in enumerate(tqdm(videos, desc="Full TTA")):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            # Reset DiT to base weights before each video
            reset_dit_weights(dit, base_state)

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

            # Setup early stopper for this video
            if early_stopper is not None and val_latents is not None:
                def _es_forward_fn(hs, ts, ncl):
                    return dit(
                        hidden_states=hs, timestep=ts,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_mask,
                        num_cond_latents=ncl,
                    )

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
                    # Note: For full TTA, no save_fn — too expensive to snapshot full model
                )

            # Fine-tune
            train_result = finetune_full_on_conditioning(
                dit=dit,
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
                optimizer_type=args.optimizer,
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

                output_path = os.path.join(videos_dir, f"{video_name}_full.mp4")
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

            result["total_time"] = train_result["train_time"] + gen_time

            loss_str = f"Loss: {result['final_loss']:.4f}" if result.get('final_loss') is not None else "Loss: N/A (0 steps)"
            print(f"  Train: {train_result['train_time']:.1f}s, "
                  f"{loss_str}"
                  + (f", Gen: {gen_time:.1f}s" if not args.skip_generation else "")
                  + (f", PSNR={result.get('psnr', 'N/A')}" if 'psnr' in result else ""))

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
            torch_gc()

        # Save checkpoint after each video
        save_checkpoint(
            {"next_idx": idx + 1, "results": all_results},
            ckpt_path,
        )

    # Save final results
    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "full_tta",
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "total_params": total_params,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "num_failed": len(all_results) - len(successful),
        "avg_train_time": (
            np.mean([r["train_time"] for r in successful])
            if successful
            else 0
        ),
        "avg_final_loss": (
            float(np.mean(valid_losses)) if (valid_losses := [r["final_loss"] for r in successful if r.get("final_loss") is not None]) else None
        ),
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))

    print("\n" + "=" * 70)
    print("Full-Model TTA Complete!")
    print("=" * 70)
    print(f"Successful: {len(successful)}/{len(all_results)}")
    if successful:
        print(f"Avg train time: {summary['avg_train_time']:.1f}s")
        avg_loss = summary['avg_final_loss']
        if avg_loss is not None and not np.isnan(avg_loss):
            print(f"Avg final loss: {avg_loss:.4f}")
        else:
            print("Avg final loss: N/A (0 training steps)")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
