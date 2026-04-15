#!/usr/bin/env python3
"""
TinyLoRA TTA for LongCat-Video continuation.

Adapts the TinyLoRA method (Morris et al., 2026 — "Learning to Reason in
13 Parameters") for per-video test-time adaptation on video DiT models.

The approach injects ultra-low-rank SVD-based adapters into the DiT's
linear layers and optimises the (tiny) trainable vectors with the same
flow-matching loss used by the Delta-A/B/C experiments.

Key differences from standard LoRA:
    * SVD-based decomposition — only a vector v in R^r is trained per layer
    * Optional weight tying (--n-tie) shares v across groups of layers
    * Typical parameter counts: 2–200 (vs 100K+ for rank-8 LoRA)

Usage:
    python run_tinylora.py \\
        --checkpoint-dir /path/to/longcat-video-checkpoints \\
        --data-dir /path/to/dataset \\
        --output-dir results/tinylora \\
        --svd-rank 2 --n-tie 1 --alpha 1.0 \\
        --tta-steps 20 --tta-lr 1e-3
"""

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    compute_flow_matching_loss_conditioned,
    split_tta_latents,
    save_results,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    load_ucf101_video_list,
)
from early_stopping import (
    add_early_stopping_args,
    build_early_stopper_from_args,
)
from tinylora_layers import (
    TinyLoRAConfig,
    TinyLoRAWrapper,
    TARGET_PRESETS,
)


# ============================================================================
# Optimisation loop
# ============================================================================


def optimize_tinylora(
    wrapper: TinyLoRAWrapper,
    cond_latents: torch.Tensor,
    train_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_steps: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    early_stopper=None,
) -> Dict:
    """Optimise TinyLoRA v-parameters on conditioning latents."""
    trainable = wrapper.get_trainable_params()
    optimizer = AdamW(trainable, lr=lr, betas=(0.9, 0.999), eps=1e-15)

    def _save_fn():
        return {n: p.detach().clone() for n, p in wrapper.named_parameters() if p.requires_grad}

    wrapper.train()
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        loss = compute_flow_matching_loss_conditioned(
            dit=wrapper,
            cond_latents=cond_latents,
            target_latents=train_latents,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            device=device,
            dtype=dtype,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()

        losses.append(loss.item())

        if early_stopper is not None:
            should_stop, es_info = early_stopper.step(
                step + 1, save_fn=_save_fn,
            )
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    es_state = None
    if early_stopper is not None:
        early_stopper.restore(
            restore_fn=lambda sd: wrapper.load_state_dict(sd, strict=False)
        )
        es_state = early_stopper.state

    v_norms = [p.detach().norm().item() for p in wrapper.get_trainable_params()]
    return {
        "losses": losses,
        "v_norms": v_norms,
        "mean_v_norm": float(np.mean(v_norms)),
        "early_stopping_info": es_state,
    }


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TinyLoRA TTA for LongCat-Video"
    )

    g = parser.add_argument_group("Model / data")
    g.add_argument("--checkpoint-dir", type=str, required=True)
    g.add_argument("--data-dir", type=str, required=True)
    g.add_argument("--output-dir", type=str, required=True)
    g.add_argument("--max-videos", type=int, default=100)

    g = parser.add_argument_group("TinyLoRA config")
    g.add_argument(
        "--svd-rank", type=int, default=2,
        help="Frozen SVD rank (paper recommends 2)",
    )
    g.add_argument(
        "--alpha", type=float, default=1.0,
        help="LoRA scaling factor (scaling = alpha / svd_rank)",
    )
    g.add_argument(
        "--n-tie", type=int, default=1,
        help="Weight tying: number of modules sharing one v vector",
    )
    g.add_argument(
        "--target-preset", type=str, default="qkv_proj",
        choices=list(TARGET_PRESETS.keys()),
        help="Predefined set of target modules",
    )
    g.add_argument(
        "--target-modules", type=str, default=None,
        help="Comma-separated target module paths (overrides --target-preset)",
    )

    g = parser.add_argument_group("TTA optimisation")
    g.add_argument("--tta-steps", type=int, default=20)
    g.add_argument("--tta-lr", type=float, default=1e-3)

    g = parser.add_argument_group("Video settings")
    g.add_argument("--num-cond-frames", type=int, default=13)
    g.add_argument("--num-frames", type=int, default=93)
    g.add_argument("--resolution", type=str, default="480p")

    g = parser.add_argument_group("General")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=str, default="cuda")
    g.add_argument("--skip-generation", action="store_true")

    add_early_stopping_args(parser)
    return parser


# ============================================================================
# Main
# ============================================================================


def main():
    parser = build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve target modules
    if args.target_modules:
        target_modules = [t.strip() for t in args.target_modules.split(",")]
    else:
        target_modules = TARGET_PRESETS[args.target_preset]

    config = TinyLoRAConfig(
        svd_rank=args.svd_rank,
        alpha=args.alpha,
        n_tie=args.n_tie,
        target_modules=target_modules,
    )

    # Resume support
    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0

    print("=" * 70)
    print("TinyLoRA TTA for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint dir  : {args.checkpoint_dir}")
    print(f"Data dir        : {args.data_dir}")
    print(f"Output dir      : {args.output_dir}")
    print(f"SVD rank        : {config.svd_rank}")
    print(f"Alpha           : {config.alpha}")
    print(f"Weight tying    : n_tie={config.n_tie}")
    print(f"Target modules  : {config.target_modules}")
    print(f"TTA steps       : {args.tta_steps}")
    print(f"TTA LR          : {args.tta_lr}")
    print(f"Resume from idx : {start_idx}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load model components
    # ------------------------------------------------------------------
    print("\nLoading model components...")
    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16
    )
    dit = components["dit"]
    vae = components["vae"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    # ------------------------------------------------------------------
    # Inject TinyLoRA (once — SVDs computed here, reused across videos)
    # ------------------------------------------------------------------
    print("\nInjecting TinyLoRA adapters...")
    wrapper = TinyLoRAWrapper(dit, config)
    summary = wrapper.param_summary()
    print(f"  Adapted layers    : {summary['num_adapted_layers']}")
    print(f"  Unique v vectors  : {summary['num_v_vectors']}")
    print(f"  Trainable params  : {summary['tinylora_trainable']}")
    print(
        f"  Total model params: {summary['total_model_params']:,} "
        f"({summary['tinylora_trainable'] / summary['total_model_params'] * 100:.6f}%)"
    )

    # ------------------------------------------------------------------
    # Load video list
    # ------------------------------------------------------------------
    videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed
    )
    print(f"\nTotal videos: {len(videos)}")

    early_stopper = build_early_stopper_from_args(args)
    all_results = []

    # ------------------------------------------------------------------
    # Per-video TTA loop
    # ------------------------------------------------------------------
    for idx, entry in enumerate(videos):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            wrapper.reset_v()

            pixel_frames = load_video_frames(
                video_path, args.num_cond_frames, height=480, width=832
            ).to(args.device, torch.bfloat16)

            all_latents = encode_video(vae, pixel_frames, normalize=True)

            vae_t_scale = 4
            num_ctx_lat = 1 + (args.num_cond_frames - 1) // vae_t_scale
            cond_latents, train_latents, val_latents = split_tta_latents(
                all_latents, num_ctx_lat,
                holdout_fraction=getattr(args, "es_holdout_fraction", 0.25),
            )

            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder, caption,
                device=args.device, dtype=torch.bfloat16,
            )

            if early_stopper is not None and val_latents is not None:
                pe, pm = prompt_embeds, prompt_mask

                def _es_forward_fn(hs, ts, ncl):
                    return wrapper(
                        hidden_states=hs,
                        timestep=ts,
                        encoder_hidden_states=pe,
                        encoder_attention_mask=pm,
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
                    save_fn=lambda: {
                        n: p.detach().clone()
                        for n, p in wrapper.named_parameters() if p.requires_grad
                    },
                )

            t0 = time.time()
            opt_result = optimize_tinylora(
                wrapper=wrapper,
                cond_latents=cond_latents,
                train_latents=train_latents,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                num_steps=args.tta_steps,
                lr=args.tta_lr,
                device=args.device,
                dtype=torch.bfloat16,
                early_stopper=early_stopper if val_latents is not None else None,
            )
            train_time = time.time() - t0

            result = {
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "train_time": train_time,
                "final_loss": (
                    opt_result["losses"][-1] if opt_result["losses"] else None
                ),
                "mean_v_norm": opt_result["mean_v_norm"],
                "early_stopping_info": opt_result.get("early_stopping_info"),
            }

            print(
                f"  Train time: {train_time:.1f}s, "
                f"Final loss: {result['final_loss']:.4f}, "
                f"Mean ||v||: {result['mean_v_norm']:.6f}"
            )

            all_results.append(result)

            del all_latents, cond_latents, train_latents, val_latents
            del pixel_frames, prompt_embeds, prompt_mask
            torch_gc()

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({"video_name": video_name, "error": str(e)})

        save_checkpoint({"next_idx": idx + 1}, ckpt_path)

    # ------------------------------------------------------------------
    # Cleanup and save
    # ------------------------------------------------------------------
    wrapper.remove()

    experiment_summary = {
        "method": "tinylora",
        "config": {
            "svd_rank": config.svd_rank,
            "alpha": config.alpha,
            "n_tie": config.n_tie,
            "target_modules": config.target_modules,
        },
        "param_summary": summary,
        "tta_steps": args.tta_steps,
        "tta_lr": args.tta_lr,
        "num_videos": len(all_results),
        "avg_train_time": float(
            np.mean(
                [r.get("train_time", 0) for r in all_results if "train_time" in r]
            )
        ) if any("train_time" in r for r in all_results) else 0.0,
        "results": all_results,
    }
    save_results(
        experiment_summary, os.path.join(args.output_dir, "summary.json")
    )
    print(f"\nResults saved to {args.output_dir}/summary.json")
    print(f"Avg train time: {experiment_summary['avg_train_time']:.1f}s")


if __name__ == "__main__":
    main()
