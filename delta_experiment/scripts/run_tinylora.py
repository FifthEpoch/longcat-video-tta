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
import copy
import functools
import gc
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.checkpoint import checkpoint as _ckpt_fn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    compute_flow_matching_loss_conditioned,
    generate_video_continuation,
    split_tta_latents,
    save_results,
    save_video_from_numpy,
    rename_videos_with_metrics,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    load_ucf101_video_list,
    build_augmented_latent_variants,
    add_augmentation_args,
    add_tta_frame_args,
    add_caption_guard_args,
    add_caption_override_args,
    add_feature_frame_guard_args,
    add_clip_gate_args,
    add_online_eval_args,
    OnlineFrechetAccumulator,
    finalize_online_eval,
    aggregate_quality_metrics,
    parse_speed_factors,
    evaluate_clip_gate,
    summarize_clip_gate_stats,
    validate_caption_quality,
    apply_fixed_caption,
    validate_tta_feature_budget,
    evaluate_generation_metrics,
)
from early_stopping import (
    AnchoredEarlyStopper,
    add_early_stopping_args,
    build_early_stopper_from_args,
)
from tinylora_layers import (
    TinyLoRAConfig,
    TinyLoRAWrapper,
    TARGET_PRESETS,
    parse_target_blocks,
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
    train_latents_variants: Optional[List[Dict]] = None,
) -> Dict:
    """Optimise TinyLoRA v-parameters on conditioning latents."""
    trainable = wrapper.get_trainable_params()
    optimizer = AdamW(trainable, lr=lr, betas=(0.9, 0.999), eps=1e-15)

    if train_latents_variants is None:
        train_latents_variants = [{"latents": train_latents, "name": "orig"}]

    def _save_fn():
        return [p.detach().clone() for p in wrapper.get_trainable_params()]

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
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
        def _restore_fn(saved_params):
            for p, sp in zip(wrapper.get_trainable_params(), saved_params):
                p.data.copy_(sp)
        early_stopper.restore(restore_fn=_restore_fn)
        es_state = early_stopper.state

    v_norms = [p.detach().norm().item() for p in wrapper.get_trainable_params()]
    return {
        "losses": losses,
        "v_norms": v_norms,
        "mean_v_norm": float(np.mean(v_norms)),
        "es_check_time": es_check_time,
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
    g.add_argument("--start-video-idx", type=int, default=0,
                   help="Start processing from this index in the video list (for chunked runs)")
    g.add_argument("--chunk-size", type=int, default=0,
                   help="Number of videos to process from start-video-idx (0 = all remaining)")

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
    g.add_argument(
        "--target-blocks", type=str, default="all",
        help="Which DiT blocks to inject into: 'all', 'last_N', 'first_N', "
             "or comma-separated indices (e.g. 'last_5'). Adapting only "
             "late blocks cuts backward-pass time proportionally.",
    )

    g = parser.add_argument_group("TTA optimisation")
    g.add_argument("--tta-steps", type=int, default=20)
    g.add_argument("--tta-lr", type=float, default=1e-3)

    g = parser.add_argument_group("Video / generation settings")
    g.add_argument("--num-cond-frames", type=int, default=14)
    g.add_argument("--num-frames", type=int, default=28)
    g.add_argument("--gen-start-frame", type=int, default=48,
                    help="Fixed anchor frame where generation starts.")
    g.add_argument("--num-inference-steps", type=int, default=50)
    g.add_argument("--guidance-scale", type=float, default=4.0)
    g.add_argument("--resolution", type=str, default="480p")
    g.add_argument("--no-save-videos", action="store_true",
                    help="Delete generated videos after evaluation to save disk space")

    g = parser.add_argument_group("General")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=str, default="cuda")
    g.add_argument("--skip-generation", action="store_true")

    add_early_stopping_args(parser)
    add_augmentation_args(parser)
    add_tta_frame_args(parser)
    add_caption_guard_args(parser)
    add_caption_override_args(parser)
    add_feature_frame_guard_args(parser)
    add_online_eval_args(parser)
    add_clip_gate_args(parser)
    return parser


# ============================================================================
# Main
# ============================================================================


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.tta_total_frames is None:
        args.tta_total_frames = args.gen_start_frame
    if args.tta_context_frames is None or args.tta_context_frames > args.tta_total_frames:
        args.tta_context_frames = args.num_cond_frames
    if args.tta_total_frames > args.gen_start_frame:
        print(
            f"[WARN] tta_total_frames ({args.tta_total_frames}) exceeds "
            f"gen_start_frame ({args.gen_start_frame}); clamping."
        )
        args.tta_total_frames = args.gen_start_frame
    if args.tta_context_frames > args.tta_total_frames:
        args.tta_context_frames = args.tta_total_frames
    validate_tta_feature_budget(args, context="tinylora")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    if args.target_modules:
        target_modules = [t.strip() for t in args.target_modules.split(",")]
    else:
        target_modules = TARGET_PRESETS[args.target_preset]

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0

    print("=" * 70)
    print("TinyLoRA TTA for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint dir  : {args.checkpoint_dir}")
    print(f"Data dir        : {args.data_dir}")
    print(f"Output dir      : {args.output_dir}")
    print(f"SVD rank        : {args.svd_rank}")
    print(f"Alpha           : {args.alpha}")
    print(f"Weight tying    : n_tie={args.n_tie}")
    print(f"Target modules  : {target_modules}")
    print(f"Target blocks   : {args.target_blocks}")
    print(f"TTA steps       : {args.tta_steps}")
    print(f"TTA LR          : {args.tta_lr}")
    print(f"Augmentation    : {args.aug_enabled}")
    print(f"Gen start frame : {args.gen_start_frame}")
    print(f"Num cond frames : {args.num_cond_frames}")
    print(f"Num frames      : {args.num_frames}")
    print(f"Guidance scale  : {args.guidance_scale}")
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
    pipe = components["pipe"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    # Enable gradient checkpointing (critical for fitting in GPU memory)
    dit.gradient_checkpointing = True
    dit._gradient_checkpointing_func = functools.partial(
        _ckpt_fn, use_reentrant=False
    )
    print("Gradient checkpointing: ENABLED (use_reentrant=False)")

    # ------------------------------------------------------------------
    # Build TinyLoRA config (deferred until model is loaded so we
    # can resolve --target-blocks against the actual block count)
    # ------------------------------------------------------------------
    num_blocks = len(dit.blocks)
    resolved_blocks = parse_target_blocks(args.target_blocks, num_blocks)

    config = TinyLoRAConfig(
        svd_rank=args.svd_rank,
        alpha=args.alpha,
        n_tie=args.n_tie,
        target_modules=target_modules,
        target_blocks=resolved_blocks,
    )

    # ------------------------------------------------------------------
    # Inject TinyLoRA (once — SVDs computed here, reused across videos)
    # ------------------------------------------------------------------
    print("\nInjecting TinyLoRA adapters...")
    wrapper = TinyLoRAWrapper(dit, config)
    param_summary = wrapper.param_summary()
    print(f"  Adapted layers    : {param_summary['num_adapted_layers']}")
    print(f"  Unique v vectors  : {param_summary['num_v_vectors']}")
    print(f"  Trainable params  : {param_summary['tinylora_trainable']}")
    print(
        f"  Total model params: {param_summary['total_model_params']:,} "
        f"({param_summary['tinylora_trainable'] / param_summary['total_model_params'] * 100:.6f}%)"
    )

    # ------------------------------------------------------------------
    # Load video list
    # ------------------------------------------------------------------
    videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed,
        validate_decodable=True,
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
    if args.start_video_idx > 0 or args.chunk_size > 0:
        end = len(videos)
        if args.chunk_size > 0:
            end = min(args.start_video_idx + args.chunk_size, end)
        videos = videos[args.start_video_idx:end]
        print(f"Chunk: videos [{args.start_video_idx}:{end}] → {len(videos)} videos")

    print(f"\nTotal videos: {len(videos)}")

    early_stopper = build_early_stopper_from_args(args)
    all_results = []
    videos_dir = os.path.join(args.output_dir, "videos")
    fvd_accumulator = OnlineFrechetAccumulator(
        device=args.device, compute_fid=args.compute_fid,
        min_videos=args.min_fvd_videos,
        gt_cache_path=getattr(args, "gt_features_cache", None),
    ) if args.compute_fvd else None
    if not args.skip_generation and not args.no_save_videos:
        os.makedirs(videos_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Per-video TTA loop
    # ------------------------------------------------------------------
    for idx, entry in enumerate(videos):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n{'='*70}")
        print(f"[{idx + 1}/{len(videos)}] {video_name}")

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
                f"decision={clip_gate_info['clip_gate_decision']}"
            )

        try:
            train_time = 0.0
            timing = {k: 0.0 for k in [
                "load_frames", "encode_latents", "encode_prompt",
                "aug_build", "aug_encode", "tta_train", "es_setup",
                "es_check", "tta_train_net", "train_total",
            ]}

            if clip_gate_info.get("tta_skipped", False):
                gen_cond_start = args.gen_start_frame - args.num_cond_frames
                gen_pixel_frames = load_video_frames(
                    video_path, args.num_cond_frames,
                    height=480, width=832, start_frame=max(0, gen_cond_start),
                ).to(args.device, torch.bfloat16).cpu()
                opt_result = {
                    "losses": [], "mean_v_norm": 0.0,
                    "es_check_time": 0.0, "early_stopping_info": None,
                }
            else:
                # Reset v vectors to zero for this video
                wrapper.reset_v()

                tta_start = args.gen_start_frame - args.tta_total_frames
                _t = time.time()
                pixel_frames = load_video_frames(
                    video_path, args.tta_total_frames, height=480, width=832,
                    start_frame=max(0, tta_start),
                ).to(args.device, torch.bfloat16)
                timing["load_frames"] = time.time() - _t

                _t = time.time()
                all_latents = encode_video(vae, pixel_frames, normalize=True)
                timing["encode_latents"] = time.time() - _t

                vae_t_scale = 4
                num_ctx_lat = 1 + (args.tta_context_frames - 1) // vae_t_scale
                cond_latents, train_latents, val_latents = split_tta_latents(
                    all_latents, num_ctx_lat,
                    holdout_fraction=getattr(args, "es_holdout_fraction", 0.25),
                )

                _t = time.time()
                prompt_embeds, prompt_mask = encode_prompt(
                    tokenizer, text_encoder, caption,
                    device=args.device, dtype=torch.bfloat16,
                )
                timing["encode_prompt"] = time.time() - _t

                _cached_gen_cond_frames = None
                if args.tta_total_frames >= args.num_cond_frames:
                    _cached_gen_cond_frames = pixel_frames[:, :, -args.num_cond_frames:].clone()

                del all_latents, pixel_frames
                torch_gc()

                if _cached_gen_cond_frames is not None:
                    gen_pixel_frames = _cached_gen_cond_frames
                else:
                    gen_cond_start = args.gen_start_frame - args.num_cond_frames
                    gen_pixel_frames = load_video_frames(
                        video_path, args.num_cond_frames,
                        height=480, width=832, start_frame=max(0, gen_cond_start),
                    ).to(args.device, torch.bfloat16).cpu()

                # Offload VAE + text encoder to CPU during training
                vae.to("cpu")
                text_encoder.to("cpu")
                torch.cuda.empty_cache()

                # Early stopper setup
                if early_stopper is not None and val_latents is not None:
                    pe, pm = prompt_embeds, prompt_mask

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
                        cond_latents=cond_latents,
                        val_latents=val_latents,
                        prompt_embeds=prompt_embeds,
                        prompt_mask=prompt_mask,
                        device=args.device,
                        dtype=torch.bfloat16,
                        forward_fn=_es_forward_fn,
                        video_id=video_name,
                        save_fn=lambda: [
                            p.detach().clone()
                            for p in wrapper.get_trainable_params()
                        ],
                    )
                    timing["es_setup"] = time.time() - _t

                # Build augmented variants
                train_latents_variants = None
                if args.aug_enabled:
                    from common import build_augmented_pixel_variants
                    _tta_start = args.gen_start_frame - args.tta_total_frames
                    _t = time.time()
                    _pf = load_video_frames(
                        entry["video_path"], args.tta_total_frames,
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
                            train_latents_variants.append({"latents": train_latents, "name": "orig"})
                        else:
                            aug_lat = encode_video(vae, pv["pixel_frames"], normalize=True)
                            t_start = cond_latents.shape[2]
                            t_end = t_start + train_latents.shape[2]
                            train_latents_variants.append({
                                "latents": aug_lat[:, :, t_start:t_end],
                                "name": pv["name"],
                            })
                    vae.to("cpu")
                    torch.cuda.empty_cache()
                    timing["aug_encode"] = time.time() - _t
                    del _pf

                # Train
                _t_train_start = time.time()
                _t = time.time()
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
                    early_stopper=early_stopper if (val_latents is not None) else None,
                    train_latents_variants=train_latents_variants,
                )
                timing["tta_train"] = time.time() - _t
                train_time = time.time() - _t_train_start
                timing["es_check"] = opt_result.get("es_check_time", 0.0)
                timing["tta_train_net"] = timing["tta_train"] - timing["es_check"]
                timing["train_total"] = train_time

                print(f"  Train time: {train_time:.1f}s, "
                      f"Mean ||v||: {opt_result['mean_v_norm']:.6f}")

                # Bring back VAE + text encoder for generation
                vae.to(args.device)
                text_encoder.to(args.device)

            # -- Build result dict --
            result = {
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "train_time": train_time,
                "es_check_time": opt_result.get("es_check_time", 0.0),
                "final_loss": opt_result["losses"][-1] if opt_result["losses"] else None,
                "mean_v_norm": opt_result["mean_v_norm"],
                "early_stopping_info": opt_result.get("early_stopping_info"),
                "timing": timing,
                "success": True,
            }
            result.update(clip_gate_info)

            # -- Generate and evaluate --
            gen_time = 0.0
            if not args.skip_generation:
                from PIL import Image

                gen_pf = gen_pixel_frames.to(args.device)
                pf = gen_pf.squeeze(0)
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

                output_path = os.path.join(videos_dir, f"{video_name}_tinylora.mp4")
                if not args.no_save_videos:
                    save_video_from_numpy(
                        gen_frames, output_path, fps=24,
                        num_cond_frames=args.num_cond_frames,
                    )
                    result["output_path"] = output_path

                num_gen = args.num_frames - args.num_cond_frames
                metrics = evaluate_generation_metrics(
                    gen_output=gen_frames,
                    video_path=video_path,
                    num_cond_frames=args.num_cond_frames,
                    num_gen_frames=num_gen,
                    gen_start_frame=args.gen_start_frame,
                    device=args.device,
                    return_gt_frames=(fvd_accumulator is not None),
                )
                _gt_for_fvd = metrics.pop("gt_frames_hwc", None)
                result.update(metrics)
                if fvd_accumulator is not None:
                    fvd_accumulator.update(
                        gen_frames, video_path,
                        args.num_cond_frames, num_gen, args.gen_start_frame,
                        gt_frames_hwc=_gt_for_fvd,
                    )
                print(f"    Metrics: PSNR={metrics['psnr']:.2f}, "
                      f"SSIM={metrics['ssim']:.4f}, "
                      f"LPIPS={metrics['lpips']:.4f}")

                del gen_pf
                torch_gc()

            result["total_time"] = (
                float(clip_gate_info.get("clip_gate_eval_time", 0.0))
                + train_time
                + gen_time
            )
            all_results.append(result)
            save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)
            torch_gc()

        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "video_name": video_name,
                "video_path": video_path,
                "error": str(e),
                "success": False,
            })
            save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)
            torch_gc()

    # ------------------------------------------------------------------
    # Cleanup and save
    # ------------------------------------------------------------------
    wrapper.remove()

    successful = [r for r in all_results if r.get("success", False)]
    experiment_summary = {
        "method": "tinylora",
        "config": {
            "svd_rank": config.svd_rank,
            "alpha": config.alpha,
            "n_tie": config.n_tie,
            "target_modules": config.target_modules,
            "target_blocks": args.target_blocks,
        },
        "param_summary": param_summary,
        "tta_steps": args.tta_steps,
        "tta_lr": args.tta_lr,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "gen_start_frame": args.gen_start_frame,
        "guidance_scale": args.guidance_scale,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "avg_train_time": float(np.mean([r.get("train_time", 0) for r in successful])) if successful else 0,
        "avg_gen_time": float(np.mean([r.get("gen_time", 0) for r in successful])) if successful else 0,
        "avg_total_time": float(np.mean([r.get("total_time", 0) for r in successful])) if successful else 0,
        "avg_timing": {
            k: float(np.mean([r.get("timing", {}).get(k, 0.0) for r in successful]))
            for k in [
                "load_frames", "encode_latents", "encode_prompt",
                "aug_build", "aug_encode", "tta_train", "tta_train_net",
                "es_setup", "es_check", "train_total",
            ]
        } if successful else {},
        "aug_enabled": args.aug_enabled,
        "es_disable": getattr(args, "es_disable", False),
        "clip_gate_enabled": args.clip_gate_enabled,
        "clip_gate_stats": summarize_clip_gate_stats(successful),
        "results": all_results,
    }
    aggregate_quality_metrics(experiment_summary)
    finalize_online_eval(fvd_accumulator, experiment_summary, videos_dir, args)
    save_results(
        experiment_summary, os.path.join(args.output_dir, "summary.json")
    )
    if not args.no_save_videos:
        rename_videos_with_metrics(experiment_summary, videos_dir)
    print(f"\nResults saved to {args.output_dir}/summary.json")
    if successful:
        print(f"Avg train time : {experiment_summary['avg_train_time']:.1f}s")
        print(f"Avg gen time   : {experiment_summary['avg_gen_time']:.1f}s")
        print(f"Avg total time : {experiment_summary['avg_total_time']:.1f}s")


if __name__ == "__main__":
    main()
