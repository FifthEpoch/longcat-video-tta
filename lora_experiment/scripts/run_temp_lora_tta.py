#!/usr/bin/env python3
"""
SlowFast-VGen (Temp-LoRA) test-time adaptation baseline for LongCat-Video.

Reference: Hu et al., "SlowFast-VGen: Slow-Fast Learning for Action-Driven
Long Video Generation", ICLR 2025. The "fast" pathway is a Temporal-LoRA
(Temp-LoRA) module updated *during* generation to store episodic memory of
the sequence, so later chunks stay consistent with earlier ones.

This is a COMPARISON BASELINE for AdaSteer (parameter-space TTA). The defining
difference from plain LoRA-TTA (`run_lora_tta.py`, which trains one LoRA once
on the whole context and then freezes it for generation) is that Temp-LoRA is
updated SEQUENTIALLY, chunk by chunk:

  1. Warm start (optional): stream the Temp-LoRA over the OBSERVED context
     window in temporal chunks, taking a few gradient steps per chunk so the
     adapter accumulates a recency-weighted "episodic memory" of this video's
     dynamics. Uses only observed pre-gen_start frames -> leakage-free.
  2. Generate the future. In a multi-chunk rollout, after each generated chunk
     the Temp-LoRA is fast-updated on that generated chunk (self-supervised;
     the model's own output, never future GT) before generating the next.

Leakage guarantees (matches AdaSteer / run_lora_tta):
  - `tta_total_frames` is clamped to `gen_start_frame`; the adaptation window
    is strictly pre-gen_start.
  - Ground-truth future frames are used for scoring ONLY, never for any
    Temp-LoRA update.

Short-horizon comparison (paper): 14 cond + 14 gen @ gen_start=48, single
chunk -> the warm-start streaming update is what distinguishes it from
LoRA-TTA. Long-horizon rollouts additionally exercise the per-chunk update.

Integration: wired into the sweep harness as METHOD=temp_lora
(see sweep_experiment/sbatch/run_sweep.sbatch), so it reuses the standard
summary.json / chunking / FVD-FID / VBench plumbing.
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

# common.py + early_stopping.py live under delta_experiment/scripts
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_DELTA_SCRIPTS = _REPO_ROOT / "delta_experiment" / "scripts"
sys.path.insert(0, str(_SCRIPT_DIR))          # for run_lora_tta import
sys.path.insert(0, str(_DELTA_SCRIPTS))
sys.path.insert(0, str(_REPO_ROOT))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    generate_video_continuation,
    save_results,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    load_ucf101_video_list,
    split_tta_latents,
    evaluate_generation_metrics,
    add_tta_frame_args,
    add_caption_guard_args,
    add_caption_override_args,
    add_tta_disable_caption_args,
    tta_caption_for,
    add_feature_frame_guard_args,
    validate_caption_quality,
    apply_fixed_caption,
    validate_tta_feature_budget,
    add_online_eval_args,
    OnlineFrechetAccumulator,
    finalize_online_eval,
    aggregate_quality_metrics,
)

# Reuse the (well-tested) LoRA machinery from the LoRA-TTA runner.
from run_lora_tta import (
    inject_builtin_lora_into_dit,
    inject_lora_into_dit,
    get_builtin_lora_parameters,
    get_lora_parameters,
    count_builtin_lora_parameters,
    count_lora_parameters,
    reset_builtin_lora_weights,
    reset_lora_weights,
    finetune_lora_on_conditioning,
    save_video_from_numpy,
    gpu_mem_stats,
    log_gpu_mem,
)

_VAE_T_SCALE = 4


def _num_latent_frames(num_pixel_frames: int) -> int:
    """VAE temporal compression: T pixel frames -> 1 + (T-1)//4 latent frames."""
    return 1 + (num_pixel_frames - 1) // _VAE_T_SCALE


def stream_temp_lora_updates(
    dit,
    lora_modules,
    all_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    ctx_latents: int,
    step_latents: int,
    steps_per_chunk: int,
    lr: float,
    warmup_steps: int,
    weight_decay: float,
    max_grad_norm: float,
    device: str,
    dtype: torch.dtype,
    lora_param_fn,
    anchor_x0_weight: float = 0.0,
) -> dict:
    """Sequentially fast-update the Temp-LoRA over a latent sequence.

    Slides a (cond -> target) window along the temporal axis of ``all_latents``
    WITHOUT resetting the adapter between windows, so the LoRA accumulates an
    episodic, recency-weighted memory of the sequence (the SlowFast-VGen "fast"
    pathway). ``all_latents`` must contain ONLY observed / already-generated
    frames (never future GT).

    cond window : latents[:, :, s : s + ctx_latents]
    target      : latents[:, :, s + ctx_latents : s + ctx_latents + step_latents]
    for s = 0, step_latents, 2*step_latents, ... while a full target fits.
    """
    L = all_latents.shape[2]
    losses: List[float] = []
    n_chunks = 0
    t0 = time.time()
    s = 0
    while s + ctx_latents + step_latents <= L:
        cond = all_latents[:, :, s:s + ctx_latents].contiguous()
        target = all_latents[:, :, s + ctx_latents:s + ctx_latents + step_latents].contiguous()
        res = finetune_lora_on_conditioning(
            dit=dit, lora_modules=lora_modules,
            cond_latents=cond, train_latents=target,
            prompt_embeds=prompt_embeds, prompt_mask=prompt_mask,
            num_steps=steps_per_chunk, lr=lr,
            warmup_steps=min(warmup_steps, steps_per_chunk),
            weight_decay=weight_decay, max_grad_norm=max_grad_norm,
            device=device, dtype=dtype,
            early_stopper=None,
            lora_param_fn=lora_param_fn,
            anchor_x0_weight=anchor_x0_weight,
        )
        losses.extend(res.get("losses", []))
        n_chunks += 1
        s += step_latents
        del cond, target
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "losses": losses,
        "num_chunks": n_chunks,
        "train_time": time.time() - t0,
        "final_loss": losses[-1] if losses else None,
    }


def main():
    parser = argparse.ArgumentParser(description="SlowFast-VGen (Temp-LoRA) TTA for LongCat-Video")

    # Data / output
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--start-video-idx", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--restart", action="store_true")

    # LoRA arguments (Temp-LoRA is a LoRA under the hood)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--target-ffn", action="store_true")
    parser.add_argument("--target-modules", type=str, default="qkv,proj")
    parser.add_argument("--lora-target-blocks", type=str, default="all")
    parser.add_argument("--use-builtin-lora", action="store_true")

    # Training arguments (per-chunk fast update)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-steps", type=int, default=10,
                        help="Gradient steps per Temp-LoRA fast-update chunk.")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=10.0)
    parser.add_argument("--anchor-x0-weight", type=float, default=0.0)

    # Temp-LoRA streaming controls
    parser.add_argument("--temp-lora-ctx-latents", type=int, default=0,
                        help="Context latent frames used as cond in each streaming "
                             "update (0 = derive from --tta-context-frames).")
    parser.add_argument("--temp-lora-step-latents", type=int, default=1,
                        help="Target latent frames advanced per streaming chunk.")
    parser.add_argument("--warm-start-on-context", action="store_true", default=True,
                        help="Stream Temp-LoRA over the observed context before generating "
                             "(default on; distinguishes Temp-LoRA from plain LoRA-TTA).")
    parser.add_argument("--no-warm-start-on-context", dest="warm_start_on_context",
                        action="store_false")
    parser.add_argument("--update-during-rollout", action="store_true", default=True,
                        help="Fast-update Temp-LoRA on each generated chunk during rollout "
                             "(default on; the SlowFast-VGen episodic mechanism).")
    parser.add_argument("--no-update-during-rollout", dest="update_during_rollout",
                        action="store_false")

    # Generation / eval
    parser.add_argument("--num-cond-frames", type=int, default=14)
    parser.add_argument("--num-frames", type=int, default=28)
    parser.add_argument("--gen-start-frame", type=int, default=48)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--no-save-videos", action="store_true")
    parser.add_argument("--save-only-list", type=str, default=None)
    parser.add_argument("--rollout-steps", type=int, default=1)

    add_tta_frame_args(parser)
    add_caption_guard_args(parser)
    add_caption_override_args(parser)
    add_tta_disable_caption_args(parser)
    add_feature_frame_guard_args(parser)
    add_online_eval_args(parser)

    args = parser.parse_args()

    # --- TTA window sizing / leakage guards (identical policy to run_lora_tta) ---
    if args.tta_total_frames is None:
        args.tta_total_frames = args.gen_start_frame
    if args.tta_context_frames is None or args.tta_context_frames > args.tta_total_frames:
        args.tta_context_frames = args.num_cond_frames
    if args.tta_total_frames > args.gen_start_frame:
        print(f"[WARN] tta_total_frames ({args.tta_total_frames}) exceeds "
              f"gen_start_frame ({args.gen_start_frame}); clamping to avoid GT leakage.")
        args.tta_total_frames = args.gen_start_frame
    if args.tta_context_frames > args.tta_total_frames:
        args.tta_context_frames = args.tta_total_frames
    validate_tta_feature_budget(args, context="temp_lora_tta")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = 0
    all_results = []
    if ckpt and not args.restart:
        start_idx = ckpt.get("next_idx", 0)
        all_results = ckpt.get("results", [])

    target_modules = [m.strip() for m in args.target_modules.split(",")]

    ctx_lat = args.temp_lora_ctx_latents or _num_latent_frames(args.tta_context_frames)
    step_lat = max(1, args.temp_lora_step_latents)

    print("=" * 70)
    print("SlowFast-VGen (Temp-LoRA) Test-Time Adaptation for LongCat-Video")
    print("=" * 70)
    print(f"Data dir            : {args.data_dir}")
    print(f"Output dir          : {args.output_dir}")
    print(f"LoRA rank/alpha     : {args.lora_rank}/{args.lora_alpha}")
    print(f"Per-chunk steps/lr  : {args.num_steps} / {args.learning_rate}")
    print(f"Stream ctx/step lat : {ctx_lat} / {step_lat}")
    print(f"Warm-start context  : {args.warm_start_on_context}")
    print(f"Update in rollout   : {args.update_during_rollout}")
    print(f"Geometry            : cond={args.num_cond_frames} total={args.num_frames} "
          f"gen_start={args.gen_start_frame} rollout={args.rollout_steps}")
    print(f"TTA window          : total={args.tta_total_frames} ctx={args.tta_context_frames}")
    print(f"Resume from idx     : {start_idx}")
    print("=" * 70)

    print("\nLoading LongCat-Video model components...")
    components = load_longcat_components(args.checkpoint_dir, device=args.device, dtype=torch.bfloat16)
    dit = components["dit"]
    vae = components["vae"]
    pipe = components["pipe"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    import functools
    from torch.utils.checkpoint import checkpoint as _ckpt_fn
    dit.gradient_checkpointing = True
    dit._gradient_checkpointing_func = functools.partial(_ckpt_fn, use_reentrant=False)

    for p in dit.parameters():
        p.requires_grad = False

    use_builtin = args.use_builtin_lora
    if use_builtin:
        lora_modules = inject_builtin_lora_into_dit(
            dit, rank=args.lora_rank, alpha=args.lora_alpha,
            target_modules=target_modules, target_ffn=args.target_ffn,
            target_blocks=args.lora_target_blocks,
        )
        param_counts = count_builtin_lora_parameters(lora_modules)
        _get_lora_params = lambda: get_builtin_lora_parameters(lora_modules)
        _reset_lora = lambda: reset_builtin_lora_weights(lora_modules)
    else:
        lora_modules = inject_lora_into_dit(
            dit, rank=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout,
            target_modules=target_modules, target_ffn=args.target_ffn,
            target_blocks=args.lora_target_blocks,
        )
        param_counts = count_lora_parameters(lora_modules)
        _get_lora_params = lambda: get_lora_parameters(lora_modules)
        _reset_lora = lambda: reset_lora_weights(lora_modules)

    print(f"Temp-LoRA modules: {len(lora_modules)}, trainable params: {param_counts['trainable']:,}")
    mem_after_load = log_gpu_mem("After model+LoRA load")

    exp_config = {
        "method": "temp_lora_tta",
        "reference": "SlowFast-VGen (ICLR 2025), Temp-LoRA fast pathway",
        "lora": {
            "rank": args.lora_rank, "alpha": args.lora_alpha,
            "target_modules": target_modules, "target_blocks": args.lora_target_blocks,
            "target_ffn": args.target_ffn, "num_modules": len(lora_modules),
        },
        "temp_lora": {
            "ctx_latents": ctx_lat, "step_latents": step_lat,
            "steps_per_chunk": args.num_steps, "learning_rate": args.learning_rate,
            "warm_start_on_context": args.warm_start_on_context,
            "update_during_rollout": args.update_during_rollout,
        },
        "generation": {
            "num_cond_frames": args.num_cond_frames, "num_frames": args.num_frames,
            "gen_start_frame": args.gen_start_frame, "rollout_steps": args.rollout_steps,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale, "resolution": args.resolution,
        },
        "seed": args.seed, "max_videos": args.max_videos,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    eval_videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed, validate_decodable=True
    )
    eval_videos = apply_fixed_caption(eval_videos, args.fixed_caption, context="eval")
    validate_caption_quality(
        eval_videos, mode=args.caption_guard_mode,
        min_nonempty_ratio=args.caption_guard_min_nonempty_ratio,
        min_unique_ratio=args.caption_guard_min_unique_ratio,
        max_top1_ratio=args.caption_guard_max_top1_ratio,
        max_generic_top1_ratio=args.caption_guard_max_generic_top1_ratio,
        top_k=args.caption_guard_topk, context="eval",
    )
    if args.start_video_idx > 0 or args.chunk_size > 0:
        end = len(eval_videos)
        if args.chunk_size > 0:
            end = min(args.start_video_idx + args.chunk_size, end)
        eval_videos = eval_videos[args.start_video_idx:end]
        print(f"Chunk: videos [{args.start_video_idx}:{end}] -> {len(eval_videos)} videos")

    print(f"\nEvaluation videos: {len(eval_videos)}")

    videos_dir = os.path.join(args.output_dir, "videos")
    fvd_accumulator = OnlineFrechetAccumulator(
        device=args.device, compute_fid=args.compute_fid,
        min_videos=args.min_fvd_videos,
        gt_cache_path=getattr(args, "gt_features_cache", None),
    ) if args.compute_fvd else None
    fvd_ckpt_path = os.path.join(args.output_dir, "fvd_checkpoint.npz")
    if fvd_accumulator is not None and start_idx > 0:
        fvd_accumulator.load_stats(fvd_ckpt_path)

    retain_set = set()
    if args.save_only_list:
        with open(args.save_only_list) as _f:
            retain_set = set(json.load(_f).get("all", []))
    if not args.no_save_videos or retain_set:
        os.makedirs(videos_dir, exist_ok=True)

    from PIL import Image

    for idx, eval_entry in enumerate(tqdm(eval_videos, desc="Temp-LoRA TTA")):
        if idx < start_idx:
            continue
        video_path = eval_entry["video_path"]
        caption = eval_entry["caption"]
        video_name = Path(video_path).stem

        try:
            # ── Load observed TTA window (strictly pre-gen_start) ──
            tta_start = args.gen_start_frame - args.tta_total_frames
            pixel_frames = load_video_frames(
                video_path, args.tta_total_frames, height=480, width=832,
                start_frame=max(0, tta_start),
            ).to(args.device, torch.bfloat16)
            all_latents = encode_video(vae, pixel_frames, normalize=True)

            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder, tta_caption_for(args, caption),
                device=args.device, dtype=torch.bfloat16,
            )

            # Fresh Temp-LoRA per video (episodic memory is per-sequence).
            _reset_lora()

            # ── Warm-start: stream fast-updates over OBSERVED context ──
            train_time = 0.0
            stream_info = {"num_chunks": 0, "final_loss": None}
            if args.warm_start_on_context:
                stream_info = stream_temp_lora_updates(
                    dit=dit, lora_modules=lora_modules,
                    all_latents=all_latents,
                    prompt_embeds=prompt_embeds, prompt_mask=prompt_mask,
                    ctx_latents=ctx_lat, step_latents=step_lat,
                    steps_per_chunk=args.num_steps, lr=args.learning_rate,
                    warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
                    max_grad_norm=args.max_grad_norm,
                    device=args.device, dtype=torch.bfloat16,
                    lora_param_fn=_get_lora_params,
                    anchor_x0_weight=args.anchor_x0_weight,
                )
                train_time += stream_info["train_time"]

            # Fallback: if the window was too short to form any streaming chunk,
            # adapt once on (context -> remaining observed frames) so the adapter
            # is not left at its zero-init identity.
            if stream_info["num_chunks"] == 0:
                num_ctx_lat = _num_latent_frames(args.tta_context_frames)
                cond_latents, train_latents, _ = split_tta_latents(
                    all_latents, num_ctx_lat, holdout_fraction=0.0,
                )
                if train_latents is not None and train_latents.shape[2] > 0:
                    res = finetune_lora_on_conditioning(
                        dit=dit, lora_modules=lora_modules,
                        cond_latents=cond_latents, train_latents=train_latents,
                        prompt_embeds=prompt_embeds, prompt_mask=prompt_mask,
                        num_steps=args.num_steps, lr=args.learning_rate,
                        warmup_steps=args.warmup_steps, weight_decay=args.weight_decay,
                        max_grad_norm=args.max_grad_norm,
                        device=args.device, dtype=torch.bfloat16,
                        lora_param_fn=_get_lora_params,
                        anchor_x0_weight=args.anchor_x0_weight,
                    )
                    train_time += res["train_time"]
                    stream_info["final_loss"] = res["losses"][-1] if res["losses"] else None
                del cond_latents, train_latents

            _cached_gen_cond = pixel_frames[:, :, -args.num_cond_frames:].clone()
            del all_latents, pixel_frames
            torch_gc()

            result = {
                "idx": idx, "video_name": video_name, "video_path": video_path,
                "caption": caption, "train_time": train_time,
                "temp_lora_chunks": stream_info["num_chunks"],
                "final_loss": stream_info.get("final_loss"),
                "success": True,
            }

            # ── Generation (with optional per-chunk Temp-LoRA updates) ──
            gen_time = 0.0
            if not args.skip_generation:
                num_gen = args.num_frames - args.num_cond_frames
                rollout_steps = args.rollout_steps

                pf = _cached_gen_cond.squeeze(0)
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_images = [
                    Image.fromarray(
                        (pf[:, t].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                    ) for t in range(pf.shape[1])
                ]

                all_step_metrics = []
                prev_gen_frames = None
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(args.device)

                for step_i in range(rollout_steps):
                    step_gen_start_frame = args.gen_start_frame + step_i * num_gen

                    if step_i > 0:
                        tail = prev_gen_frames[num_gen:]
                        cond_images = [
                            Image.fromarray((np.clip(tail[t], 0, 1) * 255).astype(np.uint8))
                            for t in range(tail.shape[0])
                        ]

                    t_gen0 = time.time()
                    gen_frames = generate_video_continuation(
                        pipe=pipe, video_frames=cond_images, prompt=caption,
                        num_cond_frames=args.num_cond_frames, num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed + idx + step_i,
                        resolution=args.resolution, device=args.device,
                    )
                    gen_time += time.time() - t_gen0

                    step_metrics = evaluate_generation_metrics(
                        gen_output=gen_frames, video_path=video_path,
                        num_cond_frames=args.num_cond_frames, num_gen_frames=num_gen,
                        gen_start_frame=step_gen_start_frame, device=args.device,
                        return_gt_frames=(step_i == 0 and fvd_accumulator is not None),
                    )
                    _gt_for_fvd = step_metrics.pop("gt_frames_hwc", None)
                    all_step_metrics.append(step_metrics)

                    if step_i == 0 and fvd_accumulator is not None:
                        fvd_accumulator.update(
                            gen_frames, video_path, args.num_cond_frames, num_gen,
                            args.gen_start_frame, gt_frames_hwc=_gt_for_fvd,
                        )
                    if step_i == 0:
                        should_save = (not args.no_save_videos) or (video_name in retain_set)
                        if should_save:
                            out_path = os.path.join(videos_dir, f"{video_name}_temp_lora.mp4")
                            save_video_from_numpy(gen_frames, out_path, fps=24)
                            result["output_path"] = out_path

                    # SlowFast-VGen fast update: learn the just-generated chunk
                    # (self-supervised episodic memory) before the next chunk.
                    # Uses the model's OWN output only -> leakage-free.
                    if args.update_during_rollout and step_i < rollout_steps - 1:
                        gen_gen = gen_frames[args.num_cond_frames:
                                             args.num_cond_frames + num_gen]  # [T,H,W,3] in [0,1]
                        gen_t = torch.from_numpy(
                            np.ascontiguousarray(gen_gen.transpose(3, 0, 1, 2))
                        ).unsqueeze(0).to(args.device, torch.bfloat16) * 2.0 - 1.0
                        with torch.no_grad():
                            gen_lat = encode_video(vae, gen_t, normalize=True)
                        num_ctx_lat = _num_latent_frames(args.num_cond_frames)
                        if gen_lat.shape[2] > num_ctx_lat:
                            c = gen_lat[:, :, :num_ctx_lat].contiguous()
                            tgt = gen_lat[:, :, num_ctx_lat:].contiguous()
                            t_u0 = time.time()
                            finetune_lora_on_conditioning(
                                dit=dit, lora_modules=lora_modules,
                                cond_latents=c, train_latents=tgt,
                                prompt_embeds=prompt_embeds, prompt_mask=prompt_mask,
                                num_steps=args.num_steps, lr=args.learning_rate,
                                warmup_steps=min(args.warmup_steps, args.num_steps),
                                weight_decay=args.weight_decay, max_grad_norm=args.max_grad_norm,
                                device=args.device, dtype=torch.bfloat16,
                                lora_param_fn=_get_lora_params,
                                anchor_x0_weight=args.anchor_x0_weight,
                            )
                            train_time += time.time() - t_u0
                            del c, tgt
                        del gen_t, gen_lat
                        torch_gc()

                    prev_gen_frames = gen_frames

                result["gen_time"] = gen_time
                result["train_time"] = train_time
                result["rollout_steps"] = rollout_steps
                result["gpu_mem_gen"] = gpu_mem_stats(args.device) if torch.cuda.is_available() else {}

                for si, sm in enumerate(all_step_metrics):
                    for mk in ("psnr", "ssim", "lpips"):
                        result["step_%d_%s" % (si + 1, mk)] = sm.get(mk)
                avg_metrics = {}
                for mk in ("psnr", "ssim", "lpips"):
                    vals = [sm[mk] for sm in all_step_metrics
                            if sm.get(mk) is not None and sm[mk] == sm[mk]]
                    avg_metrics[mk] = float(np.mean(vals)) if vals else float("nan")
                result.update(avg_metrics)
                print("    Metrics: PSNR=%.2f, SSIM=%.4f, LPIPS=%.4f (chunks=%d)" % (
                    avg_metrics["psnr"], avg_metrics["ssim"], avg_metrics["lpips"],
                    stream_info["num_chunks"]))

            result["total_time"] = train_time + gen_time
            del _cached_gen_cond, prompt_embeds, prompt_mask
            torch_gc()
            all_results.append(result)

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "idx": idx, "video_name": video_name, "video_path": video_path,
                "error": str(e), "success": False,
            })

        save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)
        if fvd_accumulator is not None:
            fvd_accumulator.save_stats(fvd_ckpt_path)

    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "temp_lora_tta",
        "reference": "SlowFast-VGen (ICLR 2025)",
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "temp_lora_ctx_latents": ctx_lat,
        "temp_lora_step_latents": step_lat,
        "warm_start_on_context": args.warm_start_on_context,
        "update_during_rollout": args.update_during_rollout,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "gen_start_frame": args.gen_start_frame,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "num_failed": len(all_results) - len(successful),
        "avg_train_time": float(np.mean([r["train_time"] for r in successful])) if successful else 0,
        "avg_gen_time": float(np.mean([r.get("gen_time", 0.0) for r in successful])) if successful else 0,
        "avg_total_time": float(np.mean([r.get("total_time", 0.0) for r in successful])) if successful else 0,
        "results": all_results,
    }
    aggregate_quality_metrics(summary)
    finalize_online_eval(fvd_accumulator, summary, videos_dir, args)
    save_results(summary, os.path.join(args.output_dir, "summary.json"))

    print("\n" + "=" * 70)
    print("Temp-LoRA (SlowFast-VGen) TTA Complete: %d/%d ok" % (
        len(successful), len(all_results)))
    print("Results saved to: %s" % args.output_dir)
    print("=" * 70)


if __name__ == "__main__":
    main()
