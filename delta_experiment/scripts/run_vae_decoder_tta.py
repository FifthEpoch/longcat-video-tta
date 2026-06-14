#!/usr/bin/env python3
"""
VAE-Decoder-Only TTA (Modification 2 of
sweep_experiment/reports/LITERATURE_tta_recipe_modifications_2026-06-12.md).

Freeze the DiT entirely. Adapt only the LongCat-Video VAE *decoder* weights
per video, on the VAE round-trip reconstruction objective:

    pred_pixels = VAE.decode(VAE.encode(pixel_frames_train))
    L_total     = MSE(pred_pixels, pixel_frames_train)
                + lpips_weight * LPIPS(pred_pixels, pixel_frames_train)   [if > 0]

At inference, the DiT runs as usual and the per-video-tuned decoder is used
for the final latents-to-pixels step. The decoder is restored to its
snapshot at the end of each video so adapter state does not carry across
the stream.

Hypothesis (REVIEW §4.1; Mod 2 of the LITERATURE doc):
the binding constraint on per-video continuation TTA may not be the DiT
prediction but the VAE round-trip fidelity. If the VAE decoder is the
bottleneck (it cannot faithfully reconstruct certain motion patterns or
high-frequency texture for *this specific video*), per-video decoder
adaptation should improve trajectory PSNR/SSIM without touching the DiT.

Falsification (per LITERATURE §3, Mod 2):
held-out ΔPSNR on the four §2.3 beneficiary videos (panda_0461, panda_0555,
panda_0862, panda_0431) under VAE-decoder TTA does not exceed +1.0 dB on at
least 3 of 4 → bottleneck is somewhere else (DiT capacity, prompt
conditioning, or fundamentally the test-time supervisory signal).

Usage (env-var driven via run_sweep.sbatch case METHOD=vae_decoder):
    python run_vae_decoder_tta.py \
        --checkpoint-dir /path/to/longcat-video-checkpoints \
        --data-dir       /path/to/dataset \
        --output-dir     results/vae_decoder_tta \
        --vae-tta-steps  10 \
        --vae-tta-lr     1e-5 \
        --vae-tta-lpips-weight 0.0
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
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_video_frames,
    encode_video,
    save_results,
    save_video_from_numpy,
    rename_videos_with_metrics,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    add_tta_frame_args,
    add_caption_guard_args,
    add_caption_override_args,
    add_tta_disable_caption_args,
    add_feature_frame_guard_args,
    add_clip_gate_args,
    evaluate_clip_gate,
    summarize_clip_gate_stats,
    validate_caption_quality,
    apply_fixed_caption,
    validate_tta_feature_budget,
    add_online_eval_args,
    OnlineFrechetAccumulator,
    finalize_online_eval,
    aggregate_quality_metrics,
    load_ucf101_video_list,
)


# ============================================================================
# Two-phase load: VAE-only on GPU for decoder TTA; DiT lazy-loaded for inference
# ============================================================================

_DIT_GPU_MEM_WARN_GB = 25.0


def _gpu_mem_allocated_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return float(torch.cuda.memory_allocated() / (1024 ** 3))


def log_gpu_mem(tag: str) -> float:
    """Log PyTorch CUDA allocated memory; return GiB."""
    mem_gb = _gpu_mem_allocated_gb()
    print(f"  [mem] {tag}: {mem_gb:.2f} GiB PyTorch allocated")
    return mem_gb


def assert_vae_only_gpu_budget(tag: str, max_gb: float = _DIT_GPU_MEM_WARN_GB) -> None:
    """Fail fast before decoder TTA if DiT (or other bulk weights) stayed on GPU."""
    mem_gb = log_gpu_mem(tag)
    if mem_gb > max_gb:
        raise RuntimeError(
            f"GPU memory {mem_gb:.1f} GiB exceeds {max_gb:.1f} GiB budget at {tag} — "
            "DiT must not be resident during VAE-decoder TTA. "
            "Use two-phase load (VAE-only, lazy DiT)."
        )


def purge_generation_refs(components: Dict[str, object], tag: str = "") -> None:
    """Null pipeline / DiT handles so nothing generation-related stays on GPU."""
    pipe = components.get("pipe")
    dit = components.get("dit")
    if pipe is not None:
        if hasattr(pipe, "dit"):
            pipe.dit = None
        components["pipe"] = None
    if dit is not None:
        del dit
    components["dit"] = None
    components["pipe"] = None
    text_encoder = components.get("text_encoder")
    if text_encoder is not None:
        text_encoder.to("cpu")
        components["text_encoder"] = text_encoder
    torch_gc()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if tag:
        log_gpu_mem(tag)


def materialize_vae_on_gpu(vae, device: str, dtype: torch.dtype) -> None:
    """Force VAE weights onto GPU before budget checks (avoids lazy-load surprises)."""
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 5, 64, 64, device=device, dtype=dtype)
        try:
            vae.encode(dummy)
        except Exception as exc:
            print(f"  [WARN] VAE warmup encode failed ({exc}); continuing anyway.")
    torch_gc()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    log_gpu_mem("after VAE warmup encode")


def load_vae_only_components(
    checkpoint_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cp_split_hw: Optional[List[int]] = None,
) -> Dict[str, object]:
    """Load tokenizer, text_encoder (CPU), VAE (GPU), scheduler. DiT is deferred."""
    if cp_split_hw is None:
        cp_split_hw = [1, 1]

    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer", torch_dtype=dtype, use_fast=False,
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", torch_dtype=dtype,
    ).to("cpu")
    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_dir, subfolder="vae", torch_dtype=dtype,
    ).to(device)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=dtype,
    )

    materialize_vae_on_gpu(vae, device, dtype)

    torch_gc()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "scheduler": scheduler,
        "dit": None,
        "pipe": None,
        "checkpoint_dir": checkpoint_dir,
        "cp_split_hw": cp_split_hw,
        "dtype": dtype,
    }


def load_dit_and_build_pipe(
    components: Dict[str, object],
    device: str,
) -> Dict[str, object]:
    """Lazy-load DiT onto GPU and assemble the pipeline for generation."""
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

    checkpoint_dir = components["checkpoint_dir"]
    dtype = components["dtype"]
    cp_split_hw = components["cp_split_hw"]

    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir,
        subfolder="dit",
        cp_split_hw=cp_split_hw,
        enable_flashattn2=True,
        torch_dtype=dtype,
    ).to(device)
    dit.eval()
    for p in dit.parameters():
        p.requires_grad = False

    text_encoder = components["text_encoder"].to(device)
    pipe = LongCatVideoPipeline(
        tokenizer=components["tokenizer"],
        text_encoder=text_encoder,
        vae=components["vae"],
        scheduler=components["scheduler"],
        dit=dit,
    )

    torch_gc()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    components["dit"] = dit
    components["pipe"] = pipe
    components["text_encoder"] = text_encoder
    log_gpu_mem("after DiT lazy-load for inference")
    return components


def release_dit_after_inference(components: Dict[str, object]) -> None:
    """Delete DiT and move text_encoder off GPU; keep only VAE on GPU for TTA."""
    pipe = components.get("pipe")
    dit = components.get("dit")
    text_encoder = components.get("text_encoder")

    if pipe is not None:
        pipe.dit = None
    if dit is not None:
        del dit
    components["dit"] = None
    components["pipe"] = None

    if text_encoder is not None:
        text_encoder.to("cpu")
        components["text_encoder"] = text_encoder

    torch_gc()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    log_gpu_mem("after DiT release (VAE-only for next video)")


# ============================================================================
# Decoder snapshot helpers
# ============================================================================

def snapshot_decoder_state(vae) -> Dict[str, torch.Tensor]:
    """Snapshot the VAE decoder state_dict on CPU in the decoder's native dtype.

    Native-dtype + CPU keeps the snapshot safe from in-place GPU updates and
    makes per-video restoration byte-identical (no dtype round-trip).
    Restoring per-video uses ``load_state_dict``.
    """
    snap = {}
    for name, p in vae.decoder.state_dict().items():
        snap[name] = p.detach().to("cpu").clone()
    return snap


def restore_decoder_state(vae, snapshot: Dict[str, torch.Tensor]) -> None:
    """Restore the VAE decoder from a snapshot.

    Casts to the decoder's current device; preserves the snapshot's dtype
    (which is the decoder's native dtype by construction). This guarantees
    per-video reset is byte-identical to the post-load state.
    """
    target_device = next(vae.decoder.parameters()).device
    new_state = {
        name: t.to(device=target_device)
        for name, t in snapshot.items()
    }
    vae.decoder.load_state_dict(new_state)


def compute_decoder_drift(vae, snapshot: Dict[str, torch.Tensor]) -> float:
    """L2 norm of (current decoder weights − snapshot), summed over all params.

    Used for logging only — a sanity check that the decoder actually moved.
    """
    total_sq = 0.0
    with torch.no_grad():
        for name, p in vae.decoder.state_dict().items():
            ref = snapshot[name].to(p.device, torch.float32)
            d = p.to(torch.float32) - ref
            total_sq += float((d * d).sum().item())
    return float(total_sq ** 0.5)


# ============================================================================
# VAE-decoder TTA optimization
# ============================================================================

def optimize_vae_decoder(
    vae,
    pixel_frames: torch.Tensor,
    *,
    num_steps: int,
    lr: float,
    lpips_weight: float = 0.0,
    lpips_model: Optional[nn.Module] = None,
    device: str = "cuda",
    weight_decay: float = 0.0,
    grad_clip: Optional[float] = 1.0,
) -> Dict[str, object]:
    """Adapt vae.decoder to better reconstruct *pixel_frames* (a single clip).

    Encoder is held frozen; latents are computed once with no_grad and reused
    as the input to the decoder across all TTA steps. The reconstruction
    target is *pixel_frames* itself.

    Returns a dict with keys:
      - losses          : list[float], per-step total loss
      - pix_losses      : list[float], per-step pixel MSE component
      - lpips_losses    : list[float], per-step LPIPS component (or None)
      - grad_norms      : list[float], pre-clip total gradient L2 norm
      - num_steps       : int (= num_steps unless something aborts)
    """
    log_gpu_mem("optimize_vae_decoder: entry (before encode)")
    # 1. Encode the visible clip once (no grad on the encoder).
    with torch.no_grad():
        latents = encode_video(vae, pixel_frames, normalize=True)
    log_gpu_mem("optimize_vae_decoder: after encode_video")

    # 2. Set up decoder for training.
    vae.decoder.train()
    for p in vae.encoder.parameters():
        p.requires_grad = False
    for p in vae.decoder.parameters():
        p.requires_grad = True

    trainable = [p for p in vae.decoder.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=lr, weight_decay=weight_decay)

    losses: List[float] = []
    pix_losses: List[float] = []
    lpips_losses: List[float] = []
    grad_norms: List[float] = []

    from common import denormalize_latents

    # vae.decode expects latents in the *unnormalised* (raw VAE-output) space.
    # Denorm once; reuse the normalised latents tensor across Adam steps.
    lat_for_decode = denormalize_latents(vae, latents)
    T_lat = lat_for_decode.shape[2]
    pix_target = pixel_frames.to(torch.float32)

    for step in range(num_steps):
        optimizer.zero_grad(set_to_none=True)

        # Decode one latent temporal slice at a time (mirrors Wan VAE _decode) so
        # peak activation memory stays bounded during the decoder backward pass.
        pix_loss_sum = torch.zeros((), device=pix_target.device, dtype=torch.float32)
        lpips_loss_sum = torch.zeros((), device=pix_target.device, dtype=torch.float32)
        n_pix_elems = 0
        n_lpips_frames = 0
        pixel_offset = 0

        for t in range(T_lat):
            if step == 0 and t == 0:
                log_gpu_mem("optimize_vae_decoder: first decode slice (step=0,t=0)")
            lat_slice = lat_for_decode[:, :, t:t + 1].to(vae.dtype)
            dec_slice = vae.decode(lat_slice, return_dict=False)[0]
            t_pix = dec_slice.shape[2]
            pix_slice = pix_target[:, :, pixel_offset:pixel_offset + t_pix]
            n = min(pix_slice.shape[2], t_pix)
            if n <= 0:
                break
            pix_slice = pix_slice[:, :, :n]
            dec_slice = dec_slice[:, :, :n]

            diff = dec_slice.to(torch.float32) - pix_slice
            pix_loss_sum = pix_loss_sum + (diff * diff).sum()
            n_pix_elems += diff.numel()

            if lpips_weight > 0.0 and lpips_model is not None:
                B, C, T, H, W = dec_slice.shape
                d2 = dec_slice.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                p2 = pix_slice.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                lpips_loss_sum = lpips_loss_sum + lpips_model(
                    d2.clamp(-1, 1), p2.clamp(-1, 1)
                ).sum()
                n_lpips_frames += B * T

            pixel_offset += t_pix
            del dec_slice, lat_slice, diff, pix_slice

        pix_loss = pix_loss_sum / max(n_pix_elems, 1)
        if lpips_weight > 0.0 and lpips_model is not None and n_lpips_frames > 0:
            lpips_loss = lpips_loss_sum / n_lpips_frames
        else:
            lpips_loss = torch.zeros((), device=pix_loss.device, dtype=torch.float32)

        total = pix_loss + lpips_weight * lpips_loss
        total.backward()

        grad_sq = 0.0
        for p in trainable:
            if p.grad is None:
                continue
            grad_sq += float(p.grad.detach().to(torch.float32).pow(2).sum().item())
        grad_norm = grad_sq ** 0.5
        grad_norms.append(grad_norm)

        if grad_clip is not None and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip)

        optimizer.step()

        losses.append(float(total.detach().item()))
        pix_losses.append(float(pix_loss.detach().item()))
        lpips_losses.append(float(lpips_loss.detach().item()) if lpips_weight > 0.0 else None)

        torch_gc()

    vae.decoder.eval()
    for p in vae.decoder.parameters():
        p.requires_grad = False

    return {
        "losses": losses,
        "pix_losses": pix_losses,
        "lpips_losses": lpips_losses,
        "grad_norms": grad_norms,
        "num_steps": len(losses),
    }


# ============================================================================
# Main per-video loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="VAE-Decoder-Only TTA for LongCat-Video")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--start-video-idx", type=int, default=0,
                        help="Start processing from this index (for chunked runs)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Number of videos to process from start-video-idx (0 = all remaining)")

    # The recipe-specific knobs.
    parser.add_argument("--vae-tta-steps", type=int, default=10,
                        help="Number of TTA optimisation steps per video.")
    parser.add_argument("--vae-tta-lr", type=float, default=1e-5,
                        help="LR for the VAE decoder optimiser. Tiny by default — "
                             "the decoder is large and we adapt every parameter, "
                             "so the per-step movement is small to prevent decoder drift.")
    parser.add_argument("--vae-tta-lpips-weight", type=float, default=0.0,
                        help="Optional LPIPS auxiliary weight on the reconstruction loss. "
                             "Default 0.0 = pure pixel MSE.")
    parser.add_argument("--vae-tta-grad-clip", type=float, default=1.0,
                        help="Per-step grad-clip on the decoder parameters. "
                             "Default 1.0; set to 0 or negative to disable.")
    parser.add_argument("--vae-tta-weight-decay", type=float, default=0.0,
                        help="AdamW weight decay on the decoder parameters. Default 0.")

    # Standard frame / inference / eval / IO flags.
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
                        help="Skip video generation (only train decoder)")
    parser.add_argument("--no-save-videos", action="store_true",
                        help="Delete generated videos after evaluation to save disk space")

    add_tta_frame_args(parser)
    add_caption_guard_args(parser)
    add_caption_override_args(parser)
    add_tta_disable_caption_args(parser)
    add_feature_frame_guard_args(parser)
    add_online_eval_args(parser)
    add_clip_gate_args(parser)

    args = parser.parse_args()

    # Default tta_total_frames to gen_start_frame
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
    validate_tta_feature_budget(args, context="vae_decoder_tta")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resume support (phase-aware: tta → gen).
    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    run_phase = "tta"
    tta_start_idx = 0
    gen_start_idx = 0
    tta_records: Dict[str, dict] = {}
    all_results: List[dict] = []
    if ckpt:
        run_phase = ckpt.get("phase", "tta")
        tta_start_idx = ckpt.get("tta_next_idx", ckpt.get("next_idx", 0))
        gen_start_idx = ckpt.get("gen_next_idx", 0)
        for rec in ckpt.get("tta_records", []):
            tta_records[rec["video_name"]] = rec
        if ckpt.get("tta_complete"):
            run_phase = "gen"
        if ckpt.get("results"):
            all_results = list(ckpt["results"])

    log_gpu_mem("startup after imports, before model load")

    print("=" * 70)
    print("VAE-Decoder-Only TTA for LongCat-Video (Modification 2)")
    print("=" * 70)
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Data dir       : {args.data_dir}")
    print(f"Output dir     : {args.output_dir}")
    print(f"VAE TTA steps  : {args.vae_tta_steps}")
    print(f"VAE TTA LR     : {args.vae_tta_lr}")
    print(f"VAE LPIPS wt   : {args.vae_tta_lpips_weight}")
    print(f"Grad clip      : {args.vae_tta_grad_clip}")
    print(f"Weight decay   : {args.vae_tta_weight_decay}")
    print(f"TTA frames     : {args.tta_total_frames}  (ctx {args.tta_context_frames})")
    print(f"Resume phase   : {run_phase}  (tta_idx={tta_start_idx}, gen_idx={gen_start_idx})")
    print("=" * 70)

    # Load VAE-only (DiT deferred until generation phase — never during TTA).
    print("\nLoading VAE-only components (DiT deferred)...")
    components = load_vae_only_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16
    )
    purge_generation_refs(components, "after VAE-only load (purge gen refs)")
    vae = components["vae"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]
    log_gpu_mem("after VAE-only load")
    assert_vae_only_gpu_budget("post-load VAE-only budget check")

    # Freeze the VAE encoder once. We never train the encoder.
    for p in vae.encoder.parameters():
        p.requires_grad = False
    vae.encoder.eval()

    print("\nSnapshotting pristine VAE decoder weights (per-video restore baseline)...")
    decoder_snapshot = snapshot_decoder_state(vae)
    n_dec_params = sum(p.numel() for p in vae.decoder.parameters())
    print(f"  Decoder params : {n_dec_params/1e6:.1f}M")

    # Optional LPIPS model.
    lpips_model = None
    if args.vae_tta_lpips_weight > 0.0:
        try:
            import lpips as _lpips
            lpips_model = _lpips.LPIPS(net="alex").to(args.device).eval()
            for p in lpips_model.parameters():
                p.requires_grad = False
            print(f"  LPIPS model loaded (net=alex) for lpips_weight={args.vae_tta_lpips_weight}")
        except Exception as _e:
            print(f"  [WARN] failed to load LPIPS ({_e}); falling back to lpips_weight=0")
            args.vae_tta_lpips_weight = 0.0

    # Load eval videos.
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

    decoder_adapted_dir = os.path.join(args.output_dir, "decoder_adapted")
    gen_cond_dir = os.path.join(args.output_dir, "gen_cond")
    os.makedirs(decoder_adapted_dir, exist_ok=True)
    os.makedirs(gen_cond_dir, exist_ok=True)

    videos_dir = os.path.join(args.output_dir, "videos")
    fvd_accumulator = OnlineFrechetAccumulator(
        device=args.device, compute_fid=args.compute_fid,
        min_videos=args.min_fvd_videos,
        gt_cache_path=getattr(args, "gt_features_cache", None),
    ) if args.compute_fvd else None
    fvd_ckpt_path = os.path.join(args.output_dir, "fvd_checkpoint.npz")
    if fvd_accumulator is not None and gen_start_idx > 0:
        fvd_accumulator.load_stats(fvd_ckpt_path)
    if not args.skip_generation and not args.no_save_videos:
        os.makedirs(videos_dir, exist_ok=True)

    first_optimize_logged = False

    # ══════════════════════════════════════════════════════════════════════
    # Phase 1 — decoder TTA for every video (VAE-only on GPU; DiT absent)
    # ══════════════════════════════════════════════════════════════════════
    if run_phase == "tta":
        print("\n" + "=" * 70)
        print("PHASE 1: VAE-decoder TTA (DiT not loaded)")
        print("=" * 70)
        purge_generation_refs(components, "phase-1 start")

        for v_idx, eval_entry in enumerate(eval_videos):
            if v_idx < tta_start_idx:
                continue

            eval_name = Path(eval_entry["video_path"]).stem
            print(f"\n{'='*70}")
            print(f"[TTA {v_idx + 1}/{len(eval_videos)}] {eval_name}")

            if v_idx == tta_start_idx:
                log_gpu_mem("first video loop iteration (phase 1)")

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

            timing = {
                "load_frames": 0.0, "encode_latents": 0.0, "tta_train": 0.0,
                "decoder_restore": 0.0,
            }
            opt_result = {
                "losses": [], "pix_losses": [], "lpips_losses": [], "grad_norms": [],
                "num_steps": 0,
            }
            train_time = 0.0
            decoder_drift = 0.0
            adapted_decoder_path = os.path.join(decoder_adapted_dir, f"{eval_name}.pt")
            gen_cond_path = os.path.join(gen_cond_dir, f"{eval_name}.pt")

            try:
                purge_generation_refs(components, "pre-video TTA purge")

                tta_start = args.gen_start_frame - args.tta_total_frames
                _t = time.time()
                pixel_frames = load_video_frames(
                    eval_entry["video_path"], args.tta_total_frames,
                    height=480, width=832,
                    start_frame=max(0, tta_start),
                ).to(args.device, torch.bfloat16)
                timing["load_frames"] = time.time() - _t

                if args.tta_total_frames >= args.num_cond_frames:
                    gen_cond_frames_cpu = pixel_frames[:, :, -args.num_cond_frames:].clone().cpu()
                else:
                    gen_cond_start = args.gen_start_frame - args.num_cond_frames
                    gen_cond_frames_cpu = load_video_frames(
                        eval_entry["video_path"], args.num_cond_frames,
                        height=480, width=832, start_frame=max(0, gen_cond_start),
                    ).cpu()
                torch.save(gen_cond_frames_cpu, gen_cond_path)

                if not clip_gate_info.get("tta_skipped", False) and args.vae_tta_steps > 0:
                    assert_vae_only_gpu_budget("pre-decoder-TTA budget check")
                    if not first_optimize_logged:
                        log_gpu_mem("immediately before first optimize_vae_decoder call")
                        first_optimize_logged = True

                    _t_train_start = time.time()
                    opt_result = optimize_vae_decoder(
                        vae=vae,
                        pixel_frames=pixel_frames,
                        num_steps=args.vae_tta_steps,
                        lr=args.vae_tta_lr,
                        lpips_weight=args.vae_tta_lpips_weight,
                        lpips_model=lpips_model,
                        device=args.device,
                        weight_decay=args.vae_tta_weight_decay,
                        grad_clip=(args.vae_tta_grad_clip if args.vae_tta_grad_clip > 0 else None),
                    )
                    train_time = time.time() - _t_train_start
                    decoder_drift = compute_decoder_drift(vae, decoder_snapshot)
                    timing["tta_train"] = train_time

                    torch.save(snapshot_decoder_state(vae), adapted_decoder_path)

                    print(f"  Train time: {train_time:.1f}s   ({len(opt_result['losses'])} steps)")
                    if opt_result["pix_losses"]:
                        print(f"  Pix MSE:   first={opt_result['pix_losses'][0]:.5f}  "
                              f"last={opt_result['pix_losses'][-1]:.5f}")
                    if opt_result["losses"] and args.vae_tta_lpips_weight > 0:
                        print(f"  LPIPS:     first={opt_result['lpips_losses'][0]:.5f}  "
                              f"last={opt_result['lpips_losses'][-1]:.5f}")
                    if opt_result["grad_norms"]:
                        gn = opt_result["grad_norms"]
                        print(f"  Grad norm: first={gn[0]:.3f}  median={np.median(gn):.3f}  last={gn[-1]:.3f}")
                    print(f"  Decoder drift (||Δw||_2): {decoder_drift:.4f}")
                    log_gpu_mem("after decoder TTA optimize")
                elif clip_gate_info.get("tta_skipped", False):
                    print("  CLIP gate triggered: skip TTA (decoder remains pristine).")
                else:
                    torch.save(snapshot_decoder_state(vae), adapted_decoder_path)

                tta_rec = {
                    "video_name": eval_name,
                    "video_path": eval_entry["video_path"],
                    "caption": eval_entry["caption"],
                    "train_time": train_time,
                    "vae_tta_steps_actual": opt_result["num_steps"],
                    "vae_tta_losses": opt_result["losses"],
                    "vae_tta_pix_losses": opt_result["pix_losses"],
                    "vae_tta_lpips_losses": opt_result["lpips_losses"],
                    "vae_tta_grad_norms": opt_result["grad_norms"],
                    "decoder_drift": decoder_drift,
                    "timing": timing,
                    "adapted_decoder_path": adapted_decoder_path,
                    "gen_cond_path": gen_cond_path,
                }
                tta_rec.update(clip_gate_info)
                tta_records[eval_name] = tta_rec

                _t = time.time()
                restore_decoder_state(vae, decoder_snapshot)
                timing["decoder_restore"] = time.time() - _t
                tta_rec["timing"]["decoder_restore"] = timing["decoder_restore"]

                del pixel_frames, gen_cond_frames_cpu
                torch_gc()

                save_checkpoint({
                    "phase": "tta",
                    "tta_next_idx": v_idx + 1,
                    "gen_next_idx": 0,
                    "tta_complete": False,
                    "tta_records": list(tta_records.values()),
                }, ckpt_path)

            except Exception as e:
                import traceback
                print(f"  ERROR (phase 1): {e}")
                traceback.print_exc()
                try:
                    restore_decoder_state(vae, decoder_snapshot)
                except Exception:
                    pass
                tta_records[eval_name] = {
                    "video_name": eval_name,
                    "video_path": eval_entry["video_path"],
                    "error": str(e),
                    "tta_failed": True,
                }
                save_checkpoint({
                    "phase": "tta",
                    "tta_next_idx": v_idx + 1,
                    "gen_next_idx": 0,
                    "tta_complete": False,
                    "tta_records": list(tta_records.values()),
                }, ckpt_path)
                torch_gc()

        assert_vae_only_gpu_budget("end of phase-1 decoder TTA (all videos)")
        save_checkpoint({
            "phase": "gen",
            "tta_next_idx": len(eval_videos),
            "gen_next_idx": 0,
            "tta_complete": True,
            "tta_records": list(tta_records.values()),
        }, ckpt_path)
        run_phase = "gen"

    # ══════════════════════════════════════════════════════════════════════
    # Phase 2 — generation with lazy DiT load (once), after all decoder TTA
    # ══════════════════════════════════════════════════════════════════════
    if not args.skip_generation and run_phase == "gen":
        from PIL import Image
        from common import evaluate_generation_metrics, generate_video_continuation

        print("\n" + "=" * 70)
        print("PHASE 2: generation (lazy DiT load — after all decoder TTA)")
        print("=" * 70)

        if not tta_records and ckpt:
            for rec in ckpt.get("tta_records", []):
                tta_records[rec["video_name"]] = rec

        load_dit_and_build_pipe(components, args.device)
        pipe = components["pipe"]

        for v_idx, eval_entry in enumerate(eval_videos):
            if v_idx < gen_start_idx:
                continue

            eval_name = Path(eval_entry["video_path"]).stem
            tta_rec = tta_records.get(eval_name, {})
            if tta_rec.get("tta_failed"):
                all_results.append({
                    "video_name": eval_name,
                    "video_path": eval_entry["video_path"],
                    "error": tta_rec.get("error", "tta_failed"),
                    "success": False,
                })
                save_checkpoint({
                    "phase": "gen",
                    "tta_next_idx": len(eval_videos),
                    "gen_next_idx": v_idx + 1,
                    "tta_complete": True,
                    "tta_records": list(tta_records.values()),
                    "results": all_results,
                }, ckpt_path)
                continue

            print(f"\n{'='*70}")
            print(f"[GEN {v_idx + 1}/{len(eval_videos)}] {eval_name}")

            clip_gate_info = {k: tta_rec.get(k) for k in (
                "clip_gate_enabled", "clip_gate_decision", "clip_gate_reason",
                "tta_skipped", "clip_gate_eval_time",
            ) if k in tta_rec}
            train_time = float(tta_rec.get("train_time", 0.0))
            gen_time = 0.0

            try:
                adapted_path = tta_rec.get("adapted_decoder_path",
                                            os.path.join(decoder_adapted_dir, f"{eval_name}.pt"))
                gen_cond_path = tta_rec.get("gen_cond_path",
                                            os.path.join(gen_cond_dir, f"{eval_name}.pt"))
                if os.path.isfile(adapted_path):
                    restore_decoder_state(vae, torch.load(adapted_path, map_location="cpu"))
                gen_cond_frames_cpu = torch.load(gen_cond_path, map_location="cpu")

                gen_pf = gen_cond_frames_cpu.to(args.device)
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
                    prompt=eval_entry["caption"],
                    num_cond_frames=args.num_cond_frames,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed + v_idx,
                    resolution=args.resolution,
                    device=args.device,
                )
                gen_time = time.time() - gen_start

                step_metrics = evaluate_generation_metrics(
                    gen_output=gen_frames,
                    video_path=eval_entry["video_path"],
                    num_cond_frames=args.num_cond_frames,
                    num_gen_frames=args.num_frames - args.num_cond_frames,
                    gen_start_frame=args.gen_start_frame,
                    device=args.device,
                    return_gt_frames=(fvd_accumulator is not None),
                )
                _gt_for_fvd = step_metrics.pop("gt_frames_hwc", None)
                if fvd_accumulator is not None:
                    fvd_accumulator.update(
                        gen_frames, eval_entry["video_path"],
                        args.num_cond_frames,
                        args.num_frames - args.num_cond_frames,
                        args.gen_start_frame,
                        gt_frames_hwc=_gt_for_fvd,
                    )

                result = {
                    "video_name": eval_name,
                    "video_path": eval_entry["video_path"],
                    "caption": eval_entry["caption"],
                    "train_time": train_time,
                    "vae_tta_steps_actual": tta_rec.get("vae_tta_steps_actual", 0),
                    "vae_tta_losses": tta_rec.get("vae_tta_losses", []),
                    "vae_tta_pix_losses": tta_rec.get("vae_tta_pix_losses", []),
                    "vae_tta_lpips_losses": tta_rec.get("vae_tta_lpips_losses", []),
                    "vae_tta_grad_norms": tta_rec.get("vae_tta_grad_norms", []),
                    "decoder_drift": tta_rec.get("decoder_drift", 0.0),
                    "timing": tta_rec.get("timing", {}),
                    "gen_time": gen_time,
                    "total_time": (
                        float(tta_rec.get("clip_gate_eval_time", 0.0))
                        + train_time
                        + gen_time
                    ),
                    "success": True,
                }
                result.update(clip_gate_info)

                if not args.no_save_videos:
                    output_path = os.path.join(videos_dir, f"{eval_name}.mp4")
                    save_video_from_numpy(
                        gen_frames, output_path, fps=24,
                        num_cond_frames=args.num_cond_frames,
                    )
                    result["output_path"] = output_path

                for mk in ("psnr", "ssim", "lpips"):
                    result[mk] = step_metrics.get(mk)

                psnr = result["psnr"] if result["psnr"] is not None else float("nan")
                ssim = result["ssim"] if result["ssim"] is not None else float("nan")
                lpv = result["lpips"] if result["lpips"] is not None else float("nan")
                print(f"    Metrics: PSNR={psnr:.2f}, SSIM={ssim:.4f}, LPIPS={lpv:.4f}")

                del gen_pf, gen_cond_frames_cpu
                torch_gc()
                restore_decoder_state(vae, decoder_snapshot)

                all_results.append(result)
                save_checkpoint({
                    "phase": "gen",
                    "tta_next_idx": len(eval_videos),
                    "gen_next_idx": v_idx + 1,
                    "tta_complete": True,
                    "tta_records": list(tta_records.values()),
                    "results": all_results,
                }, ckpt_path)
                if fvd_accumulator is not None:
                    fvd_accumulator.save_stats(fvd_ckpt_path)

            except Exception as e:
                import traceback
                print(f"  ERROR (phase 2): {e}")
                traceback.print_exc()
                try:
                    restore_decoder_state(vae, decoder_snapshot)
                except Exception:
                    pass
                all_results.append({
                    "video_name": eval_name,
                    "video_path": eval_entry["video_path"],
                    "error": str(e),
                    "success": False,
                })
                save_checkpoint({
                    "phase": "gen",
                    "tta_next_idx": len(eval_videos),
                    "gen_next_idx": v_idx + 1,
                    "tta_complete": True,
                    "tta_records": list(tta_records.values()),
                    "results": all_results,
                }, ckpt_path)
                if fvd_accumulator is not None:
                    fvd_accumulator.save_stats(fvd_ckpt_path)
                torch_gc()

        release_dit_after_inference(components)

    elif args.skip_generation:
        # TTA-only run: promote tta_records to results without generation metrics.
        for eval_entry in eval_videos:
            eval_name = Path(eval_entry["video_path"]).stem
            tta_rec = tta_records.get(eval_name, {})
            if tta_rec.get("tta_failed"):
                all_results.append({
                    "video_name": eval_name,
                    "video_path": eval_entry["video_path"],
                    "error": tta_rec.get("error", "tta_failed"),
                    "success": False,
                })
            else:
                rec = dict(tta_rec)
                rec["success"] = True
                rec["gen_time"] = 0.0
                rec["total_time"] = float(tta_rec.get("clip_gate_eval_time", 0.0)) + float(
                    tta_rec.get("train_time", 0.0)
                )
                all_results.append(rec)

    # ── Final summary ──
    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "vae_decoder_tta",
        "vae_tta_steps": args.vae_tta_steps,
        "vae_tta_lr": args.vae_tta_lr,
        "vae_tta_lpips_weight": args.vae_tta_lpips_weight,
        "vae_tta_grad_clip": args.vae_tta_grad_clip,
        "vae_tta_weight_decay": args.vae_tta_weight_decay,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "gen_start_frame": args.gen_start_frame,
        "tta_total_frames": args.tta_total_frames,
        "tta_context_frames": args.tta_context_frames,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "avg_train_time": (
            float(np.mean([r.get("train_time", 0) for r in successful]))
            if successful else 0
        ),
        "avg_gen_time": (
            float(np.mean([r.get("gen_time", 0.0) for r in successful]))
            if successful else 0
        ),
        "avg_total_time": (
            float(np.mean([r.get("total_time", 0.0) for r in successful]))
            if successful else 0
        ),
        "avg_decoder_drift": (
            float(np.mean([r.get("decoder_drift", 0.0) for r in successful]))
            if successful else 0
        ),
        "avg_timing": {
            k: float(np.mean([r.get("timing", {}).get(k, 0.0) for r in successful]))
            for k in ["load_frames", "encode_latents", "tta_train", "decoder_restore"]
        } if successful else {},
        "clip_gate_enabled": args.clip_gate_enabled,
        "clip_gate_stats": summarize_clip_gate_stats(successful),
        "tta_disable_caption": getattr(args, "tta_disable_caption", False),
        "results": all_results,
    }
    aggregate_quality_metrics(summary)
    finalize_online_eval(fvd_accumulator, summary, videos_dir, args)
    save_results(summary, os.path.join(args.output_dir, "summary.json"))
    if not args.no_save_videos:
        rename_videos_with_metrics(summary, videos_dir)

    print(f"\nResults saved to {args.output_dir}/summary.json")
    if successful:
        print(f"Avg train time   : {summary['avg_train_time']:.1f}s")
        print(f"Avg gen time     : {summary['avg_gen_time']:.1f}s")
        print(f"Avg total time   : {summary['avg_total_time']:.1f}s")
        print(f"Avg decoder drift: {summary['avg_decoder_drift']:.4f}")


if __name__ == "__main__":
    main()
