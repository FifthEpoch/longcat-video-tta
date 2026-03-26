#!/usr/bin/env python3
"""
Shared utilities for LongCat-Video TTA experiments.

Handles model loading, video encoding, flow-matching loss computation,
dataset loading, and evaluation metrics.
"""

import gc
import ast
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup – add LongCat-Video to PYTHONPATH
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]  # sgs/
_LONGCAT_DIR = _REPO_ROOT / "LongCat-Video"
sys.path.insert(0, str(_LONGCAT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from transformers import AutoTokenizer, UMT5EncoderModel
from longcat_video.modules.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
from longcat_video.pipeline_longcat_video import LongCatVideoPipeline, retrieve_latents


# ============================================================================
# Model loading helpers
# ============================================================================

def load_longcat_components(
    checkpoint_dir: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    cp_split_hw: Optional[list] = None,
) -> Dict[str, object]:
    """Load all LongCat-Video components from a checkpoint directory.

    Returns a dict with keys: tokenizer, text_encoder, vae, scheduler, dit, pipe.
    """
    if cp_split_hw is None:
        cp_split_hw = [1, 1]

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_dir, subfolder="tokenizer", torch_dtype=dtype
    )
    text_encoder = UMT5EncoderModel.from_pretrained(
        checkpoint_dir, subfolder="text_encoder", torch_dtype=dtype
    )
    vae = AutoencoderKLWan.from_pretrained(
        checkpoint_dir, subfolder="vae", torch_dtype=dtype
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        checkpoint_dir, subfolder="scheduler", torch_dtype=dtype
    )
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw,
        enable_flashattn2=True, torch_dtype=dtype,
    )

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        scheduler=scheduler,
        dit=dit,
    )

    # Move to device
    text_encoder = text_encoder.to(device)
    vae = vae.to(device)
    dit = dit.to(device)

    return {
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "scheduler": scheduler,
        "dit": dit,
        "pipe": pipe,
    }


# ============================================================================
# Video / latent helpers
# ============================================================================

def load_video_frames(
    video_path: str,
    num_frames: int,
    height: int = 480,
    width: int = 832,
    start_frame: int = 0,
) -> torch.Tensor:
    """Load video frames as a tensor [1, C, T, H, W] in [-1, 1].

    Args:
        video_path: path to video file.
        num_frames: how many frames to return.
        height, width: target spatial resolution.
        start_frame: skip this many decoded frames before collecting.
                     Useful for the anchor-based scheme where conditioning
                     starts at ``anchor - num_cond`` instead of frame 0.
    """
    import av
    container = av.open(video_path)
    frames = []
    decoded = 0
    for frame in container.decode(video=0):
        if decoded < start_frame:
            decoded += 1
            continue
        if len(frames) >= num_frames:
            break
        img = frame.to_ndarray(format="rgb24")
        frames.append(img)
        decoded += 1
    container.close()

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from {video_path}")
    # Pad with last frame if not enough
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames_np = np.stack(frames[:num_frames], axis=0)  # (T, H, W, 3)
    frames_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()  # (3, T, H, W)
    frames_t = frames_t / 255.0

    # Resize
    frames_t = F.interpolate(
        frames_t.unsqueeze(0),
        size=(frames_t.shape[1], height, width),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)

    # Normalize to [-1, 1]
    frames_t = frames_t * 2.0 - 1.0
    return frames_t.unsqueeze(0)  # (1, 3, T, H, W)


def encode_video(
    vae: AutoencoderKLWan,
    pixel_frames: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Encode pixel frames [B, C, T, H, W] into VAE latents.

    If *normalize* is True, applies the VAE's latent normalization
    (mean/std shift) that LongCat-Video expects.
    """
    with torch.no_grad():
        posterior = vae.encode(pixel_frames)
        latents = retrieve_latents(posterior)

    if normalize:
        latents = normalize_latents(vae, latents)
    return latents


def normalize_latents(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    """Apply VAE mean/std normalization to latents."""
    mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    inv_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    return (latents - mean) * inv_std


def denormalize_latents(vae: AutoencoderKLWan, latents: torch.Tensor) -> torch.Tensor:
    """Undo VAE mean/std normalization."""
    mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    inv_std = (
        1.0
        / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    return latents / inv_std + mean


def decode_latents(
    vae: AutoencoderKLWan,
    latents: torch.Tensor,
    denorm: bool = True,
) -> torch.Tensor:
    """Decode latents to pixel frames [B, C, T, H, W] in [0, 1]."""
    if denorm:
        latents = denormalize_latents(vae, latents)
    with torch.no_grad():
        video = vae.decode(latents.to(vae.dtype), return_dict=False)[0]
    # video is in [-1, 1]
    video = (video + 1.0) / 2.0
    return video.clamp(0, 1)


# ============================================================================
# Text encoding helpers
# ============================================================================

def encode_prompt(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    prompt: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a text prompt. Returns (prompt_embeds, attention_mask)."""
    inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        embeds = text_encoder(input_ids, mask).last_hidden_state

    embeds = embeds.to(dtype=dtype, device=device)
    # Shape for LongCat-Video: [B, 1, N, C]
    embeds = embeds.unsqueeze(1)
    return embeds, mask


# ============================================================================
# Flow-matching loss
# ============================================================================

def _get_model_config(dit):
    """Get model config, handling both direct models and delta wrappers."""
    if hasattr(dit, "config"):
        return dit.config
    if hasattr(dit, "dit") and hasattr(dit.dit, "config"):
        return dit.dit.config
    raise AttributeError(
        f"Cannot find config on {type(dit).__name__}. "
        "Ensure the model or its .dit attribute inherits from ConfigMixin."
    )


def compute_flow_matching_loss(
    dit: LongCatVideoTransformer3DModel,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_train_timesteps: int = 1000,
    sigma_min: float = 0.001,
    sigma_max: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    forward_fn=None,
) -> torch.Tensor:
    """Compute flow-matching MSE loss on given latents.

    This is the standard rectified-flow loss:
        x_t = (1 - sigma) * x_0 + sigma * noise
        target = noise - x_0
        loss = MSE(model(x_t, t), target)

    Parameters
    ----------
    dit : the DiT model (or a delta wrapper with a .dit attribute)
    latents : clean latents [B, C, T, H, W]
    prompt_embeds : text embeddings [B, 1, N, C]
    prompt_mask : attention mask [B, N]
    forward_fn : optional custom forward function accepting (noisy_latents, timestep)
        If None, uses the standard dit forward pass.

    Returns
    -------
    loss : scalar MSE loss
    """
    cfg = _get_model_config(dit)
    B, C, T_lat, H_lat, W_lat = latents.shape

    # Sample random sigma uniformly in [sigma_min, sigma_max]
    sigma = torch.rand(B, device=device, dtype=torch.float32) * (sigma_max - sigma_min) + sigma_min
    # Expand sigma for broadcasting: [B, 1, 1, 1, 1]
    sigma_expanded = sigma.view(B, 1, 1, 1, 1)

    # Generate noise
    noise = torch.randn_like(latents)

    # Forward noising: x_t = (1 - sigma) * x_0 + sigma * noise
    noisy_latents = (1.0 - sigma_expanded) * latents + sigma_expanded * noise

    # Timestep: [B, T_lat] - same sigma for all frames in a sample
    patch_t = cfg.patch_size[0]
    N_t = T_lat // patch_t
    timestep = (sigma * num_train_timesteps).unsqueeze(1).expand(B, N_t).to(dtype)

    # Forward pass
    noisy_latents_input = noisy_latents.to(dtype)

    if forward_fn is not None:
        pred = forward_fn(noisy_latents_input, timestep)
    else:
        pred = dit(
            hidden_states=noisy_latents_input,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
        )

    # Target velocity: v = noise - x_0
    target = (noise - latents).to(torch.float32)
    pred = pred.to(torch.float32)

    loss = F.mse_loss(pred, target)
    return loss


def compute_flow_matching_loss_fixed(
    dit: LongCatVideoTransformer3DModel,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    fixed_sigmas: List[float],
    noise_draws: int = 1,
    num_train_timesteps: int = 1000,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    forward_fn=None,
) -> torch.Tensor:
    """Compute flow-matching loss at fixed timesteps with fixed noise seeds.

    Used by the AnchoredEarlyStopper for deterministic loss evaluation.

    Parameters
    ----------
    fixed_sigmas : list of sigma values to evaluate at
    noise_draws  : number of independent noise draws per sigma
    """
    cfg = _get_model_config(dit)
    B, C, T_lat, H_lat, W_lat = latents.shape
    total_loss = 0.0
    count = 0

    patch_t = cfg.patch_size[0]
    N_t = T_lat // patch_t

    for sigma_val in fixed_sigmas:
        sigma = torch.tensor([sigma_val], device=device, dtype=torch.float32)
        sigma_expanded = sigma.view(1, 1, 1, 1, 1)
        timestep = (sigma * num_train_timesteps).unsqueeze(1).expand(B, N_t).to(dtype)

        for draw_idx in range(noise_draws):
            # Use deterministic noise for reproducibility
            gen = torch.Generator(device=device)
            gen.manual_seed(42 + draw_idx)
            noise = torch.randn(
                latents.shape, generator=gen, device=device, dtype=latents.dtype
            )

            noisy_latents = (1.0 - sigma_expanded) * latents + sigma_expanded * noise
            noisy_latents_input = noisy_latents.to(dtype)

            with torch.no_grad():
                if forward_fn is not None:
                    pred = forward_fn(noisy_latents_input, timestep)
                else:
                    pred = dit(
                        hidden_states=noisy_latents_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_mask,
                    )

            target = (noise - latents).to(torch.float32)
            pred = pred.to(torch.float32)
            total_loss += F.mse_loss(pred, target).item()
            count += 1

    return total_loss / max(count, 1)


# ============================================================================
# Conditioning-aware flow-matching loss
# ============================================================================

def compute_flow_matching_loss_conditioned(
    dit,
    cond_latents: torch.Tensor,
    target_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_train_timesteps: int = 1000,
    sigma_min: float = 0.001,
    sigma_max: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    forward_fn=None,
) -> torch.Tensor:
    """Conditioning-aware flow-matching loss matching LongCat inference.

    During LongCat-Video inference, conditioning latent tokens receive
    ``timestep=0`` while noise tokens receive ``timestep=sigma*1000``, and
    ``num_cond_latents`` is passed to the DiT so that the attention module
    treats them differently.

    This function replicates that logic during TTA training:
        hidden_states = cat([cond_latents (clean), target_latents (noised)])
        timestep      = [0, ..., 0, sigma*1000, ..., sigma*1000]
        pred = dit(hidden_states, timestep, ..., num_cond_latents=T_cond)
        loss = MSE on target portion only

    Parameters
    ----------
    cond_latents   : clean conditioning latents [B, C, T_cond, H, W]
    target_latents : training target latents [B, C, T_target, H, W]
    forward_fn     : optional custom forward; signature
                     ``(hidden_states, timestep, num_cond_latents) -> pred``
    """
    cfg = _get_model_config(dit)
    B, C, T_cond, H_lat, W_lat = cond_latents.shape
    T_target = target_latents.shape[2]
    T_total = T_cond + T_target

    patch_t = cfg.patch_size[0]
    N_cond = T_cond // patch_t
    N_target = T_target // patch_t
    N_total = T_total // patch_t

    # Sample sigma
    sigma = torch.rand(B, device=device, dtype=torch.float32) * (sigma_max - sigma_min) + sigma_min
    sigma_expanded = sigma.view(B, 1, 1, 1, 1)

    # Noise only the target portion
    noise = torch.randn_like(target_latents)
    noisy_target = (1.0 - sigma_expanded) * target_latents + sigma_expanded * noise

    # Concatenate: [cond_clean, noisy_target]
    hidden_states = torch.cat([cond_latents, noisy_target], dim=2).to(dtype)

    # Build per-token timestep: cond=0, target=sigma*1000
    timestep = torch.zeros(B, N_total, device=device, dtype=dtype)
    timestep[:, N_cond:] = (sigma * num_train_timesteps).unsqueeze(1).expand(B, N_target).to(dtype)

    # Forward
    if forward_fn is not None:
        pred = forward_fn(hidden_states, timestep, N_cond)
    else:
        pred = dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            num_cond_latents=N_cond,
        )

    # Loss only on target portion
    pred_target = pred[:, :, T_cond:].to(torch.float32)
    velocity_target = (noise - target_latents).to(torch.float32)

    loss = F.mse_loss(pred_target, velocity_target)
    return loss


def compute_flow_matching_loss_conditioned_fixed(
    dit,
    cond_latents: torch.Tensor,
    target_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    fixed_sigmas: List[float],
    fixed_noises: List[torch.Tensor],
    num_train_timesteps: int = 1000,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    forward_fn=None,
) -> float:
    """Conditioning-aware flow-matching loss at fixed sigmas/noises.

    Used by the early stopper for deterministic anchor-val evaluation.
    ``fixed_noises`` is a list of pre-generated noise tensors (one per draw)
    with the same shape as ``target_latents``.

    Parameters
    ----------
    fixed_sigmas  : list of sigma values to evaluate at
    fixed_noises  : list of pre-drawn noise tensors [B, C, T_target, H, W]
    forward_fn    : optional custom forward; signature
                    ``(hidden_states, timestep, num_cond_latents) -> pred``
    """
    cfg = _get_model_config(dit)
    B, C, T_cond, H_lat, W_lat = cond_latents.shape
    T_target = target_latents.shape[2]
    T_total = T_cond + T_target

    patch_t = cfg.patch_size[0]
    N_cond = T_cond // patch_t
    N_target = T_target // patch_t
    N_total = T_total // patch_t

    total_loss = 0.0
    count = 0

    for sigma_val in fixed_sigmas:
        sigma = torch.tensor([sigma_val], device=device, dtype=torch.float32)
        sigma_expanded = sigma.view(1, 1, 1, 1, 1)

        for noise in fixed_noises:
            noisy_target = (1.0 - sigma_expanded) * target_latents + sigma_expanded * noise
            hidden_states = torch.cat([cond_latents, noisy_target], dim=2).to(dtype)

            timestep = torch.zeros(B, N_total, device=device, dtype=dtype)
            timestep[:, N_cond:] = (sigma * num_train_timesteps).unsqueeze(1).expand(B, N_target).to(dtype)

            with torch.no_grad():
                if forward_fn is not None:
                    pred = forward_fn(hidden_states, timestep, N_cond)
                else:
                    pred = dit(
                        hidden_states=hidden_states,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_mask,
                        num_cond_latents=N_cond,
                    )

            pred_target = pred[:, :, T_cond:].to(torch.float32)
            velocity_target = (noise - target_latents).to(torch.float32)
            total_loss += F.mse_loss(pred_target, velocity_target).item()
            count += 1

    return total_loss / max(count, 1)


# ============================================================================
# Video generation helper (using pipeline)
# ============================================================================

def generate_video_continuation(
    pipe: LongCatVideoPipeline,
    video_frames: list,
    prompt: str,
    num_cond_frames: int = 13,
    num_frames: int = 93,
    num_inference_steps: int = 50,
    guidance_scale: float = 4.0,
    seed: int = 42,
    resolution: str = "480p",
    device: str = "cuda",
    use_kv_cache: bool = True,
) -> np.ndarray:
    """Generate video continuation using the pipeline.

    Parameters
    ----------
    video_frames : list of PIL Images (conditioning frames)
    prompt : text prompt
    Returns np.ndarray of shape [N, H, W, 3] in [0, 1].
    """
    from PIL import Image

    vae_temporal_factor = 4
    num_frames_valid = (
        ((num_frames - 1 + vae_temporal_factor - 1) // vae_temporal_factor)
        * vae_temporal_factor + 1
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    output = pipe.generate_vc(
        video=video_frames,
        prompt=prompt,
        resolution=resolution,
        num_frames=num_frames_valid,
        num_cond_frames=num_cond_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        use_kv_cache=use_kv_cache,
        offload_kv_cache=False,
    )[0]

    return output  # np array [N, H, W, 3]


# ============================================================================
# Evaluation metrics
# ============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between pred and target tensors (both in [0, 1])."""
    mse = F.mse_loss(pred.float(), target.float())
    if mse == 0:
        return float("inf")
    return (10.0 * torch.log10(1.0 / mse)).item()


def compute_ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute average SSIM across frames. Tensors shape [T, C, H, W] in [0,1]."""
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(pred.device)
        return ssim_fn(pred, target).item()
    except ImportError:
        # Fallback: simple SSIM approximation
        mu_pred = pred.mean(dim=[2, 3], keepdim=True)
        mu_target = target.mean(dim=[2, 3], keepdim=True)
        sigma_pred_sq = ((pred - mu_pred) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma_target_sq = ((target - mu_target) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma_cross = ((pred - mu_pred) * (target - mu_target)).mean(dim=[2, 3], keepdim=True)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = (
            (2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)
        ) / (
            (mu_pred ** 2 + mu_target ** 2 + c1) * (sigma_pred_sq + sigma_target_sq + c2)
        )
        return ssim_map.mean().item()


def compute_lpips_batch(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute average LPIPS across frames. Tensors [T, C, H, W] in [0,1]."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net="alex", verbose=False).to(pred.device)
        # LPIPS expects [-1, 1]
        pred_scaled = pred * 2.0 - 1.0
        target_scaled = target * 2.0 - 1.0
        with torch.no_grad():
            scores = loss_fn(pred_scaled, target_scaled)
        return scores.mean().item()
    except ImportError:
        return float("nan")


def evaluate_generation_metrics(
    gen_output: np.ndarray,
    video_path: str,
    num_cond_frames: int,
    num_gen_frames: int,
    gen_start_frame: int,
    device: str = "cuda",
) -> Dict[str, float]:
    """Compute PSNR, SSIM, LPIPS between generated and ground truth frames.

    Parameters
    ----------
    gen_output : np.ndarray
        Full pipeline output of shape [N, H, W, 3] in [0, 1], where
        N = num_cond_frames + num_generated.
    video_path : str
        Path to the source video for loading GT frames.
    num_cond_frames : int
        Number of conditioning frames at the start of gen_output.
    num_gen_frames : int
        Number of generated frames to evaluate (may be less than total gen).
    gen_start_frame : int
        Anchor frame index — GT starts at this frame in the source video.
    device : str
        Device for LPIPS computation.

    Returns
    -------
    dict with keys: psnr, ssim, lpips
    """
    from PIL import Image
    import av

    gen_frames = gen_output[num_cond_frames:num_cond_frames + num_gen_frames]
    out_h, out_w = gen_frames.shape[1], gen_frames.shape[2]

    container = av.open(video_path)
    gt_pil = []
    decoded = 0
    for frame in container.decode(video=0):
        if decoded < gen_start_frame:
            decoded += 1
            continue
        if len(gt_pil) >= num_gen_frames:
            break
        gt_pil.append(frame.to_image())
        decoded += 1
    container.close()

    n_compare = min(len(gen_frames), len(gt_pil))
    if n_compare == 0:
        return {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}

    gt_np = np.stack([
        np.array(img.resize((out_w, out_h), Image.LANCZOS)) / 255.0
        for img in gt_pil[:n_compare]
    ], axis=0).astype(np.float32)  # [T, H, W, 3]
    gen_np = gen_frames[:n_compare].astype(np.float32)

    # PSNR (per-frame, then average)
    psnr_vals = []
    for i in range(n_compare):
        mse = np.mean((gen_np[i] - gt_np[i]) ** 2)
        if mse < 1e-10:
            psnr_vals.append(50.0)
        else:
            psnr_vals.append(float(10.0 * np.log10(1.0 / mse)))
    psnr = float(np.mean(psnr_vals))

    # SSIM (per-frame, then average)
    ssim_vals = []
    for i in range(n_compare):
        p = torch.from_numpy(gen_np[i]).permute(2, 0, 1).unsqueeze(0).float()
        g = torch.from_numpy(gt_np[i]).permute(2, 0, 1).unsqueeze(0).float()
        ssim_vals.append(_ssim_single(p, g))
    ssim = float(np.mean(ssim_vals))

    # LPIPS
    try:
        import lpips as lpips_lib
        loss_fn = lpips_lib.LPIPS(net="alex", verbose=False).to(device)
        lpips_vals = []
        for i in range(n_compare):
            p = torch.from_numpy(gen_np[i]).permute(2, 0, 1).unsqueeze(0).float().to(device)
            g = torch.from_numpy(gt_np[i]).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                s = loss_fn(p * 2 - 1, g * 2 - 1)
            lpips_vals.append(s.item())
        del loss_fn
        torch.cuda.empty_cache()
        lpips_val = float(np.mean(lpips_vals))
    except ImportError:
        lpips_val = float("nan")

    return {"psnr": psnr, "ssim": ssim, "lpips": lpips_val}


def _ssim_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM for a single frame pair. Both [1, C, H, W] in [0, 1]."""
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)
        return ssim_fn(pred, target).item()
    except ImportError:
        mu_p = pred.mean(dim=[2, 3], keepdim=True)
        mu_g = target.mean(dim=[2, 3], keepdim=True)
        sig_p = ((pred - mu_p) ** 2).mean(dim=[2, 3], keepdim=True)
        sig_g = ((target - mu_g) ** 2).mean(dim=[2, 3], keepdim=True)
        sig_pg = ((pred - mu_p) * (target - mu_g)).mean(dim=[2, 3], keepdim=True)
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu_p * mu_g + c1) * (2 * sig_pg + c2)) / (
            (mu_p ** 2 + mu_g ** 2 + c1) * (sig_p + sig_g + c2)
        )
        return ssim_map.mean().item()


# ============================================================================
# Dataset helpers
# ============================================================================

def _normalize_caption(raw: Any) -> str:
    """Normalize metadata captions to a clean string.

    Handles:
    - plain strings
    - list/tuple values (pick first non-empty)
    - list-like strings such as "['cap1', 'cap2']"
    """
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)):
        for item in raw:
            s = str(item).strip()
            if s:
                return s
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                for item in parsed:
                    t = str(item).strip()
                    if t:
                        return t
        except (ValueError, SyntaxError):
            pass
    return s

def load_ucf101_video_list(
    data_dir: str,
    max_videos: int = 100,
    seed: int = 42,
    stratified: bool = True,
    validate_decodable: bool = False,
) -> List[Dict]:
    """Load a list of video entries from a dataset directory.

    Reads metadata.csv if present (columns: filename, caption, category).
    Falls back to scanning for video files and using directory names.

    Returns list of dicts with keys: video_path, caption, class_name.
    """
    import csv as _csv

    data_dir = Path(data_dir)
    video_entries = []

    meta_path = data_dir / "metadata.csv"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", row.get("video_path", ""))
                vp = data_dir / "videos" / fname
                if not vp.exists():
                    vp = data_dir / fname
                if not vp.exists():
                    continue
                caption = _normalize_caption(row.get("caption", row.get("text", "")))
                class_name = row.get("category", row.get("class_name", "unknown"))
                video_entries.append({
                    "video_path": str(vp),
                    "caption": caption,
                    "class_name": class_name,
                })
        if video_entries:
            print(f"  Loaded {len(video_entries)} videos from {meta_path}")
    
    if not video_entries:
        for ext in ("*.mp4", "*.avi"):
            for vp in sorted(data_dir.rglob(ext)):
                class_name = vp.parent.name if vp.parent != data_dir else "unknown"
                caption = class_name.replace("_", " ")
                video_entries.append({
                    "video_path": str(vp),
                    "caption": caption,
                    "class_name": class_name,
                })

    if not video_entries:
        raise FileNotFoundError(f"No video files found in {data_dir}")

    if validate_decodable:
        valid_entries = []
        bad_examples: List[str] = []
        for entry in video_entries:
            vp = entry.get("video_path")
            ok = True
            try:
                import av
                container = av.open(str(vp))
                try:
                    next(container.decode(video=0))
                except StopIteration:
                    ok = False
                finally:
                    container.close()
            except Exception:
                ok = False
            if ok:
                valid_entries.append(entry)
            elif len(bad_examples) < 5:
                bad_examples.append(str(vp))

        dropped = len(video_entries) - len(valid_entries)
        if dropped > 0:
            print(f"  Dropped {dropped} undecodable videos during dataset load.")
            for ex in bad_examples:
                print(f"    bad_video: {ex}")
        video_entries = valid_entries

    if not video_entries:
        raise FileNotFoundError(f"No decodable video files found in {data_dir}")

    rng = np.random.RandomState(seed)
    # Panda datasets should use plain random sampling to honor max_videos.
    data_dir_lower = str(data_dir).lower()
    if "panda" in data_dir_lower and stratified:
        print("  Stratified sampling disabled for Panda dataset path.")
        stratified = False

    if stratified:
        from collections import defaultdict
        from collections import Counter
        by_class = defaultdict(list)
        for entry in video_entries:
            by_class[entry["class_name"]].append(entry)

        n_classes = len(by_class)
        class_sizes = Counter(entry["class_name"] for entry in video_entries)
        singleton_ratio = (
            sum(1 for _, c in class_sizes.items() if c == 1) / max(n_classes, 1)
        )
        # Panda metadata often has many singleton "classes", which collapses
        # sampling to ~number of unique labels (e.g., 86). Fall back to random
        # sampling in that case so max_videos is respected.
        if singleton_ratio > 0.5 and n_classes > max_videos // 2:
            print(
                "  Stratified sampling disabled: many singleton classes "
                f"(classes={n_classes}, singleton_ratio={singleton_ratio:.2f})."
            )
            rng.shuffle(video_entries)
            return video_entries[:max_videos]

        per_class = max(1, max_videos // n_classes)
        selected = []
        leftover = []
        for cls in sorted(by_class.keys()):
            entries = by_class[cls]
            rng.shuffle(entries)
            selected.extend(entries[:per_class])
            leftover.extend(entries[per_class:])

        if len(selected) < max_videos and leftover:
            rng.shuffle(leftover)
            selected.extend(leftover[: max_videos - len(selected)])

        rng.shuffle(selected)
        return selected[:max_videos]
    else:
        rng.shuffle(video_entries)
        return video_entries[:max_videos]


def load_panda70m_video_list(
    data_dir: str,
    meta_path: Optional[str] = None,
    max_videos: int = 100,
    seed: int = 42,
    validate_decodable: bool = False,
) -> List[Dict]:
    """Load video list from a Panda-70M style dataset."""
    data_dir = Path(data_dir)
    video_entries = []

    if meta_path and os.path.exists(meta_path):
        import csv
        with open(meta_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", row.get("video_path", ""))
                vp = data_dir / "videos" / fname
                if not vp.exists():
                    vp = data_dir / fname
                if vp.exists():
                    video_entries.append({
                        "video_path": str(vp),
                        "caption": _normalize_caption(row.get("caption", row.get("text", ""))),
                        "class_name": "panda70m",
                    })
    else:
        for vp in sorted(data_dir.rglob("*.mp4")):
            video_entries.append({
                "video_path": str(vp),
                "caption": "A video clip",
                "class_name": "panda70m",
            })

    if validate_decodable:
        valid_entries = []
        bad_examples: List[str] = []
        for entry in video_entries:
            vp = entry.get("video_path")
            ok = True
            try:
                import av
                container = av.open(str(vp))
                try:
                    next(container.decode(video=0))
                except StopIteration:
                    ok = False
                finally:
                    container.close()
            except Exception:
                ok = False
            if ok:
                valid_entries.append(entry)
            elif len(bad_examples) < 5:
                bad_examples.append(str(vp))

        dropped = len(video_entries) - len(valid_entries)
        if dropped > 0:
            print(f"  Dropped {dropped} undecodable Panda videos during dataset load.")
            for ex in bad_examples:
                print(f"    bad_video: {ex}")
        video_entries = valid_entries

    rng = np.random.RandomState(seed)
    rng.shuffle(video_entries)
    return video_entries[:max_videos]


# ============================================================================
# Caption quality guard
# ============================================================================

_GENERIC_CAPTIONS = {
    "",
    "video",
    "videos",
    "a video",
    "a video clip",
    "video clip",
    "unknown",
    "none",
    "nan",
}


def analyze_caption_quality(
    video_entries: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Compute caption quality stats for a list of video entries."""
    from collections import Counter

    total = len(video_entries)
    captions = [str(v.get("caption", "")).strip() for v in video_entries]
    nonempty = [c for c in captions if c]
    lower = [c.lower() for c in nonempty]

    counter = Counter(lower)
    unique_count = len(counter)
    nonempty_count = len(nonempty)
    nonempty_ratio = (nonempty_count / total) if total else 0.0
    unique_ratio = (unique_count / nonempty_count) if nonempty_count else 0.0

    top_items = counter.most_common(max(int(top_k), 1))
    top1_caption = top_items[0][0] if top_items else ""
    top1_count = top_items[0][1] if top_items else 0
    top1_ratio = (top1_count / nonempty_count) if nonempty_count else 0.0

    avg_len = float(np.mean([len(c) for c in nonempty])) if nonempty else 0.0

    return {
        "total": total,
        "nonempty_count": nonempty_count,
        "nonempty_ratio": nonempty_ratio,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        "top1_caption": top1_caption,
        "top1_count": top1_count,
        "top1_ratio": top1_ratio,
        "avg_caption_len": avg_len,
        "top_captions": top_items,
    }


def validate_caption_quality(
    video_entries: List[Dict[str, Any]],
    *,
    mode: str = "fail",
    min_nonempty_ratio: float = 0.95,
    min_unique_ratio: float = 0.10,
    max_top1_ratio: float = 0.50,
    max_generic_top1_ratio: float = 0.20,
    top_k: int = 5,
    context: str = "",
) -> Dict[str, Any]:
    """Validate caption quality and optionally fail-fast on suspicious inputs."""
    mode = (mode or "fail").lower()
    if mode not in {"fail", "warn", "off"}:
        raise ValueError(f"Invalid caption guard mode: {mode}")

    stats = analyze_caption_quality(video_entries, top_k=top_k)
    total = stats["total"]
    prefix = f"[caption_guard:{context}]" if context else "[caption_guard]"

    print(
        f"{prefix} total={total} nonempty_ratio={stats['nonempty_ratio']:.4f} "
        f"unique_ratio={stats['unique_ratio']:.4f} top1_ratio={stats['top1_ratio']:.4f} "
        f"avg_len={stats['avg_caption_len']:.1f}"
    )
    if stats["top_captions"]:
        print(f"{prefix} top captions:")
        for cap, count in stats["top_captions"]:
            print(f"  - {count:4d} | {cap[:180]}")

    if mode == "off" or total < 20:
        return stats

    reasons: List[str] = []
    if stats["nonempty_ratio"] < float(min_nonempty_ratio):
        reasons.append(
            f"nonempty_ratio={stats['nonempty_ratio']:.4f} < {float(min_nonempty_ratio):.4f}"
        )
    if stats["unique_ratio"] < float(min_unique_ratio):
        reasons.append(
            f"unique_ratio={stats['unique_ratio']:.4f} < {float(min_unique_ratio):.4f}"
        )
    if stats["top1_ratio"] > float(max_top1_ratio):
        reasons.append(
            f"top1_ratio={stats['top1_ratio']:.4f} > {float(max_top1_ratio):.4f}"
        )
    if (
        stats["top1_caption"] in _GENERIC_CAPTIONS
        and stats["top1_ratio"] > float(max_generic_top1_ratio)
    ):
        reasons.append(
            "generic top caption dominates "
            f"('{stats['top1_caption']}' ratio={stats['top1_ratio']:.4f} > "
            f"{float(max_generic_top1_ratio):.4f})"
        )

    if reasons:
        msg = f"{prefix} suspicious captions detected: " + "; ".join(reasons)
        if mode == "warn":
            print(f"WARNING: {msg}")
        else:
            raise RuntimeError(msg)

    return stats


def apply_fixed_caption(
    video_entries: List[Dict[str, Any]],
    fixed_caption: Optional[str],
    *,
    context: str = "eval",
) -> List[Dict[str, Any]]:
    """Override all captions with one fixed caption string."""
    if fixed_caption is None:
        return video_entries
    cap = str(fixed_caption).strip()
    # If callers accidentally pass shell-quoted literals (e.g. '"videos"'),
    # normalize back to plain text to avoid silent caption drift.
    if len(cap) >= 2 and cap[0] == cap[-1] and cap[0] in ("'", '"'):
        cap = cap[1:-1]
    for row in video_entries:
        row["caption"] = cap
    print(f"[caption_override:{context}] applied fixed caption to {len(video_entries)} videos: {cap!r}")
    return video_entries


# ============================================================================
# Data augmentation for TTA
# ============================================================================

def parse_speed_factors(raw: str) -> List[float]:
    """Parse comma-separated speed factors string, e.g. '0.5,2.0'."""
    if not raw:
        return []
    parts = [p.strip() for p in raw.split(",")]
    return [float(p) for p in parts if p]


def _rotation_scale(h: int, w: int, degrees: float) -> float:
    """Compute zoom scale needed to avoid black borders after rotation."""
    rad = abs(degrees) * (3.141592653589793 / 180.0)
    cos_a = abs(torch.cos(torch.tensor(rad)).item())
    sin_a = abs(torch.sin(torch.tensor(rad)).item())
    new_w = w * cos_a + h * sin_a
    new_h = h * cos_a + w * sin_a
    return max(new_w / w, new_h / h)


def _rotate_clip(pixel_frames: torch.Tensor, degrees: float, zoom: bool = False) -> torch.Tensor:
    """Rotate all frames of a clip by *degrees*.

    Parameters
    ----------
    pixel_frames : [1, C, T, H, W]
    degrees : rotation angle (can be negative)
    zoom : if True, scale up so rotated content fills the canvas

    Returns
    -------
    Rotated clip of same shape [1, C, T, H, W].
    """
    import torchvision.transforms.functional as TF
    from torchvision.transforms.functional import InterpolationMode

    clip = pixel_frames[0].permute(1, 0, 2, 3)  # [T, C, H, W]
    h, w = int(clip.shape[-2]), int(clip.shape[-1])
    scale = _rotation_scale(h, w, degrees) if zoom else 1.0
    rotated = torch.stack(
        [
            TF.affine(
                frame,
                degrees,
                translate=[0, 0],
                scale=scale,
                shear=[0.0, 0.0],
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            )
            for frame in clip
        ],
        dim=0,
    )
    return rotated.permute(1, 0, 2, 3).unsqueeze(0)


def build_augmented_pixel_variants(
    pixel_frames: torch.Tensor,
    *,
    enable_flip: bool = False,
    rotate_deg: float = 0.0,
    rotate_random_min: float = 5.0,
    rotate_random_max: float = 15.0,
    rotate_random_count: int = 2,
    rotate_random_step: float = 1.0,
    rotate_zoom: bool = True,
    speed_factors: Optional[Iterable[float]] = None,
) -> List[Dict[str, Any]]:
    """Build augmented pixel variants from a conditioning clip.

    Each variant is a dict with keys: pixel_frames, name.

    Parameters
    ----------
    pixel_frames : [1, C, T, H, W] in [-1, 1]
    enable_flip : include a horizontal-flip variant
    rotate_deg : if > 0, add fixed ±deg rotations
    rotate_random_min/max/count/step : random-rotation variants
    rotate_zoom : zoom to fill canvas after rotating
    speed_factors : temporal speed changes (e.g. 0.5 = slow, 2.0 = fast)

    Returns
    -------
    List of variant dicts.  The first is always the original clip.
    """
    variants: List[Dict[str, Any]] = []
    t_len = int(pixel_frames.shape[2])

    # Original (always first)
    variants.append({"pixel_frames": pixel_frames, "name": "orig"})

    # Horizontal flip
    if enable_flip:
        variants.append({
            "pixel_frames": pixel_frames.flip(dims=[4]),
            "name": "flip_h",
        })

    # Fixed rotations
    if rotate_deg and rotate_deg > 0:
        for deg in (-rotate_deg, rotate_deg):
            variants.append({
                "pixel_frames": _rotate_clip(pixel_frames, deg, zoom=rotate_zoom),
                "name": f"rotate_{deg:+.1f}",
            })

    # Random rotations
    if rotate_random_count and rotate_random_count > 0:
        rmin = rotate_random_min if rotate_random_min is not None else 0.0
        rmax = rotate_random_max if rotate_random_max is not None else 0.0
        if rmin > rmax:
            rmin, rmax = rmax, rmin

        if rotate_random_step and rotate_random_step > 0:
            options = torch.arange(rmin, rmax + 1e-6, rotate_random_step)
            if len(options) == 0:
                options = torch.tensor([rmin])
            idx = torch.randint(0, len(options), (rotate_random_count,))
            angles = options[idx].tolist()
        else:
            angles = torch.empty(rotate_random_count).uniform_(rmin, rmax).tolist()

        for deg in angles:
            if abs(deg) < 1e-6:
                continue
            variants.append({
                "pixel_frames": _rotate_clip(pixel_frames, float(deg), zoom=rotate_zoom),
                "name": f"rotate_rand_{float(deg):+.1f}",
            })

    # Temporal speed changes
    if speed_factors:
        device = pixel_frames.device
        for factor in speed_factors:
            if factor == 1.0:
                continue
            if factor > 1.0:
                stride = max(2, int(round(factor)))
                idx = torch.arange(0, t_len, step=stride, device=device)
                variants.append({
                    "pixel_frames": pixel_frames[:, :, idx, :, :],
                    "name": f"speed_{stride}x",
                })
            elif factor < 1.0:
                repeat = max(2, int(round(1.0 / factor)))
                idx = torch.arange(t_len, device=device).repeat_interleave(repeat)[:t_len]
                variants.append({
                    "pixel_frames": pixel_frames[:, :, idx, :, :],
                    "name": f"slow_{repeat}x",
                })

    return variants


def build_augmented_latent_variants(
    pixel_frames: torch.Tensor,
    base_latents: torch.Tensor,
    vae: AutoencoderKLWan,
    *,
    enable_flip: bool = False,
    rotate_deg: float = 0.0,
    rotate_random_min: float = 5.0,
    rotate_random_max: float = 15.0,
    rotate_random_count: int = 2,
    rotate_random_step: float = 1.0,
    rotate_zoom: bool = True,
    speed_factors: Optional[Iterable[float]] = None,
) -> List[Dict[str, Any]]:
    """Build augmented latent variants from a conditioning clip.

    Augments at the pixel level, then encodes through the VAE.
    The 'orig' variant reuses the already-encoded *base_latents*
    to avoid redundant encoding.

    Returns list of dicts with keys: latents, name.
    """
    pixel_variants = build_augmented_pixel_variants(
        pixel_frames,
        enable_flip=enable_flip,
        rotate_deg=rotate_deg,
        rotate_random_min=rotate_random_min,
        rotate_random_max=rotate_random_max,
        rotate_random_count=rotate_random_count,
        rotate_random_step=rotate_random_step,
        rotate_zoom=rotate_zoom,
        speed_factors=speed_factors,
    )

    latent_variants: List[Dict[str, Any]] = []
    for item in pixel_variants:
        if item["name"] == "orig":
            latents = base_latents
        else:
            latents = encode_video(vae, item["pixel_frames"], normalize=True)
        latent_variants.append({
            "latents": latents,
            "name": item["name"],
        })

    return latent_variants


def split_tta_latents(
    latents: torch.Tensor,
    num_context_latents: int,
    holdout_fraction: float = 0.25,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split latents into context / train / val along the temporal axis.

    Parameters
    ----------
    latents : [B, C, T, H, W] – all conditioning-region latents
    num_context_latents : number of leading latent frames to keep as clean
                          context (these are never noised and always get
                          ``timestep=0``).
    holdout_fraction : fraction of the remaining frames held out for val.

    Returns
    -------
    (cond_latents, train_latents, val_latents)
        cond_latents  : [B, C, T_cond, H, W]  – clean context
        train_latents : [B, C, T_train, H, W] – used for the flow-matching loss
        val_latents   : [B, C, T_val, H, W]   – held out for early stopping
    """
    T_total = latents.shape[2]
    T_cond = min(num_context_latents, T_total - 1)  # at least 1 non-cond
    remainder = T_total - T_cond
    T_val = max(1, int(remainder * holdout_fraction))
    T_train = remainder - T_val

    if T_train < 1:
        # Not enough frames: use everything for training, no val
        T_train = remainder
        T_val = 0

    cond = latents[:, :, :T_cond].contiguous()
    train = latents[:, :, T_cond:T_cond + T_train].contiguous()
    val = latents[:, :, T_cond + T_train:].contiguous() if T_val > 0 else None
    return cond, train, val


def add_tta_frame_args(parser):
    """Add TTA-specific frame split CLI arguments."""
    g = parser.add_argument_group("TTA frame split")
    g.add_argument(
        "--tta-total-frames", type=int, default=None,
        help="Total pixel frames before the anchor to load for TTA "
             "(default: same as --num-cond-frames, i.e. no separate train/val)."
    )
    g.add_argument(
        "--tta-context-frames", type=int, default=None,
        help="Number of leading pixel frames treated as clean context "
             "(timestep=0). Defaults to --num-cond-frames."
    )
    return parser


def add_caption_guard_args(parser):
    """Add caption quality guard CLI flags used across all TTA runners."""
    g = parser.add_argument_group("Caption quality guard")
    g.add_argument(
        "--caption-guard-mode",
        type=str,
        default="fail",
        choices=["fail", "warn", "off"],
        help="How to handle suspicious caption distributions.",
    )
    g.add_argument(
        "--caption-guard-min-nonempty-ratio",
        type=float,
        default=0.95,
        help="Fail/warn if non-empty caption ratio drops below this.",
    )
    g.add_argument(
        "--caption-guard-min-unique-ratio",
        type=float,
        default=0.10,
        help="Fail/warn if unique/non-empty caption ratio drops below this.",
    )
    g.add_argument(
        "--caption-guard-max-top1-ratio",
        type=float,
        default=0.50,
        help="Fail/warn if top-1 caption ratio exceeds this.",
    )
    g.add_argument(
        "--caption-guard-max-generic-top1-ratio",
        type=float,
        default=0.20,
        help="Fail/warn if generic top caption dominates above this ratio.",
    )
    g.add_argument(
        "--caption-guard-topk",
        type=int,
        default=5,
        help="How many top captions to print in guard diagnostics.",
    )
    return parser


def add_caption_override_args(parser):
    """Add global caption override CLI flag used for controlled ablations."""
    g = parser.add_argument_group("Caption override")
    g.add_argument(
        "--fixed-caption",
        type=str,
        default=None,
        help="If set, override every sample caption with this exact string.",
    )
    return parser


def add_feature_frame_guard_args(parser):
    """Add guard mode for feature frame-budget validation."""
    g = parser.add_argument_group("Feature frame budget guard")
    g.add_argument(
        "--feature-frame-guard-mode",
        type=str,
        default="fail",
        choices=["fail", "warn", "off"],
        help="How to handle incompatible TTA frame geometry for enabled features (ES/CLIP).",
    )
    return parser


def _estimate_latent_len(num_pixel_frames: int, vae_t_scale: int = 4) -> int:
    n = max(1, int(num_pixel_frames))
    return 1 + (n - 1) // int(vae_t_scale)


def estimate_tta_split_budget(
    tta_total_frames: int,
    tta_context_frames: int,
    holdout_fraction: float = 0.25,
    vae_t_scale: int = 4,
) -> Dict[str, int]:
    """Estimate latent split sizes used by split_tta_latents()."""
    t_total = _estimate_latent_len(tta_total_frames, vae_t_scale=vae_t_scale)
    t_ctx_req = _estimate_latent_len(tta_context_frames, vae_t_scale=vae_t_scale)

    # Mirror split_tta_latents() behavior
    t_cond = min(t_ctx_req, t_total - 1)
    remainder = t_total - t_cond
    t_val = max(1, int(remainder * float(holdout_fraction)))
    t_train = remainder - t_val
    if t_train < 1:
        t_train = remainder
        t_val = 0

    return {
        "total_latents": int(t_total),
        "cond_latents": int(t_cond),
        "train_latents": int(t_train),
        "val_latents": int(t_val),
    }


def _estimate_clip_candidate_frames(
    tta_total_frames: int,
    sampling_mode: str,
    late_fraction: float,
) -> int:
    window_len = max(1, int(tta_total_frames))
    mode = (sampling_mode or "full_window").lower()
    if mode == "late_only":
        frac = min(max(float(late_fraction), 1e-6), 1.0)
        return max(1, int(round(window_len * frac)))
    return window_len


def validate_tta_feature_budget(args, context: str = "") -> Dict[str, Any]:
    """Validate that enabled ES/CLIP features have sufficient TTA frame budget."""
    mode = str(getattr(args, "feature_frame_guard_mode", "fail")).lower()
    if mode not in {"fail", "warn", "off"}:
        mode = "fail"

    info: Dict[str, Any] = {}
    issues: List[str] = []
    prefix = f"[feature_budget:{context}]" if context else "[feature_budget]"

    tta_total = int(getattr(args, "tta_total_frames", 0) or 0)
    tta_context = int(getattr(args, "tta_context_frames", 0) or 0)
    holdout = float(getattr(args, "es_holdout_fraction", 0.25) or 0.25)
    split = estimate_tta_split_budget(tta_total, tta_context, holdout_fraction=holdout)
    info["split_budget"] = split

    es_enabled = not bool(getattr(args, "es_disable", False))
    if es_enabled and split["val_latents"] < 1:
        issues.append(
            "ES is enabled but estimated val_latents=0 "
            f"(tta_total_frames={tta_total}, tta_context_frames={tta_context}, "
            f"holdout={holdout}). Increase tta_total_frames and/or reduce tta_context_frames."
        )

    clip_enabled = bool(getattr(args, "clip_gate_enabled", False))
    if clip_enabled:
        sampling_mode = str(getattr(args, "clip_gate_sampling_mode", "full_window"))
        if bool(getattr(args, "clip_gate_late_only", False)):
            sampling_mode = "late_only"
        late_fraction = float(getattr(args, "clip_gate_late_fraction", 0.4) or 0.4)
        sample_frames = int(getattr(args, "clip_gate_sample_frames", 4) or 4)
        backend = str(getattr(args, "clip_gate_backend", "clip") or "clip").lower()
        required_frames = sample_frames if backend != "xclip" else 8
        candidates = _estimate_clip_candidate_frames(
            tta_total_frames=tta_total,
            sampling_mode=sampling_mode,
            late_fraction=late_fraction,
        )
        info["clip_candidates"] = int(candidates)
        info["clip_required_frames"] = int(required_frames)
        if candidates < required_frames:
            issues.append(
                "CLIP gate is enabled but candidate frames are fewer than required "
                f"(candidates={candidates}, required={required_frames}, "
                f"tta_total_frames={tta_total}, sampling_mode={sampling_mode}, "
                f"late_fraction={late_fraction}). Increase tta_total_frames and/or adjust sampling."
            )

    if mode != "off":
        print(
            f"{prefix} split(total={split['total_latents']}, cond={split['cond_latents']}, "
            f"train={split['train_latents']}, val={split['val_latents']})"
        )
        if "clip_candidates" in info:
            print(
                f"{prefix} clip_candidates={info['clip_candidates']} "
                f"required={info['clip_required_frames']}"
            )

    if issues:
        msg = f"{prefix} " + " | ".join(issues)
        if mode == "warn":
            print(f"WARNING: {msg}")
        elif mode == "fail":
            raise RuntimeError(msg)
    return info


def add_clip_gate_args(parser):
    """Add CLIP gate CLI flags used across all TTA runners."""
    g = parser.add_argument_group("CLIP gate")
    g.add_argument(
        "--clip-gate-enabled",
        action="store_true",
        help="Enable CLIP-based per-sample gate before TTA optimization.",
    )
    g.add_argument(
        "--clip-gate-threshold",
        type=float,
        default=0.0,
        help="Skip TTA when CLIP alignment score is below this threshold.",
    )
    g.add_argument(
        "--clip-gate-backend",
        type=str,
        default="clip",
        choices=["clip", "xclip"],
        help="Alignment backend: image CLIP (clip) or video-native X-CLIP (xclip).",
    )
    g.add_argument(
        "--clip-gate-model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="HF model id for selected backend "
             "(e.g., openai/clip-vit-large-patch14 or microsoft/xclip-base-patch32).",
    )
    g.add_argument(
        "--clip-gate-sample-frames",
        type=int,
        default=4,
        help="Number of frames sampled from the TTA window for CLIP scoring.",
    )
    g.add_argument(
        "--clip-gate-aggregation",
        type=str,
        default="mean",
        choices=["mean", "min", "max"],
        help="How to aggregate per-frame CLIP scores into one gate score.",
    )
    g.add_argument(
        "--clip-gate-sampling-mode",
        type=str,
        default="full_window",
        choices=["full_window", "late_only"],
        help="Frame sampling region inside TTA window for CLIP scoring.",
    )
    g.add_argument(
        "--clip-gate-late-fraction",
        type=float,
        default=0.4,
        help="Fraction of trailing TTA window to use when mode is late_only.",
    )
    g.add_argument(
        "--clip-gate-late-only",
        action="store_true",
        help="Compatibility alias for --clip-gate-sampling-mode late_only.",
    )
    g.add_argument(
        "--clip-gate-fail-open",
        action="store_true",
        help="If CLIP scoring fails, continue with TTA instead of failing run.",
    )
    g.add_argument(
        "--clip-gate-fail-closed",
        action="store_false",
        dest="clip_gate_fail_open",
        help="If CLIP scoring fails, raise an error instead of continuing.",
    )
    g.add_argument(
        "--clip-gate-log-only",
        action="store_true",
        help="Compute and log CLIP scores but never skip TTA.",
    )
    parser.set_defaults(clip_gate_fail_open=True)
    return parser


def add_augmentation_args(parser):
    """Add common augmentation CLI flags to an argparse parser."""
    group = parser.add_argument_group("Data augmentation")
    group.add_argument("--aug-enabled", action="store_true",
                       help="Enable data augmentation during TTA")
    group.add_argument("--aug-flip", action="store_true",
                       help="Enable horizontal flip augmentation")
    group.add_argument("--aug-rotate-deg", type=float, default=10.0,
                       help="Fixed rotation degrees (applies ±deg)")
    group.add_argument("--aug-rotate-random-min", type=float, default=5.0,
                       help="Random rotation min degrees")
    group.add_argument("--aug-rotate-random-max", type=float, default=15.0,
                       help="Random rotation max degrees")
    group.add_argument("--aug-rotate-random-count", type=int, default=2,
                       help="Number of random rotation variants")
    group.add_argument("--aug-rotate-random-step", type=float, default=1.0,
                       help="Discrete step size for random rotations")
    group.add_argument(
        "--no-aug-rotate-zoom",
        action="store_false",
        dest="aug_rotate_zoom",
        help="Disable zoom when rotating (may introduce black borders)",
    )
    parser.set_defaults(aug_rotate_zoom=True)
    group.add_argument("--aug-speed-factors", type=str, default="",
                       help="Comma-separated speed factors (e.g. 0.5,2.0)")
    return parser


_CLIP_SCORER_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
_XCLIP_SCORER_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}
_XCLIP_DEFAULT_MODEL = "microsoft/xclip-base-patch32"


def _get_clip_scorer(model_name: str, device: str):
    """Lazy-load and cache a CLIP model+processor pair."""
    key = (model_name, device)
    if key in _CLIP_SCORER_CACHE:
        return _CLIP_SCORER_CACHE[key]

    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    _CLIP_SCORER_CACHE[key] = (model, processor)
    return model, processor


def _get_xclip_scorer(model_name: str, device: str):
    """Lazy-load and cache an X-CLIP model+processor pair."""
    key = (model_name, device)
    if key in _XCLIP_SCORER_CACHE:
        return _XCLIP_SCORER_CACHE[key]

    from transformers import XCLIPModel, XCLIPProcessor

    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    _XCLIP_SCORER_CACHE[key] = (model, processor)
    return model, processor


def _sample_clip_frame_offsets(
    window_len: int,
    sample_frames: int,
    sampling_mode: str = "full_window",
    late_fraction: float = 0.4,
) -> List[int]:
    """Pick frame offsets inside a TTA window for CLIP scoring."""
    if window_len <= 0:
        return []

    if sampling_mode == "late_only":
        frac = min(max(float(late_fraction), 1e-6), 1.0)
        late_len = max(1, int(round(window_len * frac)))
        candidate_start = max(0, window_len - late_len)
        candidates = list(range(candidate_start, window_len))
    else:
        candidates = list(range(window_len))

    if not candidates:
        return []

    k = max(1, min(int(sample_frames), len(candidates)))
    if k == 1:
        return [candidates[-1]]

    pos = np.linspace(0, len(candidates) - 1, num=k, dtype=int)
    return [candidates[int(i)] for i in pos]


def _decode_video_window_for_clip(
    video_path: str,
    start_frame: int,
    num_frames: int,
) -> List[Any]:
    """Decode a contiguous frame window and return PIL frames."""
    import av

    if num_frames <= 0:
        return []

    container = av.open(video_path)
    frames = []
    decoded = 0
    for frame in container.decode(video=0):
        if decoded < start_frame:
            decoded += 1
            continue
        if len(frames) >= num_frames:
            break
        frames.append(frame.to_image())
        decoded += 1
    container.close()

    if not frames:
        return []
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return frames[:num_frames]


def evaluate_clip_gate(
    *,
    video_path: str,
    caption: str,
    gen_start_frame: int,
    tta_total_frames: int,
    device: str = "cuda",
    enabled: bool = False,
    threshold: float = 0.0,
    backend: str = "clip",
    model_name: str = "openai/clip-vit-large-patch14",
    sample_frames: int = 4,
    aggregation: str = "mean",
    sampling_mode: str = "full_window",
    late_fraction: float = 0.4,
    late_only: bool = False,
    fail_open: bool = True,
    log_only: bool = False,
) -> Dict[str, Any]:
    """Compute CLIP gate decision for a single sample.

    Returns a dict with stable logging fields used by all runners.
    """
    if late_only:
        sampling_mode = "late_only"

    t_eval_start = time.time()
    info: Dict[str, Any] = {
        "clip_gate_enabled": bool(enabled),
        "clip_gate_backend": backend,
        "clip_gate_threshold": float(threshold),
        "clip_gate_model": model_name,
        "clip_gate_sample_frames": int(sample_frames),
        "clip_gate_aggregation": aggregation,
        "clip_gate_sampling_mode": sampling_mode,
        "clip_gate_late_fraction": float(late_fraction),
        "clip_gate_log_only": bool(log_only),
        "clip_gate_fail_open": bool(fail_open),
        "clip_alignment_score": None,
        "clip_gate_window_start_frame": max(0, int(gen_start_frame - tta_total_frames)),
        "clip_gate_window_end_frame": int(gen_start_frame),
        "clip_gate_sampled_frames": [],
        "clip_gate_decision": "run_tta",
        "clip_gate_reason": "gate_disabled",
        "tta_skipped": False,
        "clip_gate_eval_time": 0.0,
    }

    if not enabled:
        info["clip_gate_eval_time"] = float(time.time() - t_eval_start)
        return info
    if not caption or not caption.strip():
        info["clip_gate_reason"] = "empty_caption"
        info["clip_gate_eval_time"] = float(time.time() - t_eval_start)
        return info

    try:
        backend = (backend or "clip").lower()
        if backend not in {"clip", "xclip"}:
            raise ValueError(f"Unsupported clip gate backend: {backend}")
        if backend == "xclip" and model_name == "openai/clip-vit-large-patch14":
            model_name = _XCLIP_DEFAULT_MODEL
            info["clip_gate_model"] = model_name
        info["clip_gate_backend"] = backend

        window_start = info["clip_gate_window_start_frame"]
        window_len = max(1, int(tta_total_frames))
        window_frames = _decode_video_window_for_clip(
            video_path=video_path,
            start_frame=window_start,
            num_frames=window_len,
        )
        if not window_frames:
            raise RuntimeError("No frames available in CLIP gate window.")

        offsets = _sample_clip_frame_offsets(
            window_len=len(window_frames),
            sample_frames=sample_frames,
            sampling_mode=sampling_mode,
            late_fraction=late_fraction,
        )
        if not offsets:
            raise RuntimeError("No frame offsets selected for CLIP scoring.")
        images = [window_frames[i] for i in offsets]
        info["clip_gate_sampled_frames"] = [window_start + i for i in offsets]

        if backend == "xclip":
            model, processor = _get_xclip_scorer(model_name, device)
            required_frames = int(getattr(model.config, "num_frames", len(images)) or len(images))
            if required_frames > 0 and len(images) != required_frames:
                if len(images) < required_frames:
                    pad_n = required_frames - len(images)
                    images = images + [images[-1].copy() for _ in range(pad_n)]
                    if info["clip_gate_sampled_frames"]:
                        last_idx = info["clip_gate_sampled_frames"][-1]
                        info["clip_gate_sampled_frames"].extend([last_idx] * pad_n)
                else:
                    keep = np.linspace(0, len(images) - 1, num=required_frames, dtype=int).tolist()
                    images = [images[i] for i in keep]
                    info["clip_gate_sampled_frames"] = [
                        info["clip_gate_sampled_frames"][i] for i in keep
                    ]
            # X-CLIP expects each video as a list of RGB frames, not a stacked
            # ndarray batch. Passing a stacked array can trigger PIL conversion
            # shape errors in some transformers versions.
            frames_rgb = [img.convert("RGB") for img in images]
            proc = processor(
                text=[caption],
                videos=[frames_rgb],
                return_tensors="pt",
                truncation=True,
                padding=True,
            )
            proc = {
                k: (v.to(device) if torch.is_tensor(v) else v)
                for k, v in proc.items()
            }
            text_inputs = {
                k: proc[k]
                for k in ("input_ids", "attention_mask")
                if k in proc
            }
            video_inputs = {
                k: proc[k]
                for k in ("pixel_values", "pixel_mask")
                if k in proc
            }

            with torch.no_grad():
                text_feat = model.get_text_features(**text_inputs)
                video_feat = model.get_video_features(**video_inputs)
                text_feat = F.normalize(text_feat, dim=-1)
                video_feat = F.normalize(video_feat, dim=-1)
                frame_scores = (video_feat @ text_feat.T).view(1)
        else:
            model, processor = _get_clip_scorer(model_name, device)
            text_inputs = processor(text=[caption], return_tensors="pt", truncation=True).to(device)
            image_inputs = processor(images=images, return_tensors="pt").to(device)

            with torch.no_grad():
                text_feat = model.get_text_features(**text_inputs)
                image_feat = model.get_image_features(**image_inputs)
                text_feat = F.normalize(text_feat, dim=-1)
                image_feat = F.normalize(image_feat, dim=-1)
                frame_scores = (image_feat @ text_feat.T).squeeze(-1)

        if aggregation == "min":
            score = frame_scores.min()
        elif aggregation == "max":
            score = frame_scores.max()
        else:
            score = frame_scores.mean()

        score_val = float(score.item())
        info["clip_alignment_score"] = score_val

        if log_only:
            info["clip_gate_decision"] = "run_tta"
            info["clip_gate_reason"] = "log_only"
            info["tta_skipped"] = False
        elif score_val < threshold:
            info["clip_gate_decision"] = "skip_tta"
            info["clip_gate_reason"] = "low_alignment"
            info["tta_skipped"] = True
        else:
            info["clip_gate_decision"] = "run_tta"
            info["clip_gate_reason"] = "passed_threshold"
            info["tta_skipped"] = False
        info["clip_gate_eval_time"] = float(time.time() - t_eval_start)
        return info
    except Exception as exc:
        if fail_open:
            info["clip_gate_decision"] = "run_tta"
            info["clip_gate_reason"] = "fail_open_error"
            info["clip_gate_error"] = str(exc)
            info["tta_skipped"] = False
            info["clip_gate_eval_time"] = float(time.time() - t_eval_start)
            return info
        raise


def summarize_clip_gate_stats(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate CLIP-gate statistics for summary.json."""
    enabled_rows = [r for r in results if r.get("clip_gate_enabled")]
    scores = [
        float(r["clip_alignment_score"])
        for r in enabled_rows
        if r.get("clip_alignment_score") is not None
    ]
    skipped = [r for r in enabled_rows if r.get("tta_skipped")]
    total_enabled = len(enabled_rows)
    skip_rate = (len(skipped) / total_enabled) if total_enabled else 0.0

    stats: Dict[str, Any] = {
        "num_enabled": total_enabled,
        "num_scored": len(scores),
        "num_skipped": len(skipped),
        "skip_rate": skip_rate,
    }
    if scores:
        stats.update({
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
        })
    return stats


# ============================================================================
# Video saving helper
# ============================================================================

def save_video_from_numpy(frames: np.ndarray, output_path: str, fps: int = 24):
    """Save video from numpy array [N, H, W, 3] in [0, 1]."""
    import imageio

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames_u8 = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    imageio.mimwrite(output_path, frames_u8, fps=fps, codec="libx264", quality=9)


# ============================================================================
# Checkpoint / result helpers
# ============================================================================

def save_results(results: dict, output_path: str):
    """Save experiment results to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_checkpoint(checkpoint_path: str) -> Optional[dict]:
    """Load a checkpoint JSON if it exists."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint: dict, checkpoint_path: str):
    """Save checkpoint JSON."""
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)


# ============================================================================
# Retrieval-based batch helpers
# ============================================================================

def build_retrieval_pool(
    pool_entries: List[Dict],
    model_name: str = "all-MiniLM-L6-v2",
) -> Tuple[np.ndarray, Any]:
    """Pre-compute normalised sentence embeddings for a pool of videos.

    Returns (embeddings, sentence_transformer_model) so the model can be
    reused for encoding query captions without reloading.
    """
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(model_name)
    captions = [v.get("caption", "") for v in pool_entries]
    embeddings = st_model.encode(
        captions, show_progress_bar=True, normalize_embeddings=True,
    )
    print(f"  Retrieval pool: {len(pool_entries)} videos, "
          f"embedding dim={embeddings.shape[1]}")
    return embeddings, st_model


def retrieve_neighbors(
    query_entry: Dict,
    pool_entries: List[Dict],
    pool_embeddings: np.ndarray,
    st_model: Any,
    k: int,
) -> List[Dict]:
    """Return up to *k-1* nearest neighbours from the pool for *query_entry*.

    The intended batch for TTA training is ``[query_entry] + neighbours`` so
    the total size is *k*.  When *k* <= 1, an empty list is returned.

    Matching is by cosine similarity of text-prompt embeddings.  The query
    video itself is excluded from the returned list (matched by absolute path).
    """
    if k <= 1:
        return []

    query_emb = st_model.encode(
        [query_entry.get("caption", "")], normalize_embeddings=True,
    )
    sims = (pool_embeddings @ query_emb.T).squeeze()

    query_path = os.path.abspath(query_entry.get("video_path", ""))
    ranked = np.argsort(-sims)

    neighbors: List[Dict] = []
    for idx in ranked:
        if len(neighbors) >= k - 1:
            break
        pool_path = os.path.abspath(pool_entries[int(idx)].get("video_path", ""))
        if pool_path == query_path:
            continue
        neighbors.append(pool_entries[int(idx)])

    return neighbors


# ============================================================================
# GPU memory helpers
# ============================================================================

def torch_gc():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================================
# Online FVD / FID accumulator  (no video files required on disk)
# ============================================================================

_I3D_HF_REPO = "kiwhansong/DFoT"
_I3D_HF_FILE = "metrics_models/i3d_torchscript.pt"
_I3D_FEATURE_DIM = 400
_FID_FEATURE_DIM = 2048
_MIN_I3D_FRAMES = 9
_DEFAULT_MIN_FVD_VIDEOS = 256
_COV_EPS = 1e-6


def _load_i3d_model(device: str) -> "torch.jit.ScriptModule":
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=_I3D_HF_REPO, filename=_I3D_HF_FILE)
    model = torch.jit.load(path, map_location=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _load_inception_v3_model(device: str) -> nn.Module:
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _pad_for_i3d(x: torch.Tensor) -> torch.Tensor:
    """Symmetric first/last-frame padding to >= 9 frames (DFoT protocol)."""
    T = x.shape[1]
    if T < _MIN_I3D_FRAMES:
        pad = (10 - T) // 2
        x = torch.cat(
            [
                x[:, 0:1].expand(-1, pad, -1, -1, -1).clone(),
                x,
                x[:, -1:].expand(-1, pad, -1, -1, -1).clone(),
            ],
            dim=1,
        )
    return x


def _frames_np_to_i3d_tensor(
    frames_np: np.ndarray, size: int = 224,
) -> torch.Tensor:
    """Convert [T, H, W, 3] float32 in [0,1] -> [1, T, C, H, W] tensor."""
    from torchvision.transforms import functional as TF
    from PIL import Image

    tensors = []
    for i in range(frames_np.shape[0]):
        img = Image.fromarray(
            (np.clip(frames_np[i], 0, 1) * 255).astype(np.uint8)
        )
        img = TF.resize(img, size, interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.center_crop(img, size)
        tensors.append(TF.to_tensor(img))
    return torch.stack(tensors, dim=0).unsqueeze(0)  # [1, T, C, H, W]


def _compute_frechet_distance(
    sum_a: np.ndarray,
    cov_sum_a: np.ndarray,
    n_a: int,
    sum_b: np.ndarray,
    cov_sum_b: np.ndarray,
    n_b: int,
    eps: float = _COV_EPS,
) -> float:
    """Frechet distance from running sums (float64)."""
    from scipy.linalg import sqrtm

    mu_a = sum_a / n_a
    mu_b = sum_b / n_b
    sigma_a = cov_sum_a / n_a - np.outer(mu_a, mu_a)
    sigma_b = cov_sum_b / n_b - np.outer(mu_b, mu_b)

    sigma_a += eps * np.eye(sigma_a.shape[0])
    sigma_b += eps * np.eye(sigma_b.shape[0])

    diff = mu_a - mu_b
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean))


class OnlineFrechetAccumulator:
    """Incrementally accumulate I3D (and optionally InceptionV3) features
    for online FVD / FID computation.  No video files on disk required."""

    def __init__(
        self,
        device: str = "cuda",
        compute_fid: bool = False,
        min_videos: int = _DEFAULT_MIN_FVD_VIDEOS,
    ):
        self.device = device
        self.compute_fid = compute_fid
        self.min_videos = min_videos

        self._i3d: Optional["torch.jit.ScriptModule"] = None
        self._inception: Optional[nn.Module] = None

        d = _I3D_FEATURE_DIM
        self._gen_sum = np.zeros(d, dtype=np.float64)
        self._gen_cov = np.zeros((d, d), dtype=np.float64)
        self._ref_sum = np.zeros(d, dtype=np.float64)
        self._ref_cov = np.zeros((d, d), dtype=np.float64)
        self._count = 0

        if compute_fid:
            fd = _FID_FEATURE_DIM
            self._fid_gen_sum = np.zeros(fd, dtype=np.float64)
            self._fid_gen_cov = np.zeros((fd, fd), dtype=np.float64)
            self._fid_ref_sum = np.zeros(fd, dtype=np.float64)
            self._fid_ref_cov = np.zeros((fd, fd), dtype=np.float64)
            self._fid_gen_frames = 0
            self._fid_ref_frames = 0

    def _ensure_models(self):
        if self._i3d is None:
            print("[FVD] Loading I3D (Kinetics-400 TorchScript)...")
            self._i3d = _load_i3d_model(self.device)
        if self.compute_fid and self._inception is None:
            print("[FID] Loading InceptionV3...")
            self._inception = _load_inception_v3_model(self.device)

    def _i3d_features(self, clip: torch.Tensor) -> np.ndarray:
        """clip: [1, T, C, H, W] in [0, 1] -> 400-dim feature (float64)."""
        clip = _pad_for_i3d(clip.to(self.device))
        clip = torch.clamp(2.0 * clip - 1.0, -1.0, 1.0)
        clip = clip.permute(0, 2, 1, 3, 4).contiguous()
        with torch.no_grad():
            feats = self._i3d(clip, rescale=False, resize=True, return_features=True)
        return feats.cpu().to(torch.float64).numpy().squeeze(0)  # (400,)

    def _inception_features(self, frames_np: np.ndarray) -> np.ndarray:
        """frames_np: [T, H, W, 3] in [0,1] -> [T, 2048] float64."""
        from torchvision.transforms import functional as TF
        from PIL import Image

        feats_list = []
        with torch.no_grad():
            for i in range(frames_np.shape[0]):
                img = Image.fromarray(
                    (np.clip(frames_np[i], 0, 1) * 255).astype(np.uint8)
                )
                img = TF.resize(img, 299, interpolation=TF.InterpolationMode.BILINEAR)
                img = TF.center_crop(img, 299)
                t = TF.normalize(
                    TF.to_tensor(img),
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ).unsqueeze(0).to(self.device)
                f = self._inception(t).cpu().to(torch.float64).numpy()  # (1, 2048)
                feats_list.append(f)
        return np.concatenate(feats_list, axis=0)  # (T, 2048)

    @staticmethod
    def _accumulate(
        feats: np.ndarray, feat_sum: np.ndarray, cov_sum: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Add a single feature vector (or rows of a matrix) to running stats."""
        if feats.ndim == 1:
            feat_sum += feats
            cov_sum += np.outer(feats, feats)
        else:
            feat_sum += feats.sum(axis=0)
            cov_sum += feats.T @ feats
        return feat_sum, cov_sum

    def update(
        self,
        gen_output: np.ndarray,
        video_path: str,
        num_cond_frames: int,
        num_gen_frames: int,
        gen_start_frame: int,
    ):
        """Feed one video's generated output + GT source for feature accumulation.

        Parameters match ``evaluate_generation_metrics`` so callers can pass
        the same arguments to both functions.
        """
        import av
        from PIL import Image

        self._ensure_models()

        gen_frames = gen_output[num_cond_frames:num_cond_frames + num_gen_frames]
        if gen_frames.shape[0] == 0:
            return

        try:
            container = av.open(video_path)
            gt_pil: list = []
            decoded = 0
            for frame in container.decode(video=0):
                if decoded < gen_start_frame:
                    decoded += 1
                    continue
                if len(gt_pil) >= num_gen_frames:
                    break
                gt_pil.append(frame.to_image())
                decoded += 1
            container.close()
        except Exception:
            return

        if len(gt_pil) == 0:
            return

        out_h, out_w = gen_frames.shape[1], gen_frames.shape[2]
        n_compare = min(len(gen_frames), len(gt_pil))
        gt_np = np.stack([
            np.array(img.resize((out_w, out_h), Image.LANCZOS)) / 255.0
            for img in gt_pil[:n_compare]
        ], axis=0).astype(np.float32)
        gen_np = gen_frames[:n_compare].astype(np.float32)

        gen_clip = _frames_np_to_i3d_tensor(gen_np)
        ref_clip = _frames_np_to_i3d_tensor(gt_np)

        gen_feat = self._i3d_features(gen_clip)
        ref_feat = self._i3d_features(ref_clip)

        self._gen_sum, self._gen_cov = self._accumulate(
            gen_feat, self._gen_sum, self._gen_cov,
        )
        self._ref_sum, self._ref_cov = self._accumulate(
            ref_feat, self._ref_sum, self._ref_cov,
        )
        self._count += 1

        if self.compute_fid and self._inception is not None:
            gen_fid = self._inception_features(gen_np)
            ref_fid = self._inception_features(gt_np)
            self._fid_gen_sum, self._fid_gen_cov = self._accumulate(
                gen_fid, self._fid_gen_sum, self._fid_gen_cov,
            )
            self._fid_ref_sum, self._fid_ref_cov = self._accumulate(
                ref_fid, self._fid_ref_sum, self._fid_ref_cov,
            )
            self._fid_gen_frames += gen_fid.shape[0]
            self._fid_ref_frames += ref_fid.shape[0]

    def compute(self) -> Dict[str, Any]:
        """Return FVD (and FID) metrics from accumulated statistics."""
        result: Dict[str, Any] = {}

        if self._count < 2:
            result["fvd"] = None
            result["fvd_num_videos"] = self._count
            result["fvd_error"] = "Need at least 2 videos for FVD"
            return result

        fvd = _compute_frechet_distance(
            self._gen_sum, self._gen_cov, self._count,
            self._ref_sum, self._ref_cov, self._count,
        )
        result["fvd"] = round(fvd, 6)
        result["fvd_num_videos"] = self._count
        result["fvd_feature_extractor"] = "i3d_kinetics400_torchscript"
        result["fvd_feature_dim"] = _I3D_FEATURE_DIM

        if self._count < self.min_videos:
            result["fvd_sample_size_warning"] = (
                f"FVD computed with {self._count} videos "
                f"(recommended >= {self.min_videos}). "
                f"Covariance estimate may be unreliable."
            )

        if self.compute_fid and self._fid_gen_frames >= 2:
            fid = _compute_frechet_distance(
                self._fid_gen_sum, self._fid_gen_cov, self._fid_gen_frames,
                self._fid_ref_sum, self._fid_ref_cov, self._fid_ref_frames,
            )
            result["fid"] = round(fid, 6)
            result["fid_num_frames_gen"] = self._fid_gen_frames
            result["fid_num_frames_ref"] = self._fid_ref_frames
            result["fid_feature_extractor"] = "inception_v3_imagenet"
            result["fid_feature_dim"] = _FID_FEATURE_DIM

        return result


# ============================================================================
# Online eval CLI args + finalization
# ============================================================================

def add_online_eval_args(parser: "argparse.ArgumentParser"):
    """Add --compute-fvd, --compute-fid, --compute-vbench, --min-fvd-videos."""
    grp = parser.add_argument_group("Online distributional metrics")
    grp.add_argument("--compute-fvd", action="store_true",
                     help="Compute FVD online (no saved videos needed)")
    grp.add_argument("--compute-fid", action="store_true",
                     help="Compute per-frame FID online (requires --compute-fvd)")
    grp.add_argument("--compute-vbench", action="store_true",
                     help="Run VBench++ at end of job (requires saved videos)")
    grp.add_argument("--min-fvd-videos", type=int,
                     default=_DEFAULT_MIN_FVD_VIDEOS,
                     help="Minimum videos before FVD is considered reliable "
                          f"(default: {_DEFAULT_MIN_FVD_VIDEOS})")


def aggregate_quality_metrics(summary: dict):
    """Compute avg PSNR, SSIM, LPIPS from per-video results and merge into summary."""
    successful = [r for r in summary.get("results", []) if r.get("success")]
    for key in ("psnr", "ssim", "lpips"):
        values = [r[key] for r in successful if r.get(key) is not None]
        summary[key] = round(float(np.mean(values)), 6) if values else None


def finalize_online_eval(
    accumulator: Optional[OnlineFrechetAccumulator],
    summary: dict,
    videos_dir: str,
    args,
):
    """Compute FVD/FID from the accumulator and optionally run VBench++.

    Merges results into *summary* in-place so they are saved alongside
    existing PSNR / SSIM / LPIPS metrics.
    """
    if accumulator is not None:
        print("\n[Online eval] Computing FVD/FID from accumulated features...")
        fvd_results = accumulator.compute()
        summary.update(fvd_results)
        for k, v in fvd_results.items():
            print(f"  {k}: {v}")

    vbench_skipped = True
    if getattr(args, "compute_vbench", False):
        mp4s = sorted(Path(videos_dir).glob("*.mp4")) if os.path.isdir(videos_dir) else []
        if mp4s:
            print(f"\n[VBench++] Running on {len(mp4s)} videos in {videos_dir}...")
            try:
                from vbench import VBench

                _VBENCH_DIMS = [
                    "subject_consistency",
                    "motion_smoothness",
                    "temporal_flickering",
                    "aesthetic_quality",
                    "imaging_quality",
                ]
                vb = VBench(device="cuda", full_json_dir=None)
                vbench_scores: Dict[str, Any] = {}
                video_paths = [str(p) for p in mp4s]
                for dim in _VBENCH_DIMS:
                    try:
                        score = vb.evaluate(
                            videos_path=video_paths,
                            name=dim,
                            dimension_list=[dim],
                            mode="i2v",
                        )
                        vbench_scores[dim] = (
                            float(score) if isinstance(score, (int, float)) else score
                        )
                        print(f"  {dim}: {vbench_scores[dim]}")
                    except Exception as exc:
                        print(f"  WARNING: VBench++ {dim} failed: {exc}",
                              file=sys.stderr)
                        vbench_scores[dim] = None
                summary["vbench"] = vbench_scores
                vbench_skipped = False
            except ImportError:
                print("  WARNING: vbench not installed, skipping VBench++",
                      file=sys.stderr)
        else:
            print("[VBench++] No saved videos found; skipping "
                  "(requires NO_SAVE_VIDEOS=0).")

    summary["vbench_skipped"] = vbench_skipped
