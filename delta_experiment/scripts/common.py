#!/usr/bin/env python3
"""
Shared utilities for LongCat-Video TTA experiments.

Handles model loading, video encoding, flow-matching loss computation,
dataset loading, and evaluation metrics.
"""

import gc
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
        checkpoint_dir, subfolder="dit", cp_split_hw=cp_split_hw, torch_dtype=dtype
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

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    output = pipe.generate_vc(
        video=video_frames,
        prompt=prompt,
        resolution=resolution,
        num_frames=num_frames,
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


# ============================================================================
# Dataset helpers
# ============================================================================

def load_ucf101_video_list(
    data_dir: str,
    max_videos: int = 100,
    seed: int = 42,
    stratified: bool = True,
) -> List[Dict]:
    """Load a list of video entries from a UCF101-style dataset directory.

    Returns list of dicts with keys: video_path, caption, class_name.
    """
    data_dir = Path(data_dir)
    video_entries = []

    # Look for video files
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

    # Stratified sampling
    rng = np.random.RandomState(seed)
    if stratified:
        # Group by class
        from collections import defaultdict
        by_class = defaultdict(list)
        for entry in video_entries:
            by_class[entry["class_name"]].append(entry)

        # Sample proportionally from each class
        n_classes = len(by_class)
        per_class = max(1, max_videos // n_classes)
        selected = []
        for cls in sorted(by_class.keys()):
            entries = by_class[cls]
            rng.shuffle(entries)
            selected.extend(entries[:per_class])

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
) -> List[Dict]:
    """Load video list from a Panda-70M style dataset."""
    data_dir = Path(data_dir)
    video_entries = []

    if meta_path and os.path.exists(meta_path):
        import csv
        with open(meta_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vp = data_dir / row.get("filename", row.get("video_path", ""))
                if vp.exists():
                    video_entries.append({
                        "video_path": str(vp),
                        "caption": row.get("caption", row.get("text", "")),
                        "class_name": "panda70m",
                    })
    else:
        for vp in sorted(data_dir.rglob("*.mp4")):
            video_entries.append({
                "video_path": str(vp),
                "caption": "A video clip",
                "class_name": "panda70m",
            })

    rng = np.random.RandomState(seed)
    rng.shuffle(video_entries)
    return video_entries[:max_videos]


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
# GPU memory helpers
# ============================================================================

def torch_gc():
    """Free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
