#!/usr/bin/env python3
"""
Shared utilities for CogVideoX-5B-I2V TTA backbone experiments.

Handles model loading, video encoding/decoding, flow-matching loss computation,
dataset loading, evaluation metrics, and the Delta-A wrapper for CogVideoX.

CogVideoX-5B-I2V architecture:
  - CogVideoXTransformer3DModel (30 blocks, 30 heads, 64 dim/head)
  - time_embed_dim: 512
  - in/out channels: 16
  - text_embed_dim: 4096 (T5 encoder)
  - VAE: AutoencoderKLCogVideoX (temporal 4x, spatial 8x)
  - Resolution: 720x480 fixed
  - Frames: 49 (6s @ 8fps)
"""

import csv
import gc
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Model loading
# ============================================================================

def load_cogvideo_components(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, Any]:
    """Load CogVideoX-5B-I2V components from local path or HuggingFace.

    Returns dict with keys: transformer, vae, text_encoder, tokenizer,
    scheduler, pipe.
    """
    from diffusers import CogVideoXImageToVideoPipeline

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        model_path, torch_dtype=dtype
    )

    return {
        "transformer": pipe.transformer,
        "vae": pipe.vae,
        "text_encoder": pipe.text_encoder,
        "tokenizer": pipe.tokenizer,
        "scheduler": pipe.scheduler,
        "pipe": pipe,
    }


# ============================================================================
# Video / latent helpers
# ============================================================================

def load_video_frames(
    video_path: str,
    num_frames: int,
    height: int = 480,
    width: int = 720,
    start_frame: int = 0,
) -> torch.Tensor:
    """Load video frames as a tensor [1, C, T, H, W] in [-1, 1].

    CogVideoX-5B-I2V expects 720x480 resolution. Panda-70M videos
    (typically 832x480) are resized to fit.
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
        frames.append(frame.to_ndarray(format="rgb24"))
        decoded += 1
    container.close()

    if len(frames) == 0:
        raise ValueError(f"No frames decoded from {video_path}")
    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames_np = np.stack(frames[:num_frames], axis=0)  # (T, H, W, 3)
    frames_t = torch.from_numpy(frames_np).permute(3, 0, 1, 2).float()  # (3, T, H, W)
    frames_t = frames_t / 255.0

    frames_t = F.interpolate(
        frames_t.unsqueeze(0),
        size=(frames_t.shape[1], height, width),
        mode="trilinear",
        align_corners=False,
    ).squeeze(0)

    frames_t = frames_t * 2.0 - 1.0
    return frames_t.unsqueeze(0)  # (1, 3, T, H, W)


def encode_video_cogvideo(
    vae,
    pixel_frames: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Encode pixel frames [B, C, T, H, W] in [-1,1] to latents.

    CogVideoX VAE: temporal 4x, spatial 8x compression.
    Returns [B, 16, T//4, H//8, W//8].
    """
    vae = vae.to(device)
    with torch.no_grad():
        latents = vae.encode(
            pixel_frames.to(device, vae.dtype)
        ).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents


def decode_latents_cogvideo(
    vae,
    latents: torch.Tensor,
    device: str = "cuda",
) -> torch.Tensor:
    """Decode latents back to pixel frames [B, C, T, H, W] in [-1,1]."""
    vae = vae.to(device)
    latents_scaled = latents / vae.config.scaling_factor
    with torch.no_grad():
        video = vae.decode(latents_scaled.to(device, vae.dtype)).sample
    return video


# ============================================================================
# Text encoding
# ============================================================================

def encode_prompt_cogvideo(
    tokenizer,
    text_encoder,
    prompt: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    max_length: int = 226,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode text prompt using T5 encoder for CogVideoX.

    Returns (prompt_embeds, attention_mask).
    """
    inputs = tokenizer(
        [prompt],
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        embeds = text_encoder(input_ids, attention_mask=attention_mask)[0]

    return embeds.to(dtype), attention_mask


# ============================================================================
# Flow-matching loss
# ============================================================================

def compute_flow_matching_loss_cogvideo(
    transformer,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    sigma_min: float = 0.001,
    sigma_max: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Compute rectified flow loss for CogVideoX transformer.

    CogVideoXTransformer3DModel forward signature:
        transformer(hidden_states, encoder_hidden_states, timestep, ...)
    """
    B = latents.shape[0]

    sigma = torch.rand(B, device=device) * (sigma_max - sigma_min) + sigma_min
    sigma_expanded = sigma.view(B, 1, 1, 1, 1)

    noise = torch.randn_like(latents)
    noisy_latents = (1.0 - sigma_expanded) * latents + sigma_expanded * noise

    timestep = (sigma * 1000).long()

    pred = transformer(
        hidden_states=noisy_latents.to(dtype),
        encoder_hidden_states=prompt_embeds,
        timestep=timestep,
        return_dict=False,
    )[0]

    target = (noise - latents).to(torch.float32)
    pred = pred.to(torch.float32)

    return F.mse_loss(pred, target)


# ============================================================================
# Delta-A wrapper
# ============================================================================

class DeltaAWrapperCogVideo(nn.Module):
    """Delta-A wrapper for CogVideoX: adds learnable vector to timestep embedding.

    CogVideoX uses time_embed_dim=512 for its timestep embedding.
    The timestep goes through a TimestepEmbedding module that produces
    the modulation vector used by all transformer blocks.
    """

    def __init__(self, transformer: nn.Module, time_embed_dim: int = 512):
        super().__init__()
        self.transformer = transformer
        for p in self.transformer.parameters():
            p.requires_grad = False

        self.delta = nn.Parameter(torch.zeros(time_embed_dim))
        self._hook = None

    @property
    def config(self):
        return self.transformer.config

    def apply_hook(self):
        """Hook the time_embedding module to inject delta."""
        delta = self.delta
        target_module = None
        if hasattr(self.transformer, "time_embedding"):
            target_module = self.transformer.time_embedding
        elif hasattr(self.transformer, "time_embed"):
            target_module = self.transformer.time_embed

        if target_module is not None:
            def _hook(_mod, _inp, output):
                return output + delta.unsqueeze(0).to(output.dtype)

            self._hook = target_module.register_forward_hook(_hook)

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def forward(self, hidden_states, encoder_hidden_states, timestep, **kwargs):
        self.apply_hook()
        try:
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                **kwargs,
            )
        finally:
            self.remove_hook()


# ============================================================================
# Video generation
# ============================================================================

def generate_video_cogvideo(
    pipe,
    image,
    prompt: str,
    num_frames: int = 49,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    seed: int = 42,
    device: str = "cuda",
) -> np.ndarray:
    """Generate video continuation using CogVideoX I2V pipeline.

    Parameters
    ----------
    pipe : CogVideoXImageToVideoPipeline
    image : PIL.Image — the conditioning frame
    prompt : text prompt
    num_frames : number of output frames (default 49)
    num_inference_steps : denoising steps
    guidance_scale : classifier-free guidance scale
    seed : random seed

    Returns
    -------
    np.ndarray [N, H, W, 3] in [0, 1].
    """
    generator = torch.Generator(device=device).manual_seed(seed)

    output = pipe(
        prompt=prompt,
        image=image,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).frames[0]

    frames = np.stack(
        [np.array(f) for f in output], axis=0
    ).astype(np.float32) / 255.0
    return frames


# ============================================================================
# Dataset helpers
# ============================================================================

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

    rng = np.random.RandomState(seed)
    if stratified:
        by_class = defaultdict(list)
        for entry in video_entries:
            by_class[entry["class_name"]].append(entry)

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


# ============================================================================
# Evaluation metrics
# ============================================================================

def compute_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute PSNR between pred and target tensors (both in [0, 1])."""
    mse = F.mse_loss(pred.float(), target.float())
    if mse == 0:
        return float("inf")
    return (10.0 * torch.log10(1.0 / mse)).item()


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
    gen_output : np.ndarray [N, H, W, 3] in [0, 1].
    video_path : path to source video.
    num_cond_frames : conditioning frames at start of gen_output.
    num_gen_frames : generated frames to evaluate.
    gen_start_frame : anchor frame index in source video.
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
    ], axis=0).astype(np.float32)
    gen_np = gen_frames[:n_compare].astype(np.float32)

    # PSNR (per-frame average)
    psnr_vals = []
    for i in range(n_compare):
        mse = np.mean((gen_np[i] - gt_np[i]) ** 2)
        if mse < 1e-10:
            psnr_vals.append(50.0)
        else:
            psnr_vals.append(float(10.0 * np.log10(1.0 / mse)))
    psnr = float(np.mean(psnr_vals))

    # SSIM (per-frame average)
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


# ============================================================================
# Video saving
# ============================================================================

def save_video_from_numpy(frames: np.ndarray, output_path: str, fps: int = 8):
    """Save video from numpy array [N, H, W, 3] in [0, 1].

    CogVideoX generates at 8 fps by default.
    """
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
