#!/usr/bin/env python3
"""Shared utilities for Open-Sora v2.0 TTA backbone experiments.

Handles model loading, video encoding/decoding, flow-matching loss computation,
patchify/unpatchify, and the Delta-A wrapper for the MMDiT architecture.
"""

import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup – add Open-Sora v2.0 to PYTHONPATH
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]  # LongCat-Video-Experiment/

_OPENSORA_CANDIDATES = [
    _REPO_ROOT / ".." / "Open-Sora-2.0",
    Path("/scratch/wc3013/longcat-video-tta/../Open-Sora-2.0"),
    Path(os.environ.get("OPENSORA_ROOT", "")),
]
for _candidate in _OPENSORA_CANDIDATES:
    if _candidate.exists():
        sys.path.insert(0, str(_candidate.resolve()))
        break


# ============================================================================
# Model loading helpers
# ============================================================================

def load_opensora_components(
    config_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> Dict[str, object]:
    """Load all Open-Sora v2.0 model components from an inference config.

    The *config_path* should point to one of the inference .py configs under
    ``configs/diffusion/inference/`` (e.g. ``256px.py``).

    Returns a dict with keys: dit, vae, t5, clip, config.
    """
    from opensora.utils.config import read_config
    from opensora.utils.sampling import prepare_models

    cfg = read_config(config_path)

    # Override AE_SPATIAL_COMPRESSION env var if config specifies it
    if cfg.get("ae_spatial_compression", None) is not None:
        os.environ["AE_SPATIAL_COMPRESSION"] = str(cfg.ae_spatial_compression)

    dit, vae, t5, clip, optional_models = prepare_models(
        cfg, device=device, dtype=dtype, offload_model=False,
    )

    return {
        "dit": dit,
        "vae": vae,
        "t5": t5,
        "clip": clip,
        "optional_models": optional_models,
        "config": cfg,
    }


# ============================================================================
# Text encoding helpers
# ============================================================================

def encode_prompt_opensora(
    t5,
    clip,
    prompt: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_img_tokens: int = 0,
) -> Dict[str, torch.Tensor]:
    """Encode a text prompt using the Open-Sora T5 + CLIP embedders.

    Returns dict with keys: txt, txt_ids, y_vec.
    """
    with torch.no_grad():
        txt = t5([prompt], added_tokens=num_img_tokens, seq_align=1)
        y_vec = clip([prompt])

    bs = 1
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    return {
        "txt": txt.to(device, dtype),
        "txt_ids": txt_ids.to(device, dtype),
        "y_vec": y_vec.to(device, dtype),
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

    Pads by repeating the last frame when the source video is too short.
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


def encode_video_opensora(
    vae,
    pixel_frames: torch.Tensor,
) -> torch.Tensor:
    """Encode pixel frames [B, C, T, H, W] in [-1,1] to latents.

    The VAE internally applies scale_factor / shift_factor normalisation.
    """
    with torch.no_grad():
        latents = vae.encode(pixel_frames.to(next(vae.parameters()).dtype))
    return latents


def decode_latents_opensora(
    vae,
    latents: torch.Tensor,
) -> torch.Tensor:
    """Decode latents to pixel frames [B, C, T, H, W].

    The VAE internally reverses the scale_factor / shift_factor normalisation.
    Output is roughly in [-1, 1].
    """
    with torch.no_grad():
        video = vae.decode(latents.to(next(vae.parameters()).dtype))
    return video


# ============================================================================
# Patchify / unpatchify helpers
# ============================================================================

def patchify_latents(
    latents: torch.Tensor,
    patch_size: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert latents [B, C, T, H, W] to patchified tokens + positional IDs.

    Returns:
        img : [B, T*Hp*Wp, C*p*p]
        img_ids : [B, T*Hp*Wp, 3]  (t, h, w coordinates)
    """
    from einops import rearrange, repeat

    B, C, T, H, W = latents.shape
    Hp = H // patch_size
    Wp = W // patch_size

    img = rearrange(
        latents,
        "b c t (h ph) (w pw) -> b (t h w) (c ph pw)",
        ph=patch_size, pw=patch_size,
    )

    img_ids = torch.zeros(T, Hp, Wp, 3)
    img_ids[..., 0] = img_ids[..., 0] + torch.arange(T)[:, None, None]
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(Hp)[None, :, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(Wp)[None, None, :]
    img_ids = repeat(img_ids, "t h w c -> b (t h w) c", b=B)

    return img, img_ids.to(latents.device, latents.dtype)


def unpatchify_tokens(
    tokens: torch.Tensor,
    T: int,
    H: int,
    W: int,
    patch_size: int = 2,
) -> torch.Tensor:
    """Reverse patchification: [B, N, C*p*p] -> [B, C, T, H, W]."""
    from einops import rearrange

    D = int(os.environ.get("AE_SPATIAL_COMPRESSION", 16))
    return rearrange(
        tokens,
        "b (t h w) (c ph pw) -> b c t (h ph) (w pw)",
        h=math.ceil(H / D),
        w=math.ceil(W / D),
        t=T,
        ph=patch_size,
        pw=patch_size,
    )


# ============================================================================
# Flow-matching loss for Open-Sora v2.0
# ============================================================================

def compute_flow_matching_loss_opensora(
    dit: nn.Module,
    latents: torch.Tensor,
    text_inputs: Dict[str, torch.Tensor],
    sigma_min: float = 0.001,
    sigma_max: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    guidance: float = 7.5,
    patch_size: int = 2,
    forward_fn=None,
) -> torch.Tensor:
    """Flow-matching loss for Open-Sora's MMDiT.

    Rectified flow: x_t = (1-t)*x_0 + t*eps, target = eps - x_0.
    The model predicts velocity v = eps - x_0.
    """
    B = latents.shape[0]

    sigma = torch.rand(B, device=device, dtype=torch.float32) * (sigma_max - sigma_min) + sigma_min
    sigma_expanded = sigma.view(B, 1, 1, 1, 1)

    noise = torch.randn_like(latents)
    noisy_latents = (1.0 - sigma_expanded) * latents + sigma_expanded * noise

    img, img_ids = patchify_latents(noisy_latents, patch_size=patch_size)
    img = img.to(dtype)

    txt = text_inputs["txt"].to(dtype)
    txt_ids = text_inputs["txt_ids"].to(device, dtype)
    y_vec = text_inputs["y_vec"].to(dtype)

    guidance_vec = torch.full((B,), guidance, device=device, dtype=dtype)

    fwd_kwargs = dict(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        timesteps=sigma.to(dtype),
        y_vec=y_vec,
        guidance=guidance_vec,
    )

    if forward_fn is not None:
        pred = forward_fn(**fwd_kwargs)
    else:
        pred = dit(**fwd_kwargs)

    pred_latents = unpatchify_tokens(
        pred,
        T=latents.shape[2],
        H=latents.shape[3] * patch_size,
        W=latents.shape[4] * patch_size,
        patch_size=patch_size,
    )

    target = (noise - latents).to(torch.float32)
    pred_latents = pred_latents.to(torch.float32)

    return F.mse_loss(pred_latents, target)


# ============================================================================
# Conditioning-aware flow-matching loss
# ============================================================================

def build_cond_embed(
    masks: torch.Tensor,
    masked_ref: torch.Tensor,
    patch_size: int = 2,
) -> torch.Tensor:
    """Build the conditioning tensor expected by MMDiT's cond_embed mode.

    Concatenates masks and masked_ref along channel dim, then patchifies.

    masks     : [B, 1, T, H, W]   (binary)
    masked_ref: [B, C, T, H, W]   (latent-space reference)
    Returns   : [B, N, (1+C)*p*p]
    """
    from einops import rearrange

    cond = torch.cat((masks, masked_ref), dim=1)
    cond = rearrange(
        cond,
        "b c t (h ph) (w pw) -> b (t h w) (c ph pw)",
        ph=patch_size, pw=patch_size,
    )
    return cond


def compute_flow_matching_loss_conditioned_opensora(
    dit: nn.Module,
    cond_latents: torch.Tensor,
    target_latents: torch.Tensor,
    text_inputs: Dict[str, torch.Tensor],
    sigma_min: float = 0.001,
    sigma_max: float = 1.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    guidance: float = 7.5,
    patch_size: int = 2,
    forward_fn=None,
) -> torch.Tensor:
    """Conditioning-aware flow-matching loss for Open-Sora v2.0.

    Mimics the inference-time v2v/i2v conditioning: the reference frames are
    masked into the ``cond`` input, while noise is only applied to the target
    portion of the latent volume.
    """
    B, C, T_cond, H, W = cond_latents.shape
    T_target = target_latents.shape[2]
    T_total = T_cond + T_target

    sigma = torch.rand(B, device=device, dtype=torch.float32) * (sigma_max - sigma_min) + sigma_min
    sigma_expanded = sigma.view(B, 1, 1, 1, 1)

    noise = torch.randn_like(target_latents)
    noisy_target = (1.0 - sigma_expanded) * target_latents + sigma_expanded * noise

    full_latents = torch.cat([cond_latents, noisy_target], dim=2)

    masks = torch.zeros(B, 1, T_total, H, W, device=device, dtype=dtype)
    masks[:, :, :T_cond] = 1.0
    masked_ref = torch.zeros(B, C, T_total, H, W, device=device, dtype=dtype)
    masked_ref[:, :, :T_cond] = cond_latents.to(dtype)

    cond = build_cond_embed(masks, masked_ref, patch_size=patch_size)

    img, img_ids = patchify_latents(full_latents, patch_size=patch_size)
    img = img.to(dtype)

    txt = text_inputs["txt"].to(dtype)
    txt_ids = text_inputs["txt_ids"].to(device, dtype)
    y_vec = text_inputs["y_vec"].to(dtype)
    guidance_vec = torch.full((B,), guidance, device=device, dtype=dtype)

    fwd_kwargs = dict(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        timesteps=sigma.to(dtype),
        y_vec=y_vec,
        guidance=guidance_vec,
        cond=cond,
    )

    if forward_fn is not None:
        pred = forward_fn(**fwd_kwargs)
    else:
        pred = dit(**fwd_kwargs)

    pred_latents = unpatchify_tokens(
        pred,
        T=T_total,
        H=H * patch_size,
        W=W * patch_size,
        patch_size=patch_size,
    )

    pred_target = pred_latents[:, :, T_cond:].to(torch.float32)
    velocity_target = (noise - target_latents).to(torch.float32)

    return F.mse_loss(pred_target, velocity_target)


# ============================================================================
# Delta-A wrapper for Open-Sora v2.0
# ============================================================================

class DeltaAWrapperOpenSora(nn.Module):
    """Wraps MMDiTModel to inject a learnable delta into the timestep embedding.

    In Open-Sora v2.0 the timestep embedding path is:
        vec = time_in(timestep_embedding(t, 256)) + vector_in(y_vec)
    We add delta to ``vec`` after ``time_in`` via a forward hook, so that the
    same delta is seen by all double/single stream blocks via their AdaLN
    modulation.
    """

    def __init__(self, dit: nn.Module, hidden_size: int = None):
        super().__init__()
        self.dit = dit
        for p in self.dit.parameters():
            p.requires_grad = False

        if hidden_size is None:
            hidden_size = dit.hidden_size
        self.delta = nn.Parameter(torch.zeros(hidden_size))
        self._hook = None

    @property
    def config(self):
        return self.dit.config

    @property
    def patch_size(self):
        return self.dit.patch_size

    @property
    def hidden_size(self):
        return self.dit.hidden_size

    # ------------------------------------------------------------------
    # Hook-based injection
    # ------------------------------------------------------------------

    def apply_hook(self):
        """Install hook on time_in to add delta to timestep embedding."""
        delta = self.delta

        def _hook(_mod, _inp, output):
            return output + delta.unsqueeze(0).to(output.dtype)

        self._hook = self.dit.time_in.register_forward_hook(_hook)

    def remove_hook(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None

    def forward(self, **kwargs):
        """Forward with delta injected via hook. Safe for both train and eval."""
        self.apply_hook()
        try:
            return self.dit(**kwargs)
        finally:
            self.remove_hook()


# ============================================================================
# Dataset helpers
# ============================================================================

def load_ucf101_video_list(
    data_dir: str,
    max_videos: int = 100,
    seed: int = 42,
    stratified: bool = True,
) -> List[Dict]:
    """Load a list of video entries from a UCF101-style dataset directory."""
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
        from collections import defaultdict
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
    """PSNR between pred and target (both in [0, 1])."""
    mse = F.mse_loss(pred.float(), target.float())
    if mse == 0:
        return float("inf")
    return (10.0 * torch.log10(1.0 / mse)).item()


def compute_ssim_single(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute SSIM for a single frame pair [1, C, H, W] in [0, 1]."""
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
    gen_frames: np.ndarray,
    video_path: str,
    num_cond_frames: int,
    num_gen_frames: int,
    gen_start_frame: int,
    device: str = "cuda",
) -> Dict[str, float]:
    """Compute PSNR, SSIM, LPIPS between generated and ground truth frames.

    gen_frames : np.ndarray [N, H, W, 3] in [0, 1]
    """
    from PIL import Image
    import av

    gen = gen_frames[num_cond_frames:num_cond_frames + num_gen_frames]
    out_h, out_w = gen.shape[1], gen.shape[2]

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

    n_compare = min(len(gen), len(gt_pil))
    if n_compare == 0:
        return {"psnr": float("nan"), "ssim": float("nan"), "lpips": float("nan")}

    gt_np = np.stack([
        np.array(img.resize((out_w, out_h), Image.LANCZOS)) / 255.0
        for img in gt_pil[:n_compare]
    ], axis=0).astype(np.float32)
    gen_np = gen[:n_compare].astype(np.float32)

    psnr_vals = []
    for i in range(n_compare):
        mse = np.mean((gen_np[i] - gt_np[i]) ** 2)
        psnr_vals.append(50.0 if mse < 1e-10 else float(10.0 * np.log10(1.0 / mse)))
    psnr = float(np.mean(psnr_vals))

    ssim_vals = []
    for i in range(n_compare):
        p = torch.from_numpy(gen_np[i]).permute(2, 0, 1).unsqueeze(0).float()
        g = torch.from_numpy(gt_np[i]).permute(2, 0, 1).unsqueeze(0).float()
        ssim_vals.append(compute_ssim_single(p, g))
    ssim = float(np.mean(ssim_vals))

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
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_checkpoint(checkpoint_path: str) -> Optional[dict]:
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return None


def save_checkpoint(checkpoint: dict, checkpoint_path: str):
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)


# ============================================================================
# GPU memory helpers
# ============================================================================

def torch_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
