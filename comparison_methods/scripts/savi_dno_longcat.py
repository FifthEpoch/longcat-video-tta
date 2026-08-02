#!/usr/bin/env python3
"""
SAVi-DNO with LongCat backbone for fair comparison with LongCat TTA methods.

Implements Algorithm 1 from arXiv:2511.18255 (SAVi-DNO) using LongCat-Video
as the denoising backbone instead of PVDM. This enables an apples-to-apples
comparison where only the TTA strategy differs (noise optimization vs
LoRA/full fine-tuning).

Key differences from PVDM-based SAVi-DNO:
  - Flow matching (Euler) instead of DDPM/DDIM
  - DiT instead of UNet
  - AutoencoderKLWan instead of ViT autoencoder
  - 480x832 resolution, 14 cond + 14 gen frames
  - Conditioning via temporal concatenation + per-token timesteps

Fair comparison protocol (default, leakage-free):
  The sequence-adaptive noise is optimized ONLY on an observed history
  segment (predict [gen_start-num_gen, gen_start) from the frames before it),
  then that adapted noise seeds the sampler to predict the true UNSEEN future
  [gen_start, gen_start+num_gen). The scored future frames never enter
  optimization -> apples-to-apples with AdaSteer / LoRA-TTA, which adapt on
  pre-gen_start context only.

  --oracle-leak reverts to optimizing the noise directly against the scored
  future frames (the previous behaviour). That is an ORACLE UPPER BOUND, not a
  fair baseline, and must be labelled as such if ever reported.
"""

import sys
import os
import csv
import gc
import json
import math
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt_fn
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

# ---------------------------------------------------------------------------
# Path setup — add LongCat-Video to PYTHONPATH
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_LONGCAT_DIR = _REPO_ROOT / "LongCat-Video"
sys.path.insert(0, str(_LONGCAT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from delta_experiment.scripts.common import (
    load_longcat_components,
    encode_video,
    decode_latents,
    normalize_latents,
    denormalize_latents,
    encode_prompt,
    load_video_frames,
    _get_model_config,
    _extract_vbench_score,
)

# ---------------------------------------------------------------------------
# Metric helpers (reused from savi_dno.py to stay self-contained)
# ---------------------------------------------------------------------------

_I3D_HF_REPO = "kiwhansong/DFoT"
_I3D_HF_FILE = "metrics_models/i3d_torchscript.pt"
_I3D_FEATURE_DIM = 400
_FID_FEATURE_DIM = 2048
_MIN_I3D_FRAMES = 9
_COV_EPS = 1e-6


def load_feature_model(device):
    """Load ResNet3D-18 pretrained on Kinetics for feature loss."""
    from torchvision.models.video import r3d_18
    try:
        from torchvision.models.video import R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    except (ImportError, TypeError):
        model = r3d_18(pretrained=True)
    model.fc = nn.Identity()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_lpips_model(device):
    try:
        import lpips
        model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        return model
    except ImportError:
        print("WARNING: lpips not installed, LPIPS will be NaN")
        return None


def compute_lpips(lpips_model, pred_np, gt_np, device):
    """pred_np, gt_np: [T, C, H, W] in [0,1]. Returns mean LPIPS."""
    if lpips_model is None:
        return float("nan")
    vals = []
    with torch.no_grad():
        for t in range(pred_np.shape[0]):
            p = torch.from_numpy(pred_np[t:t + 1]).float().to(device) * 2 - 1
            g = torch.from_numpy(gt_np[t:t + 1]).float().to(device) * 2 - 1
            vals.append(lpips_model(p, g).item())
    return float(np.mean(vals))


def compute_metrics(pred_np, gt_np):
    """pred_np, gt_np: [T, C, H, W] in [0,1]."""
    T = pred_np.shape[0]
    psnrs, ssims = [], []
    for t in range(T):
        p = pred_np[t].transpose(1, 2, 0)
        g = gt_np[t].transpose(1, 2, 0)
        psnrs.append(skimage_psnr(g, p, data_range=1.0))
        ssims.append(skimage_ssim(g, p, data_range=1.0, channel_axis=2))
    return float(np.mean(psnrs)), float(np.mean(ssims))


def load_i3d_model(device):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=_I3D_HF_REPO, filename=_I3D_HF_FILE)
    model = torch.jit.load(path, map_location=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_inception_model(device):
    from torchvision.models import inception_v3, Inception_V3_Weights
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = nn.Identity()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def pad_for_i3d(x):
    T = x.shape[1]
    if T < _MIN_I3D_FRAMES:
        pad = (10 - T) // 2
        x = torch.cat([
            x[:, 0:1].expand(-1, pad, -1, -1, -1).clone(),
            x,
            x[:, -1:].expand(-1, pad, -1, -1, -1).clone(),
        ], dim=1)
    return x


def frames_to_i3d_tensor(frames_np, size=224):
    from torchvision.transforms import functional as TF
    from PIL import Image
    tensors = []
    for i in range(frames_np.shape[0]):
        if frames_np.shape[1] == 3:
            arr = (np.clip(frames_np[i].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        else:
            arr = (np.clip(frames_np[i], 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = TF.resize(img, size, interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.center_crop(img, size)
        tensors.append(TF.to_tensor(img))
    return torch.stack(tensors, dim=0).unsqueeze(0)


def i3d_features(model, clip, device):
    clip = pad_for_i3d(clip.to(device))
    clip = torch.clamp(2.0 * clip - 1.0, -1.0, 1.0)
    clip = clip.permute(0, 2, 1, 3, 4).contiguous()
    with torch.no_grad():
        feats = model(clip, rescale=False, resize=True, return_features=True)
    return feats.cpu().to(torch.float64).numpy().squeeze(0)


def inception_features(model, frames_np, device):
    from torchvision.transforms import functional as TF
    from PIL import Image
    feats_list = []
    with torch.no_grad():
        for i in range(frames_np.shape[0]):
            arr = (np.clip(frames_np[i].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(arr)
            img = TF.resize(img, 299, interpolation=TF.InterpolationMode.BILINEAR)
            img = TF.center_crop(img, 299)
            t = TF.normalize(
                TF.to_tensor(img),
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ).unsqueeze(0).to(device)
            f = model(t).cpu().to(torch.float64).numpy()
            feats_list.append(f)
    return np.concatenate(feats_list, axis=0)


def compute_frechet_distance(sum_a, cov_a, n_a, sum_b, cov_b, n_b, eps=_COV_EPS):
    from scipy.linalg import sqrtm
    mu_a = sum_a / n_a
    mu_b = sum_b / n_b
    sigma_a = cov_a / n_a - np.outer(mu_a, mu_a)
    sigma_b = cov_b / n_b - np.outer(mu_b, mu_b)
    sigma_a += eps * np.eye(sigma_a.shape[0])
    sigma_b += eps * np.eye(sigma_b.shape[0])
    diff = mu_a - mu_b
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean))


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ============================================================================
# In-distribution noise regularizers for published noise-optimization methods
# ============================================================================
#
# The two published, peer-reviewed noise-optimization methods we add on top of
# this differentiable LongCat sampler differ from SAVi-DNO only in *how they
# keep the optimized initial noise in-distribution* (SAVi-DNO uses fresh-noise
# interpolation, param ``p``; these use an explicit regularizer term):
#
#   * dno              -> decorrelation regularizer.  Karunratanakul et al.,
#                         "Optimizing Diffusion Noise Can Serve As Universal
#                         Motion Priors", CVPR 2024 (arXiv:2312.11994).
#   * direct_noise_opt -> probability ("Gaussian-shell") regularizer.  Tang et
#                         al., "Inference-Time Alignment of Diffusion Models
#                         with Direct Noise Optimization", ICML 2025
#                         (arXiv:2405.18881).
#
# Both were designed to optimize the noise against a reward on the SAME sample
# being generated.  In our video-*prediction* setting there is no test-time
# reward for the unseen future, so we optimize the noise against an OBSERVED
# history segment and transfer it to the future exactly as the leakage-free
# SAVi-DNO protocol does (see `predict_and_optimize` / `generate_with_optimized_eps`).
# This is the only fair, deployable way to bring these reward-driven methods
# into prediction, and it keeps them apples-to-apples with SAVi-DNO / AdaSteer.


def decorrelation_loss(eps: torch.Tensor) -> torch.Tensor:
    """DNO decorrelation regularizer (Karunratanakul et al., CVPR 2024).

    Keeps the optimized noise close to white i.i.d. Gaussian by penalizing
    autocorrelation along the sequence axis.  DNO applies this along the
    motion-sequence (time) axis; the video analogue is the latent temporal
    axis ``T``.  We penalize the squared off-diagonal entries of the
    normalized cross-frame correlation matrix (zero = temporally white noise).
    """
    B, C, T, H, W = eps.shape
    if T < 2:
        return torch.zeros((), device=eps.device, dtype=torch.float32)
    x = eps.permute(0, 2, 1, 3, 4).reshape(B, T, -1).float()  # [B, T, M]
    x = x - x.mean(dim=1, keepdim=True)
    gram = torch.matmul(x, x.transpose(1, 2))                  # [B, T, T]
    diag = torch.diagonal(gram, dim1=1, dim2=2).clamp_min(1e-8)
    denom = torch.sqrt(diag.unsqueeze(2) * diag.unsqueeze(1))
    corr = gram / denom
    eye = torch.eye(T, device=eps.device).unsqueeze(0)
    off = corr * (1.0 - eye)
    return (off ** 2).sum() / (B * T * (T - 1))


def gaussian_shell_penalty(eps: torch.Tensor) -> torch.Tensor:
    """Direct Noise Optimization probability regularizer (Tang et al., ICML 2025).

    For isotropic Gaussian noise in ``d`` dimensions, ``||z||^2`` concentrates
    on the shell ``||z||^2 ~ d`` (the Gaussian typical set).  Penalizing
    deviation from this shell keeps the optimized noise in the support of the
    pretrained prior and prevents out-of-distribution reward hacking (the
    failure mode DNO-direct's probability regularization was introduced to
    fix).  Dimension-normalized so the scale is independent of latent size.
    """
    B = eps.shape[0]
    z = eps.reshape(B, -1).float()
    d = z.shape[1]
    sq_norm = (z ** 2).sum(dim=1)
    return ((sq_norm / d) - 1.0).pow(2).mean()


_REGULARIZERS = {
    "none": None,
    "decorr": decorrelation_loss,
    "gaussian_shell": gaussian_shell_penalty,
}

# method name -> (regularizer key, use noise interpolation, default reg weight,
#                 output method_name)
_NOISE_OPT_METHODS = {
    "savi_dno": ("none", True, 0.0, "savi_dno_longcat"),
    "dno": ("decorr", False, 1.0, "dno_longcat"),
    "direct_noise_opt": ("gaussian_shell", False, 0.01, "direct_noise_opt_longcat"),
}


# ============================================================================
# 2-GPU model parallelism for DiT
# ============================================================================

def split_dit_across_gpus(dit, split_block=24, device0="cuda:0", device1="cuda:1"):
    """Split the DiT transformer blocks across 2 GPUs.

    Moves blocks[split_block:] and final_layer to device1, keeps embedders
    and blocks[:split_block] on device0.  Monkey-patches dit.forward to
    insert device transfers at the split boundary so gradients flow through
    both GPUs seamlessly.
    """
    import types

    for i in range(split_block, len(dit.blocks)):
        dit.blocks[i] = dit.blocks[i].to(device1)
    dit.final_layer = dit.final_layer.to(device1)

    print(f"[2-GPU] Blocks 0-{split_block - 1} on {device0}, "
          f"blocks {split_block}-{len(dit.blocks) - 1} + final_layer on {device1}")

    _split = split_block
    _dev0 = device0
    _dev1 = device1

    def _forward_2gpu(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        **kwargs,
    ):
        # Pin inputs to device0 — during checkpoint recomputation the
        # default CUDA device may have shifted to device1, which would
        # cause tensors created with device="cuda" to land on the wrong GPU.
        hidden_states = hidden_states.to(_dev0)
        timestep = timestep.to(_dev0)
        encoder_hidden_states = encoder_hidden_states.to(_dev0)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(_dev0)

        B, _, T, H, W = hidden_states.shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)

        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = self.x_embedder(hidden_states)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            t = self.t_embedder(
                timestep.float().flatten(), dtype=torch.float32
            ).reshape(B, N_t, -1)

        encoder_hidden_states = self.y_embedder(encoder_hidden_states)

        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = (
                encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            )
            encoder_attention_mask = (
                (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = (
                encoder_hidden_states.squeeze(1)
                .masked_select(encoder_attention_mask.unsqueeze(-1) != 0)
                .view(1, -1, hidden_states.shape[-1])
            )
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(
                1, -1, hidden_states.shape[-1]
            )

        latent_shape = (N_t, N_h, N_w)

        for i, block in enumerate(self.blocks):
            if i == _split:
                hidden_states = hidden_states.to(_dev1)
                t = t.to(_dev1)
                encoder_hidden_states = encoder_hidden_states.to(_dev1)

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(
                    block, hidden_states, encoder_hidden_states, t,
                    y_seqlens, latent_shape, num_cond_latents,
                    False, None, False,
                )
            else:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, t,
                    y_seqlens, latent_shape, num_cond_latents,
                )

        hidden_states = self.final_layer(hidden_states, t, latent_shape)
        hidden_states = self.unpatchify(hidden_states, N_t, N_h, N_w)
        hidden_states = hidden_states.to(torch.float32)

        return hidden_states.to(_dev0)

    dit.forward = types.MethodType(_forward_2gpu, dit)
    return dit


def enable_single_gpu_block_checkpointing(dit, device="cuda:0"):
    """Monkey-patch the DiT forward to checkpoint EACH transformer block on one GPU.

    The stock LongCat DiT forward runs its 48-block loop without checkpointing,
    so a single differentiable pass materialises every block's activations at
    once (~139 GiB) and OOMs one H200 by ~200 MiB (confirmed: memory-in-use is
    the same with or without the T5 encoder offloaded, i.e. it is the DiT, not
    T5). Wrapping each block in torch.utils.checkpoint caps the live activation
    at ~one block (~tens of GiB), so it fits one H200 with large headroom, at
    the cost of one recompute per block in backward. Mirrors the embedder /
    masking / final-layer logic of ``split_dit_across_gpus`` but keeps
    everything on one device. Pair with SAViDNO.gradient_checkpointing=False so
    we do single-level (per-block) checkpointing, not nested step+block.
    """
    import types
    _dev = device

    def _forward_1gpu_ckpt(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        **kwargs,
    ):
        hidden_states = hidden_states.to(_dev)
        timestep = timestep.to(_dev)
        encoder_hidden_states = encoder_hidden_states.to(_dev)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(_dev)

        B, _, T, H, W = hidden_states.shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        if len(timestep.shape) == 1:
            timestep = timestep.unsqueeze(1).expand(-1, N_t)

        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.to(dtype)
        timestep = timestep.to(dtype)
        encoder_hidden_states = encoder_hidden_states.to(dtype)

        hidden_states = self.x_embedder(hidden_states)

        with torch.amp.autocast("cuda", dtype=torch.float32):
            t = self.t_embedder(
                timestep.float().flatten(), dtype=torch.float32
            ).reshape(B, N_t, -1)

        encoder_hidden_states = self.y_embedder(encoder_hidden_states)

        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            encoder_hidden_states = (
                encoder_hidden_states * encoder_attention_mask[:, None, :, None]
            )
            encoder_attention_mask = (
                (encoder_attention_mask * 0 + 1).to(encoder_attention_mask.dtype)
            )

        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.squeeze(1).squeeze(1)
            encoder_hidden_states = (
                encoder_hidden_states.squeeze(1)
                .masked_select(encoder_attention_mask.unsqueeze(-1) != 0)
                .view(1, -1, hidden_states.shape[-1])
            )
            y_seqlens = encoder_attention_mask.sum(dim=1).tolist()
        else:
            y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
            encoder_hidden_states = encoder_hidden_states.squeeze(1).view(
                1, -1, hidden_states.shape[-1]
            )

        latent_shape = (N_t, N_h, N_w)

        for block in self.blocks:
            if torch.is_grad_enabled():
                hidden_states = ckpt_fn(
                    block, hidden_states, encoder_hidden_states, t,
                    y_seqlens, latent_shape, num_cond_latents,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(
                    hidden_states, encoder_hidden_states, t,
                    y_seqlens, latent_shape, num_cond_latents,
                )

        hidden_states = self.final_layer(hidden_states, t, latent_shape)
        hidden_states = self.unpatchify(hidden_states, N_t, N_h, N_w)
        hidden_states = hidden_states.to(torch.float32)
        return hidden_states

    dit.forward = types.MethodType(_forward_1gpu_ckpt, dit)
    return dit


# ============================================================================
# SAViDNO_LongCat — core class
# ============================================================================

class SAViDNO_LongCat:
    """SAVi-DNO noise optimization using LongCat-Video as backbone.

    Flow-matching version: the initial noise at t=1.0 is optimized so that
    the Euler-sampled prediction at t=0.0 matches the ground-truth future
    frames.  Gradients flow through the entire denoising chain to the
    optimizable noise tensor.
    """

    def __init__(
        self,
        dit,
        vae,
        scheduler,
        tokenizer,
        text_encoder,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        num_inference_steps: int = 10,
        guidance_scale: float = 4.0,
        lr: float = 0.01,
        lam: float = 0.0012,
        p: float = 0.9,
        feature_model: Optional[nn.Module] = None,
        gradient_checkpointing: bool = True,
        latent_loss: bool = False,
        max_grad_norm: float = 1.0,
        regularizer: str = "none",
        reg_weight: float = 0.0,
        noise_interp: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.dit = dit
        self.vae = vae
        self.scheduler = scheduler
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder

        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.lr = lr
        self.lam = lam
        self.p = p
        self.gradient_checkpointing = gradient_checkpointing
        self.latent_loss = latent_loss
        self.max_grad_norm = max_grad_norm

        # In-distribution noise regularizer for published noise-opt methods.
        # regularizer="none" + noise_interp=True reproduces SAVi-DNO exactly.
        if regularizer not in _REGULARIZERS:
            raise ValueError(
                "regularizer must be one of %s (got %r)"
                % (list(_REGULARIZERS), regularizer)
            )
        self.regularizer = regularizer
        self.reg_fn = _REGULARIZERS[regularizer]
        self.reg_weight = float(reg_weight)
        self.noise_interp = bool(noise_interp)

        self.feature_model = None
        if feature_model is not None and not latent_loss:
            self.feature_model = feature_model
            for param in self.feature_model.parameters():
                param.requires_grad = False

        self.eps_optimized = None
        self.optimizer = None

        # Cache null prompt embeddings for CFG
        self._null_embeds = None
        self._null_mask = None

    def _get_null_embeds(self):
        """Compute and cache empty-string embeddings for CFG."""
        if self._null_embeds is None:
            self._null_embeds, self._null_mask = encode_prompt(
                self.tokenizer, self.text_encoder,
                prompt="",
                device=self.device, dtype=self.dtype,
            )
        return self._null_embeds, self._null_mask

    def _build_sigmas(self):
        """Build flow-matching sigma schedule from ~1.0 to 0.0.

        Returns scheduler.sigmas (N+1 values including terminal 0.0),
        NOT scheduler.timesteps (which are sigma*1000).
        """
        self.scheduler.set_timesteps(self.num_inference_steps, device=self.device)
        return self.scheduler.sigmas

    def _dit_forward_step(
        self,
        x_t: torch.Tensor,
        cond_latents: torch.Tensor,
        t_value: float,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Single DiT forward pass with conditioning.

        Concatenates [cond_clean, x_t] along temporal dim, builds per-token
        timesteps (cond=0, target=t*1000), and runs the DiT.

        Returns the velocity prediction for the TARGET portion only.
        """
        cfg = _get_model_config(self.dit)
        patch_t = cfg.patch_size[0]

        B, C, T_target, H_lat, W_lat = x_t.shape
        T_cond = cond_latents.shape[2]
        T_total = T_cond + T_target
        N_cond = T_cond // patch_t
        N_target = T_target // patch_t
        N_total = N_cond + N_target

        hidden_states = torch.cat([cond_latents, x_t], dim=2).to(self.dtype)

        timestep = torch.zeros(B, N_total, device=self.device, dtype=self.dtype)
        timestep[:, N_cond:] = t_value * 1000.0

        pred = self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=prompt_embeds,
            encoder_attention_mask=prompt_mask,
            num_cond_latents=N_cond,
        )

        return pred[:, :, T_cond:]

    def _dit_forward_step_cfg(
        self,
        x_t: torch.Tensor,
        cond_latents: torch.Tensor,
        t_value: float,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """DiT forward with classifier-free guidance."""
        null_embeds, null_mask = self._get_null_embeds()

        v_cond = self._dit_forward_step(
            x_t, cond_latents, t_value, prompt_embeds, prompt_mask,
        )
        v_uncond = self._dit_forward_step(
            x_t, cond_latents, t_value, null_embeds, null_mask,
        )

        return v_uncond + self.guidance_scale * (v_cond - v_uncond)

    def _flow_euler_sample_differentiable(
        self,
        cond_latents: torch.Tensor,
        eps_init: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        use_cfg: bool = False,
    ) -> torch.Tensor:
        """Euler flow-matching sampling with gradient flow to eps_init.

        Starts from pure noise (eps_init at t=1.0) and steps toward clean
        signal (t=0.0) using the velocity prediction from the DiT.

        PVDM-style (default): single DiT pass per step, no CFG.
        Matches SAVi-DNO paper: "For PVDM, we do not use guidance during
        inference."
        """
        sigmas = self._build_sigmas()
        x_t = eps_init

        step_fn = self._dit_forward_step_cfg if use_cfg else self._dit_forward_step

        for i in range(len(sigmas) - 1):
            t_curr = sigmas[i].item()
            t_next = sigmas[i + 1].item()
            dt = t_next - t_curr

            if self.gradient_checkpointing:
                v_pred = ckpt_fn(
                    step_fn,
                    x_t, cond_latents, t_curr, prompt_embeds, prompt_mask,
                    use_reentrant=False,
                )
            else:
                v_pred = step_fn(
                    x_t, cond_latents, t_curr, prompt_embeds, prompt_mask,
                )

            x_t = x_t + dt * v_pred.to(x_t.dtype)

        return x_t

    def _noise_interpolation(self, eps_opt, eps_fresh):
        """h(p, eps_s, eps) = (p*eps_s + (1-p)*eps) / sqrt(p^2 + (1-p)^2)"""
        pp = self.p
        norm = math.sqrt(pp ** 2 + (1 - pp) ** 2)
        return (pp * eps_opt + (1 - pp) * eps_fresh) / norm

    def encode(self, pixel_frames: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames [B, C, T, H, W] in [-1,1] to normalized latents."""
        with torch.no_grad():
            return encode_video(self.vae, pixel_frames, normalize=True)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode normalized latents to pixel frames [B, C, T, H, W] in [0,1].

        Uses torch.enable_grad so gradients flow through the decode for the
        pixel-space loss.
        """
        z = denormalize_latents(self.vae, latents)
        with torch.enable_grad():
            video = self.vae.decode(z.to(self.vae.dtype), return_dict=False)[0]
            video = (video + 1.0) / 2.0
        return video.clamp(0, 1)

    def predict_and_optimize(
        self,
        cond_latents: torch.Tensor,
        gt_frames: torch.Tensor,
        target_latent_shape: tuple,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ):
        """One step of SAVi-DNO: predict via flow matching, optimize noise.

        Parameters
        ----------
        cond_latents : [B, C, T_cond, H, W] — clean conditioning latents
        gt_frames    : [B, C, T_gen, H, W] — ground-truth future frames in [0,1]
        target_latent_shape : shape for the optimizable noise tensor
        prompt_embeds, prompt_mask : encoded text prompt

        Returns
        -------
        pred_pixels : [B, C, T_gen, H, W] in [0,1], detached
        loss_val : float or None
        """
        if self.eps_optimized is None:
            self.eps_optimized = torch.randn(
                target_latent_shape, device=self.device,
                dtype=torch.float32, requires_grad=True,
            )
            self.optimizer = torch.optim.Adam([self.eps_optimized], lr=self.lr)

        # SAVi-DNO regularizes via fresh-noise interpolation (noise_interp=True);
        # DNO / Direct-Noise-Optimization instead optimize the raw noise with an
        # explicit in-distribution regularizer (noise_interp=False).
        if self.noise_interp:
            eps_fresh = torch.randn_like(self.eps_optimized)
            eps_mixed = self._noise_interpolation(self.eps_optimized, eps_fresh)
        else:
            eps_mixed = self.eps_optimized

        # Regularizer is always computed on the raw optimization variable
        # (self.eps_optimized), matching both papers' formulations.
        reg_loss = None
        if self.reg_fn is not None and self.reg_weight > 0.0:
            reg_loss = self.reg_fn(self.eps_optimized)

        z_pred = self._flow_euler_sample_differentiable(
            cond_latents, eps_mixed, prompt_embeds, prompt_mask,
        )

        loss_val = None
        if gt_frames is not None:
            if self.latent_loss:
                # Vista-style: L1 loss in latent space (Eq. 6 from SAVi-DNO).
                # Avoids decoding during optimization, saving significant memory
                # for large models. Matches the paper's methodology for Vista.
                gt_for_vae = (gt_frames.to(self.device) * 2.0 - 1.0).to(self.dtype)
                with torch.no_grad():
                    z_gt = encode_video(self.vae, gt_for_vae, normalize=True)

                total_loss = F.l1_loss(z_pred, z_gt.to(z_pred.dtype), reduction="mean")
                if reg_loss is not None:
                    total_loss = total_loss + self.reg_weight * reg_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_([self.eps_optimized], self.max_grad_norm)
                self.optimizer.step()
                loss_val = total_loss.item()

                self.dit.zero_grad(set_to_none=True)

                with torch.no_grad():
                    pred_pixels = decode_latents(
                        self.vae, z_pred.detach(), denorm=True,
                    )
            else:
                # PVDM-style: pixel + feature loss (Eq. 3+4 from SAVi-DNO).
                pred_pixels = self.decode(z_pred)

                gt = gt_frames.to(self.device)
                T_pred = pred_pixels.shape[2]
                T_gt = gt.shape[2]
                if T_pred < T_gt:
                    gt = gt[:, :, :T_pred]
                elif T_gt < T_pred:
                    pred_pixels = pred_pixels[:, :, :T_gt]

                loss_pixel = F.l1_loss(pred_pixels, gt, reduction="mean")

                loss_feature = torch.tensor(0.0, device=self.device)
                if self.feature_model is not None and self.lam > 0:
                    pred_3d = pred_pixels.float()
                    gt_3d = gt.float()
                    feat_pred = self.feature_model(pred_3d)
                    feat_gt = self.feature_model(gt_3d)
                    loss_feature = F.mse_loss(feat_pred, feat_gt, reduction="mean")

                total_loss = loss_pixel + self.lam * loss_feature
                if reg_loss is not None:
                    total_loss = total_loss + self.reg_weight * reg_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_([self.eps_optimized], self.max_grad_norm)
                self.optimizer.step()
                loss_val = total_loss.item()

                self.dit.zero_grad(set_to_none=True)
                self.vae.zero_grad(set_to_none=True)
                if self.feature_model is not None:
                    self.feature_model.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                pred_pixels = decode_latents(
                    self.vae, z_pred.detach(), denorm=True,
                )

        return pred_pixels.detach(), loss_val

    def predict_no_optimize(
        self,
        cond_latents: torch.Tensor,
        target_latent_shape: tuple,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ):
        """Baseline: generate without noise optimization (random noise)."""
        with torch.no_grad():
            eps = torch.randn(target_latent_shape, device=self.device, dtype=torch.float32)
            z_pred = self._flow_euler_sample_differentiable(
                cond_latents, eps, prompt_embeds, prompt_mask,
            )
            pred_pixels = decode_latents(self.vae, z_pred, denorm=True)
        return pred_pixels

    def generate_with_optimized_eps(
        self,
        cond_latents: torch.Tensor,
        target_latent_shape: tuple,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
    ):
        """Generate the (unseen) future segment using the noise that was
        adapted on OBSERVED history frames.

        This is the leakage-free "apply" step of the fair SAVi-DNO protocol:
        the sequence-adaptive noise ``self.eps_optimized`` was fit against
        already-observed frames (see ``predict_and_optimize`` called on the
        history segment), and here we simply seed the sampler with it to
        predict the true future.  No gradient / no ground-truth of the scored
        segment is touched.  Falls back to fresh Gaussian noise if adaptation
        never ran.
        """
        with torch.no_grad():
            if self.eps_optimized is not None:
                eps = self.eps_optimized.detach().to(self.device, torch.float32)
                if tuple(eps.shape) != tuple(target_latent_shape):
                    # History segment had a different geometry than the target
                    # window; cannot transfer noise 1:1 -> reseed.
                    eps = torch.randn(
                        target_latent_shape, device=self.device, dtype=torch.float32,
                    )
            else:
                eps = torch.randn(
                    target_latent_shape, device=self.device, dtype=torch.float32,
                )
            z_pred = self._flow_euler_sample_differentiable(
                cond_latents, eps, prompt_embeds, prompt_mask,
            )
            pred_pixels = decode_latents(self.vae, z_pred, denorm=True)
        return pred_pixels

    def reset(self):
        self.eps_optimized = None
        self.optimizer = None


# ============================================================================
# Evaluation driver
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SAVi-DNO with LongCat backbone evaluation",
    )
    parser.add_argument("--checkpoint-dir", required=True,
                        help="LongCat-Video checkpoint directory")
    parser.add_argument("--data-dir", required=True,
                        help="Dataset directory with videos/ and metadata.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--num-inference-steps", type=int, default=10,
                        help="Euler steps (10 matches SAVi-DNO paper, needs 2 GPUs)")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs (2 for model-parallel DiT)")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Adam LR for noise optimization")
    parser.add_argument("--lam", type=float, default=0.0012,
                        help="Feature loss weight (PVDM-style pixel+feature loss)")
    parser.add_argument("--p", type=float, default=0.7,
                        help="Noise interpolation parameter")
    parser.add_argument("--no-optimize", action="store_true",
                        help="LongCat baseline without noise optimization")
    parser.add_argument("--latent-loss", action="store_true",
                        help="Use latent-space L1 loss (Vista style, for OOM)")
    parser.add_argument("--pixel-loss", action="store_true",
                        help="Use pixel + feature loss (default, PVDM style)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-cond-frames", type=int, default=14)
    parser.add_argument("--num-frames", type=int, default=28,
                        help="Total frames (cond + gen)")
    parser.add_argument("--gen-start-frame", type=int, default=48,
                        help="Video frame index where generation starts")
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--save-only-list", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-videos", action="store_true",
                        help="Save EVERY generated mp4 (not just --save-only-list ids). "
                             "Needed to bank videos for the figure bank / future "
                             "experiments and to enable VBench backfill.")
    parser.add_argument("--compute-vbench", action="store_true",
                        help="After generation, run VBench++ (7 dims) on the saved "
                             "mp4s so SAVi-DNO has the same metric set as AdaSteer. "
                             "Requires saved videos (--save-videos or --save-only-list).")
    parser.add_argument("--gt-features-cache", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="Disable checkpoint/resume. By default the run "
                             "persists FVD/FID sufficient-statistics, running "
                             "metric totals, per-video results and a next-index "
                             "cursor to <output_dir>/resume_state.npz after every "
                             "video, and on restart skips already-processed videos "
                             "and resumes the accumulators exactly. This makes the "
                             "job safe to requeue after a low-GPU-util / preemption "
                             "cancellation without corrupting the pooled FVD/FID.")
    parser.add_argument("--rollout-steps", type=int, default=10,
                        help="Number of noise optimization (Adam) steps taken on the "
                             "OBSERVED history segment before predicting the future")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm for eps_optimized (0=disable)")
    parser.add_argument("--oracle-leak", action="store_true",
                        help="DEBUG/UPPER-BOUND ONLY: optimize the noise directly against "
                             "the scored future frames (ground-truth leakage). This is NOT a "
                             "fair baseline; it reproduces the old behaviour as an oracle "
                             "upper bound. Leave OFF for the paper comparison.")
    parser.add_argument("--method", type=str, default="savi_dno",
                        choices=list(_NOISE_OPT_METHODS),
                        help="Which published noise-optimization method to run. All share "
                             "the same differentiable LongCat sampler + leakage-free "
                             "prediction protocol and differ only in the in-distribution "
                             "regularizer: 'savi_dno' (fresh-noise interpolation, arXiv:2511.18255, "
                             "default/unchanged), 'dno' (decorrelation reg, Karunratanakul CVPR 2024), "
                             "'direct_noise_opt' (Gaussian-shell probability reg, Tang ICML 2025).")
    parser.add_argument("--reg-weight", type=float, default=-1.0,
                        help="Weight of the in-distribution regularizer for --method dno / "
                             "direct_noise_opt. <0 (default) uses the method's built-in default "
                             "(dno=1.0, direct_noise_opt=0.01). Ignored for --method savi_dno.")
    args = parser.parse_args()

    # Resolve the noise-opt method into (regularizer, noise_interp, reg_weight).
    _reg_key, _noise_interp, _default_reg_w, _method_out_name = _NOISE_OPT_METHODS[args.method]
    reg_weight = _default_reg_w if args.reg_weight < 0 else args.reg_weight

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Selective video retention
    retain_set = set()
    if args.save_only_list:
        with open(args.save_only_list) as f:
            retain_set = set(json.load(f).get("all", []))
        print("[Retain] Will save %d videos" % len(retain_set))
    save_dir = Path(args.save_dir) if args.save_dir else output_dir / "videos"
    if retain_set or args.save_videos:
        save_dir.mkdir(parents=True, exist_ok=True)
        if args.save_videos:
            print("[Save] Saving ALL generated videos -> %s" % save_dir)

    # Derive geometry
    num_gen_frames = args.num_frames - args.num_cond_frames
    height = 480 if args.resolution == "480p" else 720
    width = 832 if args.resolution == "480p" else 1280
    vae_t_factor = 4

    # Resolve loss mode: --latent-loss and --pixel-loss are mutually exclusive.
    # Default to latent loss (shorter gradient path, more stable on large DiT).
    use_latent_loss = not args.pixel_loss
    loss_mode = "latent (Vista-style)" if use_latent_loss else "pixel+feature (PVDM-style, no CFG)"

    print("=" * 70)
    print("Noise-Optimization TTA with LongCat Backbone")
    print("=" * 70)
    print("  Method       : %s -> %s" % (args.method, _method_out_name))
    print("  Regularizer  : %s (weight=%g, noise_interp=%s)" % (
        _reg_key, reg_weight, _noise_interp))
    print("  Checkpoint   : %s" % args.checkpoint_dir)
    print("  Resolution   : %dx%d" % (height, width))
    print("  Cond frames  : %d" % args.num_cond_frames)
    print("  Gen frames   : %d" % num_gen_frames)
    print("  Euler steps  : %d" % args.num_inference_steps)
    print("  Guidance     : %.1f%s" % (args.guidance_scale,
          " (CFG off — PVDM-style)" if not use_latent_loss else " (CFG on — Vista-style)"))
    print("  DNO LR       : %g" % args.lr)
    print("  Loss mode    : %s" % loss_mode)
    print("  Feature lam  : %g%s" % (args.lam, " (unused — latent loss)" if use_latent_loss else ""))
    print("  Noise interp : %.2f" % args.p)
    print("  Rollout steps: %d" % args.rollout_steps)
    print("  Max grad norm: %g" % args.max_grad_norm)
    print("  No-optimize  : %s" % args.no_optimize)
    print("  Grad ckpt    : %s" % (not args.no_gradient_checkpointing))
    print("  Num GPUs     : %d" % args.num_gpus)
    print("=" * 70)

    # --- Load LongCat model ---
    print("\nLoading LongCat-Video model...")
    components = load_longcat_components(
        args.checkpoint_dir, device=device, dtype=torch.bfloat16,
    )
    dit = components["dit"]
    vae = components["vae"]
    scheduler = components["scheduler"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    # Freeze all model weights
    for param in dit.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    for param in text_encoder.parameters():
        param.requires_grad = False
    dit.eval()
    vae.eval()
    text_encoder.eval()

    # Memory strategy by GPU count:
    #   2 GPUs -> split the DiT blocks across devices (halves per-GPU activation)
    #   1 GPU  -> per-block gradient checkpointing so a single differentiable DiT
    #             pass keeps only ~one block live instead of all 48 (~139 GiB ->
    #             OOMs one H200 by ~200 MiB). Confirmed the DiT (not T5) is the
    #             cost: memory-in-use was identical with T5 offloaded. Block
    #             checkpointing fits one H200 with headroom (extra recompute in
    #             backward -> slower, but ~100% util and no OOM).
    single_gpu = (args.num_gpus < 2 or torch.cuda.device_count() < 2)
    if not single_gpu:
        split_dit_across_gpus(dit, split_block=24,
                              device0="cuda:0", device1="cuda:1")
    else:
        enable_single_gpu_block_checkpointing(dit, device=str(device))
        print("[mem] single-GPU: enabled per-block gradient checkpointing in the "
              "DiT forward (caps live activation at ~1 block) to fit one H200.")

    # Feature model only needed for pixel-loss mode
    feature_model = None
    if not args.no_optimize and not use_latent_loss and args.lam > 0:
        print("Loading ResNet3D feature extractor...")
        feature_model = load_feature_model(device)

    savi = SAViDNO_LongCat(
        dit=dit, vae=vae, scheduler=scheduler,
        tokenizer=tokenizer, text_encoder=text_encoder,
        device=device, dtype=torch.bfloat16,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        lr=args.lr, lam=args.lam, p=args.p,
        feature_model=feature_model,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        latent_loss=use_latent_loss,
        max_grad_norm=args.max_grad_norm,
        regularizer=_reg_key,
        reg_weight=reg_weight,
        noise_interp=_noise_interp,
    )

    # Single GPU already checkpoints at the block level inside the DiT forward,
    # so turn OFF the Euler-step-level checkpoint to avoid nested (step x block)
    # double recompute. On 2 GPUs we keep step-level checkpointing as before.
    if single_gpu:
        savi.gradient_checkpointing = False

    # --- Load metric models ---
    print("Loading metric models (LPIPS, I3D, InceptionV3)...")
    lpips_model = load_lpips_model(device)
    i3d_model = load_i3d_model(device)
    incep_model = load_inception_model(device)

    # --- Load dataset ---
    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    with open(metadata_path) as f:
        video_list = list(csv.DictReader(f))
    if args.max_videos > 0:
        video_list = video_list[:args.max_videos]

    if args.no_optimize:
        method_name = "longcat_baseline"
    elif args.oracle_leak:
        method_name = "%s_oracle" % _method_out_name
    else:
        method_name = _method_out_name
    leakage_free = not args.oracle_leak
    protocol = ("no_optimize" if args.no_optimize
                else ("oracle_leak_UPPER_BOUND" if args.oracle_leak
                      else "fair_streaming_observed_history"))
    print("\nProcessing %d videos (%s)..." % (len(video_list), method_name))
    print("  Protocol     : %s (leakage_free=%s)" % (protocol, leakage_free))
    if args.oracle_leak:
        print("  *** WARNING: --oracle-leak optimizes noise against the SCORED "
              "future. Oracle upper bound only, NOT a fair baseline. ***")

    # --- FVD/FID accumulators ---
    d_fvd = _I3D_FEATURE_DIM
    gen_fvd_sum = np.zeros(d_fvd, dtype=np.float64)
    gen_fvd_cov = np.zeros((d_fvd, d_fvd), dtype=np.float64)
    ref_fvd_sum = np.zeros(d_fvd, dtype=np.float64)
    ref_fvd_cov = np.zeros((d_fvd, d_fvd), dtype=np.float64)
    fvd_count = 0
    ref_fvd_count = 0

    d_fid = _FID_FEATURE_DIM
    gen_fid_sum = np.zeros(d_fid, dtype=np.float64)
    gen_fid_cov = np.zeros((d_fid, d_fid), dtype=np.float64)
    ref_fid_sum = np.zeros(d_fid, dtype=np.float64)
    ref_fid_cov = np.zeros((d_fid, d_fid), dtype=np.float64)
    fid_gen_frames = 0
    fid_ref_frames = 0

    gt_cached = False
    if args.gt_features_cache:
        print("[FVD/FID] Loading GT cache from %s" % args.gt_features_cache)
        cache = np.load(args.gt_features_cache, allow_pickle=True)
        ref_fvd_sum = cache["ref_fvd_sum"].astype(np.float64)
        ref_fvd_cov = cache["ref_fvd_cov"].astype(np.float64)
        ref_fvd_count = int(cache["ref_fvd_count"])
        ref_fid_sum = cache["ref_fid_sum"].astype(np.float64)
        ref_fid_cov = cache["ref_fid_cov"].astype(np.float64)
        fid_ref_frames = int(cache["ref_fid_count"])
        gt_cached = True
        print("[FVD/FID] GT cache: %d ref videos, %d FID frames" %
              (ref_fvd_count, fid_ref_frames))

    results = []
    total_psnr = total_ssim = total_lpips = 0.0
    n_ok = 0
    gpu_peak_per_video = []

    # ------------------------------------------------------------------
    # Checkpoint / resume.  We persist the pooled FVD/FID sufficient
    # statistics (sum + outer-product covariance), running PSNR/SSIM/LPIPS
    # totals and per-video results after EVERY video, and restore them exactly
    # on restart so a job cancelled by the low-GPU-util policy / preemption can
    # be requeued and finish the pooled metrics without double-counting.
    #
    # Resume is SUCCESS-based: we skip only videos already recorded as
    # successful, and re-attempt failures (OOM/transient). Failure entries are
    # dropped on reload so they line up with the accumulators (which only ever
    # counted successes) and are retried cleanly. This is what lets you simply
    # resubmit after the memory fix -- the earlier all-OOM run recorded 0
    # successes, so nothing is skipped.
    # ------------------------------------------------------------------
    resume_state_path = output_dir / "resume_state.npz"
    done_ids: set = set()

    def _save_resume_state():
        tmp = str(resume_state_path) + ".tmp.npz"
        np.savez(
            tmp,
            gen_fvd_sum=gen_fvd_sum, gen_fvd_cov=gen_fvd_cov,
            gen_fid_sum=gen_fid_sum, gen_fid_cov=gen_fid_cov,
            ref_fvd_sum=ref_fvd_sum, ref_fvd_cov=ref_fvd_cov,
            ref_fid_sum=ref_fid_sum, ref_fid_cov=ref_fid_cov,
            counts=np.array([len(done_ids), fvd_count, fid_gen_frames,
                             ref_fvd_count, fid_ref_frames, n_ok],
                            dtype=np.int64),
            totals=np.array([total_psnr, total_ssim, total_lpips],
                            dtype=np.float64),
            results=np.array(results, dtype=object),
            meta=np.array([str(method_name), str(int(args.max_videos)),
                           str(int(args.seed))], dtype=object),
        )
        os.replace(tmp, str(resume_state_path))

    if (not args.no_resume) and resume_state_path.exists():
        try:
            st = np.load(str(resume_state_path), allow_pickle=True)
            meta = list(st["meta"])
            same_run = (meta[0] == str(method_name)
                        and meta[1] == str(int(args.max_videos)))
            if not same_run:
                print("  [resume] state at %s is for a DIFFERENT run "
                      "(method/max_videos mismatch: %s) -> ignoring, "
                      "starting fresh." % (resume_state_path, meta))
            else:
                counts = st["counts"]
                fvd_count = int(counts[1]); fid_gen_frames = int(counts[2])
                n_ok = int(counts[5])
                gen_fvd_sum = st["gen_fvd_sum"].astype(np.float64)
                gen_fvd_cov = st["gen_fvd_cov"].astype(np.float64)
                gen_fid_sum = st["gen_fid_sum"].astype(np.float64)
                gen_fid_cov = st["gen_fid_cov"].astype(np.float64)
                tot = st["totals"]
                total_psnr = float(tot[0]); total_ssim = float(tot[1])
                total_lpips = float(tot[2])
                # Keep only SUCCESS entries; failures are dropped and retried.
                results = [r for r in list(st["results"]) if r.get("success")]
                done_ids = {r["video"] for r in results}
                # Only restore the reference accumulators when they are being
                # built online (no GT cache). With a GT cache the reference is
                # static and already loaded above -> do NOT overwrite it.
                if not gt_cached:
                    ref_fvd_sum = st["ref_fvd_sum"].astype(np.float64)
                    ref_fvd_cov = st["ref_fvd_cov"].astype(np.float64)
                    ref_fid_sum = st["ref_fid_sum"].astype(np.float64)
                    ref_fid_cov = st["ref_fid_cov"].astype(np.float64)
                    ref_fvd_count = int(counts[3]); fid_ref_frames = int(counts[4])
                print("  [resume] restored state from %s: %d videos already "
                      "succeeded (n_ok=%d, fvd_count=%d) -> skipping those, "
                      "retrying the rest."
                      % (resume_state_path, len(done_ids), n_ok, fvd_count))
        except Exception as _e:
            print("  [resume] failed to load %s (%s) -> starting fresh."
                  % (resume_state_path, _e))
            done_ids = set()

    for idx, entry in enumerate(tqdm(video_list, desc=method_name)):
        video_name = entry.get("video_name", entry.get("filename", ""))
        if video_name in done_ids:
            continue
        video_filename = entry.get("filename", video_name)
        video_path = os.path.join(args.data_dir, "videos", video_filename)

        if not os.path.exists(video_path):
            results.append({"video": video_name, "success": False, "error": "not_found"})
            if not args.no_resume:
                _save_resume_state()
            continue

        try:
            caption = entry.get("caption", entry.get("prompt", ""))
            savi.reset()
            t_start = time.time()

            # Encode text prompt once per video.
            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder,
                prompt=caption, device=device, dtype=torch.bfloat16,
            )

            # ----------------------------------------------------------------
            # Frame windows (leakage-free fair protocol):
            #   cond       = [gen_start - num_cond, gen_start)       real context
            #   target     = [gen_start, gen_start + num_gen)        UNSEEN future (scored)
            #   adapt_tgt  = [gen_start - num_gen,  gen_start)       OBSERVED (adapt on this)
            #   adapt_cond = [adapt_tgt - num_cond, adapt_tgt)       OBSERVED
            # SAVi-DNO fits the sequence-adaptive noise on the observed history
            # segment (adapt_cond -> adapt_tgt), then seeds the sampler with that
            # noise to predict the true future WITHOUT optimizing against it.
            # No future GT enters optimization -> apples-to-apples with AdaSteer.
            # ----------------------------------------------------------------
            cond_start = args.gen_start_frame - args.num_cond_frames
            pixel_cond = load_video_frames(
                video_path, args.num_cond_frames,
                height=height, width=width,
                start_frame=max(0, cond_start),
            ).to(device, torch.bfloat16)

            # True future frames [1, C, T_gen, H, W] in [-1, 1] -> [0,1] (SCORED ONLY)
            pixel_gt = load_video_frames(
                video_path, num_gen_frames,
                height=height, width=width,
                start_frame=args.gen_start_frame,
            ).to(device, torch.bfloat16)
            gt_01 = (pixel_gt + 1.0) / 2.0

            # Encode conditioning to latents
            cond_latents = savi.encode(pixel_cond)

            # Compute target latent shape
            T_gen_latent = 1 + (num_gen_frames - 1) // vae_t_factor
            target_shape = (
                1,
                cond_latents.shape[1],
                T_gen_latent,
                cond_latents.shape[3],
                cond_latents.shape[4],
            )

            adapt_cond_latents = None
            # --- Run SAVi-DNO or baseline ---
            if args.no_optimize:
                pred_pixels = savi.predict_no_optimize(
                    cond_latents, target_shape, prompt_embeds, prompt_mask,
                )
                loss_val = None
            elif args.oracle_leak:
                # UPPER BOUND ONLY — leaks the scored future into optimization.
                # Not a fair baseline; kept for a labelled oracle row.
                loss_val = None
                for _opt_step in range(args.rollout_steps):
                    pred_pixels, loss_val = savi.predict_and_optimize(
                        cond_latents, gt_01, target_shape,
                        prompt_embeds, prompt_mask,
                    )
            else:
                # FAIR leakage-free protocol.
                adapt_tgt_start = args.gen_start_frame - num_gen_frames
                adapt_cond_start = adapt_tgt_start - args.num_cond_frames
                loss_val = None
                if adapt_cond_start < 0:
                    # Not enough observed history to form an adaptation segment
                    # of matching geometry -> leakage-free no-optimization fallback.
                    if idx == 0:
                        print("  [warn] gen_start=%d too small for an observed "
                              "adaptation segment (need >= %d); using no-optimize "
                              "fallback." % (args.gen_start_frame,
                                             num_gen_frames + args.num_cond_frames))
                    pred_pixels = savi.predict_no_optimize(
                        cond_latents, target_shape, prompt_embeds, prompt_mask,
                    )
                else:
                    pixel_adapt_cond = load_video_frames(
                        video_path, args.num_cond_frames,
                        height=height, width=width,
                        start_frame=adapt_cond_start,
                    ).to(device, torch.bfloat16)
                    pixel_adapt_tgt = load_video_frames(
                        video_path, num_gen_frames,
                        height=height, width=width,
                        start_frame=adapt_tgt_start,
                    ).to(device, torch.bfloat16)
                    adapt_tgt_01 = (pixel_adapt_tgt + 1.0) / 2.0
                    adapt_cond_latents = savi.encode(pixel_adapt_cond)

                    # Adapt the noise on the OBSERVED history segment only.
                    for _opt_step in range(args.rollout_steps):
                        _, loss_val = savi.predict_and_optimize(
                            adapt_cond_latents, adapt_tgt_01, target_shape,
                            prompt_embeds, prompt_mask,
                        )

                    # Apply the sequence-adaptive noise to the UNSEEN future.
                    pred_pixels = savi.generate_with_optimized_eps(
                        cond_latents, target_shape, prompt_embeds, prompt_mask,
                    )
                    del pixel_adapt_cond, pixel_adapt_tgt, adapt_tgt_01

            elapsed = time.time() - t_start

            # Track GPU peak memory
            if torch.cuda.is_available():
                peaks = {}
                for gi in range(torch.cuda.device_count()):
                    peaks[gi] = torch.cuda.max_memory_allocated(gi) / (1024**3)
                    torch.cuda.reset_peak_memory_stats(gi)
                gpu_peak_per_video.append(peaks)
                if idx == 0:
                    total_per_gpu = {gi: torch.cuda.get_device_properties(gi).total_memory / (1024**3)
                                     for gi in range(torch.cuda.device_count())}
                    print("\n  [GPU mem] First video peaks: %s" %
                          ", ".join("GPU%d=%.1f/%.1f GiB" % (gi, peaks[gi], total_per_gpu[gi])
                                   for gi in sorted(peaks)))

            # Convert to numpy [T, C, H, W] for metrics
            pred_np = pred_pixels.squeeze(0).float().cpu().numpy()
            gt_np = gt_01.squeeze(0).float().cpu().numpy()

            # Trim to matching lengths
            n_compare = min(pred_np.shape[1], gt_np.shape[1])
            pred_np = pred_np[:, :n_compare]
            gt_np = gt_np[:, :n_compare]

            # Reshape: [C, T, H, W] -> [T, C, H, W]
            pred_np = pred_np.transpose(1, 0, 2, 3)
            gt_np = gt_np.transpose(1, 0, 2, 3)

            psnr, ssim = compute_metrics(pred_np, gt_np)
            lpips_val = compute_lpips(lpips_model, pred_np, gt_np, device)

            # FVD/FID accumulation
            gen_clip = frames_to_i3d_tensor(pred_np)
            gen_feat = i3d_features(i3d_model, gen_clip, device)
            gen_fvd_sum += gen_feat
            gen_fvd_cov += np.outer(gen_feat, gen_feat)
            fvd_count += 1

            gen_fid_feat = inception_features(incep_model, pred_np, device)
            gen_fid_sum += gen_fid_feat.sum(axis=0)
            gen_fid_cov += gen_fid_feat.T @ gen_fid_feat
            fid_gen_frames += gen_fid_feat.shape[0]

            if not gt_cached:
                ref_clip = frames_to_i3d_tensor(gt_np)
                ref_feat = i3d_features(i3d_model, ref_clip, device)
                ref_fvd_sum += ref_feat
                ref_fvd_cov += np.outer(ref_feat, ref_feat)
                ref_fvd_count += 1

                ref_fid_feat = inception_features(incep_model, gt_np, device)
                ref_fid_sum += ref_fid_feat.sum(axis=0)
                ref_fid_cov += ref_fid_feat.T @ ref_fid_feat
                fid_ref_frames += ref_fid_feat.shape[0]

            entry_result = {
                "video": video_name, "success": True,
                "psnr": psnr, "ssim": ssim, "lpips": lpips_val,
                "loss": loss_val, "time": elapsed,
            }
            results.append(entry_result)

            # Save video: everything when --save-videos, else only retain-set ids
            video_stem = Path(video_name).stem
            if args.save_videos or (video_stem in retain_set):
                import imageio
                frames_hwc = pred_np.transpose(0, 2, 3, 1)
                frames_uint8 = np.clip(frames_hwc * 255, 0, 255).astype(np.uint8)
                out_path = save_dir / ("%s_%s.mp4" % (video_stem, method_name))
                writer = imageio.get_writer(
                    str(out_path), fps=24, codec="libx264",
                    output_params=["-crf", "23"],
                )
                for frame in frames_uint8:
                    writer.append_data(frame)
                writer.close()

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_val if lpips_val == lpips_val else 0.0
            n_ok += 1
            done_ids.add(video_name)

            # Free memory
            del pred_pixels, pixel_cond, pixel_gt, gt_01, cond_latents
            del prompt_embeds, prompt_mask
            if adapt_cond_latents is not None:
                del adapt_cond_latents
            torch_gc()

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"video": video_name, "success": False, "error": str(e)})

        # Persist resume state after every video so a low-GPU-util / preemption
        # cancellation can be requeued without losing progress or corrupting the
        # pooled FVD/FID sufficient statistics.
        if not args.no_resume:
            _save_resume_state()

    # --- Compute FVD/FID ---
    fvd_val = None
    fid_val = None
    effective_ref = ref_fvd_count if gt_cached else fvd_count
    if fvd_count >= 2:
        fvd_val = compute_frechet_distance(
            gen_fvd_sum, gen_fvd_cov, fvd_count,
            ref_fvd_sum, ref_fvd_cov, effective_ref,
        )
        print("[FVD] %.4f (%d gen / %d ref)" % (fvd_val, fvd_count, effective_ref))
    if fid_gen_frames >= 2:
        fid_val = compute_frechet_distance(
            gen_fid_sum, gen_fid_cov, fid_gen_frames,
            ref_fid_sum, ref_fid_cov, fid_ref_frames,
        )
        print("[FID] %.4f (%d gen / %d ref frames)" % (
            fid_val, fid_gen_frames, fid_ref_frames))

    # --- Save summary ---
    summary = {
        "method": method_name,
        "noise_opt_method": args.method,
        "regularizer": _reg_key,
        "reg_weight": reg_weight,
        "noise_interp": _noise_interp,
        "backbone": "longcat",
        "protocol": protocol,
        "leakage_free": leakage_free,
        "num_videos": len(video_list),
        "num_successful": n_ok,
        "avg_psnr": total_psnr / max(n_ok, 1),
        "avg_ssim": total_ssim / max(n_ok, 1),
        "avg_lpips": total_lpips / max(n_ok, 1),
        "fvd": round(fvd_val, 6) if fvd_val is not None else None,
        "fvd_num_videos": fvd_count,
        "fvd_num_ref_videos": effective_ref,
        "fvd_gt_cached": gt_cached,
        "fid": round(fid_val, 6) if fid_val is not None else None,
        "fid_num_frames_gen": fid_gen_frames,
        "fid_num_frames_ref": fid_ref_frames,
        "resolution": "%dx%d" % (height, width),
        "num_cond_frames": args.num_cond_frames,
        "num_gen_frames": num_gen_frames,
        "gen_start_frame": args.gen_start_frame,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "lr": args.lr,
        "lam": args.lam,
        "p": args.p,
        "rollout_steps": args.rollout_steps,
        "num_gpus": args.num_gpus,
    }
    if gpu_peak_per_video:
        num_devs = max(len(p) for p in gpu_peak_per_video)
        gpu_mem = {}
        for gi in range(num_devs):
            dev_peaks = [p[gi] for p in gpu_peak_per_video if gi in p]
            total_gib = torch.cuda.get_device_properties(gi).total_memory / (1024**3)
            gpu_mem["gpu%d" % gi] = {
                "total_gib": round(total_gib, 1),
                "peak_mean_gib": round(sum(dev_peaks) / len(dev_peaks), 2),
                "peak_max_gib": round(max(dev_peaks), 2),
                "headroom_gib": round(total_gib - max(dev_peaks), 1),
            }
        summary["gpu_memory"] = gpu_mem
    summary["results"] = results

    # --- VBench++ (optional): run on the saved mp4s so SAVi-DNO carries the
    # same 7-dim scores as the AdaSteer runs. Purely post-hoc on saved videos;
    # does not touch the PSNR/SSIM/LPIPS/FVD/FID path above. ---
    if args.compute_vbench:
        mp4s = sorted(save_dir.glob("*.mp4")) if save_dir.is_dir() else []
        if not mp4s:
            print("[VBench] SKIPPED: no saved mp4s (need --save-videos or "
                  "--save-only-list).")
        else:
            print("\n[VBench++] Running on %d videos in %s ..." % (len(mp4s), save_dir))
            try:
                from vbench import VBench
                import vbench as _vbench_pkg
                _VBENCH_DIMS = [
                    "subject_consistency", "background_consistency",
                    "motion_smoothness", "dynamic_degree",
                    "aesthetic_quality", "imaging_quality",
                ]
                pkg_dir = os.path.dirname(_vbench_pkg.__file__)
                full_info_json = os.path.join(pkg_dir, "VBench_full_info.json")
                if not os.path.exists(full_info_json):
                    full_info_json = os.path.join(
                        os.path.dirname(pkg_dir), "vbench", "VBench_full_info.json")
                vbench_output = str(output_dir / "vbench_results")
                os.makedirs(vbench_output, exist_ok=True)
                vb = VBench(torch.device("cuda"), full_info_json, vbench_output)
                vbench_scores = {}
                for dim in _VBENCH_DIMS:
                    try:
                        print("  Evaluating %s..." % dim)
                        vb.evaluate(videos_path=str(save_dir), name="vbench_%s" % dim,
                                    dimension_list=[dim], mode="custom_input")
                        rf = os.path.join(vbench_output, "vbench_%s_eval_results.json" % dim)
                        if os.path.exists(rf):
                            with open(rf) as _f:
                                score = _extract_vbench_score(dim, json.load(_f))
                            if score is not None:
                                vbench_scores[dim] = score
                    except Exception as _ve:
                        print("    [VBench] %s failed: %s" % (dim, _ve))
                if vbench_scores:
                    summary["vbench"] = vbench_scores
                    summary["vbench_num_videos"] = len(mp4s)
                    print("  VBench dims: %s" % vbench_scores)
            except Exception as ve:
                import traceback
                traceback.print_exc()
                print("[VBench] FAILED (metrics above are unaffected): %s" % ve)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 70)
    print("%s Complete" % method_name.upper())
    print("=" * 70)
    print("  Videos: %d/%d" % (n_ok, len(video_list)))
    print("  Avg PSNR:  %.4f" % summary["avg_psnr"])
    print("  Avg SSIM:  %.4f" % summary["avg_ssim"])
    print("  Avg LPIPS: %.4f" % summary["avg_lpips"])
    if fvd_val is not None:
        print("  FVD:       %.4f" % fvd_val)
    if fid_val is not None:
        print("  FID:       %.4f" % fid_val)
    if "gpu_memory" in summary:
        for gname, ginfo in summary["gpu_memory"].items():
            print("  %s: peak %.1f / %.1f GiB (headroom %.1f GiB)" %
                  (gname, ginfo["peak_max_gib"], ginfo["total_gib"], ginfo["headroom_gib"]))
    print("  Results: %s" % str(output_dir / "summary.json"))
    print("=" * 70)


if __name__ == "__main__":
    main()
