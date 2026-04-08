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

        eps_fresh = torch.randn_like(self.eps_optimized)
        eps_mixed = self._noise_interpolation(self.eps_optimized, eps_fresh)

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
                self.optimizer.zero_grad()
                total_loss.backward()
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
                self.optimizer.zero_grad()
                total_loss.backward()
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
                        help="Euler steps (default 10 to match SAVi-DNO PVDM)")
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Adam LR for noise optimization")
    parser.add_argument("--lam", type=float, default=0.0012,
                        help="Feature loss weight (PVDM-style pixel+feature loss)")
    parser.add_argument("--p", type=float, default=0.9,
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
    parser.add_argument("--gt-features-cache", type=str, default=None)
    parser.add_argument("--rollout-steps", type=int, default=1)
    args = parser.parse_args()

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
    if retain_set:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Derive geometry
    num_gen_frames = args.num_frames - args.num_cond_frames
    height = 480 if args.resolution == "480p" else 720
    width = 832 if args.resolution == "480p" else 1280
    vae_t_factor = 4

    # Resolve loss mode: --latent-loss and --pixel-loss are mutually exclusive.
    # Default to pixel+feature loss (PVDM-style) when neither is specified.
    use_latent_loss = args.latent_loss and not args.pixel_loss
    loss_mode = "latent (Vista-style)" if use_latent_loss else "pixel+feature (PVDM-style, no CFG)"

    print("=" * 70)
    print("SAVi-DNO with LongCat Backbone")
    print("=" * 70)
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
    print("  No-optimize  : %s" % args.no_optimize)
    print("  Grad ckpt    : %s" % (not args.no_gradient_checkpointing))
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
    )

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

    method_name = "longcat_baseline" if args.no_optimize else "savi_dno_longcat"
    print("\nProcessing %d videos (%s)..." % (len(video_list), method_name))

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

    for idx, entry in enumerate(tqdm(video_list, desc=method_name)):
        video_name = entry.get("video_name", entry.get("filename", ""))
        video_filename = entry.get("filename", video_name)
        video_path = os.path.join(args.data_dir, "videos", video_filename)

        if not os.path.exists(video_path):
            results.append({"video": video_name, "success": False, "error": "not_found"})
            continue

        try:
            caption = entry.get("caption", entry.get("prompt", ""))
            savi.reset()
            t_start = time.time()

            # Encode text prompt once per video
            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder,
                prompt=caption, device=device, dtype=torch.bfloat16,
            )

            # Load conditioning frames [1, C, T, H, W] in [-1, 1]
            cond_start = args.gen_start_frame - args.num_cond_frames
            pixel_cond = load_video_frames(
                video_path, args.num_cond_frames,
                height=height, width=width,
                start_frame=max(0, cond_start),
            ).to(device, torch.bfloat16)

            # Load GT frames [1, C, T_gen, H, W] in [-1, 1]
            pixel_gt = load_video_frames(
                video_path, num_gen_frames,
                height=height, width=width,
                start_frame=args.gen_start_frame,
            ).to(device, torch.bfloat16)
            # Convert GT to [0, 1] for loss computation
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

            # --- Run SAVi-DNO or baseline ---
            if args.no_optimize:
                pred_pixels = savi.predict_no_optimize(
                    cond_latents, target_shape, prompt_embeds, prompt_mask,
                )
                loss_val = None
            else:
                pred_pixels, loss_val = savi.predict_and_optimize(
                    cond_latents, gt_01, target_shape,
                    prompt_embeds, prompt_mask,
                )

            elapsed = time.time() - t_start

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

            # Save video if in retain set
            video_stem = Path(video_name).stem
            if video_stem in retain_set:
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

            # Free memory
            del pred_pixels, pixel_cond, pixel_gt, gt_01, cond_latents
            del prompt_embeds, prompt_mask
            torch_gc()

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"video": video_name, "success": False, "error": str(e)})

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
        "backbone": "longcat",
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
        "results": results,
    }
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
    print("  Results: %s" % str(output_dir / "summary.json"))
    print("=" * 70)


if __name__ == "__main__":
    main()
