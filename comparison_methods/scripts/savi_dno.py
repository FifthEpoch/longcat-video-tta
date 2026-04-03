#!/usr/bin/env python3
"""
SAVi-DNO: Sequence-Adaptive Video Prediction with Diffusion Noise Optimization.

Implements Algorithm 1 from arXiv:2511.18255 on top of PVDM.

The key idea: instead of fine-tuning model weights, optimize the initial
diffusion noise epsilon_s to minimize reconstruction loss against the
observed next frames. The optimized noise carries forward to the next
prediction step.

Paper hyperparameters for UCF-101 (10 DDIM steps):
  - Adam LR: 0.01
  - lambda (feature loss weight): 0.0012
  - p (noise interpolation): 0.9
  - DDIM eta: 0 (deterministic sampling for gradient flow)
"""

import sys
import os
import math
import time
import csv
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim


def make_beta_schedule(n_timestep, linear_start=1e-4, linear_end=2e-2):
    betas = (
        torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep,
                        dtype=torch.float64) ** 2
    )
    return betas


class SAViDNO:
    """SAVi-DNO noise optimization wrapper around PVDM."""

    def __init__(self, autoencoder, diffusion_model, ddpm_config,
                 device, ddim_steps=10, lr=0.01, lam=0.0012,
                 p=0.9, feature_model=None, w=0.0):
        self.device = device
        self.autoencoder = autoencoder.to(device).eval()
        self.ddim_steps = ddim_steps
        self.lr = lr
        self.lam = lam
        self.p = p
        self.w = w

        betas = make_beta_schedule(
            int(ddpm_config.timesteps),
            linear_start=float(ddpm_config.linear_start),
            linear_end=float(ddpm_config.linear_end),
        )
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).float().to(device)
        self.num_timesteps = int(ddpm_config.timesteps)

        self.diffusion_model = diffusion_model.to(device).eval()

        self.feature_model = None
        if feature_model is not None:
            self.feature_model = feature_model.to(device).eval()
            for p_param in self.feature_model.parameters():
                p_param.requires_grad = False

        self.eps_optimized = None
        self.optimizer = None

    def _get_ddim_timesteps(self):
        total = self.num_timesteps
        steps = self.ddim_steps
        times = torch.linspace(-1, total - 1, steps=steps + 1)
        times = list(reversed(times.int().tolist()))
        return list(zip(times[:-1], times[1:]))

    def _ddim_sample_differentiable(self, z_cond, eps_init):
        """Deterministic DDIM sampling (eta=0) with gradient flow."""
        batch = z_cond.shape[0]
        time_pairs = self._get_ddim_timesteps()
        img = eps_init

        for time_now, time_next in time_pairs:
            t = torch.full((batch,), time_now, device=self.device, dtype=torch.long)

            if z_cond is not None and self.w > 0:
                model_out = (1 + self.w) * self.diffusion_model(img, z_cond, t) \
                            - self.w * self.diffusion_model(img, z_cond * 0, t)
            else:
                model_out = self.diffusion_model(img, z_cond, t)

            alpha = self.alphas_cumprod[time_now]
            x_start = (img - (1 - alpha).sqrt() * model_out) / alpha.sqrt()
            x_start = x_start.clamp(-1., 1.)
            pred_noise = model_out

            if time_next < 0:
                img = x_start
                continue

            alpha_next = self.alphas_cumprod[time_next]
            c = (1 - alpha_next).sqrt()
            img = x_start * alpha_next.sqrt() + c * pred_noise

        return img

    def _noise_interpolation(self, eps_opt, eps_fresh):
        """h(p, eps_s, eps) = (p*eps_s + (1-p)*eps) / sqrt(p^2 + (1-p)^2)"""
        pp = self.p
        norm = math.sqrt(pp ** 2 + (1 - pp) ** 2)
        return (pp * eps_opt + (1 - pp) * eps_fresh) / norm

    def encode(self, frames):
        """Encode pixel frames [B, T, C, H, W] in [0,1] to latent."""
        with torch.no_grad():
            x = rearrange(frames * 2 - 1, 'b t c h w -> b c t h w')
            z = self.autoencoder.extract(x)
        return z

    def decode(self, z):
        """Decode latent to pixel frames in [0,1]."""
        with torch.enable_grad():
            x = self.autoencoder.decode_from_sample(z)
            x = (x.clamp(-1, 1) + 1) / 2
        return x

    def predict_and_optimize(self, z_cond, gt_frames_next, latent_shape):
        """One step of SAVi-DNO: predict, then optimize noise using GT."""
        if self.eps_optimized is None:
            self.eps_optimized = torch.randn(latent_shape, device=self.device,
                                              requires_grad=True)
            self.optimizer = torch.optim.Adam([self.eps_optimized], lr=self.lr)

        eps_fresh = torch.randn_like(self.eps_optimized)
        eps_mixed = self._noise_interpolation(self.eps_optimized, eps_fresh)

        z_pred = self._ddim_sample_differentiable(z_cond, eps_mixed)
        pred_frames = self.decode(z_pred)

        loss_val = None
        if gt_frames_next is not None:
            gt_flat = rearrange(gt_frames_next, 'b t c h w -> (b t) c h w').to(self.device)
            loss_pixel = F.l1_loss(pred_frames, gt_flat, reduction='mean')

            loss_feature = torch.tensor(0.0, device=self.device)
            if self.feature_model is not None and self.lam > 0:
                T = gt_frames_next.shape[1]
                pred_3d = rearrange(pred_frames, '(b t) c h w -> b c t h w', t=T)
                gt_3d = rearrange(gt_flat, '(b t) c h w -> b c t h w', t=T)
                feat_pred = self.feature_model(pred_3d)
                feat_gt = self.feature_model(gt_3d)
                loss_feature = F.mse_loss(feat_pred, feat_gt, reduction='mean')

            total_loss = loss_pixel + self.lam * loss_feature
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            loss_val = total_loss.item()

            self.diffusion_model.zero_grad(set_to_none=True)
            self.autoencoder.zero_grad(set_to_none=True)
            if self.feature_model is not None:
                self.feature_model.zero_grad(set_to_none=True)

        return pred_frames.detach(), loss_val

    def reset(self):
        self.eps_optimized = None
        self.optimizer = None


def load_pvdm_models(pvdm_dir, pretrained_dir, device):
    """Load PVDM autoencoder and diffusion model."""
    from models.autoencoder.autoencoder_vit import ViTAutoencoder
    from models.ddpm.unet import UNetModel, DiffusionWrapper
    from omegaconf import OmegaConf

    pvdm_dir = Path(pvdm_dir)
    ae_config = OmegaConf.load(pvdm_dir / "configs" / "autoencoder" / "base.yaml")
    diff_config = OmegaConf.load(pvdm_dir / "configs" / "latent-diffusion" / "base.yaml")

    autoencoder = ViTAutoencoder(
        ae_config.model.params.embed_dim,
        ae_config.model.params.ddconfig,
    )
    ae_ckpt = torch.load(str(Path(pretrained_dir) / "pvdm_ucf101_autoencoder.pth"),
                          map_location="cpu")
    autoencoder.load_state_dict(ae_ckpt)
    autoencoder = autoencoder.to(device).eval()

    unet = UNetModel(**diff_config.model.params.unet_config)
    diff_wrapper = DiffusionWrapper(unet)
    diff_ckpt = torch.load(str(Path(pretrained_dir) / "pvdm_ucf101_diffusion.pth"),
                            map_location="cpu")
    diff_wrapper.load_state_dict(diff_ckpt)
    diff_wrapper = diff_wrapper.to(device).eval()

    return autoencoder, diff_wrapper, ae_config, diff_config


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


def load_video_frames(video_path, num_frames, size=256):
    """Load video frames, center-crop and resize."""
    from PIL import Image
    import torchvision.transforms as T

    frames = []
    try:
        import imageio.v3 as iio
        reader = iio.imread(str(video_path), plugin="pyav")
    except Exception:
        import imageio
        reader = imageio.mimread(str(video_path), memtest=False)

    for i, frame in enumerate(reader):
        if len(frames) >= num_frames:
            break
        if isinstance(frame, np.ndarray):
            img = Image.fromarray(frame)
        else:
            img = frame
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top = (h - s) // 2
        img = img.crop((left, top, left + s, top + s))
        img = img.resize((size, size), Image.BILINEAR)
        frames.append(T.ToTensor()(img))

    while len(frames) < num_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames)


def compute_metrics(pred_np, gt_np):
    T = pred_np.shape[0]
    psnrs, ssims = [], []
    for t in range(T):
        p = pred_np[t].transpose(1, 2, 0)
        g = gt_np[t].transpose(1, 2, 0)
        psnrs.append(skimage_psnr(g, p, data_range=1.0))
        ssims.append(skimage_ssim(g, p, data_range=1.0, channel_axis=2))
    return float(np.mean(psnrs)), float(np.mean(ssims))


# ============================================================================
# LPIPS, FVD, FID computation
# ============================================================================

_I3D_HF_REPO = "kiwhansong/DFoT"
_I3D_HF_FILE = "metrics_models/i3d_torchscript.pt"
_I3D_FEATURE_DIM = 400
_FID_FEATURE_DIM = 2048
_MIN_I3D_FRAMES = 9
_COV_EPS = 1e-6


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
    """pred_np, gt_np: [(B*T), C, H, W] in [0,1]. Returns mean LPIPS."""
    if lpips_model is None:
        return float("nan")
    vals = []
    with torch.no_grad():
        for t in range(pred_np.shape[0]):
            p = torch.from_numpy(pred_np[t:t+1]).float().to(device) * 2 - 1
            g = torch.from_numpy(gt_np[t:t+1]).float().to(device) * 2 - 1
            vals.append(lpips_model(p, g).item())
    return float(np.mean(vals))


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
    """Symmetric first/last-frame padding to >= 9 frames."""
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
    """[T, C, H, W] float32 [0,1] -> [1, T, C, H, W] resized tensor."""
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
    """clip: [1, T, C, H, W] in [0,1] -> 400-dim feature."""
    clip = pad_for_i3d(clip.to(device))
    clip = torch.clamp(2.0 * clip - 1.0, -1.0, 1.0)
    clip = clip.permute(0, 2, 1, 3, 4).contiguous()
    with torch.no_grad():
        feats = model(clip, rescale=False, resize=True, return_features=True)
    return feats.cpu().to(torch.float64).numpy().squeeze(0)


def inception_features(model, frames_np, device):
    """frames_np: [T, C, H, W] in [0,1] -> [T, 2048] float64."""
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


def main():
    parser = argparse.ArgumentParser(description="SAVi-DNO Evaluation on UCF-101")
    parser.add_argument("--pvdm-dir", required=True)
    parser.add_argument("--pretrained-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--max-videos", type=int, default=500)
    parser.add_argument("--ddim-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=0.0012)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--no-optimize", action="store_true",
                        help="Run PVDM baseline without noise optimization")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sys.path.insert(0, args.pvdm_dir)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading PVDM models...")
    autoencoder, diff_model, ae_config, diff_config = load_pvdm_models(
        args.pvdm_dir, args.pretrained_dir, device)

    feature_model = None
    if not args.no_optimize and args.lam > 0:
        print("Loading ResNet3D feature extractor...")
        feature_model = load_feature_model(device)

    savi = SAViDNO(
        autoencoder=autoencoder,
        diffusion_model=diff_model,
        ddpm_config=diff_config.model.params,
        device=device,
        ddim_steps=args.ddim_steps,
        lr=args.lr,
        lam=args.lam if not args.no_optimize else 0.0,
        p=args.p,
        feature_model=feature_model,
        w=float(diff_config.model.params.w),
    )

    # Load metric models
    print("Loading LPIPS model...")
    lpips_model = load_lpips_model(device)
    print("Loading I3D model for FVD...")
    i3d_model = load_i3d_model(device)
    print("Loading InceptionV3 model for FID...")
    incep_model = load_inception_model(device)

    with open(args.mapping_csv) as f:
        video_list = list(csv.DictReader(f))
    if args.max_videos > 0:
        video_list = video_list[:args.max_videos]

    method_name = "pvdm_baseline" if args.no_optimize else "savi_dno"
    print("Processing %d videos (%s)..." % (len(video_list), method_name))
    print("  data_dir: %s" % args.data_dir)
    print("  mapping_csv: %s" % args.mapping_csv)
    sys.stdout.flush()

    results = []
    total_psnr = total_ssim = total_lpips = 0.0
    n_ok = 0
    not_found_count = 0

    # FVD/FID accumulators
    d_fvd = _I3D_FEATURE_DIM
    gen_fvd_sum = np.zeros(d_fvd, dtype=np.float64)
    gen_fvd_cov = np.zeros((d_fvd, d_fvd), dtype=np.float64)
    ref_fvd_sum = np.zeros(d_fvd, dtype=np.float64)
    ref_fvd_cov = np.zeros((d_fvd, d_fvd), dtype=np.float64)
    fvd_count = 0

    d_fid = _FID_FEATURE_DIM
    gen_fid_sum = np.zeros(d_fid, dtype=np.float64)
    gen_fid_cov = np.zeros((d_fid, d_fid), dtype=np.float64)
    ref_fid_sum = np.zeros(d_fid, dtype=np.float64)
    ref_fid_cov = np.zeros((d_fid, d_fid), dtype=np.float64)
    fid_gen_frames = 0
    fid_ref_frames = 0

    for idx, entry in enumerate(tqdm(video_list, desc=method_name)):
        pvdm_path = os.path.join(args.data_dir, entry["pvdm_path"])
        original = entry["original_filename"]

        if not os.path.exists(pvdm_path):
            if not_found_count < 3:
                print("  NOT FOUND [%d]: %s" % (idx, pvdm_path), file=sys.stderr)
            not_found_count += 1
            results.append({"video": original, "success": False, "error": "not_found: %s" % pvdm_path})
            continue

        try:
            all_frames = load_video_frames(pvdm_path, 32, size=256)
            cond_frames = all_frames[:16].unsqueeze(0).to(device)
            gt_frames = all_frames[16:32].unsqueeze(0).to(device)

            savi.reset()
            z_cond = savi.encode(cond_frames)
            latent_shape = (1, z_cond.shape[1], z_cond.shape[2])

            t_start = time.time()
            if args.no_optimize:
                with torch.no_grad():
                    eps = torch.randn(latent_shape, device=device)
                    z_pred = savi._ddim_sample_differentiable(z_cond, eps)
                    pred_frames = savi.decode(z_pred)
                loss_val = None
            else:
                pred_frames, loss_val = savi.predict_and_optimize(
                    z_cond, gt_frames, latent_shape)
            elapsed = time.time() - t_start

            pred_np = pred_frames.cpu().numpy()  # [(B*T), C, H, W]
            gt_np = rearrange(gt_frames, 'b t c h w -> (b t) c h w').cpu().numpy()

            psnr, ssim = compute_metrics(pred_np, gt_np)
            lpips_val = compute_lpips(lpips_model, pred_np, gt_np, device)

            # FVD: accumulate I3D features
            gen_clip = frames_to_i3d_tensor(pred_np)
            ref_clip = frames_to_i3d_tensor(gt_np)
            gen_feat = i3d_features(i3d_model, gen_clip, device)
            ref_feat = i3d_features(i3d_model, ref_clip, device)
            gen_fvd_sum += gen_feat
            gen_fvd_cov += np.outer(gen_feat, gen_feat)
            ref_fvd_sum += ref_feat
            ref_fvd_cov += np.outer(ref_feat, ref_feat)
            fvd_count += 1

            # FID: accumulate InceptionV3 features
            gen_fid_feat = inception_features(incep_model, pred_np, device)
            ref_fid_feat = inception_features(incep_model, gt_np, device)
            gen_fid_sum += gen_fid_feat.sum(axis=0)
            gen_fid_cov += gen_fid_feat.T @ gen_fid_feat
            ref_fid_sum += ref_fid_feat.sum(axis=0)
            ref_fid_cov += ref_fid_feat.T @ ref_fid_feat
            fid_gen_frames += gen_fid_feat.shape[0]
            fid_ref_frames += ref_fid_feat.shape[0]

            results.append({
                "video": original, "success": True,
                "psnr": psnr, "ssim": ssim, "lpips": lpips_val,
                "loss": loss_val, "time": elapsed,
            })
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_val
            n_ok += 1

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"video": original, "success": False, "error": str(e)})

    if not_found_count > 0:
        print("WARNING: %d/%d videos not found!" % (not_found_count, len(video_list)))

    # Compute FVD and FID
    fvd_val = None
    fid_val = None
    if fvd_count >= 2:
        fvd_val = compute_frechet_distance(
            gen_fvd_sum, gen_fvd_cov, fvd_count,
            ref_fvd_sum, ref_fvd_cov, fvd_count)
        print("[Online FVD] FVD = %.4f (%d videos)" % (fvd_val, fvd_count))
    if fid_gen_frames >= 2:
        fid_val = compute_frechet_distance(
            gen_fid_sum, gen_fid_cov, fid_gen_frames,
            ref_fid_sum, ref_fid_cov, fid_ref_frames)
        print("[Online FID] FID = %.4f (%d gen / %d ref frames)" % (
            fid_val, fid_gen_frames, fid_ref_frames))

    summary = {
        "method": method_name,
        "num_videos": len(video_list),
        "num_successful": n_ok,
        "num_not_found": not_found_count,
        "avg_psnr": total_psnr / max(n_ok, 1),
        "avg_ssim": total_ssim / max(n_ok, 1),
        "avg_lpips": total_lpips / max(n_ok, 1),
        "fvd": round(fvd_val, 6) if fvd_val is not None else None,
        "fvd_num_videos": fvd_count,
        "fvd_feature_extractor": "i3d_kinetics400_torchscript",
        "fid": round(fid_val, 6) if fid_val is not None else None,
        "fid_num_frames_gen": fid_gen_frames,
        "fid_num_frames_ref": fid_ref_frames,
        "fid_feature_extractor": "inception_v3_imagenet",
        "resolution": "256x256",
        "ddim_steps": args.ddim_steps,
        "lr": args.lr,
        "lam": args.lam,
        "p": args.p,
        "results": results,
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 60)
    print("%s Complete" % method_name.upper())
    print("  Videos: %d/%d" % (n_ok, len(video_list)))
    print("  Avg PSNR:  %.4f" % summary["avg_psnr"])
    print("  Avg SSIM:  %.4f" % summary["avg_ssim"])
    print("  Avg LPIPS: %.4f" % summary["avg_lpips"])
    if fvd_val is not None:
        print("  FVD:       %.4f" % fvd_val)
    if fid_val is not None:
        print("  FID:       %.4f" % fid_val)
    print("  Results: %s" % str(output_dir / "summary.json"))
    print("=" * 60)


if __name__ == "__main__":
    main()
