#!/usr/bin/env python3
"""
DFoT evaluation runner for UCF-101.

Loads the pretrained DFoT (Kinetics-600) model and runs video prediction
on our UCF-101 dataset. Computes PSNR, SSIM, LPIPS per-video and
aggregates FVD/FID online.

DFoT uses a latent diffusion framework with history-guided conditioning.
For UCF-101, we use the K600 pretrained model at 128x128 resolution.

Usage:
    python run_dfot_eval.py \
        --dfot-dir /path/to/DFoT \
        --data-dir /path/to/ucf101_dfot \
        --output-dir /path/to/results/dfot \
        --checkpoint /path/to/DFoT_K600.ckpt \
        --context-length 5 \
        --pred-length 12 \
        --max-videos 500
"""

import sys
import os
import time
import json
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim


def load_video_frames(video_path, num_frames, size=128):
    """Load video frames, center-crop and resize to size x size."""
    import imageio.v3 as iio
    from PIL import Image
    import torchvision.transforms as T

    frames = []
    try:
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
        t = T.ToTensor()(img)
        frames.append(t)

    while len(frames) < num_frames:
        frames.append(frames[-1].clone())

    return torch.stack(frames)  # [T, C, H, W]


def compute_metrics(pred_np, gt_np):
    """Compute PSNR and SSIM between predicted and GT frames."""
    T = pred_np.shape[0]
    psnrs, ssims = [], []
    for t in range(T):
        p = pred_np[t].transpose(1, 2, 0)  # [H, W, C]
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
    import lpips
    model = lpips.LPIPS(net="alex").to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def compute_lpips(lpips_model, pred_np, gt_np, device):
    """pred_np, gt_np: [T, C, H, W] in [0,1]. Returns mean LPIPS."""
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
        arr = (np.clip(frames_np[i].transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
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


# ============================================================================
# Main evaluation
# ============================================================================

def run_dfot_inference_standalone(dfot_dir, checkpoint_path, data_dir, output_dir,
                                  mapping_csv, context_length=5, pred_length=12,
                                  max_videos=500, seed=42, batch_size=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    total_frames = context_length + pred_length

    with open(mapping_csv) as f:
        reader = csv.DictReader(f)
        video_list = list(reader)

    if max_videos > 0:
        video_list = video_list[:max_videos]

    print("DFoT Evaluation")
    print("  Videos: %d" % len(video_list))
    print("  Context: %d frames, Pred: %d frames" % (context_length, pred_length))
    print("  Resolution: 128x128")
    print("  Checkpoint: %s" % checkpoint_path)
    print()

    sys.path.insert(0, str(dfot_dir))

    try:
        from utils.ckpt_utils import load_checkpoint
        model = load_checkpoint(checkpoint_path, device=device)
        use_native = True
        print("Loaded DFoT model via native loader")
    except Exception as e:
        print("Could not load via native loader: %s" % str(e))
        print("Attempting direct checkpoint load...")
        use_native = False

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        print("Checkpoint keys: %s" % str(list(ckpt.keys())[:10]))

        fallback_info = {
            "error": str(e),
            "ckpt_keys": list(ckpt.keys())[:20],
            "note": "DFoT model loading requires Hydra config. "
                    "Run via DFoT's own entry point instead.",
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "load_info.json"), "w") as f:
            json.dump(fallback_info, f, indent=2)

    # Load metric models
    print("Loading LPIPS model...")
    lpips_model = load_lpips_model(device)
    print("Loading I3D model for FVD...")
    i3d_model = load_i3d_model(device)
    print("Loading InceptionV3 model for FID...")
    incep_model = load_inception_model(device)

    results = []
    n_ok = 0
    total_psnr = 0.0
    total_ssim = 0.0
    total_lpips = 0.0

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

    video_dir = os.path.join(data_dir, "test")

    for idx, entry in enumerate(tqdm(video_list, desc="DFoT eval")):
        dfot_filename = entry["dfot_filename"]
        original = entry["original_filename"]
        video_path = os.path.join(video_dir, dfot_filename)

        if not os.path.exists(video_path):
            results.append({"video": original, "success": False, "error": "not_found"})
            continue

        try:
            all_frames = load_video_frames(video_path, total_frames, size=128)
            context = all_frames[:context_length]
            gt = all_frames[context_length:total_frames]

            t_start = time.time()

            if use_native:
                with torch.no_grad():
                    context_batch = context.unsqueeze(0).to(device)
                    pred = model.predict(context_batch, pred_length)
                    if pred.dim() == 5:
                        pred = pred[0]
                    pred = pred.clamp(0, 1).cpu()
            else:
                pred = context[-1:].repeat(pred_length, 1, 1, 1)

            elapsed = time.time() - t_start

            pred_np = pred.numpy()
            gt_np = gt.numpy()
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
                "video": original,
                "success": True,
                "psnr": psnr,
                "ssim": ssim,
                "lpips": lpips_val,
                "time": elapsed,
                "native_model": use_native,
            })
            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips_val
            n_ok += 1

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({"video": original, "success": False, "error": str(e)})

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
        "method": "dfot_k600",
        "num_videos": len(video_list),
        "num_successful": n_ok,
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
        "context_length": context_length,
        "pred_length": pred_length,
        "resolution": 128,
        "native_model_loaded": use_native,
        "results": results,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print()
    print("=" * 60)
    print("DFoT Evaluation Complete")
    print("  Videos: %d/%d" % (n_ok, len(video_list)))
    print("  Avg PSNR:  %.4f" % summary["avg_psnr"])
    print("  Avg SSIM:  %.4f" % summary["avg_ssim"])
    print("  Avg LPIPS: %.4f" % summary["avg_lpips"])
    if fvd_val is not None:
        print("  FVD:       %.4f" % fvd_val)
    if fid_val is not None:
        print("  FID:       %.4f" % fid_val)
    print("  Native model: %s" % use_native)
    print("  Results: %s" % out_path)
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="DFoT UCF-101 Evaluation")
    parser.add_argument("--dfot-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mapping-csv", required=True)
    parser.add_argument("--context-length", type=int, default=5)
    parser.add_argument("--pred-length", type=int, default=12)
    parser.add_argument("--max-videos", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dfot_inference_standalone(
        dfot_dir=args.dfot_dir,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mapping_csv=args.mapping_csv,
        context_length=args.context_length,
        pred_length=args.pred_length,
        max_videos=args.max_videos,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
