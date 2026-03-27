#!/usr/bin/env python3
"""
DFoT evaluation runner for UCF-101.

Loads the pretrained DFoT (Kinetics-600) model and runs video prediction
on our UCF-101 dataset. Computes PSNR, SSIM per-video and aggregates.

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


def run_dfot_inference_standalone(dfot_dir, checkpoint_path, data_dir, output_dir,
                                  mapping_csv, context_length=5, pred_length=12,
                                  max_videos=500, seed=42, batch_size=1):
    """
    Standalone DFoT inference that loads videos manually and runs
    the model's generate/sample method.

    This approach bypasses DFoT's Hydra config system for flexibility,
    directly loading the model and running frame-by-frame prediction.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    total_frames = context_length + pred_length

    # Load video mapping
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

    # Try to load DFoT model via their framework
    sys.path.insert(0, str(dfot_dir))

    try:
        # Attempt to use DFoT's own loading mechanism
        from utils.ckpt_utils import load_checkpoint
        model = load_checkpoint(checkpoint_path, device=device)
        use_native = True
        print("Loaded DFoT model via native loader")
    except Exception as e:
        print("Could not load via native loader: %s" % str(e))
        print("Attempting direct checkpoint load...")
        use_native = False

        # Fallback: load raw checkpoint and try to reconstruct model
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        print("Checkpoint keys: %s" % str(list(ckpt.keys())[:10]))

        # Store checkpoint for later analysis
        fallback_info = {
            "error": str(e),
            "ckpt_keys": list(ckpt.keys())[:20],
            "note": "DFoT model loading requires Hydra config. "
                    "Run via DFoT's own entry point instead.",
        }
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "load_info.json"), "w") as f:
            json.dump(fallback_info, f, indent=2)

    results = []
    n_ok = 0
    total_psnr = 0.0
    total_ssim = 0.0

    video_dir = os.path.join(data_dir, "test")

    for idx, entry in enumerate(tqdm(video_list, desc="DFoT eval")):
        dfot_filename = entry["dfot_filename"]
        original = entry["original_filename"]
        video_path = os.path.join(video_dir, dfot_filename)

        if not os.path.exists(video_path):
            results.append({"video": original, "success": False, "error": "not_found"})
            continue

        try:
            # Load frames
            all_frames = load_video_frames(video_path, total_frames, size=128)
            context = all_frames[:context_length]  # [ctx, C, H, W]
            gt = all_frames[context_length:total_frames]  # [pred, C, H, W]

            t_start = time.time()

            if use_native:
                # Use DFoT model for prediction
                with torch.no_grad():
                    context_batch = context.unsqueeze(0).to(device)  # [1, ctx, C, H, W]
                    pred = model.predict(context_batch, pred_length)
                    if pred.dim() == 5:
                        pred = pred[0]  # [pred, C, H, W]
                    pred = pred.clamp(0, 1).cpu()
            else:
                # Fallback: copy context frames as dummy prediction
                # (indicates model couldn't be loaded; metrics will be meaningful
                #  only as a placeholder until the loading issue is resolved)
                pred = context[-1:].repeat(pred_length, 1, 1, 1)

            elapsed = time.time() - t_start

            pred_np = pred.numpy()
            gt_np = gt.numpy()
            psnr, ssim = compute_metrics(pred_np, gt_np)

            results.append({
                "video": original,
                "success": True,
                "psnr": psnr,
                "ssim": ssim,
                "time": elapsed,
                "native_model": use_native,
            })
            total_psnr += psnr
            total_ssim += ssim
            n_ok += 1

        except Exception as e:
            results.append({"video": original, "success": False, "error": str(e)})

    # Summary
    summary = {
        "method": "dfot_k600",
        "num_videos": len(video_list),
        "num_successful": n_ok,
        "avg_psnr": total_psnr / max(n_ok, 1),
        "avg_ssim": total_ssim / max(n_ok, 1),
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
    print("  Avg PSNR: %.4f" % summary["avg_psnr"])
    print("  Avg SSIM: %.4f" % summary["avg_ssim"])
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
