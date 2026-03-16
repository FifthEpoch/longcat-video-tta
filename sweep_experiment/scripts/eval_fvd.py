#!/usr/bin/env python3
"""
Compute Frechet Video Distance (FVD) between generated and reference video sets.

Uses a pretrained I3D network for feature extraction. Requires 256+ videos
for reliable covariance estimation -- use UCF-101 test set, not Panda-70M.

Usage:
    python sweep_experiment/scripts/eval_fvd.py \
        --gen-dir sweep_experiment/results/ucf101_full/best/videos \
        --ref-dir /scratch/wc3013/datasets/ucf101_test_480p/videos \
        --num-frames 16 \
        --output results/ucf101_full_fvd.json
"""
import argparse, json, os, sys
from pathlib import Path
from typing import List

import numpy as np
import torch


def load_video_as_tensor(video_path: str, num_frames: int = 16, 
                          size: int = 224) -> torch.Tensor:
    """Load video as [1, T, C, H, W] float32 tensor in [0,1], center-cropped and resized."""
    import av
    from torchvision.transforms import functional as TF
    from PIL import Image

    container = av.open(video_path)
    frames = []
    for frame in container.decode(video=0):
        if len(frames) >= num_frames:
            break
        img = frame.to_image()
        frames.append(img)
    container.close()

    if len(frames) == 0:
        return None

    # Pad with last frame if needed
    while len(frames) < num_frames:
        frames.append(frames[-1])

    tensors = []
    for img in frames[:num_frames]:
        # Resize to (size, size) with center crop
        img = TF.resize(img, size, interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.center_crop(img, size)
        t = TF.to_tensor(img)  # [C, H, W] in [0, 1]
        tensors.append(t)

    return torch.stack(tensors, dim=0).unsqueeze(0)  # [1, T, C, H, W]


def extract_i3d_features(videos: List[torch.Tensor], device: str = "cuda",
                          batch_size: int = 8) -> np.ndarray:
    """Extract I3D features from video tensors using pytorch_i3d or torchvision."""
    try:
        from torchvision.models.video import r3d_18, R3D_18_Weights
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        # Remove the final classification layer to get features
        model.fc = torch.nn.Identity()
        model = model.to(device).eval()
    except Exception:
        # Fallback: use a simple mean pooling over frames
        print("WARNING: Could not load R3D-18, using mean pooling features",
              file=sys.stderr)
        features = []
        for v in videos:
            feat = v.mean(dim=[1, 3, 4]).numpy()  # [1, C]
            features.append(feat.flatten())
        return np.stack(features)

    features = []
    with torch.no_grad():
        for i in range(0, len(videos), batch_size):
            batch = videos[i:i + batch_size]
            # Stack and permute to [B, C, T, H, W]
            batch_tensor = torch.cat(batch, dim=0).permute(0, 2, 1, 3, 4).to(device)
            feats = model(batch_tensor).cpu().numpy()
            features.append(feats)

    return np.concatenate(features, axis=0)


def compute_fvd(feats_gen: np.ndarray, feats_ref: np.ndarray) -> float:
    """Compute Frechet distance between two sets of features."""
    mu_gen = np.mean(feats_gen, axis=0)
    mu_ref = np.mean(feats_ref, axis=0)
    sigma_gen = np.cov(feats_gen, rowvar=False)
    sigma_ref = np.cov(feats_ref, rowvar=False)

    from scipy.linalg import sqrtm

    diff = mu_gen - mu_ref
    covmean, _ = sqrtm(sigma_gen @ sigma_ref, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fvd = diff @ diff + np.trace(sigma_gen + sigma_ref - 2 * covmean)
    return float(fvd)


def main():
    parser = argparse.ArgumentParser(description="Compute FVD between video sets")
    parser.add_argument("--gen-dir", type=str, required=True,
                        help="Directory of generated videos")
    parser.add_argument("--ref-dir", type=str, required=True,
                        help="Directory of reference/ground truth videos")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    gen_videos = sorted(Path(args.gen_dir).glob("*.mp4"))
    ref_videos = sorted(Path(args.ref_dir).glob("*.mp4"))

    if len(gen_videos) < 256:
        print(f"WARNING: Only {len(gen_videos)} generated videos. "
              f"FVD requires 256+ for reliable estimation.", file=sys.stderr)
    if len(ref_videos) < 256:
        print(f"WARNING: Only {len(ref_videos)} reference videos.", file=sys.stderr)

    n = min(len(gen_videos), len(ref_videos))
    print(f"Computing FVD with {n} video pairs...")

    print("Loading generated videos...")
    gen_tensors = []
    for vp in gen_videos[:n]:
        t = load_video_as_tensor(str(vp), args.num_frames)
        if t is not None:
            gen_tensors.append(t)

    print("Loading reference videos...")
    ref_tensors = []
    for vp in ref_videos[:n]:
        t = load_video_as_tensor(str(vp), args.num_frames)
        if t is not None:
            ref_tensors.append(t)

    n_final = min(len(gen_tensors), len(ref_tensors))
    print(f"  Valid pairs: {n_final}")

    print("Extracting features (generated)...")
    feats_gen = extract_i3d_features(gen_tensors[:n_final], args.device)
    print("Extracting features (reference)...")
    feats_ref = extract_i3d_features(ref_tensors[:n_final], args.device)

    fvd = compute_fvd(feats_gen, feats_ref)
    print(f"\nFVD: {fvd:.2f}")

    result = {
        "fvd": fvd,
        "num_videos": n_final,
        "gen_dir": args.gen_dir,
        "ref_dir": args.ref_dir,
        "num_frames": args.num_frames,
    }

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
