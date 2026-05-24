#!/usr/bin/env python3
"""
Compute Frechet Video Distance (FVD) and optionally per-frame Frechet
Inception Distance (FID) between generated and reference video sets.

FVD uses the canonical I3D network (Kinetics-400, TorchScript) matching
DFoT (ICML 2025), CVP, and SAVi-DNO baselines.  FID uses InceptionV3
(ImageNet, 2048-dim pool features) per frame -- reported by DFoT only.

Requires 256+ videos for reliable covariance estimation (configurable via
--min-videos).  Use --force to override the sample-size check.

Usage:
    python sweep_experiment/scripts/eval_fvd.py \\
        --gen-dir results/best/videos \\
        --ref-dir /scratch/datasets/ucf101_test/videos \\
        --num-frames 16 \\
        --output results/fvd_score.json

    # With FID, forced small sample, and self-consistency check:
    python sweep_experiment/scripts/eval_fvd.py \\
        --gen-dir results/best/videos \\
        --ref-dir /scratch/datasets/ucf101_test/videos \\
        --compute-fid --force --self-check \\
        --output results/fvd_fid_score.json
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from scipy.linalg import sqrtm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_I3D_HF_REPO = "kiwhansong/DFoT"
_I3D_HF_FILE = "metrics_models/i3d_torchscript.pt"
_I3D_FEATURE_DIM = 400
_FID_FEATURE_DIM = 2048
_MIN_I3D_FRAMES = 9
_DEFAULT_MIN_VIDEOS = 256
_COV_EPS = 1e-6


# ---------------------------------------------------------------------------
# I3D model loading (DFoT-compatible TorchScript)
# ---------------------------------------------------------------------------
def _download_i3d() -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=_I3D_HF_REPO, filename=_I3D_HF_FILE)


def _load_i3d(device: str) -> torch.jit.ScriptModule:
    model_path = _download_i3d()
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _i3d_file_hash() -> str:
    path = _download_i3d()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# InceptionV3 for per-frame FID
# ---------------------------------------------------------------------------
def _load_inception_v3(device: str) -> torch.nn.Module:
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ---------------------------------------------------------------------------
# Video loading
# ---------------------------------------------------------------------------
def load_video_as_tensor(
    video_path: str,
    num_frames: int = 16,
    size: int = 224,
) -> Optional[torch.Tensor]:
    """Load video as [1, T, C, H, W] float32 tensor in [0, 1]."""
    import av
    from torchvision.transforms import functional as TF

    try:
        container = av.open(video_path)
    except Exception as exc:
        print(f"  SKIP (open failed): {video_path} -- {exc}", file=sys.stderr)
        return None

    frames: list = []
    try:
        for frame in container.decode(video=0):
            if len(frames) >= num_frames:
                break
            frames.append(frame.to_image())
    except Exception as exc:
        print(f"  SKIP (decode failed): {video_path} -- {exc}", file=sys.stderr)
        return None
    finally:
        container.close()

    if len(frames) == 0:
        return None

    while len(frames) < num_frames:
        frames.append(frames[-1])

    tensors = []
    for img in frames[:num_frames]:
        img = TF.resize(img, size, interpolation=TF.InterpolationMode.BILINEAR)
        img = TF.center_crop(img, size)
        tensors.append(TF.to_tensor(img))  # [C, H, W] in [0, 1]

    return torch.stack(tensors, dim=0).unsqueeze(0)  # [1, T, C, H, W]


# ---------------------------------------------------------------------------
# I3D feature extraction (DFoT protocol)
# ---------------------------------------------------------------------------
def _pad_for_i3d(x: torch.Tensor) -> torch.Tensor:
    """Symmetric padding to >= 9 frames, matching DFoT."""
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


def extract_i3d_features(
    videos: List[torch.Tensor],
    model: torch.jit.ScriptModule,
    device: str = "cuda",
    batch_size: int = 4,
) -> np.ndarray:
    """Extract 400-dim I3D features following DFoT's exact protocol.

    Input videos: list of [1, T, C, H, W] tensors in [0, 1].
    Pipeline: pad -> normalize to [-1, 1] -> permute to B,C,T,H,W -> I3D.
    """
    features = []
    with torch.no_grad():
        for i in range(0, len(videos), batch_size):
            batch = torch.cat(videos[i : i + batch_size], dim=0).to(device)
            batch = _pad_for_i3d(batch)
            batch = torch.clamp(2.0 * batch - 1.0, -1.0, 1.0)
            batch = batch.permute(0, 2, 1, 3, 4).contiguous()  # B,C,T,H,W
            feats = model(batch, rescale=False, resize=True, return_features=True)
            features.append(feats.cpu().to(torch.float64).numpy())

    return np.concatenate(features, axis=0)


# ---------------------------------------------------------------------------
# FID feature extraction (per-frame InceptionV3)
# ---------------------------------------------------------------------------
def extract_fid_features(
    videos: List[torch.Tensor],
    model: torch.nn.Module,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    """Extract 2048-dim InceptionV3 features from every frame."""
    from torchvision.transforms import functional as TF

    all_frames: list = []
    for v in videos:
        for t in range(v.shape[1]):
            frame = v[0, t]  # [C, H, W] in [0, 1]
            frame = TF.resize(frame, 299, interpolation=TF.InterpolationMode.BILINEAR)
            frame = TF.center_crop(frame, 299)
            frame = TF.normalize(
                frame,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            all_frames.append(frame)

    features = []
    with torch.no_grad():
        for i in range(0, len(all_frames), batch_size):
            batch = torch.stack(all_frames[i : i + batch_size]).to(device)
            feats = model(batch)  # [B, 2048] (fc replaced with Identity)
            features.append(feats.cpu().to(torch.float64).numpy())

    return np.concatenate(features, axis=0)


# ---------------------------------------------------------------------------
# Frechet distance (shared by FVD and FID)
# ---------------------------------------------------------------------------
def compute_frechet_distance(
    feats_a: np.ndarray,
    feats_b: np.ndarray,
    eps: float = _COV_EPS,
) -> float:
    """Frechet distance with float64 precision and covariance regularization."""
    feats_a = feats_a.astype(np.float64)
    feats_b = feats_b.astype(np.float64)

    mu_a = np.mean(feats_a, axis=0)
    mu_b = np.mean(feats_b, axis=0)
    sigma_a = np.cov(feats_a, rowvar=False)
    sigma_b = np.cov(feats_b, rowvar=False)

    sigma_a += eps * np.eye(sigma_a.shape[0])
    sigma_b += eps * np.eye(sigma_b.shape[0])

    diff = mu_a - mu_b
    covmean, _ = sqrtm(sigma_a @ sigma_b, disp=False)

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            print(
                "WARNING: sqrtm produced non-negligible imaginary components; "
                "taking real part.",
                file=sys.stderr,
            )
        covmean = covmean.real

    return float(diff @ diff + np.trace(sigma_a + sigma_b - 2 * covmean))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compute FVD (and optionally FID) between video sets "
        "using canonical I3D (Kinetics-400) features.",
    )
    parser.add_argument(
        "--gen-dir", type=str, required=True,
        help="Directory of generated videos (.mp4)",
    )
    parser.add_argument(
        "--ref-dir", type=str, required=True,
        help="Directory of reference/ground-truth videos (.mp4)",
    )
    parser.add_argument(
        "--num-frames", type=int, default=16,
        help="Frames per clip (default: 16)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size for I3D feature extraction (default: 4)",
    )
    parser.add_argument(
        "--min-videos", type=int, default=_DEFAULT_MIN_VIDEOS,
        help="Hard minimum number of valid video pairs "
        f"(default: {_DEFAULT_MIN_VIDEOS})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Override the minimum sample-size check "
        "(result will carry a warning)",
    )
    parser.add_argument(
        "--compute-fid", action="store_true",
        help="Also compute per-frame FID (InceptionV3, 2048-dim)",
    )
    parser.add_argument(
        "--self-check", action="store_true",
        help="Run a self-consistency test (FVD of ref vs ref should be ~0)",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    t0 = time.time()

    # ------------------------------------------------------------------ discover videos
    gen_paths = sorted(Path(args.gen_dir).glob("*.mp4"))
    ref_paths = sorted(Path(args.ref_dir).glob("*.mp4"))
    print(f"Found {len(gen_paths)} generated, {len(ref_paths)} reference videos.")

    n = min(len(gen_paths), len(ref_paths))
    if n == 0:
        print("ERROR: No videos found in one or both directories.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ load videos
    print("Loading generated videos...")
    gen_tensors: List[torch.Tensor] = []
    for vp in gen_paths[:n]:
        t = load_video_as_tensor(str(vp), args.num_frames)
        if t is not None:
            gen_tensors.append(t)

    print("Loading reference videos...")
    ref_tensors: List[torch.Tensor] = []
    for vp in ref_paths[:n]:
        t = load_video_as_tensor(str(vp), args.num_frames)
        if t is not None:
            ref_tensors.append(t)

    n_valid = min(len(gen_tensors), len(ref_tensors))
    print(f"Valid video pairs: {n_valid}")

    # ------------------------------------------------------------------ sample-size guard
    sample_size_warning: Optional[str] = None
    if n_valid < args.min_videos:
        msg = (
            f"Only {n_valid} valid videos, below --min-videos={args.min_videos}. "
            f"FVD covariance estimation is unreliable with fewer than "
            f"{args.min_videos} samples."
        )
        if not args.force:
            print(f"ERROR: {msg}  Use --force to override.", file=sys.stderr)
            sys.exit(1)
        sample_size_warning = msg
        print(f"WARNING (--force): {msg}", file=sys.stderr)

    gen_tensors = gen_tensors[:n_valid]
    ref_tensors = ref_tensors[:n_valid]

    # ------------------------------------------------------------------ load I3D
    print("Loading I3D (Kinetics-400 TorchScript)...")
    i3d_model = _load_i3d(args.device)
    i3d_hash = _i3d_file_hash()
    print(f"  I3D weights SHA-256: {i3d_hash[:16]}...")
    print(f"  Feature dim: {_I3D_FEATURE_DIM}")

    # ------------------------------------------------------------------ I3D features
    print("Extracting I3D features (generated)...")
    feats_gen = extract_i3d_features(
        gen_tensors, i3d_model, args.device, args.batch_size,
    )
    print(f"  shape: {feats_gen.shape}")

    print("Extracting I3D features (reference)...")
    feats_ref = extract_i3d_features(
        ref_tensors, i3d_model, args.device, args.batch_size,
    )
    print(f"  shape: {feats_ref.shape}")

    assert feats_gen.shape[1] == _I3D_FEATURE_DIM, (
        f"Expected {_I3D_FEATURE_DIM}-dim I3D features, got {feats_gen.shape[1]}"
    )

    # ------------------------------------------------------------------ FVD
    fvd_score = compute_frechet_distance(feats_gen, feats_ref)
    print(f"\nFVD: {fvd_score:.4f}")

    result = {
        "fvd": round(fvd_score, 6),
        "fid": None,
        "num_gen_videos": len(gen_tensors),
        "num_ref_videos": len(ref_tensors),
        "num_valid_pairs": n_valid,
        "num_frames_per_clip": args.num_frames,
        "feature_extractor": "i3d_kinetics400_torchscript",
        "feature_dim": _I3D_FEATURE_DIM,
        "normalization": "[-1, 1]",
        "i3d_weights_sha256": i3d_hash,
        "sample_size_warning": sample_size_warning,
        "gen_dir": str(args.gen_dir),
        "ref_dir": str(args.ref_dir),
    }

    # ------------------------------------------------------------------ optional FID
    if args.compute_fid:
        print("\nLoading InceptionV3 for per-frame FID...")
        inception = _load_inception_v3(args.device)

        print("Extracting InceptionV3 features (generated frames)...")
        fid_feats_gen = extract_fid_features(
            gen_tensors, inception, args.device,
        )
        print(f"  shape: {fid_feats_gen.shape}")

        print("Extracting InceptionV3 features (reference frames)...")
        fid_feats_ref = extract_fid_features(
            ref_tensors, inception, args.device,
        )
        print(f"  shape: {fid_feats_ref.shape}")

        fid_score = compute_frechet_distance(fid_feats_gen, fid_feats_ref)
        print(f"FID: {fid_score:.4f}")
        result["fid"] = round(fid_score, 6)
        result["fid_feature_extractor"] = "inception_v3_imagenet"
        result["fid_feature_dim"] = _FID_FEATURE_DIM
        result["fid_num_frames_gen"] = fid_feats_gen.shape[0]
        result["fid_num_frames_ref"] = fid_feats_ref.shape[0]

    # ------------------------------------------------------------------ self-check
    if args.self_check:
        print("\nSelf-consistency check (ref vs ref)...")
        fvd_self = compute_frechet_distance(feats_ref, feats_ref)
        print(f"  FVD(ref, ref) = {fvd_self:.6f}  (expected ~0)")
        result["self_check_fvd"] = round(fvd_self, 6)
        if fvd_self > 1.0:
            print(
                f"WARNING: Self-check FVD = {fvd_self:.4f} is unexpectedly large.",
                file=sys.stderr,
            )

    # ------------------------------------------------------------------ wrap up
    elapsed = time.time() - t0
    result["elapsed_seconds"] = round(elapsed, 1)
    print(f"\nTotal time: {elapsed:.1f}s")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
