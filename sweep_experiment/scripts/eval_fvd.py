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
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.linalg import sqrtm

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.caption_utils import canonical_video_id

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
    *,
    num_cond_frames: int = 0,
    num_gen_frames: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """Load video as [1, T, C, H, W] float32 tensor in [0, 1].

    When ``num_cond_frames`` > 0, saved LongCat clips are assumed to be
    ``[cond | gen]`` (e.g. 14+14).  Only the generated tail is used so
    offline FVD matches the online accumulator in ``common.py``:
    ``gen_output[num_cond_frames : num_cond_frames + num_gen_frames]``.
    """
    import av
    from torchvision.transforms import functional as TF

    if num_gen_frames is not None:
        clip_len = num_cond_frames + num_gen_frames
    else:
        clip_len = num_frames

    try:
        container = av.open(video_path)
    except Exception as exc:
        print(f"  SKIP (open failed): {video_path} -- {exc}", file=sys.stderr)
        return None

    frames: list = []
    try:
        for frame in container.decode(video=0):
            if len(frames) >= clip_len:
                break
            frames.append(frame.to_image())
    except Exception as exc:
        print(f"  SKIP (decode failed): {video_path} -- {exc}", file=sys.stderr)
        return None
    finally:
        container.close()

    if len(frames) == 0:
        return None

    if num_cond_frames > 0:
        if len(frames) <= num_cond_frames:
            print(
                f"  SKIP (too short for cond skip): {video_path} "
                f"({len(frames)} frames, need >{num_cond_frames})",
                file=sys.stderr,
            )
            return None
        frames = frames[num_cond_frames:]

    eval_frames = num_gen_frames if num_gen_frames is not None else num_frames
    while len(frames) < eval_frames:
        frames.append(frames[-1])

    tensors = []
    for img in frames[:eval_frames]:
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
def _canonical_video_id_from_path(path: Path) -> str:
    """Map any saved mp4 name back to ``panda_XXXX`` (or ``ucf_XXXX``)."""
    stem = path.stem
    vid = canonical_video_id(stem)
    if vid:
        return vid
    m = re.search(r"(\d+)", stem)
    if m:
        prefix = "panda" if "panda" in stem.lower() else "video"
        return f"{prefix}_{int(m.group(1)):04d}"
    return stem


def _pair_videos_by_id(
    gen_paths: List[Path],
    ref_paths: List[Path],
) -> List[Tuple[Path, Path]]:
    """Pair generated and reference clips by canonical video id."""
    ref_by_id: Dict[str, Path] = {}
    for rp in ref_paths:
        vid = _canonical_video_id_from_path(rp)
        if vid:
            ref_by_id[vid] = rp

    pairs: List[Tuple[Path, Path]] = []
    missing_ref: List[str] = []
    identical: List[str] = []
    for gp in gen_paths:
        vid = _canonical_video_id_from_path(gp)
        if not vid or vid not in ref_by_id:
            missing_ref.append(vid or gp.name)
            continue
        rp = ref_by_id[vid]
        if gp.resolve() == rp.resolve():
            identical.append(vid)
            continue
        pairs.append((gp, rp))

    if missing_ref:
        print(
            f"WARNING: {len(missing_ref)} generated videos have no reference "
            f"match (showing up to 5): {missing_ref[:5]}",
            file=sys.stderr,
        )
    if identical:
        print(
            f"ERROR: {len(identical)} generated/reference pairs resolve to the "
            f"same file (gen==ref → FVD≈0). First examples: {identical[:5]}",
            file=sys.stderr,
        )
        print(
            "  This usually means oracle/gen symlinks fell back to GT source "
            "videos. Rebuild policy dirs with build_oracle_policy_dirs.py.",
            file=sys.stderr,
        )
        sys.exit(1)

    return pairs


def compute_frechet_from_sufficient_stats(
    sum_a: np.ndarray,
    cov_a: np.ndarray,
    n_a: int,
    sum_b: np.ndarray,
    cov_b: np.ndarray,
    n_b: int,
    eps: float = _COV_EPS,
) -> float:
    """Frechet distance from running sums (matches merge_chunks.py)."""
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
        "--ref-dir", type=str, default=None,
        help="Directory of reference/ground-truth videos (.mp4). "
        "Required unless --gt-cache is set.",
    )
    parser.add_argument(
        "--num-frames", type=int, default=16,
        help="Frames per clip when --num-gen-frames is unset (default: 16)",
    )
    parser.add_argument(
        "--num-cond-frames", type=int, default=0,
        help="Skip this many leading conditioning frames in each saved mp4 "
        "(LongCat standard: 14). Matches online FVD gen-only window.",
    )
    parser.add_argument(
        "--num-gen-frames", type=int, default=None,
        help="Generated frames to evaluate after --num-cond-frames "
        "(LongCat standard: 14). Overrides --num-frames when set.",
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
    parser.add_argument(
        "--gt-cache", type=str, default=None,
        help="Optional precomputed GT .npz (from precompute_gt_features.py). "
        "When set, reference I3D stats are loaded from cache instead of "
        "--ref-dir, matching the online-FVD protocol used in headline sweeps.",
    )
    parser.add_argument(
        "--pair-by-id", action=argparse.BooleanOptionalAction, default=True,
        help="Pair gen/ref clips by canonical video id (panda_XXXX). "
        "Disable only for legacy sorted-order pairing.",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.gt_cache is None and not args.ref_dir:
        print("ERROR: --ref-dir is required unless --gt-cache is provided.",
              file=sys.stderr)
        sys.exit(1)

    t0 = time.time()

    # ------------------------------------------------------------------ discover videos
    gen_paths = sorted(Path(args.gen_dir).glob("*.mp4"))
    print(f"Found {len(gen_paths)} generated videos in {args.gen_dir}.")

    if not gen_paths:
        print("ERROR: No generated videos found.", file=sys.stderr)
        sys.exit(1)

    use_gt_cache = args.gt_cache is not None
    ref_paths: List[Path] = []
    pairs: List[Tuple[Path, Path]] = []
    if use_gt_cache:
        if not Path(args.gt_cache).exists():
            print(f"ERROR: GT cache not found: {args.gt_cache}", file=sys.stderr)
            sys.exit(1)
        print(f"Using GT feature cache: {args.gt_cache}")
    else:
        ref_paths = sorted(Path(args.ref_dir).glob("*.mp4"))
        print(f"Found {len(ref_paths)} reference videos in {args.ref_dir}.")
        if not ref_paths:
            print("ERROR: No reference videos found.", file=sys.stderr)
            sys.exit(1)
        if args.pair_by_id:
            pairs = _pair_videos_by_id(gen_paths, ref_paths)
        else:
            n = min(len(gen_paths), len(ref_paths))
            pairs = list(zip(gen_paths[:n], ref_paths[:n]))

    # ------------------------------------------------------------------ load videos
    print("Loading generated videos...")
    gen_tensors: List[torch.Tensor] = []
    gen_paths_valid: List[Path] = []
    if use_gt_cache:
        load_targets = gen_paths
    else:
        load_targets = [gp for gp, _ in pairs]

    load_kwargs = {
        "num_frames": args.num_frames,
        "num_cond_frames": args.num_cond_frames,
        "num_gen_frames": args.num_gen_frames,
    }
    for vp in load_targets:
        t = load_video_as_tensor(str(vp), **load_kwargs)
        if t is not None:
            gen_tensors.append(t)
            gen_paths_valid.append(vp)

    ref_tensors: List[torch.Tensor] = []
    if not use_gt_cache:
        print("Loading reference videos...")
        ref_by_gen = {gp: rp for gp, rp in pairs}
        for gp in gen_paths_valid:
            rp = ref_by_gen.get(gp)
            if rp is None:
                continue
            t = load_video_as_tensor(str(rp), **load_kwargs)
            if t is not None:
                ref_tensors.append(t)

    n_valid = len(gen_tensors) if use_gt_cache else min(len(gen_tensors), len(ref_tensors))
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
    if not use_gt_cache:
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

    ref_from_cache = False
    cache = None
    ref_count = 0
    if use_gt_cache:
        cache = np.load(args.gt_cache, allow_pickle=True)
        ref_sum = cache["ref_fvd_sum"].astype(np.float64)
        ref_cov = cache["ref_fvd_cov"].astype(np.float64)
        ref_count = int(cache["ref_fvd_count"])
        ref_from_cache = True
        print(f"GT cache reference: {ref_count} videos")
    else:
        print("Extracting I3D features (reference)...")
        feats_ref = extract_i3d_features(
            ref_tensors, i3d_model, args.device, args.batch_size,
        )
        print(f"  shape: {feats_ref.shape}")

    assert feats_gen.shape[1] == _I3D_FEATURE_DIM, (
        f"Expected {_I3D_FEATURE_DIM}-dim I3D features, got {feats_gen.shape[1]}"
    )

    # ------------------------------------------------------------------ FVD
    if use_gt_cache:
        gen_sum = feats_gen.sum(axis=0)
        gen_cov = feats_gen.T @ feats_gen
        gen_count = feats_gen.shape[0]
        fvd_score = compute_frechet_from_sufficient_stats(
            gen_sum, gen_cov, gen_count,
            ref_sum, ref_cov, ref_count,
        )
    else:
        fvd_score = compute_frechet_distance(feats_gen, feats_ref)
    print(f"\nFVD: {fvd_score:.4f}")

    # ---- FVD decomposition (diagnostic): mean-shift term vs covariance term --
    # FVD = ||mu_gen - mu_ref||^2  +  Tr(S_gen + S_ref - 2*sqrt(S_gen S_ref)).
    # A large mean_term => the generated feature distribution is systematically
    # SHIFTED from GT (e.g. a consistent TTA bias); a large trace_term => a
    # variance/covariance mismatch. This helps explain FVD moves that pixel PSNR
    # does not (PSNR is per-video; the mean_term is a pooled-distribution shift).
    fvd_mean_term: Optional[float] = None
    fvd_trace_term: Optional[float] = None
    try:
        if use_gt_cache:
            mu_g = gen_sum / gen_count
            sig_g = gen_cov / gen_count - np.outer(mu_g, mu_g)
            mu_r = ref_sum / ref_count
            sig_r = ref_cov / ref_count - np.outer(mu_r, mu_r)
        else:
            mu_g = np.mean(feats_gen.astype(np.float64), axis=0)
            sig_g = np.cov(feats_gen.astype(np.float64), rowvar=False)
            mu_r = np.mean(feats_ref.astype(np.float64), axis=0)
            sig_r = np.cov(feats_ref.astype(np.float64), rowvar=False)
        sig_g = sig_g + _COV_EPS * np.eye(sig_g.shape[0])
        sig_r = sig_r + _COV_EPS * np.eye(sig_r.shape[0])
        d = mu_g - mu_r
        fvd_mean_term = float(d @ d)
        covmean_d, _ = sqrtm(sig_g @ sig_r, disp=False)
        if np.iscomplexobj(covmean_d):
            covmean_d = covmean_d.real
        fvd_trace_term = float(np.trace(sig_g + sig_r - 2 * covmean_d))
        print(f"  decomposition: mean_term={fvd_mean_term:.4f}  "
              f"trace_term={fvd_trace_term:.4f}  "
              f"(mean_frac={fvd_mean_term / max(fvd_mean_term + fvd_trace_term, 1e-9):.2%})")
    except Exception as exc:  # noqa: BLE001
        print(f"  (FVD decomposition failed: {exc})", file=sys.stderr)

    result = {
        "fvd": round(fvd_score, 6),
        "fid": None,
        "num_gen_videos": len(gen_tensors),
        "num_ref_videos": ref_count if use_gt_cache else len(ref_tensors),
        "num_valid_pairs": n_valid,
        "num_frames_per_clip": (
            args.num_gen_frames if args.num_gen_frames is not None else args.num_frames
        ),
        "num_cond_frames": args.num_cond_frames,
        "num_gen_frames": args.num_gen_frames,
        "feature_extractor": "i3d_kinetics400_torchscript",
        "feature_dim": _I3D_FEATURE_DIM,
        "normalization": "[-1, 1]",
        "i3d_weights_sha256": i3d_hash,
        "sample_size_warning": sample_size_warning,
        "fvd_mean_term": round(fvd_mean_term, 6) if fvd_mean_term is not None else None,
        "fvd_trace_term": round(fvd_trace_term, 6) if fvd_trace_term is not None else None,
        "gen_dir": str(args.gen_dir),
        "ref_dir": str(args.ref_dir) if args.ref_dir else None,
        "gt_cache": str(args.gt_cache) if use_gt_cache else None,
        "pair_by_id": args.pair_by_id,
        "ref_from_gt_cache": ref_from_cache,
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

        if use_gt_cache and "ref_fid_sum" in cache:
            fid_gen_sum = fid_feats_gen.sum(axis=0)
            fid_gen_cov = fid_feats_gen.T @ fid_feats_gen
            fid_gen_frames = fid_feats_gen.shape[0]
            fid_score = compute_frechet_from_sufficient_stats(
                fid_gen_sum, fid_gen_cov, fid_gen_frames,
                cache["ref_fid_sum"].astype(np.float64),
                cache["ref_fid_cov"].astype(np.float64),
                int(cache["ref_fid_count"]),
            )
        elif not use_gt_cache:
            print("Extracting InceptionV3 features (reference frames)...")
            fid_feats_ref = extract_fid_features(
                ref_tensors, inception, args.device,
            )
            print(f"  shape: {fid_feats_ref.shape}")
            fid_score = compute_frechet_distance(fid_feats_gen, fid_feats_ref)
        else:
            fid_score = None
            print("WARNING: --compute-fid with --gt-cache but cache lacks FID stats",
                  file=sys.stderr)

        if fid_score is not None:
            print(f"FID: {fid_score:.4f}")
            result["fid"] = round(fid_score, 6)
            result["fid_feature_extractor"] = "inception_v3_imagenet"
            result["fid_feature_dim"] = _FID_FEATURE_DIM
            result["fid_num_frames_gen"] = fid_feats_gen.shape[0]
            if use_gt_cache and "ref_fid_count" in cache:
                result["fid_num_frames_ref"] = int(cache["ref_fid_count"])
            elif not use_gt_cache:
                result["fid_num_frames_ref"] = fid_feats_ref.shape[0]

    # ------------------------------------------------------------------ self-check
    if args.self_check:
        if use_gt_cache:
            print("\nSelf-consistency check skipped with --gt-cache "
                  "(ref distribution is fixed).", file=sys.stderr)
        else:
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
