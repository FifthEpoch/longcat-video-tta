#!/usr/bin/env python3
"""
Pre-compute and cache I3D (FVD) and InceptionV3 (FID) features for ground
truth video frames so the reference distribution is computed once and reused.

Supports three modes:
  longcat  - LongCat 480p pipeline  (frames via PyAV, [T,H,W,3])
  pvdm     - PVDM/SAVi-DNO 256x256 (frames via imageio, [T,C,H,W])
  dfot     - DFoT 128x128           (frames via imageio, [T,C,H,W])

Output: .npz with per-video features AND pre-aggregated sufficient statistics.
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_I3D_HF_REPO = "kiwhansong/DFoT"
_I3D_HF_FILE = "metrics_models/i3d_torchscript.pt"
_I3D_FEATURE_DIM = 400
_FID_FEATURE_DIM = 2048
_MIN_I3D_FRAMES = 9


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


def _np_to_pil_resize_tensor(frame_hwc, size):
    """Convert HWC float32 frame to resized+cropped tensor."""
    from torchvision.transforms import functional as TF
    from PIL import Image
    img = Image.fromarray((np.clip(frame_hwc, 0, 1) * 255).astype(np.uint8))
    img = TF.resize(img, size, interpolation=TF.InterpolationMode.BILINEAR)
    img = TF.center_crop(img, size)
    return img


def extract_i3d_features(i3d_model, frames_hwc, device, size=224):
    """frames_hwc: [T, H, W, 3] float32 [0,1] -> 400-dim float64."""
    from torchvision.transforms import functional as TF
    tensors = []
    for i in range(frames_hwc.shape[0]):
        img = _np_to_pil_resize_tensor(frames_hwc[i], size)
        tensors.append(TF.to_tensor(img))
    clip = torch.stack(tensors, dim=0).unsqueeze(0)

    clip = pad_for_i3d(clip.to(device))
    clip = torch.clamp(2.0 * clip - 1.0, -1.0, 1.0)
    clip = clip.permute(0, 2, 1, 3, 4).contiguous()
    with torch.no_grad():
        feats = i3d_model(clip, rescale=False, resize=True, return_features=True)
    return feats.cpu().to(torch.float64).numpy().squeeze(0)


def extract_inception_features(incep_model, frames_hwc, device):
    """frames_hwc: [T, H, W, 3] float32 [0,1] -> [T, 2048] float64."""
    from torchvision.transforms import functional as TF
    feats_list = []
    with torch.no_grad():
        for i in range(frames_hwc.shape[0]):
            img = _np_to_pil_resize_tensor(frames_hwc[i], 299)
            t = TF.normalize(
                TF.to_tensor(img),
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ).unsqueeze(0).to(device)
            f = incep_model(t).cpu().to(torch.float64).numpy()
            feats_list.append(f)
    return np.concatenate(feats_list, axis=0)


def ensure_hwc(frames_np, layout):
    """Ensure frames are [T, H, W, 3] regardless of input layout."""
    if layout == "chw":
        return np.transpose(frames_np, (0, 2, 3, 1))
    return frames_np


def load_gt_frames_longcat(video_path, gen_start_frame, num_gen_frames, target_size=None):
    """Returns [T, H, W, 3] float32 in [0,1] or None."""
    import av
    from PIL import Image

    try:
        container = av.open(str(video_path))
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
    except Exception as e:
        print("  WARNING: Failed to decode %s: %s" % (video_path, e))
        return None

    if len(gt_pil) < num_gen_frames:
        return None

    if target_size is not None:
        out_w, out_h = target_size
        gt_np = np.stack([
            np.array(img.resize((out_w, out_h), Image.LANCZOS)) / 255.0
            for img in gt_pil[:num_gen_frames]
        ], axis=0).astype(np.float32)
    else:
        gt_np = np.stack([
            np.array(img) / 255.0
            for img in gt_pil[:num_gen_frames]
        ], axis=0).astype(np.float32)

    return gt_np


def load_gt_frames_pvdm(video_path, gt_start=16, gt_end=32, size=256):
    """Returns [T, C, H, W] float32 in [0,1] or None."""
    from PIL import Image
    import torchvision.transforms as T

    try:
        import imageio.v3 as iio
        reader = iio.imread(str(video_path), plugin="pyav")
    except Exception:
        try:
            import imageio
            reader = imageio.mimread(str(video_path), memtest=False)
        except Exception as e:
            print("  WARNING: Failed to load %s: %s" % (video_path, e))
            return None

    frames = []
    for i, frame in enumerate(reader):
        if len(frames) >= gt_end:
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

    if len(frames) < gt_end:
        return None

    gt_frames = torch.stack(frames[gt_start:gt_end])
    return gt_frames.numpy().astype(np.float32)


def load_gt_frames_dfot(video_path, context_length=5, pred_length=12, size=128):
    """Returns [T, C, H, W] float32 in [0,1] or None."""
    from PIL import Image
    import torchvision.transforms as T

    num_needed = context_length + pred_length

    try:
        import imageio.v3 as iio
        reader = iio.imread(str(video_path), plugin="pyav")
    except Exception:
        try:
            import imageio
            reader = imageio.mimread(str(video_path), memtest=False)
        except Exception as e:
            print("  WARNING: Failed to load %s: %s" % (video_path, e))
            return None

    frames = []
    for i, frame in enumerate(reader):
        if len(frames) >= num_needed:
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

    if len(frames) < num_needed:
        return None

    gt_frames = torch.stack(frames[context_length:context_length + pred_length])
    return gt_frames.numpy().astype(np.float32)


def load_video_list(data_dir, mode):
    data_dir = Path(data_dir)
    entries = []

    if mode == "pvdm":
        mapping_csv = data_dir / "mapping.csv"
        if mapping_csv.exists():
            with open(mapping_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pvdm_path = data_dir / row.get("pvdm_path", row.get("filename", ""))
                    name = Path(row.get("original_filename",
                                        row.get("filename", ""))).stem
                    entries.append((name, str(pvdm_path)))
        else:
            for mp4 in sorted((data_dir / "test").glob("*.mp4")):
                entries.append((mp4.stem, str(mp4)))

    elif mode == "dfot":
        mapping_csv = data_dir / "mapping.csv"
        if mapping_csv.exists():
            with open(mapping_csv) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dfot_fn = row.get("dfot_filename", row.get("filename", ""))
                    name = Path(row.get("original_filename",
                                        row.get("filename", ""))).stem
                    video_path = data_dir / "test" / dfot_fn
                    entries.append((name, str(video_path)))
        else:
            for mp4 in sorted((data_dir / "test").glob("*.mp4")):
                entries.append((mp4.stem, str(mp4)))

    else:
        meta_path = data_dir / "metadata.csv"
        if meta_path.exists():
            with open(meta_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename", row.get("video_path", ""))
                    vp = data_dir / "videos" / fname
                    if not vp.exists():
                        vp = data_dir / fname
                    name = Path(fname).stem
                    entries.append((name, str(vp)))
        else:
            vid_dir = data_dir / "videos"
            if vid_dir.exists():
                for mp4 in sorted(vid_dir.glob("*.mp4")):
                    entries.append((mp4.stem, str(mp4)))

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute GT I3D/Inception features for FVD/FID caching")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--mode", choices=["longcat", "pvdm", "dfot"],
                        default="longcat")
    parser.add_argument("--device", default="cuda")

    grp_lc = parser.add_argument_group("LongCat mode options")
    grp_lc.add_argument("--gen-start-frame", type=int, default=48)
    grp_lc.add_argument("--num-gen-frames", type=int, default=14)
    grp_lc.add_argument("--num-cond-frames", type=int, default=14)
    grp_lc.add_argument("--target-width", type=int, default=None)
    grp_lc.add_argument("--target-height", type=int, default=None)

    grp_pvdm = parser.add_argument_group("PVDM mode options")
    grp_pvdm.add_argument("--pvdm-gt-start", type=int, default=16)
    grp_pvdm.add_argument("--pvdm-gt-end", type=int, default=32)
    grp_pvdm.add_argument("--pvdm-size", type=int, default=256)

    grp_dfot = parser.add_argument_group("DFoT mode options")
    grp_dfot.add_argument("--context-length", type=int, default=5)
    grp_dfot.add_argument("--pred-length", type=int, default=12)
    grp_dfot.add_argument("--dfot-size", type=int, default=128)

    args = parser.parse_args()

    if not torch.cuda.is_available() and args.device == "cuda":
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    target_size = None
    if args.mode == "longcat" and args.target_width and args.target_height:
        target_size = (args.target_width, args.target_height)

    print("=" * 60)
    print("Pre-computing GT features")
    print("  Mode:     %s" % args.mode)
    print("  Data dir: %s" % args.data_dir)
    print("  Output:   %s" % args.output)
    print("  Device:   %s" % args.device)
    if args.mode == "longcat":
        print("  gen_start_frame: %d" % args.gen_start_frame)
        print("  num_gen_frames:  %d" % args.num_gen_frames)
        if target_size:
            print("  target_size: %dx%d" % target_size)
    elif args.mode == "pvdm":
        print("  GT frames: [%d, %d)" % (args.pvdm_gt_start, args.pvdm_gt_end))
        print("  Resolution: %dx%d" % (args.pvdm_size, args.pvdm_size))
    elif args.mode == "dfot":
        print("  context_length: %d" % args.context_length)
        print("  pred_length:    %d" % args.pred_length)
        print("  Resolution: %dx%d" % (args.dfot_size, args.dfot_size))
    print("=" * 60)

    video_list = load_video_list(args.data_dir, args.mode)
    print("Found %d videos" % len(video_list))

    if not video_list:
        print("ERROR: No videos found!", file=sys.stderr)
        sys.exit(1)

    print("Loading I3D model...")
    i3d_model = load_i3d_model(args.device)
    print("Loading InceptionV3 model...")
    incep_model = load_inception_model(args.device)

    is_chw = args.mode in ("pvdm", "dfot")

    video_names = []
    all_i3d_feats = []
    all_incep_feats = []
    incep_frames_per_video = []
    skipped = 0

    t0 = time.time()
    for idx, (name, vpath) in enumerate(video_list):
        if args.mode == "longcat":
            gt_frames = load_gt_frames_longcat(
                vpath, args.gen_start_frame, args.num_gen_frames,
                target_size=target_size)
        elif args.mode == "pvdm":
            gt_frames = load_gt_frames_pvdm(
                vpath, args.pvdm_gt_start, args.pvdm_gt_end, args.pvdm_size)
        elif args.mode == "dfot":
            gt_frames = load_gt_frames_dfot(
                vpath, args.context_length, args.pred_length, args.dfot_size)

        if gt_frames is None:
            skipped += 1
            if skipped <= 5:
                print("  SKIP [%d]: %s" % (idx, name))
            continue

        frames_hwc = ensure_hwc(gt_frames, "chw" if is_chw else "hwc")

        i3d_feat = extract_i3d_features(i3d_model, frames_hwc, args.device)
        incep_feat = extract_inception_features(incep_model, frames_hwc, args.device)

        video_names.append(name)
        all_i3d_feats.append(i3d_feat)
        all_incep_feats.append(incep_feat)
        incep_frames_per_video.append(incep_feat.shape[0])

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            print("  [%d/%d] %.1f videos/sec  (skipped %d)" %
                  (idx + 1, len(video_list), rate, skipped))

    elapsed = time.time() - t0
    n_ok = len(video_names)
    print("\nExtraction complete: %d/%d videos in %.1f sec (skipped %d)" %
          (n_ok, len(video_list), elapsed, skipped))

    if n_ok == 0:
        print("ERROR: No features extracted!", file=sys.stderr)
        sys.exit(1)

    i3d_arr = np.stack(all_i3d_feats)
    incep_arr = np.concatenate(all_incep_feats, axis=0)
    incep_fpv = np.array(incep_frames_per_video, dtype=np.int32)

    ref_fvd_sum = i3d_arr.sum(axis=0)
    ref_fvd_cov = i3d_arr.T @ i3d_arr
    ref_fvd_count = n_ok

    ref_fid_sum = incep_arr.sum(axis=0)
    ref_fid_cov = incep_arr.T @ incep_arr
    ref_fid_count = incep_arr.shape[0]

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "mode": args.mode,
        "data_dir": args.data_dir,
        "num_videos": n_ok,
        "num_skipped": skipped,
        "total_inception_frames": int(ref_fid_count),
    }
    if args.mode == "longcat":
        metadata["gen_start_frame"] = args.gen_start_frame
        metadata["num_gen_frames"] = args.num_gen_frames
        metadata["num_cond_frames"] = args.num_cond_frames
        if target_size:
            metadata["target_size"] = list(target_size)
    elif args.mode == "pvdm":
        metadata["gt_start"] = args.pvdm_gt_start
        metadata["gt_end"] = args.pvdm_gt_end
        metadata["resolution"] = args.pvdm_size
    elif args.mode == "dfot":
        metadata["context_length"] = args.context_length
        metadata["pred_length"] = args.pred_length
        metadata["resolution"] = args.dfot_size

    np.savez(
        args.output,
        video_names=np.array(video_names, dtype=object),
        i3d_features=i3d_arr,
        inception_features=incep_arr,
        inception_frames_per_video=incep_fpv,
        ref_fvd_sum=ref_fvd_sum,
        ref_fvd_cov=ref_fvd_cov,
        ref_fvd_count=np.array(ref_fvd_count, dtype=np.int64),
        ref_fid_sum=ref_fid_sum,
        ref_fid_cov=ref_fid_cov,
        ref_fid_count=np.array(ref_fid_count, dtype=np.int64),
        metadata=np.array(json.dumps(metadata)),
    )

    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print("\nSaved: %s (%.1f MB)" % (args.output, file_size_mb))
    print("  Videos:           %d" % n_ok)
    print("  I3D features:     [%d, %d]" % i3d_arr.shape)
    print("  Inception frames: %d total" % ref_fid_count)
    print("  ref_fvd_sum:      [%d]" % ref_fvd_sum.shape[0])
    print("  ref_fvd_cov:      [%d, %d]" % ref_fvd_cov.shape)
    print("  ref_fid_sum:      [%d]" % ref_fid_sum.shape[0])
    print("  ref_fid_cov:      [%d, %d]" % ref_fid_cov.shape)


if __name__ == "__main__":
    main()
