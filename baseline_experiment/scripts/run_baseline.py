#!/usr/bin/env python3
"""
Baseline inference for LongCat-Video on Panda-70M videos.

For each video:
  1. Load frames at 15 fps
  2. Use first 2 frames as conditioning
  3. Generate 15 frames via video continuation (480p)
  4. Compare 14 generated frames vs ground truth
  5. Report PSNR, SSIM, LPIPS

Requires single GPU (H200). No torchrun needed — sets up distributed
env internally for compatibility with LongCat-Video's context parallel.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_LONGCAT_DIR = _REPO_ROOT / "LongCat-Video"
sys.path.insert(0, str(_LONGCAT_DIR))
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Single-GPU distributed init (needed by LongCat-Video internals)
# ---------------------------------------------------------------------------
def init_single_gpu():
    """Set up minimal distributed env for single-GPU inference."""
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))

    torch.cuda.set_device(0)

    from longcat_video.context_parallel.context_parallel_util import (
        init_context_parallel,
    )
    init_context_parallel(context_parallel_size=1, global_rank=0, world_size=1)


# ---------------------------------------------------------------------------
# Video frame loading
# ---------------------------------------------------------------------------
def load_video_frames_pil(
    video_path: str,
    num_frames: int = 17,
    target_fps: float = 15.0,
) -> list[Image.Image]:
    """Load video frames as PIL Images, subsampled to target_fps.

    Returns exactly `num_frames` PIL Images (RGB) at native resolution.
    """
    import av

    container = av.open(video_path)
    stream = container.streams.video[0]
    original_fps = float(stream.average_rate or stream.guessed_rate or 30)
    stride = max(1, round(original_fps / target_fps))

    all_frames = []
    for frame in container.decode(video=0):
        all_frames.append(frame)
        if len(all_frames) >= num_frames * stride + stride:
            break
    container.close()

    # Subsample
    frames = all_frames[::stride][:num_frames]

    # Convert to PIL
    pil_frames = [f.to_image() for f in frames]

    # Pad if not enough frames
    while len(pil_frames) < num_frames:
        pil_frames.append(pil_frames[-1])

    return pil_frames[:num_frames]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    """PSNR between two float arrays in [0, 1]."""
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse < 1e-10:
        return 60.0
    return float(10.0 * np.log10(1.0 / mse))


def ssim_single(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM for a single frame pair (H, W, C) in [0, 1]."""
    try:
        from skimage.metrics import structural_similarity
        return structural_similarity(pred, gt, channel_axis=2, data_range=1.0)
    except ImportError:
        # Simple fallback
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        mu_p, mu_g = pred.mean(), gt.mean()
        sig_p, sig_g = pred.var(), gt.var()
        sig_pg = ((pred - mu_p) * (gt - mu_g)).mean()
        num = (2 * mu_p * mu_g + c1) * (2 * sig_pg + c2)
        den = (mu_p ** 2 + mu_g ** 2 + c1) * (sig_p + sig_g + c2)
        return float(num / den)


def lpips_batch(pred_frames: np.ndarray, gt_frames: np.ndarray, device="cuda") -> float:
    """Average LPIPS over frame pairs. Arrays shape (T, H, W, 3) in [0,1]."""
    try:
        import lpips
    except ImportError:
        return float("nan")

    loss_fn = lpips.LPIPS(net="alex", verbose=False).to(device)
    scores = []
    for i in range(len(pred_frames)):
        p = torch.from_numpy(pred_frames[i]).permute(2, 0, 1).unsqueeze(0).float().to(device)
        g = torch.from_numpy(gt_frames[i]).permute(2, 0, 1).unsqueeze(0).float().to(device)
        # LPIPS expects [-1, 1]
        with torch.no_grad():
            s = loss_fn(p * 2 - 1, g * 2 - 1)
        scores.append(s.item())
    del loss_fn
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="LongCat-Video baseline inference")
    p.add_argument("--checkpoint-dir", type=str, required=True,
                   help="Path to LongCat-Video model weights")
    p.add_argument("--data-dir", type=str, required=True,
                   help="Path to panda_100 directory (contains videos/ and metadata.csv)")
    p.add_argument("--output-dir", type=str, required=True,
                   help="Where to save results")
    p.add_argument("--num-cond-frames", type=int, default=2,
                   help="Number of conditioning frames")
    p.add_argument("--num-gen-frames", type=int, default=14,
                   help="Number of frames to generate and evaluate")
    p.add_argument("--resolution", type=str, default="480p",
                   choices=["480p", "720p"])
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-videos", type=int, default=100)
    p.add_argument("--save-videos", action="store_true",
                   help="Save generated video frames as mp4")
    return p.parse_args()


def main():
    args = parse_args()

    # Directories
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Total frames needed: cond + gen + 1 (because num_frames must satisfy (n-1)%4==0)
    # For 2 cond + 14 gen = 16 → nearest valid = 17 (since (17-1)/4 = 4)
    num_total = args.num_cond_frames + args.num_gen_frames
    vae_temporal_factor = 4
    # Round up to valid: (n-1) % vae_temporal_factor == 0
    num_frames = ((num_total - 1 + vae_temporal_factor - 1) // vae_temporal_factor) * vae_temporal_factor + 1
    num_gen_actual = num_frames - args.num_cond_frames
    print(f"Frame config: {args.num_cond_frames} cond + {num_gen_actual} gen = {num_frames} total")
    print(f"  (evaluating first {args.num_gen_frames} of {num_gen_actual} generated frames)")

    # ── Init distributed ──────────────────────────────────────────────
    print("Initializing single-GPU distributed env...")
    init_single_gpu()

    # ── Load model ────────────────────────────────────────────────────
    print(f"Loading LongCat-Video from {args.checkpoint_dir} ...")
    t0 = time.time()

    from transformers import AutoTokenizer, UMT5EncoderModel
    from longcat_video.modules.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler,
    )
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
    from longcat_video.pipeline_longcat_video import LongCatVideoPipeline

    dtype = torch.bfloat16
    ckpt = args.checkpoint_dir

    tokenizer = AutoTokenizer.from_pretrained(ckpt, subfolder="tokenizer", torch_dtype=dtype)
    text_encoder = UMT5EncoderModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=dtype)
    vae = AutoencoderKLWan.from_pretrained(ckpt, subfolder="vae", torch_dtype=dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ckpt, subfolder="scheduler", torch_dtype=dtype)
    dit = LongCatVideoTransformer3DModel.from_pretrained(
        ckpt, subfolder="dit", cp_split_hw=[1, 1], torch_dtype=dtype
    )

    pipe = LongCatVideoPipeline(
        tokenizer=tokenizer, text_encoder=text_encoder,
        vae=vae, scheduler=scheduler, dit=dit,
    )
    pipe.to("cuda")

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB allocated")

    # ── Load video list ───────────────────────────────────────────────
    meta_csv = data_dir / "metadata.csv"
    videos_dir = data_dir / "videos"

    video_list = []
    if meta_csv.exists():
        with open(meta_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "")
                vp = videos_dir / fname
                if vp.exists():
                    caption = row.get("caption", "A video clip")
                    # Clean caption — if it's a list-like string, take first item
                    if caption.startswith("[") and "'" in caption:
                        try:
                            captions = eval(caption)
                            caption = captions[0] if captions else "A video clip"
                        except Exception:
                            pass
                    video_list.append({"path": str(vp), "caption": caption, "filename": fname})
    else:
        # Fallback: glob for mp4 files
        for vp in sorted(videos_dir.glob("*.mp4")):
            video_list.append({"path": str(vp), "caption": "A video clip", "filename": vp.name})

    video_list = video_list[:args.max_videos]
    print(f"Found {len(video_list)} videos")

    if not video_list:
        print("ERROR: No videos found!")
        sys.exit(1)

    # ── Inference loop ────────────────────────────────────────────────
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)

    results = []
    all_psnr, all_ssim, all_lpips = [], [], []

    for vi, entry in enumerate(video_list):
        video_path = entry["path"]
        caption = entry["caption"]
        filename = entry["filename"]

        print(f"\n[{vi+1}/{len(video_list)}] {filename}")

        try:
            # Load frames at native resolution (pipeline picks 480p bucket
            # based on the video's natural aspect ratio)
            pil_frames = load_video_frames_pil(video_path, num_frames=num_frames, target_fps=15.0)
            gt_pil = pil_frames[args.num_cond_frames:args.num_cond_frames + args.num_gen_frames]

            # Run video continuation
            t1 = time.time()
            output = pipe.generate_vc(
                video=pil_frames,
                prompt=caption,
                resolution=args.resolution,
                num_frames=num_frames,
                num_cond_frames=args.num_cond_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=generator,
                use_kv_cache=True,
                offload_kv_cache=True,  # offload KV cache to CPU to save GPU mem
            )[0]  # numpy array (num_frames, H, W, 3) in [0, 1]
            elapsed = time.time() - t1

            # Extract generated frames
            gen_frames = output[args.num_cond_frames:args.num_cond_frames + args.num_gen_frames]
            out_h, out_w = gen_frames.shape[1], gen_frames.shape[2]

            # Resize GT to match output resolution
            gt_resized = []
            for img in gt_pil:
                img_resized = img.resize((out_w, out_h), Image.BICUBIC)
                gt_resized.append(np.array(img_resized).astype(np.float64) / 255.0)
            gt_np = np.array(gt_resized)
            gen_np = gen_frames.astype(np.float64)

            # Compute metrics
            frame_psnrs = [psnr(gen_np[i], gt_np[i]) for i in range(len(gen_np))]
            frame_ssims = [ssim_single(gen_np[i].astype(np.float64), gt_np[i].astype(np.float64))
                           for i in range(len(gen_np))]
            vid_psnr = float(np.mean(frame_psnrs))
            vid_ssim = float(np.mean(frame_ssims))
            vid_lpips = lpips_batch(gen_np.astype(np.float32), gt_np.astype(np.float32))

            all_psnr.append(vid_psnr)
            all_ssim.append(vid_ssim)
            all_lpips.append(vid_lpips)

            entry_result = {
                "index": vi,
                "filename": filename,
                "psnr": round(vid_psnr, 4),
                "ssim": round(vid_ssim, 4),
                "lpips": round(vid_lpips, 4),
                "resolution": f"{out_h}x{out_w}",
                "time_s": round(elapsed, 1),
            }
            results.append(entry_result)

            print(f"  PSNR={vid_psnr:.2f}  SSIM={vid_ssim:.4f}  LPIPS={vid_lpips:.4f}  "
                  f"({out_h}x{out_w}, {elapsed:.1f}s)")

            if vi == 0:
                peak = torch.cuda.max_memory_allocated() / 1e9
                print(f"  *** Peak GPU memory after 1st video: {peak:.1f} GB ***")

            # Optionally save generated video
            if args.save_videos:
                save_dir = output_dir / "generated_videos"
                save_dir.mkdir(exist_ok=True)
                out_uint8 = (gen_frames * 255).clip(0, 255).astype(np.uint8)
                out_tensor = torch.from_numpy(out_uint8)
                from torchvision.io import write_video
                write_video(
                    str(save_dir / filename),
                    out_tensor, fps=15,
                    video_codec="libx264", options={"crf": "18"},
                )

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "index": vi, "filename": filename,
                "psnr": None, "ssim": None, "lpips": None,
                "error": str(e),
            })

        # Clean up GPU memory between videos
        gc.collect()
        torch.cuda.empty_cache()

    # ── Save results ──────────────────────────────────────────────────
    # Per-video CSV
    csv_path = output_dir / "per_video_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["index", "filename", "psnr", "ssim", "lpips", "resolution", "time_s"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    # Summary JSON
    summary = {
        "experiment": "baseline_inference",
        "model": "LongCat-Video",
        "checkpoint_dir": args.checkpoint_dir,
        "resolution": args.resolution,
        "num_cond_frames": args.num_cond_frames,
        "num_gen_frames": args.num_gen_frames,
        "num_frames_total": num_frames,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "num_videos": len(video_list),
        "num_successful": len(all_psnr),
        "metrics": {
            "psnr": {"mean": round(np.mean(all_psnr), 4), "std": round(np.std(all_psnr), 4),
                     "min": round(min(all_psnr), 4), "max": round(max(all_psnr), 4)} if all_psnr else {},
            "ssim": {"mean": round(np.mean(all_ssim), 4), "std": round(np.std(all_ssim), 4),
                     "min": round(min(all_ssim), 4), "max": round(max(all_ssim), 4)} if all_ssim else {},
            "lpips": {"mean": round(np.mean(all_lpips), 4), "std": round(np.std(all_lpips), 4),
                      "min": round(min(all_lpips), 4), "max": round(max(all_lpips), 4)} if all_lpips else {},
        },
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("BASELINE INFERENCE COMPLETE")
    print("=" * 60)
    if all_psnr:
        print(f"  Videos:  {len(all_psnr)}/{len(video_list)}")
        print(f"  PSNR:    {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f}")
        print(f"  SSIM:    {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
        print(f"  LPIPS:   {np.mean(all_lpips):.4f} ± {np.std(all_lpips):.4f}")
    print(f"\n  CSV:     {csv_path}")
    print(f"  Summary: {summary_path}")

    # Cleanup distributed
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
