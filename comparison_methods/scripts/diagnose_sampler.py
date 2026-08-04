#!/usr/bin/env python3
"""Localize WHY the differentiable LongCat sampler produces garbage (PSNR ~9 dB,
SSIM ~0.05) while the real LongCat pipeline gets ~19 dB on the same clips.

The noise-opt smoke runs produce near-identical output with CFG on/off and with
10 vs 50 Euler steps -> the sampler is not actually denoising (output ≈ decoded
initial noise). This script isolates the failing layer by comparing, on the SAME
2 videos and the SAME [cond|gen] window used by savi_dno_longcat.py:

  (1) VAE ceiling (GT)   : decode(encode(gt_future))    vs gt_future
  (2) VAE ceiling (cond) : decode(encode(cond))         vs cond
        -> tests the EXACT encode_video/decode_latents + normalization path savi
           uses. If low (~9 dB), the bug is the VAE round-trip / latent
           normalization and everything downstream is doomed.
  (3) real pipe.generate_vc : the ~19 dB LongCat path, on this window
        -> printed under BOTH frame alignments (with/without cond prefix) so a
           cond-inclusion offset can't be mistaken for a failure. If high, the
           protocol/window/GT is fine and savi's own sampler is the bug.
  (4) savi predict_no_optimize (CFG+50) : reproduces the ~9 dB
        -> confirms the failure is in the reimplemented sampler, not the data.

Run on a GPU node, e.g.:
  python3 comparison_methods/scripts/diagnose_sampler.py \
    --checkpoint-dir /scratch/wc3013/longcat-video-checkpoints \
    --data-dir datasets/panda_ood_budget_1000v_preview_480p \
    --num-videos 2
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "LongCat-Video"))
sys.path.insert(0, str(_REPO_ROOT))

from delta_experiment.scripts.common import (  # noqa: E402
    load_longcat_components,
    encode_video,
    decode_latents,
    encode_prompt,
    load_video_frames,
    generate_video_continuation,
)


def to_thwc(video_5d: torch.Tensor) -> np.ndarray:
    """[1, C, T, H, W] in [0,1] -> np [T, H, W, C] float32."""
    v = video_5d.squeeze(0).float().cpu().numpy()  # [C,T,H,W]
    return np.transpose(v, (1, 2, 3, 0))


def pixels_to_pil(t_5d: torch.Tensor):
    """[1, C, T, H, W] in [-1,1] -> list of PIL.Image."""
    from PIL import Image
    v = t_5d.squeeze(0)  # [C,T,H,W]
    arr = ((v.permute(1, 2, 3, 0).float().cpu().numpy() + 1.0) / 2.0)
    arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)  # [T,H,W,C]
    return [Image.fromarray(f) for f in arr]


def psnr_ssim(pred_thwc: np.ndarray, gt_thwc: np.ndarray):
    n = min(pred_thwc.shape[0], gt_thwc.shape[0])
    ps, ss = [], []
    for i in range(n):
        p = np.clip(pred_thwc[i], 0, 1)
        g = np.clip(gt_thwc[i], 0, 1)
        ps.append(skimage_psnr(g, p, data_range=1.0))
        ss.append(skimage_ssim(g, p, data_range=1.0, channel_axis=2))
    return float(np.mean(ps)), float(np.mean(ss)), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--num-videos", type=int, default=2)
    ap.add_argument("--num-cond-frames", type=int, default=14)
    ap.add_argument("--num-gen-frames", type=int, default=14)
    ap.add_argument("--gen-start-frame", type=int, default=48)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=4.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("Loading LongCat components...", flush=True)
    comp = load_longcat_components(args.checkpoint_dir, device=device, dtype=dtype)
    vae = comp["vae"]
    dit = comp["dit"]
    scheduler = comp["scheduler"]
    tokenizer = comp["tokenizer"]
    text_encoder = comp["text_encoder"]
    pipe = comp["pipe"]

    # savi engine for path (4)
    from comparison_methods.scripts.savi_dno_longcat import SAViDNO_LongCat
    savi = SAViDNO_LongCat(
        dit=dit, vae=vae, scheduler=scheduler, tokenizer=tokenizer,
        text_encoder=text_encoder, device=device, dtype=dtype,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generation_use_cfg=True, generation_steps=args.num_inference_steps,
    )

    with open(os.path.join(args.data_dir, "metadata.csv")) as f:
        rows = list(csv.DictReader(f))[: args.num_videos]

    vae_t = 4
    cond_start = args.gen_start_frame - args.num_cond_frames

    for i, entry in enumerate(rows):
        fn = entry.get("filename", entry.get("video_name", ""))
        caption = entry.get("caption", entry.get("prompt", ""))
        vp = os.path.join(args.data_dir, "videos", fn)
        print(f"\n================ video {i}: {fn} ================", flush=True)
        if not os.path.exists(vp):
            print("  MISSING, skipping"); continue

        cond = load_video_frames(vp, args.num_cond_frames, height=args.height,
                                 width=args.width, start_frame=max(0, cond_start)).to(device, dtype)
        gt = load_video_frames(vp, args.num_gen_frames, height=args.height,
                               width=args.width, start_frame=args.gen_start_frame).to(device, dtype)
        gt01 = to_thwc((gt + 1.0) / 2.0)
        cond01 = to_thwc((cond + 1.0) / 2.0)

        # (1) VAE ceiling on GT future  (exact savi encode/decode path)
        z_gt = encode_video(vae, gt, normalize=True)
        rec_gt = decode_latents(vae, z_gt, denorm=True)  # [0,1]
        p, s, n = psnr_ssim(to_thwc(rec_gt), gt01)
        print(f"  (1) VAE ceiling  GT : PSNR {p:6.2f}  SSIM {s:.4f}  (N={n})  z_gt shape {tuple(z_gt.shape)}")

        # (2) VAE ceiling on cond
        z_c = encode_video(vae, cond, normalize=True)
        rec_c = decode_latents(vae, z_c, denorm=True)
        p, s, n = psnr_ssim(to_thwc(rec_c), cond01)
        print(f"  (2) VAE ceiling cond: PSNR {p:6.2f}  SSIM {s:.4f}  (N={n})")

        # (3) real pipeline generate_vc on this window (both alignments)
        try:
            gen_frames = max(args.num_gen_frames, 17)
            out = generate_video_continuation(
                pipe, pixels_to_pil(cond), caption,
                num_cond_frames=args.num_cond_frames, num_frames=gen_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, seed=42,
                resolution="480p", device=device,
            )
            out = np.asarray(out, dtype=np.float32)
            if out.max() > 1.5:
                out = out / 255.0
            print(f"      real generate_vc output shape {out.shape}")
            pa, sa, na = psnr_ssim(out, gt01)                          # align: out[0:]=gen
            nc = args.num_cond_frames
            pb, sb, nb = psnr_ssim(out[nc:] if out.shape[0] > nc else out, gt01)  # align: skip cond prefix
            print(f"  (3) real generate_vc: PSNR {pa:6.2f}/{pb:6.2f}  SSIM {sa:.4f}/{sb:.4f}  "
                  f"(no-offset / skip-{nc}-cond)")
        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  (3) real generate_vc: FAILED -> {e}")
            traceback.print_exc()

        # (4) savi reimplemented sampler (CFG + N steps), no optimization
        savi.reset()
        pe, pm = encode_prompt(tokenizer, text_encoder, prompt=caption,
                               device=device, dtype=dtype)
        cond_latents = savi.encode(cond)
        T_gen_latent = 1 + (args.num_gen_frames - 1) // vae_t
        tgt_shape = (1, cond_latents.shape[1], T_gen_latent,
                     cond_latents.shape[3], cond_latents.shape[4])
        pred = savi.predict_no_optimize(cond_latents, tgt_shape, pe, pm)
        p, s, n = psnr_ssim(to_thwc(pred), gt01)
        print(f"  (4) savi sampler    : PSNR {p:6.2f}  SSIM {s:.4f}  (N={n})  pred shape {tuple(pred.shape)}")

    print("\n---- READ ----")
    print("  (1) low  -> VAE round-trip / latent normalization is broken (fix first).")
    print("  (1) high + (3) high + (4) low -> savi reimplemented sampler is the bug.")
    print("  (1) high + (3) low            -> the [cond|gen] window/protocol/GT is off.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
