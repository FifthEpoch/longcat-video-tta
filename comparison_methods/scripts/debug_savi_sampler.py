#!/usr/bin/env python3
"""Bounded debug for the SAVi-DNO LongCat sampler (why PSNR ~7, A==B).

On the SAME clips (metadata.csv, identical cond frames + prompt + geometry as the
real runs) it scores four things with the SAME metric (savi_dno_longcat.compute_metrics):

  [REF  ] standard LongCat pipeline (generate_video_continuation)        -> expect ~19
  [CUST0] SAVi custom differentiable sampler, CFG OFF (predict_no_optimize) -> expect ~7
  [CUST1] SAVi custom differentiable sampler, CFG ON  (CFG hypothesis)
  [probe] conditioning sensitivity: ||v(cond) - v(0)|| / ||v(cond)|| at a mid sigma
           ~0  => the custom _dit_forward_step IGNORES the context latents (smoking gun)
           >>0 => conditioning is wired; bug is sigma-sign / latent-norm / CFG

No noise optimization (irrelevant to the generation bug). Runs on 1 GPU.
Interpretation:
  REF~19 & CUST0==CUST1==~7 & probe~0  -> conditioning not applied in custom sampler.
  CUST1 >> CUST0                       -> CFG-off was the dominant culprit.
  REF also ~7                          -> data/geometry problem, not the sampler.
"""
import sys
import os
import csv
import argparse
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
sys.path.insert(0, str(_REPO_ROOT / "LongCat-Video"))
sys.path.insert(0, str(_REPO_ROOT / "delta_experiment" / "scripts"))
sys.path.insert(0, str(_REPO_ROOT))

from delta_experiment.scripts.common import (  # noqa: E402
    load_longcat_components, load_video_frames, encode_prompt,
    decode_latents, generate_video_continuation,
)
from comparison_methods.scripts.savi_dno_longcat import (  # noqa: E402
    SAViDNO_LongCat, compute_metrics,
)


def _gen_arm_np(pred_pixels: torch.Tensor) -> np.ndarray:
    """SAVi decode [1,C,T,H,W] in [0,1] -> [T,C,H,W]."""
    x = pred_pixels.squeeze(0).float().cpu().numpy()  # [C,T,H,W]
    return np.clip(x.transpose(1, 0, 2, 3), 0, 1)


def _score(pred_tchw, gt_tchw, tag):
    n = min(pred_tchw.shape[0], gt_tchw.shape[0])
    psnr, ssim = compute_metrics(pred_tchw[:n], gt_tchw[:n])
    print("    [%-6s] PSNR=%7.3f  SSIM=%.4f  (gen frames=%d)" % (tag, psnr, ssim, n))
    return psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--max-videos", type=int, default=2)
    p.add_argument("--num-cond-frames", type=int, default=14)
    p.add_argument("--num-frames", type=int, default=28)
    p.add_argument("--gen-start-frame", type=int, default=48)
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--resolution", type=str, default="480p")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    height, width = (480, 832) if args.resolution == "480p" else (720, 1280)
    num_gen = args.num_frames - args.num_cond_frames
    vae_t = 4

    print("=" * 72)
    print("SAVi-DNO sampler debug — REF vs custom(CFG off/on) + conditioning probe")
    print("  cond=%d gen=%d gen_start=%d steps=%d guidance=%.1f res=%s"
          % (args.num_cond_frames, num_gen, args.gen_start_frame,
             args.num_inference_steps, args.guidance_scale, args.resolution))
    print("=" * 72)

    comp = load_longcat_components(args.checkpoint_dir, device=device, dtype=torch.bfloat16)
    dit, vae, scheduler = comp["dit"], comp["vae"], comp["scheduler"]
    tok, te, pipe = comp["tokenizer"], comp["text_encoder"], comp["pipe"]
    for m in (dit, vae, te):
        for prm in m.parameters():
            prm.requires_grad = False
        m.eval()

    engine = SAViDNO_LongCat(
        dit=dit, vae=vae, scheduler=scheduler, tokenizer=tok, text_encoder=te,
        device=device, dtype=torch.bfloat16,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale, gradient_checkpointing=False,
    )

    sig = engine._build_sigmas()
    print("sigmas[:3]=%s ... sigmas[-3:]=%s (n=%d)"
          % ([round(float(s), 4) for s in sig[:3]],
             [round(float(s), 4) for s in sig[-3:]], len(sig)))

    with open(os.path.join(args.data_dir, "metadata.csv")) as f:
        video_list = list(csv.DictReader(f))
    if args.max_videos > 0:
        video_list = video_list[:args.max_videos]

    for idx, entry in enumerate(video_list):
        video_filename = entry.get("filename", entry.get("video_name", ""))
        video_path = os.path.join(args.data_dir, "videos", video_filename)
        caption = entry.get("caption", entry.get("prompt", ""))
        if not os.path.exists(video_path):
            print("\n[%d] MISSING %s" % (idx, video_filename))
            continue
        print("\n[%d] %s" % (idx, Path(video_filename).stem))

        cond_start = max(0, args.gen_start_frame - args.num_cond_frames)
        pixel_cond = load_video_frames(video_path, args.num_cond_frames, height=height,
                                       width=width, start_frame=cond_start).to(device, torch.bfloat16)
        pixel_gt = load_video_frames(video_path, num_gen, height=height, width=width,
                                     start_frame=args.gen_start_frame).to(device, torch.bfloat16)
        gt_tchw = np.clip(((pixel_gt.squeeze(0).float().cpu().numpy() + 1.0) / 2.0)
                          .transpose(1, 0, 2, 3), 0, 1)

        emb, mask = encode_prompt(tok, te, prompt=caption, device=device, dtype=torch.bfloat16)
        cond_lat = engine.encode(pixel_cond)
        T_gen_lat = 1 + (num_gen - 1) // vae_t
        tgt_shape = (1, cond_lat.shape[1], T_gen_lat, cond_lat.shape[3], cond_lat.shape[4])

        # [REF] standard pipeline — build cond PIL images exactly like run_lora_tta
        pf = ((pixel_cond.squeeze(0) + 1.0) / 2.0).clamp(0, 1)
        cond_imgs = [Image.fromarray((pf[:, t].permute(1, 2, 0).float().cpu().numpy() * 255)
                                     .astype(np.uint8)) for t in range(pf.shape[1])]
        try:
            ref = generate_video_continuation(
                pipe=pipe, video_frames=cond_imgs, prompt=caption,
                num_cond_frames=args.num_cond_frames, num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale, seed=args.seed + idx,
                resolution=args.resolution, device=device)
            ref = np.asarray(ref)  # [num_frames, H, W, 3] in [0,1]
            ref_gen = np.clip(ref[args.num_cond_frames:args.num_cond_frames + num_gen], 0, 1)
            _score(ref_gen.transpose(0, 3, 1, 2), gt_tchw, "REF")
        except Exception as ex:
            print("    [REF   ] FAILED: %s" % ex)

        # custom sampler — shared eps for CFG off/on
        gen_t = torch.Generator(device=device).manual_seed(args.seed + idx)
        eps = torch.randn(tgt_shape, device=device, dtype=torch.float32, generator=gen_t)
        for tag, use_cfg in (("CUST0", False), ("CUST1", True)):
            with torch.no_grad():
                z = engine._flow_euler_sample_differentiable(cond_lat, eps, emb, mask, use_cfg=use_cfg)
                pix = decode_latents(vae, z, denorm=True)
            _score(_gen_arm_np(pix), gt_tchw, tag)

        # conditioning-sensitivity probe at a mid sigma
        with torch.no_grad():
            t_mid = float(sig[len(sig) // 2].item())
            v_cond = engine._dit_forward_step(eps, cond_lat, t_mid, emb, mask).float()
            v_zero = engine._dit_forward_step(eps, torch.zeros_like(cond_lat), t_mid, emb, mask).float()
            rel = (v_cond - v_zero).norm().item() / (v_cond.norm().item() + 1e-8)
        verdict = "IGNORED (conditioning bug!)" if rel < 0.02 else "wired (bug elsewhere)"
        print("    [probe ] ||v(cond)-v(0)||/||v(cond)|| = %.4f @ sigma=%.3f -> conditioning %s"
              % (rel, t_mid, verdict))

    print("\n" + "=" * 72)
    print("REF~19 & CUST0==CUST1==~7 & probe~0 -> conditioning not applied in custom sampler.")
    print("CUST1 >> CUST0 -> CFG-off was the culprit.  REF also ~7 -> data/geometry issue.")
    print("=" * 72)


if __name__ == "__main__":
    main()
