#!/usr/bin/env python3
"""
Pathwise Test-Time Correction (TTC) on the LongCat-Video backbone.

LONG-HORIZON COMPARISON ONLY. TTC is a *training-free*, sampling-space
intervention for autoregressive / long video generation: it suppresses
error accumulation ("appearance drift") by periodically RE-ANCHORING the
appearance of the sampled trajectory to the first frame during the low-noise
appearance-refinement steps of the diffusion sampler. No model parameters are
updated (contrast with AdaSteer / LoRA-TTA parameter-space adaptation).

This makes TTC only meaningful for LONG horizons where drift accumulates. We
therefore run it in the long-context Panda setting (93 frames = 14 cond + 79
gen @ gen_start=14), matching `panda_longctx_1000v`, and DO NOT report it for
the short-horizon (14+14) table.

Implementation notes / honesty caveat:
  The original TTC paper targets T2V long generation. Here we adapt its core
  mechanism -- pathwise appearance re-anchoring during low-noise steps -- to
  LongCat frame-conditioned continuation. Because LongCat uses a deterministic
  flow-matching Euler sampler (not a DDPM), the correction is applied to the
  running x0 estimate:

      x0_hat   = x_t - sigma * v(x_t)                    (flow-matching x0)
      x0_corr  = re_anchor(x0_hat, anchor)               (appearance only)
      v_corr   = (x_t - x0_corr) / sigma
      x_{t+1}  = x_t + dt * v_corr

  applied only when sigma <= --ttc-sigma-threshold (the low-noise appearance
  band) and every --ttc-cadence steps. "Appearance only" (default) shifts each
  frame's per-channel spatial-mean statistics toward the first-frame anchor,
  leaving spatial structure / motion untouched; --ttc-full-latent blends the
  whole latent instead. Correction strength = --ttc-weight.

  These knobs are exposed because the exact TTC schedule must be tuned/validated
  for this backbone; defaults are a sensible starting point, not the paper's
  verbatim constants.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_LONGCAT_DIR = _REPO_ROOT / "LongCat-Video"
_DELTA_SCRIPTS = _REPO_ROOT / "delta_experiment" / "scripts"
sys.path.insert(0, str(_LONGCAT_DIR))
sys.path.insert(0, str(_DELTA_SCRIPTS))
sys.path.insert(0, str(_REPO_ROOT))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    decode_latents,
    load_ucf101_video_list,
    evaluate_generation_metrics,
    save_results,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    apply_fixed_caption,
    add_online_eval_args,
    OnlineFrechetAccumulator,
    finalize_online_eval,
    aggregate_quality_metrics,
)

# Reuse the SAVi-DNO LongCat sampling engine (DiT forward + sigma schedule +
# CFG) so TTC shares the exact conditioning semantics as the other baselines.
from comparison_methods.scripts.savi_dno_longcat import SAViDNO_LongCat


class TTC_LongCat:
    """Pathwise Test-Time Correction sampler on LongCat flow-matching Euler."""

    def __init__(self, engine: SAViDNO_LongCat,
                 sigma_threshold: float = 0.3,
                 cadence: int = 1,
                 weight: float = 0.1,
                 appearance_only: bool = True,
                 use_cfg: bool = True):
        self.e = engine
        self.sigma_threshold = sigma_threshold
        self.cadence = max(1, cadence)
        self.weight = weight
        self.appearance_only = appearance_only
        self.use_cfg = use_cfg

    def _re_anchor(self, x0: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """Shift x0's appearance toward the first-frame anchor.

        anchor : [B, C, 1, H, W] (first cond latent frame), broadcast over T.
        appearance_only: match per-(frame,channel) spatial mean; else blend the
        whole latent.
        """
        w = self.weight
        if self.appearance_only:
            x0_mean = x0.mean(dim=(3, 4), keepdim=True)
            anchor_mean = anchor.mean(dim=(3, 4), keepdim=True)
            return x0 + w * (anchor_mean - x0_mean)
        return (1.0 - w) * x0 + w * anchor

    @torch.no_grad()
    def sample(self, cond_latents, target_shape, prompt_embeds, prompt_mask,
               generator=None):
        """Euler flow sampling with periodic low-noise appearance re-anchoring."""
        e = self.e
        sigmas = e._build_sigmas()
        eps = torch.randn(target_shape, device=e.device, dtype=torch.float32,
                          generator=generator)
        x_t = eps
        # First-frame anchor, broadcast across the target temporal dim.
        anchor = cond_latents[:, :, :1].to(torch.float32).expand(
            -1, -1, target_shape[2], -1, -1
        )
        step_fn = e._dit_forward_step_cfg if self.use_cfg else e._dit_forward_step

        n_corr = 0
        for i in range(len(sigmas) - 1):
            t_curr = sigmas[i].item()
            t_next = sigmas[i + 1].item()
            dt = t_next - t_curr

            v = step_fn(x_t, cond_latents, t_curr, prompt_embeds, prompt_mask)
            v = v.to(x_t.dtype)

            if (t_curr <= self.sigma_threshold and t_curr > 1e-6
                    and (i % self.cadence == 0) and self.weight > 0):
                x0 = x_t - t_curr * v
                x0 = self._re_anchor(x0, anchor)
                v = (x_t - x0) / t_curr
                n_corr += 1

            x_t = x_t + dt * v

        pred_pixels = decode_latents(e.vae, x_t, denorm=True)
        return pred_pixels, n_corr


def _to_hwc_uint01(pred_pixels: torch.Tensor) -> np.ndarray:
    """[1,C,T,H,W] in [0,1] -> [T,H,W,C] float in [0,1]."""
    x = pred_pixels.squeeze(0).float().cpu().numpy()   # [C,T,H,W]
    return np.clip(x.transpose(1, 2, 3, 0), 0, 1)      # [T,H,W,C]


def main():
    p = argparse.ArgumentParser(description="TTC (pathwise correction) on LongCat -- long horizon only")
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--max-videos", type=int, default=1000)
    p.add_argument("--start-video-idx", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # Long-horizon geometry (matches panda_longctx_1000v)
    p.add_argument("--num-cond-frames", type=int, default=14)
    p.add_argument("--num-frames", type=int, default=93)
    p.add_argument("--gen-start-frame", type=int, default=14)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--resolution", type=str, default="480p")

    # TTC knobs
    p.add_argument("--ttc-sigma-threshold", type=float, default=0.3,
                   help="Apply correction only when sigma <= this (low-noise band).")
    p.add_argument("--ttc-cadence", type=int, default=1,
                   help="Apply correction every N low-noise steps.")
    p.add_argument("--ttc-weight", type=float, default=0.1,
                   help="Correction strength toward the first-frame anchor (0=off).")
    p.add_argument("--ttc-full-latent", action="store_true",
                   help="Blend the full latent instead of appearance-only mean shift.")
    p.add_argument("--no-cfg", action="store_true",
                   help="Disable classifier-free guidance during sampling.")
    p.add_argument("--no-correction", action="store_true",
                   help="Baseline: run the same custom Euler sampler with NO correction.")

    p.add_argument("--no-save-videos", action="store_true")
    p.add_argument("--save-only-list", type=str, default=None)
    p.add_argument("--fixed-caption", type=str, default=None)
    add_online_eval_args(p)
    args = p.parse_args()

    if args.gen_start_frame < args.num_cond_frames:
        # Long-context setting anchors generation at frame 14 with 14 cond frames
        # taken from [0,14); cond_start clamps to 0.
        pass

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    height = 480 if args.resolution == "480p" else 720
    width = 832 if args.resolution == "480p" else 1280
    num_gen = args.num_frames - args.num_cond_frames
    vae_t = 4

    os.makedirs(args.output_dir, exist_ok=True)
    videos_dir = os.path.join(args.output_dir, "videos")
    retain_set = set()
    if args.save_only_list:
        with open(args.save_only_list) as f:
            retain_set = set(json.load(f).get("all", []))
    if not args.no_save_videos or retain_set:
        os.makedirs(videos_dir, exist_ok=True)

    method_name = "ttc_baseline" if args.no_correction else "ttc_longcat"
    print("=" * 70)
    print("Pathwise TTC on LongCat (LONG-HORIZON comparison only)")
    print("=" * 70)
    print("  Geometry     : cond=%d gen=%d gen_start=%d (total=%d)"
          % (args.num_cond_frames, num_gen, args.gen_start_frame, args.num_frames))
    print("  Correction   : %s" % ("OFF (baseline)" if args.no_correction else "ON"))
    if not args.no_correction:
        print("  sigma<=       : %.3f  cadence=%d  weight=%.3f  mode=%s"
              % (args.ttc_sigma_threshold, args.ttc_cadence, args.ttc_weight,
                 "full_latent" if args.ttc_full_latent else "appearance_only"))
    print("  CFG          : %s" % (not args.no_cfg))
    print("=" * 70)

    components = load_longcat_components(args.checkpoint_dir, device=device, dtype=torch.bfloat16)
    dit, vae = components["dit"], components["vae"]
    scheduler = components["scheduler"]
    tokenizer, text_encoder = components["tokenizer"], components["text_encoder"]
    for m in (dit, vae, text_encoder):
        for prm in m.parameters():
            prm.requires_grad = False
    dit.eval(); vae.eval(); text_encoder.eval()

    engine = SAViDNO_LongCat(
        dit=dit, vae=vae, scheduler=scheduler,
        tokenizer=tokenizer, text_encoder=text_encoder,
        device=device, dtype=torch.bfloat16,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        gradient_checkpointing=False,
    )
    ttc = TTC_LongCat(
        engine,
        sigma_threshold=args.ttc_sigma_threshold,
        cadence=args.ttc_cadence,
        weight=(0.0 if args.no_correction else args.ttc_weight),
        appearance_only=not args.ttc_full_latent,
        use_cfg=not args.no_cfg,
    )

    eval_videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed, validate_decodable=True
    )
    eval_videos = apply_fixed_caption(eval_videos, args.fixed_caption, context="eval")
    if args.start_video_idx > 0 or args.chunk_size > 0:
        end = len(eval_videos)
        if args.chunk_size > 0:
            end = min(args.start_video_idx + args.chunk_size, end)
        eval_videos = eval_videos[args.start_video_idx:end]

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0
    all_results = ckpt.get("results", []) if ckpt else []

    fvd_accumulator = OnlineFrechetAccumulator(
        device=device, compute_fid=args.compute_fid, min_videos=args.min_fvd_videos,
        gt_cache_path=getattr(args, "gt_features_cache", None),
    ) if args.compute_fvd else None
    fvd_ckpt_path = os.path.join(args.output_dir, "fvd_checkpoint.npz")
    if fvd_accumulator is not None and start_idx > 0:
        fvd_accumulator.load_stats(fvd_ckpt_path)

    from lora_experiment.scripts.run_lora_tta import save_video_from_numpy

    for idx, entry in enumerate(tqdm(eval_videos, desc=method_name)):
        if idx < start_idx:
            continue
        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem
        try:
            cond_start = max(0, args.gen_start_frame - args.num_cond_frames)
            pixel_cond = load_video_frames(
                video_path, args.num_cond_frames, height=height, width=width,
                start_frame=cond_start,
            ).to(device, torch.bfloat16)

            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder, caption, device=device, dtype=torch.bfloat16)
            cond_latents = encode_video(vae, pixel_cond, normalize=True)

            T_gen_latent = 1 + (num_gen - 1) // vae_t
            target_shape = (1, cond_latents.shape[1], T_gen_latent,
                            cond_latents.shape[3], cond_latents.shape[4])

            gen = torch.Generator(device=device).manual_seed(args.seed + idx)
            t0 = time.time()
            pred_pixels, n_corr = ttc.sample(
                cond_latents, target_shape, prompt_embeds, prompt_mask, generator=gen)
            elapsed = time.time() - t0

            # Build gen_output = [cond | gen] as [N,H,W,3] so we can reuse the
            # standard metric harness (it slices [num_cond : num_cond+num_gen]).
            cond_hwc = np.clip(
                ((pixel_cond.squeeze(0).float().cpu().numpy() + 1.0) / 2.0
                 ).transpose(1, 2, 3, 0), 0, 1)
            gen_hwc = _to_hwc_uint01(pred_pixels)
            gen_output = np.concatenate([cond_hwc, gen_hwc], axis=0)

            step_metrics = evaluate_generation_metrics(
                gen_output=gen_output, video_path=video_path,
                num_cond_frames=args.num_cond_frames, num_gen_frames=num_gen,
                gen_start_frame=args.gen_start_frame, device=device,
                return_gt_frames=(fvd_accumulator is not None),
            )
            _gt = step_metrics.pop("gt_frames_hwc", None)
            if fvd_accumulator is not None:
                fvd_accumulator.update(
                    gen_output, video_path, args.num_cond_frames, num_gen,
                    args.gen_start_frame, gt_frames_hwc=_gt)

            should_save = (not args.no_save_videos) or (video_name in retain_set)
            out_path = None
            if should_save:
                out_path = os.path.join(videos_dir, f"{video_name}_ttc.mp4")
                save_video_from_numpy(gen_hwc, out_path, fps=24)

            res = {
                "idx": idx, "video_name": video_name, "video_path": video_path,
                "caption": caption, "success": True,
                "gen_time": elapsed, "ttc_corrections": n_corr,
                "psnr": step_metrics.get("psnr"),
                "ssim": step_metrics.get("ssim"),
                "lpips": step_metrics.get("lpips"),
            }
            if out_path:
                res["output_path"] = out_path
            all_results.append(res)
            print("    PSNR=%.2f SSIM=%.4f LPIPS=%.4f (corr steps=%d)" % (
                step_metrics.get("psnr", float("nan")),
                step_metrics.get("ssim", float("nan")),
                step_metrics.get("lpips", float("nan")), n_corr))

            del pixel_cond, cond_latents, pred_pixels, prompt_embeds, prompt_mask
            torch_gc()
        except Exception as ex:
            import traceback
            traceback.print_exc()
            all_results.append({"idx": idx, "video_name": video_name,
                                "video_path": video_path, "error": str(ex),
                                "success": False})

        save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)
        if fvd_accumulator is not None:
            fvd_accumulator.save_stats(fvd_ckpt_path)

    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": method_name,
        "backbone": "longcat",
        "horizon": "long",
        "reference": "Pathwise Test-Time Correction (TTC)",
        "correction_enabled": not args.no_correction,
        "ttc_sigma_threshold": args.ttc_sigma_threshold,
        "ttc_cadence": args.ttc_cadence,
        "ttc_weight": (0.0 if args.no_correction else args.ttc_weight),
        "ttc_mode": "full_latent" if args.ttc_full_latent else "appearance_only",
        "use_cfg": not args.no_cfg,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "gen_start_frame": args.gen_start_frame,
        "num_gen_frames": num_gen,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "avg_gen_time": float(np.mean([r.get("gen_time", 0.0) for r in successful])) if successful else 0,
        "results": all_results,
    }
    aggregate_quality_metrics(summary)
    finalize_online_eval(fvd_accumulator, summary, videos_dir, args)
    save_results(summary, os.path.join(args.output_dir, "summary.json"))

    print("\n" + "=" * 70)
    print("TTC LongCat Complete: %d/%d ok -> %s" % (
        len(successful), len(all_results), args.output_dir))
    print("=" * 70)


if __name__ == "__main__":
    main()
