#!/usr/bin/env python3
"""Best-of-k seed generation for frozen LongCat-Video (no TTA).

Motivation: the AdaSteer budget-grid router is unroutable and its oracle headroom
is max-over-noise (see paper_tables/2026-07-31_router_significance_1000v.md),
because parameter-delta TTA produces near-identical videos. The lever with
*demonstrated real* headroom (SAVi-DNO, CVPR'24) is the INITIAL NOISE / seed:
different seeds give genuinely different continuations with a real quality
spread. This harness measures that headroom on our own pool.

For each video it generates K continuations from the FROZEN model using K
distinct seeds (candidate 0 == the deployed single-seed reference), computes
per-candidate PSNR/SSIM/LPIPS on the generated-only window (via
evaluate_generation_metrics, no GT leakage), and records cheap GT-free selector
signals (seam continuity, motion, sharpness) so an offline probe can test
whether best-of-k is *routable* with a deploy-legitimate ranker.

No DiT fine-tuning happens here — this isolates seed stochasticity of the base
model. Reuses delta_experiment/scripts/common.py (same generation + metric path
as run_full_tta.py / run_delta_a.py), so geometry (cond/gen window, seeds) lines
up with the existing pool.

Usage:
    python bestofk_experiment/scripts/run_bestofk_seeds.py \
      --checkpoint-dir /path/to/longcat-checkpoints \
      --data-dir /path/to/panda_ood_budget_1000v_preview \
      --output-dir bestofk_experiment/results/panda_1000v_k8 \
      --num-seeds 8 --num-cond-frames 14 --num-frames 28 --gen-start-frame 48
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_DELTA_SCRIPTS = _REPO_ROOT / "delta_experiment" / "scripts"
sys.path.insert(0, str(_DELTA_SCRIPTS))
sys.path.insert(0, str(_REPO_ROOT))

from common import (  # noqa: E402
    load_longcat_components,
    load_video_frames,
    generate_video_continuation,
    evaluate_generation_metrics,
    save_results,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
    load_ucf101_video_list,
    apply_fixed_caption,
    add_caption_override_args,
)


def save_video_from_numpy(frames: np.ndarray, output_path: str, fps: int = 24):
    import imageio

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frames_u8 = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    imageio.mimwrite(output_path, frames_u8, fps=fps, codec="libx264", quality=9)


def _gt_free_signals(gen_frames: np.ndarray, num_gen: int) -> Dict[str, float]:
    """Cheap deploy-legitimate signals from a candidate (no future GT).

    gen_frames: [T, H, W, 3] in [0,1]; the last `num_gen` are the generated tail
    and (when T>num_gen) the frame before them is the last conditioning frame.
    """
    f = np.asarray(gen_frames, dtype=np.float32)
    if f.ndim != 4:
        return {}
    T = f.shape[0]
    tail = f[-num_gen:] if T >= num_gen else f
    gray = tail.mean(axis=-1)  # [t,H,W]

    # motion: mean squared inter-frame difference across the generated tail
    if gray.shape[0] >= 2:
        temporal_l2 = float(np.mean((gray[1:] - gray[:-1]) ** 2))
    else:
        temporal_l2 = float("nan")

    # seam continuity: MSE between last cond frame and first generated frame
    if T > num_gen:
        last_cond = f[-num_gen - 1]
        first_gen = f[-num_gen]
        seam_l2 = float(np.mean((last_cond - first_gen) ** 2))
    else:
        seam_l2 = float("nan")

    # sharpness: mean Laplacian variance over the generated tail (blurry->low)
    lap_vars = []
    for t in range(gray.shape[0]):
        g = gray[t]
        lap = (
            -4.0 * g[1:-1, 1:-1]
            + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
        )
        lap_vars.append(float(lap.var()))
    sharpness = float(np.mean(lap_vars)) if lap_vars else float("nan")

    return {
        "sig_temporal_l2": temporal_l2,
        "sig_seam_l2": seam_l2,
        "sig_sharpness": sharpness,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Best-of-k seed generation (frozen LongCat)")
    ap.add_argument("--checkpoint-dir", type=str, required=True)
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--max-videos", type=int, default=200)
    ap.add_argument("--start-video-idx", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42, help="base seed; candidate 0 == deployed reference")
    ap.add_argument("--num-seeds", type=int, default=8, help="K candidates per video")
    ap.add_argument("--seed-stride", type=int, default=1000,
                    help="candidate s seed = seed + idx + s*stride (distinct per candidate)")
    ap.add_argument("--num-cond-frames", type=int, default=14)
    ap.add_argument("--num-frames", type=int, default=28)
    ap.add_argument("--gen-start-frame", type=int, default=48)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--guidance-scale", type=float, default=4.0)
    ap.add_argument("--resolution", type=str, default="480p")
    ap.add_argument("--no-save-videos", action="store_true",
                    help="skip mp4 saving (metrics + signals still recorded)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--restart", action="store_true")
    add_caption_override_args(ap)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    videos_dir = os.path.join(args.output_dir, "videos")
    if not args.no_save_videos:
        os.makedirs(videos_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = 0
    all_results: List[dict] = []
    if ckpt and not args.restart:
        start_idx = ckpt.get("next_idx", 0)
        all_results = ckpt.get("results", [])

    num_gen = args.num_frames - args.num_cond_frames

    print("=" * 70)
    print("Best-of-k seed generation (frozen LongCat-Video, no TTA)")
    print("=" * 70)
    print(f"Data dir     : {args.data_dir}")
    print(f"Output dir    : {args.output_dir}")
    print(f"K seeds/video : {args.num_seeds}  (stride {args.seed_stride})")
    print(f"Geometry      : cond={args.num_cond_frames} frames={args.num_frames} "
          f"gen_start={args.gen_start_frame} (num_gen={num_gen})")
    print(f"Resume from   : {start_idx}")
    print("=" * 70)

    print("\nLoading LongCat-Video components...")
    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16
    )
    pipe = components["pipe"]
    dit = components["dit"]
    for p in dit.parameters():
        p.requires_grad = False
    dit.eval()

    eval_videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed, validate_decodable=True
    )
    if args.start_video_idx > 0 or args.chunk_size > 0:
        end = len(eval_videos)
        if args.chunk_size > 0:
            end = min(args.start_video_idx + args.chunk_size, end)
        eval_videos = eval_videos[args.start_video_idx:end]
        print(f"Chunk: videos [{args.start_video_idx}:{end}] -> {len(eval_videos)} videos")
    eval_videos = apply_fixed_caption(eval_videos, args.fixed_caption, context="eval")

    exp_config = {
        "method": "bestofk_seeds",
        "num_seeds": args.num_seeds,
        "seed": args.seed,
        "seed_stride": args.seed_stride,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "gen_start_frame": args.gen_start_frame,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "resolution": args.resolution,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(exp_config, f, indent=2)

    from PIL import Image

    print(f"\nProcessing {len(eval_videos) - start_idx} videos "
          f"({args.num_seeds} seeds each)...\n")

    for idx, entry in enumerate(eval_videos):
        if idx < start_idx:
            continue
        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem
        print(f"[{idx + 1}/{len(eval_videos)}] {video_name}: {caption}")

        try:
            gen_cond_start = args.gen_start_frame - args.num_cond_frames
            gen_pixel_frames = load_video_frames(
                video_path, args.num_cond_frames, height=480, width=832,
                start_frame=max(0, gen_cond_start),
            ).to(args.device, torch.bfloat16)
            pf = gen_pixel_frames.squeeze(0)
            pf = ((pf + 1.0) / 2.0).clamp(0, 1)
            cond_images = []
            for t_idx in range(pf.shape[1]):
                arr = (pf[:, t_idx].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                cond_images.append(Image.fromarray(arr))
            del gen_pixel_frames

            candidates: List[dict] = []
            for s in range(args.num_seeds):
                seed_s = args.seed + idx + s * args.seed_stride
                t0 = time.time()
                gen_frames = generate_video_continuation(
                    pipe=pipe, video_frames=cond_images, prompt=caption,
                    num_cond_frames=args.num_cond_frames,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=seed_s,
                    resolution=args.resolution, device=args.device,
                )
                gen_time = time.time() - t0

                metrics = evaluate_generation_metrics(
                    gen_output=gen_frames, video_path=video_path,
                    num_cond_frames=args.num_cond_frames,
                    num_gen_frames=num_gen,
                    gen_start_frame=args.gen_start_frame, device=args.device,
                )
                signals = _gt_free_signals(gen_frames, num_gen)

                cand = {
                    "seed_index": s,
                    "seed": seed_s,
                    "psnr": metrics.get("psnr"),
                    "ssim": metrics.get("ssim"),
                    "lpips": metrics.get("lpips"),
                    "gen_time": gen_time,
                }
                cand.update(signals)

                if not args.no_save_videos:
                    out_path = os.path.join(videos_dir, f"{video_name}_seed{s}.mp4")
                    save_video_from_numpy(gen_frames, out_path, fps=24)
                    cand["output_path"] = out_path
                candidates.append(cand)
                print(f"    seed[{s}] #{seed_s}: PSNR={cand['psnr']}, "
                      f"seam={cand.get('sig_seam_l2'):.4g}, "
                      f"sharp={cand.get('sig_sharpness'):.4g}  ({gen_time:.1f}s)")
                del gen_frames
                torch_gc()

            psnrs = [c["psnr"] for c in candidates if c.get("psnr") is not None]
            best = max(psnrs) if psnrs else None
            ref = candidates[0].get("psnr") if candidates else None
            all_results.append({
                "idx": idx,
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "num_seeds": args.num_seeds,
                "candidates": candidates,
                "ref_psnr": ref,
                "best_psnr": best,
                "bestofk_gain_vs_ref": (best - ref) if (best is not None and ref is not None) else None,
                "success": True,
            })
            if best is not None and ref is not None:
                print(f"  best-of-{args.num_seeds} PSNR={best:.3f}  ref(seed0)={ref:.3f}  "
                      f"gain={best - ref:+.3f} dB")
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            all_results.append({
                "idx": idx, "video_name": video_name,
                "video_path": video_path, "error": str(exc), "success": False,
            })
            torch_gc()

        save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)

    successful = [r for r in all_results if r.get("success")]
    gains = [r["bestofk_gain_vs_ref"] for r in successful
             if r.get("bestofk_gain_vs_ref") is not None]
    summary = {
        "method": "bestofk_seeds",
        "num_seeds": args.num_seeds,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "gen_start_frame": args.gen_start_frame,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "mean_bestofk_gain_vs_ref": float(np.mean(gains)) if gains else None,
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))

    print("\n" + "=" * 70)
    print("Best-of-k seed generation complete")
    print(f"  Successful: {len(successful)}/{len(all_results)}")
    if gains:
        print(f"  Mean best-of-{args.num_seeds} PSNR gain vs seed0: "
              f"{np.mean(gains):+.3f} dB (oracle, needs routability probe)")
    print(f"  Results: {args.output_dir}/summary.json")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
