#!/usr/bin/env python3
"""
Long-horizon drift diagnostic (NOTTA) for LongCat-Video.

WHY THIS EXISTS
---------------
Every intervention we have tried (AdaSteer delta, placement EXP2, TANGO EXP3)
is ~null on our SHORT, in-domain, single-chunk 14->14 continuation. The
problem-difficulty audit
(``sweep_experiment/reports/paper_tables/2026-08-06_problem_difficulty_field_geometry.md``)
concluded the base model (LongCat-Video 13.6B, RLHF, continuation-pretrained)
is *too strong* for that task, so headroom is small BY CONSTRUCTION. The field
finds its headroom in LONG-HORIZON AUTOREGRESSIVE ROLLOUT, where error
accumulates chunk-over-chunk (Rolling Forcing 2509.25161, Pathwise TTC
2602.05871, BAgger 2512.12080, Self-Forcing, Meta-ARVDM 2503.10704).

This script is the DECISIVE, cheap first step recommended by that memo: run
NOTTA true autoregressive rollout (feed the model's own generated tail back as
conditioning) for K chunks on a handful of clips and measure per-chunk quality
degradation. If quality degrades with chunk index -> the headroom exists and
every steering/correction idea suddenly has room to show an effect. If it does
NOT degrade -> LongCat is too strong for this framing and we should switch base
models. Either outcome resolves the question empirically.

WHAT IT MEASURES (per chunk index)
----------------------------------
The long-video literature names the drift signatures explicitly: over-smoothing,
over-saturation, and loss of motion diversity (BAgger 2512.12080). We track all
three GT-FREE (so they are defined for the ENTIRE rollout, even after the source
video's ground truth runs out), plus cross-chunk seam discontinuity, plus the
usual GT metrics (PSNR/SSIM/LPIPS) wherever GT still overlaps the rollout:

  * sharpness      = variance of the Laplacian (over-smoothing -> DECREASES)
  * colorfulness   = Hasler-Susstrunk metric (over-saturation -> INCREASES)
  * temporal_motion= mean |frame[t+1]-frame[t]| within the chunk (motion collapse
                     -> DECREASES)
  * brightness/contrast = mean / std of luma (drift can push either)
  * seam_jump      = mean |first_gen_frame - last_cond_frame| (a hard cut at the
                     chunk boundary; report as ratio over within-chunk motion)
  * psnr/ssim/lpips= vs the true future frames, ONLY for chunks whose GT window
                     is still inside the source clip (nan otherwise)

GEOMETRY
--------
Per-chunk geometry is IDENTICAL to the AdaSteer / placement / EXP3 runs
(cond=14, num_frames=28 -> num_gen=14, gen_start=48, seed=42, 50 steps, CFG=4.0)
so a drift curve here is directly comparable to those experiments. We simply
CHAIN K chunks instead of stopping at one. The rollout re-conditioning is copied
verbatim from ``run_delta_a.py`` (tail = prev_gen[num_gen:]) so this is the same
autoregressive path the deployable code would use -- just NOTTA and with richer
per-chunk logging.

This script trains NOTHING (NOTTA): it runs the plain pipeline, no delta wrapper.

Usage
-----
    python diag_longhorizon_drift.py \
        --checkpoint-dir /scratch/wc3013/longcat-video-checkpoints \
        --data-dir /scratch/.../panda_ood_budget_1000v_preview_480p \
        --output-dir sweep_experiment/results/diag_longhorizon_drift \
        --max-videos 24 --num-chunks 8 \
        --num-cond-frames 14 --num-frames 28 --gen-start-frame 48
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    load_longcat_components,
    load_video_frames,
    generate_video_continuation,
    evaluate_generation_metrics,
    save_video_from_numpy,
    load_ucf101_video_list,
    save_results,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
)


# ============================================================================
# GT-free per-frame / per-chunk drift signals (numpy only, no cv2)
# ============================================================================

def _to_gray(frames: np.ndarray) -> np.ndarray:
    """[T,H,W,3] float[0,1] -> [T,H,W] luma."""
    return (0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2])


def _laplacian_var(gray_t: np.ndarray) -> float:
    """Variance of the 4-neighbour discrete Laplacian; a standard focus/blur
    measure. Over-smoothing (a classic AR drift signature) lowers this."""
    g = gray_t
    lap = (
        4.0 * g
        - np.roll(g, 1, axis=0) - np.roll(g, -1, axis=0)
        - np.roll(g, 1, axis=1) - np.roll(g, -1, axis=1)
    )
    # drop the 1px border where np.roll wraps around
    lap = lap[1:-1, 1:-1]
    return float(np.var(lap))


def _colorfulness(frame: np.ndarray) -> float:
    """Hasler & Susstrunk (2003) colorfulness. Over-saturation raises this."""
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_rg, std_yb = float(np.std(rg)), float(np.std(yb))
    mean_rg, mean_yb = float(np.mean(rg)), float(np.mean(yb))
    return (
        (std_rg ** 2 + std_yb ** 2) ** 0.5
        + 0.3 * (mean_rg ** 2 + mean_yb ** 2) ** 0.5
    )


def _saturation(frame: np.ndarray) -> float:
    """Mean HSV saturation = (max-min)/max over channels."""
    mx = frame.max(axis=-1)
    mn = frame.min(axis=-1)
    sat = np.where(mx > 1e-6, (mx - mn) / (mx + 1e-6), 0.0)
    return float(np.mean(sat))


def gen_free_signals(gen_only: np.ndarray, last_cond_frame: np.ndarray) -> Dict[str, float]:
    """Compute GT-free drift signals on the GENERATED frames [T,H,W,3] float[0,1].

    ``last_cond_frame`` [H,W,3] is the final conditioning frame fed to this
    chunk, used for the cross-chunk seam metric.
    """
    gen_only = np.clip(gen_only.astype(np.float32), 0.0, 1.0)
    T = gen_only.shape[0]
    gray = _to_gray(gen_only)

    sharp = float(np.mean([_laplacian_var(gray[t]) for t in range(T)]))
    colorful = float(np.mean([_colorfulness(gen_only[t]) for t in range(T)]))
    sat = float(np.mean([_saturation(gen_only[t]) for t in range(T)]))
    brightness = float(np.mean(gray))
    contrast = float(np.mean([np.std(gray[t]) for t in range(T)]))

    # within-chunk temporal motion: mean abs consecutive-frame diff
    if T >= 2:
        motion = float(np.mean(np.abs(gen_only[1:] - gen_only[:-1])))
    else:
        motion = float("nan")

    # cross-chunk seam: jump from the last conditioning frame into the first
    # generated frame, normalised by the within-chunk motion (a value >> 1 means
    # a visible cut / re-anchoring discontinuity at the chunk boundary).
    seam_jump = float(np.mean(np.abs(gen_only[0] - np.clip(last_cond_frame, 0, 1))))
    seam_ratio = seam_jump / (motion + 1e-6) if motion == motion and motion > 0 else float("nan")

    return {
        "sharpness": sharp,
        "colorfulness": colorful,
        "saturation": sat,
        "brightness": brightness,
        "contrast": contrast,
        "temporal_motion": motion,
        "seam_jump": seam_jump,
        "seam_ratio": seam_ratio,
    }


# ============================================================================
# Aggregation
# ============================================================================

_GEN_FREE_KEYS = [
    "sharpness", "colorfulness", "saturation", "brightness",
    "contrast", "temporal_motion", "seam_jump", "seam_ratio",
]
_GT_KEYS = ["psnr", "ssim", "lpips"]
_ALL_KEYS = _GEN_FREE_KEYS + _GT_KEYS


def _finite(vals: List[float]) -> List[float]:
    return [v for v in vals if v is not None and isinstance(v, (int, float)) and v == v]


def build_drift_curves(results: List[Dict], num_chunks: int) -> Dict:
    """Mean +/- std per chunk index across all successful videos, plus a
    simple drift verdict (slope + last/first ratio) for the headline signals."""
    per_chunk = {k: [[] for _ in range(num_chunks)] for k in _ALL_KEYS}
    for r in results:
        if not r.get("success"):
            continue
        for ch in r.get("chunks", []):
            ci = ch["chunk"] - 1
            if not (0 <= ci < num_chunks):
                continue
            for k in _ALL_KEYS:
                per_chunk[k][ci].append(ch.get(k))

    curves = {}
    for k in _ALL_KEYS:
        curves[k] = {"mean": [], "std": [], "n": []}
        for ci in range(num_chunks):
            fv = _finite(per_chunk[k][ci])
            curves[k]["mean"].append(float(np.mean(fv)) if fv else None)
            curves[k]["std"].append(float(np.std(fv)) if fv else None)
            curves[k]["n"].append(len(fv))

    # Drift verdict on the headline GT-free signals + PSNR.
    verdict = {}
    for k in ["sharpness", "temporal_motion", "colorfulness", "contrast", "psnr", "lpips"]:
        means = curves[k]["mean"]
        pts = [(i + 1, m) for i, m in enumerate(means) if m is not None]
        if len(pts) >= 2:
            xs = np.array([p[0] for p in pts], dtype=float)
            ys = np.array([p[1] for p in pts], dtype=float)
            slope = float(np.polyfit(xs, ys, 1)[0])
            first, last = ys[0], ys[-1]
            verdict[k] = {
                "first_chunk": float(first),
                "last_chunk": float(last),
                "abs_change": float(last - first),
                "pct_change": float((last - first) / (abs(first) + 1e-9) * 100.0),
                "slope_per_chunk": slope,
                "n_chunks_with_data": len(pts),
            }
    return {"curves": curves, "verdict": verdict}


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    p = argparse.ArgumentParser(description="Long-horizon NOTTA drift diagnostic")
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--data-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--max-videos", type=int, default=24)
    p.add_argument("--start-video-idx", type=int, default=0)
    p.add_argument("--chunk-size", type=int, default=0,
                   help="Number of videos to process from start-video-idx (0=all).")
    p.add_argument("--num-chunks", type=int, default=8,
                   help="Autoregressive rollout length (chunks chained).")
    p.add_argument("--num-cond-frames", type=int, default=14)
    p.add_argument("--num-frames", type=int, default=28,
                   help="Per-chunk window (num_gen = num_frames - num_cond_frames).")
    p.add_argument("--gen-start-frame", type=int, default=48)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=4.0)
    p.add_argument("--resolution", type=str, default="480p")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--no-save-videos", action="store_true",
                   help="Do not write the stitched rollout mp4 per video.")
    args = p.parse_args()

    num_gen = args.num_frames - args.num_cond_frames
    if num_gen <= 0:
        print("ERROR: num_frames must exceed num_cond_frames", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    videos_dir = os.path.join(args.output_dir, "videos")
    if not args.no_save_videos:
        os.makedirs(videos_dir, exist_ok=True)

    print("=" * 70)
    print("Long-horizon NOTTA drift diagnostic for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint : {args.checkpoint_dir}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Geometry   : cond={args.num_cond_frames} frames={args.num_frames} "
          f"num_gen={num_gen} gen_start={args.gen_start_frame}")
    print(f"Rollout    : {args.num_chunks} chunks (autoregressive, NOTTA)")
    print(f"Sampler    : steps={args.num_inference_steps} cfg={args.guidance_scale} "
          f"seed={args.seed}")
    print("=" * 70)

    # Resume support
    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0
    all_results = list(ckpt.get("results", [])) if ckpt else []

    print("\nLoading model components...")
    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16
    )
    pipe = components["pipe"]

    eval_videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed,
        validate_decodable=True,
    )
    if args.start_video_idx > 0 or args.chunk_size > 0:
        end = len(eval_videos)
        if args.chunk_size > 0:
            end = min(args.start_video_idx + args.chunk_size, end)
        eval_videos = eval_videos[args.start_video_idx:end]
        print(f"Chunk: videos [{args.start_video_idx}:{end}] -> {len(eval_videos)}")
    print(f"\nEvaluation videos: {len(eval_videos)}")

    from PIL import Image

    for v_idx, entry in enumerate(eval_videos):
        if v_idx < start_idx:
            continue
        eval_name = Path(entry["video_path"]).stem
        print(f"\n{'='*70}\n[{v_idx+1}/{len(eval_videos)}] {eval_name}")

        try:
            # Initial conditioning window = video[gen_start-cond : gen_start]
            cond_start = max(0, args.gen_start_frame - args.num_cond_frames)
            gen_pf = load_video_frames(
                entry["video_path"], args.num_cond_frames,
                height=480, width=832, start_frame=cond_start,
            ).to(args.device, torch.bfloat16)
            pf = ((gen_pf.squeeze(0) + 1.0) / 2.0).clamp(0, 1)
            cond_images = [
                Image.fromarray(
                    (pf[:, t].permute(1, 2, 0).float().cpu().numpy() * 255).astype(np.uint8)
                )
                for t in range(pf.shape[1])
            ]

            chunk_records: List[Dict] = []
            stitched: List[np.ndarray] = []
            # seed the stitched clip with the true conditioning frames
            stitched.append(np.stack([np.asarray(im) / 255.0 for im in cond_images], axis=0))

            prev_gen = None
            gen_time = 0.0
            for step_i in range(args.num_chunks):
                if step_i > 0:
                    tail = prev_gen[num_gen:]
                    cond_images = [
                        Image.fromarray((np.clip(tail[t], 0, 1) * 255).astype(np.uint8))
                        for t in range(tail.shape[0])
                    ]

                last_cond_frame = np.asarray(cond_images[-1]).astype(np.float32) / 255.0

                t0 = time.time()
                gen_frames = generate_video_continuation(
                    pipe=pipe,
                    video_frames=cond_images,
                    prompt=entry["caption"],
                    num_cond_frames=args.num_cond_frames,
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=args.seed + v_idx + step_i,
                    resolution=args.resolution,
                    device=args.device,
                )
                gen_time += time.time() - t0

                gen_only = gen_frames[args.num_cond_frames:args.num_cond_frames + num_gen]

                # GT metrics where the source clip still overlaps the rollout.
                step_gen_start = args.gen_start_frame + step_i * num_gen
                gt_metrics = evaluate_generation_metrics(
                    gen_output=gen_frames,
                    video_path=entry["video_path"],
                    num_cond_frames=args.num_cond_frames,
                    num_gen_frames=num_gen,
                    gen_start_frame=step_gen_start,
                    device=args.device,
                    return_gt_frames=False,
                )
                free = gen_free_signals(gen_only, last_cond_frame)

                rec = {"chunk": step_i + 1, "gen_start_frame": step_gen_start,
                       "gt_available": gt_metrics.get("psnr") == gt_metrics.get("psnr")}
                rec.update({k: gt_metrics.get(k) for k in _GT_KEYS})
                rec.update(free)
                chunk_records.append(rec)
                stitched.append(np.clip(gen_only.astype(np.float32), 0, 1))

                print(
                    f"  chunk {step_i+1:2d}: sharp={free['sharpness']:.4f} "
                    f"motion={free['temporal_motion']:.4f} "
                    f"colorful={free['colorfulness']:.4f} "
                    f"seam_ratio={free['seam_ratio']:.2f} "
                    f"psnr={rec['psnr'] if rec['psnr'] == rec['psnr'] else float('nan'):.2f}"
                    + ("" if rec["gt_available"] else " (no GT)")
                )
                prev_gen = gen_frames

            record = {
                "video_name": eval_name,
                "video_path": entry["video_path"],
                "caption": entry["caption"],
                "num_chunks": args.num_chunks,
                "num_gen_per_chunk": num_gen,
                "gen_time": gen_time,
                "chunks": chunk_records,
                "success": True,
            }
            all_results.append(record)

            if not args.no_save_videos:
                out_mp4 = os.path.join(videos_dir, f"{eval_name}_rollout.mp4")
                save_video_from_numpy(np.concatenate(stitched, axis=0), out_mp4, fps=24)
                record["output_path"] = out_mp4

            save_checkpoint({"next_idx": v_idx + 1, "results": all_results}, ckpt_path)
            del gen_pf
            torch_gc()

        except Exception as e:  # noqa: BLE001
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "video_name": eval_name, "video_path": entry["video_path"],
                "error": str(e), "success": False,
            })
            save_checkpoint({"next_idx": v_idx + 1, "results": all_results}, ckpt_path)
            torch_gc()

    successful = [r for r in all_results if r.get("success")]
    drift = build_drift_curves(successful, args.num_chunks)
    summary = {
        "method": "notta_longhorizon_drift",
        "num_chunks": args.num_chunks,
        "num_cond_frames": args.num_cond_frames,
        "num_frames": args.num_frames,
        "num_gen_per_chunk": num_gen,
        "gen_start_frame": args.gen_start_frame,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "drift_curves": drift["curves"],
        "drift_verdict": drift["verdict"],
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))

    print("\n" + "=" * 70)
    print(f"Drift verdict over {len(successful)} videos x {args.num_chunks} chunks:")
    for k, v in drift["verdict"].items():
        print(f"  {k:16s}: chunk1={v['first_chunk']:.4f} -> "
              f"chunk{v['n_chunks_with_data']}={v['last_chunk']:.4f} "
              f"({v['pct_change']:+.1f}%, slope={v['slope_per_chunk']:+.5f}/chunk)")
    print("=" * 70)
    print(f"Saved: {args.output_dir}/summary.json")
    print("Plot:  python scripts/plot_drift_curves.py --summary "
          f"{args.output_dir}/summary.json --out-dir {args.output_dir}/plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())
