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
Default (``--rollout-mode reencode``) per-chunk geometry is IDENTICAL to the
AdaSteer / placement / EXP3 runs (cond=14, num_frames=28 -> num_gen=14,
gen_start=48, seed=42, 50 steps, CFG=4.0) so a drift curve here is directly
comparable to those experiments. We simply CHAIN K chunks instead of stopping
at one. The rollout re-conditioning is copied verbatim from ``run_delta_a.py``
(tail = prev_gen[num_gen:]) so this is the same autoregressive path the
deployable code would use.

TWO CONTROLS ADDED (2026-08-07)
-------------------------------
1. ``--rollout-mode native``: LongCat's ``generate_vc`` is single-window and has
   NO KV-cache carryover across windows -- its native long-horizon IS this
   external-rollout re-conditioning. The only off-native knob is GEOMETRY: our
   short 14-cond/14-gen window creates many re-anchoring seams per unit of
   generated time, which could inflate apparent drift. ``native`` runs the SAME
   chaining at LongCat's idiomatic 13-cond/93-frame (80-gen) window so we can
   distinguish inherent long-horizon drift from a short-window re-conditioning
   artifact. If drift persists under ``native`` -> it is real.

2. ``--method delta``: train an AdaSteer delta ONCE on the observed frames
   (identical recipe to ``run_delta_a.py``: same TTA window, latent split,
   VAE/text-encoder offload, optimizer call) and hold it FIXED across the whole
   rollout. Question: does a fixed context-0 delta flatten the drift curves, or
   does it go stale as the rollout leaves the trained distribution (which would
   motivate a streaming / per-chunk re-fit delta)? NOTTA and delta share seeds
   per chunk, so the comparison is paired.

Usage
-----
    # NOTTA control at native geometry
    python diag_longhorizon_drift.py --rollout-mode native --method notta ...
    # AdaSteer delta held fixed across the rollout (reencode geometry)
    python diag_longhorizon_drift.py --method delta ...
        --checkpoint-dir /scratch/wc3013/longcat-video-checkpoints \
        --data-dir /scratch/.../panda_ood_budget_1000v_preview_480p \
        --output-dir sweep_experiment/results/diag_longhorizon_drift \
        --max-videos 24 --num-chunks 8
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
    encode_video,
    encode_prompt,
    split_tta_latents,
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
# GT-free drift verifier for best-of-N test-time search
# ============================================================================
# Test-time scaling (Video-T1, ICCV'25; MCTS-TTS, ICLR'26 sub; Verifier Matters,
# BMVC'25) reframes generation as search over the noise space with a verifier.
# Our contribution is the VERIFIER: a physically-grounded, GT-free drift score
# for autoregressive continuation. Each candidate continuation is scored by how
# far its GT-free statistics deviate from a FIXED real-frame reference (the
# initial conditioning frames -- the only ground-truth-real, always-available
# anchor in a deployable setting), plus a cross-chunk seam-continuity penalty.
# LOWER = more stable = preferred. Because candidate 0 reuses the NOTTA seed,
# best-of-N is a strict superset of NOTTA: it can only match or beat NOTTA when
# the verifier is informative. We also log every candidate so a post-hoc oracle
# (per-chunk best candidate) bounds the achievable headroom vs what the GT-free
# verifier actually captures.

_VERIFIER_SIGNALS = ["sharpness", "colorfulness", "contrast", "temporal_motion"]


def reference_signals(frames_01: np.ndarray) -> Dict[str, float]:
    """GT-free reference statistics from real frames [T,H,W,3] float[0,1]."""
    return gen_free_signals(frames_01, frames_01[0])


def verifier_score(free: Dict[str, float], ref: Dict[str, float],
                   seam_weight: float = 1.0) -> float:
    """Composite drift score (LOWER=better): sum of relative deviations of the
    GT-free signals from the real-frame reference + a seam-continuity penalty
    normalised by the reference motion scale. Two-sided deviation (not minimise)
    so it preserves the reference motion level rather than collapsing to a
    still frame."""
    s = 0.0
    for k in _VERIFIER_SIGNALS:
        rv, cv = ref.get(k), free.get(k)
        if rv is None or cv is None or rv != rv or cv != cv:
            continue
        s += abs(cv - rv) / (abs(rv) + 1e-6)
    seam_jump = free.get("seam_jump")
    ref_motion = ref.get("temporal_motion")
    if (seam_jump is not None and seam_jump == seam_jump
            and ref_motion is not None and ref_motion == ref_motion):
        s += seam_weight * seam_jump / (ref_motion + 1e-6)
    return float(s)


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
# Intervention: AdaSteer delta (trained once on observed frames, held fixed
# across the whole rollout -- identical recipe to run_delta_a.py)
# ============================================================================

def train_delta_for_video(
    components: Dict,
    video_path: str,
    caption: str,
    *,
    gen_start_frame: int,
    tta_total_frames: int,
    tta_context_frames: int,
    delta_steps: int,
    delta_lr: float,
    placement: str,
    device: str,
):
    """Train an AdaSteer delta on the OBSERVED frames before ``gen_start_frame``.

    Returns ``(wrapper, delta_norm, clean_bundle)``. ``clean_bundle`` holds the
    CLEAN chunk-0 latents + prompt embeds (on CPU) so a clean-anchored streaming
    re-fit can reuse the real-frame target without re-encoding the source video.
    The wrapper is NOT yet applied; caller must ``wrapper.apply_to_dit()`` before
    the rollout and ``remove_from_dit()`` after. This mirrors ``run_delta_a.py``
    exactly (same window, same latent split, same VAE/text-encoder offload dance,
    same optimizer call) so the delta arm is apples-to-apples with the standard
    AdaSteer runs -- the only difference is that here the trained delta is held
    fixed (or streamed) across a K-chunk autoregressive rollout.
    """
    # Lazy import so the NOTTA arm never pulls in run_delta_a.
    from run_delta_a import DeltaAWrapper, optimize_delta_a  # noqa: E402

    dtype = torch.bfloat16
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]
    dit = components["dit"]
    adaln_dim = getattr(dit.config, "adaln_tembed_dim", 512)

    tta_start = gen_start_frame - tta_total_frames
    pixel_frames = load_video_frames(
        video_path, tta_total_frames, height=480, width=832,
        start_frame=max(0, tta_start),
    ).to(device, dtype)
    all_latents = encode_video(vae, pixel_frames, normalize=True)

    vae_t_scale = 4
    num_ctx_lat = 1 + (tta_context_frames - 1) // vae_t_scale
    # holdout=0.0 -> no val split, no early stopping (plain fixed-step delta).
    cond_latents, train_latents, _ = split_tta_latents(
        all_latents, num_ctx_lat, holdout_fraction=0.0,
    )
    prompt_embeds, prompt_mask = encode_prompt(
        tokenizer, text_encoder, caption, device=device, dtype=dtype,
    )

    wrapper = DeltaAWrapper(
        dit, adaln_tembed_dim=adaln_dim, placement=placement,
    ).to(device)

    # Offload VAE + text encoder to CPU during training (frees VRAM for the
    # DiT+delta optimisation), then restore them for generation.
    vae.to("cpu")
    text_encoder.to("cpu")
    torch.cuda.empty_cache()
    try:
        opt_result = optimize_delta_a(
            wrapper=wrapper,
            cond_latents=cond_latents.to(device),
            train_latents=train_latents.to(device),
            prompt_embeds=prompt_embeds.to(device),
            prompt_mask=prompt_mask.to(device) if prompt_mask is not None else None,
            num_steps=delta_steps,
            lr=delta_lr,
            device=device,
            dtype=dtype,
        )
    finally:
        vae.to(device)
        text_encoder.to(device)
        torch.cuda.empty_cache()

    clean_bundle = {
        "train_latents": train_latents.detach().to("cpu"),
        "prompt_embeds": prompt_embeds.detach().to("cpu"),
        "prompt_mask": prompt_mask.detach().to("cpu") if prompt_mask is not None else None,
    }
    return wrapper, float(opt_result.get("delta_norm", float("nan"))), clean_bundle


def refit_delta_on_window(
    components: Dict,
    wrapper,
    delta_init,
    window_np: np.ndarray,
    caption: str,
    *,
    tta_context_frames: int,
    refit_steps: int,
    refit_lr: float,
    device: str,
    clean_target_latents=None,
    clean_prompt_embeds=None,
    clean_prompt_mask=None,
):
    """Streaming re-fit (EXP4): adapt the delta using the CURRENT (drifted)
    context, initialised from ``delta_init``. ``window_np`` is [T,H,W,3] float[0,1].

    Two target modes:
      * ``clean_target_latents is None`` (GENERATED target): flow-match the delta
        to the window's own tail -> self-supervised on the model's drifted output
        (the 2026-08-09 null: it partly REPRODUCES drift).
      * ``clean_target_latents`` provided (CLEAN target): condition on the drifted
        context but flow-match toward the CLEAN chunk-0 real-frame latents, i.e.
        teach the low-capacity bias "from where you've drifted, steer back to the
        clean distribution." This removes the train-on-own-drift flaw.

    Returns the re-fit delta tensor (caller blends it toward the chunk-0 delta).

    IMPORTANT: the caller must have REMOVED the generation hooks before calling
    (optimize_delta_a runs the wrapper.forward training path, which adds the
    delta via forward args; if the hooks were still installed the delta would be
    double-applied). Caller re-applies hooks afterwards.
    """
    from run_delta_a import optimize_delta_a  # noqa: E402

    dtype = torch.bfloat16
    vae = components["vae"]
    text_encoder = components["text_encoder"]
    tokenizer = components["tokenizer"]

    x = torch.from_numpy(np.clip(window_np, 0.0, 1.0).astype(np.float32))
    x = x.permute(3, 0, 1, 2).unsqueeze(0)          # [1,3,T,H,W]
    x = (x * 2.0 - 1.0).to(device, dtype)            # [-1,1], VAE convention
    ctx_latents = encode_video(vae, x, normalize=True)

    num_ctx_lat = 1 + (tta_context_frames - 1) // 4
    if clean_target_latents is None:
        # generated target: split the window into cond + its own continuation.
        cond_latents, train_latents, _ = split_tta_latents(
            ctx_latents, num_ctx_lat, holdout_fraction=0.0,
        )
    else:
        # clean target: the passed window IS the context (its first num_ctx_lat
        # latents condition the next chunk); target = clean chunk-0 latents.
        cond_latents = ctx_latents[:, :, :num_ctx_lat]
        train_latents = clean_target_latents.to(device, dtype)

    if clean_prompt_embeds is not None:
        prompt_embeds = clean_prompt_embeds.to(device, dtype)
        prompt_mask = clean_prompt_mask.to(device) if clean_prompt_mask is not None else None
    else:
        prompt_embeds, prompt_mask = encode_prompt(
            tokenizer, text_encoder, caption, device=device, dtype=dtype,
        )
    wrapper.delta.data.copy_(delta_init.to(device, wrapper.delta.dtype))

    vae.to("cpu")
    text_encoder.to("cpu")
    torch.cuda.empty_cache()
    try:
        optimize_delta_a(
            wrapper=wrapper,
            cond_latents=cond_latents.to(device),
            train_latents=train_latents.to(device),
            prompt_embeds=prompt_embeds.to(device),
            prompt_mask=prompt_mask.to(device) if prompt_mask is not None else None,
            num_steps=refit_steps,
            lr=refit_lr,
            device=device,
            dtype=dtype,
        )
    finally:
        vae.to(device)
        text_encoder.to(device)
        torch.cuda.empty_cache()

    return wrapper.delta.detach().clone()


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
    # --- Rollout protocol -------------------------------------------------
    # LongCat's generate_vc is single-window; it has NO KV-cache carryover
    # across windows. Native long-horizon == external rollout re-conditioned on
    # the last num_cond_frames GENERATED frames (which this script already
    # does). The only "off-native" knob is GEOMETRY: our short 14-cond/14-gen
    # window creates many re-anchoring seams per unit time, which could inflate
    # apparent drift. --rollout-mode native runs the SAME chaining at LongCat's
    # idiomatic 13-cond/93-frame (80-gen) window so we can tell inherent drift
    # from a short-window re-conditioning artifact.
    p.add_argument("--rollout-mode", type=str, default="reencode",
                   choices=["reencode", "native"],
                   help="reencode=short 14/28 window (default, comparable to "
                        "EXP2/EXP3); native=LongCat idiomatic 13-cond/93-frame "
                        "window. 'native' sets cond=13/frames=93 UNLESS you "
                        "explicitly override --num-cond-frames/--num-frames.")
    # --- Intervention -----------------------------------------------------
    p.add_argument("--method", type=str, default="notta",
                   choices=["notta", "delta", "delta_stream", "bestof"],
                   help="notta=plain pipeline; delta=train an AdaSteer delta on "
                        "the observed frames once and hold it FIXED across the "
                        "rollout; delta_stream=EXP4, re-fit the delta each chunk "
                        "on the most recent generated window (anchored toward the "
                        "chunk-0 delta) so it tracks the drifting distribution; "
                        "bestof=best-of-N test-time search: generate --search-k "
                        "candidate continuations per chunk (candidate 0 reuses "
                        "the NOTTA seed) and keep the one a GT-free drift verifier "
                        "judges most stable, so a bad chunk never poisons the "
                        "context. Strict superset of NOTTA.")
    p.add_argument("--delta-steps", type=int, default=10)
    p.add_argument("--delta-lr", type=float, default=1e-3)
    p.add_argument("--delta-placement", type=str, default="adaln",
                   choices=["adaln", "residual"])
    p.add_argument("--tta-total-frames", type=int, default=0,
                   help="Frames before gen_start used for TTA (0=use gen_start).")
    p.add_argument("--tta-context-frames", type=int, default=0,
                   help="Context frames within the TTA window (0=num_cond_frames).")
    # --- Streaming delta (EXP4) knobs ------------------------------------
    p.add_argument("--stream-refit-steps", type=int, default=5,
                   help="delta_stream: optimizer steps for each per-chunk re-fit.")
    p.add_argument("--stream-refit-lr", type=float, default=0.0,
                   help="delta_stream: re-fit LR (0=use --delta-lr).")
    p.add_argument("--stream-blend", type=float, default=0.5,
                   help="delta_stream: anchor weight lambda. applied delta = "
                        "(1-lambda)*refit + lambda*delta0. Higher=more anchored to "
                        "the real-data chunk-0 delta (guards against adapting to "
                        "the model's own drift).")
    p.add_argument("--stream-target", type=str, default="generated",
                   choices=["generated", "clean"],
                   help="delta_stream target: 'generated' flow-matches to the "
                        "drifted window's own tail (self-supervised on drift; the "
                        "2026-08-09 null); 'clean' conditions on the drifted "
                        "context but flow-matches toward the CLEAN chunk-0 real "
                        "latents (steer-back-to-clean; removes the flaw).")
    # --- Best-of-N test-time search knobs --------------------------------
    p.add_argument("--search-k", type=int, default=4,
                   help="bestof: candidate continuations per chunk (candidate 0 "
                        "== the NOTTA seed, so k>=1). Per-chunk cost scales x k.")
    p.add_argument("--search-seam-weight", type=float, default=1.0,
                   help="bestof: weight of the seam-continuity penalty in the "
                        "GT-free verifier score.")
    args = p.parse_args()

    # Native protocol: apply LongCat's idiomatic window geometry unless the
    # user explicitly set the geometry flags (argparse can't tell default from
    # an explicit equal value, so we compare against the reencode defaults).
    if args.rollout_mode == "native":
        if args.num_cond_frames == 14:
            args.num_cond_frames = 13
        if args.num_frames == 28:
            args.num_frames = 93

    num_gen = args.num_frames - args.num_cond_frames
    if num_gen <= 0:
        print("ERROR: num_frames must exceed num_cond_frames", file=sys.stderr)
        return 2

    # TTA window defaults (mirror run_delta_a.py).
    tta_total_frames = args.tta_total_frames or args.gen_start_frame
    tta_context_frames = args.tta_context_frames or args.num_cond_frames
    stream_refit_lr = args.stream_refit_lr or args.delta_lr
    uses_delta = args.method in ("delta", "delta_stream")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    videos_dir = os.path.join(args.output_dir, "videos")
    if not args.no_save_videos:
        os.makedirs(videos_dir, exist_ok=True)

    print("=" * 70)
    print("Long-horizon drift diagnostic for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint : {args.checkpoint_dir}")
    print(f"Data dir   : {args.data_dir}")
    print(f"Output dir : {args.output_dir}")
    print(f"Method     : {args.method}" + (
        f" (steps={args.delta_steps} lr={args.delta_lr} "
        f"placement={args.delta_placement})" if uses_delta else ""))
    if args.method == "delta_stream":
        print(f"Stream     : target={args.stream_target} "
              f"refit_steps={args.stream_refit_steps} "
              f"refit_lr={stream_refit_lr} blend(anchor)={args.stream_blend}")
    if args.method == "bestof":
        print(f"Search     : best-of-{args.search_k} (cand0=NOTTA seed) "
              f"GT-free drift verifier (seam_weight={args.search_seam_weight})")
    print(f"Roll mode  : {args.rollout_mode}")
    print(f"Geometry   : cond={args.num_cond_frames} frames={args.num_frames} "
          f"num_gen={num_gen} gen_start={args.gen_start_frame}")
    if uses_delta:
        print(f"TTA window : total={tta_total_frames} context={tta_context_frames}")
    print(f"Rollout    : {args.num_chunks} chunks (autoregressive)")
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

            # best-of-N verifier reference = the initial REAL conditioning frames
            # (fixed across the rollout; the deployable ground-truth anchor).
            ref_sig = (reference_signals(stitched[0].astype(np.float32))
                       if args.method == "bestof" else None)

            # Intervention: train an AdaSteer delta on the observed frames.
            # delta         -> held FIXED across the rollout.
            # delta_stream  -> re-fit each chunk on the most recent generated
            #                  window, anchored toward this chunk-0 delta (delta0).
            # Hook installed once, removed in finally so it can never leak into
            # the next video.
            wrapper = None
            delta_norm = None
            delta0 = None
            clean_bundle = None
            stream_norms = []
            if uses_delta:
                _td = time.time()
                wrapper, delta_norm, clean_bundle = train_delta_for_video(
                    components, entry["video_path"], entry["caption"],
                    gen_start_frame=args.gen_start_frame,
                    tta_total_frames=tta_total_frames,
                    tta_context_frames=tta_context_frames,
                    delta_steps=args.delta_steps,
                    delta_lr=args.delta_lr,
                    placement=args.delta_placement,
                    device=args.device,
                )
                delta0 = wrapper.delta.detach().clone()
                wrapper.apply_to_dit()
                print(f"  delta0 trained: norm={delta_norm:.4f} "
                      f"({time.time()-_td:.1f}s)")

            prev_gen = None
            gen_time = 0.0
            try:
                for step_i in range(args.num_chunks):
                    if step_i > 0:
                        tail = prev_gen[num_gen:]
                        cond_images = [
                            Image.fromarray((np.clip(tail[t], 0, 1) * 255).astype(np.uint8))
                            for t in range(tail.shape[0])
                        ]
                        # Streaming re-fit: adapt delta to the previous full
                        # window, then re-anchor toward delta0. Hooks OFF during
                        # the training forward (else delta double-applies), back
                        # ON for generation.
                        if args.method == "delta_stream":
                            _ts = time.time()
                            wrapper.remove_from_dit()
                            if args.stream_target == "clean":
                                # condition on the drifted tail (what conditions
                                # the next chunk); target = CLEAN chunk-0 latents.
                                refit = refit_delta_on_window(
                                    components, wrapper,
                                    wrapper.delta.detach().clone(),
                                    tail, entry["caption"],
                                    tta_context_frames=tta_context_frames,
                                    refit_steps=args.stream_refit_steps,
                                    refit_lr=stream_refit_lr,
                                    device=args.device,
                                    clean_target_latents=clean_bundle["train_latents"],
                                    clean_prompt_embeds=clean_bundle["prompt_embeds"],
                                    clean_prompt_mask=clean_bundle["prompt_mask"],
                                )
                            else:
                                refit = refit_delta_on_window(
                                    components, wrapper,
                                    wrapper.delta.detach().clone(),
                                    prev_gen, entry["caption"],
                                    tta_context_frames=tta_context_frames,
                                    refit_steps=args.stream_refit_steps,
                                    refit_lr=stream_refit_lr,
                                    device=args.device,
                                )
                            blended = (1.0 - args.stream_blend) * refit \
                                + args.stream_blend * delta0
                            wrapper.delta.data.copy_(blended)
                            wrapper.apply_to_dit()
                            snorm = float(wrapper.delta.detach().norm().item())
                            stream_norms.append(snorm)
                            print(f"    stream re-fit chunk {step_i+1} "
                                  f"[{args.stream_target}]: norm={snorm:.4f} "
                                  f"({time.time()-_ts:.1f}s)")

                    last_cond_frame = np.asarray(cond_images[-1]).astype(np.float32) / 255.0
                    step_gen_start = args.gen_start_frame + step_i * num_gen
                    base_seed = args.seed + v_idx + step_i

                    def _gen(seed):
                        return generate_video_continuation(
                            pipe=pipe, video_frames=cond_images,
                            prompt=entry["caption"],
                            num_cond_frames=args.num_cond_frames,
                            num_frames=args.num_frames,
                            num_inference_steps=args.num_inference_steps,
                            guidance_scale=args.guidance_scale,
                            seed=seed, resolution=args.resolution,
                            device=args.device,
                        )

                    def _gt(gf):
                        # GT metrics where the source clip still overlaps rollout.
                        return evaluate_generation_metrics(
                            gen_output=gf, video_path=entry["video_path"],
                            num_cond_frames=args.num_cond_frames,
                            num_gen_frames=num_gen, gen_start_frame=step_gen_start,
                            device=args.device, return_gt_frames=False,
                        )

                    cand_log = None
                    chosen = 0
                    if args.method == "bestof":
                        # Generate --search-k candidates; candidate 0 reuses the
                        # NOTTA seed so best-of-N is a strict superset of NOTTA.
                        cands = []
                        for c in range(max(1, args.search_k)):
                            cseed = base_seed if c == 0 else base_seed + c * 100003
                            t0 = time.time()
                            cg = _gen(cseed)
                            gen_time += time.time() - t0
                            cg_only = cg[args.num_cond_frames:args.num_cond_frames + num_gen]
                            c_free = gen_free_signals(cg_only, last_cond_frame)
                            c_gt = _gt(cg)
                            c_score = verifier_score(c_free, ref_sig, args.search_seam_weight)
                            cands.append({"cand": c, "seed": cseed, "score": c_score,
                                          "gf": cg, "only": cg_only, "free": c_free, "gt": c_gt})
                        chosen = min(range(len(cands)), key=lambda i: cands[i]["score"])
                        best = cands[chosen]
                        gen_frames, gen_only = best["gf"], best["only"]
                        free, gt_metrics = best["free"], best["gt"]
                        cand_log = [{"cand": c["cand"], "seed": c["seed"],
                                     "score": c["score"], "chosen": c["cand"] == chosen,
                                     **{k: c["free"].get(k) for k in _GEN_FREE_KEYS},
                                     **{k: c["gt"].get(k) for k in _GT_KEYS}}
                                    for c in cands]
                    else:
                        t0 = time.time()
                        gen_frames = _gen(base_seed)
                        gen_time += time.time() - t0
                        gen_only = gen_frames[args.num_cond_frames:args.num_cond_frames + num_gen]
                        gt_metrics = _gt(gen_frames)
                        free = gen_free_signals(gen_only, last_cond_frame)

                    rec = {"chunk": step_i + 1, "gen_start_frame": step_gen_start,
                           "gt_available": gt_metrics.get("psnr") == gt_metrics.get("psnr")}
                    rec.update({k: gt_metrics.get(k) for k in _GT_KEYS})
                    rec.update(free)
                    if args.method == "bestof":
                        rec["chosen_cand"] = int(chosen)
                        rec["search_k"] = len(cand_log)
                        rec["candidates"] = cand_log
                    chunk_records.append(rec)
                    stitched.append(np.clip(gen_only.astype(np.float32), 0, 1))

                    pick = (f" pick={chosen}/{len(cand_log)}" if args.method == "bestof" else "")
                    print(
                        f"  chunk {step_i+1:2d}: sharp={free['sharpness']:.4f} "
                        f"motion={free['temporal_motion']:.4f} "
                        f"colorful={free['colorfulness']:.4f} "
                        f"seam_ratio={free['seam_ratio']:.2f} "
                        f"psnr={rec['psnr'] if rec['psnr'] == rec['psnr'] else float('nan'):.2f}"
                        + pick + ("" if rec["gt_available"] else " (no GT)")
                    )
                    prev_gen = gen_frames
            finally:
                if wrapper is not None:
                    wrapper.remove_from_dit()
                    del wrapper
                    torch_gc()

            record = {
                "video_name": eval_name,
                "video_path": entry["video_path"],
                "caption": entry["caption"],
                "method": args.method,
                "rollout_mode": args.rollout_mode,
                "ref_signals": ref_sig if args.method == "bestof" else None,
                "search_k": args.search_k if args.method == "bestof" else None,
                "delta_norm": delta_norm,
                "stream_delta_norms": stream_norms if args.method == "delta_stream" else None,
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
        "method": args.method,
        "diagnostic": "longhorizon_drift",
        "rollout_mode": args.rollout_mode,
        "delta_steps": args.delta_steps if uses_delta else None,
        "delta_lr": args.delta_lr if uses_delta else None,
        "delta_placement": args.delta_placement if uses_delta else None,
        "tta_total_frames": tta_total_frames if uses_delta else None,
        "tta_context_frames": tta_context_frames if uses_delta else None,
        "stream_refit_steps": args.stream_refit_steps if args.method == "delta_stream" else None,
        "stream_refit_lr": stream_refit_lr if args.method == "delta_stream" else None,
        "stream_blend": args.stream_blend if args.method == "delta_stream" else None,
        "stream_target": args.stream_target if args.method == "delta_stream" else None,
        "search_k": args.search_k if args.method == "bestof" else None,
        "search_seam_weight": args.search_seam_weight if args.method == "bestof" else None,
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
