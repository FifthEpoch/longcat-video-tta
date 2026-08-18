#!/usr/bin/env python3
"""Chunked Wan I2V: NOTTA, always-BoN, or gated-BoN (cand0 = NOTTA seed).

Official CausalInferencePipeline.inference() only KV-caches initial_latent[:, :1]
when independent_first_frame=True, so we cannot pass a growing prefix back
through inference(). This runner replays committed latents into the KV cache
(same t=0 path as official I2V init), then denoises the next chunk.

Smoke (2 × 30 s): chunk 0 is always seed 0 (shared prefix / ref). always-BoN
searches chunks 1..N-1. Do not add TTC until this writes real video.

    python wan_experiment/scripts/run_i2v_chunked.py \
        --method notta --horizon-s 30 --n 2 --chunk-latents 24
    python wan_experiment/scripts/run_i2v_chunked.py \
        --method always_bon --search-k 4 --horizon-s 30 --n 2 --chunk-latents 24
    python wan_experiment/scripts/run_i2v_chunked.py \
        --method gated_bon --search-k 4 --gate-threshold 2.0 \
        --gate-ch1-threshold 0.8 --gate-delta 0.5 --gate-sticky \
        --horizon-s 30 --n 32
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from i2v_verifier import (  # noqa: E402
    gen_free_signals,
    reference_signals,
    score_breakdown,
)
from run_i2v_continuation import (  # noqa: E402
    FPS,
    FRAME_SEQ_PER_LATENT,
    LATENT_C,
    LATENT_H,
    LATENT_W,
    discover_items,
    encode_image,
    gen_latents_for_horizon,
    install_sdpa_attention_fallback,
    load_pipeline,
    pixel_frames,
    write_mp4,
    _cuda_mem,
)


REF_WIN = 16  # 1 s @ 16 fps, skip cond frame 0


def _json_float(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _json_signals(d: dict | None) -> dict | None:
    if not d:
        return None
    return {k: _json_float(d.get(k)) for k in d}


def _incoming_window(pixels: np.ndarray):
    win = min(REF_WIN, pixels.shape[0] - 1)
    if win < 1:
        return None
    return gen_free_signals(pixels[-win:], pixels[-win - 1])


def hybrid_gate_decision(
    chunk_idx: int,
    incoming: float,
    incoming_prev: float | None,
    search_from: int,
    t_late: float,
    t_ch1: float,
    t_delta: float,
    t_delta_prev_min: float,
) -> tuple[bool, str]:
    """Fire if ch1-early or late level or rising trend.

    Agreed 2026-08-17 after T=2.0 lost to always-on: a single global T
    cannot keep video 07 skipped (inc=0.68) and still catch 05 (0.87).
    """
    if chunk_idx < search_from:
        return False, "forced_prefix"
    reasons: list[str] = []
    if chunk_idx == 1 and incoming > t_ch1:
        reasons.append("ch1")
    if incoming > t_late:
        reasons.append("level")
    if incoming_prev is not None:
        delta = incoming - incoming_prev
        if delta > t_delta and incoming_prev > t_delta_prev_min:
            reasons.append("trend")
    if reasons:
        return True, "+".join(reasons)
    return False, "skip"


def search_while_sick_keep_on(
    incoming: float | None,
    outgoing: float | None,
    sick_min: float,
    recovery: float,
) -> tuple[bool, str | None]:
    """After a search: keep memory on only if still sick and not recovered.

    ``sick_min <= 0`` and ``recovery <= 0`` → forever stay-on (old sticky).
    Recovery is checked first (video 11: outgoing 1.11 is still >= 1.0
    but incoming fell by 1.27).
    """
    if incoming is not None and outgoing is not None and recovery > 0:
        if incoming - outgoing > recovery:
            return False, "recovered"
    if outgoing is not None and sick_min > 0 and outgoing < sick_min:
        return False, "healthy"
    return True, None


def _reset_caches(pipeline, batch_size, dtype, device) -> None:
    if pipeline.kv_cache1 is None:
        pipeline._initialize_kv_cache(batch_size, dtype, device)
    else:
        for blk in pipeline.kv_cache1:
            blk["global_end_index"].zero_()
            blk["local_end_index"].zero_()
    if getattr(pipeline, "crossattn_cache", None) is None:
        pipeline._initialize_crossattn_cache(batch_size, dtype, device)
    else:
        for blk in pipeline.crossattn_cache:
            blk["is_init"] = False


def _cache_clean_latents(pipeline, latents, conditional_dict) -> None:
    """Replay committed clean latents into KV (I2V: frame 0 alone, then blocks)."""
    import torch

    bsz, n_lat = latents.shape[:2]
    block = int(pipeline.num_frame_per_block)
    device = latents.device
    t = 0
    ts1 = torch.ones([bsz, 1], device=device, dtype=torch.int64) * 0
    pipeline.generator(
        noisy_image_or_video=latents[:, :1],
        conditional_dict=conditional_dict,
        timestep=ts1,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=0,
    )
    t = 1
    while t < n_lat:
        n = min(block, n_lat - t)
        if n != block:
            raise RuntimeError(
                f"committed latent tail {n_lat - t} is not a multiple of {block}"
            )
        ts = torch.ones([bsz, n], device=device, dtype=torch.int64) * 0
        pipeline.generator(
            noisy_image_or_video=latents[:, t:t + n],
            conditional_dict=conditional_dict,
            timestep=ts,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=t * pipeline.frame_seq_length,
        )
        t += n


def _denoise_chunk(pipeline, noise, start_frame, conditional_dict, output, rng) -> None:
    """Official Step 3 loop for one chunk. Writes output[:, start:start+n].

    ``rng`` must seed add_noise. Job 15883525 vs 15883526: chunk 0 cand0
    scores already differed (3.305 vs 2.992) because randn_like used the
    global CUDA RNG. Without this, cand0 is not a NOTTA twin.
    """
    import torch

    bsz, n_gen = noise.shape[:2]
    block = int(pipeline.num_frame_per_block)
    if n_gen % block != 0:
        raise ValueError(f"chunk n_gen={n_gen} not divisible by block={block}")
    device = noise.device
    cur = start_frame
    consumed = 0
    for _ in range(n_gen // block):
        noisy_input = noise[:, consumed:consumed + block]
        for index, current_timestep in enumerate(pipeline.denoising_step_list):
            timestep = torch.ones(
                [bsz, block], device=device, dtype=torch.int64
            ) * current_timestep
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=cur * pipeline.frame_seq_length,
            )
            if index < len(pipeline.denoising_step_list) - 1:
                next_timestep = pipeline.denoising_step_list[index + 1]
                extra = torch.randn(
                    denoised_pred.flatten(0, 1).shape,
                    device=device, dtype=denoised_pred.dtype, generator=rng,
                )
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    extra,
                    next_timestep * torch.ones(
                        [bsz * block], device=device, dtype=torch.long
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
        output[:, cur:cur + block] = denoised_pred
        context_timestep = torch.ones_like(timestep) * pipeline.args.context_noise
        pipeline.generator(
            noisy_image_or_video=denoised_pred,
            conditional_dict=conditional_dict,
            timestep=context_timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=cur * pipeline.frame_seq_length,
        )
        cur += block
        consumed += block


def _decode_pixels(pipeline, latents) -> np.ndarray:
    """latents [B,T,C,H,W] -> [T,H,W,C] float[0,1]."""
    video = pipeline.vae.decode_to_pixel(latents, use_cache=False)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    arr = video[0].float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    try:
        pipeline.vae.model.clear_cache()
    except Exception:
        pass
    return arr


def _cand_seed(base: int, cand: int) -> int:
    return int(base) if cand == 0 else int(base) + cand * 100003


def _chunk_rng(device, base: int, cand: int, chunk: int):
    """One CUDA generator per (video-seed, cand, chunk). cand0/chunk i matches NOTTA."""
    import torch

    g = torch.Generator(device=device)
    g.manual_seed(int(_cand_seed(base, cand)) + 10007 * int(chunk))
    return g


def _seed_torch(torch_mod, seed: int) -> None:
    """Process-level invariance. Per-chunk noise still uses _chunk_rng."""
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch_mod.manual_seed(int(seed))
    if torch_mod.cuda.is_available():
        torch_mod.cuda.manual_seed_all(int(seed))
    torch_mod.backends.cudnn.deterministic = True
    torch_mod.backends.cudnn.benchmark = False
    torch_mod.backends.cuda.matmul.allow_tf32 = False
    torch_mod.backends.cudnn.allow_tf32 = False
    try:
        torch_mod.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    print("rng: deterministic flags on (warn_only); per-chunk CUDA Generator")


def generate_chunked(
    pipeline,
    image_path: Path,
    prompt: str,
    n_gen: int,
    chunk_latents: int,
    seed: int,
    device,
    method: str,
    search_k: int,
    search_from_chunk: int,
    seam_weight: float,
    gate_threshold: float,
    gate_ch1_threshold: float = 0.8,
    gate_delta: float = 0.5,
    gate_delta_prev_min: float = 0.5,
    gate_sticky: bool = False,
    gate_sick_min: float = 0.0,
    gate_recovery: float = 0.0,
):
    import torch

    n_chunks = n_gen // chunk_latents
    initial = encode_image(pipeline, Path(image_path), device)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])
    total_lat = 1 + n_gen
    output = torch.zeros(
        [1, total_lat, LATENT_C, LATENT_H, LATENT_W],
        device=device, dtype=torch.bfloat16,
    )
    output[:, :1] = initial[:, :1]

    committed = 1
    ref = None
    committed_pixels = None
    incoming_prev = None
    already_on = False
    chunk_logs = []

    for ci in range(n_chunks):
        incoming_signals = None
        incoming_devs = None
        incoming_drift = None
        incoming_delta = None
        gated_fired = False
        gate_reason = "forced_prefix"
        already_on_before = already_on

        if ref is not None and committed_pixels is not None:
            incoming_signals = _incoming_window(committed_pixels)
            if incoming_signals is not None:
                incoming_devs = score_breakdown(
                    incoming_signals, ref, seam_weight=0.0,
                )
                incoming_drift = incoming_devs["score"]
                if incoming_prev is not None:
                    incoming_delta = incoming_drift - incoming_prev

        if method == "always_bon" and ci >= search_from_chunk:
            n_try = search_k
            gated_fired = True
            gate_reason = "always"
        elif method == "gated_bon" and ci >= search_from_chunk and incoming_drift is not None:
            alarm_fired, alarm_reason = hybrid_gate_decision(
                ci, incoming_drift, incoming_prev, search_from_chunk,
                t_late=gate_threshold,
                t_ch1=gate_ch1_threshold,
                t_delta=gate_delta,
                t_delta_prev_min=gate_delta_prev_min,
            )
            if gate_sticky and already_on:
                gated_fired = True
                gate_reason = alarm_reason if alarm_fired else "already_on"
            else:
                gated_fired = alarm_fired
                gate_reason = alarm_reason
            n_try = search_k if gated_fired else 1
        elif method == "notta":
            n_try = 1
            gate_reason = "notta" if ci >= search_from_chunk else "forced_prefix"
        else:
            n_try = 1
            gate_reason = "forced_prefix"

        cands = []
        for c in range(n_try):
            cseed = _cand_seed(seed, c)
            rng = _chunk_rng(device, seed, c, ci)
            noise = torch.randn(
                [1, chunk_latents, LATENT_C, LATENT_H, LATENT_W],
                device=device, dtype=torch.bfloat16, generator=rng,
            )
            _reset_caches(pipeline, 1, output.dtype, device)
            _cache_clean_latents(pipeline, output[:, :committed], conditional_dict)
            _denoise_chunk(pipeline, noise, committed, conditional_dict, output, rng)
            end = committed + chunk_latents
            pixels = _decode_pixels(pipeline, output[:, :end])
            n_committed_pix = pixel_frames(committed - 1) if committed > 1 else 1
            gen_only = pixels[n_committed_pix:]
            last_cond = pixels[n_committed_pix - 1]
            if ref is None:
                if pixels.shape[0] < 1 + REF_WIN:
                    raise RuntimeError("chunk 0 too short to build a 1 s reference")
                ref = reference_signals(pixels[1:1 + REF_WIN])
            free = gen_free_signals(gen_only, last_cond)
            br = score_breakdown(free, ref, seam_weight=seam_weight)
            score = br["score"]
            cands.append({
                "cand": c,
                "seed": cseed,
                "score": score,
                "latents": output[:, committed:end].detach().clone(),
                "pixels": pixels,
                "free": {k: free[k] for k in free},
                "breakdown": br,
            })
            print(
                f"    chunk {ci} cand{c} seed={cseed} score={score:.4f} "
                f"sharp={free['sharpness']:.4g} motion={free['temporal_motion']:.4g} "
                f"dev_sharp={br['dev_sharpness']:.3f} "
                f"dev_motion={br['dev_temporal_motion']:.3f} "
                f"seam={br['seam_term']:.3f}",
                flush=True,
            )
        chosen = min(range(len(cands)), key=lambda i: cands[i]["score"])
        best = cands[chosen]
        output[:, committed:committed + chunk_latents] = best["latents"]
        committed += chunk_latents
        committed_pixels = best["pixels"]
        cand0_score = cands[0]["score"]
        chosen_minus_cand0 = best["score"] - cand0_score
        outgoing_signals = _incoming_window(committed_pixels) if ref is not None else None
        outgoing_devs = (
            score_breakdown(outgoing_signals, ref, seam_weight=0.0)
            if outgoing_signals is not None and ref is not None else None
        )
        outgoing_drift = outgoing_devs["score"] if outgoing_devs is not None else None
        recovery = None
        if incoming_drift is not None and outgoing_drift is not None:
            recovery = incoming_drift - outgoing_drift
        gate_off_reason = None
        if gated_fired and n_try > 1:
            already_on = True
            if gate_sticky and (gate_sick_min > 0 or gate_recovery > 0):
                keep, gate_off_reason = search_while_sick_keep_on(
                    incoming_drift, outgoing_drift,
                    gate_sick_min, gate_recovery,
                )
                already_on = keep
        rec = {
            "chunk": ci,
            "chosen_cand": int(chosen),
            "search_k": n_try,
            "incoming_drift": _json_float(incoming_drift),
            "incoming_prev": _json_float(incoming_prev),
            "incoming_delta": _json_float(incoming_delta),
            "incoming_signals": _json_signals(incoming_signals),
            "incoming_devs": _json_signals(incoming_devs),
            "outgoing_drift": _json_float(outgoing_drift),
            "outgoing_devs": _json_signals(outgoing_devs),
            "recovery": _json_float(recovery),
            "gated_fired": bool(gated_fired),
            "gate_reason": gate_reason,
            "gate_sticky": bool(gate_sticky),
            "gate_sick_min": gate_sick_min,
            "gate_recovery": gate_recovery,
            "already_on_before": bool(already_on_before),
            "already_on_after": bool(already_on),
            "gate_off_reason": gate_off_reason,
            "cand0_score": _json_float(cand0_score),
            "chosen_score": _json_float(best["score"]),
            "chosen_minus_cand0": _json_float(chosen_minus_cand0),
            "chosen_breakdown": _json_signals(best["breakdown"]),
            "candidates": [
                {
                    "cand": c["cand"],
                    "seed": c["seed"],
                    "score": _json_float(c["score"]),
                    "chosen": c["cand"] == chosen,
                    "signals": _json_signals(c["free"]),
                    "devs": _json_signals(c["breakdown"]),
                    **{k: _json_float(c["free"][k]) for k in c["free"]},
                }
                for c in cands
            ],
        }
        chunk_logs.append(rec)
        if incoming_drift is not None:
            incoming_prev = incoming_drift
        gate_s = ""
        if incoming_drift is not None:
            dlt = f" Δ={incoming_delta:+.3f}" if incoming_delta is not None else ""
            off = f" off={gate_off_reason}" if gate_off_reason else ""
            gate_s = (
                f" incoming={incoming_drift:.3f}{dlt} "
                f"fire={int(gated_fired)} reason={gate_reason}{off}"
            )
        print(
            f"  chunk {ci}: pick={chosen}/{n_try} score={best['score']:.4f} "
            f"vs_cand0={chosen_minus_cand0:+.3f}{gate_s}",
            flush=True,
        )

    pixels = _decode_pixels(pipeline, output)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pixels, tuple(output.shape), ref, chunk_logs


def _bootstrap_sf(sf_root: Path):
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)
    import torch
    torch._dynamo.config.disable = True
    torch.set_grad_enabled(False)
    return torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf-root", required=True)
    ap.add_argument("--wan-dir", required=True)
    ap.add_argument("--sf-ckpt", required=True)
    ap.add_argument("--i2v-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--chunk-s", type=float, default=5.0)
    ap.add_argument("--chunk-latents", type=int, default=0,
                    help="0 = 24 if horizon>=29 else gen_latents_for_horizon(chunk-s)")
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=("notta", "always_bon", "gated_bon"),
                    default="notta")
    ap.add_argument("--search-k", type=int, default=4)
    ap.add_argument("--search-from-chunk", type=int, default=1,
                    help="first chunk index to search (0=search prefix too)")
    ap.add_argument("--seam-weight", type=float, default=1.0)
    ap.add_argument("--gate-threshold", type=float, default=2.0,
                    help="gated_bon late-drift level: fire if incoming > this")
    ap.add_argument("--gate-ch1-threshold", type=float, default=0.8,
                    help="gated_bon: fire chunk 1 if incoming > this")
    ap.add_argument("--gate-delta", type=float, default=0.5,
                    help="gated_bon: fire if incoming-incoming_prev > this")
    ap.add_argument("--gate-delta-prev-min", type=float, default=0.5,
                    help="gated_bon: trend only if incoming_prev > this (keeps 06 skipped)")
    ap.add_argument("--gate-sticky", action="store_true",
                    help="once any alarm fires on a video, keep searching later pieces")
    ap.add_argument("--gate-sick-min", type=float, default=0.0,
                    help="with --gate-sticky: turn memory off if last-second "
                         "outgoing < this (0=disabled; search-while-sick uses 1.0)")
    ap.add_argument("--gate-recovery", type=float, default=0.0,
                    help="with --gate-sticky: turn memory off if incoming-outgoing "
                         "> this (0=disabled; search-while-sick uses 0.5)")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    n_gen = gen_latents_for_horizon(args.horizon_s)
    if args.chunk_latents > 0:
        chunk_latents = args.chunk_latents
    else:
        chunk_latents = 24 if args.horizon_s >= 29 else gen_latents_for_horizon(args.chunk_s)
    if n_gen % chunk_latents != 0:
        raise SystemExit(
            f"n_gen={n_gen} not divisible by chunk_latents={chunk_latents}. "
            f"For 30 s use --chunk-latents 24 (5×24=120)."
        )
    n_pix = pixel_frames(n_gen)
    n_chunks = n_gen // chunk_latents

    sf_root = Path(args.sf_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    items = discover_items(Path(args.i2v_dir), args.n)
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
    if not items:
        print("shard is empty; nothing to do")
        return 0

    torch = _bootstrap_sf(sf_root)
    _seed_torch(torch, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} torch={torch.__version__} method={args.method} "
        f"horizon={args.horizon_s}s n_gen={n_gen} chunk_latents={chunk_latents} "
        f"n_chunks={n_chunks} n_pix={n_pix} k={args.search_k} "
        f"search_from={args.search_from_chunk} "
        f"gate_late={args.gate_threshold} gate_ch1={args.gate_ch1_threshold} "
        f"gate_delta={args.gate_delta} gate_delta_prev_min={args.gate_delta_prev_min} "
        f"gate_sticky={int(args.gate_sticky)} "
        f"gate_sick_min={args.gate_sick_min} gate_recovery={args.gate_recovery} "
        f"n_items={len(items)}"
    )
    print(
        f"KV current_start uses pipeline.frame_seq_length "
        f"(must stay {FRAME_SEQ_PER_LATENT}, not generator.seq_len)"
    )

    install_sdpa_attention_fallback()
    try:
        import wan.modules.causal_model as _cm
        from torch.nn.attention.flex_attention import flex_attention as _eager_fa
        _cm.flex_attention = _eager_fa
        print("flex_attention: eager (torch.compile disabled)")
    except Exception as e:
        print(f"flex_attention: leave as-is ({type(e).__name__}: {e})")

    t_load = time.time()
    pipeline = load_pipeline(
        sf_root, Path(args.wan_dir), Path(args.sf_ckpt),
        device, n_cache_frames=1 + n_gen + 2,
    )
    if int(pipeline.frame_seq_length) != FRAME_SEQ_PER_LATENT:
        raise RuntimeError(
            f"pipeline.frame_seq_length={pipeline.frame_seq_length} "
            f"!= {FRAME_SEQ_PER_LATENT}"
        )
    print(f"pipeline loaded in {time.time() - t_load:.1f}s")

    rows = []
    for i, item in enumerate(items):
        stem = (
            f"{i:03d}_{item['stem']}_h{int(args.horizon_s)}s_"
            f"{args.method}_k{args.search_k}_s{args.seed}"
        )
        mp4 = out_dir / f"{stem}.mp4"
        meta_path = out_dir / f"{stem}.json"
        if mp4.is_file() and mp4.stat().st_size > 10_000:
            print(f"skip existing {mp4.name}")
            rows.append({"ok": True, "skipped": True, "mp4": str(mp4), **item})
            continue
        print(f"[{i+1}/{len(items)}] {args.method} {item['file_name']!r}")
        _seed_torch(torch, args.seed)
        t0 = time.time()
        try:
            with torch.inference_mode():
                video, lat_shape, ref, chunk_logs = generate_chunked(
                    pipeline, item["image_path"], item["prompt"],
                    n_gen, chunk_latents, args.seed, device,
                    args.method, args.search_k, args.search_from_chunk,
                    args.seam_weight, args.gate_threshold,
                    args.gate_ch1_threshold, args.gate_delta,
                    args.gate_delta_prev_min, args.gate_sticky,
                    args.gate_sick_min, args.gate_recovery,
                )
            write_mp4(mp4, video, fps=FPS)
            n_div = sum(
                1 for ch in chunk_logs
                if ch["search_k"] > 1 and ch["chosen_cand"] != 0
            )
            n_fire = sum(1 for ch in chunk_logs if ch.get("gated_fired"))
            last = chunk_logs[-1] if chunk_logs else {}
            reason_counts = {}
            for ch in chunk_logs:
                r = ch.get("gate_reason") or "unknown"
                reason_counts[r] = reason_counts.get(r, 0) + 1
            rec = {
                "ok": True,
                "seconds": round(time.time() - t0, 2),
                "mp4": str(mp4),
                "n_frames": int(video.shape[0]),
                "hw": [int(video.shape[1]), int(video.shape[2])],
                "latent_shape": list(lat_shape),
                "n_gen_latent": n_gen,
                "chunk_latents": chunk_latents,
                "n_chunks": n_chunks,
                "method": args.method,
                "search_k": args.search_k,
                "search_from_chunk": args.search_from_chunk,
                "gate_threshold": args.gate_threshold,
                "gate_ch1_threshold": args.gate_ch1_threshold,
                "gate_delta": args.gate_delta,
                "gate_delta_prev_min": args.gate_delta_prev_min,
                "gate_sticky": bool(args.gate_sticky),
                "gate_sick_min": args.gate_sick_min,
                "gate_recovery": args.gate_recovery,
                "n_divergent_chunks": n_div,
                "n_gated_fired": n_fire,
                "gate_reason_counts": reason_counts,
                "incoming_series": [ch.get("incoming_drift") for ch in chunk_logs],
                "outgoing_series": [ch.get("outgoing_drift") for ch in chunk_logs],
                "gate_reasons": [ch.get("gate_reason") for ch in chunk_logs],
                "chosen_minus_cand0_series": [
                    ch.get("chosen_minus_cand0") for ch in chunk_logs
                ],
                "last_chunk_score": last.get("chosen_score"),
                "last_chunk_cand0_score": last.get("cand0_score"),
                "last_chunk_chosen_minus_cand0": last.get("chosen_minus_cand0"),
                "last_chunk_breakdown": last.get("chosen_breakdown"),
                "ref_signals": _json_signals(ref),
                "chunks": chunk_logs,
                "horizon_s_requested": args.horizon_s,
                "seed": args.seed,
                **item,
            }
            print(
                f"  wrote {mp4.name}  T={video.shape[0]}  {rec['seconds']}s  "
                f"divergent_chunks={n_div} gated_fired={n_fire} "
                f"last={rec['last_chunk_score']} reasons={reason_counts}",
                flush=True,
            )
        except Exception as e:
            rec = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "seconds": round(time.time() - t0, 2),
                "method": args.method,
                **item,
            }
            print(f"  FAIL {rec['error']}")
            print(rec["traceback"])
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            _cuda_mem("after_fail")
        meta_path.write_text(json.dumps(rec, indent=2))
        rows.append(rec)
        if rec.get("ok") and rec.get("chunks"):
            trace_path = out_dir / "gate_trace.jsonl"
            with trace_path.open("a") as tf:
                for ch in rec["chunks"]:
                    tf.write(json.dumps({
                        "file_name": rec.get("file_name"),
                        "stem": rec.get("stem"),
                        "method": args.method,
                        "seed": args.seed,
                        **{k: ch[k] for k in ch if k != "candidates"},
                        "n_candidates": len(ch.get("candidates") or []),
                    }) + "\n")

    summary = {
        "n": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "method": args.method,
        "horizon_s": args.horizon_s,
        "n_gen_latent": n_gen,
        "chunk_latents": chunk_latents,
        "n_chunks": n_chunks,
        "n_pix": n_pix,
        "search_k": args.search_k,
        "search_from_chunk": args.search_from_chunk,
        "gate_threshold": args.gate_threshold,
        "gate_ch1_threshold": args.gate_ch1_threshold,
        "gate_delta": args.gate_delta,
        "gate_delta_prev_min": args.gate_delta_prev_min,
        "gate_sticky": bool(args.gate_sticky),
        "gate_sick_min": args.gate_sick_min,
        "gate_recovery": args.gate_recovery,
        "seed": args.seed,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    return 0 if summary["n_ok"] == summary["n"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
