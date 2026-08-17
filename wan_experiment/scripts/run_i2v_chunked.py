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
        --method gated_bon --search-k 4 --gate-threshold 2.0 --horizon-s 30 --n 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from i2v_verifier import gen_free_signals, reference_signals, verifier_score  # noqa: E402
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
    chunk_logs = []

    for ci in range(n_chunks):
        incoming_drift = None
        gated_fired = False
        if method == "always_bon" and ci >= search_from_chunk:
            n_try = search_k
        elif method == "gated_bon" and ci >= search_from_chunk and ref is not None:
            pix = committed_pixels
            win = min(REF_WIN, pix.shape[0] - 1)
            incoming = gen_free_signals(pix[-win:], pix[-win - 1])
            incoming_drift = verifier_score(incoming, ref, seam_weight=0.0)
            gated_fired = incoming_drift > gate_threshold
            n_try = search_k if gated_fired else 1
        else:
            n_try = 1
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
            score = verifier_score(free, ref, seam_weight=seam_weight)
            cands.append({
                "cand": c,
                "seed": cseed,
                "score": score,
                "latents": output[:, committed:end].detach().clone(),
                "pixels": pixels,
                "free": {k: free[k] for k in free},
            })
            print(
                f"    chunk {ci} cand{c} seed={cseed} score={score:.4f} "
                f"sharp={free['sharpness']:.4g} motion={free['temporal_motion']:.4g}",
                flush=True,
            )
        chosen = min(range(len(cands)), key=lambda i: cands[i]["score"])
        best = cands[chosen]
        output[:, committed:committed + chunk_latents] = best["latents"]
        committed += chunk_latents
        committed_pixels = best["pixels"]
        chunk_logs.append({
            "chunk": ci,
            "chosen_cand": int(chosen),
            "search_k": n_try,
            "incoming_drift": incoming_drift,
            "gated_fired": gated_fired,
            "candidates": [
                {
                    "cand": c["cand"],
                    "seed": c["seed"],
                    "score": c["score"],
                    "chosen": c["cand"] == chosen,
                    **c["free"],
                }
                for c in cands
            ],
        })
        gate_s = ""
        if incoming_drift is not None:
            gate_s = f" incoming={incoming_drift:.3f} fire={int(gated_fired)}"
        print(
            f"  chunk {ci}: pick={chosen}/{n_try} score={best['score']:.4f}{gate_s}",
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
                    help="gated_bon: search iff incoming last-1s composite > this")
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
        f"search_from={args.search_from_chunk} gate={args.gate_threshold} "
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
                )
            write_mp4(mp4, video, fps=FPS)
            n_div = sum(
                1 for ch in chunk_logs
                if ch["search_k"] > 1 and ch["chosen_cand"] != 0
            )
            n_fire = sum(1 for ch in chunk_logs if ch.get("gated_fired"))
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
                "n_divergent_chunks": n_div,
                "n_gated_fired": n_fire,
                "ref_signals": ref,
                "chunks": chunk_logs,
                "horizon_s_requested": args.horizon_s,
                "seed": args.seed,
                **item,
            }
            print(
                f"  wrote {mp4.name}  T={video.shape[0]}  {rec['seconds']}s  "
                f"divergent_chunks={n_div} gated_fired={n_fire}",
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
