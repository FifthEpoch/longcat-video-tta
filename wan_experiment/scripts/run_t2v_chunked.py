#!/usr/bin/env python3
"""Chunked Wan T2V: NOTTA, always-BoN, or gated-BoN (cand0 = NOTTA seed).

Text start, then AR from own KV. Not I2V-from-still. Official
CausalInferencePipeline.inference() is one-shot; this runner replays
committed latents (T2V: no independent first frame) then denoises the
next chunk. Piece 0 is always seed 0 (shared prefix).

30 s = 6 × 21 latents (Self-Forcing native 5 s unit → 81 px × 6).
Do not add TTC.

    python wan_experiment/scripts/run_t2v_chunked.py \
        --method notta --horizon-s 30 --n 2 --prompt-file datasets/moviegen_128.txt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from i2v_verifier import (  # noqa: E402
    gen_free_signals,
    reference_signals,
    score_breakdown,
)
from run_i2v_chunked import (  # noqa: E402
    REF_WIN,
    _bootstrap_sf,
    _cand_seed,
    _chunk_rng,
    _decode_pixels,
    _denoise_chunk,
    _incoming_window,
    _json_float,
    _json_signals,
    _reset_caches,
    _seed_torch,
    hybrid_gate_decision,
)
from run_i2v_continuation import (  # noqa: E402
    FPS,
    FRAME_SEQ_PER_LATENT,
    LATENT_C,
    LATENT_H,
    LATENT_W,
    install_sdpa_attention_fallback,
    load_pipeline,
    write_mp4,
    _cuda_mem,
)


def t2v_pixel_frames(n_lat: int) -> int:
    """Wan VAE: 21 latents → 81 pixels (official Self-Forcing 5 s)."""
    if n_lat <= 0:
        return 0
    return 1 + 4 * (n_lat - 1)


def t2v_latents_for_horizon(seconds: float, chunk_latents: int) -> int:
    """Nearest multiple of chunk_latents at 16 fps / Wan VAE."""
    target_pix = max(int(round(seconds * FPS)), 9)
    n_lat = max(chunk_latents, int(round((target_pix - 1) / 4.0)) + 1)
    extra = (chunk_latents - n_lat % chunk_latents) % chunk_latents
    return n_lat + extra


def discover_prompts(prompt_file: Path, n: int) -> list[dict]:
    lines = [ln.strip() for ln in Path(prompt_file).read_text().splitlines() if ln.strip()]
    if not lines:
        raise FileNotFoundError(f"no prompts in {prompt_file}")
    items = []
    for i, prompt in enumerate(lines[:n]):
        stem = f"moviegen_{i:03d}"
        items.append({
            "prompt_index": i,
            "file_name": stem,
            "stem": stem,
            "prompt": prompt,
        })
    return items


def _cache_clean_latents_t2v(pipeline, latents, conditional_dict) -> None:
    """Replay committed T2V latents in blocks of num_frame_per_block. No still."""
    import torch

    bsz, n_lat = latents.shape[:2]
    if n_lat == 0:
        return
    block = int(pipeline.num_frame_per_block)
    if n_lat % block != 0:
        raise RuntimeError(
            f"committed T2V latents {n_lat} not a multiple of block={block}"
        )
    device = latents.device
    t = 0
    while t < n_lat:
        ts = torch.ones([bsz, block], device=device, dtype=torch.int64) * 0
        pipeline.generator(
            noisy_image_or_video=latents[:, t:t + block],
            conditional_dict=conditional_dict,
            timestep=ts,
            kv_cache=getattr(pipeline, "kv_cache_clean", None) or pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=t * pipeline.frame_seq_length,
        )
        t += block


def _cache_clean_latents_slices(pipeline, latents, conditional_dict, ranges) -> None:
    """Replay selected [start, end) latent spans at their original RoPE starts."""
    import torch

    bsz = latents.shape[0]
    block = int(pipeline.num_frame_per_block)
    device = latents.device
    for start, end in ranges:
        if end <= start:
            continue
        if (end - start) % block != 0:
            raise RuntimeError(
                f"slice [{start},{end}) length {end - start} not a multiple of {block}"
            )
        t = start
        while t < end:
            ts = torch.ones([bsz, block], device=device, dtype=torch.int64) * 0
            pipeline.generator(
                noisy_image_or_video=latents[:, t:t + block],
                conditional_dict=conditional_dict,
                timestep=ts,
                kv_cache=getattr(pipeline, "kv_cache_clean", None) or pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=t * pipeline.frame_seq_length,
            )
            t += block


def generate_chunked_t2v(
    pipeline,
    prompt: str,
    n_lat: int,
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
):
    import torch

    n_chunks = n_lat // chunk_latents
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])
    output = torch.zeros(
        [1, n_lat, LATENT_C, LATENT_H, LATENT_W],
        device=device, dtype=torch.bfloat16,
    )

    committed = 0
    ref = None
    committed_pixels = None
    incoming_prev = None
    chunk_logs = []

    for ci in range(n_chunks):
        incoming_signals = None
        incoming_devs = None
        incoming_drift = None
        incoming_delta = None
        gated_fired = False
        gate_reason = "forced_prefix"

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
            gated_fired, gate_reason = hybrid_gate_decision(
                ci, incoming_drift, incoming_prev, search_from_chunk,
                t_late=gate_threshold,
                t_ch1=gate_ch1_threshold,
                t_delta=gate_delta,
                t_delta_prev_min=gate_delta_prev_min,
            )
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
            if committed > 0:
                _cache_clean_latents_t2v(
                    pipeline, output[:, :committed], conditional_dict,
                )
            _denoise_chunk(pipeline, noise, committed, conditional_dict, output, rng)
            end = committed + chunk_latents
            pixels = _decode_pixels(pipeline, output[:, :end])
            n_committed_pix = t2v_pixel_frames(committed)
            if n_committed_pix == 0:
                gen_only = pixels
                last_cond = pixels[0]
            else:
                gen_only = pixels[n_committed_pix:]
                last_cond = pixels[n_committed_pix - 1]
            if ref is None:
                if pixels.shape[0] < REF_WIN:
                    raise RuntimeError("chunk 0 too short to build a 1 s reference")
                ref = reference_signals(pixels[:REF_WIN])
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
            "gated_fired": bool(gated_fired),
            "gate_reason": gate_reason,
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
            gate_s = (
                f" incoming={incoming_drift:.3f}{dlt} "
                f"fire={int(gated_fired)} reason={gate_reason}"
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf-root", required=True)
    ap.add_argument("--wan-dir", required=True)
    ap.add_argument("--sf-ckpt", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--chunk-latents", type=int, default=21,
                    help="Self-Forcing native 5 s unit")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=("notta", "always_bon", "gated_bon"),
                    default="notta")
    ap.add_argument("--search-k", type=int, default=4)
    ap.add_argument("--search-from-chunk", type=int, default=1)
    ap.add_argument("--seam-weight", type=float, default=1.0)
    ap.add_argument("--gate-threshold", type=float, default=2.0)
    ap.add_argument("--gate-ch1-threshold", type=float, default=0.8)
    ap.add_argument("--gate-delta", type=float, default=0.5)
    ap.add_argument("--gate-delta-prev-min", type=float, default=0.5)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    n_lat = t2v_latents_for_horizon(args.horizon_s, args.chunk_latents)
    if n_lat % args.chunk_latents != 0:
        raise SystemExit(f"n_lat={n_lat} not divisible by {args.chunk_latents}")
    n_pix = t2v_pixel_frames(n_lat)
    n_chunks = n_lat // args.chunk_latents

    sf_root = Path(args.sf_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    items = discover_prompts(Path(args.prompt_file), args.n)
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
    if not items:
        print("shard is empty; nothing to do")
        return 0

    torch = _bootstrap_sf(sf_root)
    _seed_torch(torch, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} torch={torch.__version__} task=T2V method={args.method} "
        f"horizon={args.horizon_s}s n_lat={n_lat} chunk_latents={args.chunk_latents} "
        f"n_chunks={n_chunks} n_pix={n_pix} k={args.search_k} "
        f"search_from={args.search_from_chunk} "
        f"n_items={len(items)}"
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
        device, n_cache_frames=n_lat + 2,
        independent_first_frame=False,
    )
    if int(pipeline.frame_seq_length) != FRAME_SEQ_PER_LATENT:
        raise RuntimeError(
            f"pipeline.frame_seq_length={pipeline.frame_seq_length} "
            f"!= {FRAME_SEQ_PER_LATENT}"
        )
    _iff = getattr(getattr(pipeline, "args", None), "independent_first_frame", None)
    if _iff:
        raise RuntimeError("T2V runner requires independent_first_frame=False")
    print(f"pipeline loaded in {time.time() - t_load:.1f}s")

    rows = []
    for i, item in enumerate(items):
        stem = (
            f"{item['stem']}_h{int(args.horizon_s)}s_"
            f"{args.method}_k{args.search_k}_s{args.seed}"
        )
        mp4 = out_dir / f"{stem}.mp4"
        meta_path = out_dir / f"{stem}.json"
        if mp4.is_file() and mp4.stat().st_size > 10_000:
            print(f"skip existing {mp4.name}")
            rows.append({"ok": True, "skipped": True, "mp4": str(mp4), **item})
            continue
        print(f"[{i+1}/{len(items)}] T2V {args.method} {item['file_name']}")
        _seed_torch(torch, args.seed)
        t0 = time.time()
        try:
            with torch.inference_mode():
                video, lat_shape, ref, chunk_logs = generate_chunked_t2v(
                    pipeline, item["prompt"],
                    n_lat, args.chunk_latents, args.seed, device,
                    args.method, args.search_k, args.search_from_chunk,
                    args.seam_weight, args.gate_threshold,
                    args.gate_ch1_threshold, args.gate_delta,
                    args.gate_delta_prev_min,
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
                "task": "t2v",
                "seconds": round(time.time() - t0, 2),
                "mp4": str(mp4),
                "n_frames": int(video.shape[0]),
                "hw": [int(video.shape[1]), int(video.shape[2])],
                "latent_shape": list(lat_shape),
                "n_gen_latent": n_lat,
                "chunk_latents": args.chunk_latents,
                "n_chunks": n_chunks,
                "method": args.method,
                "search_k": args.search_k,
                "search_from_chunk": args.search_from_chunk,
                "gate_threshold": args.gate_threshold,
                "gate_ch1_threshold": args.gate_ch1_threshold,
                "gate_delta": args.gate_delta,
                "gate_delta_prev_min": args.gate_delta_prev_min,
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
                "task": "t2v",
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
        "task": "t2v",
        "method": args.method,
        "horizon_s": args.horizon_s,
        "n_gen_latent": n_lat,
        "chunk_latents": args.chunk_latents,
        "n_chunks": n_chunks,
        "n_pix": n_pix,
        "search_k": args.search_k,
        "search_from_chunk": args.search_from_chunk,
        "gate_threshold": args.gate_threshold,
        "gate_ch1_threshold": args.gate_ch1_threshold,
        "gate_delta": args.gate_delta,
        "gate_delta_prev_min": args.gate_delta_prev_min,
        "seed": args.seed,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "prompt_file": str(Path(args.prompt_file).resolve()),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    return 0 if summary["n_ok"] == summary["n"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
