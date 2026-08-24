#!/usr/bin/env python3
"""Chunked Wan V2V: real video prefix, then AR. Sampling-space bake-off.

Piece 0 is a real Panda prefix (9 latents ≈ 2.1 s). Never searched.
Then 6 × 21 latents (~30 s) of generated tail.

Methods:
  notta          — one seed, default shift=8 / cfg=1
  seed_bon       — k=4 seeds, pick lowest two-sided prefix deviation
  motion_bon     — k=4 seeds, pick highest |Δframe| (falsified; keep for audit)
  shift_search   — same seed, shift in {8, 5, 12} (probe: dead on this DMD)
  backtrack      — rewind a dead tail and resample (falsified; keep for audit)
  hinge_bon      — k=4, prefix-match pick (motion hinge, no extra-twitch reward)
  late_bon       — seed_bon only when incoming motion < 0.7× prefix or last 2 chunks
  hist_drop      — full history vs last-3-latent history vs extra seeds; hinge pick
  good_backtrack — resample only if the just-written chunk collapsed *and* the
                   previous commit was good (≥0.8× prefix motion)
  cached_bon     — seed_bon pick, KV replayed once per chunk then snapshotted
  sink           — k=1, replay prefix + last window only (attention-sink approx)
  quiet_bon      — seed_bon only if real prefix_motion < 0.018; else k=1 (hot skip)
  tail_hist      — k=1, replay last 3 latents only (short history, no search)
  live_bon       — seed_bon only if prefix_motion >= 0.012 (invert quiet_bon)
  live_hist      — hist_drop candidates only if prefix is live; else k=1
  longlive_notta — LongLive-1.3B student, k=1, trained sink=3 / window=12
  longlive_sink  — LongLive + prefix+window replay (sink actually trained-in)
  longlive_prefix_sink — LongLive notta with sink_size=9 (whole prefix pinned)
  longlive_live_bon — live_bon on the LongLive student
  rolling_notta  — Rolling Forcing native sampler, real prefix, k=1
  rolling_rho_lo — RF host, per-block init-noise × (h/H)^0.5 (more early noise)
  rolling_rho_hi — RF host, per-block init-noise × (h/H)^2.0 (cleaner near)
  rolling_adapt  — RF host, ρ from prefix_motion (still=2, mid=1, hot=0.5)
  rolling_look   — RF host, k=4 lookahead on new-noise windows; seam pick
                   with trust reject (motion < 0.8× cand0 stays cand0)
  sf_roll        — SF weights + RF rolling window sampler (H1 cross)
  rf_chunk       — RF weights + SF chunked sampler (H1 cross)
  sf_recache     — SF chunked; VAE re-encode last 9 latents each chunk (H4)
  rf_recache     — RF rolling; VAE re-encode last 9 latents every 21 (H4)
  appear_bon     — k=4, pick lowest appearance/seam (motion dropped)
  live_appear    — appear_bon only if prefix_motion >= 0.012
  pseudo_gate    — generate held-out last-3 prefix latents; search tail iff
                   some seed beats notta MAE on that real B
  pseudo_appear  — same gate, appearance pick on the tail
  noise_probe    — k=1 notta, log first-step residual stats (U_t)
  noise_bon      — search extra seeds iff cand0 U_t >= tau; appear pick
  knob_probe     — first gen chunk only; grid shift × cfg; no 30 s write
  sf_rewind      — SF chunked; resample a chunk if motion < 0.8× previous
  sf_sick_search — SF chunked; k=4 only after a sick freeze; max-motion + trust
  sf_pseudo      — SF chunked; hold out last 3 prefix latents; search if extra seed wins B
  sf_always_search — SF chunked; always k=4; same motion+trust pick as sf_pseudo (no gate)
  rf_always_search — RF rolling; always k=4; same motion+trust pick as rf_sick/rf_pseudo (no gate)
  sf_sink        — SF chunked + LongLive-style sink_size (not HG-f). Not sf_roll.

No TTC. Do not scale I2V-32. Do not put these on the RF rolling sampler.

    python wan_experiment/scripts/run_v2v_chunked.py \
        --method notta --horizon-s 30 --n 2 --video-dir datasets/panda_1000_480p
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from i2v_verifier import (  # noqa: E402
    appear_score,
    gen_free_signals,
    motion_pick_score,
    prefix_match_score,
    reference_signals,
    score_breakdown,
)
from run_i2v_chunked import (  # noqa: E402
    REF_WIN,
    _active_kv,
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
)
from run_i2v_continuation import (  # noqa: E402
    FPS,
    FRAME_SEQ_PER_LATENT,
    LATENT_C,
    LATENT_H,
    LATENT_W,
    PIXEL_H,
    PIXEL_W,
    install_sdpa_attention_fallback,
    load_pipeline,
    write_mp4,
    _cuda_mem,
)
from run_t2v_chunked import (  # noqa: E402
    _cache_clean_latents_slices,
    _cache_clean_latents_t2v,
    t2v_latents_for_horizon,
    t2v_pixel_frames,
)

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".webm", ".mov"}
PREFIX_LATENTS_DEFAULT = 9
SHIFT_GRID = (8.0, 5.0, 12.0)
CFG_GRID = (1.0, 3.0, 5.0)
DEFAULT_SHIFT = 8.0
DEFAULT_CFG = 1.0
METHODS = (
    "notta", "seed_bon", "always_bon", "motion_bon",
    "shift_search", "backtrack", "knob_probe",
    "hinge_bon", "late_bon", "hist_drop", "good_backtrack",
    "cached_bon", "sink", "quiet_bon", "tail_hist",
    "live_bon", "live_hist",
    "longlive_notta", "longlive_sink", "longlive_live_bon",
    "longlive_prefix_sink", "rolling_notta",
    "rolling_rho_lo", "rolling_rho_hi", "rolling_adapt", "rolling_look",
    "sf_roll", "rf_chunk", "sf_recache", "rf_recache",
    "rf_rewind", "rf_sick_search", "rf_pseudo", "rf_sink",
    "sf_rewind", "sf_sick_search", "sf_pseudo", "sf_always_search",
    "rf_always_search", "sf_sink",
    "appear_bon", "live_appear", "pseudo_gate", "pseudo_appear",
    "noise_probe", "noise_bon",
)
TAIL_HISTORY_LATENTS = 3
SINK_WINDOW_LATENTS = 21
LATE_MOTION_FRAC = 0.7
GOOD_SAVE_FRAC = 0.8
# Search only if the real prefix is below this. N=32: seed_bon went 0/7
# on notta-tail≥0.020 (hot). 0.018 sits between mid and hot.
QUIET_SEARCH_MAX = 0.018
# Invert quiet_bon: search only when the prefix itself is moving.
# N=8 cand logs: 0007=0.070 live recovery; 0002/0003≈0.0008 stills.
LIVE_SEARCH_MIN = 0.012
_LIVE_SEARCH_MIN = LIVE_SEARCH_MIN
ROLL_STILL_MIN = 0.012
ROLL_HOT_MIN = 0.03
ROLL_TRUST_FRAC = 0.8
ROLL_LOOK_EVERY_BLOCKS = 7
PSEUDO_B_LATENTS = 3
PSEUDO_GAMMA = 0.0
_PSEUDO_GAMMA = PSEUDO_GAMMA
NOISE_TAU = 0.04
_NOISE_TAU = NOISE_TAU
RECACHE_LATENTS = 9
RECACHE_EVERY_LATENTS = 21
RF_SICK_DROP = 0.8
RF_CONTROLLER_METHODS = frozenset({
    "rf_rewind", "rf_sick_search", "rf_pseudo", "rf_sink",
    "rf_always_search",
})
ROLLING_HOST_METHODS = frozenset({"rf_chunk", "rf_recache"}) | RF_CONTROLLER_METHODS
ROLLING_SAMPLER_METHODS = frozenset({"sf_roll", "rf_recache"}) | RF_CONTROLLER_METHODS


def _v2v_host_name(method: str) -> str:
    """Host is the checkpoint, not the method prefix."""
    if method.startswith("longlive"):
        return "longlive"
    if method.startswith("rolling") or method in ROLLING_HOST_METHODS:
        return "rolling"
    return "sf"


def _uses_rolling_sampler(method: str) -> bool:
    if method == "rf_chunk":
        return False
    return method.startswith("rolling") or method in ROLLING_SAMPLER_METHODS


def prefix_pixel_count(n_lat: int) -> int:
    if n_lat <= 0:
        return 0
    return 1 + 4 * (n_lat - 1)


def _load_v2v_captions(video_dir: Path) -> dict[str, str]:
    """Best-effort file_name / stem → caption."""
    out: dict[str, str] = {}
    hits = []
    for name in (
        "captions.json", "caption_embeddings.json", "metadata.json",
        "prompts.json", "panda_captions.json",
    ):
        p = video_dir / name
        if p.is_file():
            hits.append(p)
        parent = video_dir.parent / name
        if parent.is_file():
            hits.append(parent)
    hits.extend(sorted(video_dir.glob("*caption*.json"))[:8])
    for p in hits:
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        rows = data
        if isinstance(data, dict):
            if "captions" in data and isinstance(data["captions"], list):
                rows = data["captions"]
            elif all(isinstance(v, str) for v in data.values()):
                for k, v in data.items():
                    out[Path(str(k)).name] = v
                    out[Path(str(k)).stem] = v
                continue
            else:
                rows = data.get("items") or data.get("videos") or []
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            fn = (
                item.get("file_name") or item.get("filename")
                or item.get("video") or item.get("path")
                or item.get("video_id")
            )
            cap = item.get("caption") or item.get("prompt") or item.get("text")
            if fn and cap:
                out[Path(str(fn)).name] = str(cap)
                out[Path(str(fn)).stem] = str(cap)
    return out


def discover_v2v_items(video_dir: Path, n: int) -> list[dict]:
    video_dir = video_dir.resolve()
    if not video_dir.is_dir():
        raise FileNotFoundError(f"video dir missing: {video_dir}")
    captions = _load_v2v_captions(video_dir)
    vids = sorted(
        p for p in video_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )
    if not vids:
        raise FileNotFoundError(f"no videos under {video_dir}")
    items = []
    seen: set[str] = set()
    for p in vids:
        key = p.name
        if key in seen:
            continue
        seen.add(key)
        prompt = (
            captions.get(key)
            or captions.get(p.stem)
            or p.stem.replace("_", " ")
        )
        items.append({
            "video_path": str(p),
            "file_name": key,
            "stem": p.stem[:80].replace(" ", "_"),
            "prompt": prompt,
            "prompt_source": "caption_json" if (
                key in captions or p.stem in captions
            ) else "stem",
        })
        if len(items) >= n:
            break
    return items


def encode_prefix_frames(pipeline, frames, n_latents: int, device):
    """Pixel frames → [1, n_latents, 16, 60, 104] bf16 latents."""
    import torch
    from PIL import Image
    from torchvision import transforms

    n_pix = prefix_pixel_count(n_latents)
    if len(frames) < n_pix:
        raise RuntimeError(
            f"need {n_pix} frames to encode {n_latents} latents, got {len(frames)}"
        )
    tfm = transforms.Compose([
        transforms.Resize((PIXEL_H, PIXEL_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    pix = []
    for fr in frames[:n_pix]:
        arr = np.asarray(fr)[..., :3]
        if arr.dtype != np.uint8:
            arr = np.clip(np.rint(arr.astype(np.float32) * 255.0), 0, 255).astype(
                np.uint8
            )
        pix.append(tfm(Image.fromarray(arr).convert("RGB")))
    video = torch.stack(pix, dim=1).unsqueeze(0).to(
        device=device, dtype=torch.bfloat16,
    )
    latent = pipeline.vae.encode_to_latent(video).to(
        device=device, dtype=torch.bfloat16,
    )
    if latent.shape[1] < n_latents:
        raise RuntimeError(
            f"VAE returned {latent.shape[1]} latents, wanted {n_latents}"
        )
    return latent[:, :n_latents]


def encode_prefix_video(pipeline, video_path: Path, n_latents: int, device):
    """First 1+4*(n-1) frames → [1, n_latents, 16, 60, 104] bf16 latents."""
    n_pix = prefix_pixel_count(n_latents)
    frames = []
    import imageio.v2 as imageio

    r = imageio.get_reader(str(video_path))
    try:
        for i, im in enumerate(r):
            if i >= n_pix:
                break
            frames.append(np.asarray(im)[..., :3])
    finally:
        try:
            r.close()
        except Exception:
            pass
    if len(frames) < n_pix:
        raise RuntimeError(
            f"{video_path.name}: need {n_pix} prefix frames, got {len(frames)}"
        )
    return encode_prefix_frames(pipeline, frames, n_latents, device)


def _recache_recent(pipeline, output, committed: int, n_latents: int, device):
    """Decode last n latents, VAE-encode, write back. KV is the caller's job."""
    start = max(0, int(committed) - int(n_latents))
    n = int(committed) - start
    if n <= 0:
        return None
    pix = _decode_pixels(pipeline, output[:, start:committed])
    new_lat = encode_prefix_frames(pipeline, pix, n, device)
    mae = float((output[:, start:committed].float() - new_lat.float()).abs().mean().item())
    output[:, start:committed] = new_lat
    print(f"    recache latents {start}:{committed} mae={mae:.5g}", flush=True)
    return {"start": start, "end": int(committed), "n": n, "mae": mae}


def _set_attr_chain(obj, name: str, value) -> bool:
    if obj is None or not hasattr(obj, name):
        return False
    setattr(obj, name, value)
    return True


def apply_shift(pipeline, shift: float) -> dict:
    """Best-effort FlowMatch shift. Returns which attributes were written."""
    found = {}
    val = float(shift)
    sched = getattr(pipeline, "scheduler", None)
    if _set_attr_chain(sched, "shift", val):
        found["scheduler.shift"] = val
    cfg = getattr(sched, "config", None)
    if _set_attr_chain(cfg, "shift", val):
        found["scheduler.config.shift"] = val
    args = getattr(pipeline, "args", None)
    for name in ("sample_shift", "shift", "flow_shift"):
        if _set_attr_chain(args, name, val):
            found[f"args.{name}"] = val
    return found


def apply_guidance(pipeline, scale: float) -> dict:
    found = {}
    val = float(scale)
    args = getattr(pipeline, "args", None)
    for name in ("guidance_scale", "sample_guide_scale", "cfg_scale"):
        if _set_attr_chain(args, name, val):
            found[f"args.{name}"] = val
    gen = getattr(pipeline, "generator", None)
    if _set_attr_chain(gen, "guidance_scale", val):
        found["generator.guidance_scale"] = val
    return found


def inspect_sampling_hooks(pipeline) -> dict:
    """Print live shift / cfg / sink attributes. Sink is wave 2 unless trivial."""
    info = {}
    sched = getattr(pipeline, "scheduler", None)
    if sched is not None:
        info["scheduler.shift"] = getattr(sched, "shift", None)
        cfg = getattr(sched, "config", None)
        if cfg is not None:
            info["scheduler.config.shift"] = getattr(cfg, "shift", None)
    args = getattr(pipeline, "args", None)
    if args is not None:
        for name in (
            "sample_shift", "shift", "flow_shift",
            "guidance_scale", "sample_guide_scale", "cfg_scale",
            "sink_size", "sink_size_t", "local_attn_size",
        ):
            if hasattr(args, name):
                info[f"args.{name}"] = getattr(args, name)
    gen = getattr(pipeline, "generator", None)
    if gen is not None:
        for name in ("guidance_scale", "local_attn_size", "sink_size"):
            if hasattr(gen, name):
                info[f"generator.{name}"] = getattr(gen, name)
    print("sampling_hooks:", json.dumps(info, default=str))
    sink_keys = [k for k in info if "sink" in k]
    if not sink_keys:
        print("sink: no hook on this checkpoint (wave 2 / dedicated preset)")
    return info


def _pixel_mae(a: np.ndarray, b: np.ndarray) -> float:
    n = min(a.shape[0], b.shape[0])
    if n < 1:
        return float("nan")
    return float(np.mean(np.abs(a[:n] - b[:n])))


def _should_backtrack(
    outgoing_drift: float | None,
    outgoing_motion: float | None,
    ref_motion: float | None,
    drift_threshold: float,
    motion_frac: float,
) -> tuple[bool, str]:
    # Smoke job 16069897: last-chunk composite was 6336 (prefix vs tail
    # scale clash). Ignore drift outside a sane band so backtrack does
    # not fire on every chunk.
    if (
        outgoing_drift is not None
        and 0.0 < outgoing_drift <= 100.0
        and outgoing_drift > drift_threshold
    ):
        return True, "outgoing_drift"
    if (
        outgoing_motion is not None and outgoing_motion == outgoing_motion
        and ref_motion is not None and ref_motion == ref_motion
        and ref_motion > 0
        and outgoing_motion < motion_frac * ref_motion
    ):
        return True, "motion_collapse"
    return False, "ok"


def _align_block(n: int, block: int = 3) -> int:
    return int(n) - (int(n) % int(block))


def _history_ranges(committed: int, prefix_latents: int, history: str):
    if committed <= 0:
        return []
    if history == "tail":
        start = max(0, committed - TAIL_HISTORY_LATENTS)
        start = _align_block(start)
        return [(start, committed)]
    if history == "sink":
        sink_end = int(prefix_latents)
        if committed <= sink_end + SINK_WINDOW_LATENTS:
            return [(0, committed)]
        win_start = max(sink_end, _align_block(committed - SINK_WINDOW_LATENTS))
        ranges = [(0, sink_end)]
        if win_start > sink_end:
            ranges.append((win_start, committed))
        elif committed > sink_end:
            ranges.append((sink_end, committed))
        return ranges
    return [(0, committed)]


def _replay_history(
    pipeline, output, committed, conditional_dict, history, prefix_latents,
):
    if committed <= 0:
        return
    ranges = _history_ranges(committed, prefix_latents, history)
    if not ranges:
        return
    if len(ranges) == 1 and ranges[0] == (0, committed):
        _cache_clean_latents_t2v(
            pipeline, output[:, :committed], conditional_dict,
        )
        return
    _cache_clean_latents_slices(
        pipeline, output[:, :committed], conditional_dict, ranges,
    )


def _snapshot_kv(pipeline):
    kv = []
    for blk in _active_kv(pipeline):
        end = max(
            int(blk["global_end_index"].item()),
            int(blk["local_end_index"].item()),
        )
        kv.append({
            "end": end,
            "local_end": int(blk["local_end_index"].item()),
            "global_end": int(blk["global_end_index"].item()),
            "k": blk["k"][:, : max(end, 1)].clone(),
            "v": blk["v"][:, : max(end, 1)].clone(),
        })
    cross = []
    for blk in pipeline.crossattn_cache:
        rec = {}
        for key, val in blk.items():
            rec[key] = val.clone() if hasattr(val, "clone") else val
        cross.append(rec)
    return {"kv": kv, "cross": cross}


def _restore_kv(pipeline, snap) -> None:
    for blk, saved in zip(_active_kv(pipeline), snap["kv"]):
        end = saved["end"]
        if end > 0:
            blk["k"][:, :end].copy_(saved["k"][:, :end])
            blk["v"][:, :end].copy_(saved["v"][:, :end])
        blk["global_end_index"].fill_(saved["global_end"])
        blk["local_end_index"].fill_(saved["local_end"])
    for blk, saved in zip(pipeline.crossattn_cache, snap["cross"]):
        for key, val in saved.items():
            if (
                key in blk
                and hasattr(val, "copy_")
                and hasattr(blk[key], "copy_")
            ):
                blk[key].copy_(val)
            else:
                blk[key] = val


def _build_cand_specs(
    method: str,
    ci: int,
    n_chunks: int,
    search_k: int,
    search_from_chunk: int,
    default_shift: float,
    default_cfg: float,
    incoming_motion: float | None,
    prefix_motion: float | None,
    pseudo_fire: bool = False,
    last_sick: bool = False,
):
    base = {"shift": default_shift, "cfg": default_cfg, "history": "full"}
    if method in (
        "notta", "longlive_notta", "longlive_prefix_sink",
        "rf_chunk", "sf_recache", "sf_sink", "sf_rewind",
    ) or ci < search_from_chunk:
        reason = (
            "notta" if method in (
                "notta", "longlive_notta", "longlive_prefix_sink",
                "rf_chunk", "sf_recache", "sf_sink", "sf_rewind",
            ) and ci >= search_from_chunk
            else "forced_prefix"
        )
        return [{**base, "cand": 0, "noise_id": 0}], False, reason
    if method in ("sink", "longlive_sink"):
        return (
            [{**base, "cand": 0, "noise_id": 0, "history": "sink"}],
            False,
            "sink",
        )
    if method in (
        "seed_bon", "cached_bon", "hinge_bon", "motion_bon", "appear_bon",
    ):
        return (
            [
                {**base, "cand": c, "noise_id": c}
                for c in range(search_k)
            ],
            True,
            method,
        )
    if method == "shift_search":
        return (
            [
                {**base, "cand": i, "noise_id": 0, "shift": float(s)}
                for i, s in enumerate(SHIFT_GRID)
            ],
            True,
            "shift_search",
        )
    if method == "late_bon":
        fire = False
        reason = "late_skip"
        if (
            incoming_motion is not None and incoming_motion == incoming_motion
            and prefix_motion is not None and prefix_motion == prefix_motion
            and prefix_motion > 0
            and incoming_motion < LATE_MOTION_FRAC * prefix_motion
        ):
            fire, reason = True, "late_motion"
        if ci >= n_chunks - 2:
            fire, reason = True, "late_horizon"
        if fire:
            return (
                [{**base, "cand": c, "noise_id": c} for c in range(search_k)],
                True,
                reason,
            )
        return [{**base, "cand": 0, "noise_id": 0}], False, reason
    if method == "quiet_bon":
        if (
            prefix_motion is not None and prefix_motion == prefix_motion
            and prefix_motion >= QUIET_SEARCH_MAX
        ):
            return (
                [{**base, "cand": 0, "noise_id": 0}],
                False,
                "quiet_hot",
            )
        return (
            [{**base, "cand": c, "noise_id": c} for c in range(search_k)],
            True,
            "quiet_search",
        )
    if method in ("pseudo_gate", "pseudo_appear"):
        if pseudo_fire:
            return (
                [{**base, "cand": c, "noise_id": c} for c in range(search_k)],
                True,
                "pseudo_fire",
            )
        return (
            [{**base, "cand": 0, "noise_id": 0}],
            False,
            "pseudo_skip",
        )
    if method == "noise_probe":
        return [{**base, "cand": 0, "noise_id": 0}], False, "noise_probe"
    if method == "noise_bon":
        return (
            [{**base, "cand": 0, "noise_id": 0}],
            False,
            "noise_cand0",
        )
    if method in ("live_bon", "longlive_live_bon", "live_appear"):
        live = (
            prefix_motion is not None and prefix_motion == prefix_motion
            and prefix_motion >= _LIVE_SEARCH_MIN
        )
        if live:
            return (
                [{**base, "cand": c, "noise_id": c} for c in range(search_k)],
                True,
                "live_search",
            )
        return (
            [{**base, "cand": 0, "noise_id": 0}],
            False,
            "live_skip_still",
        )
    if method == "live_hist":
        live = (
            prefix_motion is not None and prefix_motion == prefix_motion
            and prefix_motion >= _LIVE_SEARCH_MIN
        )
        if not live:
            return (
                [{**base, "cand": 0, "noise_id": 0}],
                False,
                "live_hist_skip_still",
            )
        specs = [
            {**base, "cand": 0, "noise_id": 0, "history": "full"},
            {**base, "cand": 1, "noise_id": 0, "history": "tail"},
            {**base, "cand": 2, "noise_id": 1, "history": "full"},
            {**base, "cand": 3, "noise_id": 2, "history": "full"},
        ]
        return specs[: max(2, search_k)], True, "live_hist"
    if method == "tail_hist":
        return (
            [{**base, "cand": 0, "noise_id": 0, "history": "tail"}],
            False,
            "tail_hist",
        )
    if method == "hist_drop":
        specs = [
            {**base, "cand": 0, "noise_id": 0, "history": "full"},
            {**base, "cand": 1, "noise_id": 0, "history": "tail"},
            {**base, "cand": 2, "noise_id": 1, "history": "full"},
            {**base, "cand": 3, "noise_id": 2, "history": "full"},
        ]
        return specs[: max(2, search_k)], True, "hist_drop"
    if method == "sf_sick_search":
        if last_sick:
            return (
                [{**base, "cand": c, "noise_id": c} for c in range(search_k)],
                True,
                "sick_search",
            )
        return [{**base, "cand": 0, "noise_id": 0}], False, "sick_skip"
    if method == "sf_pseudo":
        if pseudo_fire:
            return (
                [{**base, "cand": c, "noise_id": c} for c in range(search_k)],
                True,
                "pseudo_fire",
            )
        return [{**base, "cand": 0, "noise_id": 0}], False, "pseudo_skip"
    if method == "sf_always_search":
        return (
            [{**base, "cand": c, "noise_id": c} for c in range(search_k)],
            True,
            "always_search",
        )
    if method in ("backtrack", "good_backtrack"):
        return (
            [{**base, "cand": 0, "noise_id": 0}],
            False,
            f"{method}_first",
        )
    return [{**base, "cand": 0, "noise_id": 0}], False, "notta"


def _chunk_pixel_motion(pixels, committed_before: int, chunk_latents: int):
    start = t2v_pixel_frames(committed_before)
    end = t2v_pixel_frames(committed_before + chunk_latents)
    win = pixels[start:end] if pixels is not None else None
    if win is None or win.shape[0] < 2:
        return None
    return float(np.mean(np.abs(win[1:] - win[:-1])))


def _cand_temporal_motion(rec: dict):
    free = rec.get("free") or {}
    m = free.get("temporal_motion")
    if m is None:
        m = rec.get("motion_score")
    return m


def _run_one_chunk(
    pipeline,
    output,
    committed: int,
    chunk_latents: int,
    conditional_dict,
    seed: int,
    cand: int,
    ci: int,
    device,
    shift: float,
    cfg: float,
    history: str = "full",
    prefix_latents: int = PREFIX_LATENTS_DEFAULT,
    kv_snap=None,
):
    import torch

    apply_shift(pipeline, shift)
    apply_guidance(pipeline, cfg)
    rng = _chunk_rng(device, seed, cand, ci)
    noise = torch.randn(
        [1, chunk_latents, LATENT_C, LATENT_H, LATENT_W],
        device=device, dtype=torch.bfloat16, generator=rng,
    )
    if kv_snap is None:
        _reset_caches(pipeline, 1, output.dtype, device)
        if committed > 0:
            _replay_history(
                pipeline, output, committed, conditional_dict,
                history, prefix_latents,
            )
    else:
        _restore_kv(pipeline, kv_snap)
    stats_out = []
    _denoise_chunk(
        pipeline, noise, committed, conditional_dict, output, rng,
        stats_out=stats_out,
    )
    end = committed + chunk_latents
    pixels = _decode_pixels(pipeline, output[:, :end])
    latents = output[:, committed:end].detach().clone()
    noise_stats = stats_out[0] if stats_out else None
    return latents, pixels, noise_stats


def generate_chunked_v2v(
    pipeline,
    video_path: Path,
    prompt: str,
    prefix_latents: int,
    n_gen: int,
    chunk_latents: int,
    seed: int,
    device,
    method: str,
    search_k: int,
    search_from_chunk: int,
    seam_weight: float,
    backtrack_threshold: float,
    backtrack_motion_frac: float,
    default_shift: float,
    default_cfg: float,
):
    import torch

    if method == "always_bon":
        method = "seed_bon"
    n_chunks = n_gen // chunk_latents
    prefix = encode_prefix_video(pipeline, Path(video_path), prefix_latents, device)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])
    total = prefix_latents + n_gen
    output = torch.zeros(
        [1, total, LATENT_C, LATENT_H, LATENT_W],
        device=device, dtype=torch.bfloat16,
    )
    output[:, :prefix_latents] = prefix[:, :prefix_latents]
    apply_shift(pipeline, default_shift)
    apply_guidance(pipeline, default_cfg)

    committed = prefix_latents
    prefix_pix_n = t2v_pixel_frames(prefix_latents)
    ref = None
    committed_pixels = None
    incoming_prev = None
    chunk_logs = []
    prefix_only = _decode_pixels(pipeline, output[:, :prefix_latents])
    prefix_motion = (
        float(np.mean(np.abs(prefix_only[1:] - prefix_only[:-1])))
        if prefix_only.shape[0] >= 2 else None
    )
    print(f"  prefix_motion={prefix_motion}", flush=True)
    good_committed = prefix_latents
    pseudo_fire = False
    pseudo_rows = None
    last_sick = False
    rewind_logs = []
    prev_chunk_mot = prefix_motion
    if method in ("pseudo_gate", "pseudo_appear", "sf_pseudo"):
        pseudo_fire, pseudo_rows = _eval_pseudo_future(
            pipeline, output, prefix_latents, conditional_dict,
            seed, device, search_k, default_shift, default_cfg,
            prefix_only,
        )

    for ci in range(n_chunks):
        incoming_signals = None
        incoming_devs = None
        incoming_drift = None
        incoming_delta = None
        incoming_motion = None

        if ref is not None and committed_pixels is not None:
            incoming_signals = _incoming_window(committed_pixels)
            if incoming_signals is not None:
                incoming_devs = score_breakdown(
                    incoming_signals, ref, seam_weight=0.0,
                )
                incoming_drift = incoming_devs["score"]
                incoming_motion = incoming_signals.get("temporal_motion")
                if incoming_prev is not None:
                    incoming_delta = incoming_drift - incoming_prev

        cand_specs, searched, reason = _build_cand_specs(
            method, ci, n_chunks, search_k, search_from_chunk,
            default_shift, default_cfg, incoming_motion, prefix_motion,
            pseudo_fire=pseudo_fire,
            last_sick=last_sick,
        )

        def _score_cand(latents, pixels, cand, shift, cfg, history="full"):
            n_committed_pix = t2v_pixel_frames(committed)
            gen_only = pixels[n_committed_pix:]
            last_cond = pixels[n_committed_pix - 1]
            nonlocal_ref = ref
            free = gen_free_signals(gen_only, last_cond)
            br = score_breakdown(free, nonlocal_ref, seam_weight=seam_weight)
            mscore = motion_pick_score(free, nonlocal_ref)
            hscore = prefix_match_score(free, nonlocal_ref, seam_weight=seam_weight)
            ascore = appear_score(free, nonlocal_ref, seam_weight=seam_weight)
            return {
                "cand": cand,
                "seed": _cand_seed(seed, cand),
                "shift": float(shift),
                "cfg": float(cfg),
                "history": history,
                "score": br["score"],
                "hinge_score": hscore,
                "appear_score": ascore,
                "motion_score": mscore,
                "latents": latents,
                "pixels": pixels,
                "free": {k: free[k] for k in free},
                "breakdown": br,
            }

        kv_snap = None
        if method == "cached_bon" and committed > 0:
            _reset_caches(pipeline, 1, output.dtype, device)
            _replay_history(
                pipeline, output, committed, conditional_dict,
                "full", prefix_latents,
            )
            kv_snap = _snapshot_kv(pipeline)

        cands = []
        built_ref = ref
        for spec in cand_specs:
            noise_id = int(spec.get("noise_id", spec["cand"]))
            hist = spec.get("history", "full")
            latents, pixels, noise_stats = _run_one_chunk(
                pipeline, output, committed, chunk_latents,
                conditional_dict, seed, noise_id, ci, device,
                spec["shift"], spec["cfg"],
                history=hist, prefix_latents=prefix_latents,
                kv_snap=kv_snap,
            )
            if built_ref is None:
                prefix_win = pixels[: min(prefix_pix_n, pixels.shape[0])]
                if prefix_win.shape[0] < 2:
                    raise RuntimeError("prefix too short to build a V2V reference")
                ref_win = (
                    prefix_win[:REF_WIN]
                    if prefix_win.shape[0] >= REF_WIN
                    else prefix_win
                )
                built_ref = reference_signals(ref_win)
                if prefix_motion is None:
                    prefix_motion = float(np.mean(np.abs(
                        prefix_win[1:] - prefix_win[:-1]
                    )))
            ref = built_ref
            rec = _score_cand(
                latents, pixels, spec["cand"], spec["shift"], spec["cfg"],
                history=hist,
            )
            rec["noise_stats"] = noise_stats
            cands.append(rec)
            u = (noise_stats or {}).get("eps_mean_abs")
            print(
                f"    chunk {ci} cand{spec['cand']} hist={hist} "
                f"shift={spec['shift']} cfg={spec['cfg']} "
                f"score={rec['score']:.4f} hinge={rec['hinge_score']:.4f} "
                f"appear={rec['appear_score']:.4f} "
                f"motion={rec['free']['temporal_motion']:.4g} "
                f"U={u}",
                flush=True,
            )

        if method == "noise_bon" and cands:
            u0 = (cands[0].get("noise_stats") or {}).get("eps_mean_abs")
            if u0 is not None and u0 >= float(_NOISE_TAU):
                searched, reason = True, "noise_fire"
                for extra in range(1, search_k):
                    latents, pixels, noise_stats = _run_one_chunk(
                        pipeline, output, committed, chunk_latents,
                        conditional_dict, seed, extra, ci, device,
                        default_shift, default_cfg,
                        history="full", prefix_latents=prefix_latents,
                    )
                    rec = _score_cand(
                        latents, pixels, extra, default_shift, default_cfg,
                    )
                    rec["noise_stats"] = noise_stats
                    cands.append(rec)
                    print(
                        f"    chunk {ci} cand{extra} noise_fire "
                        f"appear={rec['appear_score']:.4f} U={u0:.4g}",
                        flush=True,
                    )
            else:
                reason = "noise_skip"

        if method in ("sf_sick_search", "sf_pseudo", "sf_always_search") and len(cands) > 1:
            m0 = _cand_temporal_motion(cands[0])
            feasible = []
            for c in cands:
                m = _cand_temporal_motion(c)
                if (
                    m is not None and m == m
                    and m0 is not None and m0 == m0
                    and m >= ROLL_TRUST_FRAC * m0
                ):
                    feasible.append(c)
            if not feasible:
                chosen, reason = 0, "look_trust_reject"
            else:
                best_c = max(
                    feasible,
                    key=lambda c: _cand_temporal_motion(c) or -1e9,
                )
                chosen = cands.index(best_c)
                reason = "sick_motion"
        elif method == "motion_bon" or method == "shift_search":
            chosen = max(
                range(len(cands)),
                key=lambda i: (
                    cands[i]["motion_score"]
                    if cands[i]["motion_score"] != float("-inf")
                    else -1e9
                ),
            )
        elif method in ("hinge_bon", "hist_drop", "live_hist"):
            chosen = min(range(len(cands)), key=lambda i: cands[i]["hinge_score"])
        elif method in (
            "appear_bon", "live_appear", "pseudo_appear", "noise_bon",
        ):
            chosen = min(range(len(cands)), key=lambda i: cands[i]["appear_score"])
        else:
            chosen = min(range(len(cands)), key=lambda i: cands[i]["score"])

        best = cands[chosen]
        output[:, committed:committed + chunk_latents] = best["latents"]
        committed += chunk_latents
        committed_pixels = best["pixels"]
        apply_shift(pipeline, default_shift)
        apply_guidance(pipeline, default_cfg)

        outgoing_signals = _incoming_window(committed_pixels) if ref is not None else None
        outgoing_devs = (
            score_breakdown(outgoing_signals, ref, seam_weight=0.0)
            if outgoing_signals is not None and ref is not None else None
        )
        outgoing_drift = outgoing_devs["score"] if outgoing_devs is not None else None
        outgoing_motion = (
            outgoing_signals.get("temporal_motion") if outgoing_signals else None
        )
        chunk_start = committed - chunk_latents
        chunk_mot = _chunk_pixel_motion(
            committed_pixels, chunk_start, chunk_latents,
        )
        if method == "sf_rewind" and ci >= search_from_chunk:
            ref_m = prev_chunk_mot
            sick = bool(
                chunk_mot is not None and chunk_mot == chunk_mot
                and ref_m is not None and ref_m == ref_m
                and ref_m > 0
                and chunk_mot < RF_SICK_DROP * ref_m
            )
            if sick:
                saved_lat = output[:, chunk_start:committed].clone()
                saved_pix = committed_pixels
                saved_mot = chunk_mot
                committed = chunk_start
                latents, pixels, _ns = _run_one_chunk(
                    pipeline, output, committed, chunk_latents,
                    conditional_dict, seed, 1, ci, device,
                    default_shift, default_cfg,
                    history="full", prefix_latents=prefix_latents,
                )
                mot2 = _chunk_pixel_motion(pixels, committed, chunk_latents)
                accepted = (
                    mot2 is not None and mot2 == mot2 and mot2 >= saved_mot
                )
                rewind_logs.append({
                    "chunk": ci,
                    "mot0": saved_mot,
                    "mot1": mot2,
                    "ref": ref_m,
                    "accepted": bool(accepted),
                })
                print(
                    f"    sf_rewind chunk={ci} mot {saved_mot:.5g}->"
                    f"{mot2} accept={accepted}",
                    flush=True,
                )
                if accepted:
                    output[:, committed:committed + chunk_latents] = latents
                    committed_pixels = pixels
                    chunk_mot = mot2
                    outgoing_motion = mot2
                    searched = True
                    reason = "sf_rewind_accept"
                    # Score while committed is still chunk_start. Incrementing
                    # first made gen_only empty (16266878: 24 IndexError).
                    rec2 = _score_cand(
                        latents, pixels, 1, default_shift, default_cfg,
                    )
                    cands.append(rec2)
                    chosen = len(cands) - 1
                    best = rec2
                    committed += chunk_latents
                else:
                    output[:, committed:committed + chunk_latents] = saved_lat
                    committed += chunk_latents
                    committed_pixels = saved_pix
                    chunk_mot = saved_mot
                    reason = "sf_rewind_reject"
        last_sick = bool(
            chunk_mot is not None and chunk_mot == chunk_mot
            and prev_chunk_mot is not None and prev_chunk_mot == prev_chunk_mot
            and prev_chunk_mot > 0
            and chunk_mot < RF_SICK_DROP * prev_chunk_mot
        )
        if chunk_mot is not None:
            prev_chunk_mot = chunk_mot
        backtracked = False
        backtrack_reason = None
        if method == "backtrack" and ci >= search_from_chunk:
            fire, backtrack_reason = _should_backtrack(
                outgoing_drift, outgoing_motion, prefix_motion,
                backtrack_threshold, backtrack_motion_frac,
            )
            if fire:
                print(
                    f"    backtrack chunk {ci}: {backtrack_reason} "
                    f"outgoing_drift={outgoing_drift} motion={outgoing_motion}",
                    flush=True,
                )
                committed -= chunk_latents
                latents, pixels, _ns = _run_one_chunk(
                    pipeline, output, committed, chunk_latents,
                    conditional_dict, seed, 1, ci, device,
                    default_shift, default_cfg,
                    history="full", prefix_latents=prefix_latents,
                )
                rec = _score_cand(latents, pixels, 1, default_shift, default_cfg)
                cands.append(rec)
                if rec["score"] <= best["score"] or (
                    rec["free"]["temporal_motion"]
                    > (best["free"]["temporal_motion"] or 0)
                ):
                    chosen = len(cands) - 1
                    best = rec
                output[:, committed:committed + chunk_latents] = best["latents"]
                committed += chunk_latents
                committed_pixels = best["pixels"]
                outgoing_signals = _incoming_window(committed_pixels)
                outgoing_devs = (
                    score_breakdown(outgoing_signals, ref, seam_weight=0.0)
                    if outgoing_signals is not None else None
                )
                outgoing_drift = (
                    outgoing_devs["score"] if outgoing_devs is not None else None
                )
                outgoing_motion = (
                    outgoing_signals.get("temporal_motion")
                    if outgoing_signals else None
                )
                backtracked = True
                searched = True
                reason = f"backtrack:{backtrack_reason}"
        elif method == "good_backtrack" and ci >= search_from_chunk:
            fire, backtrack_reason = _should_backtrack(
                outgoing_drift, outgoing_motion, prefix_motion,
                backtrack_threshold, backtrack_motion_frac,
            )
            prev_was_good = good_committed == committed - chunk_latents
            if fire and prev_was_good:
                print(
                    f"    good_backtrack chunk {ci}: {backtrack_reason} "
                    f"rewind_to={good_committed} motion={outgoing_motion}",
                    flush=True,
                )
                committed -= chunk_latents
                latents, pixels, _ns = _run_one_chunk(
                    pipeline, output, committed, chunk_latents,
                    conditional_dict, seed, 1, ci, device,
                    default_shift, default_cfg,
                    history="full", prefix_latents=prefix_latents,
                )
                rec = _score_cand(latents, pixels, 1, default_shift, default_cfg)
                cands.append(rec)
                chosen = len(cands) - 1
                best = rec
                output[:, committed:committed + chunk_latents] = best["latents"]
                committed += chunk_latents
                committed_pixels = best["pixels"]
                outgoing_signals = _incoming_window(committed_pixels)
                outgoing_devs = (
                    score_breakdown(outgoing_signals, ref, seam_weight=0.0)
                    if outgoing_signals is not None else None
                )
                outgoing_drift = (
                    outgoing_devs["score"] if outgoing_devs is not None else None
                )
                outgoing_motion = (
                    outgoing_signals.get("temporal_motion")
                    if outgoing_signals else None
                )
                backtracked = True
                searched = True
                reason = f"good_backtrack:{backtrack_reason}"
            elif fire and not prev_was_good:
                reason = f"good_backtrack:skip_poison:{backtrack_reason}"
        if (
            outgoing_motion is not None and outgoing_motion == outgoing_motion
            and prefix_motion is not None and prefix_motion == prefix_motion
            and prefix_motion > 0
            and outgoing_motion >= GOOD_SAVE_FRAC * prefix_motion
        ):
            good_committed = committed

        recache_info = None
        if method == "sf_recache" and ci < n_chunks - 1:
            recache_info = _recache_recent(
                pipeline, output, committed, RECACHE_LATENTS, device,
            )

        cand0_score = cands[0]["score"]
        rec = {
            "chunk": ci,
            "chosen_cand": int(chosen),
            "search_k": len(cands),
            "method": method,
            "incoming_drift": _json_float(incoming_drift),
            "incoming_prev": _json_float(incoming_prev),
            "incoming_delta": _json_float(incoming_delta),
            "incoming_motion": _json_float(incoming_motion),
            "incoming_signals": _json_signals(incoming_signals),
            "incoming_devs": _json_signals(incoming_devs),
            "outgoing_drift": _json_float(outgoing_drift),
            "outgoing_devs": _json_signals(outgoing_devs),
            "outgoing_motion": _json_float(outgoing_motion),
            "prefix_motion": _json_float(prefix_motion),
            "good_committed": int(good_committed),
            "searched": bool(searched),
            "gate_reason": reason,
            "backtracked": bool(backtracked),
            "backtrack_reason": backtrack_reason,
            "cand0_score": _json_float(cand0_score),
            "chosen_score": _json_float(best["score"]),
            "chosen_hinge_score": _json_float(best["hinge_score"]),
            "chosen_appear_score": _json_float(best.get("appear_score")),
            "chosen_motion_score": _json_float(best["motion_score"]),
            "chosen_noise_stats": best.get("noise_stats"),
            "pseudo_fire": bool(pseudo_fire),
            "pseudo_rows": pseudo_rows if ci == 0 else None,
            "last_sick": bool(last_sick),
            "chunk_motion": _json_float(chunk_mot),
            "rewind": (
                rewind_logs[-1]
                if rewind_logs and rewind_logs[-1].get("chunk") == ci
                else None
            ),
            "recache": recache_info,
            "chosen_minus_cand0": _json_float(best["score"] - cand0_score),
            "chosen_breakdown": _json_signals(best["breakdown"]),
            "candidates": [
                {
                    "cand": c["cand"],
                    "seed": c["seed"],
                    "shift": c["shift"],
                    "cfg": c["cfg"],
                    "history": c.get("history", "full"),
                    "score": _json_float(c["score"]),
                    "hinge_score": _json_float(c["hinge_score"]),
                    "appear_score": _json_float(c.get("appear_score")),
                    "motion_score": _json_float(c["motion_score"]),
                    "noise_stats": c.get("noise_stats"),
                    "chosen": c["cand"] == best["cand"] and math.isclose(
                        c["shift"], best["shift"],
                    ) and c.get("history", "full") == best.get("history", "full"),
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
        print(
            f"  chunk {ci}: pick={chosen}/{len(cands)} "
            f"score={best['score']:.4f} motion_pick={best['motion_score']:.4g} "
            f"reason={reason}",
            flush=True,
        )

    apply_shift(pipeline, default_shift)
    apply_guidance(pipeline, default_cfg)
    pixels = _decode_pixels(pipeline, output)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pixels, tuple(output.shape), ref, chunk_logs, prefix_pix_n


def _eval_pseudo_future(
    pipeline,
    output,
    prefix_latents: int,
    conditional_dict,
    seed: int,
    device,
    search_k: int,
    default_shift: float,
    default_cfg: float,
    prefix_pixels: np.ndarray,
):
    """Generate held-out last-3 prefix latents from the first 6. Real B is GT."""
    import torch

    b_lat = PSEUDO_B_LATENTS
    a_lat = int(prefix_latents) - b_lat
    if a_lat < 3 or a_lat % 3 != 0:
        raise RuntimeError(
            f"pseudo-future needs prefix={prefix_latents} with A multiple of 3"
        )
    a_pix = t2v_pixel_frames(a_lat)
    b_pix = t2v_pixel_frames(prefix_latents)
    real_b = prefix_pixels[a_pix:b_pix]
    saved = output[:, a_lat:a_lat + b_lat].clone()
    rows = []
    for c in range(max(1, int(search_k))):
        latents, pixels, _stats = _run_one_chunk(
            pipeline, output, a_lat, b_lat, conditional_dict,
            seed, c, -1, device, default_shift, default_cfg,
            history="full", prefix_latents=prefix_latents,
        )
        gen_b = pixels[a_pix:a_pix + real_b.shape[0]]
        n = min(gen_b.shape[0], real_b.shape[0])
        mae = (
            float(np.mean(np.abs(gen_b[:n] - real_b[:n])))
            if n >= 1 else float("nan")
        )
        rows.append({"cand": c, "mae": mae, "n_pix": int(n)})
        print(
            f"    pseudo B cand{c} mae={mae:.5g} vs real last-3 latents",
            flush=True,
        )
        output[:, a_lat:a_lat + b_lat] = saved
    notta_mae = rows[0]["mae"]
    best = min(rows, key=lambda r: r["mae"] if r["mae"] == r["mae"] else 1e9)
    fire = (
        best["mae"] == best["mae"]
        and notta_mae == notta_mae
        and best["mae"] < notta_mae - float(_PSEUDO_GAMMA)
        and int(best["cand"]) != 0
    )
    print(
        f"  pseudo-future notta_mae={notta_mae:.5g} best={best['mae']:.5g} "
        f"cand={best['cand']} fire={fire} gamma={_PSEUDO_GAMMA}",
        flush=True,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fire, rows


def _rf_kv(pipeline):
    kv = getattr(pipeline, "kv_cache_clean", None)
    if kv is not None:
        return kv
    return pipeline.kv_cache1


def _rf_block(pipeline) -> int:
    return int(getattr(pipeline, "num_frame_per_block", 3))


def _rf_replay_clean(pipeline, output, n_latents, conditional_dict, device):
    """Reset KV and replay clean latents 0:n_latents (prefix warmup / recache)."""
    import torch

    pipeline._initialize_kv_cache(1, output.dtype, device)
    pipeline._initialize_crossattn_cache(1, output.dtype, device)
    kv = _rf_kv(pipeline)
    block = _rf_block(pipeline)
    ts0 = torch.ones([1, block], device=device, dtype=torch.int64) * 0
    t = 0
    while t < n_latents:
        kwargs = dict(
            noisy_image_or_video=output[:, t:t + block],
            conditional_dict=conditional_dict,
            timestep=ts0,
            kv_cache=kv,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=t * pipeline.frame_seq_length,
        )
        try:
            pipeline.generator(**kwargs, updating_cache=True)
        except TypeError:
            pipeline.generator(**kwargs)
        t += block
    return kv


def _snap_kv(kv):
    if not kv:
        return None
    out = []
    for item in kv:
        if isinstance(item, dict):
            out.append({
                k: (v.clone() if hasattr(v, "clone") else v)
                for k, v in item.items()
            })
        else:
            out.append(item)
    return out


def _restore_kv(kv, snap):
    if not kv or not snap:
        return
    for dst, src in zip(kv, snap):
        if not isinstance(dst, dict) or not isinstance(src, dict):
            continue
        for k, v in src.items():
            if hasattr(v, "clone") and k in dst and hasattr(dst[k], "copy_"):
                dst[k].copy_(v)


def _rho_from_prefix(prefix_motion):
    if prefix_motion is None or prefix_motion != prefix_motion:
        return 1.0
    if prefix_motion < ROLL_STILL_MIN:
        return 2.0
    if prefix_motion >= ROLL_HOT_MIN:
        return 0.5
    return 1.0


def _scale_rf_noise(noise, block: int, rho: float):
    """Per-block init-noise × (h/H)^ρ, mean-normalized. ρ=1 is a no-op."""
    import torch

    if abs(float(rho) - 1.0) < 1e-6:
        return 1.0, []
    n_gen = int(noise.shape[1])
    num_blocks = n_gen // block
    scales = []
    for bi in range(num_blocks):
        u = (bi + 1) / max(num_blocks, 1)
        scales.append(float(u) ** float(rho))
    mean = sum(scales) / max(len(scales), 1)
    scales = [s / (mean + 1e-8) for s in scales]
    for bi, sc in enumerate(scales):
        noise[:, bi * block:(bi + 1) * block].mul_(sc)
    return float(rho), scales


def _latent_motion_seam(pred, prev_latent):
    import torch

    x = pred.float()
    if x.shape[1] >= 2:
        motion = float((x[:, 1:] - x[:, :-1]).abs().mean().item())
    else:
        motion = float("nan")
    if prev_latent is None:
        seam = 0.0
    else:
        seam = float((x[:, :1] - prev_latent.float()).abs().mean().item())
    return motion, seam


def _span_pixel_motion(pipeline, output, start: int, n: int) -> float:
    if n < 2:
        return float("nan")
    pix = _decode_pixels(pipeline, output[:, start:start + n])
    if pix.shape[0] < 2:
        return float("nan")
    return float(np.mean(np.abs(pix[1:] - pix[:-1])))


def _rf_roll_span(
    pipeline,
    output,
    start: int,
    n_lat: int,
    noise,
    conditional_dict,
    device,
    seed: int,
):
    """k=1 rolling fill of output[:, start:start+n_lat]. Replays KV to start."""
    import torch

    block = _rf_block(pipeline)
    if n_lat % block != 0:
        raise RuntimeError(f"roll span {n_lat} not divisible by block={block}")
    kv = _rf_replay_clean(pipeline, output, start, conditional_dict, device)
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    num_blocks = n_lat // block
    raw_steps = pipeline.denoising_step_list
    steps = [float(s) for s in list(raw_steps)]
    step_tensor = (
        raw_steps.to(device=device)
        if hasattr(raw_steps, "to")
        else torch.tensor(steps, device=device, dtype=torch.float32)
    )
    n_step = len(steps)
    window_num = num_blocks + n_step - 1
    noisy_cache = torch.zeros_like(output)
    shared_timestep = torch.ones(
        [1, n_step * block], device=device, dtype=torch.float32,
    )
    for index, current_timestep in enumerate(reversed(steps)):
        shared_timestep[:, index * block:(index + 1) * block] *= current_timestep
    offset = start
    for window_index in range(window_num):
        start_block = max(0, window_index - n_step + 1)
        end_block = min(num_blocks - 1, window_index)
        cur0 = offset + start_block * block
        cur1 = offset + (end_block + 1) * block
        n_frames = cur1 - cur0
        tail0 = cur0 - offset
        tail1 = cur1 - offset
        inject = n_frames == n_step * block or start_block == 0
        if n_frames == n_step * block:
            current_timestep = shared_timestep
        elif start_block == 0:
            current_timestep = shared_timestep[:, -n_frames:]
        else:
            current_timestep = shared_timestep[:, :n_frames]
        if inject:
            noisy_input = torch.cat([
                noisy_cache[:, cur0:cur1 - block],
                noise[:, tail1 - block:tail1],
            ], dim=1)
        else:
            noisy_input = noisy_cache[:, cur0:cur1]
        _, denoised_pred = pipeline.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=conditional_dict,
            timestep=current_timestep,
            kv_cache=kv,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=cur0 * pipeline.frame_seq_length,
        )
        output[:, cur0:cur1] = denoised_pred
        with torch.no_grad():
            for block_idx in range(start_block, end_block + 1):
                rel = block_idx - start_block
                block_time_step = current_timestep[
                    :, rel * block:(rel + 1) * block
                ].mean().item()
                matches = torch.abs(step_tensor.to(device) - block_time_step) < 1e-4
                idxs = torch.nonzero(matches, as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue
                block_timestep_index = int(idxs[0].item())
                if block_timestep_index == n_step - 1:
                    continue
                next_timestep = step_tensor[block_timestep_index + 1]
                noisy_cache[:, offset + block_idx * block:offset + (block_idx + 1) * block] = (
                    pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn(
                            denoised_pred.flatten(0, 1).shape,
                            device=device, dtype=denoised_pred.dtype,
                            generator=rng,
                        ),
                        next_timestep.to(device) * torch.ones(
                            [denoised_pred.shape[0] * denoised_pred.shape[1]],
                            device=device, dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])[
                        :, rel * block:(rel + 1) * block
                    ]
                )
            context_timestep = torch.ones_like(current_timestep) * float(
                getattr(getattr(pipeline, "args", None), "context_noise", 0) or 0
            )
            first = denoised_pred[:, :block]
            ctx = context_timestep[:, :block]
            kwargs = dict(
                noisy_image_or_video=first,
                conditional_dict=conditional_dict,
                timestep=ctx,
                kv_cache=kv,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=cur0 * pipeline.frame_seq_length,
            )
            try:
                pipeline.generator(**kwargs, updating_cache=True)
            except TypeError:
                pipeline.generator(**kwargs)
    return _rf_replay_clean(pipeline, output, start + n_lat, conditional_dict, device)


def generate_rolling_v2v(
    pipeline,
    video_path: Path,
    prompt: str,
    prefix_latents: int,
    n_gen: int,
    seed: int,
    device,
    rho: float = 1.0,
    rho_mode: str = "fixed",
    search_k: int = 1,
    method_name: str = "rolling_notta",
    recache_every_latents: int = 0,
):
    """Rolling Forcing tail after a real prefix. Their public
    inference_rolling_forcing() overwrites prefix frames (current_start_frame
    is also unbound on the multi-frame path), so we cache the prefix then
    roll only the tail with a start offset.
    """
    import torch

    prefix = encode_prefix_video(pipeline, Path(video_path), prefix_latents, device)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])
    block = _rf_block(pipeline)
    if prefix_latents % block != 0 or n_gen % block != 0:
        raise RuntimeError(
            f"rolling V2V needs prefix={prefix_latents} and n_gen={n_gen} "
            f"divisible by block={block}"
        )
    total = prefix_latents + n_gen
    output = torch.zeros(
        [1, total, LATENT_C, LATENT_H, LATENT_W],
        device=device, dtype=torch.bfloat16,
    )
    output[:, :prefix_latents] = prefix[:, :prefix_latents]
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed))
    noise = torch.randn(
        [1, n_gen, LATENT_C, LATENT_H, LATENT_W],
        device=device, dtype=torch.bfloat16, generator=rng,
    )
    kv = _rf_replay_clean(
        pipeline, output, prefix_latents, conditional_dict, device,
    )
    recache_logs = []

    prefix_pix_early = _decode_pixels(pipeline, output[:, :prefix_latents])
    prefix_motion_early = (
        float(np.mean(np.abs(prefix_pix_early[1:] - prefix_pix_early[:-1])))
        if prefix_pix_early.shape[0] >= 2 else None
    )
    used_rho = float(rho)
    if rho_mode == "adapt":
        used_rho = _rho_from_prefix(prefix_motion_early)
    _scale_rf_noise(noise, block, used_rho)
    look_k = max(1, int(search_k))
    look_picks = []
    rewind_logs = []
    last_sick = False
    last_chunk_motion = None
    pseudo_fire = False
    pseudo_rows = None
    if method_name == "rf_pseudo":
        import torch as _torch
        a = int(prefix_latents) - int(PSEUDO_B_LATENTS)
        real_b = output[:, a:prefix_latents].clone()
        maes = []
        for ck in range(2):
            tmp = output.clone()
            bnoise = _torch.randn(
                [1, PSEUDO_B_LATENTS, LATENT_C, LATENT_H, LATENT_W],
                device=device, dtype=output.dtype,
                generator=_torch.Generator(device=device).manual_seed(
                    int(seed) + 333 * ck
                ),
            )
            _rf_roll_span(
                pipeline, tmp, a, PSEUDO_B_LATENTS, bnoise,
                conditional_dict, device, int(seed) + 333 * ck,
            )
            mae = float((tmp[:, a:prefix_latents].float() - real_b.float()).abs().mean().item())
            maes.append(mae)
            print(f"    rf_pseudo B cand{ck} mae={mae:.5g}", flush=True)
        output[:, a:prefix_latents] = real_b
        kv = _rf_replay_clean(
            pipeline, output, prefix_latents, conditional_dict, device,
        )
        pseudo_fire = maes[1] < maes[0] - float(_PSEUDO_GAMMA)
        pseudo_rows = {"mae0": maes[0], "mae1": maes[1], "fire": bool(pseudo_fire)}
        print(f"    rf_pseudo fire={pseudo_fire} mae0={maes[0]:.5g} mae1={maes[1]:.5g}", flush=True)
        if not pseudo_fire:
            look_k = 1

    num_blocks = n_gen // block
    raw_steps = pipeline.denoising_step_list
    steps = [float(s) for s in list(raw_steps)]
    step_tensor = (
        raw_steps.to(device=device)
        if hasattr(raw_steps, "to")
        else torch.tensor(steps, device=device, dtype=torch.float32)
    )
    n_step = len(steps)
    window_num = num_blocks + n_step - 1
    noisy_cache = torch.zeros_like(output)
    shared_timestep = torch.ones(
        [1, n_step * block], device=device, dtype=torch.float32,
    )
    for index, current_timestep in enumerate(reversed(steps)):
        shared_timestep[:, index * block:(index + 1) * block] *= current_timestep

    offset = prefix_latents
    for window_index in range(window_num):
        start_block = max(0, window_index - n_step + 1)
        end_block = min(num_blocks - 1, window_index)
        cur0 = offset + start_block * block
        cur1 = offset + (end_block + 1) * block
        n_frames = cur1 - cur0
        tail0 = cur0 - offset
        tail1 = cur1 - offset
        inject = n_frames == n_step * block or start_block == 0
        if n_frames == n_step * block:
            current_timestep = shared_timestep
        elif start_block == 0:
            current_timestep = shared_timestep[:, -n_frames:]
        else:
            current_timestep = shared_timestep[:, :n_frames]
        look_here = (
            look_k > 1
            and inject
            and start_block % ROLL_LOOK_EVERY_BLOCKS == 0
        )
        if method_name == "rf_sick_search":
            look_here = look_here and last_sick
        elif method_name == "rf_pseudo":
            look_here = look_here and pseudo_fire
        prev = output[:, cur0 - 1:cur0] if cur0 > 0 else None
        n_try = look_k if look_here else 1
        kv_snap = _snap_kv(kv) if n_try > 1 else None
        saved0 = noise[:, tail1 - block:tail1].clone() if inject else None
        cands = []
        for ck in range(n_try):
            if n_try > 1:
                _restore_kv(kv, kv_snap)
            if inject:
                if ck > 0:
                    rng.manual_seed(int(seed) + 10007 * (window_index + 1) + ck)
                    noise[:, tail1 - block:tail1] = torch.randn(
                        noise[:, tail1 - block:tail1].shape,
                        device=device, dtype=noise.dtype, generator=rng,
                    )
                noisy_input = torch.cat([
                    noisy_cache[:, cur0:cur1 - block],
                    noise[:, tail1 - block:tail1],
                ], dim=1)
            else:
                noisy_input = noisy_cache[:, cur0:cur1]
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=noisy_input,
                conditional_dict=conditional_dict,
                timestep=current_timestep,
                kv_cache=kv,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=cur0 * pipeline.frame_seq_length,
            )
            motion, seam = _latent_motion_seam(denoised_pred, prev)
            cands.append({
                "cand": ck,
                "pred": denoised_pred,
                "motion": motion,
                "seam": seam,
            })
        chosen = 0
        reason = "rolling_native"
        if n_try > 1:
            m0 = cands[0]["motion"]
            skip_still = (
                method_name == "rolling_look"
                and (
                    prefix_motion_early is None
                    or prefix_motion_early != prefix_motion_early
                    or prefix_motion_early < ROLL_STILL_MIN
                )
            )
            if skip_still:
                chosen, reason = 0, "look_skip_still"
            else:
                feasible = [
                    c for c in cands
                    if c["motion"] == c["motion"]
                    and m0 == m0
                    and c["motion"] >= ROLL_TRUST_FRAC * m0
                ]
                if not feasible:
                    chosen, reason = 0, "look_trust_reject"
                elif method_name in (
                    "rf_sick_search", "rf_pseudo", "rf_always_search",
                ):
                    best = max(feasible, key=lambda c: c["motion"])
                    chosen, reason = int(best["cand"]), "sick_motion"
                else:
                    best = min(feasible, key=lambda c: c["seam"])
                    chosen, reason = int(best["cand"]), "look_seam"
            look_picks.append({
                "window": window_index,
                "start_block": start_block,
                "chosen": chosen,
                "reason": reason,
                "motions": [c["motion"] for c in cands],
                "seams": [c["seam"] for c in cands],
            })
            print(
                f"    look win{window_index} pick={chosen} {reason} "
                f"m={[round(c['motion'], 5) if c['motion'] == c['motion'] else None for c in cands]}",
                flush=True,
            )
        if n_try > 1:
            _restore_kv(kv, kv_snap)
            if inject:
                if chosen == 0:
                    noise[:, tail1 - block:tail1] = saved0
                else:
                    rng.manual_seed(int(seed) + 10007 * (window_index + 1) + chosen)
                    noise[:, tail1 - block:tail1] = torch.randn(
                        noise[:, tail1 - block:tail1].shape,
                        device=device, dtype=noise.dtype, generator=rng,
                    )
            _, denoised_pred = pipeline.generator(
                noisy_image_or_video=(
                    torch.cat([
                        noisy_cache[:, cur0:cur1 - block],
                        noise[:, tail1 - block:tail1],
                    ], dim=1) if inject else noisy_cache[:, cur0:cur1]
                ),
                conditional_dict=conditional_dict,
                timestep=current_timestep,
                kv_cache=kv,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=cur0 * pipeline.frame_seq_length,
            )
        output[:, cur0:cur1] = denoised_pred
        with torch.no_grad():
            for block_idx in range(start_block, end_block + 1):
                rel = block_idx - start_block
                block_time_step = current_timestep[
                    :, rel * block:(rel + 1) * block
                ].mean().item()
                matches = torch.abs(step_tensor.to(device) - block_time_step) < 1e-4
                idxs = torch.nonzero(matches, as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue
                block_timestep_index = int(idxs[0].item())
                if block_timestep_index == n_step - 1:
                    continue
                next_timestep = step_tensor[block_timestep_index + 1]
                noisy_cache[:, offset + block_idx * block:offset + (block_idx + 1) * block] = (
                    pipeline.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn(
                            denoised_pred.flatten(0, 1).shape,
                            device=device, dtype=denoised_pred.dtype,
                            generator=rng,
                        ),
                        next_timestep.to(device) * torch.ones(
                            [denoised_pred.shape[0] * denoised_pred.shape[1]],
                            device=device, dtype=torch.long,
                        ),
                    ).unflatten(0, denoised_pred.shape[:2])[
                        :, rel * block:(rel + 1) * block
                    ]
                )
            context_timestep = torch.ones_like(current_timestep) * float(
                getattr(getattr(pipeline, "args", None), "context_noise", 0) or 0
            )
            first = denoised_pred[:, :block]
            ctx = context_timestep[:, :block]
            kwargs = dict(
                noisy_image_or_video=first,
                conditional_dict=conditional_dict,
                timestep=ctx,
                kv_cache=kv,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=cur0 * pipeline.frame_seq_length,
            )
            try:
                pipeline.generator(**kwargs, updating_cache=True)
            except TypeError:
                pipeline.generator(**kwargs)
        print(
            f"  rolling window {window_index}/{window_num - 1} "
            f"blocks {start_block}:{end_block} frames {cur0}:{cur1}",
            flush=True,
        )
        next_start = max(0, (window_index + 1) - n_step + 1)
        frozen = next_start * block
        committed = prefix_latents + frozen
        crossed = (
            next_start > start_block
            and frozen > 0
            and frozen % RECACHE_EVERY_LATENTS == 0
            and committed <= total
        )
        if recache_every_latents > 0 and crossed and committed < total:
            info = _recache_recent(
                pipeline, output, committed, RECACHE_LATENTS, device,
            )
            kv = _rf_replay_clean(
                pipeline, output, committed, conditional_dict, device,
            )
            recache_logs.append({
                **(info or {}),
                "window": window_index,
                "frozen_tail": frozen,
            })
        if method_name in RF_CONTROLLER_METHODS and crossed:
            chunk0 = committed - RECACHE_EVERY_LATENTS
            mot = _span_pixel_motion(
                pipeline, output, chunk0, RECACHE_EVERY_LATENTS,
            )
            if frozen >= 2 * RECACHE_EVERY_LATENTS:
                ref_m = _span_pixel_motion(
                    pipeline, output,
                    committed - 2 * RECACHE_EVERY_LATENTS,
                    RECACHE_EVERY_LATENTS,
                )
            else:
                ref_m = prefix_motion_early
            last_chunk_motion = mot
            last_sick = bool(
                mot == mot and ref_m is not None and ref_m == ref_m
                and ref_m > 0 and mot < RF_SICK_DROP * ref_m
            )
            if method_name == "rf_rewind" and last_sick:
                import torch as _torch
                saved = output[:, chunk0:committed].clone()
                tail0 = chunk0 - prefix_latents
                tail1 = committed - prefix_latents
                saved_noise = noise[:, tail0:tail1].clone()
                rng.manual_seed(int(seed) + 90000 + int(frozen))
                noise[:, tail0:tail1] = _torch.randn(
                    saved_noise.shape, device=device, dtype=noise.dtype,
                    generator=rng,
                )
                kv = _rf_roll_span(
                    pipeline, output, chunk0, RECACHE_EVERY_LATENTS,
                    noise[:, tail0:tail1], conditional_dict, device,
                    int(seed) + 90000 + int(frozen),
                )
                mot2 = _span_pixel_motion(
                    pipeline, output, chunk0, RECACHE_EVERY_LATENTS,
                )
                accepted = mot2 == mot2 and mot2 >= mot
                rewind_logs.append({
                    "frozen": int(frozen),
                    "mot0": mot,
                    "mot1": mot2,
                    "ref": ref_m,
                    "accepted": bool(accepted),
                })
                print(
                    f"    rf_rewind frozen={frozen} mot {mot:.5g}->{mot2:.5g} "
                    f"accept={accepted}",
                    flush=True,
                )
                if not accepted:
                    output[:, chunk0:committed] = saved
                    noise[:, tail0:tail1] = saved_noise
                    kv = _rf_replay_clean(
                        pipeline, output, committed, conditional_dict, device,
                    )
                    last_sick = False
                    last_chunk_motion = mot

    pixels = _decode_pixels(pipeline, output)
    prefix_pix_n = t2v_pixel_frames(prefix_latents)
    prefix_only = pixels[:prefix_pix_n]
    prefix_motion = (
        float(np.mean(np.abs(prefix_only[1:] - prefix_only[:-1])))
        if prefix_only.shape[0] >= 2 else None
    )
    n_div = sum(1 for p in look_picks if int(p.get("chosen", 0)) != 0)
    chunk_logs = [{
        "chunk": 0,
        "chosen_cand": n_div,
        "search_k": look_k,
        "method": method_name,
        "searched": bool(look_k > 1),
        "gate_reason": (
            f"rolling_rho={used_rho:.3g}" if look_k <= 1
            else f"rolling_look n_div={n_div}"
        ),
        "prefix_motion": _json_float(prefix_motion),
        "rho": _json_float(used_rho),
        "rho_mode": rho_mode,
        "look_picks": look_picks,
        "recache_logs": recache_logs,
        "rewind_logs": rewind_logs,
        "last_chunk_motion": _json_float(last_chunk_motion),
        "pseudo_fire": bool(pseudo_fire),
        "pseudo_rows": pseudo_rows,
        "chosen_score": None,
        "chosen_motion_score": _json_float(last_chunk_motion),
    }]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pixels, tuple(output.shape), None, chunk_logs, prefix_pix_n


def generate_knob_probe(
    pipeline,
    video_path: Path,
    prompt: str,
    prefix_latents: int,
    chunk_latents: int,
    seed: int,
    device,
    default_shift: float,
    default_cfg: float,
):
    """One generated chunk per (shift, cfg). Compare to default (8, 1)."""
    import torch

    prefix = encode_prefix_video(pipeline, Path(video_path), prefix_latents, device)
    conditional_dict = pipeline.text_encoder(text_prompts=[prompt])
    total = prefix_latents + chunk_latents
    rows = []
    default_pixels = None
    for shift, cfg in itertools.product(SHIFT_GRID, CFG_GRID):
        output = torch.zeros(
            [1, total, LATENT_C, LATENT_H, LATENT_W],
            device=device, dtype=torch.bfloat16,
        )
        output[:, :prefix_latents] = prefix[:, :prefix_latents]
        latents, pixels, _ns = _run_one_chunk(
            pipeline, output, prefix_latents, chunk_latents,
            conditional_dict, seed, 0, 0, device, shift, cfg,
        )
        gen = pixels[t2v_pixel_frames(prefix_latents):]
        last = pixels[t2v_pixel_frames(prefix_latents) - 1]
        free = gen_free_signals(gen, last)
        if math.isclose(shift, default_shift) and math.isclose(cfg, default_cfg):
            default_pixels = pixels.copy()
        mae = (
            _pixel_mae(pixels, default_pixels)
            if default_pixels is not None else 0.0
        )
        rec = {
            "shift": float(shift),
            "cfg": float(cfg),
            "is_default": math.isclose(shift, default_shift)
            and math.isclose(cfg, default_cfg),
            "pixel_mae_vs_default": _json_float(mae),
            "n_frames": int(pixels.shape[0]),
            **{k: _json_float(free[k]) for k in free},
            "latent_norm": _json_float(float(latents.float().norm().cpu())),
        }
        rows.append(rec)
        print(
            f"  probe shift={shift} cfg={cfg} motion={free['temporal_motion']:.5g} "
            f"mae={mae:.5g} sharp={free['sharpness']:.4g}",
            flush=True,
        )
        apply_shift(pipeline, default_shift)
        apply_guidance(pipeline, default_cfg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # second pass: default pixels may have been last; recompute MAE vs default row
    default_row = next((r for r in rows if r["is_default"]), None)
    moved = []
    for r in rows:
        if default_row is None:
            r["moved"] = False
            continue
        mae = r.get("pixel_mae_vs_default")
        mot = r.get("temporal_motion")
        dmot = default_row.get("temporal_motion")
        rel_mot = None
        if mot is not None and dmot not in (None, 0):
            rel_mot = abs(mot - dmot) / (abs(dmot) + 1e-8)
        r["rel_motion_vs_default"] = _json_float(rel_mot)
        r["moved"] = bool(
            (mae is not None and mae > 1e-3)
            or (rel_mot is not None and rel_mot > 0.05)
        )
        if r["moved"] and not r["is_default"]:
            moved.append(r)
    shift_live = any(
        r["moved"] and not math.isclose(r["shift"], default_shift)
        and math.isclose(r["cfg"], default_cfg)
        for r in rows
    )
    cfg_live = any(
        r["moved"] and math.isclose(r["shift"], default_shift)
        and not math.isclose(r["cfg"], default_cfg)
        for r in rows
    )
    apply_shift(pipeline, default_shift)
    apply_guidance(pipeline, default_cfg)
    return {
        "rows": rows,
        "shift_live": bool(shift_live),
        "cfg_live": bool(cfg_live),
        "n_moved": len(moved),
        "recommendation": (
            "keep shift_search" if shift_live else "drop shift_search"
        ) + "; " + (
            "keep cfg search" if cfg_live else "drop cfg (DMD likely CFG-free)"
        ),
    }


def _video_worker_count(args) -> int:
    raw = os.environ.get("VIDEO_WORKERS")
    if raw:
        return max(1, int(raw))
    return max(1, int(getattr(args, "video_workers", 1) or 1))


def _wait_gpu_slot(out_dir: Path, worker_id: int) -> None:
    if worker_id <= 0:
        return
    prev = out_dir / f".gpu_slot_{worker_id - 1}.ready"
    print(f"video-worker {worker_id} waiting for {prev.name}", flush=True)
    t0 = time.time()
    while not prev.is_file():
        if time.time() - t0 > 1800:
            raise RuntimeError(f"timed out waiting for {prev}")
        time.sleep(2)


def _mark_gpu_slot(out_dir: Path, worker_id: int) -> None:
    (out_dir / f".gpu_slot_{worker_id}.ready").write_text("loaded\n")


def _merge_worker_summaries(out_dir: Path, workers: int) -> int:
    rows = []
    template = None
    missing = []
    for w in range(workers):
        path = out_dir / f"summary.w{w}.json"
        if not path.is_file():
            missing.append(str(path))
            continue
        blob = json.loads(path.read_text())
        if template is None:
            template = {k: v for k, v in blob.items() if k != "rows"}
        rows.extend(blob.get("rows") or [])
    if template is None:
        raise RuntimeError(
            f"no worker summaries under {out_dir} (missing {missing})"
        )
    rows.sort(key=lambda r: int(r.get("item_index", 10**9)))
    template["n"] = len(rows)
    template["n_ok"] = sum(1 for r in rows if r.get("ok"))
    template["video_workers"] = workers
    template["rows"] = rows
    (out_dir / "summary.json").write_text(json.dumps(template, indent=2))
    print(json.dumps({k: template[k] for k in template if k != "rows"}, indent=2))
    if missing:
        print(f"WARNING: missing worker summaries: {missing}", flush=True)
        return 2
    return 0 if template["n_ok"] == template["n"] else 2


def _spawn_video_workers(workers: int, out_dir: Path) -> int:
    """Two (or more) independent processes on one GPU. Same pixels as serial.

    Tensor-batching k candidates is blocked: the 137-frame KV cache is
    ~39 GB, so k=4 copies miss an H200. Videos do not share state, so
    packing them is the H200 fill that does not change the sampler.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(".gpu_slot_*.ready"):
        stale.unlink()
    procs = []
    print(
        f"packing {workers} video workers on this GPU "
        f"(candidate tensor-batch is off; KV is ~39 GB)",
        flush=True,
    )
    for w in range(workers):
        env = os.environ.copy()
        env["V2V_WORKER_ID"] = str(w)
        env["VIDEO_WORKERS"] = str(workers)
        procs.append(subprocess.Popen([sys.executable, *sys.argv], env=env))
    rc = 0
    for p in procs:
        p.wait()
        if p.returncode not in (0, None) and rc == 0:
            rc = int(p.returncode)
    merged = _merge_worker_summaries(out_dir, workers)
    return rc or merged


def _record_common(args, item, extra: dict) -> dict:
    rec = {
        "ok": True,
        "task": "v2v",
        "method": args.method if args.method != "always_bon" else "seed_bon",
        "horizon_s_requested": args.horizon_s,
        "prefix_latents": args.prefix_latents,
        "chunk_latents": args.chunk_latents,
        "seed": args.seed,
        **item,
        **extra,
    }
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf-root", required=True)
    ap.add_argument("--wan-dir", required=True)
    ap.add_argument("--sf-ckpt", required=True)
    ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--prefix-latents", type=int, default=PREFIX_LATENTS_DEFAULT)
    ap.add_argument("--chunk-latents", type=int, default=21)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--method", choices=METHODS, default="notta")
    ap.add_argument("--search-k", type=int, default=4)
    ap.add_argument("--search-from-chunk", type=int, default=0,
                    help="first GENERATED chunk index to search (prefix is not a chunk)")
    ap.add_argument("--seam-weight", type=float, default=1.0)
    ap.add_argument("--backtrack-threshold", type=float, default=2.0)
    ap.add_argument("--backtrack-motion-frac", type=float, default=0.4)
    ap.add_argument("--default-shift", type=float, default=DEFAULT_SHIFT)
    ap.add_argument("--default-cfg", type=float, default=DEFAULT_CFG)
    ap.add_argument("--live-min", type=float, default=LIVE_SEARCH_MIN)
    ap.add_argument("--pseudo-gamma", type=float, default=PSEUDO_GAMMA)
    ap.add_argument("--noise-tau", type=float, default=NOISE_TAU)
    ap.add_argument("--sink-size", type=int, default=3)
    ap.add_argument("--local-attn-size", type=int, default=12)
    ap.add_argument("--ll-root", default="")
    ap.add_argument("--ll-base", default="")
    ap.add_argument("--ll-lora", default="")
    ap.add_argument("--rf-root", default="")
    ap.add_argument("--rf-ckpt", default="")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument(
        "--video-workers", type=int, default=1,
        help="Independent videos packed on one GPU. Default 1; sbatch "
             "sets 2 on H200. Do not tensor-batch candidates (39 GB KV).",
    )
    args = ap.parse_args()

    if args.prefix_latents % 3 != 0:
        raise SystemExit(
            f"prefix_latents={args.prefix_latents} must be a multiple of "
            "num_frame_per_block=3"
        )
    n_gen = t2v_latents_for_horizon(args.horizon_s, args.chunk_latents)
    if n_gen % args.chunk_latents != 0:
        raise SystemExit(f"n_gen={n_gen} not divisible by {args.chunk_latents}")
    n_chunks = n_gen // args.chunk_latents
    n_pix = t2v_pixel_frames(args.prefix_latents + n_gen)
    method = "seed_bon" if args.method == "always_bon" else args.method
    global _LIVE_SEARCH_MIN, _PSEUDO_GAMMA, _NOISE_TAU
    _LIVE_SEARCH_MIN = float(args.live_min)
    _PSEUDO_GAMMA = float(args.pseudo_gamma)
    _NOISE_TAU = float(args.noise_tau)

    host = _v2v_host_name(method)
    if host == "longlive":
        host_root = Path(args.ll_root or args.sf_root).resolve()
    elif host == "rolling":
        host_root = Path(args.rf_root or args.sf_root).resolve()
    else:
        host_root = Path(args.sf_root).resolve()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    items = discover_v2v_items(Path(args.video_dir), args.n)
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
    if not items:
        print("shard is empty; nothing to do")
        return 0

    workers = _video_worker_count(args)
    worker_id_env = os.environ.get("V2V_WORKER_ID")
    if workers > 1 and worker_id_env is None and len(items) > 1:
        return _spawn_video_workers(workers, out_dir)

    if worker_id_env is not None:
        worker_id = int(worker_id_env)
        n_all = len(items)
        items_indexed = [
            (i, it) for i, it in enumerate(items) if i % workers == worker_id
        ]
        print(
            f"video-worker {worker_id}/{workers} "
            f"n={len(items_indexed)}/{n_all}",
            flush=True,
        )
        _wait_gpu_slot(out_dir, worker_id)
    else:
        worker_id = None
        items_indexed = list(enumerate(items))

    torch = _bootstrap_sf(host_root)
    _seed_torch(torch, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        print(
            f"gpu={torch.cuda.get_device_name(0)} "
            f"mem={prop.total_memory / 1e9:.1f}G",
            flush=True,
        )
    print(
        f"device={device} torch={torch.__version__} task=V2V method={method} "
        f"host={host} sampler={'rolling' if _uses_rolling_sampler(method) else 'chunked'} "
        f"horizon={args.horizon_s}s prefix_lat={args.prefix_latents} "
        f"n_gen={n_gen} chunk={args.chunk_latents} n_chunks={n_chunks} "
        f"n_pix={n_pix} n_items={len(items)} live_min={_LIVE_SEARCH_MIN}"
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
    n_cache = args.prefix_latents + n_gen + 2
    if host == "longlive":
        if str(_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS))
        from v2v_hosts import load_longlive_pipeline
        pipeline = load_longlive_pipeline(
            host_root, Path(args.wan_dir),
            Path(args.ll_base), Path(args.ll_lora),
            device, n_cache_frames=n_cache,
            sink_size=9 if method == "longlive_prefix_sink" else int(args.sink_size),
            local_attn_size=int(args.local_attn_size),
        )
        if args.default_shift == DEFAULT_SHIFT:
            args.default_shift = 5.0
    elif host == "rolling":
        if str(_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS))
        from v2v_hosts import load_rolling_pipeline
        pipeline = load_rolling_pipeline(
            host_root, Path(args.wan_dir), Path(args.rf_ckpt),
            device, n_cache_frames=n_cache,
        )
        if args.default_shift == DEFAULT_SHIFT:
            args.default_shift = 5.0
        if method == "rf_sink":
            from v2v_hosts import apply_sink_size
            apply_sink_size(
                pipeline, int(args.sink_size), int(args.local_attn_size),
            )
    else:
        pipeline = load_pipeline(
            host_root, Path(args.wan_dir), Path(args.sf_ckpt),
            device, n_cache_frames=n_cache,
            independent_first_frame=False,
        )
        if method == "sf_sink":
            if str(_SCRIPTS) not in sys.path:
                sys.path.insert(0, str(_SCRIPTS))
            from v2v_hosts import apply_sink_size
            apply_sink_size(
                pipeline, int(args.sink_size), int(args.local_attn_size),
            )
    if int(pipeline.frame_seq_length) != FRAME_SEQ_PER_LATENT:
        raise RuntimeError(
            f"pipeline.frame_seq_length={pipeline.frame_seq_length} "
            f"!= {FRAME_SEQ_PER_LATENT}"
        )
    hooks = inspect_sampling_hooks(pipeline)
    apply_shift(pipeline, args.default_shift)
    apply_guidance(pipeline, args.default_cfg)
    print(f"pipeline loaded in {time.time() - t_load:.1f}s")
    _cuda_mem("after_pipeline_load")
    if worker_id is not None:
        _mark_gpu_slot(out_dir, worker_id)

    rows = []
    probe_aggregate = []
    for i, item in items_indexed:
        stem = (
            f"{i:03d}_{item['stem']}_h{int(args.horizon_s)}s_"
            f"{method}_s{args.seed}"
        )
        mp4 = out_dir / f"{stem}.mp4"
        meta_path = out_dir / f"{stem}.json"
        if method != "knob_probe" and mp4.is_file() and mp4.stat().st_size > 10_000:
            print(f"skip existing {mp4.name}")
            rows.append({
                "ok": True, "skipped": True, "item_index": i,
                "mp4": str(mp4), **item,
            })
            continue
        print(f"[{i+1}/{len(items)}] V2V {method} {item['file_name']}")
        _seed_torch(torch, args.seed)
        t0 = time.time()
        try:
            with torch.inference_mode():
                if method == "knob_probe":
                    probe = generate_knob_probe(
                        pipeline, Path(item["video_path"]), item["prompt"],
                        args.prefix_latents, args.chunk_latents,
                        args.seed, device,
                        args.default_shift, args.default_cfg,
                    )
                    rec = _record_common(args, item, {
                        "item_index": i,
                        "seconds": round(time.time() - t0, 2),
                        "probe": probe,
                        "shift_live": probe["shift_live"],
                        "cfg_live": probe["cfg_live"],
                        "recommendation": probe["recommendation"],
                    })
                    probe_aggregate.append(probe)
                    print(
                        f"  probe {item['file_name']}: {probe['recommendation']} "
                        f"{rec['seconds']}s",
                        flush=True,
                    )
                else:
                    if _uses_rolling_sampler(method):
                        rho, rho_mode, look_k = 1.0, "fixed", 1
                        recache_every = 0
                        if method == "rolling_rho_lo":
                            rho = 0.5
                        elif method == "rolling_rho_hi":
                            rho = 2.0
                        elif method == "rolling_adapt":
                            rho_mode = "adapt"
                        elif method == "rolling_look":
                            look_k = max(2, int(args.search_k))
                        elif method == "rf_recache":
                            recache_every = RECACHE_EVERY_LATENTS
                        elif method in (
                            "rf_sick_search", "rf_pseudo", "rf_always_search",
                        ):
                            look_k = max(2, int(args.search_k))
                        video, lat_shape, ref, chunk_logs, prefix_pix = generate_rolling_v2v(
                            pipeline, Path(item["video_path"]), item["prompt"],
                            args.prefix_latents, n_gen, args.seed, device,
                            rho=rho, rho_mode=rho_mode, search_k=look_k,
                            method_name=method,
                            recache_every_latents=recache_every,
                        )
                    else:
                        video, lat_shape, ref, chunk_logs, prefix_pix = generate_chunked_v2v(
                            pipeline, Path(item["video_path"]), item["prompt"],
                            args.prefix_latents, n_gen, args.chunk_latents,
                            args.seed, device, method, args.search_k,
                            args.search_from_chunk, args.seam_weight,
                            args.backtrack_threshold, args.backtrack_motion_frac,
                            args.default_shift, args.default_cfg,
                        )
                    write_mp4(mp4, video, fps=FPS)
                    last = chunk_logs[-1] if chunk_logs else {}
                    tail = video[prefix_pix:] if video.shape[0] > prefix_pix else video
                    tail_motion = (
                        float(np.mean(np.abs(tail[1:] - tail[:-1])))
                        if tail.shape[0] >= 2 else float("nan")
                    )
                    rec = _record_common(args, item, {
                        "item_index": i,
                        "seconds": round(time.time() - t0, 2),
                        "mp4": str(mp4),
                        "n_frames": int(video.shape[0]),
                        "prefix_pix": int(prefix_pix),
                        "hw": [int(video.shape[1]), int(video.shape[2])],
                        "latent_shape": list(lat_shape),
                        "n_gen_latent": n_gen,
                        "n_chunks": n_chunks,
                        "search_k": args.search_k,
                        "n_divergent_chunks": sum(
                            1 for ch in chunk_logs
                            if ch["search_k"] > 1 and ch["chosen_cand"] != 0
                        ),
                        "n_backtracked": sum(
                            1 for ch in chunk_logs if ch.get("backtracked")
                        ),
                        "incoming_series": [
                            ch.get("incoming_drift") for ch in chunk_logs
                        ],
                        "outgoing_series": [
                            ch.get("outgoing_drift") for ch in chunk_logs
                        ],
                        "last_chunk_score": last.get("chosen_score"),
                        "last_chunk_motion_score": last.get("chosen_motion_score"),
                        "tail_motion": _json_float(tail_motion),
                        "ref_signals": _json_signals(ref),
                        "chunks": chunk_logs,
                    })
                    print(
                        f"  wrote {mp4.name}  T={video.shape[0]}  "
                        f"prefix_pix={prefix_pix}  tail_motion={tail_motion:.5g}  "
                        f"{rec['seconds']}s",
                        flush=True,
                    )
        except Exception as e:
            rec = {
                "ok": False,
                "task": "v2v",
                "item_index": i,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "seconds": round(time.time() - t0, 2),
                "method": method,
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
        _cuda_mem(f"after_video_{i:03d}")

    shift_live = any(r.get("shift_live") for r in rows if r.get("ok"))
    cfg_live = any(r.get("cfg_live") for r in rows if r.get("ok"))
    summary = {
        "n": len(rows),
        "n_ok": sum(1 for r in rows if r.get("ok")),
        "task": "v2v",
        "method": method,
        "host": host,
        "live_min": _LIVE_SEARCH_MIN,
        "horizon_s": args.horizon_s,
        "prefix_latents": args.prefix_latents,
        "n_gen_latent": n_gen,
        "chunk_latents": args.chunk_latents,
        "n_chunks": n_chunks,
        "n_pix": n_pix,
        "search_k": args.search_k,
        "seed": args.seed,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "video_workers": workers,
        "video_worker_id": worker_id,
        "video_dir": str(Path(args.video_dir).resolve()),
        "sampling_hooks": hooks,
        "shift_live": bool(shift_live) if method == "knob_probe" else None,
        "cfg_live": bool(cfg_live) if method == "knob_probe" else None,
        "rows": rows,
    }
    sum_name = f"summary.w{worker_id}.json" if worker_id is not None else "summary.json"
    (out_dir / sum_name).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    return 0 if summary["n_ok"] == summary["n"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
