#!/usr/bin/env python3
"""NOTTA I2V / prefix-conditioned continuation on Wan2.1-T2V-1.3B + Self-Forcing.

Uses the official CausalInferencePipeline I2V path (encode first frame → AR
rollout with KV cache). Does NOT call Self-Forcing inference.py (torchvision
2.13 dropped write_video). Writes mp4 via imageio.

Horizon math (16 fps, Wan VAE temporal 4, num_frame_per_block=3):
  n_gen must be a multiple of 3; total latent = 1 (image) + n_gen
  pixel frames ≈ 1 + 4 * n_gen
  5 s  → n_gen=21  → 85 frames (~5.3 s)
  10 s → n_gen=42  → 169 frames (~10.6 s)
  30 s → n_gen=120 → 481 frames (~30.1 s)

KV cache default in Self-Forcing is 21 frames (32760 tokens). We enlarge it
to the rollout length. Do not change local_attn_size (that would alter the
attention window).
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


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
LATENT_C, LATENT_H, LATENT_W = 16, 60, 104
PIXEL_H, PIXEL_W = 480, 832
FPS = 16
BLOCK = 3
# Tokens per latent frame at 480x832 (30x52 patches). Official default
# cache is 21 * 1560 = 32760. Do NOT use generator.seq_len (also 32760) as
# "tokens per frame" — job 15876397 did that via pipeline.frame_seq_length
# resolving to 32760 and allocated 24*32760*30*2*12*128*bf16 ≈ 135 GB.
FRAME_SEQ_PER_LATENT = 1560


def gen_latents_for_horizon(seconds: float, block: int = BLOCK) -> int:
    target_pix = max(int(round(seconds * FPS)), 9)
    n_latent = max(4, int(round((target_pix - 1) / 4.0)) + 1)
    n_gen = n_latent - 1
    n_gen = ((n_gen + block - 1) // block) * block
    return n_gen


def pixel_frames(n_gen: int) -> int:
    return 1 + 4 * n_gen


def ensure_wan_symlink(sf_root: Path, wan_dir: Path) -> Path:
    dest_parent = sf_root / "wan_models"
    dest_parent.mkdir(parents=True, exist_ok=True)
    dest = dest_parent / "Wan2.1-T2V-1.3B"
    wan_dir = wan_dir.resolve()
    if dest.is_symlink() or dest.exists():
        if dest.resolve() != wan_dir:
            if dest.is_symlink() or dest.is_file():
                dest.unlink()
            else:
                raise FileExistsError(
                    f"{dest} exists and is not the Wan checkpoint dir"
                )
            dest.symlink_to(wan_dir, target_is_directory=True)
    else:
        dest.symlink_to(wan_dir, target_is_directory=True)
    return dest


def _load_prompt_map(i2v_dir: Path) -> dict[str, str]:
    """file_name → caption from VBench-I2V json if present."""
    out: dict[str, str] = {}
    for name in ("i2v-bench-info.json", "i2v-bench-info-vertical.json"):
        hits = list(i2v_dir.rglob(name))
        for p in hits:
            try:
                data = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(data, list):
                continue
            for item in data:
                if not isinstance(item, dict):
                    continue
                fn = item.get("file_name") or item.get("filename")
                cap = item.get("caption") or item.get("prompt")
                if fn and cap:
                    out[Path(fn).name] = str(cap)
    return out


def discover_items(i2v_dir: Path, n: int, prefer_ratio: str = "16-9") -> list[dict]:
    """Pair on-disk images with VBench-I2V captions. Prefer 16-9 crops."""
    i2v_dir = i2v_dir.resolve()
    prompts = _load_prompt_map(i2v_dir)
    imgs = [p for p in i2v_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise FileNotFoundError(f"no images under {i2v_dir}")

    def score(p: Path) -> tuple:
        name = p.name
        has_prompt = 0 if name in prompts else 1
        ratio_hit = 0 if prefer_ratio in p.parts else 1
        crop_hit = 0 if "crop" in p.parts else 1
        return (has_prompt, ratio_hit, crop_hit, str(p))

    imgs = sorted(imgs, key=score)
    seen: set[str] = set()
    items = []
    for p in imgs:
        key = p.name
        if key in seen:
            continue
        seen.add(key)
        prompt = prompts.get(key) or p.stem.replace("_", " ")
        items.append({
            "image_path": str(p),
            "file_name": key,
            "prompt": prompt,
            "stem": p.stem[:80].replace(" ", "_"),
        })
        if len(items) >= n:
            break
    if not items:
        raise FileNotFoundError(f"no usable I2V items under {i2v_dir}")
    return items


def write_mp4(path: Path, video_01: "np.ndarray", fps: int = FPS) -> None:
    """video_01: [T, H, W, C] float in [0, 1]."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = (np.clip(video_01, 0.0, 1.0) * 255.0).astype(np.uint8)
    imageio.mimwrite(str(path), frames, fps=fps, codec="libx264", quality=8)


def _cuda_mem(tag: str) -> None:
    import torch
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"cuda_mem[{tag}] allocated={alloc:.2f}G reserved={reserved:.2f}G")


def enlarge_kv_cache(pipeline, n_frames: int) -> None:
    """Replace Self-Forcing's 21-frame KV cache without changing local attn."""
    import torch

    n_blocks = int(pipeline.num_transformer_blocks)
    kv_cache_size = int(n_frames) * FRAME_SEQ_PER_LATENT
    # bf16 K+V, 30 blocks, 12 heads x 128: refuse a 15876397-sized alloc.
    est_gb = n_blocks * 2 * kv_cache_size * 12 * 128 * 2 / 1e9
    reported = getattr(pipeline, "frame_seq_length", None)
    print(f"KV cache: {n_frames} frames x {FRAME_SEQ_PER_LATENT} tokens "
          f"(pipeline.frame_seq_length={reported}) -> {kv_cache_size} tokens, "
          f"est {est_gb:.2f} GB")
    if est_gb > 48:
        raise RuntimeError(
            f"KV cache estimate {est_gb:.1f} GB exceeds 48 GB safety cap "
            f"(n_frames={n_frames}, tokens={kv_cache_size}). "
            f"Refusing to allocate (see job 15876397)."
        )

    def _initialize_kv_cache(batch_size, dtype, device):
        cache = []
        for _ in range(n_blocks):
            cache.append({
                "k": torch.zeros(
                    [batch_size, kv_cache_size, 12, 128],
                    dtype=dtype, device=device,
                ),
                "v": torch.zeros(
                    [batch_size, kv_cache_size, 12, 128],
                    dtype=dtype, device=device,
                ),
                "global_end_index": torch.tensor(
                    [0], dtype=torch.long, device=device,
                ),
                "local_end_index": torch.tensor(
                    [0], dtype=torch.long, device=device,
                ),
            })
        pipeline.kv_cache1 = cache
        _cuda_mem("after_kv_init")

    pipeline._initialize_kv_cache = _initialize_kv_cache
    pipeline._cache_frames = n_frames


def install_sdpa_attention_fallback() -> None:
    """Self-Forcing's Wan blocks call flash_attention() directly.

    Job 15858704 died at `assert FLASH_ATTN_2_AVAILABLE` because we skip
    compiling flash-attn (2h TIMEOUT on 15796574). wan/modules/attention.py
    already has an SDPA path on `attention()`, but `model.py` imports
    `flash_attention` by name. Patch both modules. H200 + torch 2.13 SDPA
    uses the built-in flash kernel; padding masks (k_lens) are ignored —
    acceptable for this smoke, revisit if VBench looks off.
    """
    from wan.modules import attention as attn
    from wan.modules import model as wan_model

    if attn.FLASH_ATTN_2_AVAILABLE or attn.FLASH_ATTN_3_AVAILABLE:
        print("flash-attn available; no SDPA fallback")
        return

    import torch

    def _flash_attention_sdpa(
        q, k, v,
        q_lens=None, k_lens=None, dropout_p=0.,
        softmax_scale=None, q_scale=None, causal=False,
        window_size=(-1, -1), deterministic=False,
        dtype=torch.bfloat16, version=None,
    ):
        return attn.attention(
            q, k, v,
            q_lens=q_lens, k_lens=k_lens, dropout_p=dropout_p,
            softmax_scale=softmax_scale, q_scale=q_scale, causal=causal,
            window_size=window_size, deterministic=deterministic,
            dtype=dtype, fa_version=version,
        )

    wan_model.flash_attention = _flash_attention_sdpa
    attn.flash_attention = _flash_attention_sdpa
    print("WARNING: flash-attn missing; using PyTorch SDPA fallback")


def load_pipeline(
    sf_root: Path,
    wan_dir: Path,
    sf_ckpt: Path,
    device,
    n_cache_frames: int,
    independent_first_frame: bool = True,
):
    import torch
    from omegaconf import OmegaConf
    from pipeline import CausalInferencePipeline

    ensure_wan_symlink(sf_root, wan_dir)
    default_cfg = OmegaConf.load(str(sf_root / "configs" / "default_config.yaml"))
    dmd_cfg = OmegaConf.load(str(sf_root / "configs" / "self_forcing_dmd.yaml"))
    config = OmegaConf.merge(default_cfg, dmd_cfg)
    # Official I2V needs an independent first frame; default_config has this false
    # and then 1-frame prefix fails the num_frame_per_block assert.
    # T2V must keep this false: every latent is generated in blocks of 3.
    config.independent_first_frame = bool(independent_first_frame)

    pipeline = CausalInferencePipeline(config, device=device)
    state = torch.load(str(sf_ckpt), map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or "generator_ema" not in state:
        raise KeyError(f"expected generator_ema in {sf_ckpt}, keys={list(state)[:12] if isinstance(state, dict) else type(state)}")
    pipeline.generator.load_state_dict(state["generator_ema"])
    del state

    pipeline = pipeline.to(dtype=torch.bfloat16)
    # Official recipe: swap T5 in/out. Full umt5-xxl fp32 on GPU plus a
    # compiled flex_attention workspace filled the H200 (15877786, 138 GB)
    # even after the KV-cache size fix.
    try:
        from demo_utils.memory import DynamicSwapInstaller
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
        print("text_encoder: DynamicSwapInstaller")
    except Exception as e:
        pipeline.text_encoder.to(device=device)
        print(f"text_encoder: full GPU ({type(e).__name__}: {e})")
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    enlarge_kv_cache(pipeline, n_cache_frames)
    _cuda_mem("after_load")
    return pipeline


def encode_image(pipeline, image_path: Path, device):
    import torch
    from PIL import Image
    from torchvision import transforms

    tfm = transforms.Compose([
        transforms.Resize((PIXEL_H, PIXEL_W)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    img = Image.open(image_path).convert("RGB")
    pixel = tfm(img)  # [C, H, W]
    # official: [B, C, T=1, H, W]
    video = pixel.unsqueeze(0).unsqueeze(2).to(device=device, dtype=torch.bfloat16)
    latent = pipeline.vae.encode_to_latent(video).to(device=device, dtype=torch.bfloat16)
    return latent  # [1, 1, 16, 60, 104]


def generate_one(pipeline, image_path: Path, prompt: str, n_gen: int, seed: int, device):
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    initial = encode_image(pipeline, Path(image_path), device)
    noise = torch.randn(
        [1, n_gen, LATENT_C, LATENT_H, LATENT_W],
        device=device, dtype=torch.bfloat16,
    )
    _cuda_mem("before_inference")
    with torch.inference_mode():
        video, latents = pipeline.inference(
            noise=noise,
            text_prompts=[prompt],
            return_latents=True,
            initial_latent=initial,
            low_memory=True,
        )
    # video: [B, T, C, H, W] in [0, 1]
    arr = video[0].float().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    try:
        pipeline.vae.model.clear_cache()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arr, tuple(latents.shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf-root", required=True)
    ap.add_argument("--wan-dir", required=True)
    ap.add_argument("--sf-ckpt", required=True)
    ap.add_argument("--i2v-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--horizon-s", type=float, default=5.0)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    sf_root = Path(args.sf_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    n_gen = gen_latents_for_horizon(args.horizon_s)
    n_pix = pixel_frames(n_gen)
    items = discover_items(Path(args.i2v_dir), args.n)
    items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard_id]
    if not items:
        print("shard is empty; nothing to do")
        return 0

    # Must be set before importing torch / Self-Forcing. causal_model.py does
    # `flex_attention = torch.compile(..., mode="max-autotune-no-cudagraphs")`
    # at import time; on an H200 that autotune workspace is the remaining
    # 138 GB after the KV-cache fix (job 15877786, identical footprint).
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    # Imports from Self-Forcing require cwd + sys.path at the clone root
    # (hardcoded wan_models/Wan2.1-T2V-1.3B paths in wan_wrapper.py).
    sys.path.insert(0, str(sf_root))
    os.chdir(sf_root)

    import torch
    torch._dynamo.config.disable = True
    # Official inference.py does this. Without it, 2 denoise blocks on a
    # 32760-padded seq (WanDiffusionWrapper.seq_len) fill the H200: job
    # 15879723 was 3.1 GB at before_inference / 10 GB after KV init, then
    # 138 GB by the 3rd block. Not the KV cache (est 6.90 GB, confirmed).
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} torch={torch.__version__} "
          f"horizon={args.horizon_s}s n_gen={n_gen} n_pix={n_pix} "
          f"n_items={len(items)} shard={args.shard_id}/{args.num_shards}")

    install_sdpa_attention_fallback()
    try:
        import wan.modules.causal_model as _cm
        from torch.nn.attention.flex_attention import flex_attention as _eager_fa
        _cm.flex_attention = _eager_fa
        print("flex_attention: eager (torch.compile disabled)")
    except Exception as e:
        print(f"flex_attention: leave as-is ({type(e).__name__}: {e})")
    _cuda_mem("after_import")

    t_load = time.time()
    pipeline = load_pipeline(
        sf_root, Path(args.wan_dir), Path(args.sf_ckpt),
        device, n_cache_frames=1 + n_gen + 2,
    )
    print(f"pipeline loaded in {time.time() - t_load:.1f}s")

    rows = []
    for i, item in enumerate(items):
        stem = f"{i:03d}_{item['stem']}_h{int(args.horizon_s)}s_s{args.seed}"
        mp4 = out_dir / f"{stem}.mp4"
        meta_path = out_dir / f"{stem}.json"
        if mp4.is_file() and mp4.stat().st_size > 10_000:
            print(f"skip existing {mp4.name}")
            rows.append({"ok": True, "skipped": True, "mp4": str(mp4), **item})
            continue
        print(f"[{i+1}/{len(items)}] {item['file_name']!r}")
        t0 = time.time()
        try:
            video, lat_shape = generate_one(
                pipeline, item["image_path"], item["prompt"],
                n_gen, args.seed, device,
            )
            write_mp4(mp4, video, fps=FPS)
            rec = {
                "ok": True,
                "seconds": round(time.time() - t0, 2),
                "mp4": str(mp4),
                "n_frames": int(video.shape[0]),
                "hw": [int(video.shape[1]), int(video.shape[2])],
                "latent_shape": list(lat_shape),
                "n_gen_latent": n_gen,
                "horizon_s_requested": args.horizon_s,
                "seed": args.seed,
                **item,
            }
            print(f"  wrote {mp4.name}  T={video.shape[0]}  {rec['seconds']}s")
        except Exception as e:
            rec = {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
                "seconds": round(time.time() - t0, 2),
                **item,
            }
            print(f"  FAIL {rec['error']}")
            print(rec["traceback"])
            try:
                import torch
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
        "horizon_s": args.horizon_s,
        "n_gen_latent": n_gen,
        "n_pix": n_pix,
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
