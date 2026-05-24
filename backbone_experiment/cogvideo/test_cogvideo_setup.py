#!/usr/bin/env python3
"""
Quick sanity check for CogVideoX-5B-I2V setup.

Verifies:
  1. Model loads from local path or HuggingFace
  2. Panda-70M video loads and resizes to 720x480
  3. VAE encodes a frame to correct latent shape
  4. GPU memory usage is reported
  5. One flow-matching loss step completes
  6. Short continuation generates (5 denoising steps)
  7. Success/failure reported

Usage:
    python test_cogvideo_setup.py \\
        --model-path /path/to/CogVideoX-5b-I2V \\
        --video-path /path/to/test_video.mp4
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common_cogvideo import (
    load_cogvideo_components,
    load_video_frames,
    encode_video_cogvideo,
    encode_prompt_cogvideo,
    compute_flow_matching_loss_cogvideo,
    generate_video_cogvideo,
    torch_gc,
)


def _gpu_mem_mb() -> str:
    if not torch.cuda.is_available():
        return "N/A (no CUDA)"
    alloc = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    return f"allocated={alloc:.0f}MB, reserved={reserved:.0f}MB"


def main():
    parser = argparse.ArgumentParser(
        description="Sanity check CogVideoX-5B-I2V setup"
    )
    parser.add_argument("--model-path", type=str, required=True,
                        help="HuggingFace model ID or local path")
    parser.add_argument("--video-path", type=str, required=True,
                        help="Path to a test video (mp4)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    passed = 0
    failed = 0

    # ── Step 1: Load model ────────────────────────────────────────────
    print("=" * 60)
    print("[1/7] Loading CogVideoX-5B-I2V...")
    try:
        t0 = time.time()
        components = load_cogvideo_components(
            args.model_path, device=device, dtype=torch.bfloat16
        )
        load_time = time.time() - t0
        transformer = components["transformer"]
        vae = components["vae"]
        pipe = components["pipe"]
        tokenizer = components["tokenizer"]
        text_encoder = components["text_encoder"]

        n_blocks = len(transformer.transformer_blocks)
        n_params = sum(p.numel() for p in transformer.parameters())
        print(f"  OK — {n_blocks} blocks, {n_params:,} params, "
              f"loaded in {load_time:.1f}s")
        print(f"  GPU: {_gpu_mem_mb()}")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")
        failed += 1
        print(f"\nResult: {passed} passed, {failed} failed (cannot continue)")
        sys.exit(1)

    # ── Step 2: Load video ────────────────────────────────────────────
    print(f"\n[2/7] Loading video: {args.video_path}")
    try:
        pixel_frames = load_video_frames(
            args.video_path, num_frames=49, height=480, width=720
        )
        print(f"  OK — shape: {list(pixel_frames.shape)} "
              f"(expect [1, 3, 49, 480, 720])")
        assert pixel_frames.shape == (1, 3, 49, 480, 720), \
            f"Unexpected shape: {pixel_frames.shape}"
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")
        failed += 1
        print(f"\nResult: {passed} passed, {failed} failed (cannot continue)")
        sys.exit(1)

    # ── Step 3: Resize check ─────────────────────────────────────────
    print(f"\n[3/7] Confirming 720x480 resize...")
    try:
        h, w = pixel_frames.shape[3], pixel_frames.shape[4]
        assert h == 480 and w == 720, f"Got {w}x{h}, expected 720x480"
        val_range = (pixel_frames.min().item(), pixel_frames.max().item())
        print(f"  OK — {w}x{h}, value range: [{val_range[0]:.2f}, {val_range[1]:.2f}]")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")
        failed += 1

    # ── Step 4: VAE encode ────────────────────────────────────────────
    print(f"\n[4/7] Encoding 1 frame with CogVideoX VAE...")
    try:
        single_frame = pixel_frames[:, :, :1, :, :].to(device, torch.bfloat16)
        t0 = time.time()
        latents_1 = encode_video_cogvideo(vae, single_frame, device=device)
        enc_time = time.time() - t0
        print(f"  OK — latent shape: {list(latents_1.shape)}, "
              f"encoded in {enc_time:.2f}s")
        print(f"  GPU: {_gpu_mem_mb()}")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")
        failed += 1

    # ── Step 5: GPU memory report ─────────────────────────────────────
    print(f"\n[5/7] GPU memory report...")
    try:
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem / 1e9
            alloc = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            free = total - reserved
            print(f"  Total    : {total:.1f} GB")
            print(f"  Allocated: {alloc:.1f} GB")
            print(f"  Reserved : {reserved:.1f} GB")
            print(f"  Free     : {free:.1f} GB")
            print(f"  Device   : {torch.cuda.get_device_name(0)}")
        else:
            print("  No CUDA device available")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")
        failed += 1

    # ── Step 6: Flow-matching loss step ───────────────────────────────
    print(f"\n[6/7] Running 1 flow-matching loss step...")
    try:
        # Encode a small batch of frames for the loss computation
        small_clip = pixel_frames[:, :, :5, :, :].to(device, torch.bfloat16)
        latents_small = encode_video_cogvideo(vae, small_clip, device=device)

        prompt_embeds, prompt_mask = encode_prompt_cogvideo(
            tokenizer, text_encoder, "A test video clip",
            device=device, dtype=torch.bfloat16,
        )

        # Move transformer to device for loss computation
        transformer = transformer.to(device)
        transformer.eval()

        # Need grad for testing the loss step
        for p in transformer.parameters():
            p.requires_grad = True
        transformer.enable_gradient_checkpointing()

        t0 = time.time()
        loss = compute_flow_matching_loss_cogvideo(
            transformer=transformer,
            latents=latents_small,
            prompt_embeds=prompt_embeds,
            device=device,
            dtype=torch.bfloat16,
        )
        loss.backward()
        loss_time = time.time() - t0

        print(f"  OK — loss={loss.item():.4f}, computed in {loss_time:.2f}s")
        print(f"  GPU: {_gpu_mem_mb()}")

        # Freeze again after test
        for p in transformer.parameters():
            p.requires_grad = False
        transformer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # ── Step 7: Short generation ──────────────────────────────────────
    print(f"\n[7/7] Generating short continuation (5 denoising steps)...")
    try:
        from PIL import Image

        pf = pixel_frames.squeeze(0)  # [C, T, H, W]
        pf_01 = ((pf + 1.0) / 2.0).clamp(0, 1)
        frame_np = (
            pf_01[:, 0].permute(1, 2, 0).float().numpy() * 255
        ).astype(np.uint8)
        cond_image = Image.fromarray(frame_np)

        # Ensure all components are on device
        vae.to(device)
        text_encoder.to(device)
        transformer.to(device)

        t0 = time.time()
        gen_frames = generate_video_cogvideo(
            pipe=pipe,
            image=cond_image,
            prompt="A test video clip",
            num_frames=9,
            num_inference_steps=5,
            guidance_scale=6.0,
            seed=42,
            device=device,
        )
        gen_time = time.time() - t0

        print(f"  OK — generated {gen_frames.shape[0]} frames "
              f"({gen_frames.shape[1]}x{gen_frames.shape[2]}), "
              f"took {gen_time:.1f}s")
        print(f"  Output range: [{gen_frames.min():.3f}, {gen_frames.max():.3f}]")
        print(f"  GPU: {_gpu_mem_mb()}")
        passed += 1
    except Exception as e:
        print(f"  FAIL — {e}")
        import traceback
        traceback.print_exc()
        failed += 1

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if failed == 0:
        print(f"ALL CHECKS PASSED ({passed}/{passed + failed})")
    else:
        print(f"SOME CHECKS FAILED: {passed} passed, {failed} failed")
    print("=" * 60)

    torch_gc()
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
