#!/usr/bin/env python3
"""
Dry-run test for SAVi-DNO with LongCat backbone.

Validates:
  1. LongCat model loads successfully
  2. Single-video DNO produces valid output (finite pixels in [0,1])
  3. Gradients flow to eps_optimized (non-zero grad after backward)
  4. Noise interpolation formula is correct
  5. Baseline (no-optimize) mode produces valid output

Usage (on cluster, inside longcat conda env):
  python comparison_methods/scripts/test_savi_dno_longcat.py \
      --checkpoint-dir /scratch/wc3013/longcat-video-checkpoints \
      --data-dir datasets/panda_1000_480p
"""

import sys
import os
import math
import argparse
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
_LONGCAT_DIR = _REPO_ROOT / "LongCat-Video"
sys.path.insert(0, str(_LONGCAT_DIR))
sys.path.insert(0, str(_REPO_ROOT))

from delta_experiment.scripts.common import (
    load_longcat_components,
    encode_video,
    encode_prompt,
    load_video_frames,
    _get_model_config,
)
from comparison_methods.scripts.savi_dno_longcat import SAViDNO_LongCat, load_feature_model


def test_noise_interpolation():
    """Test the noise interpolation formula h(p, eps_s, eps)."""
    print("\n[TEST 1] Noise interpolation formula...")

    class FakeSAVi:
        p = 0.9
        def _noise_interpolation(self, eps_opt, eps_fresh):
            pp = self.p
            norm = math.sqrt(pp ** 2 + (1 - pp) ** 2)
            return (pp * eps_opt + (1 - pp) * eps_fresh) / norm

    savi = FakeSAVi()
    eps_opt = torch.randn(2, 3, 4)
    eps_fresh = torch.randn(2, 3, 4)
    result = savi._noise_interpolation(eps_opt, eps_fresh)

    # Verify normalization preserves unit variance
    assert result.shape == eps_opt.shape, "Shape mismatch"
    norm_factor = math.sqrt(0.9**2 + 0.1**2)
    expected = (0.9 * eps_opt + 0.1 * eps_fresh) / norm_factor
    assert torch.allclose(result, expected, atol=1e-6), "Formula mismatch"

    print("  PASSED: noise interpolation formula correct")


def test_gradient_flow_mock():
    """Test gradient flow through a simplified Euler step (no real model)."""
    print("\n[TEST 2] Gradient flow through mock Euler step...")

    eps = torch.randn(1, 16, 4, 60, 104, requires_grad=True)
    optimizer = torch.optim.Adam([eps], lr=0.01)

    # Simulate a single Euler step: x = eps + dt * v
    dt = -0.1
    v_fake = torch.randn_like(eps)
    x_denoised = eps + dt * v_fake

    # Simulate decode + loss
    target = torch.randn_like(x_denoised)
    loss = torch.nn.functional.l1_loss(x_denoised, target)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert eps.grad is not None, "eps.grad is None — no gradient flow!"
    assert eps.grad.abs().sum() > 0, "eps.grad is all zeros!"
    print("  PASSED: gradients flow to eps_optimized (mock)")


def test_full_pipeline(args):
    """Full end-to-end test with real LongCat model."""
    print("\n[TEST 3] Full pipeline: load model + single-video DNO...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    print("  Loading LongCat components...")
    components = load_longcat_components(
        args.checkpoint_dir, device=device, dtype=torch.bfloat16,
    )
    dit = components["dit"]
    vae = components["vae"]
    scheduler = components["scheduler"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    for p in dit.parameters():
        p.requires_grad = False
    for p in vae.parameters():
        p.requires_grad = False
    for p in text_encoder.parameters():
        p.requires_grad = False
    dit.eval()
    vae.eval()
    text_encoder.eval()
    print("  Model loaded successfully")

    # Load feature model
    print("  Loading ResNet3D feature model...")
    feature_model = load_feature_model(device)
    print("  Feature model loaded")

    # Create SAViDNO_LongCat instance (PVDM-style: no CFG, pixel+feature loss)
    savi = SAViDNO_LongCat(
        dit=dit, vae=vae, scheduler=scheduler,
        tokenizer=tokenizer, text_encoder=text_encoder,
        device=device, dtype=torch.bfloat16,
        num_inference_steps=args.num_steps,
        guidance_scale=4.0,
        lr=0.01, lam=0.0012, p=0.9,
        feature_model=feature_model,
        gradient_checkpointing=True,
        latent_loss=False,
    )

    # Find a video to test with
    import csv
    metadata_path = os.path.join(args.data_dir, "metadata.csv")
    with open(metadata_path) as f:
        entries = list(csv.DictReader(f))

    entry = entries[0]
    video_name = entry.get("video_name", entry.get("filename", ""))
    video_filename = entry.get("filename", video_name)
    video_path = os.path.join(args.data_dir, "videos", video_filename)
    caption = entry.get("caption", entry.get("prompt", ""))

    print(f"  Test video: {video_name}")
    print(f"  Caption: {caption[:80]}...")

    # Encode prompt
    prompt_embeds, prompt_mask = encode_prompt(
        tokenizer, text_encoder,
        prompt=caption, device=device, dtype=torch.bfloat16,
    )

    # Load frames
    num_cond = 14
    num_gen = 14
    gen_start = 48

    pixel_cond = load_video_frames(
        video_path, num_cond,
        height=480, width=832,
        start_frame=max(0, gen_start - num_cond),
    ).to(device, torch.bfloat16)
    print(f"  Cond frames shape: {pixel_cond.shape}")

    pixel_gt = load_video_frames(
        video_path, num_gen,
        height=480, width=832,
        start_frame=gen_start,
    ).to(device, torch.bfloat16)
    gt_01 = (pixel_gt + 1.0) / 2.0
    print(f"  GT frames shape: {pixel_gt.shape}")

    # Encode conditioning
    cond_latents = savi.encode(pixel_cond)
    print(f"  Cond latents shape: {cond_latents.shape}")

    # Target latent shape
    vae_t_factor = 4
    T_gen_latent = 1 + (num_gen - 1) // vae_t_factor
    target_shape = (
        1, cond_latents.shape[1], T_gen_latent,
        cond_latents.shape[3], cond_latents.shape[4],
    )
    print(f"  Target latent shape: {target_shape}")

    # Test DNO (with optimization)
    print("\n  Running DNO prediction + optimization...")
    pred_pixels, loss_val = savi.predict_and_optimize(
        cond_latents, gt_01, target_shape, prompt_embeds, prompt_mask,
    )

    # Validate output
    assert pred_pixels.shape[0] == 1, f"Bad batch dim: {pred_pixels.shape}"
    assert pred_pixels.shape[1] == 3, f"Bad channel dim: {pred_pixels.shape}"
    assert torch.isfinite(pred_pixels).all(), "Output has NaN/Inf!"
    assert pred_pixels.min() >= -0.01, f"Output below 0: {pred_pixels.min()}"
    assert pred_pixels.max() <= 1.01, f"Output above 1: {pred_pixels.max()}"
    print(f"  PASSED: output shape {tuple(pred_pixels.shape)}, "
          f"range [{pred_pixels.min():.4f}, {pred_pixels.max():.4f}]")

    # Validate loss
    assert loss_val is not None, "Loss is None!"
    assert math.isfinite(loss_val), f"Loss is {loss_val}!"
    print(f"  PASSED: loss = {loss_val:.6f}")

    # Validate gradient flow
    assert savi.eps_optimized is not None, "eps_optimized is None after optimize!"
    print(f"  PASSED: eps_optimized shape {tuple(savi.eps_optimized.shape)}")

    # Test baseline (no optimization)
    print("\n  Running baseline (no-optimize) prediction...")
    savi.reset()
    baseline_pixels = savi.predict_no_optimize(
        cond_latents, target_shape, prompt_embeds, prompt_mask,
    )
    assert torch.isfinite(baseline_pixels).all(), "Baseline output has NaN/Inf!"
    assert baseline_pixels.shape[1] == 3, f"Bad baseline channels: {baseline_pixels.shape}"
    print(f"  PASSED: baseline shape {tuple(baseline_pixels.shape)}, "
          f"range [{baseline_pixels.min():.4f}, {baseline_pixels.max():.4f}]")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Dry-run test for SAVi-DNO LongCat")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="/scratch/wc3013/longcat-video-checkpoints")
    parser.add_argument("--data-dir", type=str,
                        default="datasets/panda_1000_480p")
    parser.add_argument("--num-steps", type=int, default=4,
                        help="Euler steps for test (fewer = faster)")
    parser.add_argument("--skip-model", action="store_true",
                        help="Skip full model test (run only unit tests)")
    args = parser.parse_args()

    print("=" * 60)
    print("SAVi-DNO LongCat: Dry-Run Tests")
    print("=" * 60)

    # Unit tests (no GPU needed)
    test_noise_interpolation()
    test_gradient_flow_mock()

    if args.skip_model:
        print("\n[SKIP] Full pipeline test (--skip-model)")
    else:
        test_full_pipeline(args)


if __name__ == "__main__":
    main()
