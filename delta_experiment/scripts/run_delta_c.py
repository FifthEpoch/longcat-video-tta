#!/usr/bin/env python3
"""
Delta-C TTA: Output correction residual.

Adds a learnable constant residual δ_out to the denoiser's output prediction:
    v_pred' = v_pred + δ_out

δ_out has shape [C_out, 1, 1, 1] (per-channel bias) or the full output shape
depending on configuration. We default to per-channel to keep it small.

Usage:
    python run_delta_c.py \\
        --checkpoint-dir /path/to/longcat-video-checkpoints \\
        --data-dir /path/to/dataset \\
        --output-dir results/delta_c \\
        --delta-steps 20 --delta-lr 1e-3
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    load_longcat_components,
    load_video_frames,
    encode_video,
    encode_prompt,
    compute_flow_matching_loss,
    generate_video_continuation,
    save_results,
    save_video_from_numpy,
    load_checkpoint,
    save_checkpoint,
    torch_gc,
)
from early_stopping import (
    AnchoredEarlyStopper,
    add_early_stopping_args,
    build_early_stopper_from_args,
)


# ============================================================================
# Delta-C wrapper: output correction
# ============================================================================

class DeltaCWrapper(nn.Module):
    """Wraps LongCatVideoTransformer3DModel to add a learnable residual
    to the model's output prediction.

    Modes:
        - "per_channel": δ_out ∈ R^{out_channels} broadcast over T, H, W
        - "full": δ_out has the full output shape (more parameters)
    """

    def __init__(
        self,
        dit: nn.Module,
        mode: str = "per_channel",
        out_channels: int = 16,
    ):
        super().__init__()
        self.dit = dit
        self.mode = mode

        # Freeze all DiT parameters
        for p in self.dit.parameters():
            p.requires_grad = False

        if mode == "per_channel":
            # Shape [C_out] - broadcast over B, T, H, W
            self.delta_out = nn.Parameter(torch.zeros(out_channels))
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'per_channel'.")

        # Generation hook (installed/removed around pipeline calls)
        self._gen_hook = None

    @property
    def config(self):
        """Proxy config to the inner DiT."""
        return self.dit.config

    # ------------------------------------------------------------------
    # Hook-based injection for pipeline generation
    # ------------------------------------------------------------------
    def apply_to_dit(self):
        """Install a forward hook on the DiT to add output correction
        during pipeline generation (preserves KV-cache, BSA, etc.)."""
        delta_out = self.delta_out

        def _hook(_module, _input, output):
            correction = delta_out.view(1, -1, 1, 1, 1).to(
                output.dtype if not isinstance(output, tuple) else output[0].dtype
            )
            if isinstance(output, tuple):
                return (output[0] + correction,) + output[1:]
            return output + correction

        self._gen_hook = self.dit.register_forward_hook(_hook)

    def remove_from_dit(self):
        """Remove the generation hook."""
        if self._gen_hook is not None:
            self._gen_hook.remove()
            self._gen_hook = None

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        **kwargs,
    ):
        """Forward with output correction."""
        # Standard DiT forward
        pred = self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            num_cond_latents=num_cond_latents,
        )

        # Add output correction
        if self.mode == "per_channel":
            # pred shape: [B, C_out, T, H, W]
            delta = self.delta_out.view(1, -1, 1, 1, 1)
            pred = pred + delta

        return pred


# ============================================================================
# Optimization loop
# ============================================================================

def optimize_delta_c(
    wrapper: DeltaCWrapper,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    num_steps: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    early_stopper: Optional[AnchoredEarlyStopper] = None,
) -> Dict:
    """Optimize the output correction delta."""
    optimizer = AdamW([wrapper.delta_out], lr=lr, betas=(0.9, 0.999), eps=1e-15)

    wrapper.train()
    losses = []

    for step in range(num_steps):
        optimizer.zero_grad()

        loss = compute_flow_matching_loss(
            dit=wrapper,
            latents=latents,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
            device=device,
            dtype=dtype,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_([wrapper.delta_out], 1.0)
        optimizer.step()

        losses.append(loss.item())

        if early_stopper is not None:
            should_stop, es_info = early_stopper.step(step + 1)
            if should_stop:
                print(f"  Early stopping at step {step + 1}: {es_info}")
                break

    es_state = None
    if early_stopper is not None:
        early_stopper.restore(
            restore_fn=lambda sd: wrapper.load_state_dict(sd, strict=False)
        )
        es_state = early_stopper.state

    return {
        "losses": losses,
        "delta_out_norm": wrapper.delta_out.detach().norm().item(),
        "delta_out_values": wrapper.delta_out.detach().cpu().tolist(),
        "early_stopping_info": es_state,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Delta-C TTA for LongCat-Video")
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    parser.add_argument("--delta-steps", type=int, default=20)
    parser.add_argument("--delta-lr", type=float, default=1e-3)
    parser.add_argument("--delta-mode", type=str, default="per_channel",
                        choices=["per_channel"])
    parser.add_argument("--num-cond-frames", type=int, default=13)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip video generation (only train delta)")
    add_early_stopping_args(parser)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    ckpt_path = os.path.join(args.output_dir, "checkpoint.json")
    ckpt = load_checkpoint(ckpt_path)
    start_idx = ckpt.get("next_idx", 0) if ckpt else 0

    print("=" * 70)
    print("Delta-C TTA for LongCat-Video")
    print("=" * 70)
    print(f"Checkpoint dir : {args.checkpoint_dir}")
    print(f"Mode           : {args.delta_mode}")
    print(f"Delta steps    : {args.delta_steps}")
    print(f"Delta LR       : {args.delta_lr}")
    print(f"Resume from idx: {start_idx}")
    print("=" * 70)

    components = load_longcat_components(
        args.checkpoint_dir, device=args.device, dtype=torch.bfloat16
    )
    dit = components["dit"]
    vae = components["vae"]
    pipe = components["pipe"]
    tokenizer = components["tokenizer"]
    text_encoder = components["text_encoder"]

    from common import load_ucf101_video_list
    videos = load_ucf101_video_list(
        args.data_dir, max_videos=args.max_videos, seed=args.seed
    )
    print(f"\nTotal videos: {len(videos)}")

    early_stopper = build_early_stopper_from_args(args)
    all_results = []
    videos_dir = os.path.join(args.output_dir, "videos")
    if not args.skip_generation:
        os.makedirs(videos_dir, exist_ok=True)

    for idx, entry in enumerate(videos):
        if idx < start_idx:
            continue

        video_path = entry["video_path"]
        caption = entry["caption"]
        video_name = Path(video_path).stem

        print(f"\n[{idx + 1}/{len(videos)}] {video_name}: {caption}")

        try:
            pixel_frames = load_video_frames(
                video_path, args.num_cond_frames, height=480, width=832
            ).to(args.device, torch.bfloat16)

            latents = encode_video(vae, pixel_frames, normalize=True)

            prompt_embeds, prompt_mask = encode_prompt(
                tokenizer, text_encoder, caption,
                device=args.device, dtype=torch.bfloat16,
            )

            wrapper = DeltaCWrapper(
                dit, mode=args.delta_mode,
                out_channels=dit.config.out_channels,
            ).to(args.device)

            if early_stopper is not None:
                early_stopper.setup(
                    model=wrapper,
                    latents=latents,
                    prompt_embeds=prompt_embeds,
                    prompt_mask=prompt_mask,
                    device=args.device,
                    dtype=torch.bfloat16,
                    forward_fn=lambda nl, ts: wrapper(
                        hidden_states=nl, timestep=ts,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_mask,
                    ),
                )

            t0 = time.time()
            opt_result = optimize_delta_c(
                wrapper=wrapper,
                latents=latents,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                num_steps=args.delta_steps,
                lr=args.delta_lr,
                device=args.device,
                dtype=torch.bfloat16,
                early_stopper=early_stopper,
            )
            train_time = time.time() - t0

            result = {
                "video_name": video_name,
                "video_path": video_path,
                "caption": caption,
                "train_time": train_time,
                "final_loss": opt_result["losses"][-1] if opt_result["losses"] else None,
                "delta_out_norm": opt_result["delta_out_norm"],
                "early_stopping_info": opt_result.get("early_stopping_info"),
                "success": True,
            }

            print(f"  Train time: {train_time:.1f}s, "
                  f"Final loss: {result['final_loss']:.4f}, "
                  f"δ_out norm: {result['delta_out_norm']:.6f}")

            # ── Generation ──────────────────────────────────────────
            gen_time = 0.0
            if not args.skip_generation:
                from PIL import Image

                pf = pixel_frames.squeeze(0)  # [C, T, H, W]
                pf = ((pf + 1.0) / 2.0).clamp(0, 1)
                cond_images = []
                for t_idx in range(pf.shape[1]):
                    frame_np = (pf[:, t_idx].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    cond_images.append(Image.fromarray(frame_np))

                wrapper.apply_to_dit()
                try:
                    gen_start = time.time()
                    gen_frames = generate_video_continuation(
                        pipe=pipe,
                        video_frames=cond_images,
                        prompt=caption,
                        num_cond_frames=args.num_cond_frames,
                        num_frames=args.num_frames,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        seed=args.seed + idx,
                        resolution=args.resolution,
                        device=args.device,
                    )
                    gen_time = time.time() - gen_start

                    output_path = os.path.join(videos_dir, f"{video_name}_delta_c.mp4")
                    save_video_from_numpy(gen_frames, output_path, fps=24)
                    result["output_path"] = output_path
                    result["gen_time"] = gen_time
                    print(f"  Gen: {gen_time:.1f}s → {output_path}")
                finally:
                    wrapper.remove_from_dit()

            result["total_time"] = train_time + gen_time
            all_results.append(result)

            del wrapper, latents, pixel_frames, prompt_embeds, prompt_mask
            torch_gc()

        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            all_results.append({
                "video_name": video_name,
                "video_path": video_path,
                "error": str(e),
                "success": False,
            })

        save_checkpoint({"next_idx": idx + 1, "results": all_results}, ckpt_path)

    successful = [r for r in all_results if r.get("success", False)]
    summary = {
        "method": "delta_c",
        "delta_mode": args.delta_mode,
        "delta_steps": args.delta_steps,
        "delta_lr": args.delta_lr,
        "num_videos": len(all_results),
        "num_successful": len(successful),
        "avg_train_time": np.mean([r.get("train_time", 0) for r in successful]) if successful else 0,
        "results": all_results,
    }
    save_results(summary, os.path.join(args.output_dir, "summary.json"))
    print(f"\nResults saved to {args.output_dir}/summary.json")
    if successful:
        print(f"Avg train time: {summary['avg_train_time']:.1f}s")


if __name__ == "__main__":
    main()
