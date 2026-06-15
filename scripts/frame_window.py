#!/usr/bin/env python3
"""Shared frame-window geometry for LongCat TTA runs and gating extractors.

Single source of truth for which pixel frames TTA sees vs which frames offline
gating / hypothesis modules should evaluate.  Defaults match
``panda_1000v_standard`` from ``sweep_experiment/sbatch/submit_standard_1000v_chunked.sh``:

    gen_start_frame=48, tta_total_frames=48, tta_context_frames=14,
    num_cond_frames=14, num_frames=28  -> 14 generated frames at [48:62).

TTA runners load pixel frames ``[tta_start : gen_start_frame)`` where
``tta_start = max(0, gen_start_frame - tta_total_frames)``.  For the
standard config that is ``[0:48)``.

Runtime CLIP gate samples *within* that same window (typically 4 evenly spaced
frames); it does not read the generation-target region ``[48:62)``.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple

# panda_1000v_standard — sourced from submit_standard_1000v_chunked.sh env vars.
GEN_START_FRAME: int = 48
TTA_TOTAL_FRAMES: int = 48
TTA_CONTEXT_FRAMES: int = 14
NUM_FRAMES: int = 28
NUM_COND_FRAMES: int = 14
VAE_TEMPORAL_SCALE: int = 4


def tta_start_frame(gen_start_frame: int, tta_total_frames: int) -> int:
    """First pixel frame index the TTA loop loads (matches runner clamp)."""
    return max(0, int(gen_start_frame) - int(tta_total_frames))


def tta_visible_range(
    gen_start_frame: int,
    tta_total_frames: int,
) -> Tuple[int, int]:
    """Python-slice ``(start, end)`` of pre-anchor frames TTA can see."""
    start = tta_start_frame(gen_start_frame, tta_total_frames)
    return start, int(gen_start_frame)


def num_generated_frames(num_frames: int, num_cond_frames: int) -> int:
    """Count of genuinely new frames the diffusion sampler emits."""
    return int(num_frames) - int(num_cond_frames)


def gen_target_range(
    gen_start_frame: int,
    num_frames: int,
    num_cond_frames: int,
) -> Tuple[int, int]:
    """Python-slice ``(start, end)`` of GT generation-target frames (Tier 3)."""
    n_gen = num_generated_frames(num_frames, num_cond_frames)
    start = int(gen_start_frame)
    return start, start + n_gen


def num_context_latents(
    tta_context_frames: int,
    vae_temporal_scale: int = VAE_TEMPORAL_SCALE,
) -> int:
    """Clean conditioning latent count (VAE temporal scale = 4)."""
    return 1 + (max(1, int(tta_context_frames)) - 1) // int(vae_temporal_scale)


@dataclass(frozen=True)
class FrameWindowConfig:
    """Immutable frame geometry for one sweep / dataset recipe."""

    gen_start_frame: int = GEN_START_FRAME
    tta_total_frames: int = TTA_TOTAL_FRAMES
    tta_context_frames: int = TTA_CONTEXT_FRAMES
    num_frames: int = NUM_FRAMES
    num_cond_frames: int = NUM_COND_FRAMES
    vae_temporal_scale: int = VAE_TEMPORAL_SCALE

    def tta_start_frame(self) -> int:
        return tta_start_frame(self.gen_start_frame, self.tta_total_frames)

    def tta_visible_range(self) -> Tuple[int, int]:
        return tta_visible_range(self.gen_start_frame, self.tta_total_frames)

    def gen_target_range(self) -> Tuple[int, int]:
        return gen_target_range(
            self.gen_start_frame, self.num_frames, self.num_cond_frames,
        )

    def num_generated_frames(self) -> int:
        return num_generated_frames(self.num_frames, self.num_cond_frames)

    def num_context_latents(self) -> int:
        return num_context_latents(
            self.tta_context_frames, self.vae_temporal_scale,
        )


PANDA_1000V_STANDARD = FrameWindowConfig()

# Backward-compatible aliases used by existing gating scripts / docs.
AUTO_TTA_VISIBLE_RANGE: Tuple[int, int] = PANDA_1000V_STANDARD.tta_visible_range()
AUTO_GEN_TARGET_RANGE: Tuple[int, int] = PANDA_1000V_STANDARD.gen_target_range()
AUTO_GEN_TARGET_FRAMES: int = PANDA_1000V_STANDARD.num_generated_frames()


def parse_frame_range_arg(
    arg: str,
    default: Tuple[int, int],
) -> Tuple[int, int]:
    """Parse CLI ``'auto'`` or ``'start:end'`` into ``(start, end)``."""
    if not arg or str(arg).lower() == "auto":
        return default
    if ":" in str(arg):
        a, b = str(arg).split(":", 1)
        return int(a), int(b)
    raise argparse.ArgumentTypeError(
        f"frame-range arg must be 'auto' or 'start:end', got {arg!r}"
    )


def resolve_visible_range(
    arg: str,
    cfg: FrameWindowConfig = PANDA_1000V_STANDARD,
) -> Tuple[int, int]:
    """Resolve ``--tta-visible-frames`` against a config preset."""
    return parse_frame_range_arg(arg, cfg.tta_visible_range())


def format_frame_range(r: Tuple[int, int]) -> str:
    return f"{int(r[0])}:{int(r[1])}"


def sample_clip_frame_offsets(
    window_len: int,
    sample_frames: int,
    sampling_mode: str = "full_window",
    late_fraction: float = 0.4,
) -> List[int]:
    """Pick frame offsets inside a TTA-visible window for CLIP / X-CLIP scoring."""
    import numpy as np

    if window_len <= 0:
        return []

    if sampling_mode == "late_only":
        frac = min(max(float(late_fraction), 1e-6), 1.0)
        late_len = max(1, int(round(window_len * frac)))
        candidate_start = max(0, window_len - late_len)
        candidates = list(range(candidate_start, window_len))
    else:
        candidates = list(range(window_len))

    if not candidates:
        return []

    k = max(1, min(int(sample_frames), len(candidates)))
    if k == 1:
        return [candidates[-1]]

    pos = np.linspace(0, len(candidates) - 1, num=k, dtype=int)
    return [candidates[int(i)] for i in pos]


def estimate_clip_candidate_frames(
    tta_total_frames: int,
    sampling_mode: str = "full_window",
    late_fraction: float = 0.4,
) -> int:
    """Upper bound on distinct frame indices CLIP gate can sample."""
    window_len = max(1, int(tta_total_frames))
    if sampling_mode == "late_only":
        frac = min(max(float(late_fraction), 1e-6), 1.0)
        return max(1, int(round(window_len * frac)))
    return window_len
