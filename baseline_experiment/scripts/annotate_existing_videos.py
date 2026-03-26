#!/usr/bin/env python3
"""
Annotate existing generated videos with CONDITIONING / GENERATED labels.

For each generated video in --gen-dir:
  - Extracts the video index from the filename (e.g. 032_... -> panda_0032.mp4)
  - Loads the first `num_cond_frames` frames from the matching original 480p video
  - Annotates conditioning frames with green border + "CONDITIONING" label
  - Annotates generated frames with "GENERATED" label
  - Saves the combined annotated video to --out-dir

Usage (local):
  python annotate_existing_videos.py \
      --gen-dir ~/Downloads/generated_video/using_longcat/baselines \
      --orig-dir ~/Downloads/generated_video/using_longcat/originals_480p \
      --out-dir ~/Downloads/generated_video/using_longcat/annotated
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Annotation helpers (same logic as run_baseline.py)
# ---------------------------------------------------------------------------

def _get_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSMono.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def annotate_frame(
    frame_uint8: np.ndarray,
    label: str,
    border_color: tuple[int, int, int] | None = None,
    border_width: int = 3,
) -> np.ndarray:
    img = Image.fromarray(frame_uint8)
    draw = ImageDraw.Draw(img)
    h, w = frame_uint8.shape[:2]

    if border_color is not None:
        for i in range(border_width):
            draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=border_color)

    font_size = max(12, h // 25)
    font = _get_font(font_size)
    margin = (border_width + 4) if border_color else 6
    bbox = draw.textbbox((margin, margin), label, font=font)
    pad = 3
    draw.rectangle(
        [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
        fill=(0, 0, 0),
    )
    draw.text((margin, margin), label, fill="white", font=font)
    return np.array(img)


# ---------------------------------------------------------------------------
# Video I/O (using av)
# ---------------------------------------------------------------------------

def read_video_frames(path: str, max_frames: int = 999) -> list[np.ndarray]:
    """Read up to max_frames from a video, returning list of uint8 (H,W,3)."""
    import av
    container = av.open(path)
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= max_frames:
            break
    container.close()
    return frames


def save_video(frames: list[np.ndarray], path: str, fps: int = 15):
    """Save list of uint8 (H,W,3) frames as mp4."""
    import torch
    from torchvision.io import write_video
    tensor = torch.from_numpy(np.stack(frames))
    write_video(path, tensor, fps=fps, video_codec="libx264", options={"crf": "18"})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gen-dir", type=str, required=True,
                   help="Directory with generated .mp4 files")
    p.add_argument("--orig-dir", type=str, default=None,
                   help="Directory with original 480p .mp4 files (panda_XXXX.mp4). "
                        "If provided, conditioning frames are prepended from originals.")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory for annotated videos")
    p.add_argument("--num-cond-frames", type=int, default=2)
    args = p.parse_args()

    gen_dir = Path(args.gen_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_dir = Path(args.orig_dir) if args.orig_dir else None

    gen_videos = sorted(gen_dir.glob("*.mp4"))
    if not gen_videos:
        print(f"No .mp4 files found in {gen_dir}")
        sys.exit(1)

    print(f"Found {len(gen_videos)} generated videos")

    for vp in gen_videos:
        print(f"  Processing {vp.name} ...", end="", flush=True)

        # Extract index from filename like "032_some-slug_PSNR-...mp4"
        match = re.match(r"^(\d+)_", vp.name)
        idx = int(match.group(1)) if match else None

        # Load generated frames
        gen_frames = read_video_frames(str(vp))

        annotated = []

        # Try to load conditioning frames from original video
        if orig_dir is not None and idx is not None:
            orig_path = orig_dir / f"panda_{idx:04d}.mp4"
            if orig_path.exists():
                orig_frames = read_video_frames(str(orig_path), max_frames=args.num_cond_frames)
                # Resize to match generated resolution
                gen_h, gen_w = gen_frames[0].shape[:2]
                for f in orig_frames[:args.num_cond_frames]:
                    resized = np.array(
                        Image.fromarray(f).resize((gen_w, gen_h), Image.BICUBIC)
                    )
                    annotated.append(annotate_frame(
                        resized, "CONDITIONING",
                        border_color=(255, 0, 0), border_width=3))
            else:
                print(f" (no original at {orig_path})", end="")

        # Annotate generated frames
        for f in gen_frames:
            annotated.append(annotate_frame(
                f, "GENERATED",
                border_color=(0, 255, 0), border_width=3))

        out_path = out_dir / vp.name
        save_video(annotated, str(out_path))
        print(f" -> {out_path.name} ({len(annotated)} frames)")

    print(f"\nDone. Annotated videos in: {out_dir}")


if __name__ == "__main__":
    main()
