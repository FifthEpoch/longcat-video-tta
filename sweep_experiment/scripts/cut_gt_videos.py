#!/usr/bin/env python3
"""
Cut ground-truth source videos to match the temporal window of generated videos.

For each video in a run's summary.json, extracts frames
[gen_start_frame .. gen_start_frame + num_frames - 1] from the original
source video and saves the clipped version with GT naming convention:

  <idx>_<caption-with-dashes>_GT.mp4

Usage:
  python cut_gt_videos.py \
      --summary /path/to/results/sanity_100/panda_adasteer_ablation/AS_BARE/summary.json \
      --source-dir /path/to/datasets/panda_1000_480p/videos \
      --out-dir /path/to/results/sanity_100/gt_videos \
      --num-cond-frames 14 --num-frames 28 --gen-start-frame 48

  # Defaults match the standard LongCat config (14 cond, 28 total, start=48).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


def sanitize_caption(caption: str, max_len: int = 80) -> str:
    s = caption.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s)
    s = s.strip("-")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-")
    return s


def extract_index(video_name: str) -> str:
    m = re.search(r"(\d+)$", video_name)
    if m:
        return m.group(1).lstrip("0") or "0"
    return video_name


def read_video_frames(path: str, start: int, count: int) -> list[np.ndarray]:
    """Read `count` frames starting at frame index `start`."""
    import av
    container = av.open(path)
    stream = container.streams.video[0]
    frames = []
    for i, frame in enumerate(container.decode(stream)):
        if i < start:
            continue
        if i >= start + count:
            break
        frames.append(frame.to_ndarray(format="rgb24"))
    container.close()
    return frames


def save_video(frames: list[np.ndarray], path: str, fps: int = 24):
    import torch
    from torchvision.io import write_video
    tensor = torch.from_numpy(np.stack(frames))
    write_video(path, tensor, fps=fps, video_codec="libx264",
                options={"crf": "18"})


def main():
    p = argparse.ArgumentParser(
        description="Cut GT source videos to match generated video temporal window.")
    p.add_argument("--summary", type=str, required=True,
                   help="Path to a run's summary.json (used to get video list + captions)")
    p.add_argument("--source-dir", type=str, required=True,
                   help="Directory containing original source .mp4 files")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Output directory for cut GT videos")
    p.add_argument("--num-cond-frames", type=int, default=14)
    p.add_argument("--num-frames", type=int, default=28,
                   help="Total frames in generated video (cond + gen)")
    p.add_argument("--gen-start-frame", type=int, default=48,
                   help="Pixel frame index where conditioning starts in the source")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--annotate", action="store_true",
                   help="Add CONDITIONING/GT CONTINUATION labels to frames")
    args = p.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source_dir = Path(args.source_dir)

    results = summary.get("results", [])
    n_cond = args.num_cond_frames
    n_total = args.num_frames
    start = args.gen_start_frame

    print(f"Cutting GT videos: frames [{start}..{start + n_total - 1}] "
          f"({n_cond} cond + {n_total - n_cond} continuation)")
    print(f"Source: {source_dir}")
    print(f"Output: {out_dir}\n")

    count = 0
    for r in results:
        if not r.get("success", False):
            continue
        vname = r.get("video_name", "")
        caption = r.get("caption", "unknown")
        video_path = r.get("video_path", "")

        source_path = Path(video_path)
        if not source_path.exists():
            stem = source_path.stem
            candidates = list(source_dir.glob(f"{stem}.*"))
            if candidates:
                source_path = candidates[0]
            else:
                alt = source_dir / f"{vname}.mp4"
                if alt.exists():
                    source_path = alt
                else:
                    print(f"  SKIP {vname}: source not found")
                    continue

        try:
            frames = read_video_frames(str(source_path), start, n_total)
        except Exception as e:
            print(f"  SKIP {vname}: {e}")
            continue

        if len(frames) < n_total:
            print(f"  SKIP {vname}: only {len(frames)} frames "
                  f"(need {n_total} from frame {start})")
            continue

        if args.annotate:
            from PIL import Image, ImageDraw, ImageFont
            def _get_font(size):
                for fp in (
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                    "/System/Library/Fonts/Helvetica.ttc",
                ):
                    try:
                        return ImageFont.truetype(fp, size)
                    except (IOError, OSError):
                        continue
                return ImageFont.load_default()

            for i in range(len(frames)):
                img = Image.fromarray(frames[i])
                draw = ImageDraw.Draw(img)
                h, w = frames[i].shape[:2]
                bw = 4
                if i < n_cond:
                    label, color = "CONDITIONING", (0, 200, 0)
                else:
                    label, color = "GT CONTINUATION", (0, 120, 255)
                for j in range(bw):
                    draw.rectangle([j, j, w-1-j, h-1-j], outline=color)
                font = _get_font(max(14, h // 22))
                margin = bw + 5
                bbox = draw.textbbox((margin, margin), label, font=font)
                pad = 4
                draw.rectangle(
                    [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
                    fill=(0, 0, 0))
                draw.text((margin, margin), label, fill="white", font=font)
                frames[i] = np.array(img)

        idx = extract_index(vname)
        slug = sanitize_caption(caption)
        out_name = f"{idx}_{slug}_GT.mp4"
        out_path = out_dir / out_name

        save_video(frames, str(out_path), fps=args.fps)
        count += 1
        if count % 10 == 0:
            print(f"  [{count}] {out_name}")

    print(f"\nDone. Saved {count} GT videos to {out_dir}")


if __name__ == "__main__":
    main()
