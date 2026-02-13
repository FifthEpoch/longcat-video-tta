#!/usr/bin/env python3
"""
Extract annotated ground-truth clips from original 480p videos.

Cuts the first (num_cond + num_gen) frames from each video, annotates
conditioning frames with a red border / "CONDITIONING" label and ground-truth
frames with a blue border / "GROUND TRUTH" label.

Usage:
  python extract_gt_videos.py \
      --data-dir datasets/panda_100_480p \
      --out-dir baseline_experiment/results/gt_clips_panda \
      --num-cond 28 --num-gen 14

  python extract_gt_videos.py \
      --data-dir datasets/ucf101_100_480p \
      --out-dir baseline_experiment/results/gt_clips_ucf101 \
      --num-cond 28 --num-gen 14
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Annotation (mirrors run_baseline.py)
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
# Video I/O
# ---------------------------------------------------------------------------

def load_frames(video_path: str, num_frames: int, target_fps: float = 15.0) -> list[np.ndarray]:
    """Load up to num_frames from a video, subsampled to target_fps. Returns uint8 (H,W,3)."""
    import av
    container = av.open(video_path)
    stream = container.streams.video[0]
    original_fps = float(stream.average_rate or stream.guessed_rate or 30)
    stride = max(1, round(original_fps / target_fps))

    all_frames = []
    for frame in container.decode(video=0):
        all_frames.append(frame)
        if len(all_frames) >= num_frames * stride + stride:
            break
    container.close()

    frames = all_frames[::stride][:num_frames]
    pil_frames = [f.to_ndarray(format="rgb24") for f in frames]

    # Pad if short
    while len(pil_frames) < num_frames:
        pil_frames.append(pil_frames[-1])

    return pil_frames[:num_frames]


def save_video(frames: list[np.ndarray], path: str, fps: int = 15):
    from torchvision.io import write_video
    tensor = torch.from_numpy(np.stack(frames))
    write_video(path, tensor, fps=fps, video_codec="libx264", options={"crf": "18"})


def slugify(text: str, max_len: int = 50) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    text = re.sub(r"[\s]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text[:max_len]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, required=True,
                   help="Path to 480p dataset (contains videos/ and metadata.csv)")
    p.add_argument("--out-dir", type=str, required=True,
                   help="Where to save annotated GT clips")
    p.add_argument("--num-cond", type=int, default=28,
                   help="Number of conditioning frames to show")
    p.add_argument("--num-gen", type=int, default=14,
                   help="Number of ground-truth (generated) frames to show")
    p.add_argument("--max-videos", type=int, default=100)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total_frames = args.num_cond + args.num_gen

    # Load video list from metadata.csv
    meta_csv = data_dir / "metadata.csv"
    videos_dir = data_dir / "videos"
    video_list = []

    if meta_csv.exists():
        with open(meta_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                fname = row.get("filename", "")
                vp = videos_dir / fname
                if vp.exists():
                    caption = row.get("caption", "")
                    if caption.startswith("[") and "'" in caption:
                        try:
                            caption = eval(caption)[0]
                        except Exception:
                            pass
                    video_list.append({"path": str(vp), "caption": caption, "filename": fname})
    else:
        for vp in sorted(videos_dir.glob("*.mp4")):
            video_list.append({"path": str(vp), "caption": "", "filename": vp.name})

    video_list = video_list[:args.max_videos]
    print(f"Processing {len(video_list)} videos → {total_frames} frames each "
          f"({args.num_cond} cond + {args.num_gen} GT)", flush=True)

    ok = fail = 0
    for vi, entry in enumerate(video_list):
        try:
            frames = load_frames(entry["path"], total_frames)

            annotated = []
            for fi, f in enumerate(frames):
                if fi < args.num_cond:
                    annotated.append(annotate_frame(
                        f, "CONDITIONING",
                        border_color=(255, 0, 0), border_width=3))
                else:
                    annotated.append(annotate_frame(
                        f, "GROUND TRUTH",
                        border_color=(0, 120, 255), border_width=3))

            slug = slugify(entry["caption"]) if entry["caption"] else entry["filename"].replace(".mp4", "")
            out_name = f"{vi:03d}_{slug}.mp4"
            save_video(annotated, str(out_dir / out_name))
            ok += 1

            if (vi + 1) % 20 == 0:
                print(f"  [{vi+1}/{len(video_list)}] done", flush=True)

        except Exception as e:
            print(f"  ERROR [{vi}] {entry['filename']}: {e}", flush=True)
            fail += 1

    print(f"\nComplete: {ok} ok, {fail} failed", flush=True)
    print(f"Output: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
