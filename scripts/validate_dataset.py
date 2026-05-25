#!/usr/bin/env python3
"""
Validate that every video in a dataset can be loaded end-to-end by the
TTA pipeline (using the same `av` decode path as `delta_experiment.common`).

This is a stricter check than the `validate_decodable=True` hook in
`delta_experiment/scripts/common.py`, which only decodes the first frame.
Many of our 1000-video runs in the past finished with N<1000 because a
handful of clips passed the first-frame check but failed mid-decode.

Checks performed (per video):

  - File exists and size >= MIN_BYTES (default 50 KB)
  - `av.open()` succeeds
  - At least MIN_FRAMES frames decode end-to-end (default 50, enough for
    gen_start_frame=48 + a few cond frames)
  - Reported FPS >= MIN_FPS (default 6, catches obviously broken clips)
  - Reported height >= MIN_HEIGHT (default 240, allows UCF101's 240p source)
  - Metadata.csv row exists, with a non-empty caption (Panda) or category (UCF)

Exit code is 0 only if exactly --required-valid videos pass. Otherwise the
script returns non-zero so a downstream shell pipeline can bail early.

Usage:

  # Validate the new Panda 2048 dataset, must have >= 2048 valid clips:
  python scripts/validate_dataset.py \
      --dataset-dir /scratch/wc3013/longcat-video-tta/datasets/panda_2048_480p \
      --required-valid 2048

  # Validate UCF101, quiet mode (only report failures):
  python scripts/validate_dataset.py \
      --dataset-dir datasets/ucf101_2048_480p \
      --required-valid 2048 --quiet

  # Validate but write a "valid_subset.csv" that downstream training can
  # use as a drop-in metadata replacement that contains only the working
  # clips, preserving the original ordering:
  python scripts/validate_dataset.py \
      --dataset-dir datasets/panda_2048_480p \
      --required-valid 2048 \
      --write-valid-subset valid_subset.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class VideoCheck:
    filename: str
    ok: bool
    reason: str
    frames_decoded: int = 0
    fps: float = 0.0
    height: int = 0
    width: int = 0
    duration: float = 0.0
    size_bytes: int = 0
    caption: str = ""
    category: str = ""


def _check_one_video(
    video_path: Path,
    row: dict,
    min_bytes: int,
    min_frames: int,
    min_fps: float,
    min_height: int,
    require_caption: bool,
    require_category: bool,
    max_decode_frames: int,
) -> VideoCheck:
    name = video_path.name
    res = VideoCheck(filename=name, ok=False, reason="")

    if not video_path.exists():
        res.reason = "missing_file"
        return res
    res.size_bytes = video_path.stat().st_size
    if res.size_bytes < min_bytes:
        res.reason = f"too_small({res.size_bytes}B)"
        return res

    caption = (row.get("caption", "") or row.get("text", "") or "").strip()
    category = (row.get("category", "") or row.get("class_name", "") or "").strip()
    res.caption = caption
    res.category = category

    if require_caption and not caption:
        res.reason = "empty_caption"
        return res
    if require_category and not category:
        res.reason = "empty_category"
        return res

    try:
        import av
    except ImportError:
        res.reason = "av_not_installed"
        return res

    try:
        container = av.open(str(video_path))
    except Exception as exc:
        res.reason = f"av_open_failed:{type(exc).__name__}"
        return res

    try:
        stream = container.streams.video[0]
        # Stream-level info, used for fps / resolution checks.
        if stream.average_rate is not None:
            res.fps = float(stream.average_rate)
        elif stream.base_rate is not None:
            res.fps = float(stream.base_rate)
        res.height = int(stream.height or 0)
        res.width = int(stream.width or 0)

        if res.height < min_height:
            res.reason = f"height_too_small({res.height})"
            container.close()
            return res
        if res.fps and res.fps < min_fps:
            res.reason = f"fps_too_low({res.fps:.2f})"
            container.close()
            return res

        frames_decoded = 0
        for _frame in container.decode(video=0):
            frames_decoded += 1
            if frames_decoded >= max_decode_frames:
                break
        res.frames_decoded = frames_decoded

        if stream.duration is not None and stream.time_base is not None:
            try:
                res.duration = float(stream.duration * stream.time_base)
            except Exception:
                pass
    except Exception as exc:
        res.reason = f"decode_failed:{type(exc).__name__}"
        try:
            container.close()
        except Exception:
            pass
        return res

    try:
        container.close()
    except Exception:
        pass

    if res.frames_decoded < min_frames:
        res.reason = f"too_few_frames({res.frames_decoded})"
        return res

    res.ok = True
    res.reason = "ok"
    return res


def _load_metadata_rows(meta_path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v or "").strip() for k, v in row.items()})
    return rows


def _resolve_video_path(data_dir: Path, row: dict) -> Path:
    fname = row.get("filename", "") or row.get("video_path", "")
    if not fname:
        return data_dir / "_missing_filename"
    vp = data_dir / "videos" / fname
    if vp.exists():
        return vp
    return data_dir / fname


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate every video in a dataset is loadable end-to-end",
    )
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Path to dataset root (contains metadata.csv and "
                             "videos/ subdirectory)")
    parser.add_argument("--required-valid", type=int, default=None,
                        help="Required number of valid videos. If fewer are "
                             "found, exit non-zero. If not set, only reports.")
    parser.add_argument("--min-bytes", type=int, default=50_000,
                        help="Minimum file size in bytes (default 50 KB)")
    parser.add_argument("--min-frames", type=int, default=50,
                        help="Minimum frames that must decode end-to-end "
                             "(default 50; gen_start_frame=48 + buffer)")
    parser.add_argument("--min-fps", type=float, default=6.0,
                        help="Minimum FPS to accept (default 6.0)")
    parser.add_argument("--min-height", type=int, default=240,
                        help="Minimum frame height (default 240, allows UCF "
                             "240p source)")
    parser.add_argument("--no-require-caption", action="store_true",
                        help="Skip the non-empty caption check")
    parser.add_argument("--no-require-category", action="store_true",
                        help="Skip the non-empty category check (Panda has no "
                             "category in some manifests)")
    parser.add_argument("--max-decode-frames", type=int, default=80,
                        help="Stop decoding after this many frames per video "
                             "(default 80, just past gen_start_frame+windows)")
    parser.add_argument("--write-report", type=str, default=None,
                        help="Write a JSON report listing per-video status to "
                             "this path (default: <dataset-dir>/validation_report.json)")
    parser.add_argument("--write-valid-subset", type=str, default=None,
                        help="Write a metadata.csv containing only valid videos "
                             "(relative path or absolute). Useful as a drop-in "
                             "replacement when the original has extras.")
    parser.add_argument("--quiet", action="store_true",
                        help="Only print failures and the final summary")
    parser.add_argument("--limit", type=int, default=None,
                        help="Validate only the first N rows (smoke test)")
    args = parser.parse_args()

    data_dir = Path(args.dataset_dir).resolve()
    meta_path = data_dir / "metadata.csv"
    if not meta_path.exists():
        print(f"ERROR: no metadata.csv in {data_dir}", file=sys.stderr)
        return 2

    rows = _load_metadata_rows(meta_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    print(f"Validating {len(rows)} entries in {data_dir}")
    print(f"  min_bytes={args.min_bytes}  min_frames={args.min_frames} "
          f"min_fps={args.min_fps}  min_height={args.min_height}")
    print()

    checks: List[VideoCheck] = []
    start_time = time.time()
    reason_counts: dict = {}
    for i, row in enumerate(rows):
        vp = _resolve_video_path(data_dir, row)
        res = _check_one_video(
            video_path=vp,
            row=row,
            min_bytes=args.min_bytes,
            min_frames=args.min_frames,
            min_fps=args.min_fps,
            min_height=args.min_height,
            require_caption=not args.no_require_caption,
            require_category=not args.no_require_category,
            max_decode_frames=args.max_decode_frames,
        )
        checks.append(res)

        reason_counts[res.reason] = reason_counts.get(res.reason, 0) + 1

        if not res.ok and not args.quiet:
            print(f"  [{i:4d}] FAIL  {res.filename}  {res.reason}")
        elif not args.quiet and (i % 100 == 0):
            elapsed = time.time() - start_time
            rate = (i + 1) / max(elapsed, 0.01)
            print(f"  [{i:4d}] ok={sum(c.ok for c in checks)}  "
                  f"rate={rate:.1f}/s")

    elapsed = time.time() - start_time
    n_total = len(checks)
    n_ok = sum(1 for c in checks if c.ok)
    n_bad = n_total - n_ok

    print()
    print("=" * 70)
    print(f"Validation summary  ({elapsed:.1f}s)")
    print("=" * 70)
    print(f"  total      : {n_total}")
    print(f"  valid      : {n_ok}")
    print(f"  invalid    : {n_bad}")
    print()
    print("  reason breakdown:")
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"    {reason:30s} {count}")

    report_path = (
        Path(args.write_report) if args.write_report
        else data_dir / "validation_report.json"
    )
    with open(report_path, "w") as f:
        json.dump(
            {
                "dataset_dir": str(data_dir),
                "total": n_total,
                "valid": n_ok,
                "invalid": n_bad,
                "elapsed_seconds": elapsed,
                "params": {
                    "min_bytes": args.min_bytes,
                    "min_frames": args.min_frames,
                    "min_fps": args.min_fps,
                    "min_height": args.min_height,
                    "max_decode_frames": args.max_decode_frames,
                    "require_caption": not args.no_require_caption,
                    "require_category": not args.no_require_category,
                },
                "per_video": [
                    {
                        "filename": c.filename,
                        "ok": c.ok,
                        "reason": c.reason,
                        "frames_decoded": c.frames_decoded,
                        "fps": c.fps,
                        "height": c.height,
                        "width": c.width,
                        "size_bytes": c.size_bytes,
                    }
                    for c in checks
                ],
                "reason_counts": reason_counts,
            },
            f,
            indent=2,
        )
    print()
    print(f"  report     : {report_path}")

    if args.write_valid_subset:
        sub_path = Path(args.write_valid_subset)
        if not sub_path.is_absolute():
            sub_path = data_dir / sub_path
        valid_filenames = {c.filename for c in checks if c.ok}
        with open(meta_path, "r", encoding="utf-8") as f_in:
            reader = csv.DictReader(f_in)
            fieldnames = reader.fieldnames
            kept = [r for r in reader if r.get("filename") in valid_filenames]
        with open(sub_path, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
        print(f"  valid-only : {sub_path}  ({len(kept)} rows)")

    if args.required_valid is not None:
        if n_ok < args.required_valid:
            print()
            print(f"FAIL: need {args.required_valid} valid videos, got {n_ok}")
            return 1
        else:
            print()
            print(f"PASS: have {n_ok} >= {args.required_valid} valid videos")
            return 0

    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
