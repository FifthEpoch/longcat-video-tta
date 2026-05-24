#!/usr/bin/env python3
"""Build per-candidate filmstrips for cover-image visual curation.

For each video ID in --retain-json, locates the corresponding MP4 in each
method directory, samples --num-keyframes evenly-spaced frames, and
composes them into a single labeled PNG per candidate. Also writes an
HTML index ranking all candidates by dpsnr (largest gain first).

Filename matching is defensive: the script first attempts the
post-rename pattern ``<idx>_*<method>.mp4`` using the per_video_results
ordering in each method's summary.json, then falls back to globbing for
the original ``<video_name>*.mp4`` pattern. This handles both
pre-rename and post-rename directory states.

Dependencies: numpy, opencv-python, Pillow (all already installed by the
env_setup script via torchvision / matplotlib). No new packages required.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# --- Layout constants (override via CLI flags) ---
DEFAULT_FRAME_W = 240
DEFAULT_FRAME_H = 135
DEFAULT_NUM_KEYFRAMES = 6
LABEL_COL_W = 110           # left-side label column width
HEADER_BAND_H = 56          # top header band (video id + metrics + caption)
ROW_PAD = 4
COL_PAD = 4

# Method colours for label backgrounds (BGR)
METHOD_COLOURS = {
    "GT":        (200, 200, 200),
    "No-TTA":    (180, 180, 220),
    "AdaSteer":  (110, 200, 110),
    "LoRA R8":   (180, 220, 180),
}


def load_retain(p: Path) -> List[str]:
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        return list(data.get("all", []))
    if isinstance(data, list):
        return list(data)
    raise SystemExit(f"Unrecognized retain.json structure in {p}")


def load_gains(p: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if p is None or not p.exists():
        return {}
    with p.open() as f:
        return {r["video"]: r for r in csv.DictReader(f)}


def load_summary_order(method_dir: Path) -> Dict[str, int]:
    """Return {video_name: idx} from method_dir/summary.json.

    Falls back to {} if no summary.json is present.
    """
    sj = method_dir.parent / "summary.json"
    if not sj.exists():
        return {}
    try:
        data = json.loads(sj.read_text())
    except Exception:
        return {}
    pv = data.get("per_video_results") or data.get("results") or []
    out: Dict[str, int] = {}
    for i, r in enumerate(pv):
        v = r.get("video_name") or r.get("video_id")
        if v:
            out[v] = i
    return out


def find_mp4(videos_dir: Path, video_name: str,
             idx_by_name: Dict[str, int]) -> Optional[Path]:
    """Locate the MP4 for video_name in videos_dir. Defensive."""
    if not videos_dir.is_dir():
        return None
    # 1. Direct match: <video_name>.mp4 (e.g., GT case)
    direct = videos_dir / f"{video_name}.mp4"
    if direct.exists():
        return direct
    # 2. Original pre-rename pattern: <video_name>_<method>.mp4
    pre = list(videos_dir.glob(f"{video_name}*.mp4"))
    if len(pre) == 1:
        return pre[0]
    # 3. Post-rename pattern: <idx>_*.mp4
    idx = idx_by_name.get(video_name)
    if idx is not None:
        post = sorted(videos_dir.glob(f"{idx}_*.mp4"))
        if len(post) >= 1:
            # Prefer one that also matches a method suffix shape
            return post[0]
    # 4. Substring fallback
    sub = list(videos_dir.glob(f"*{video_name}*.mp4"))
    if len(sub) == 1:
        return sub[0]
    return None


def sample_frames(video_path: Path, num: int,
                  start: int = 0, count: Optional[int] = None,
                  out_w: int = DEFAULT_FRAME_W,
                  out_h: int = DEFAULT_FRAME_H) -> List[np.ndarray]:
    """Sample `num` evenly-spaced frames from `count` frames starting at `start`.

    Returns BGR uint8 arrays sized (out_h, out_w, 3). If the video has
    fewer than start+count frames, samples from what's available.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if count is None:
        count = total - start
    end = min(total, start + count)
    span = max(end - start, 1)
    if span <= 1:
        idxs = [start] * num
    else:
        # Evenly spaced including first and last
        idxs = [start + int(round(i * (span - 1) / (num - 1)))
                for i in range(num)]

    frames: List[np.ndarray] = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, bgr = cap.read()
        if not ok:
            blank = np.full((out_h, out_w, 3), 200, dtype=np.uint8)
            frames.append(blank)
            continue
        bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        frames.append(bgr)
    cap.release()
    return frames


_FONT_CACHE: Dict[int, "ImageFont.ImageFont"] = {}


def _get_font(size: int) -> "ImageFont.ImageFont":
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for name in (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "Arial.ttf",
        "Helvetica.ttf",
    ):
        try:
            font = ImageFont.truetype(name, size)
            _FONT_CACHE[size] = font
            return font
        except (OSError, IOError):
            continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def put_label(img: np.ndarray, text: str, *,
              org: Tuple[int, int], scale: float = 0.5,
              colour: Tuple[int, int, int] = (0, 0, 0),
              thickness: int = 1) -> None:
    """Draw ``text`` onto a BGR uint8 image in-place using PIL.

    Notes on the cv2 avoidance:
        On some cluster installs ``opencv-python==4.9.0`` is built against a
        different numpy ABI than what is loaded at runtime, so every cv2 call
        with a numpy array we allocated ourselves raises
        ``src is not a numpy array, neither a scalar``. The arrays cv2 itself
        creates (e.g. from ``VideoCapture.read``) still work, but anything we
        build via ``np.full`` / ``np.vstack`` does not. We therefore do BGR
        <-> RGB conversions with plain numpy indexing instead of
        ``cv2.cvtColor`` and rasterise text with PIL.
    """
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        return
    rgb = np.ascontiguousarray(img[:, :, ::-1])
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(max(10, int(round(scale * 28))))
    pil_colour = (int(colour[2]), int(colour[1]), int(colour[0]))
    # cv2.putText anchors at the text baseline; PIL anchors at the top-left.
    # Shift up by roughly the cap height so the visual placement matches.
    cap_h = max(6, int(round(scale * 22)))
    pos = (int(org[0]), int(org[1]) - cap_h)
    draw.text(pos, text, fill=pil_colour, font=font)
    new_rgb = np.array(pil_img)
    img[:, :, :] = new_rgb[:, :, ::-1]


def compose_row(label: str, frames: List[np.ndarray],
                frame_w: int, frame_h: int,
                label_col_w: int) -> np.ndarray:
    n = len(frames)
    row_w = label_col_w + n * frame_w + (n - 1) * COL_PAD
    row_h = frame_h
    row = np.full((row_h, row_w, 3), 250, dtype=np.uint8)

    # Label column background
    colour = METHOD_COLOURS.get(label, (220, 220, 220))
    row[:, :label_col_w] = colour
    put_label(row, label, org=(8, frame_h // 2 + 5), scale=0.55,
              colour=(40, 40, 40), thickness=1)

    # Frames
    x = label_col_w
    for f in frames:
        row[:, x:x + frame_w] = f
        x += frame_w + COL_PAD
    return row


def compose_header(video_id: str, theme: str, dpsnr: Optional[float],
                   dssim: Optional[float], dlpips: Optional[float],
                   caption: str, width: int, height: int) -> np.ndarray:
    header = np.full((height, width, 3), 245, dtype=np.uint8)
    title = f"{video_id}    theme: {theme}"
    if dpsnr is not None:
        title += f"    dPSNR {dpsnr:+.2f}"
    if dssim is not None:
        title += f"    dSSIM {dssim:+.3f}"
    if dlpips is not None:
        title += f"    dLPIPS {dlpips:+.3f}"
    put_label(header, title, org=(10, 22), scale=0.6,
              colour=(30, 30, 30), thickness=1)
    snippet = caption[:140].replace("\n", " ")
    put_label(header, snippet, org=(10, 46), scale=0.45,
              colour=(80, 80, 80), thickness=1)
    return header


def compose_filmstrip(rows: List[np.ndarray], header: np.ndarray) -> np.ndarray:
    width = max(r.shape[1] for r in rows + [header])
    pieces: List[np.ndarray] = []

    def _pad(img: np.ndarray) -> np.ndarray:
        if img.shape[1] == width:
            return img
        out = np.full((img.shape[0], width, 3), 250, dtype=np.uint8)
        out[:, :img.shape[1]] = img
        return out

    pieces.append(_pad(header))
    for i, row in enumerate(rows):
        pieces.append(_pad(row))
        if i < len(rows) - 1:
            sep = np.full((ROW_PAD, width, 3), 245, dtype=np.uint8)
            pieces.append(sep)
    return np.vstack(pieces)


def safe_float(x: Optional[str]) -> Optional[float]:
    if x in (None, "", "n/a"):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--retain-json", required=True, type=Path,
                   help="Path to retain.json (output of cover-image phase 1).")
    p.add_argument("--gains-csv", type=Path, default=None,
                   help="Optional per-video gains CSV for annotation + sort.")
    p.add_argument("--gt-dir", required=True, type=Path,
                   help="Source video dir (e.g., datasets/panda_cover_candidates_480p/videos).")
    p.add_argument("--notta-dir", required=True, type=Path,
                   help="No-TTA generated videos dir.")
    p.add_argument("--adasteer-dir", required=True, type=Path,
                   help="AdaSteer generated videos dir.")
    p.add_argument("--lora-dir", type=Path, default=None,
                   help="Optional LoRA generated videos dir.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Output directory for PNG filmstrips + index.html.")
    p.add_argument("--num-keyframes", type=int, default=DEFAULT_NUM_KEYFRAMES)
    p.add_argument("--frame-w", type=int, default=DEFAULT_FRAME_W)
    p.add_argument("--frame-h", type=int, default=DEFAULT_FRAME_H)
    p.add_argument("--gt-frame-start", type=int, default=48,
                   help="First frame to sample from the GT video. Default 48 "
                        "matches gen_start_frame in the eval config.")
    p.add_argument("--gt-frame-count", type=int, default=28,
                   help="Number of frames to span on the GT (matches num_frames).")
    p.add_argument("--methods", nargs="+",
                   default=["GT", "No-TTA", "AdaSteer", "LoRA R8"],
                   help="Which rows to include and in what order.")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    retain = load_retain(args.retain_json)
    gains = load_gains(args.gains_csv)

    # Build per-method idx-by-name maps (post-rename pattern support)
    method_dirs = {
        "GT": args.gt_dir,
        "No-TTA": args.notta_dir,
        "AdaSteer": args.adasteer_dir,
        "LoRA R8": args.lora_dir,
    }
    method_idx = {
        name: (load_summary_order(d) if (d is not None and name != "GT") else {})
        for name, d in method_dirs.items()
    }

    chosen_methods: List[str] = []
    for m in args.methods:
        if method_dirs.get(m) is None:
            print(f"[skip] method {m} requested but its dir is None")
            continue
        chosen_methods.append(m)

    print(f"Retain set: {len(retain)} videos")
    print(f"Methods: {chosen_methods}")
    print(f"Frame size: {args.frame_w}x{args.frame_h}  Keyframes: {args.num_keyframes}")

    successes: List[Dict] = []
    failures: List[Tuple[str, str]] = []

    for video_id in retain:
        rows: List[np.ndarray] = []
        missing: List[str] = []
        for m in chosen_methods:
            mdir = method_dirs[m]
            mp4 = find_mp4(mdir, video_id, method_idx.get(m, {}))
            if mp4 is None:
                missing.append(m)
                blank_row = compose_row(
                    m, [np.full((args.frame_h, args.frame_w, 3), 230, dtype=np.uint8)
                        for _ in range(args.num_keyframes)],
                    args.frame_w, args.frame_h, LABEL_COL_W,
                )
                rows.append(blank_row)
                continue
            if m == "GT":
                frames = sample_frames(
                    mp4, args.num_keyframes,
                    start=args.gt_frame_start, count=args.gt_frame_count,
                    out_w=args.frame_w, out_h=args.frame_h,
                )
            else:
                frames = sample_frames(
                    mp4, args.num_keyframes,
                    start=0, count=None,
                    out_w=args.frame_w, out_h=args.frame_h,
                )
            rows.append(compose_row(m, frames, args.frame_w, args.frame_h, LABEL_COL_W))

        if missing:
            failures.append((video_id, ",".join(missing)))

        g = gains.get(video_id, {})
        theme = g.get("theme", "")
        dpsnr = safe_float(g.get("dpsnr"))
        dssim = safe_float(g.get("dssim"))
        dlpips = safe_float(g.get("dlpips"))
        caption = g.get("caption", "")

        strip_w = LABEL_COL_W + args.num_keyframes * args.frame_w + (args.num_keyframes - 1) * COL_PAD
        header = compose_header(video_id, theme, dpsnr, dssim, dlpips,
                                caption, width=strip_w, height=HEADER_BAND_H)
        strip = compose_filmstrip(rows, header)

        tag = "miss" if missing else "ok"
        rank_tag = f"{dpsnr:+.2f}" if dpsnr is not None else "nan"
        out_name = f"{video_id}_dpsnr{rank_tag}.png"
        out_path = args.out_dir / out_name
        # PIL save avoids the cv2 numpy-ABI rejection on cluster wheels.
        Image.fromarray(np.ascontiguousarray(strip[:, :, ::-1])).save(out_path)
        successes.append({
            "video": video_id, "out": out_name, "dpsnr": dpsnr,
            "dssim": dssim, "dlpips": dlpips, "theme": theme,
            "caption": caption, "tag": tag, "missing": missing,
        })
        print(f"  [{tag}] {video_id}  dPSNR={rank_tag}  -> {out_name}"
              + (f"  (missing: {missing})" if missing else ""))

    print(f"\nWrote {len(successes)} filmstrip PNGs to {args.out_dir}")
    if failures:
        print(f"[warn] {len(failures)} candidates had missing methods:")
        for v, m in failures:
            print(f"  {v}: missing {m}")

    # Index HTML, sorted by dpsnr desc
    successes_sorted = sorted(
        successes,
        key=lambda r: (r["dpsnr"] is None, -(r["dpsnr"] or 0.0)),
    )
    index_path = args.out_dir / "index.html"
    rows_html: List[str] = []
    for r in successes_sorted:
        cap_safe = html.escape(r["caption"][:200])
        miss = (" <span style='color:#c00'>(missing: " + html.escape(", ".join(r["missing"])) + ")</span>"
                if r["missing"] else "")
        dpsnr_disp = f"{r['dpsnr']:+.2f}" if r['dpsnr'] is not None else "n/a"
        dssim_disp = f"{r['dssim']:+.3f}" if r['dssim'] is not None else "n/a"
        dlpips_disp = f"{r['dlpips']:+.3f}" if r['dlpips'] is not None else "n/a"
        rows_html.append(
            f"""<section style="margin: 12px 0; padding: 10px; border: 1px solid #ddd; border-radius: 6px;">
  <h3 style="margin:0 0 4px 0; font: 600 14px/1.2 system-ui, sans-serif;">
    {html.escape(r['video'])}
    <span style="font-weight:400; color:#666;">  theme: {html.escape(r['theme'])}</span>
    <span style="font-weight:400;">  dPSNR {dpsnr_disp}  dSSIM {dssim_disp}  dLPIPS {dlpips_disp}</span>{miss}
  </h3>
  <p style="margin: 2px 0 6px 0; color:#555; font: 13px/1.3 system-ui, sans-serif;">{cap_safe}</p>
  <img src="{html.escape(r['out'])}" style="max-width:100%; border:1px solid #eee;" />
</section>"""
        )
    index_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>AdaSteer cover-image candidates</title>
<style>
  body {{ background:#fafafa; color:#222; margin: 18px auto; max-width: 1700px;
         font: 14px/1.4 system-ui, -apple-system, sans-serif; }}
  h1 {{ font: 700 20px/1.2 system-ui, sans-serif; margin: 0 0 16px 0; }}
  .meta {{ color:#666; margin-bottom: 16px; }}
</style></head><body>
<h1>AdaSteer cover-image candidates ({len(successes_sorted)})</h1>
<p class="meta">Sorted by dPSNR (largest gain first). Each strip shows
{args.num_keyframes} evenly-spaced keyframes per method.</p>
{chr(10).join(rows_html)}
</body></html>
"""
    index_path.write_text(index_html)
    print(f"Wrote index: {index_path}")


if __name__ == "__main__":
    main()
