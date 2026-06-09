#!/usr/bin/env python3
"""Per-video feature extraction for the TTA-gain correlation analysis.

# ============================================================================
# TTA-VISIBLE FRAMES AUDIT  --  panda_1000v_standard  (2026-06-09)
# ============================================================================
# QUESTION ANSWERED HERE: "Which subset of each GT clip does the TTA loop
# actually see for the panda_1000v_standard runs whose per-video PSNRs we
# already have in sweep_experiment/reports/per_video_analysis/2026-06-09/
# per_video_gains.csv ?"
#
# AUDIT SOURCES (frame counts come from environment variables exported in
# the chunked submit script; runners default the TTA window to start at
# `gen_start_frame - tta_total_frames` and clamp to avoid GT-region leakage):
#
#   * sweep_experiment/sbatch/submit_standard_1000v_chunked.sh
#       NUM_FRAMES=28
#       NUM_COND_FRAMES=14
#       GEN_START_FRAME=48
#       TTA_TOTAL_FRAMES=48
#       TTA_CONTEXT_FRAMES=14
#       NUM_INFERENCE_STEPS=50, GUIDANCE_SCALE=4.0, RESOLUTION=480p
#   * delta_experiment/scripts/run_delta_a.py  (ADA / ADA_NOPROMPT)
#       lines 613-628  : defaults / safety clamps
#       lines 852-869  : tta_start = gen_start_frame - tta_total_frames;
#                        load_video_frames(...,  tta_total_frames, ...,
#                                          start_frame=max(0, tta_start))
#                        num_ctx_lat = 1 + (tta_context_frames - 1) // 4
#                        split_tta_latents(all_latents, num_ctx_lat, ...)
#   * lora_experiment/scripts/run_lora_tta.py  (LORA_R8_TTA / LORA_R8_TTA_NOPROMPT)
#       lines 817-831  : same defaults / safety clamps
#       lines 1137-1145, 1179-1188 : same tta_start / num_ctx_lat formula
#   * delta_experiment/scripts/run_tinylora.py  (TL_BARE_R2 / TL_TIED_R2)
#       lines 265-276  : same defaults / safety clamps
#       lines 471-484  : same tta_start / num_ctx_lat formula
#   * delta_experiment/scripts/common.py
#       load_video_frames():  decodes `num_frames` frames starting at byte-
#                             precise `start_frame` via PyAV, returns
#                             tensor [1, 3, T, H, W] in [-1, 1] after
#                             trilinear-resize to (T, 480, 832).
#
# RESULT: ALL THREE runners agree.  For panda_1000v_standard each video's
# TTA loop loads pixel frames [0:48] (since tta_total_frames == gen_start_frame
# == 48) and uses every one of those 48 frames:
#       - first 14 frames (tta_context_frames=14) are clean VAE-encoded
#         context  (-> 4 context latents with VAE temporal scale 4);
#       - the remaining 34 frames are the noised-and-denoised TTA training
#         target (further split into train/val per early-stopping holdout).
#   Generation then produces frames [48:62]  (num_frames=28 of which 14
#   are conditioning so 14 are genuinely new).  Frames [48:62] are GT
#   for the PSNR/SSIM/LPIPS we already have; they are NOT visible to
#   TTA (so any feature computed on them is "Tier 3 diagnostic, not
#   online-actionable" per the user's constraint).
#
# Concretely the model already had _every pre-anchor frame_ in pixel space
# for this config.  TTA sees the full pre-anchor clip; the visible window
# happens to be 100 % of the pre-anchor span.  This is the most generous
# read for the model and matches what is available at deploy time in this
# evaluation setup.
#
# Frame indices used by THIS SCRIPT (auto mode for panda_1000v_standard):
#       TTA-visible:        [0, 48)   ->  48 frames per video
#       Generation-target:  [48, 62)  ->  14 frames per video (Tier 3 only)
# ============================================================================

Tier-1 features computed on TTA-visible frames + caption (the model's actual
TTA-time information, per the user's "we only care about the subsection of
data the model has access to during TTA" constraint).

Tier-3 features compare TTA-visible frames against the generation-target
frames (which are GT, not online-actionable).  They are still emitted as
diagnostics but the corresponding columns are documented in the schema
comment block below as not-online-actionable.

CSV schema (one row per ``video_id``):

    video_id, n_frames_used, tta_visible_range, gen_target_range, caption,

    # Tier 1 -- visible-only, online-actionable
    cut_count_pyscenedetect, cut_count_histogram, cut_density_per_frame,
    clip_text_image_sim_mean, clip_text_image_sim_var,
    clip_text_image_sim_min,
    dino_temporal_l2_mean,
    laplacian_variance_mean,
    rgb_histogram_entropy_mean,

    # Tier 3 -- DIAGNOSTIC ONLY, uses GT generation-target frames
    dino_tta_vs_genregion_sim,
    clip_text_genregion_sim_mean,

    # Provenance
    clip_model, dino_model, hist_bins_per_channel, hist_bhattacharyya_thresh

Dependencies (the user's cluster env already has all of these for the
existing CLIP-gate / dynamicness pipeline; see env_setup/01_setup_longcat_env.sbatch):

    torch, transformers (CLIPModel/CLIPProcessor + AutoImageProcessor/AutoModel
        for DINOv2), av (PyAV), opencv-python, numpy, scenedetect (PySceneDetect).

PySceneDetect is treated as OPTIONAL: if the import fails the column
``cut_count_pyscenedetect`` (and ``cut_density_per_frame`` which derives
from it) become NaN and a one-line WARNING is printed.  The histogram-based
``cut_count_histogram`` is always populated as the backup.

CLI mirrors ``scripts/analyze_per_video_tta_gain.py`` conventions:

    python3 scripts/extract_video_features_for_tta.py \\
        --videos-dir datasets/panda_1000_480p \\
        --captions-csv datasets/panda_1000_480p/metadata.csv \\
        --tta-visible-frames auto \\
        --output sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv \\
        --device cuda \\
        --batch-size 16
"""
from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Lazy / guarded imports so the script can still print --help on machines
# that do not have torch installed.  All heavy imports happen inside main()
# or the per-feature helpers.


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# panda_1000v_standard frame geometry (see audit block at top of file).
AUTO_TTA_VISIBLE_RANGE: Tuple[int, int] = (0, 48)
AUTO_GEN_TARGET_RANGE: Tuple[int, int] = (48, 62)

# Histogram cut detector: Bhattacharyya distance threshold.  Calibrated so
# that obvious hard cuts in panda_100 sample clips fire (>= 0.4 is the
# OpenCV-tutorial "strong dissimilarity" band for HISTCMP_BHATTACHARYYA),
# and most slow camera pans / talking-head segments stay quiet.  Pinned
# here so the column is reproducible across runs.
HIST_BHATTACHARYYA_THRESH: float = 0.40
HIST_BINS_PER_CHANNEL: int = 8

# Default model checkpoints (small + fast, match what the existing CLIP
# gate code uses; can be overridden on the CLI).
DEFAULT_CLIP_MODEL: str = "openai/clip-vit-base-patch32"
DEFAULT_DINO_MODEL: str = "facebook/dinov2-small"

PROGRESS_EVERY: int = 50

# Tier 3 columns are emitted but flagged here so downstream consumers can
# split them out programmatically.
TIER3_COLUMNS: Tuple[str, ...] = (
    "dino_tta_vs_genregion_sim",
    "clip_text_genregion_sim_mean",
)


# ---------------------------------------------------------------------------
# Canonical video-id extraction (mirrors analyze_per_video_tta_gain.py)
# ---------------------------------------------------------------------------
_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def _canonical_video_id(s: Optional[str]) -> str:
    if not s:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


# ---------------------------------------------------------------------------
# Caption parsing (Panda metadata.csv stores a stringified Python list)
# ---------------------------------------------------------------------------
def parse_caption(raw: str) -> List[str]:
    """Return the list of caption strings encoded in `raw`.

    Panda's metadata.csv stores captions as a Python literal list, e.g.
    ``"['cap1', 'cap2', 'cap3']"``.  UCF-style entries are bare strings.
    Always returns a non-empty list when raw is non-empty.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            obj = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return [raw]
        if isinstance(obj, (list, tuple)):
            out = [str(x).strip() for x in obj if str(x).strip()]
            return out or [raw]
    return [raw]


def caption_for_clip(captions: List[str]) -> str:
    """Single canonical text used for the CLIP text encoding (joined)."""
    if not captions:
        return ""
    # Join with ". " so a multi-caption clip's text embedding is the mean
    # *content* across captions (matches what the diffusion model's text
    # encoder receives when caption is the raw stringified list at
    # generation time, but in a way CLIP's 77-token tokenizer can handle).
    joined = ". ".join(c.rstrip(".") for c in captions)
    return joined


# ---------------------------------------------------------------------------
# Video frame decoding (PyAV; matches delta_experiment/scripts/common.py)
# ---------------------------------------------------------------------------
def decode_window(
    video_path: str,
    start_frame: int,
    num_frames: int,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Decode a contiguous frame window as uint8 HxWx3 (RGB).

    Returns array shaped (T, H, W, 3) with len(T) == num_frames (padded by
    repeating the last frame if the source clip is too short).  Optional
    ``resize_hw=(h, w)`` resizes each decoded frame with OpenCV INTER_AREA.
    """
    import av
    import cv2  # noqa: F401  -- used below if resize_hw

    container = av.open(video_path)
    frames: List[np.ndarray] = []
    decoded = 0
    try:
        for frame in container.decode(video=0):
            if decoded < start_frame:
                decoded += 1
                continue
            if len(frames) >= num_frames:
                break
            img = frame.to_ndarray(format="rgb24")
            if resize_hw is not None:
                import cv2 as _cv
                h_t, w_t = resize_hw
                if img.shape[0] != h_t or img.shape[1] != w_t:
                    img = _cv.resize(img, (w_t, h_t), interpolation=_cv.INTER_AREA)
            frames.append(img)
            decoded += 1
    finally:
        container.close()

    if not frames:
        raise ValueError(f"No frames decoded from {video_path} at start={start_frame}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())
    return np.stack(frames[:num_frames], axis=0)


# ---------------------------------------------------------------------------
# Cheap CPU features (cuts + texture + colour)
# ---------------------------------------------------------------------------
def count_cuts_histogram(frames_rgb: np.ndarray,
                         bins_per_channel: int = HIST_BINS_PER_CHANNEL,
                         thresh: float = HIST_BHATTACHARYYA_THRESH) -> int:
    """Count consecutive frame pairs whose RGB joint-histogram Bhattacharyya
    distance exceeds ``thresh``.  Cheap, fully deterministic backup for
    PySceneDetect.
    """
    import cv2

    T = frames_rgb.shape[0]
    if T < 2:
        return 0
    hist_size = [bins_per_channel] * 3
    ranges = [0, 256, 0, 256, 0, 256]
    prev_hist = None
    cuts = 0
    for t in range(T):
        # cv2 expects BGR -- but Bhattacharyya on a *joint* histogram does
        # not care about channel order as long as it is consistent.
        hist = cv2.calcHist([frames_rgb[t]], [0, 1, 2], None, hist_size, ranges)
        cv2.normalize(hist, hist)
        if prev_hist is not None:
            dist = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
            if dist >= thresh:
                cuts += 1
        prev_hist = hist
    return cuts


def count_cuts_pyscenedetect(video_path: str, start_frame: int,
                             end_frame: int) -> Optional[int]:
    """Use PySceneDetect ContentDetector on the [start_frame, end_frame)
    range.  Returns ``None`` if PySceneDetect is not installed; caller is
    responsible for emitting NaN into the CSV in that case.
    """
    try:
        from scenedetect import SceneManager, open_video
        from scenedetect.detectors import ContentDetector
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] PySceneDetect unavailable ({exc}); "
              "cut_count_pyscenedetect will be NaN", file=sys.stderr)
        return None

    try:
        video = open_video(video_path)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector())
        scene_manager.detect_scenes(
            video, frame_skip=0, show_progress=False,
            duration=None, start_time=None, end_time=None,
        )
        # PySceneDetect returns scene cuts as start times of each scene
        # (in frames). The number of cuts is len(scenes) - 1 if scenes is
        # not empty.  We restrict to those falling inside [start_frame,
        # end_frame).
        scenes = scene_manager.get_scene_list()
        cut_count = 0
        for s_start, _s_end in scenes[1:]:  # skip the artificial scene-0 start
            f = s_start.get_frames()
            if start_frame <= f < end_frame:
                cut_count += 1
        return int(cut_count)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] PySceneDetect failed on {video_path}: {exc}",
              file=sys.stderr)
        return None


def laplacian_variance_mean(frames_rgb: np.ndarray) -> float:
    """Mean over frames of Var(cv2.Laplacian(gray)).  Sharpness proxy."""
    import cv2

    vals: List[float] = []
    for t in range(frames_rgb.shape[0]):
        gray = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        vals.append(float(lap.var()))
    return float(np.mean(vals)) if vals else float("nan")


def rgb_histogram_entropy_mean(frames_rgb: np.ndarray,
                               bins_per_channel: int = HIST_BINS_PER_CHANNEL,
                               ) -> float:
    """Mean over frames of Shannon entropy (bits) of the joint RGB histogram.

    A featureless monochrome frame entropy -> ~0; a richly-coloured
    natural frame approaches log2(bins_per_channel ** 3).
    """
    import cv2

    vals: List[float] = []
    hist_size = [bins_per_channel] * 3
    ranges = [0, 256, 0, 256, 0, 256]
    for t in range(frames_rgb.shape[0]):
        hist = cv2.calcHist([frames_rgb[t]], [0, 1, 2], None, hist_size, ranges)
        hist = hist.flatten()
        total = float(hist.sum())
        if total <= 0:
            continue
        p = hist / total
        p = p[p > 0]
        vals.append(float(-(p * np.log2(p)).sum()))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# CLIP image / text encoder (HuggingFace transformers; matches CLIP-gate path)
# ---------------------------------------------------------------------------
class _CLIPScorer:
    def __init__(self, model_name: str, device: str):
        from transformers import CLIPModel, CLIPProcessor
        import torch

        self.device = device
        self.model_name = model_name
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.torch = torch

    def encode_images(self, frames_rgb: np.ndarray, batch_size: int = 16):
        """frames_rgb: (T, H, W, 3) uint8 -> (T, D) normalised image embeds."""
        torch = self.torch
        T = frames_rgb.shape[0]
        out = []
        with torch.inference_mode():
            for s in range(0, T, batch_size):
                chunk = [frames_rgb[i] for i in range(s, min(s + batch_size, T))]
                # CLIPProcessor accepts a list of HWC uint8 numpy / PIL.
                inputs = self.processor(images=chunk, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                feats = self.model.get_image_features(pixel_values=pixel_values)
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)

    def encode_text(self, text: str):
        torch = self.torch
        with torch.inference_mode():
            inputs = self.processor(text=[text], return_tensors="pt",
                                    truncation=True, padding=True, max_length=77)
            input_ids = inputs["input_ids"].to(self.device)
            attn = inputs.get("attention_mask")
            if attn is not None:
                attn = attn.to(self.device)
            feat = self.model.get_text_features(
                input_ids=input_ids,
                attention_mask=attn,
            )
            feat = feat / feat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return feat.float().cpu().numpy()[0]


# ---------------------------------------------------------------------------
# DINOv2 image encoder
# ---------------------------------------------------------------------------
class _DINOScorer:
    def __init__(self, model_name: str, device: str):
        from transformers import AutoImageProcessor, AutoModel
        import torch

        self.device = device
        self.model_name = model_name
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.torch = torch

    def encode_images(self, frames_rgb: np.ndarray, batch_size: int = 16):
        """frames_rgb: (T, H, W, 3) uint8 -> (T, D) L2-normalised CLS embeddings."""
        torch = self.torch
        T = frames_rgb.shape[0]
        out = []
        with torch.inference_mode():
            for s in range(0, T, batch_size):
                chunk = [frames_rgb[i] for i in range(s, min(s + batch_size, T))]
                inputs = self.processor(images=chunk, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self.device)
                feats = self.model(pixel_values=pixel_values).last_hidden_state[:, 0]
                feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                out.append(feats.float().cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)


# ---------------------------------------------------------------------------
# Caption CSV loader (mirrors analyze_per_video_tta_gain.py.load_captions)
# ---------------------------------------------------------------------------
def load_captions_csv(path: Path) -> Dict[str, str]:
    """Return {canonical_video_id -> raw caption string}.  Tolerant of
    Panda's metadata.csv schema (filename, caption) and the simpler UCF-101
    one (filename, text)."""
    out: Dict[str, str] = {}
    if not path.exists():
        print(f"[warn] captions CSV not found at {path}; rows will use empty "
              "captions and CLIP text features will be NaN",
              file=sys.stderr)
        return out
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (row.get("filename") or row.get("video_path")
                     or row.get("path") or row.get("video"))
            if not fname:
                continue
            vid = _canonical_video_id(fname)
            if not vid:
                continue
            cap = row.get("caption") or row.get("text") or ""
            out[vid] = cap
    return out


# ---------------------------------------------------------------------------
# Video file enumeration (mirrors load_ucf101_video_list logic but file-only)
# ---------------------------------------------------------------------------
def list_video_paths(videos_dir: Path) -> List[Path]:
    """Return sorted (by canonical id) list of video files under
    ``videos_dir``.  Honours the ``videos/`` subdir layout the Panda
    datasets use; falls back to a recursive .mp4/.avi scan otherwise."""
    candidates: List[Path] = []
    subdir = videos_dir / "videos"
    if subdir.is_dir():
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(subdir.glob(ext))
    if not candidates:
        for ext in ("*.mp4", "*.avi"):
            candidates.extend(videos_dir.rglob(ext))
    return sorted(candidates, key=lambda p: _canonical_video_id(p.name))


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------
def _parse_visible_arg(arg: str) -> Tuple[int, int]:
    """Parse --tta-visible-frames.  'auto' -> AUTO_TTA_VISIBLE_RANGE; 'A:B' -> (A, B)."""
    if not arg or arg.lower() == "auto":
        return AUTO_TTA_VISIBLE_RANGE
    if ":" in arg:
        a, b = arg.split(":", 1)
        return int(a), int(b)
    raise argparse.ArgumentTypeError(
        f"--tta-visible-frames must be 'auto' or 'A:B', got {arg!r}"
    )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--videos-dir", type=Path, required=True,
                    help="Dataset root containing videos/ subdir or *.mp4 directly.")
    ap.add_argument("--captions-csv", type=Path, required=True,
                    help="Panda metadata.csv (or UCF-style: filename,text).")
    ap.add_argument("--tta-visible-frames", type=str, default="auto",
                    help="'auto' (resolved from panda_1000v_standard audit: "
                         f"{AUTO_TTA_VISIBLE_RANGE[0]}:{AUTO_TTA_VISIBLE_RANGE[1]}) "
                         "or an explicit 'A:B' python-slice range.")
    ap.add_argument("--gen-target-frames", type=str, default="auto",
                    help="'auto' (resolves to "
                         f"{AUTO_GEN_TARGET_RANGE[0]}:{AUTO_GEN_TARGET_RANGE[1]}) "
                         "or 'A:B'.  Used ONLY by Tier-3 diagnostic columns. "
                         "Pass 'none' (or omit if visible == full clip) to "
                         "leave those columns NaN.")
    ap.add_argument("--output", type=Path, required=True,
                    help="CSV path. Idempotent — existing rows are reused "
                         "unless --force is passed.")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Torch device. 'cpu' works but is much slower.")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="Frames per CLIP/DINO forward pass within a video.")
    ap.add_argument("--max-videos", type=int, default=0,
                    help="0 = all. Otherwise process the first N (by canonical id).")
    ap.add_argument("--force", action="store_true",
                    help="Recompute even if the row already exists in --output.")
    ap.add_argument("--clip-model", type=str, default=DEFAULT_CLIP_MODEL,
                    help=f"HuggingFace CLIP id. Default: {DEFAULT_CLIP_MODEL}.")
    ap.add_argument("--dino-model", type=str, default=DEFAULT_DINO_MODEL,
                    help=f"HuggingFace DINOv2 id. Default: {DEFAULT_DINO_MODEL}.")
    ap.add_argument("--resize-h", type=int, default=480,
                    help="Decode/resize frames to this height before featurization.")
    ap.add_argument("--resize-w", type=int, default=832,
                    help="Decode/resize frames to this width before featurization.")
    ap.add_argument("--skip-pyscenedetect", action="store_true",
                    help="Force-skip PySceneDetect even if installed "
                         "(use for environments where it's known flaky).")
    return ap.parse_args()


def _load_existing_csv(path: Path) -> Tuple[List[dict], List[str]]:
    """Return (rows, fieldnames) for an existing output CSV (idempotency)."""
    if not path.exists():
        return [], []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        fieldnames = list(reader.fieldnames) if reader.fieldnames else []
    return rows, fieldnames


def _format_row(row: dict, fieldnames: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k in fieldnames:
        v = row.get(k)
        if v is None:
            out[k] = ""
        elif isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                out[k] = ""
            else:
                out[k] = f"{v:.6f}"
        else:
            out[k] = str(v)
    return out


# ---------------------------------------------------------------------------
# Per-video extraction (one entry point so the body is easy to read)
# ---------------------------------------------------------------------------
def extract_one_video(
    video_path: Path,
    caption_raw: str,
    visible_range: Tuple[int, int],
    gen_range: Optional[Tuple[int, int]],
    *,
    clip: _CLIPScorer,
    dino: _DINOScorer,
    resize_hw: Tuple[int, int],
    batch_size: int,
    skip_pyscenedetect: bool,
) -> dict:
    vs, ve = visible_range
    n_visible = ve - vs
    captions = parse_caption(caption_raw)
    caption_text = caption_for_clip(captions)

    # ---- decode TTA-visible frames once ------------------------------------
    visible_frames = decode_window(
        str(video_path), start_frame=vs, num_frames=n_visible,
        resize_hw=resize_hw,
    )

    # ---- cheap CPU features -----------------------------------------------
    cut_hist = count_cuts_histogram(visible_frames)
    cut_psd: Optional[int] = (
        None if skip_pyscenedetect
        else count_cuts_pyscenedetect(str(video_path), start_frame=vs, end_frame=ve)
    )
    cut_density = (
        (cut_psd / n_visible) if (cut_psd is not None and n_visible > 0)
        else float("nan")
    )
    lap_var_mean = laplacian_variance_mean(visible_frames)
    hist_ent_mean = rgb_histogram_entropy_mean(visible_frames)

    # ---- CLIP image embeddings (visible region) ---------------------------
    clip_img_visible = clip.encode_images(visible_frames, batch_size=batch_size)
    clip_text_vec = (
        clip.encode_text(caption_text)
        if caption_text else None
    )
    if clip_text_vec is not None and clip_img_visible.shape[0] > 0:
        sims_visible = clip_img_visible @ clip_text_vec  # cosine since normalised
        ct_mean = float(sims_visible.mean())
        ct_var = float(sims_visible.var(ddof=0))
        ct_min = float(sims_visible.min())
    else:
        ct_mean = ct_var = ct_min = float("nan")

    # ---- DINO temporal coherence on visible region ------------------------
    dino_visible = dino.encode_images(visible_frames, batch_size=batch_size)
    if dino_visible.shape[0] >= 2:
        diffs = dino_visible[1:] - dino_visible[:-1]
        l2 = np.linalg.norm(diffs, axis=1)
        dino_temp_l2 = float(l2.mean())
    else:
        dino_temp_l2 = float("nan")

    # ---- Tier 3: generation-target region (diagnostic only) ---------------
    dino_tta_vs_gen = float("nan")
    clip_text_gen_mean = float("nan")
    if gen_range is not None:
        gs, ge = gen_range
        n_gen = ge - gs
        if n_gen > 0:
            try:
                gen_frames = decode_window(
                    str(video_path), start_frame=gs, num_frames=n_gen,
                    resize_hw=resize_hw,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] {video_path.name}: gen-region decode failed "
                      f"({exc}); Tier-3 columns NaN", file=sys.stderr)
                gen_frames = None
            if gen_frames is not None:
                dino_gen = dino.encode_images(gen_frames, batch_size=batch_size)
                if dino_gen.shape[0] > 0 and dino_visible.shape[0] > 0:
                    mu_tta = dino_visible.mean(axis=0)
                    mu_tta = mu_tta / np.linalg.norm(mu_tta).clip(min=1e-12)
                    mu_gen = dino_gen.mean(axis=0)
                    mu_gen = mu_gen / np.linalg.norm(mu_gen).clip(min=1e-12)
                    dino_tta_vs_gen = float(mu_tta @ mu_gen)
                if clip_text_vec is not None:
                    clip_img_gen = clip.encode_images(gen_frames, batch_size=batch_size)
                    if clip_img_gen.shape[0] > 0:
                        sims_gen = clip_img_gen @ clip_text_vec
                        clip_text_gen_mean = float(sims_gen.mean())

    vid_id = _canonical_video_id(video_path.name)
    return {
        "video_id": vid_id,
        "n_frames_used": int(n_visible),
        "tta_visible_range": f"{vs}:{ve}",
        "gen_target_range": (
            f"{gen_range[0]}:{gen_range[1]}" if gen_range is not None else ""
        ),
        "caption_text": caption_text,
        # Tier 1
        "cut_count_pyscenedetect": (
            float("nan") if cut_psd is None else int(cut_psd)
        ),
        "cut_count_histogram": int(cut_hist),
        "cut_density_per_frame": float(cut_density),
        "clip_text_image_sim_mean": float(ct_mean),
        "clip_text_image_sim_var": float(ct_var),
        "clip_text_image_sim_min": float(ct_min),
        "dino_temporal_l2_mean": float(dino_temp_l2),
        "laplacian_variance_mean": float(lap_var_mean),
        "rgb_histogram_entropy_mean": float(hist_ent_mean),
        # Tier 3 (DIAGNOSTIC ONLY)
        "dino_tta_vs_genregion_sim": float(dino_tta_vs_gen),
        "clip_text_genregion_sim_mean": float(clip_text_gen_mean),
        # Provenance
        "clip_model": clip.model_name,
        "dino_model": dino.model_name,
        "hist_bins_per_channel": int(HIST_BINS_PER_CHANNEL),
        "hist_bhattacharyya_thresh": float(HIST_BHATTACHARYYA_THRESH),
    }


def _fieldnames() -> List[str]:
    return [
        "video_id", "n_frames_used", "tta_visible_range", "gen_target_range",
        "caption_text",
        "cut_count_pyscenedetect", "cut_count_histogram",
        "cut_density_per_frame",
        "clip_text_image_sim_mean", "clip_text_image_sim_var",
        "clip_text_image_sim_min",
        "dino_temporal_l2_mean",
        "laplacian_variance_mean",
        "rgb_histogram_entropy_mean",
        "dino_tta_vs_genregion_sim",
        "clip_text_genregion_sim_mean",
        "clip_model", "dino_model",
        "hist_bins_per_channel", "hist_bhattacharyya_thresh",
    ]


def main() -> int:
    args = _parse_args()
    visible_range = _parse_visible_arg(args.tta_visible_frames)
    if args.gen_target_frames.lower() == "none":
        gen_range: Optional[Tuple[int, int]] = None
    else:
        gen_range = _parse_visible_arg(args.gen_target_frames)
        # Sanity: don't let gen overlap visible (would mean Tier 3 == Tier 1
        # which would silently inflate diagnostic correlations).
        if gen_range[0] < visible_range[1]:
            print(f"[warn] gen-target range {gen_range} starts inside the "
                  f"visible range {visible_range}; Tier-3 columns will be "
                  "computed but flagged as overlapping.",
                  file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Per-video feature extraction (TTA-visible scope)")
    print("=" * 70)
    print(f"Videos dir         : {args.videos_dir}")
    print(f"Captions CSV       : {args.captions_csv}")
    print(f"TTA-visible frames : {visible_range[0]}:{visible_range[1]}  "
          f"(n={visible_range[1] - visible_range[0]})")
    print(f"Gen-target frames  : "
          f"{'none' if gen_range is None else f'{gen_range[0]}:{gen_range[1]}'}")
    print(f"Output             : {args.output}")
    print(f"Device             : {args.device}")
    print(f"Batch size         : {args.batch_size}")
    print(f"CLIP model         : {args.clip_model}")
    print(f"DINO model         : {args.dino_model}")
    print(f"PySceneDetect      : {'skipped' if args.skip_pyscenedetect else 'enabled (if importable)'}")
    print("=" * 70)

    # ---- existing rows for idempotency ------------------------------------
    fieldnames = _fieldnames()
    existing_rows, existing_fields = _load_existing_csv(args.output)
    existing_by_id: Dict[str, dict] = {}
    if existing_rows and not args.force:
        for r in existing_rows:
            vid = r.get("video_id", "").strip()
            if vid:
                existing_by_id[vid] = r
        # Warn if the existing CSV's schema differs from ours -- we'll keep
        # extra historical columns by extending fieldnames.
        for extra in existing_fields:
            if extra and extra not in fieldnames:
                fieldnames.append(extra)
        print(f"[info] Loaded {len(existing_by_id)} existing rows from "
              f"{args.output}; will skip videos already covered "
              "(pass --force to recompute).")

    # ---- enumerate videos + load captions ---------------------------------
    captions_by_id = load_captions_csv(args.captions_csv)
    video_paths = list_video_paths(args.videos_dir)
    if args.max_videos > 0:
        video_paths = video_paths[: args.max_videos]
    if not video_paths:
        print(f"[error] no video files found under {args.videos_dir}",
              file=sys.stderr)
        return 2
    print(f"Videos discovered  : {len(video_paths)}")
    print(f"Captions loaded    : {len(captions_by_id)}")
    print()

    # ---- preload heavy models ---------------------------------------------
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"[error] torch not importable ({exc}); aborting", file=sys.stderr)
        return 2

    print("Loading CLIP model...")
    clip = _CLIPScorer(args.clip_model, args.device)
    print("Loading DINOv2 model...")
    dino = _DINOScorer(args.dino_model, args.device)
    resize_hw = (args.resize_h, args.resize_w)
    print()

    # ---- per-video loop ---------------------------------------------------
    n_done = 0
    n_skipped = 0
    n_errored = 0
    new_rows: List[dict] = []
    t0 = time.time()
    last_print_t = t0

    example_row: Optional[dict] = None

    for v_idx, vp in enumerate(video_paths):
        vid = _canonical_video_id(vp.name)
        if vid in existing_by_id and not args.force:
            n_skipped += 1
            continue
        cap = captions_by_id.get(vid, "")
        try:
            row = extract_one_video(
                vp, cap, visible_range, gen_range,
                clip=clip, dino=dino,
                resize_hw=resize_hw,
                batch_size=args.batch_size,
                skip_pyscenedetect=args.skip_pyscenedetect,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            print(f"[error] {vp.name}: {exc}", file=sys.stderr)
            traceback.print_exc()
            n_errored += 1
            continue
        new_rows.append(row)
        if example_row is None:
            example_row = row
        n_done += 1

        if (v_idx + 1) % PROGRESS_EVERY == 0 or (v_idx + 1) == len(video_paths):
            dt = time.time() - last_print_t
            last_print_t = time.time()
            print(f"  [{v_idx + 1}/{len(video_paths)}] "
                  f"new={n_done}  skip={n_skipped}  err={n_errored}  "
                  f"(+{dt:.1f}s)  vid={vid}",
                  flush=True)

    # ---- write merged CSV (existing rows preserved, new rows appended) ----
    final_rows_by_id: Dict[str, dict] = {}
    if not args.force:
        # preserve every existing row exactly as it was
        for r in existing_rows:
            v = r.get("video_id", "").strip()
            if v:
                final_rows_by_id[v] = r
    for r in new_rows:
        final_rows_by_id[r["video_id"]] = r

    sorted_rows = sorted(final_rows_by_id.values(),
                         key=lambda r: r.get("video_id", ""))
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted_rows:
            writer.writerow(_format_row(row, fieldnames))

    dt_total = time.time() - t0
    print()
    print(f"Wrote {args.output}")
    print(f"  total rows now : {len(sorted_rows)}")
    print(f"  new this run   : {n_done}")
    print(f"  skipped existing: {n_skipped}")
    print(f"  errored        : {n_errored}")
    print(f"  wall time      : {dt_total:.1f}s")

    if example_row is not None:
        print()
        print("Example row (one video, post-extraction):")
        for k in fieldnames:
            v = example_row.get(k)
            if isinstance(v, float):
                v_s = f"{v:.6f}" if (v == v and not math.isinf(v)) else "NaN"
            else:
                v_s = str(v)
            if k == "caption_text" and len(v_s) > 96:
                v_s = v_s[:93] + "..."
            print(f"  {k:32s} = {v_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
