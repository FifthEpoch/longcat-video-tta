#!/usr/bin/env python3
"""
deploy_pool_extensions_v2.py
============================

Cluster-side deploy script for the **UCF-101 chunking** upgrade. This is
a *focused delta* on top of v1 (`deploy_pool_extensions.py`):

    v1: built Panda 25K pool, UCF 12K pool (1-clip-per-source),
        embedding cache, plus retrieval-pool patches to common.py +
        run_delta_a.py.

    v2 (this script): replaces the UCF builder with a CHUNKED version
        (~28-32K chunks, ~50 GB), and patches common.py's neighbour
        selection to exclude *all* pool chunks sharing the eval clip's
        source video.

Expected order of operations on the cluster:

    1. (Optional but recommended) Run v1 if not already applied:
         python scripts/deploy_pool_extensions.py --apply

    2. Pull this script onto the cluster and run v2:
         python scripts/deploy_pool_extensions_v2.py            # dry-run
         python scripts/deploy_pool_extensions_v2.py --apply    # write

    3. Kick off the chunked UCF pool build:
         sbatch --account=torch_pr_36_mren \\
             datasets/build_ucf101_retrieval_pool.sbatch

    4. Pre-compute embeddings for the chunked pool:
         sbatch --account=torch_pr_36_mren \\
             --export=ALL,POOL_DIR=/scratch/wc3013/longcat-video-tta/datasets/ucf101_pool_max \\
             delta_experiment/sbatch/precompute_pool_embeddings.sbatch

    5. Launch the new UCF retrieval sweep:
         bash sweep_experiment/sbatch/submit_ucf101_2048v_poolmax_retrieval_chunked.sh

What this script changes:
-------------------------
* WRITE   scripts/build_ucf101_retrieval_pool.py
            Replaces v1 (1-clip-per-source) with the chunked builder.
            Adds --chunk-seconds / --chunk-gap / --max-chunks-per-source /
            --no-chunking. Writes `source_video_id` into metadata.csv +
            manifest.jsonl. Chunk naming: <source>_t<N>.mp4.
* WRITE   datasets/build_ucf101_retrieval_pool.sbatch
            Bumps wall-time 8h -> 12h, exports CHUNKING / CHUNK_SECONDS /
            CHUNK_GAP / MAX_CHUNKS_PER_SOURCE env knobs, removes the
            stale --existing-dataset flag (chunked builder seeds from
            raw UCF, not the 1000-clip subset).
* PATCH   delta_experiment/scripts/common.py:
            - Insert `_entry_source_id()` helper just before
              `sample_neighbors_random` (handles `source_video_id` /
              `source_videoID` fields, falls back to `videoID` with the
              `_t<N>` chunk suffix stripped).
            - Replace `sample_neighbors_random` to filter out
              same-source pool entries.
            - Replace `retrieve_neighbors` to filter out same-source
              pool entries.
            (No-op on Panda pools where source_video_id is missing
             and videoIDs have no `_t<N>` suffix -- the helper just
             returns the videoID unchanged.)

The patches are idempotent: if the NEW text is already present, the
script reports "already applied" instead of failing.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
# Helpers
# ============================================================================

def _md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def write_file(
    target: Path, content: str, *, apply: bool, label: str = "",
) -> str:
    target = target.resolve()
    label = label or str(target)
    existing = ""
    existing_md5 = ""
    if target.exists():
        existing = target.read_text(encoding="utf-8")
        existing_md5 = _md5(existing)
    new_md5 = _md5(content)
    if existing and existing_md5 == new_md5:
        return f"  [skip] {label} (already up-to-date, md5={existing_md5})"
    target.parent.mkdir(parents=True, exist_ok=True)
    if apply:
        target.write_text(content, encoding="utf-8")
        return (f"  [WRITE] {label}\n"
                f"          old md5: {existing_md5 or '(absent)'}\n"
                f"          new md5: {new_md5}")
    return (f"  [dry-run] would write {label}\n"
            f"          old md5: {existing_md5 or '(absent)'}\n"
            f"          new md5: {new_md5}")


def apply_patch(
    target: Path, old: str, new: str, *, apply: bool, label: str = "",
) -> str:
    target = target.resolve()
    label = label or str(target)
    if not target.exists():
        return f"  [SKIP] {label}: target missing -> {target}"
    content = target.read_text(encoding="utf-8")
    if old in content:
        if new in content:
            # Both old and new present -- ambiguous, leave alone.
            return (f"  [skip] {label}: BOTH old and new anchors present "
                    f"(no-op)")
        new_content = content.replace(old, new, 1)
        if not apply:
            return (f"  [dry-run] would patch {label}\n"
                    f"            old md5: {_md5(content)}\n"
                    f"            new md5: {_md5(new_content)}")
        target.write_text(new_content, encoding="utf-8")
        return (f"  [PATCH] {label}\n"
                f"          old md5: {_md5(content)}\n"
                f"          new md5: {_md5(new_content)}")
    if new in content:
        return f"  [skip] {label}: already applied"
    return (f"  [ERROR] {label}: neither old nor new anchor found.\n"
            f"          The target may have drifted from the expected\n"
            f"          version. Inspect manually and re-run.")


# ============================================================================
# New file contents
# ============================================================================

BUILD_UCF101_PY = r'''#!/usr/bin/env python3
"""
Build a maximum-size CHUNKED UCF-101 retrieval pool for similarity- or
random-augmented TTA.

Compared to a 1-clip-per-source UCF pool (which would top out at ~12K
entries because UCF-101 only has 13,320 source videos), this builder
slices each source clip into multiple non-overlapping 3-second chunks
and emits each chunk as a separate pool entry. The result is ~28-32K
chunks at ~50 GB on disk, reaching Panda-25K-pool parity.

Why chunking is sound for TTA retrieval:
  TTA training reads only the first `tta_total_frames` (=48) of each
  pool clip, which is ~1.92 s at UCF's typical 25 fps. A 3.0 s chunk
  covers that frame budget with margin and discards the rest. So every
  chunk we emit is a "valid training segment" -- no fake samples.

Same-source bias:
  Two chunks from the same source clip are highly visually correlated
  (next/prev frames of the same person). Retrieval must therefore
  exclude not just the eval clip itself, but ALL pool chunks that
  share the eval clip's source video. The pool builder emits a
  `source_video_id` field; `delta_experiment/scripts/common.py` has
  been patched (see deploy_pool_extensions.py) to use it during
  neighbour selection.

Captions on UCF (unchanged from the 1-clip-per-source builder):
  Class names humanised into stock natural-language captions, so every
  clip in a class shares an identical caption. Similarity-mode
  retrieval collapses to same-class retrieval; chunking does not
  change this.

Pipeline:
  1. Download UCF101.rar from CRCV (~6.5 GB) if not already cached.
  2. Extract to <staging>/UCF-101/<ClassName>/v_<ClassName>_g<NN>_c<NN>.avi
     using `unrar` (or `7z` / `bsdtar` as fallbacks).
  3. For each source clip:
       - Probe duration with ffprobe.
       - Plan chunk start times: stride = chunk_seconds + chunk_gap,
         cap = max_chunks_per_source.
       - For each planned chunk, ffmpeg-trim+scale to 832x480 with
         `-ss <start> -i <src> -t <chunk_seconds>` (fast input seek).
  4. Walk the chunks dir and write metadata.csv + manifest.jsonl with
     the chunk-level schema (`videoID` = chunk-level ID,
     `source_video_id` = original UCF clip ID).
  5. Validate (>= 48 decodable frames per chunk).
  6. Truncate metadata.csv to the first `target_valid` valid rows.

Chunk naming:
  source video : v_BasketballDunk_g01_c01
  chunk t=0    : v_BasketballDunk_g01_c01_t0   (start 0.0 s)
  chunk t=1    : v_BasketballDunk_g01_c01_t1   (start 3.5 s)
  chunk t=2    : v_BasketballDunk_g01_c01_t2   (start 7.0 s)
  ...

  The `_t<N>` suffix is what `_entry_source_id()` in common.py looks
  for as a same-source fallback when `source_video_id` is missing.

Resumable:
  - Already-encoded chunks (mp4 exists and size > 100 KB) are skipped.
  - Sources whose `source_video_id` already appears in manifest.jsonl
    are not re-probed/re-chunked.
  - Re-running the script continues from where it stopped.

Usage:
  python scripts/build_ucf101_retrieval_pool.py \\
      --new-dataset /scratch/wc3013/longcat-video-tta/datasets/ucf101_pool_max \\
      --rar-source  /scratch/wc3013/longcat-video-tta/datasets/ucf101_source \\
      --chunk-seconds 3.0 --chunk-gap 0.5 --max-chunks-per-source 10

  # Disable chunking (legacy 1-clip-per-source mode, ~12K pool):
  python scripts/build_ucf101_retrieval_pool.py ... --no-chunking

CPU-only job. Submit via datasets/build_ucf101_retrieval_pool.sbatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]

UCF_RAR_URL = "https://www.crcv.ucf.edu/data/UCF101/UCF101.rar"
UCF_RAR_EXPECTED_SIZE_MIN = 6_500_000_000
TARGET_WIDTH = 832
TARGET_HEIGHT = 480
RESIZE_SIZE_THRESHOLD_BYTES = 100 * 1024

CHUNK_SECONDS_DEFAULT = 3.0
CHUNK_GAP_DEFAULT = 0.5
MAX_CHUNKS_PER_SOURCE_DEFAULT = 10
MIN_CHUNK_SECONDS = 2.0


# ============================================================================
# Caption humanisation (unchanged from v1; same class -> same caption)
# ============================================================================

_UCF_CAPTION_OVERRIDES: Dict[str, str] = {
    "ApplyEyeMakeup": "a person applying eye makeup",
    "ApplyLipstick":  "a person applying lipstick",
    "Archery":        "a person doing archery",
    "BabyCrawling":   "a baby crawling",
    "BalanceBeam":    "a person on a balance beam",
    "BandMarching":   "a marching band performing",
    "BaseballPitch":  "a baseball pitch being thrown",
    "Basketball":     "a person playing basketball",
    "BasketballDunk": "a basketball dunk",
    "BenchPress":     "a person doing a bench press",
    "Biking":         "a person biking",
    "Billiards":      "a person playing billiards",
    "BlowDryHair":    "a person blow-drying their hair",
    "BlowingCandles": "a person blowing out candles",
    "BodyWeightSquats": "a person doing bodyweight squats",
    "Bowling":        "a person bowling",
    "BoxingPunchingBag": "a person punching a heavy bag",
    "BoxingSpeedBag": "a person hitting a speed bag",
    "BreastStroke":   "a person swimming breaststroke",
    "BrushingTeeth":  "a person brushing their teeth",
    "CleanAndJerk":   "a clean-and-jerk weightlifting lift",
    "CliffDiving":    "a person cliff diving",
    "CricketBowling": "a cricket bowler bowling",
    "CricketShot":    "a cricket batter playing a shot",
    "CuttingInKitchen": "a person cutting food in a kitchen",
    "Diving":         "a person diving",
    "Drumming":       "a person drumming",
    "Fencing":        "two people fencing",
    "FieldHockeyPenalty": "a field hockey penalty",
    "FloorGymnastics": "a person doing floor gymnastics",
    "FrisbeeCatch":   "a person catching a frisbee",
    "FrontCrawl":     "a person swimming front crawl",
    "GolfSwing":      "a person taking a golf swing",
    "Haircut":        "a person getting a haircut",
    "HammerThrow":    "a hammer throw event",
    "Hammering":      "a person hammering",
    "HandstandPushups": "a person doing handstand pushups",
    "HandstandWalking": "a person walking on their hands",
    "HeadMassage":    "a person getting a head massage",
    "HighJump":       "a person performing a high jump",
    "HorseRace":      "a horse race",
    "HorseRiding":    "a person riding a horse",
    "HulaHoop":       "a person using a hula hoop",
    "IceDancing":     "a couple ice dancing",
    "JavelinThrow":   "a javelin throw",
    "JugglingBalls":  "a person juggling balls",
    "JumpRope":       "a person jumping rope",
    "JumpingJack":    "a person doing jumping jacks",
    "Kayaking":       "a person kayaking",
    "Knitting":       "a person knitting",
    "LongJump":       "a person doing a long jump",
    "Lunges":         "a person doing lunges",
    "MilitaryParade": "a military parade",
    "Mixing":         "a person mixing batter in a bowl",
    "MoppingFloor":   "a person mopping a floor",
    "Nunchucks":      "a person twirling nunchucks",
    "ParallelBars":   "a gymnast on parallel bars",
    "PizzaTossing":   "a person tossing pizza dough",
    "PlayingCello":   "a person playing cello",
    "PlayingDaf":     "a person playing a daf drum",
    "PlayingDhol":    "a person playing a dhol drum",
    "PlayingFlute":   "a person playing flute",
    "PlayingGuitar":  "a person playing guitar",
    "PlayingPiano":   "a person playing piano",
    "PlayingSitar":   "a person playing sitar",
    "PlayingTabla":   "a person playing tabla",
    "PlayingViolin":  "a person playing violin",
    "PoleVault":      "a person performing a pole vault",
    "PommelHorse":    "a gymnast on the pommel horse",
    "PullUps":        "a person doing pull-ups",
    "Punch":          "a person throwing a punch",
    "PushUps":        "a person doing pushups",
    "Rafting":        "people rafting on whitewater",
    "RockClimbingIndoor": "a person rock climbing indoors",
    "RopeClimbing":   "a person rope climbing",
    "Rowing":         "a person rowing a boat",
    "SalsaSpin":      "two people doing a salsa spin",
    "ShavingBeard":   "a person shaving their beard",
    "Shotput":        "a person putting the shot",
    "SkateBoarding":  "a person skateboarding",
    "Skiing":         "a person skiing",
    "Skijet":         "a person on a jet ski",
    "SkyDiving":      "a person skydiving",
    "SoccerJuggling": "a person juggling a soccer ball",
    "SoccerPenalty":  "a soccer penalty kick",
    "StillRings":     "a gymnast on still rings",
    "SumoWrestling":  "two sumo wrestlers competing",
    "Surfing":        "a person surfing",
    "Swing":          "a child on a swing",
    "TableTennisShot": "a table tennis shot",
    "TaiChi":         "a person doing tai chi",
    "TennisSwing":    "a person taking a tennis swing",
    "ThrowDiscus":    "a person throwing a discus",
    "TrampolineJumping": "a person jumping on a trampoline",
    "Typing":         "a person typing on a keyboard",
    "UnevenBars":     "a gymnast on uneven bars",
    "VolleyballSpiking": "a volleyball player spiking the ball",
    "WalkingWithDog": "a person walking with a dog",
    "WallPushups":    "a person doing wall pushups",
    "WritingOnBoard": "a person writing on a board",
    "YoYo":           "a person playing with a yo-yo",
}


def _camel_split(name: str) -> str:
    out_chars: List[str] = []
    for i, ch in enumerate(name):
        if i > 0 and ch.isupper() and (not name[i - 1].isupper() or
                                       (i + 1 < len(name) and name[i + 1].islower())):
            out_chars.append(" ")
        out_chars.append(ch.lower())
    return "".join(out_chars)


def _humanise_class(class_name: str) -> str:
    override = _UCF_CAPTION_OVERRIDES.get(class_name)
    if override:
        return override
    return f"a person performing {_camel_split(class_name)}"


# ============================================================================
# Source acquisition (unchanged from v1)
# ============================================================================

def _download_ucf_rar(rar_source: Path) -> Path:
    rar_source.mkdir(parents=True, exist_ok=True)
    rar_path = rar_source / "UCF101.rar"
    if rar_path.exists() and rar_path.stat().st_size >= UCF_RAR_EXPECTED_SIZE_MIN:
        print(f"  UCF101.rar already present: {rar_path} "
              f"({rar_path.stat().st_size / 1024 / 1024:.0f} MB)")
        return rar_path

    print(f"  Downloading UCF101.rar from {UCF_RAR_URL}")
    print(f"    -> {rar_path}")
    if shutil.which("wget"):
        cmd = ["wget", "-c", "-O", str(rar_path), UCF_RAR_URL]
    elif shutil.which("curl"):
        cmd = ["curl", "-L", "-C", "-", "-o", str(rar_path), UCF_RAR_URL]
    else:
        print("ERROR: neither wget nor curl is available", file=sys.stderr)
        sys.exit(2)

    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"ERROR: download exited {rc}", file=sys.stderr)
        sys.exit(rc)
    if not rar_path.exists() or rar_path.stat().st_size < UCF_RAR_EXPECTED_SIZE_MIN:
        print(f"ERROR: downloaded file is too small ("
              f"{rar_path.stat().st_size if rar_path.exists() else 0} bytes); "
              f"expected >= {UCF_RAR_EXPECTED_SIZE_MIN}", file=sys.stderr)
        sys.exit(2)
    return rar_path


def _detect_extractor() -> List[str]:
    if shutil.which("unrar"):
        return ["unrar", "x", "-o+"]
    if shutil.which("7z"):
        return ["7z", "x", "-y"]
    if shutil.which("bsdtar"):
        return ["bsdtar", "-xf"]
    print("ERROR: no RAR extractor found (tried unrar, 7z, bsdtar). "
          "Install one via `conda install -c conda-forge unrar` and retry.",
          file=sys.stderr)
    sys.exit(2)


def _extract_ucf_rar(rar_path: Path, extract_dir: Path) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    ucf_root = extract_dir / "UCF-101"
    if ucf_root.is_dir() and any(ucf_root.iterdir()):
        n_classes = sum(1 for p in ucf_root.iterdir() if p.is_dir())
        print(f"  UCF-101 already extracted at {ucf_root} "
              f"({n_classes} class directories)")
        return ucf_root

    extractor = _detect_extractor()
    print(f"  Extracting {rar_path.name} with: {' '.join(extractor)}")
    if extractor[0] == "bsdtar":
        cmd = extractor + [str(rar_path), "-C", str(extract_dir)]
    else:
        cmd = extractor + [str(rar_path)]
    rc = subprocess.run(cmd, cwd=str(extract_dir)).returncode
    if rc != 0:
        print(f"ERROR: extraction exited {rc}", file=sys.stderr)
        sys.exit(rc)
    if not ucf_root.is_dir():
        for cand in extract_dir.iterdir():
            if cand.is_dir() and any(
                p.is_dir() and re.match(r"^[A-Za-z]", p.name)
                for p in cand.iterdir()
            ):
                ucf_root = cand
                break
        if not ucf_root.is_dir():
            print(f"ERROR: could not find UCF-101 root after extraction",
                  file=sys.stderr)
            sys.exit(2)
    return ucf_root


# ============================================================================
# Chunk planning + ffmpeg invocation
# ============================================================================

def _probe_duration(path: Path) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=15)
        return float(out.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return 0.0


def _plan_chunk_starts(
    duration_sec: float,
    chunk_seconds: float,
    chunk_gap: float,
    max_chunks: int,
    min_chunk_seconds: float = MIN_CHUNK_SECONDS,
) -> List[float]:
    """Return list of chunk start times (seconds). Each chunk consumes
    `chunk_seconds` of content (or up to clip-end for the last chunk),
    separated from the next by `chunk_gap` of unused content.

    Returns [] if the source is too short for even one usable chunk
    (< min_chunk_seconds duration).
    """
    if duration_sec < min_chunk_seconds:
        return []
    stride = chunk_seconds + chunk_gap
    out: List[float] = []
    t = 0.0
    while len(out) < max_chunks:
        if t + min_chunk_seconds > duration_sec + 1e-6:
            break
        out.append(t)
        t += stride
    return out


def _encode_one_chunk(
    src: Path,
    dst: Path,
    start_sec: float,
    chunk_seconds: float,
    target_w: int = TARGET_WIDTH,
    target_h: int = TARGET_HEIGHT,
    crf: int = 23,
    preset: str = "fast",
) -> bool:
    """Encode a single chunk. Idempotent on dst existing + > threshold."""
    if dst.exists() and dst.stat().st_size > RESIZE_SIZE_THRESHOLD_BYTES:
        return True
    tmp = dst.with_suffix(".chunk.mp4")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-nostdin",
        "-ss", f"{start_sec:.3f}",
        "-i", str(src),
        "-t", f"{chunk_seconds:.3f}",
        "-vf", f"scale={target_w}:{target_h}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-an",
        str(tmp),
    ]
    try:
        res = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False
    if res.returncode != 0 or not tmp.exists() or \
            tmp.stat().st_size <= RESIZE_SIZE_THRESHOLD_BYTES:
        tmp.unlink(missing_ok=True)
        return False
    try:
        os.replace(str(tmp), str(dst))
    except OSError:
        tmp.unlink(missing_ok=True)
        return False
    return True


def _process_source(
    class_name: str,
    src: Path,
    new_videos: Path,
    chunk_seconds: float,
    chunk_gap: float,
    max_chunks_per_source: int,
    chunking_enabled: bool,
) -> List[Dict]:
    """Process one UCF source clip into 1 or more chunks. Returns list of
    chunk records: each is a dict with chunk_video_id, source_video_id,
    class_name, dst_path, chunk_index, start_sec, duration_sec.
    Empty list if the source was too short."""
    source_video_id = src.stem  # v_<Class>_g<NN>_c<NN>
    duration = _probe_duration(src)
    if duration <= 0.0:
        return []

    if not chunking_enabled:
        # Legacy single-clip mode: emit one chunk covering the whole clip.
        starts = [0.0]
        per_chunk_seconds = min(duration, 30.0)  # cap at 30s for sanity
    else:
        starts = _plan_chunk_starts(
            duration, chunk_seconds, chunk_gap, max_chunks_per_source,
        )
        per_chunk_seconds = chunk_seconds

    if not starts:
        return []

    results: List[Dict] = []
    for t_idx, start_sec in enumerate(starts):
        if chunking_enabled:
            chunk_video_id = f"{source_video_id}_t{t_idx}"
            chunk_filename = f"{chunk_video_id}.mp4"
        else:
            chunk_video_id = source_video_id
            chunk_filename = f"{source_video_id}.mp4"
        dst = new_videos / chunk_filename
        ok = _encode_one_chunk(
            src=src, dst=dst,
            start_sec=start_sec, chunk_seconds=per_chunk_seconds,
        )
        if not ok:
            continue
        results.append({
            "chunk_video_id": chunk_video_id,
            "source_video_id": source_video_id,
            "class_name": class_name,
            "dst": dst,
            "chunk_index": t_idx,
            "start_sec": start_sec,
            "duration_sec": per_chunk_seconds,
        })
    return results


# ============================================================================
# Metadata probing of finished chunks
# ============================================================================

def _probe_chunk_stream(path: Path) -> Tuple[float, float, int, int, int]:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=15)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return (0.0, 0.0, 0, 0, 0)
    kv: Dict[str, str] = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()
    w = int(kv.get("width") or 0)
    h = int(kv.get("height") or 0)
    fps_raw = kv.get("r_frame_rate", "0/1")
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 0.0
    except (ValueError, ZeroDivisionError):
        fps = 0.0
    try:
        duration = float(kv.get("duration", "0") or 0.0)
    except ValueError:
        duration = 0.0
    try:
        nb_frames = int(kv.get("nb_frames", "0") or 0)
    except ValueError:
        nb_frames = 0
    if nb_frames == 0 and fps > 0 and duration > 0:
        nb_frames = int(round(fps * duration))
    return (duration, fps, nb_frames, w, h)


# ============================================================================
# Main build
# ============================================================================

def _list_source_clips(ucf_root: Path) -> List[Tuple[str, Path]]:
    pairs: List[Tuple[str, Path]] = []
    for class_dir in sorted(ucf_root.iterdir()):
        if not class_dir.is_dir():
            continue
        for avi in sorted(class_dir.glob("v_*.avi")):
            pairs.append((class_dir.name, avi))
    return pairs


def _build_pool(
    ucf_root: Path,
    new_dir: Path,
    chunk_seconds: float,
    chunk_gap: float,
    max_chunks_per_source: int,
    chunking_enabled: bool,
    num_workers: int,
) -> int:
    new_videos = new_dir / "videos"
    new_videos.mkdir(parents=True, exist_ok=True)
    manifest_path = new_dir / "manifest.jsonl"

    pairs = _list_source_clips(ucf_root)
    print(f"  Found {len(pairs)} UCF source clips under {ucf_root}")
    if chunking_enabled:
        print(f"  Chunking: chunk={chunk_seconds}s, gap={chunk_gap}s, "
              f"max={max_chunks_per_source}/source")
    else:
        print(f"  Chunking: DISABLED (one clip per source, legacy v1 mode)")

    # Existing source_video_ids in manifest -> skip those sources entirely.
    existing_sources: Set[str] = set()
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = entry.get("source_video_id")
                if not sid:
                    vid = str(entry.get("videoID", ""))
                    m = re.match(r"^(.+)_t\d+$", vid)
                    sid = m.group(1) if m else vid
                if sid:
                    existing_sources.add(sid)
        print(f"  Skipping {len(existing_sources)} sources already processed")

    todo = [(cn, src) for cn, src in pairs if src.stem not in existing_sources]
    print(f"  {len(todo)} sources to process")
    if not todo:
        return 0

    # Parallel chunk encoding (one worker = one source -> 1..N chunks).
    t_start = time.time()
    last_print = t_start
    n_done = 0
    n_chunks_total = 0
    n_sources_with_chunks = 0
    all_chunk_records: List[Dict] = []
    with ThreadPoolExecutor(max_workers=num_workers,
                            thread_name_prefix="ucfchunk") as ex:
        futures = {
            ex.submit(
                _process_source, class_name, src, new_videos,
                chunk_seconds, chunk_gap, max_chunks_per_source,
                chunking_enabled,
            ): (class_name, src)
            for class_name, src in todo
        }
        for fut in as_completed(futures):
            class_name, src = futures[fut]
            try:
                records = fut.result()
            except Exception:
                records = []
            if records:
                n_sources_with_chunks += 1
                n_chunks_total += len(records)
                all_chunk_records.extend(records)
            n_done += 1
            now = time.time()
            if now - last_print >= 15.0 or n_done == len(todo):
                rate = n_done / max(1.0, now - t_start)
                eta = (len(todo) - n_done) / max(1e-6, rate)
                print(f"    [chunk] {n_done}/{len(todo)} sources, "
                      f"{n_chunks_total} chunks, "
                      f"{rate:.1f} sources/s, eta {eta/60:.1f} min",
                      flush=True)
                last_print = now

    print(f"  Chunk summary: {n_sources_with_chunks}/{len(todo)} sources "
          f"yielded >=1 chunk; total {n_chunks_total} chunks; "
          f"elapsed {(time.time() - t_start)/60:.1f} min")

    # ------------------------------------------------------------------
    # Build the full per-chunk metadata + manifest. We rebuild metadata.csv
    # from in-memory state every time (cheap) so columns are deterministic.
    # ------------------------------------------------------------------
    metadata_rows: List[Dict[str, str]] = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                metadata_rows.append({
                    "index": str(e.get("index", "")),
                    "filename": str(e.get("filename", "")),
                    "videoID": str(e.get("videoID", "")),
                    "source_video_id": str(e.get("source_video_id", "")
                                           or e.get("videoID", "")),
                    "video_path": str(e.get("path", "") or e.get("video_path", "")),
                    "caption": str(e.get("caption", "")),
                    "category": str(e.get("category", "")),
                    "chunk_index": str(e.get("chunk_index", "0")),
                    "chunk_start_sec": str(e.get("chunk_start_sec", "0.0")),
                    "duration": str(e.get("duration", "")),
                    "fps": str(e.get("fps", "")),
                    "frames": str(e.get("frames", "")),
                    "width": str(e.get("width", "")),
                    "height": str(e.get("height", "")),
                })

    next_index = len(metadata_rows)
    n_appended = 0
    with open(manifest_path, "a", encoding="utf-8") as f_man:
        for rec in all_chunk_records:
            dst = rec["dst"]
            if not dst.exists():
                continue
            duration, fps, n_frames, width, height = _probe_chunk_stream(dst)
            caption = _humanise_class(rec["class_name"])
            row = {
                "index": str(next_index),
                "filename": dst.name,
                "videoID": rec["chunk_video_id"],
                "source_video_id": rec["source_video_id"],
                "video_path": str(dst),
                "caption": caption,
                "category": rec["class_name"],
                "chunk_index": str(rec["chunk_index"]),
                "chunk_start_sec": f"{rec['start_sec']:.3f}",
                "duration": f"{duration:.3f}",
                "fps": f"{fps:.3f}",
                "frames": str(n_frames),
                "width": str(width),
                "height": str(height),
            }
            metadata_rows.append(row)
            f_man.write(json.dumps({
                "index": next_index,
                "filename": dst.name,
                "videoID": rec["chunk_video_id"],
                "source_video_id": rec["source_video_id"],
                "caption": caption,
                "category": rec["class_name"],
                "chunk_index": rec["chunk_index"],
                "chunk_start_sec": float(f"{rec['start_sec']:.3f}"),
                "duration": float(f"{duration:.3f}"),
                "fps": float(f"{fps:.3f}"),
                "frames": n_frames,
                "width": width,
                "height": height,
                "path": str(dst),
            }) + "\n")
            next_index += 1
            n_appended += 1

    meta_csv_path = new_dir / "metadata.csv"
    fieldnames = [
        "index", "filename", "videoID", "source_video_id", "video_path",
        "caption", "category", "chunk_index", "chunk_start_sec",
        "duration", "fps", "frames", "width", "height",
    ]
    with open(meta_csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    print(f"  Wrote {len(metadata_rows)} rows to {meta_csv_path} "
          f"({n_appended} newly appended)")
    return len(metadata_rows)


# ============================================================================
# Validate + finalize (unchanged from v1)
# ============================================================================

def _run_validate(new_dir: Path, required_valid: int, min_frames: int) -> int:
    validate_script = REPO_ROOT / "scripts" / "validate_dataset.py"
    if not validate_script.exists():
        print(f"  validate_dataset.py not found; skipping validation step")
        return 0
    cmd = [
        sys.executable, str(validate_script),
        "--dataset-dir", str(new_dir),
        "--required-valid", str(required_valid),
        "--min-frames", str(min_frames),
        "--write-valid-subset", "valid_subset.csv",
        "--no-require-category",
    ]
    print("\nInvoking validator:")
    print("  " + " ".join(cmd))
    return int(subprocess.run(cmd).returncode)


def _finalize_metadata_to_n(new_dir: Path, target_valid: int) -> int:
    valid_path = new_dir / "valid_subset.csv"
    if not valid_path.exists():
        meta_path = new_dir / "metadata.csv"
        if meta_path.exists():
            n = 0
            with open(meta_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                n = sum(1 for _ in reader)
            print(f"\n(No valid_subset.csv; keeping metadata.csv as-is "
                  f"with {n} rows)")
            return n
        return -1
    with open(valid_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if len(rows) < target_valid:
        print(f"\nNote: only {len(rows)} valid rows available, fewer than "
              f"target {target_valid}. Keeping all available valid rows "
              f"(UCF-101 is dataset-bounded, so this is expected).")
        kept = rows
    else:
        kept = rows[:target_valid]
    final_path = new_dir / "metadata.csv"
    with open(final_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    print(f"\nFinal metadata.csv written ({len(kept)} rows) -> {final_path}")
    return len(kept)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--new-dataset", type=str, required=True)
    parser.add_argument("--rar-source", type=str, required=True,
                        help="Directory holding UCF101.rar + extracted UCF-101/")
    parser.add_argument("--target-valid", type=int, default=30000,
                        help="Cap pool at this many valid rows (default 30000 "
                             "for the chunked pool; ~12000 covers the legacy "
                             "1-clip-per-source mode)")
    parser.add_argument("--min-frames", type=int, default=48,
                        help="Validator min-frames threshold (default 48; "
                             "matches tta_total_frames)")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--chunk-seconds", type=float,
                        default=CHUNK_SECONDS_DEFAULT,
                        help=f"Chunk length in seconds (default "
                             f"{CHUNK_SECONDS_DEFAULT})")
    parser.add_argument("--chunk-gap", type=float, default=CHUNK_GAP_DEFAULT,
                        help=f"Gap between consecutive chunks of the same "
                             f"source, in seconds (default {CHUNK_GAP_DEFAULT})")
    parser.add_argument("--max-chunks-per-source", type=int,
                        default=MAX_CHUNKS_PER_SOURCE_DEFAULT,
                        help=f"Hard cap on chunks emitted per source video "
                             f"(default {MAX_CHUNKS_PER_SOURCE_DEFAULT}; "
                             f"prevents a small number of long clips from "
                             f"dominating the pool)")
    parser.add_argument("--no-chunking", action="store_true",
                        help="Disable chunking; emit one clip per source "
                             "(legacy v1 mode, ~12K pool)")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-resize", action="store_true",
                        help="Skip chunk encoding (use after a partial run)")
    parser.add_argument("--skip-validate", action="store_true")
    args = parser.parse_args()

    new_dir = Path(args.new_dataset).resolve()
    rar_source = Path(args.rar_source).resolve()
    new_dir.mkdir(parents=True, exist_ok=True)

    chunking_enabled = not args.no_chunking

    print("=" * 70)
    print("Build UCF-101 retrieval pool"
          + (" (CHUNKED)" if chunking_enabled else " (legacy 1-clip-per-source)"))
    print("=" * 70)
    print(f"  new dir          : {new_dir}")
    print(f"  rar source       : {rar_source}")
    print(f"  target valid     : {args.target_valid}")
    print(f"  resize workers   : {args.num_workers}")
    if chunking_enabled:
        print(f"  chunk seconds    : {args.chunk_seconds}")
        print(f"  chunk gap        : {args.chunk_gap}")
        print(f"  max chunks/src   : {args.max_chunks_per_source}")
    print()

    t0 = time.time()

    if not args.skip_download:
        rar_path = _download_ucf_rar(rar_source)
        ucf_root = _extract_ucf_rar(rar_path, rar_source)
    else:
        ucf_root = rar_source / "UCF-101"
        if not ucf_root.is_dir():
            print(f"ERROR: --skip-download set but {ucf_root} does not exist",
                  file=sys.stderr)
            return 2

    if not args.skip_resize:
        _build_pool(
            ucf_root=ucf_root,
            new_dir=new_dir,
            chunk_seconds=args.chunk_seconds,
            chunk_gap=args.chunk_gap,
            max_chunks_per_source=args.max_chunks_per_source,
            chunking_enabled=chunking_enabled,
            num_workers=args.num_workers,
        )

    if not args.skip_validate:
        _run_validate(
            new_dir=new_dir,
            required_valid=args.target_valid,
            min_frames=args.min_frames,
        )

    n_final = _finalize_metadata_to_n(new_dir, args.target_valid)

    elapsed = time.time() - t0
    print()
    print("=" * 70)
    print(f"DONE in {elapsed/60:.1f} min")
    print("=" * 70)
    print(f"  pool       : {new_dir}")
    print(f"  metadata   : {new_dir / 'metadata.csv'} ({n_final} rows)")
    print()
    print("Next step: pre-compute caption embeddings.")
    print(f"  sbatch --account=torch_pr_36_mren \\")
    print(f"      --export=ALL,POOL_DIR={new_dir} \\")
    print(f"      delta_experiment/sbatch/precompute_pool_embeddings.sbatch")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

BUILD_UCF101_SBATCH = r'''#!/bin/bash
#SBATCH --job-name=build_ucf101_pool_max
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=datasets/slurm_log/build_ucf101_pool_max_%j.out
#SBATCH --error=datasets/slurm_log/build_ucf101_pool_max_%j.err

# ============================================================================
# Build a CHUNKED UCF-101 retrieval pool (~28-32K chunks, ~50 GB).
#
# Why chunked: UCF-101 has only 13,320 source clips, so a 1-clip-per-source
# pool would top out at ~12K entries. The TTA pipeline only reads the first
# `tta_total_frames` (=48) of each pool clip (~1.92 s at 25 fps), so slicing
# each source into multiple 3-second non-overlapping chunks creates extra
# valid training segments without inventing fake samples. With default
# settings (chunk=3.0s, gap=0.5s, max=10/source) the pool reaches Panda
# 25K-pool parity.
#
# CPU + network only -- no GPU. ffprobe + ffmpeg are parallelised across
# CPU workers (default 8). Wall-time is generous (12h) to cover the initial
# RAR download + per-source probe + ~30K chunk encodes.
#
# Submit (default = chunking ENABLED, target 30K valid):
#   sbatch --account=torch_pr_36_mren datasets/build_ucf101_retrieval_pool.sbatch
#
# Resume (idempotent -- already-encoded chunks and already-manifested sources
# are skipped):
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,SKIP_DOWNLOAD=1 datasets/build_ucf101_retrieval_pool.sbatch
#
# Legacy 1-clip-per-source mode (~12K pool, no chunking):
#   sbatch --account=torch_pr_36_mren \
#       --export=ALL,CHUNKING=0,TARGET_VALID=12000 \
#       datasets/build_ucf101_retrieval_pool.sbatch
# ============================================================================

set -euo pipefail
export PYTHONNOUSERSITE=1

SCRATCH_BASE="/scratch/wc3013"
PROJECT_ROOT="${SCRATCH_BASE}/longcat-video-tta"
ENV_PATH="${SCRATCH_BASE}/conda-envs/longcat"

NEW="${NEW:-${PROJECT_ROOT}/datasets/ucf101_pool_max}"
RAR_SOURCE="${RAR_SOURCE:-${PROJECT_ROOT}/datasets/ucf101_source}"
TARGET_VALID="${TARGET_VALID:-30000}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Chunking knobs.
CHUNKING="${CHUNKING:-1}"
CHUNK_SECONDS="${CHUNK_SECONDS:-3.0}"
CHUNK_GAP="${CHUNK_GAP:-0.5}"
MAX_CHUNKS_PER_SOURCE="${MAX_CHUNKS_PER_SOURCE:-10}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_RESIZE="${SKIP_RESIZE:-0}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"

export TMPDIR="${SCRATCH_BASE}/tmp"
export PIP_CACHE_DIR="${SCRATCH_BASE}/pip-cache"
mkdir -p "${TMPDIR}" "${PIP_CACHE_DIR}"

echo "=============================================================================="
echo "Build UCF-101 retrieval pool (CHUNKED max-size)"
echo "=============================================================================="
echo "  New              : ${NEW}"
echo "  Rar source       : ${RAR_SOURCE}"
echo "  Target valid     : ${TARGET_VALID}"
echo "  Resize workers   : ${NUM_WORKERS}"
echo "  Chunking enabled : ${CHUNKING}"
echo "  Chunk seconds    : ${CHUNK_SECONDS}"
echo "  Chunk gap        : ${CHUNK_GAP}"
echo "  Max chunks/src   : ${MAX_CHUNKS_PER_SOURCE}"
echo "  Skip download    : ${SKIP_DOWNLOAD}"
echo "  Skip resize      : ${SKIP_RESIZE}"
echo "  Skip validate    : ${SKIP_VALIDATE}"
echo "=============================================================================="

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"
unset PYTHONHOME
unset PYTHONPATH
export PATH="${ENV_PATH}/bin:${PATH}"
PYTHON="${ENV_PATH}/bin/python"

# Tooling: ffmpeg + ffprobe + a rar extractor.
"${PYTHON}" -c "import av" 2>/dev/null || pip install av --quiet
if ! command -v ffmpeg &>/dev/null || ! command -v ffprobe &>/dev/null; then
    conda install -y -c conda-forge ffmpeg --quiet 2>/dev/null || true
fi
if ! command -v unrar &>/dev/null && ! command -v 7z &>/dev/null \
        && ! command -v bsdtar &>/dev/null; then
    echo "Installing unrar via conda..."
    conda install -y -c conda-forge unrar --quiet 2>/dev/null || true
fi
if ! command -v unrar &>/dev/null && ! command -v 7z &>/dev/null \
        && ! command -v bsdtar &>/dev/null; then
    echo "ERROR: no RAR extractor available (tried unrar, 7z, bsdtar). "
    echo "       Try: conda install -c conda-forge unrar"
    exit 2
fi

mkdir -p "${PROJECT_ROOT}/datasets/slurm_log"
cd "${PROJECT_ROOT}"

EXTRA_FLAGS=""
[ "${SKIP_DOWNLOAD}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-download"
[ "${SKIP_RESIZE}"   = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-resize"
[ "${SKIP_VALIDATE}" = "1" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --skip-validate"
[ "${CHUNKING}"      = "0" ] && EXTRA_FLAGS="${EXTRA_FLAGS} --no-chunking"

"${PYTHON}" scripts/build_ucf101_retrieval_pool.py \
    --new-dataset "${NEW}" \
    --rar-source "${RAR_SOURCE}" \
    --target-valid "${TARGET_VALID}" \
    --num-workers "${NUM_WORKERS}" \
    --chunk-seconds "${CHUNK_SECONDS}" \
    --chunk-gap "${CHUNK_GAP}" \
    --max-chunks-per-source "${MAX_CHUNKS_PER_SOURCE}" \
    ${EXTRA_FLAGS}

echo ""
echo "=============================================================================="
echo "UCF pool build job finished"
echo "  Dataset    : ${NEW}"
echo "  Videos     : $(ls ${NEW}/videos/*.mp4 2>/dev/null | wc -l)"
echo "  Du -sh     : $(du -sh ${NEW} 2>/dev/null)"
echo "  Report     : ${NEW}/validation_report.json"
echo "=============================================================================="
echo ""
echo "Next step: pre-compute caption embeddings"
echo "  sbatch --account=torch_pr_36_mren \\"
echo "      --export=ALL,POOL_DIR=${NEW} \\"
echo "      delta_experiment/sbatch/precompute_pool_embeddings.sbatch"
echo "=============================================================================="
'''


# ============================================================================
# common.py patches
# ============================================================================

#: Anchor for the `sample_neighbors_random` patch. Must match the FUNCTION
#: signature + body BEFORE any v2 modification.
COMMON_SAMPLE_OLD = '''def sample_neighbors_random(
    query_entry: Dict,
    pool_entries: List[Dict],
    k: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Return up to ``k-1`` *random* neighbours from the pool, excluding the query.

    This is the non-similarity counterpart to :func:`retrieve_neighbors`.
    Used when ``batch_method='random'`` to test whether having any
    additional clips in the training batch helps (variance reduction),
    independent of whether the neighbours are topically related.

    No caption-embedding model is required.
    """
    if k <= 1:
        return []
    if rng is None:
        rng = np.random.default_rng()

    query_path = os.path.abspath(query_entry.get("video_path", ""))
    candidates = [
        e for e in pool_entries
        if os.path.abspath(e.get("video_path", "")) != query_path
    ]
    n_pick = min(k - 1, len(candidates))
    if n_pick == 0:
        return []
    idx = rng.choice(len(candidates), size=n_pick, replace=False)
    return [candidates[int(i)] for i in idx]
'''

COMMON_SAMPLE_NEW = '''def _entry_source_id(entry: Dict) -> str:
    """Extract a *source*-video identifier from a pool/query entry.

    Prefers explicit ``source_video_id`` / ``source_videoID`` fields (written
    by the chunked UCF-101 retrieval pool, where multiple chunks share an
    underlying source clip). Falls back to ``videoID``, and finally to the
    filename stem with any trailing ``_t<digits>`` chunk suffix stripped.

    This means same-source exclusion works even when the pool's metadata.csv
    predates the chunked schema, as long as chunk videoIDs followed the
    ``<source>_t<N>`` naming convention. On legacy pools where neither
    `source_video_id` nor `_t<N>` videoIDs are present, this returns the
    plain `videoID`, so retrieval behaviour is unchanged.
    """
    for k in ("source_video_id", "source_videoID"):
        v = entry.get(k)
        if v:
            return str(v)
    vid = entry.get("videoID")
    if not vid:
        fn = entry.get("filename") or entry.get("video_path") or ""
        if fn:
            vid = Path(fn).stem
    if not vid:
        return ""
    import re as _re
    m = _re.match(r"^(.+)_t\\d+$", str(vid))
    return m.group(1) if m else str(vid)


def sample_neighbors_random(
    query_entry: Dict,
    pool_entries: List[Dict],
    k: int,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """Return up to ``k-1`` *random* neighbours from the pool, excluding the
    query and any pool entries sharing the same source-video ID.

    This is the non-similarity counterpart to :func:`retrieve_neighbors`.
    Used when ``batch_method='random'`` to test whether having any
    additional clips in the training batch helps (variance reduction),
    independent of whether the neighbours are topically related.

    Same-source exclusion is what prevents the chunked UCF-101 retrieval
    pool from returning neighbour chunks belonging to the *same* underlying
    source clip as the eval query.

    No caption-embedding model is required.
    """
    if k <= 1:
        return []
    if rng is None:
        rng = np.random.default_rng()

    query_path = os.path.abspath(query_entry.get("video_path", ""))
    query_source = _entry_source_id(query_entry)
    candidates: List[Dict] = []
    for e in pool_entries:
        if os.path.abspath(e.get("video_path", "")) == query_path:
            continue
        if query_source and _entry_source_id(e) == query_source:
            continue
        candidates.append(e)
    n_pick = min(k - 1, len(candidates))
    if n_pick == 0:
        return []
    idx = rng.choice(len(candidates), size=n_pick, replace=False)
    return [candidates[int(i)] for i in idx]
'''

#: Anchor for `retrieve_neighbors`.
COMMON_RETRIEVE_OLD = '''def retrieve_neighbors(
    query_entry: Dict,
    pool_entries: List[Dict],
    pool_embeddings: np.ndarray,
    st_model: Any,
    k: int,
) -> List[Dict]:
    """Return up to *k-1* nearest neighbours from the pool for *query_entry*.

    The intended batch for TTA training is ``[query_entry] + neighbours`` so
    the total size is *k*.  When *k* <= 1, an empty list is returned.

    Matching is by cosine similarity of text-prompt embeddings.  The query
    video itself is excluded from the returned list (matched by absolute path).
    """
    if k <= 1:
        return []

    query_emb = st_model.encode(
        [query_entry.get("caption", "")], normalize_embeddings=True,
    )
    sims = (pool_embeddings @ query_emb.T).squeeze()

    query_path = os.path.abspath(query_entry.get("video_path", ""))
    ranked = np.argsort(-sims)

    neighbors: List[Dict] = []
    for idx in ranked:
        if len(neighbors) >= k - 1:
            break
        pool_path = os.path.abspath(pool_entries[int(idx)].get("video_path", ""))
        if pool_path == query_path:
            continue
        neighbors.append(pool_entries[int(idx)])

    return neighbors
'''

COMMON_RETRIEVE_NEW = '''def retrieve_neighbors(
    query_entry: Dict,
    pool_entries: List[Dict],
    pool_embeddings: np.ndarray,
    st_model: Any,
    k: int,
) -> List[Dict]:
    """Return up to *k-1* nearest neighbours from the pool for *query_entry*.

    The intended batch for TTA training is ``[query_entry] + neighbours`` so
    the total size is *k*.  When *k* <= 1, an empty list is returned.

    Matching is by cosine similarity of text-prompt embeddings.  The query
    video itself is excluded from the returned list (matched by absolute
    path), and so are any pool entries sharing the same source-video ID
    (matched via :func:`_entry_source_id`). The source-ID check is what
    prevents the chunked UCF-101 retrieval pool from returning the query's
    own neighbouring chunks as nearest neighbours.
    """
    if k <= 1:
        return []

    query_emb = st_model.encode(
        [query_entry.get("caption", "")], normalize_embeddings=True,
    )
    sims = (pool_embeddings @ query_emb.T).squeeze()

    query_path = os.path.abspath(query_entry.get("video_path", ""))
    query_source = _entry_source_id(query_entry)
    ranked = np.argsort(-sims)

    neighbors: List[Dict] = []
    for idx in ranked:
        if len(neighbors) >= k - 1:
            break
        pool_entry = pool_entries[int(idx)]
        pool_path = os.path.abspath(pool_entry.get("video_path", ""))
        if pool_path == query_path:
            continue
        if query_source and _entry_source_id(pool_entry) == query_source:
            continue
        neighbors.append(pool_entry)

    return neighbors
'''


# ============================================================================
# Main
# ============================================================================

PATCHES: List[Dict[str, str]] = [
    {
        "path": "delta_experiment/scripts/common.py",
        "label": "common.py: insert _entry_source_id helper + patch sample_neighbors_random",
        "old": COMMON_SAMPLE_OLD,
        "new": COMMON_SAMPLE_NEW,
    },
    {
        "path": "delta_experiment/scripts/common.py",
        "label": "common.py: patch retrieve_neighbors (same-source exclusion)",
        "old": COMMON_RETRIEVE_OLD,
        "new": COMMON_RETRIEVE_NEW,
    },
]

NEW_FILES: List[Tuple[str, str]] = [
    ("scripts/build_ucf101_retrieval_pool.py", BUILD_UCF101_PY),
    ("datasets/build_ucf101_retrieval_pool.sbatch", BUILD_UCF101_SBATCH),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Deploy UCF chunking upgrade (v2 delta).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually write files / apply patches (default: dry-run)."
    )
    parser.add_argument(
        "--repo-root", type=str, default=None,
        help="Repository root (default: parent of this script's directory)."
    )
    args = parser.parse_args()

    here = Path(__file__).resolve()
    repo_root = Path(args.repo_root).resolve() if args.repo_root \
        else here.parent.parent

    print("=" * 72)
    print("deploy_pool_extensions_v2.py")
    print("=" * 72)
    print(f"  repo root : {repo_root}")
    print(f"  mode      : {'APPLY' if args.apply else 'dry-run'}")
    print()
    print(f"Plan: write {len(NEW_FILES)} files, apply {len(PATCHES)} patches.")
    print()

    print("Files:")
    for rel, content in NEW_FILES:
        print(write_file(repo_root / rel, content,
                         apply=args.apply, label=rel))
    print()

    print("Patches:")
    n_err = 0
    for patch in PATCHES:
        msg = apply_patch(
            repo_root / patch["path"], patch["old"], patch["new"],
            apply=args.apply, label=patch["label"],
        )
        if msg.lstrip().startswith("[ERROR]"):
            n_err += 1
        print(msg)
    print()

    print("=" * 72)
    if n_err:
        print(f"FAILED: {n_err} patch(es) could not be applied.")
        return 2
    if args.apply:
        print("OK -- changes applied. Next steps:")
        print()
        print("  sbatch --account=torch_pr_36_mren \\")
        print("      datasets/build_ucf101_retrieval_pool.sbatch")
        print()
        print("After the build finishes (~2-3 h):")
        print()
        print("  sbatch --account=torch_pr_36_mren \\")
        print("      --export=ALL,POOL_DIR=/scratch/wc3013/longcat-video-tta/"
              "datasets/ucf101_pool_max \\")
        print("      delta_experiment/sbatch/precompute_pool_embeddings.sbatch")
    else:
        print("OK -- dry-run clean. Re-run with --apply to write changes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
