#!/usr/bin/env python3
"""Official VBench quality scoring on a flat Wan I2V mp4 directory.

The controller loop stays GT-free. This script is the *outcome* scorecard:
do-nothing / always-search / gated-search mp4s are judged with the same
VBench dimensions used on the Panda 1000v tables.

Does NOT compute PSNR/SSIM/LPIPS. The 32 VBench-I2V stills have no paired
30 s ground-truth video.

Must run in the vbench-backfill env (numpy 1.x / torch 2.5), not
self_forcing and not longcat.

    python wan_experiment/scripts/score_i2v_vbench.py \
        --video-dir wan_experiment/results/i2v_bon_32v_hybrid/notta_h30s_shard0 \
        --clip full \
        --out-dir wan_experiment/results/i2v_bon_32v_hybrid/notta_h30s_shard0/vbench_full
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

WINDOW_RE = re.compile(r"^w(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$")


QUALITY_DIMS = [
    "subject_consistency",
    "background_consistency",
    "aesthetic_quality",
    "imaging_quality",
    "motion_smoothness",
    "dynamic_degree",
    "temporal_flickering",
]

I2V_DIMS = ["i2v_subject", "i2v_background"]


def _find_full_info_json() -> str:
    import vbench as _v

    pkg_dir = os.path.dirname(_v.__file__)
    candidates = [
        os.path.join(pkg_dir, "VBench_full_info.json"),
        os.path.join(os.path.dirname(pkg_dir), "vbench", "VBench_full_info.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "VBench_full_info.json not found alongside vbench package. "
        f"Tried: {candidates}"
    )


def _result_file(out_dir: Path, dim: str) -> Path:
    return out_dir / f"vbench_{dim}_eval_results.json"


def _mp4s(video_dir: Path) -> List[Path]:
    return sorted(p for p in video_dir.glob("*.mp4") if p.is_file())


def window_bounds(clip: str, tail_s: float) -> Tuple[str, Optional[float], Optional[float]]:
    """Return (kind, start_s, end_s). kind is full|tail|head|window."""
    if clip == "full":
        return "full", None, None
    if clip == "last5":
        return "tail", None, float(tail_s)
    if clip == "first5":
        return "head", 0.0, float(tail_s)
    m = WINDOW_RE.match(clip)
    if m:
        return "window", float(m.group(1)), float(m.group(2))
    raise ValueError(
        f"unknown clip {clip!r}; use full, last5, first5, or wSTART_END (e.g. w10_15)"
    )


def windows_for_horizon(horizon_s: float, window_s: float) -> List[str]:
    if window_s <= 0 or horizon_s <= 0:
        raise ValueError("horizon and window must be positive")
    n = max(1, int(round(horizon_s / window_s)))
    names = []
    for i in range(n):
        start = i * window_s
        end = (i + 1) * window_s
        names.append(f"w{int(start)}_{int(end)}")
    return names


def _slice_frames(frames, src_fps: float, clip: str, tail_s: float):
    kind, start_s, end_s = window_bounds(clip, tail_s)
    n_all = len(frames)
    if kind == "full":
        return frames
    if kind == "tail":
        n = max(1, int(round(float(end_s) * src_fps)))
        return frames[-n:]
    start_f = max(0, int(round(float(start_s) * src_fps)))
    end_f = min(n_all, int(round(float(end_s) * src_fps)))
    if kind == "window" and end_f >= n_all - 1:
        end_f = n_all
    sliced = frames[start_f:end_f]
    if not sliced:
        raise RuntimeError(f"{clip} empty on {n_all} frames @ {src_fps} fps")
    return sliced


def _extract_clip(src: Path, dst: Path, clip: str, fps: float, tail_s: float) -> None:
    import cv2

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {src}")
    frames = []
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"0 frames in {src}")

    src_fps = float(src_fps or fps)
    frames = _slice_frames(frames, src_fps, clip, tail_s)

    dst.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(dst),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(src_fps or fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"cannot write {dst}")
    for fr in frames:
        writer.write(fr)
    writer.release()


def ensure_clip_dir(video_dir: Path, clip: str, fps: float, tail_s: float) -> Path:
    if clip == "full":
        return video_dir
    out = video_dir / "vbench_clips" / clip
    out.mkdir(parents=True, exist_ok=True)
    srcs = _mp4s(video_dir)
    if not srcs:
        raise FileNotFoundError(f"no mp4 in {video_dir}")
    for src in srcs:
        dst = out / src.name
        if dst.is_file() and dst.stat().st_size > 10_000:
            continue
        print(f"  clip {clip}: {src.name} -> {dst.name}", flush=True)
        _extract_clip(src, dst, clip, fps=fps, tail_s=tail_s)
    return out


def _scalar(item) -> Optional[float]:
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        for k in ("video_results", "video_score", "score"):
            v = item.get(k)
            if isinstance(v, (int, float)):
                return float(v)
    return None


def extract_per_video(parsed: dict, dim: str) -> Dict[str, float]:
    """Map video stem -> scalar from a VBench 0.1.5 result json."""
    if not isinstance(parsed, dict):
        return {}
    body = parsed.get(dim)
    if body is None and len(parsed) == 1:
        body = next(iter(parsed.values()))
    if body is None:
        return {}

    items = body
    if isinstance(body, list) and body:
        if isinstance(body[0], (int, float)) and len(body) >= 2:
            items = body[1]
        else:
            items = body

    out: Dict[str, float] = {}
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            path = item.get("video_path") or item.get("video") or item.get("path")
            score = _scalar(item)
            if path is None or score is None:
                continue
            out[Path(str(path)).stem] = score
    elif isinstance(items, dict):
        for path, val in items.items():
            score = _scalar(val) if not isinstance(val, (int, float)) else float(val)
            if score is None:
                continue
            out[Path(str(path)).stem] = score
    return out


def _load_summary_rows(video_dir: Path) -> List[dict]:
    p = video_dir / "summary.json"
    if not p.is_file():
        return []
    data = json.loads(p.read_text())
    return [r for r in (data.get("rows") or []) if r.get("ok")]


def _row_stem(row: dict) -> str:
    mp4 = row.get("mp4")
    if mp4:
        return Path(str(mp4)).stem
    return str(row.get("stem") or row.get("file_name") or "")


def join_rows(
    rows: List[dict],
    per_dim: Dict[str, Dict[str, float]],
) -> List[dict]:
    joined = []
    for row in rows:
        stem = _row_stem(row)
        rec = {
            "stem": stem,
            "file_name": row.get("file_name"),
            "last_chunk_score": row.get("last_chunk_score"),
            "method": row.get("method"),
            "vbench": {},
        }
        for dim, by_stem in per_dim.items():
            if stem in by_stem:
                rec["vbench"][dim] = by_stem[stem]
                continue
            # VBench sometimes returns a basename without our suffix noise.
            hits = [v for k, v in by_stem.items() if stem in k or k in stem]
            if len(hits) == 1:
                rec["vbench"][dim] = hits[0]
        joined.append(rec)
    if joined:
        return joined
    stems = sorted({s for by in per_dim.values() for s in by})
    for stem in stems:
        rec = {"stem": stem, "file_name": None, "last_chunk_score": None,
               "method": None, "vbench": {}}
        for dim, by_stem in per_dim.items():
            if stem in by_stem:
                rec["vbench"][dim] = by_stem[stem]
        joined.append(rec)
    return joined


def _pop(values: List[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
    }


def score_dir(
    video_dir: Path,
    out_dir: Path,
    dimensions: List[str],
    clip: str,
    mode: str,
    force: bool,
    fps: float,
    tail_s: float,
) -> int:
    video_dir = video_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"video_dir : {video_dir}")
    print(f"out_dir   : {out_dir}")
    print(f"clip      : {clip}")
    print(f"dims      : {dimensions}")
    print(f"mode      : {mode}")

    score_dir_path = ensure_clip_dir(video_dir, clip, fps=fps, tail_s=tail_s)
    mp4s = _mp4s(score_dir_path)
    print(f"mp4s      : {len(mp4s)} in {score_dir_path}")
    if not mp4s:
        return 2

    import torch
    from vbench import VBench

    full_info = _find_full_info_json()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device    : {device}")
    print(f"full_info : {full_info}")
    vb = VBench(device, full_info, str(out_dir))

    n_run = n_skip = n_fail = 0
    failed = []
    for dim in dimensions:
        res_path = _result_file(out_dir, dim)
        if res_path.exists() and not force:
            print(f"  {dim}: SKIP (exists)")
            n_skip += 1
            continue
        print(f"  {dim}: running on {len(mp4s)} videos ...", end=" ", flush=True)
        t0 = time.time()
        try:
            vb.evaluate(
                videos_path=str(score_dir_path),
                name=f"vbench_{dim}",
                dimension_list=[dim],
                mode=mode,
            )
            dt = time.time() - t0
            if res_path.exists():
                print(f"OK ({dt:.1f}s)")
                n_run += 1
            else:
                print(f"WARN: no result file after {dt:.1f}s")
                n_fail += 1
                failed.append((dim, "no result file written"))
        except Exception as exc:
            dt = time.time() - t0
            print(f"FAIL ({dt:.1f}s): {type(exc).__name__}: {exc}")
            traceback.print_exc()
            n_fail += 1
            failed.append((dim, f"{type(exc).__name__}: {exc}"))

    per_dim: Dict[str, Dict[str, float]] = {}
    for dim in dimensions:
        res_path = _result_file(out_dir, dim)
        if not res_path.is_file():
            continue
        try:
            parsed = json.loads(res_path.read_text())
        except Exception as exc:
            print(f"  [warn] parse {res_path.name}: {exc}", file=sys.stderr)
            continue
        per_dim[dim] = extract_per_video(parsed, dim)

    rows = _load_summary_rows(video_dir)
    joined = join_rows(rows, per_dim)
    population = {
        dim: _pop([rec["vbench"][dim] for rec in joined if dim in rec["vbench"]])
        for dim in dimensions
    }
    summary = {
        "video_dir": str(video_dir),
        "score_dir": str(score_dir_path),
        "out_dir": str(out_dir),
        "clip": clip,
        "mode": mode,
        "n_mp4": len(mp4s),
        "n_joined": len(joined),
        "dimensions": dimensions,
        "population": population,
        "ran": n_run,
        "skipped": n_skip,
        "failed": n_fail,
        "failures": failed,
        "per_video": joined,
    }
    (out_dir / "joined.json").write_text(json.dumps(summary, indent=2))
    print()
    print("=" * 70)
    print(f"VBench {clip}  {video_dir.name}")
    print("=" * 70)
    for dim in dimensions:
        pop = population.get(dim) or {}
        print(f"  {dim:24s}  n={pop.get('n', 0):>3}  "
              f"mean={pop.get('mean')}  median={pop.get('median')}")
    print(f"  ran={n_run} skip={n_skip} fail={n_fail}")
    print(f"  wrote {out_dir / 'joined.json'}")
    return 0 if n_fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--clip", default="full",
        help="full (official VBench++), last5, first5, wSTART_END (e.g. w10_15), "
             "or windows (every --window-s seconds over --horizon-s).",
    )
    ap.add_argument("--horizon-s", type=float, default=30.0)
    ap.add_argument("--window-s", type=float, default=5.0)
    ap.add_argument("--tail-s", type=float, default=5.0)
    ap.add_argument("--fps", type=float, default=16.0)
    ap.add_argument("--dimensions", nargs="+", default=QUALITY_DIMS)
    ap.add_argument("--mode", default="custom_input",
                    choices=["custom_input", "i2v", "t2v"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    unknown = [d for d in args.dimensions if d in I2V_DIMS]
    if unknown:
        print(
            f"[warn] {unknown} need vbench2_beta_i2v + name-matched images; "
            "this script scores regular VBench quality dims. Dropping I2V dims.",
            file=sys.stderr,
        )
        args.dimensions = [d for d in args.dimensions if d not in I2V_DIMS]
    clips = (
        windows_for_horizon(args.horizon_s, args.window_s)
        if args.clip == "windows"
        else [args.clip]
    )
    status = 0
    for clip in clips:
        try:
            window_bounds(clip, args.tail_s)
        except ValueError as e:
            raise SystemExit(str(e))
        out_dir = args.out_dir if (args.out_dir and args.clip != "windows") else (
            args.video_dir / f"vbench_{clip}"
        )
        rc = score_dir(
            args.video_dir, out_dir, args.dimensions, clip,
            args.mode, args.force, args.fps, args.tail_s,
        )
        if rc != 0:
            status = rc
    return status


if __name__ == "__main__":
    sys.exit(main())
