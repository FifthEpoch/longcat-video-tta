#!/usr/bin/env python3
"""Add ``chunk_index`` (segment index) to Panda eval ``metadata.csv``.

Panda-70M rows store multi-segment captions as Python-literal list strings.
The download pipeline often writes the *full* list into ``metadata.csv`` without
recording which segment was actually downloaded.  This script:

  1. Loads ``manifest.jsonl`` from the original download dir (``panda_1000``)
     or the eval dir if present — entries carry ``videoID``, optional
     ``start``/``end``, and ``filename``.
  2. Optionally matches each clip to Panda-70M segment metadata (CSV) by
     ``videoID`` + start timestamp.
  3. Writes ``chunk_index`` (and optionally single-segment ``caption``) back
     into ``metadata.csv`` so ``scripts/caption_utils.py`` resolves the correct
     segment caption.

Usage on cluster:
    python scripts/patch_panda_metadata_segment_index.py \\
        --metadata datasets/panda_1000_480p/metadata.csv \\
        --manifest datasets/panda_1000/manifest.jsonl \\
        --panda-csv datasets/panda_pool_10k/panda70m_training_2m.csv \\
        --in-place

    # Dry-run first:
    python scripts/patch_panda_metadata_segment_index.py \\
        --metadata datasets/panda_1000_480p/metadata.csv \\
        --manifest datasets/panda_1000/manifest.jsonl \\
        --output datasets/panda_1000_480p/metadata_patched.csv
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.caption_utils import parse_caption_list, resolve_caption_for_clip

_TS_RE = re.compile(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$")


def _parse_timestamp(ts: Any) -> Optional[float]:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return float(ts)
    s = str(ts).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    m = _TS_RE.match(s)
    if m:
        h, m_, sec = m.groups()
        return int(h) * 3600 + int(m_) * 60 + float(sec)
    parts = s.split(":")
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
    except ValueError:
        pass
    return None


def _safe_literal_list(raw: str) -> Optional[list]:
    if not raw:
        return None
    try:
        v = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    return v if isinstance(v, list) else None


def load_manifest(path: Path) -> Dict[str, dict]:
    """Map ``filename`` and ``panda_XXXX`` stem -> manifest entry."""
    by_filename: Dict[str, dict] = {}
    if not path.exists():
        return by_filename
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            fname = obj.get("filename") or ""
            if fname:
                by_filename[fname] = obj
                by_filename[Path(fname).stem] = obj
    return by_filename


def load_panda_segment_index(csv_path: Path) -> Dict[str, List[Tuple[float, float, int]]]:
    """Map YouTube ``videoID`` -> [(start_sec, end_sec, seg_idx), ...]."""
    out: Dict[str, List[Tuple[float, float, int]]] = {}
    captions_by_vid: Dict[str, List[Tuple[int, str]]] = {}
    if not csv_path.exists():
        return out

    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    with csv_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("videoID") or ""
            if not vid or vid in out:
                continue
            timestamps = _safe_literal_list(row.get("timestamp", ""))
            cap_list = _safe_literal_list(row.get("caption", ""))
            if not timestamps:
                continue
            segs: List[Tuple[float, float, int]] = []
            cap_pairs: List[Tuple[int, str]] = []
            for idx, ts_pair in enumerate(timestamps):
                if not isinstance(ts_pair, (list, tuple)) or len(ts_pair) < 2:
                    continue
                start = _parse_timestamp(ts_pair[0])
                end = _parse_timestamp(ts_pair[1])
                if start is None or end is None:
                    continue
                segs.append((start, end, idx))
                if cap_list and idx < len(cap_list) and isinstance(cap_list[idx], str):
                    cap_pairs.append((idx, cap_list[idx].strip()))
            if segs:
                out[vid] = segs
            if cap_pairs:
                captions_by_vid[vid] = cap_pairs
    return out, captions_by_vid


def infer_segment_index(
    *,
    video_id: Optional[str],
    start_sec: Optional[float],
    end_sec: Optional[float],
    panda_segments: Dict[str, List[Tuple[float, float, int]]],
    panda_captions: Dict[str, List[Tuple[int, str]]],
    caption_first: Optional[str],
    tolerance: float = 1.5,
) -> Optional[int]:
    if not video_id or video_id not in panda_segments:
        return None
    segs = panda_segments[video_id]
    if start_sec is not None:
        best_idx = None
        best_dist = float("inf")
        for s, e, idx in segs:
            dist = abs(s - start_sec)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is not None and best_dist <= tolerance:
            return best_idx
    if end_sec is not None:
        for s, e, idx in segs:
            if abs(e - end_sec) <= tolerance:
                return idx
    if caption_first and video_id in panda_captions:
        matches = [
            idx for idx, cap in panda_captions[video_id]
            if cap == caption_first.strip()
        ]
        if len(matches) == 1:
            return matches[0]
    if len(segs) == 1:
        return segs[0][2]
    return None


def patch_metadata_rows(
    rows: List[dict],
    fieldnames: List[str],
    manifest: Dict[str, dict],
    panda_segments: Dict[str, List[Tuple[float, float, int]]],
    panda_captions: Dict[str, List[Tuple[int, str]]],
    *,
    rewrite_caption: bool,
    tolerance: float,
) -> Tuple[List[dict], int, int]:
    if "chunk_index" not in fieldnames:
        fieldnames = list(fieldnames) + ["chunk_index"]

    n_patched = 0
    n_already = 0
    for row in rows:
        existing = row.get("chunk_index") or row.get("segment_index")
        if existing not in (None, ""):
            try:
                int(existing)
                n_already += 1
                continue
            except (TypeError, ValueError):
                pass

        fname = row.get("filename") or ""
        manifest_entry = manifest.get(fname) or manifest.get(Path(fname).stem)
        video_id = (
            row.get("videoID")
            or (manifest_entry or {}).get("videoID")
        )
        start_sec = _parse_timestamp(
            row.get("start")
            or row.get("chunk_start_sec")
            or (manifest_entry or {}).get("start")
        )
        end_sec = _parse_timestamp(
            row.get("end")
            or (manifest_entry or {}).get("end")
        )

        caps = parse_caption_list(row.get("caption") or "")
        caption_first = caps[0] if caps else None

        seg_idx = infer_segment_index(
            video_id=video_id,
            start_sec=start_sec,
            end_sec=end_sec,
            panda_segments=panda_segments,
            panda_captions=panda_captions,
            caption_first=caption_first,
            tolerance=tolerance,
        )

        if seg_idx is None:
            if len(caps) == 1:
                seg_idx = 0
            else:
                continue

        row["chunk_index"] = str(seg_idx)
        n_patched += 1

        if rewrite_caption:
            raw = row.get("caption") or ""
            resolved = resolve_caption_for_clip(raw, segment_index=seg_idx)
            if resolved:
                row["caption"] = resolved

    return rows, n_patched, n_already


def main() -> int:
    ap = argparse.ArgumentParser(description="Patch Panda metadata chunk_index")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="manifest.jsonl (try datasets/panda_1000/manifest.jsonl on cluster)",
    )
    ap.add_argument(
        "--panda-csv",
        type=Path,
        default=None,
        help="Panda-70M segment CSV for timestamp matching",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--in-place", action="store_true",
        help="Overwrite --metadata (writes .bak backup first)",
    )
    ap.add_argument(
        "--rewrite-caption", action="store_true",
        help="Replace list-string caption with single segment caption",
    )
    ap.add_argument(
        "--tolerance-sec", type=float, default=1.5,
        help="Max |start - seg_start| for segment match (default 1.5s)",
    )
    args = ap.parse_args()

    if not args.metadata.exists():
        print(f"ERROR: {args.metadata} not found", file=sys.stderr)
        return 2

    manifest_path = args.manifest
    if manifest_path is None:
        for cand in (
            args.metadata.parent / "manifest.jsonl",
            _REPO_ROOT / "datasets/panda_1000/manifest.jsonl",
            _REPO_ROOT / "datasets/panda_1000_480p/manifest.jsonl",
        ):
            if cand.exists():
                manifest_path = cand
                break

    panda_csv = args.panda_csv
    if panda_csv is None:
        for cand in (
            _REPO_ROOT / "datasets/panda_pool_10k/panda70m_training_2m.csv",
            _REPO_ROOT / "datasets/panda70m_training_2m.csv",
        ):
            if cand.exists():
                panda_csv = cand
                break

    manifest = load_manifest(manifest_path) if manifest_path else {}
    panda_segments: Dict[str, List[Tuple[float, float, int]]] = {}
    panda_captions: Dict[str, List[Tuple[int, str]]] = {}
    if panda_csv:
        panda_segments, panda_captions = load_panda_segment_index(panda_csv)

    print(f"metadata : {args.metadata}")
    print(f"manifest : {manifest_path} ({len(manifest)} keys)" if manifest_path else "manifest : (none)")
    print(f"panda csv: {panda_csv} ({len(panda_segments)} videoIDs)" if panda_csv else "panda csv: (none)")

    with args.metadata.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    rows, n_patched, n_already = patch_metadata_rows(
        rows, fieldnames, manifest, panda_segments, panda_captions,
        rewrite_caption=args.rewrite_caption,
        tolerance=args.tolerance_sec,
    )

    out_path = args.output
    if args.in_place:
        backup = args.metadata.with_suffix(".csv.bak")
        if not backup.exists():
            backup.write_text(args.metadata.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"backup   : {backup}")
        out_path = args.metadata
    elif out_path is None:
        out_path = args.metadata.with_name("metadata_patched.csv")

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    n_with_idx = sum(
        1 for r in rows
        if str(r.get("chunk_index", "")).strip() != ""
    )
    print(f"patched  : {n_patched} rows newly assigned chunk_index")
    print(f"existing : {n_already} rows already had segment/chunk index")
    print(f"total    : {n_with_idx}/{len(rows)} rows with chunk_index")
    print(f"output   : {out_path}")
    return 0 if n_with_idx > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
