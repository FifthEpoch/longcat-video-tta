#!/usr/bin/env python3
"""Segment-aligned caption selection shared across TTA, CLIP, and Phase-0 features.

Panda-70M metadata often stores multiple segment captions as a Python-literal
list string (e.g. ``"['cap seg0', 'cap seg1']"``). Different code paths
previously disagreed: TTA loaders took the *first* caption while CLIP/OOD
feature scripts *joined* all captions. This module picks one caption per clip
using ``segment_index`` / ``chunk_index`` when present in metadata, otherwise
the first non-empty list entry (legacy TTA behaviour).

Fallback order for ``resolve_caption_for_clip``:
  1. ``segment_index`` (or ``chunk_index`` from the CSV row) selects
     ``captions[segment_index]`` when in range.
  2. Otherwise the first non-empty parsed caption (index 0).
  3. Plain (non-list) strings are returned stripped.
  4. Empty string if nothing usable.
"""
from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

_CANONICAL_PREFIX_RE = re.compile(r"^([A-Za-z][A-Za-z0-9]*_\d+)")


def canonical_video_id(s: Optional[str]) -> str:
    """Strip method suffixes so ``panda_0010_delta_a`` -> ``panda_0010``."""
    if not s:
        return ""
    stem = Path(str(s)).stem
    m = _CANONICAL_PREFIX_RE.match(stem)
    return m.group(1) if m else stem


def parse_caption_list(raw: Any) -> List[str]:
    """Return non-empty caption strings encoded in ``raw``."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        out = [str(x).strip() for x in raw if str(x).strip()]
        return out
    s = str(raw).strip()
    if not s:
        return []
    if s.startswith("[") and s.endswith("]"):
        try:
            obj = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return [s]
        if isinstance(obj, (list, tuple)):
            out = [str(x).strip() for x in obj if str(x).strip()]
            return out or [s]
    return [s]


def segment_index_from_metadata(
    row: Optional[dict] = None,
    *,
    segment_index: Optional[int] = None,
    chunk_index: Optional[int] = None,
) -> Optional[int]:
    """Read segment/chunk index from explicit args or a metadata CSV row."""
    if segment_index is not None:
        try:
            return int(segment_index)
        except (TypeError, ValueError):
            pass
    if chunk_index is not None:
        try:
            return int(chunk_index)
        except (TypeError, ValueError):
            pass
    if not row:
        return None
    for key in ("segment_index", "chunk_index", "seg_idx", "segment_idx"):
        val = row.get(key)
        if val is None or str(val).strip() == "":
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return None


def resolve_caption_for_clip(
    raw: Any,
    *,
    segment_index: Optional[int] = None,
) -> str:
    """Return one segment-aligned caption string for encode_prompt / CLIP."""
    captions = parse_caption_list(raw)
    if not captions:
        return ""
    if len(captions) == 1:
        return captions[0]
    if segment_index is not None and 0 <= segment_index < len(captions):
        return captions[segment_index]
    return captions[0]


def resolve_caption_from_row(row: dict) -> str:
    """Resolve caption from a metadata.csv row (caption + segment columns)."""
    raw = row.get("caption") or row.get("text") or ""
    seg = segment_index_from_metadata(row)
    return resolve_caption_for_clip(raw, segment_index=seg)


def load_resolved_captions_csv(
    path: Path,
    *,
    canonical_id: Callable[[str], str] = canonical_video_id,
    warn_missing: bool = True,
) -> Dict[str, str]:
    """Return ``{canonical_video_id -> resolved caption string}``."""
    out: Dict[str, str] = {}
    if not path.exists():
        if warn_missing:
            print(
                f"[warn] captions CSV not found at {path}; "
                "rows will use empty captions",
                file=sys.stderr,
            )
        return out
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = (
                row.get("filename") or row.get("video_path")
                or row.get("path") or row.get("video")
            )
            if not fname:
                continue
            vid = canonical_id(fname)
            if not vid:
                continue
            out[vid] = resolve_caption_from_row(row)
    return out
