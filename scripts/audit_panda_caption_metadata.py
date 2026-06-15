#!/usr/bin/env python3
"""Audit Panda eval metadata for multi-segment captions and segment-index columns.

Run on the cluster against ``datasets/panda_1000_480p`` (or any Panda eval set)
to verify whether ``metadata.csv`` stores list captions and whether
``segment_index`` / ``chunk_index`` is present for segment-aligned selection.

Usage:
    python3 scripts/audit_panda_caption_metadata.py \\
        --metadata datasets/panda_1000_480p/metadata.csv \\
        --manifest datasets/panda_1000_480p/manifest.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.caption_utils import (
    parse_caption_list,
    resolve_caption_from_row,
    segment_index_from_metadata,
)


def _load_manifest(path: Path) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    if not path.exists():
        return out
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
                out[fname] = obj
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Panda caption metadata")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--show-examples", type=int, default=5)
    args = ap.parse_args()

    if not args.metadata.exists():
        print(f"[error] metadata not found: {args.metadata}", file=sys.stderr)
        return 2

    manifest = _load_manifest(args.manifest) if args.manifest else {}

    n_rows = 0
    n_list_captions = 0
    n_with_segment_col = 0
    n_mismatch_first_vs_resolved = 0
    examples: List[str] = []

    with args.metadata.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        print(f"metadata: {args.metadata}")
        print(f"columns : {fieldnames}")
        print()

        for row in reader:
            n_rows += 1
            raw = row.get("caption") or row.get("text") or ""
            caps = parse_caption_list(raw)
            if len(caps) > 1:
                n_list_captions += 1
                if len(examples) < args.show_examples:
                    fname = row.get("filename", "?")
                    seg = segment_index_from_metadata(row)
                    resolved = resolve_caption_from_row(row)
                    examples.append(
                        f"  {fname}: n_caps={len(caps)} segment_index={seg} "
                        f"resolved={resolved[:80]!r}..."
                    )
            if segment_index_from_metadata(row) is not None:
                n_with_segment_col += 1
            first_only = caps[0] if caps else ""
            resolved = resolve_caption_from_row(row)
            if caps and len(caps) > 1 and first_only != resolved:
                n_mismatch_first_vs_resolved += 1

    print(f"rows                         : {n_rows}")
    print(f"multi-caption list rows      : {n_list_captions}")
    print(f"rows with segment/chunk index: {n_with_segment_col}")
    print(f"rows where index != first cap: {n_mismatch_first_vs_resolved}")
    print(f"manifest entries loaded      : {len(manifest)}")
    print()
    if examples:
        print("Examples (multi-caption rows):")
        for ex in examples:
            print(ex)
    else:
        print("No multi-caption list rows found (captions may already be single-string).")

    if n_list_captions and not n_with_segment_col:
        print()
        print(
            "[note] Multi-caption rows lack segment_index/chunk_index. "
            "All paths now fall back to caption[0]. To align with the "
            "downloaded clip segment, add chunk_index to metadata.csv "
            "(see datasets/download_panda70m_subset.py manifest) or rebuild "
            "via scripts/build_panda_segment_pool.py for per-segment captions."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
