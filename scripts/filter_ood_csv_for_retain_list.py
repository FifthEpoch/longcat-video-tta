#!/usr/bin/env python3
"""Filter a diffusion OOD CSV to videos listed in a retain-list JSON.

Used for the 1000v preview set: reuse segment-pool OOD scores (no re-GPU)
for the 1000 sampled video_ids.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.caption_utils import canonical_video_id  # noqa: E402


def _load_want_ids(retain_json: Path) -> set[str]:
    data = json.loads(retain_json.read_text(encoding="utf-8"))
    if data.get("all"):
        return {canonical_video_id(v) for v in data["all"] if v}
    return {
        canonical_video_id(v.get("video_id", ""))
        for v in data.get("videos", [])
        if v.get("video_id")
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ood-csv", type=Path, required=True)
    ap.add_argument("--retain-json", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    if not args.ood_csv.is_file():
        print(f"[error] OOD CSV not found: {args.ood_csv}", file=sys.stderr)
        return 2
    if not args.retain_json.is_file():
        print(f"[error] retain JSON not found: {args.retain_json}", file=sys.stderr)
        return 2

    want = _load_want_ids(args.retain_json)
    if not want:
        print("[error] retain JSON has no video ids", file=sys.stderr)
        return 2

    with args.ood_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("[error] empty OOD CSV", file=sys.stderr)
            return 2
        fieldnames = list(reader.fieldnames)
        rows = [
            r for r in reader
            if canonical_video_id(r.get("video_id", "")) in want
        ]

    got = {canonical_video_id(r.get("video_id", "")) for r in rows}
    missing = sorted(want - got)
    if missing:
        print(
            f"[warn] {len(missing)} retain ids missing from OOD CSV "
            f"(first: {missing[:5]})",
            file=sys.stderr,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(
        f"Wrote {args.output} ({len(rows)} rows; "
        f"wanted {len(want)}; missing {len(missing)})",
        file=sys.stderr,
    )
    return 0 if len(rows) >= len(want) * 0.95 else 1


if __name__ == "__main__":
    raise SystemExit(main())
