#!/usr/bin/env python3
"""Split a cluster dump (FILE: ... blocks) into local_archive/reports/."""
from __future__ import annotations

import re
import sys
from pathlib import Path


def split_dump(dump_text: str, archive_dir: Path) -> None:
    reports = archive_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(
        r"^#{70}\n### FILE: (?P<path>.+?)\n#{70}\n(?P<body>.*?)"
        r"(?=^#{70}\n### FILE: |\n#{70}\n### CSV HEAD: |\Z)",
        re.MULTILINE | re.DOTALL,
    )
    for m in pattern.finditer(dump_text):
        src = m.group("path").strip()
        body = m.group("body").rstrip()
        name = Path(src.replace("/scratch/wc3013/longcat-video-tta/", "")).name
        if body.startswith("MISSING:"):
            (reports / f"MISSING__{name}.txt").write_text(body + "\n", encoding="utf-8")
        else:
            (reports / name).write_text(body + "\n", encoding="utf-8")

    inv = re.search(
        r"(========== FILE INVENTORY ==========.*?)(?=\n#{70}\n### FILE:)",
        dump_text,
        re.DOTALL,
    )
    if inv:
        (archive_dir / "file_inventory.txt").write_text(inv.group(1).strip() + "\n", encoding="utf-8")

    csv_m = re.search(
        r"### CSV HEAD:.*?\n#{70}\n(.*?)\n\.\.\.\n(.*per_video_vbench_gains\.csv)",
        dump_text,
        re.DOTALL,
    )
    if csv_m:
        (archive_dir / "csv_head.txt").write_text(
            csv_m.group(1).strip() + "\n...\n" + csv_m.group(2).strip() + "\n",
            encoding="utf-8",
        )

    png = re.search(r"(### PNG LIST:.*?)(?=\n={10}|$)", dump_text, re.DOTALL)
    if png:
        (archive_dir / "png_list.txt").write_text(png.group(1).strip() + "\n", encoding="utf-8")


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: split_cluster_dump.py DUMP.txt ARCHIVE_DIR", file=sys.stderr)
        return 2
    dump_path = Path(sys.argv[1])
    archive_dir = Path(sys.argv[2])
    split_dump(dump_path.read_text(encoding="utf-8"), archive_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
