#!/usr/bin/env python3
"""
Extract prompts from Panda-70M metadata CSV and write them in Open-Sora's
expected CSV format (columns: text, name).

Usage:
    python t2v_experiment/scripts/prepare_opensora_prompts.py \
        --meta-csv datasets/panda_100/metadata.csv \
        --output t2v_experiment/opensora_prompts.csv \
        --max-videos 100
"""
from __future__ import annotations

import argparse
import csv


def parse_caption(raw: str) -> str:
    if raw.startswith("[") and "'" in raw:
        try:
            captions = eval(raw)
            return captions[0] if captions else "A video clip"
        except Exception:
            pass
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-videos", type=int, default=100)
    args = parser.parse_args()

    rows = []
    with open(args.meta_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get("filename", f"video_{len(rows):04d}.mp4")
            name = fname.replace(".mp4", "")
            caption = parse_caption(row.get("caption", "A video clip"))
            rows.append({"text": caption, "name": name})
            if len(rows) >= args.max_videos:
                break

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} prompts to {args.output}")


if __name__ == "__main__":
    main()
