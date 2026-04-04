#!/usr/bin/env python3
"""
Select top-20 and bottom-20 PSNR videos from a no-TTA summary.json.

Writes a retain_videos.json that other runner scripts use via --save-only-list
to selectively save generated MP4s for qualitative comparison.

Usage:
    python select_retain_videos.py \
        --summary sweep_experiment/results/panda_notta/NOTTA/summary.json \
        --output  sweep_experiment/retain_videos.json \
        --top 20 --bottom 20
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Select top/bottom PSNR videos from no-TTA summary")
    parser.add_argument("--summary", required=True,
                        help="Path to no-TTA summary.json")
    parser.add_argument("--output", required=True,
                        help="Path to write retain_videos.json")
    parser.add_argument("--top", type=int, default=20,
                        help="Number of highest-PSNR videos to retain")
    parser.add_argument("--bottom", type=int, default=20,
                        help="Number of lowest-PSNR videos to retain")
    args = parser.parse_args()

    with open(args.summary) as f:
        summary = json.load(f)

    results = summary.get("results", [])
    scored = [r for r in results if r.get("success") and r.get("psnr") is not None]

    if not scored:
        print("ERROR: No successful videos with PSNR found in %s" % args.summary,
              file=sys.stderr)
        sys.exit(1)

    scored.sort(key=lambda r: r["psnr"])

    n_bottom = min(args.bottom, len(scored))
    n_top = min(args.top, len(scored))

    bottom = scored[:n_bottom]
    top = scored[-n_top:]

    def video_stem(r):
        name = r.get("video_name") or r.get("video", "")
        return Path(name).stem

    bottom_list = [
        {"video_name": video_stem(r), "psnr": r["psnr"], "rank": i + 1}
        for i, r in enumerate(bottom)
    ]
    top_list = [
        {"video_name": video_stem(r), "psnr": r["psnr"],
         "rank": len(scored) - n_top + i + 1}
        for i, r in enumerate(top)
    ]

    all_names = sorted(set(
        [e["video_name"] for e in bottom_list] +
        [e["video_name"] for e in top_list]
    ))

    output = {
        "source": args.summary,
        "total_videos": len(scored),
        "top_n": n_top,
        "bottom_n": n_bottom,
        "top_20": top_list,
        "bottom_20": bottom_list,
        "all": all_names,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print("Retain list written to %s" % args.output)
    print("  Total scored videos: %d" % len(scored))
    print("  Bottom %d PSNR range: %.4f - %.4f" % (
        n_bottom, bottom[0]["psnr"], bottom[-1]["psnr"]))
    print("  Top %d PSNR range: %.4f - %.4f" % (
        n_top, top[0]["psnr"], top[-1]["psnr"]))
    print("  Unique videos to retain: %d" % len(all_names))


if __name__ == "__main__":
    main()
