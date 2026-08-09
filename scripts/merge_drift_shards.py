#!/usr/bin/env python3
"""Merge sharded long-horizon drift runs into one combined summary + verdict.

Each shard is an independent ``diag_longhorizon_drift.py`` run (its own
output dir + checkpoint) covering a slice of the video list. This concatenates
the successful per-video records across shards and recomputes the per-chunk
drift curves + verdict over the pooled set.

Usage:
    python3 scripts/merge_drift_shards.py \
        --shards-root sweep_experiment/results/longhorizon_sweep_notta_native \
        --out sweep_experiment/results/longhorizon_sweep_notta_native/merged_summary.json
"""
import argparse
import glob
import json
import os

import numpy as np

GEN_FREE = ["sharpness", "colorfulness", "saturation", "brightness",
            "contrast", "temporal_motion", "seam_jump", "seam_ratio"]
GT = ["psnr", "ssim", "lpips"]
ALL = GEN_FREE + GT
VERDICT_KEYS = ["sharpness", "temporal_motion", "colorfulness", "contrast", "psnr", "ssim", "lpips"]


def _load_results(shard_dir):
    """Prefer summary.json (finished shard), else checkpoint.json (partial)."""
    for name in ("summary.json", "checkpoint.json"):
        p = os.path.join(shard_dir, name)
        if os.path.isfile(p):
            with open(p) as f:
                d = json.load(f)
            return d.get("results", []), name
    return [], None


def _finite(vals):
    return [v for v in vals if isinstance(v, (int, float)) and v == v]


def build_curves(results, num_chunks):
    per = {k: [[] for _ in range(num_chunks)] for k in ALL}
    for r in results:
        if not r.get("success"):
            continue
        for ch in r.get("chunks", []):
            i = ch["chunk"] - 1
            if 0 <= i < num_chunks:
                for k in ALL:
                    v = ch.get(k)
                    if isinstance(v, (int, float)) and v == v:
                        per[k][i].append(v)
    curves = {}
    for k in ALL:
        curves[k] = {
            "mean": [float(np.mean(per[k][i])) if per[k][i] else None for i in range(num_chunks)],
            "std":  [float(np.std(per[k][i])) if per[k][i] else None for i in range(num_chunks)],
            "n":    [len(per[k][i]) for i in range(num_chunks)],
        }
    verdict = {}
    for k in VERDICT_KEYS:
        pts = [(i + 1, m) for i, m in enumerate(curves[k]["mean"]) if m is not None]
        if len(pts) >= 2:
            xs = np.array([p[0] for p in pts], float)
            ys = np.array([p[1] for p in pts], float)
            verdict[k] = {
                "first_chunk": float(ys[0]), "last_chunk": float(ys[-1]),
                "abs_change": float(ys[-1] - ys[0]),
                "pct_change": float((ys[-1] - ys[0]) / (abs(ys[0]) + 1e-9) * 100.0),
                "slope_per_chunk": float(np.polyfit(xs, ys, 1)[0]),
                "n_chunks_with_data": len(pts),
            }
    return curves, verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards-root", required=True,
                    help="Parent dir containing shard_* subdirs (or pass --shard-glob).")
    ap.add_argument("--shard-glob", default=None,
                    help="Explicit glob for shard dirs (default: <root>/shard_*).")
    ap.add_argument("--num-chunks", type=int, default=0,
                    help="0 = infer from the shards' max chunk count.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pattern = args.shard_glob or os.path.join(args.shards_root, "shard_*")
    shard_dirs = sorted(d for d in glob.glob(pattern) if os.path.isdir(d))
    if not shard_dirs:
        print(f"No shard dirs matched {pattern}")
        return 2

    pooled, meta = [], None
    print(f"Merging {len(shard_dirs)} shards from {pattern}")
    for d in shard_dirs:
        res, src = _load_results(d)
        ok = [r for r in res if r.get("success")]
        print(f"  {os.path.basename(d):24s} {src or '<none>':16s} "
              f"{len(ok)}/{len(res)} successful")
        pooled.extend(res)
        if meta is None and res:
            meta = {k: res[0].get(k) for k in ("method", "rollout_mode", "num_chunks",
                                               "num_gen_per_chunk")}

    successful = [r for r in pooled if r.get("success")]
    nch = args.num_chunks or max((len(r.get("chunks", [])) for r in successful), default=0)
    curves, verdict = build_curves(successful, nch)

    summary = {
        "merged": True, "num_shards": len(shard_dirs),
        "num_videos": len(pooled), "num_successful": len(successful),
        "num_chunks": nch, "meta": meta or {},
        "drift_curves": curves, "drift_verdict": verdict,
        "results": pooled,
    }
    out = args.out or os.path.join(args.shards_root, "merged_summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nPooled N={len(successful)} videos x {nch} chunks")
    print("-- merged drift verdict (chunk1 -> last non-nan) --")
    for k in VERDICT_KEYS:
        if k in verdict:
            v = verdict[k]
            print(f"  {k:16s} {v['first_chunk']:9.4f} -> {v['last_chunk']:9.4f} "
                  f"({v['pct_change']:+.1f}%, slope={v['slope_per_chunk']:+.5f})")
    print(f"\nSaved: {out}")
    print(f"Plot:  python scripts/plot_drift_curves.py --summary {out} "
          f"--out-dir {os.path.dirname(out)}/plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
