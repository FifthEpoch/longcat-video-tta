#!/usr/bin/env python3
"""Per-video x per-chunk drift breakdown across arms (NOTTA vs delta variants).

Answers two things the population verdict hides:
  (1) Is the intervention's effect HETEROGENEOUS -- helped some videos, hurt
      others (so a per-video "apply TTA or not" gate could matter)?
  (2) For which videos / at which chunk would NO-TTA have been better?

All arms are paired (same seed/pool => same videos + per-chunk seeds), so this
is a clean per-video comparison. GT-free signals are defined for the whole
rollout; GT metrics (psnr/ssim/lpips) run out after ~1-2 chunks.

Usage:
  python scripts/analyze_drift_per_video.py \
    --arm NOTTA=sweep_experiment/results/longhorizon_sweep_notta_native_12ch/merged_summary.json \
    --arm gen=sweep_experiment/results/longhorizon_sweep_delta_stream_native_12ch/merged_summary.json \
    --arm clean=sweep_experiment/results/longhorizon_sweep_delta_stream_clean_native_12ch/merged_summary.json \
    --baseline NOTTA \
    --out-dir sweep_experiment/results/longhorizon_sweep_delta_stream_clean_native_12ch/per_video
"""
import argparse
import csv
import json
import os

import numpy as np

GEN_FREE = ["sharpness", "temporal_motion", "colorfulness", "contrast"]
GT = ["psnr", "ssim", "lpips"]
ALL = GEN_FREE + GT


def _load(path):
    with open(path) as f:
        return json.load(f)


def _video_map(summary):
    """video_name -> {chunk_idx(int): {signal: value}}"""
    out = {}
    for r in summary.get("results", []):
        if not r.get("success"):
            continue
        chunks = {}
        for ch in r.get("chunks", []):
            chunks[int(ch["chunk"])] = ch
        out[r["video_name"]] = chunks
    return out


def _series(chunks, key):
    pts = []
    for ci in sorted(chunks):
        v = chunks[ci].get(key)
        if isinstance(v, (int, float)) and v == v:
            pts.append((ci, float(v)))
    return pts


def _drift(chunks, key):
    """signed and abs drift = last_finite - first_finite (None if <2 points)."""
    s = _series(chunks, key)
    if len(s) < 2:
        return None, None
    d = s[-1][1] - s[0][1]
    return d, abs(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True,
                    help="NAME=path/to/merged_summary.json (repeatable, >=2).")
    ap.add_argument("--baseline", default=None,
                    help="arm name treated as no-TTA reference (default: first).")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = _video_map(_load(path))
    names = list(arms)
    base = args.baseline or names[0]
    if base not in arms:
        raise SystemExit(f"baseline {base} not among arms {names}")
    others = [n for n in names if n != base]
    os.makedirs(args.out_dir, exist_ok=True)

    common = sorted(set.intersection(*[set(m) for m in arms.values()]))
    print(f"Arms: {names}   baseline={base}   paired videos: {len(common)}\n")

    # ---- full per-video x per-chunk dump -------------------------------
    dump_path = os.path.join(args.out_dir, "per_video_per_chunk.csv")
    with open(dump_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "chunk", "signal"] + names)
        for v in common:
            max_ch = max(max(arms[n][v]) for n in names if arms[n][v])
            for ci in range(1, max_ch + 1):
                for sig in ALL:
                    row = [v, ci, sig]
                    for n in names:
                        val = arms[n][v].get(ci, {}).get(sig)
                        row.append(f"{val:.6f}" if isinstance(val, (int, float)) and val == val else "")
                    w.writerow(row)
    print(f"wrote full dump -> {dump_path}\n")

    # ---- per-signal heterogeneity + per-video winner -------------------
    verdict = {"baseline": base, "arms": names, "n_videos": len(common), "signals": {}}
    for sig in ALL:
        # abs drift per arm per video (only videos where all arms have drift)
        per = {n: {} for n in names}
        for v in common:
            ok = True
            for n in names:
                _, dabs = _drift(arms[n][v], sig)
                if dabs is None:
                    ok = False
                    break
                per[n][v] = dabs
            if not ok:
                for n in names:
                    per[n].pop(v, None)
        vids = sorted(set.intersection(*[set(per[n]) for n in names])) if names else []
        if len(vids) < 2:
            continue

        arr = {n: np.array([per[n][v] for v in vids], float) for n in names}
        # oracle per-video = pick the arm with least drift for that video
        stacked = np.vstack([arr[n] for n in names])          # [arms, vids]
        oracle = stacked.min(axis=0)
        best_arm_idx = stacked.argmin(axis=0)
        best_counts = {n: int((best_arm_idx == i).sum()) for i, n in enumerate(names)}

        sig_row = {
            "n": len(vids),
            "mean_abs_drift": {n: float(arr[n].mean()) for n in names},
            "n_videos_best": best_counts,
            "oracle_mean_abs_drift": float(oracle.mean()),
        }
        # helps/hurts vs baseline for each other arm
        for n in others:
            diff = arr[base] - arr[n]                          # >0 => arm drifts less than baseline (helps)
            sig_row.setdefault("vs_baseline", {})[n] = {
                "helps": int((diff > 0).sum()),
                "hurts": int((diff < 0).sum()),
                "ties": int((diff == 0).sum()),
                "mean_reduction": float(diff.mean()),
            }
        verdict["signals"][sig] = sig_row

        # print (GT-free first; GT flagged)
        tag = "" if sig in GEN_FREE else "  [GT: n small, ~1-2 chunks]"
        print(f"== {sig}{tag}   (n={len(vids)}) ==")
        for n in names:
            print(f"   mean|drift| {n:8s} = {arr[n].mean():.5f}   best-for {best_counts[n]:>2d}/{len(vids)} videos")
        print(f"   oracle (per-video min) = {oracle.mean():.5f}"
              f"   [inflated by max/min-over-noise; upper bound]")
        for n in others:
            vb = sig_row["vs_baseline"][n]
            print(f"   {n} vs {base}: helps {vb['helps']}  hurts {vb['hurts']}  "
                  f"(mean reduction {vb['mean_reduction']:+.5f})")
        print()

    # ---- per-video net verdict across GT-free signals ------------------
    print("== per-video net verdict (GT-free signals; which arm drifts least) ==")
    net = {}
    for v in common:
        wins = {n: 0 for n in names}
        counted = 0
        for sig in GEN_FREE:
            dabs = {}
            ok = True
            for n in names:
                _, da = _drift(arms[n][v], sig)
                if da is None:
                    ok = False
                    break
                dabs[n] = da
            if not ok:
                continue
            counted += 1
            wins[min(dabs, key=dabs.get)] += 1
        net[v] = {"wins": wins, "signals_counted": counted}
        winner = max(wins, key=wins.get) if counted else "n/a"
        print(f"   {v:22s} winner={winner:8s}  " +
              "  ".join(f"{n}:{wins[n]}" for n in names))
    verdict["per_video_net"] = net

    # baseline-better tally
    base_better = sum(1 for v in net if net[v]["signals_counted"] and
                      net[v]["wins"][base] == max(net[v]["wins"].values()) and
                      net[v]["wins"][base] > max([net[v]["wins"][n] for n in others] + [0]))
    print(f"\n   videos where NO-TTA ({base}) strictly wins the most GT-free signals: "
          f"{base_better}/{len(common)}")

    with open(os.path.join(args.out_dir, "per_video_verdict.json"), "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"\nsaved -> {args.out_dir}/per_video_verdict.json  (+ per_video_per_chunk.csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
