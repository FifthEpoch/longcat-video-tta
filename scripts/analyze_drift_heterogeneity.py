#!/usr/bin/env python3
"""Is the per-video 'which arm wins' heterogeneity REAL (routable) or noise?

The per-video breakdown showed NOTTA/gen/clean each win for different videos and
that some videos have a consistent winner across signals. Before spending compute
chasing a per-video router, test cheaply whether that consistency exceeds chance:

  * CROSS-SIGNAL CONSISTENCY: per video, do the GT-free signals agree on the best
    arm more than a null that preserves each signal's marginal win-rates but breaks
    the within-video coupling (permute each signal's winners across videos)? If yes
    => the winner is a property of the VIDEO (routable). If ~chance => the oracle
    gap is max/min-over-noise and a router can't realise it (cf. the PSNR router).
  * ORACLE vs RANDOM-PICK: how much of the oracle (per-video best) beats picking a
    fixed arm, and beats a random per-video pick (the noise-only expectation)?

Small N (=8) is underpowered; read this as a GATE (is it worth scaling N?), not a
verdict.

Usage: same --arm NAME=path args as analyze_drift_per_video.py.
"""
import argparse
import itertools
import json
import os

import numpy as np

GEN_FREE = ["sharpness", "temporal_motion", "colorfulness", "contrast"]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _video_map(summary):
    out = {}
    for r in summary.get("results", []):
        if not r.get("success"):
            continue
        out[r["video_name"]] = {int(ch["chunk"]): ch for ch in r.get("chunks", [])}
    return out


def _drift_abs(chunks, key):
    pts = [(ci, float(chunks[ci][key])) for ci in sorted(chunks)
           if isinstance(chunks[ci].get(key), (int, float)) and chunks[ci][key] == chunks[ci][key]]
    return abs(pts[-1][1] - pts[0][1]) if len(pts) >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--nperm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    arms = {}
    for spec in args.arm:
        name, _, path = spec.partition("=")
        arms[name] = _video_map(_load(path))
    names = list(arms)
    base = args.baseline or names[0]
    common = sorted(set.intersection(*[set(m) for m in arms.values()]))
    rng = np.random.default_rng(args.seed)

    # winner matrix W[signal][video] = arm index with least |drift| (only videos
    # where every arm has a finite drift for that signal).
    W, drift = {}, {}
    for sig in GEN_FREE:
        col, dcol = {}, {}
        for v in common:
            das = [_drift_abs(arms[n][v], sig) for n in names]
            if any(d is None for d in das):
                continue
            col[v] = int(np.argmin(das))
            dcol[v] = das
        if len(col) >= 3:
            W[sig] = col
            drift[sig] = dcol
    sigs = list(W)
    vids = sorted(set.intersection(*[set(W[s]) for s in sigs])) if sigs else []
    print(f"Arms {names}  baseline={base}  signals={sigs}  videos={len(vids)}\n")
    if len(vids) < 3 or len(sigs) < 2:
        print("too few paired videos/signals for a consistency test")
        return 0

    # ---- cross-signal consistency (pairwise winner agreement) ----------
    def pairwise_agreement(Wmat):
        # Wmat: dict sig->{vid->arm}; mean over signal-pairs of frac(videos) that agree.
        ags = []
        for a, b in itertools.combinations(sigs, 2):
            same = np.mean([Wmat[a][v] == Wmat[b][v] for v in vids])
            ags.append(same)
        return float(np.mean(ags))

    obs = pairwise_agreement(W)
    # null: permute each signal's winners across videos independently (keeps each
    # signal's marginal arm-win distribution, breaks within-video coupling).
    null = np.empty(args.nperm)
    base_arr = {s: np.array([W[s][v] for v in vids]) for s in sigs}
    for k in range(args.nperm):
        Wp = {s: dict(zip(vids, rng.permutation(base_arr[s]))) for s in sigs}
        null[k] = pairwise_agreement(Wp)
    p_consistency = float((null >= obs).mean())
    print("== cross-signal consistency (is the winner a video property?) ==")
    print(f"   observed mean pairwise winner-agreement = {obs:.3f}")
    print(f"   null mean (shuffled)                     = {null.mean():.3f} "
          f"[{np.percentile(null,2.5):.3f},{np.percentile(null,97.5):.3f}]")
    print(f"   p(consistency >= observed | null)        = {p_consistency:.4f}"
          + ("   *REAL structure*" if p_consistency < 0.05 else "   (~chance -> likely noise)"))
    print()

    # ---- oracle vs fixed-arm vs random-pick ----------------------------
    print("== oracle (per-video best) vs fixed arm vs random-pick ==")
    routing = {}
    for sig in sigs:
        D = np.array([drift[sig][v] for v in vids])          # [vids, arms]
        oracle = D.min(axis=1).mean()
        fixed = {names[i]: D[:, i].mean() for i in range(len(names))}
        best_fixed = min(fixed.values())
        random_pick = D.mean()                               # E[random arm] per video
        realizable = best_fixed - oracle                     # gap a PERFECT router could close
        noise_gap = random_pick - oracle                     # gap from noise-only min-over-arms
        routing[sig] = {"oracle": oracle, "fixed": fixed, "best_fixed": best_fixed,
                        "random_pick": random_pick}
        print(f"   {sig:16s} oracle={oracle:.5f}  best_fixed={best_fixed:.5f}  "
              f"random={random_pick:.5f}")
        print(f"                     perfect-router gain vs best_fixed = {realizable:+.5f}"
              f"   (noise-only min gain = {noise_gap:.5f})")
    print()
    print("   Reading: if 'perfect-router gain vs best_fixed' is ~0, no arm-selection")
    print("   helps beyond the single best fixed arm. If cross-signal p<0.05 AND the")
    print("   router gain is sizable, per-video routing is worth scaling N to test.")

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "heterogeneity.json"), "w") as f:
            json.dump({"arms": names, "signals": sigs, "n_videos": len(vids),
                       "consistency_obs": obs, "consistency_null_mean": float(null.mean()),
                       "p_consistency": p_consistency, "routing": routing}, f, indent=2)
        print(f"\nsaved -> {args.out_dir}/heterogeneity.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
