#!/usr/bin/env python3
"""Chunk-interaction gate: would a TIME-SCHEDULED (ramped) delta help?

The three prior deltas (fixed / streaming-generated / streaming-clean) all used
a CONSTANT blend. A scheduled delta -- small early (content near-clean), larger
late (content degraded) -- is only justified if the constant delta's per-video
paired effect CROSSES OVER: net-harmful (or neutral) on early chunks, net-helpful
on late chunks. If the effect is flat noise at every chunk, no schedule can help.

This reads the SAME merged_summary.json files as the paired test (no GPU) and,
per GT-free signal, computes the paired intervention effect at each chunk:

    drift_A(v,t) = |signal_A(v,t) - signal_A(v,1)|            (A = NOTTA baseline)
    drift_B(v,t) = |signal_B(v,t) - signal_B(v,1)|            (B = a delta arm)
    effect(v,t)  = drift_A(v,t) - drift_B(v,t)   ( >0 => delta reduces drift )

Reports, per signal:
  * per-chunk mean effect (+ bootstrap 95% CI over videos)
  * relative effect = effect / (|drift_A| + eps)  (removes "drift grows with t")
  * early window (chunks 2..4) vs late window (last third) mean effect
  * slope of effect vs chunk index
  * VERDICT: crossover (hurts early, helps late) => a ramp is worth building;
             flat/noise or anti-crossover => skip, run NOTTA-only capstone.

Usage:
  python scripts/analyze_delta_chunk_interaction.py \
    --notta sweep_experiment/results/longhorizon_sweep_notta_native_12ch/merged_summary.json \
    --delta clean=sweep_experiment/results/longhorizon_sweep_delta_stream_clean_native_12ch/merged_summary.json \
    --delta gen=sweep_experiment/results/longhorizon_sweep_delta_stream_native_12ch/merged_summary.json \
    --out-dir sweep_experiment/results/longhorizon_sweep_delta_stream_clean_native_12ch/chunk_interaction
"""
import argparse
import json
import os

import numpy as np

GEN_FREE = ["sharpness", "temporal_motion", "colorfulness", "contrast"]
EPS = 1e-9


def _load(path):
    with open(path) as f:
        return json.load(f)


def _video_map(summary):
    out = {}
    for r in summary.get("results", []):
        if not r.get("success"):
            continue
        chunks = {}
        for ch in r.get("chunks", []):
            chunks[int(ch["chunk"])] = ch
        out[r["video_name"]] = chunks
    return out


def _val(chunks, ci, key):
    v = chunks.get(ci, {}).get(key)
    return float(v) if isinstance(v, (int, float)) and v == v else None


def _boot_ci(x, nboot=10000, seed=0):
    x = np.asarray(x, float)
    if x.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(nboot, x.size))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze_arm(notta, delta, name, out_dir):
    common = sorted(set(notta) & set(delta))
    # max chunk index common to all videos in both arms
    max_ch = min(
        min(max(notta[v]) for v in common),
        min(max(delta[v]) for v in common),
    )
    print(f"\n########## delta arm: {name}   (paired videos={len(common)}, chunks<= {max_ch}) ##########")

    result = {"arm": name, "n_videos": len(common), "max_chunk": max_ch, "signals": {}}
    for sig in GEN_FREE:
        # per-chunk paired effect across videos
        eff_by_chunk = {}     # ci -> list of per-video absolute effects
        rel_by_chunk = {}     # ci -> list of per-video relative effects
        for ci in range(2, max_ch + 1):
            eff, rel = [], []
            for v in common:
                a1 = _val(notta[v], 1, sig)
                b1 = _val(delta[v], 1, sig)
                at = _val(notta[v], ci, sig)
                bt = _val(delta[v], ci, sig)
                if None in (a1, b1, at, bt):
                    continue
                dA = abs(at - a1)
                dB = abs(bt - b1)
                eff.append(dA - dB)
                rel.append((dA - dB) / (dA + EPS))
            if eff:
                eff_by_chunk[ci] = eff
                rel_by_chunk[ci] = rel

        chunks = sorted(eff_by_chunk)
        if len(chunks) < 3:
            continue
        mean_eff = np.array([np.mean(eff_by_chunk[c]) for c in chunks])
        mean_rel = np.array([np.mean(rel_by_chunk[c]) for c in chunks])
        cis = [_boot_ci(eff_by_chunk[c], seed=c) for c in chunks]

        # early vs late windows
        early_c = [c for c in chunks if c <= min(4, chunks[len(chunks) // 3])]
        late_c = chunks[-max(1, len(chunks) // 3):]
        early_eff = float(np.mean([np.mean(eff_by_chunk[c]) for c in early_c]))
        late_eff = float(np.mean([np.mean(eff_by_chunk[c]) for c in late_c]))
        # slope of mean effect vs chunk index
        slope = float(np.polyfit(chunks, mean_eff, 1)[0])

        result["signals"][sig] = {
            "chunks": chunks,
            "mean_effect": [float(x) for x in mean_eff],
            "mean_effect_ci": cis,
            "mean_rel_effect": [float(x) for x in mean_rel],
            "early_window": early_c, "late_window": late_c,
            "early_effect": early_eff, "late_effect": late_eff,
            "slope_per_chunk": slope,
        }

        print(f"\n== {sig}  (effect>0 => delta reduces drift; paired over videos) ==")
        print("  chunk   mean_eff        95%CI            rel_eff")
        for c, m, ci, r in zip(chunks, mean_eff, cis, mean_rel):
            star = "*" if (ci[0] > 0 or ci[1] < 0) else " "
            print(f"  {c:>4d}  {m:+.5f}  [{ci[0]:+.5f},{ci[1]:+.5f}]{star}  {r:+.3f}")
        print(f"  early(chunks {early_c[0]}-{early_c[-1]}) mean = {early_eff:+.5f}   "
              f"late(chunks {late_c[0]}-{late_c[-1]}) mean = {late_eff:+.5f}   "
              f"slope = {slope:+.6f}/chunk")

        # verdict per signal
        crossover = early_eff < 0 and late_eff > 0 and slope > 0
        if crossover:
            v = "CROSSOVER: hurts early, helps late -> a ramp schedule is justified"
        elif slope > 0 and late_eff > 0:
            v = "improving-late (no early harm) -> ramp may add little over constant"
        elif abs(late_eff) < 1e-4 and abs(early_eff) < 1e-4:
            v = "FLAT ~0 at all chunks -> no schedule can help"
        elif slope < 0:
            v = "ANTI-crossover: effect worsens late -> ramp would HURT"
        else:
            v = "mixed/noise"
        result["signals"][sig]["verdict"] = v
        print(f"  VERDICT: {v}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"chunk_interaction_{name}.json")
        with open(p, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nsaved -> {p}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notta", required=True, help="NOTTA baseline merged_summary.json")
    ap.add_argument("--delta", action="append", required=True,
                    help="NAME=path/to/delta merged_summary.json (repeatable)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    notta = _video_map(_load(args.notta))
    results = []
    for spec in args.delta:
        name, _, path = spec.partition("=")
        results.append(analyze_arm(notta, _video_map(_load(path)), name, args.out_dir))

    print("\n" + "=" * 72)
    print("OVERALL READ: build a ramped/scheduled delta ONLY if a GT-free signal")
    print("shows CROSSOVER (early_effect<0, late_effect>0, slope>0) in a delta arm.")
    print("Otherwise the schedule has no signal to exploit -> run NOTTA-only capstone.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
