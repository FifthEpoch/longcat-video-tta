#!/usr/bin/env python3
"""Analyze a best-of-N drift-verifier search rollout (method=bestof).

The bestof arm logs EVERY candidate per chunk (candidate 0 == the NOTTA seed),
each scored by the GT-free drift verifier and evaluated against GT where the
source clip still overlaps. This script turns that into the core evidence for a
test-time-search method:

  1. SEARCH ACTIVITY  -- how often the verifier picks a non-NOTTA candidate
     (if it always picks candidate 0, search is a no-op).
  2. VERIFIER EFFECT  -- per-chunk GT-free composite of the chosen candidate vs
     candidate 0 (the NOTTA seed): how much closer to the real-frame reference
     the selected continuation is.
  3. TRUE-QUALITY CHECK (the credibility test) -- on chunks with GT, does the
     GT-FREE pick also improve PSNR / LPIPS over candidate 0, and how much of
     the ORACLE (best candidate by the GT metric itself) does it capture? A
     GT-free selector that also lifts GT quality is the publishable result.
  4. PER-SIGNAL ORACLE ceiling -- best-achievable per-signal deviation reduction.

The end-of-rollout DRIFT reduction vs the separate NOTTA run is a paired test:
  python scripts/compare_drift_paired.py --notta <notta>/merged_summary.json \
    --delta <bestof>/merged_summary.json --out-dir <bestof>/paired --label-b bestof

Usage:
  python scripts/analyze_bestof_search.py \
    --summary sweep_experiment/results/longhorizon_sweep_bestof_k4_native_12ch/merged_summary.json \
    --out-dir sweep_experiment/results/longhorizon_sweep_bestof_k4_native_12ch/search_analysis
"""
import argparse
import json
import os

import numpy as np

GEN_FREE = ["sharpness", "colorfulness", "contrast", "temporal_motion"]


def _load(path):
    with open(path) as f:
        return json.load(f)


def _dev(sig_val, ref_val):
    """relative deviation |x-ref|/|ref|."""
    if sig_val is None or ref_val is None or sig_val != sig_val or ref_val != ref_val:
        return None
    return abs(sig_val - ref_val) / (abs(ref_val) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="bestof merged_summary.json")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    s = _load(args.summary)
    recs = [r for r in s.get("results", []) if r.get("success") and r.get("method") == "bestof"]
    if not recs:
        raise SystemExit("no successful bestof records found in summary")

    k = s.get("search_k") or (recs[0].get("search_k"))
    print(f"best-of-{k} search analysis   videos={len(recs)}  "
          f"chunks/video={s.get('num_chunks')}\n")

    n_chunks = 0
    n_divergent = 0
    comp_cand0, comp_chosen, comp_rand = [], [], []
    # GT (where available)
    psnr_c0, psnr_ch, psnr_or, psnr_rand = [], [], [], []
    lpips_c0, lpips_ch, lpips_or, lpips_rand = [], [], [], []
    n_gt = 0
    psnr_chosen_beats_c0 = 0
    # per-signal deviation (chosen vs cand0 vs random-pick vs per-signal oracle)
    sig_dev = {sg: {"c0": [], "ch": [], "rand": [], "or": []} for sg in GEN_FREE}

    for r in recs:
        ref = r.get("ref_signals") or {}
        for ch in r.get("chunks", []):
            cands = ch.get("candidates")
            if not cands:
                continue
            n_chunks += 1
            chosen = ch.get("chosen_cand", 0)
            if chosen != 0:
                n_divergent += 1
            c0, best = cands[0], cands[chosen]
            comp_cand0.append(c0.get("score"))
            comp_chosen.append(best.get("score"))
            _scores = [c.get("score") for c in cands
                       if isinstance(c.get("score"), (int, float)) and c.get("score") == c.get("score")]
            if _scores:
                comp_rand.append(float(np.mean(_scores)))  # E[random pick]

            # per-signal relative deviation from the real-frame reference
            for sg in GEN_FREE:
                d0 = _dev(c0.get(sg), ref.get(sg))
                dch = _dev(best.get(sg), ref.get(sg))
                devs = [_dev(c.get(sg), ref.get(sg)) for c in cands]
                devs = [d for d in devs if d is not None]
                if d0 is not None:
                    sig_dev[sg]["c0"].append(d0)
                if dch is not None:
                    sig_dev[sg]["ch"].append(dch)
                if devs:
                    sig_dev[sg]["rand"].append(float(np.mean(devs)))  # E[random pick]
                    sig_dev[sg]["or"].append(min(devs))

            # true-quality check on chunks with GT
            if ch.get("gt_available"):
                p0, pch = c0.get("psnr"), best.get("psnr")
                psnrs = [c.get("psnr") for c in cands if isinstance(c.get("psnr"), (int, float)) and c.get("psnr") == c.get("psnr")]
                l0, lch = c0.get("lpips"), best.get("lpips")
                lpipss = [c.get("lpips") for c in cands if isinstance(c.get("lpips"), (int, float)) and c.get("lpips") == c.get("lpips")]
                if p0 == p0 and pch == pch and psnrs:
                    n_gt += 1
                    psnr_c0.append(p0); psnr_ch.append(pch); psnr_or.append(max(psnrs))
                    psnr_rand.append(float(np.mean(psnrs)))  # E[random pick]
                    if pch > p0:
                        psnr_chosen_beats_c0 += 1
                if l0 == l0 and lch == lch and lpipss:
                    lpips_c0.append(l0); lpips_ch.append(lch); lpips_or.append(min(lpipss))
                    lpips_rand.append(float(np.mean(lpipss)))  # E[random pick]

    mean = lambda x: float(np.mean(x)) if x else float("nan")

    print("== 1. search activity ==")
    print(f"   chunks total = {n_chunks};  verifier picked a NON-NOTTA candidate "
          f"in {n_divergent} ({100.0*n_divergent/max(1,n_chunks):.1f}%)")
    if n_divergent == 0:
        print("   (search is a no-op -> verifier never overrides the NOTTA seed)")

    print("\n== 2. GT-free verifier composite (lower=closer to real-frame ref) ==")
    print(f"   candidate 0 (NOTTA seed) mean = {mean(comp_cand0):.4f}")
    print(f"   random-pick   E[.]       mean = {mean(comp_rand):.4f}   <- noise floor")
    print(f"   chosen (verifier)        mean = {mean(comp_chosen):.4f}")
    print(f"   verifier gain vs random  = {mean(comp_rand)-mean(comp_chosen):+.4f}   "
          f"(>0 => selection beats chance, i.e. real signal not max-over-noise)")

    print("\n== 3. TRUE-QUALITY check on chunks with GT (n=%d) ==" % n_gt)
    if n_gt:
        print(f"   PSNR   cand0={mean(psnr_c0):.3f}  random={mean(psnr_rand):.3f}  "
              f"chosen={mean(psnr_ch):.3f}  oracle(byPSNR)={mean(psnr_or):.3f}")
        print(f"          chosen-cand0 = {mean(psnr_ch)-mean(psnr_c0):+.3f} dB   "
              f"chosen-random = {mean(psnr_ch)-mean(psnr_rand):+.3f} dB (gate)   "
              f"oracle-random = {mean(psnr_or)-mean(psnr_rand):+.3f} dB (noise floor)   "
              f"(chosen beats cand0 in {psnr_chosen_beats_c0}/{n_gt})")
        if lpips_ch:
            print(f"   LPIPS  cand0={mean(lpips_c0):.4f}  random={mean(lpips_rand):.4f}  "
                  f"chosen={mean(lpips_ch):.4f}  oracle={mean(lpips_or):.4f}  "
                  f"(chosen-random = {mean(lpips_ch)-mean(lpips_rand):+.4f}, lower=better)")
        print("   GATE: the verifier only has REAL signal if chosen beats RANDOM "
              "(not just cand0). If chosen≈random, the apparent oracle gap is "
              "max-over-noise (the PSNR-router trap) -> the GT-free signal doesn't "
              "track quality; try anchor-similarity verifier / larger k.")
    else:
        print("   no GT-overlapping chunks (rollout ran past the source clip).")

    print("\n== 4. per-signal deviation-from-reference (chosen vs cand0 vs random vs oracle) ==")
    for sg in GEN_FREE:
        c0m, chm = mean(sig_dev[sg]["c0"]), mean(sig_dev[sg]["ch"])
        rdm, orm = mean(sig_dev[sg]["rand"]), mean(sig_dev[sg]["or"])
        print(f"   {sg:16s} cand0={c0m:.4f} random={rdm:.4f} chosen={chm:.4f} oracle={orm:.4f}  "
              f"(vs_random {rdm-chm:+.4f}; oracle_vs_random {rdm-orm:+.4f}=noise floor)")

    out = {
        "search_k": k, "n_videos": len(recs), "n_chunks": n_chunks,
        "divergent_frac": n_divergent / max(1, n_chunks),
        "composite_cand0": mean(comp_cand0), "composite_random": mean(comp_rand),
        "composite_chosen": mean(comp_chosen),
        "n_gt_chunks": n_gt,
        "psnr_cand0": mean(psnr_c0), "psnr_random": mean(psnr_rand),
        "psnr_chosen": mean(psnr_ch), "psnr_oracle": mean(psnr_or),
        "psnr_chosen_beats_cand0": psnr_chosen_beats_c0,
        "lpips_cand0": mean(lpips_c0), "lpips_random": mean(lpips_rand),
        "lpips_chosen": mean(lpips_ch), "lpips_oracle": mean(lpips_or),
        "per_signal": {sg: {"cand0": mean(sig_dev[sg]["c0"]),
                            "random": mean(sig_dev[sg]["rand"]),
                            "chosen": mean(sig_dev[sg]["ch"]),
                            "oracle": mean(sig_dev[sg]["or"])} for sg in GEN_FREE},
    }
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        p = os.path.join(args.out_dir, "search_analysis.json")
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nsaved -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
