#!/usr/bin/env python3
"""
Phase 1 Diagnostics: Investigate why TTA methods failed to improve over no-TTA baseline.

Usage:
    python sweep_experiment/scripts/phase1_diagnostics.py \
        --output-file sweep_experiment/phase1_diagnostics_report.txt
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np


def build_lookup(summary):
    return {r["video_name"]: r for r in summary.get("results", []) if r.get("success")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-file", type=str,
                        default="sweep_experiment/phase1_diagnostics_report.txt")
    parser.add_argument("--results-dir", type=str,
                        default="sweep_experiment/results")
    args = parser.parse_args()

    out = open(args.output_file, "w")

    def pr(msg=""):
        print(msg, file=out, flush=True)

    paths = {
        "notta":           os.path.join(args.results_dir, "ucf500_notta/NOTTA/summary.json"),
        "full_base":       os.path.join(args.results_dir, "ucf500_tta_baseline/FULL/summary.json"),
        "lora_base":       os.path.join(args.results_dir, "ucf500_tta_baseline/LORA/summary.json"),
        "adasteer_base":   os.path.join(args.results_dir, "ucf500_tta_baseline/DBL4/summary.json"),
        "full_esclip":     os.path.join(args.results_dir, "ucf500_tta_es_clip/FULL/summary.json"),
        "adasteer_esclip": os.path.join(args.results_dir, "ucf500_tta_es_clip/DBL4/summary.json"),
    }

    runs = {}
    for key, path in paths.items():
        if os.path.exists(path):
            runs[key] = json.load(open(path))
            pr(f"Loaded: {key} ({path})")
        else:
            pr(f"NOT FOUND: {key} ({path})")

    if "notta" not in runs:
        pr("ERROR: no-TTA baseline not found. Cannot proceed.")
        out.close()
        return

    notta_lk = build_lookup(runs["notta"])

    # ========================================================================
    pr()
    pr("=" * 80)
    pr("PHASE 1a: PER-VIDEO PAIRED COMPARISON (TTA vs No-TTA)")
    pr("=" * 80)

    for method_key, method_label in [
        ("full_base", "Full (baseline, no ES)"),
        ("full_esclip", "Full (ES+CLIP)"),
        ("lora_base", "LoRA (baseline, no ES)"),
        ("adasteer_base", "AdaSteer (baseline, no ES)"),
        ("adasteer_esclip", "AdaSteer (ES+CLIP)"),
    ]:
        if method_key not in runs:
            continue
        lk = build_lookup(runs[method_key])

        deltas = []
        for vname, notta_r in notta_lk.items():
            if vname not in lk:
                continue
            notta_psnr = notta_r.get("psnr")
            tta_psnr = lk[vname].get("psnr")
            if notta_psnr is None or tta_psnr is None:
                continue
            deltas.append(tta_psnr - notta_psnr)

        deltas = np.array(deltas)
        improved = int(np.sum(deltas > 0.01))
        degraded = int(np.sum(deltas < -0.01))
        unchanged = int(np.sum(np.abs(deltas) <= 0.01))

        pr(f"\n--- {method_label} ---")
        pr(f"  Paired videos: {len(deltas)}")
        pr(f"  Improved (>+0.01 dB): {improved} ({100*improved/len(deltas):.1f}%)")
        pr(f"  Degraded (<-0.01 dB): {degraded} ({100*degraded/len(deltas):.1f}%)")
        pr(f"  Unchanged:            {unchanged} ({100*unchanged/len(deltas):.1f}%)")
        pr(f"  Mean dPSNR:   {deltas.mean():+.4f} dB")
        pr(f"  Median dPSNR: {np.median(deltas):+.4f} dB")
        pr(f"  Std dPSNR:    {deltas.std():.4f} dB")
        pr(f"  Min dPSNR:    {deltas.min():+.4f} dB (worst degradation)")
        pr(f"  Max dPSNR:    {deltas.max():+.4f} dB (best improvement)")
        for pct in [5, 25, 50, 75, 95]:
            pr(f"  P{pct:02d} dPSNR:   {np.percentile(deltas, pct):+.4f} dB")

    # ========================================================================
    pr()
    pr("=" * 80)
    pr("PHASE 1b: EARLY STOPPING ANALYSIS (Full ES+CLIP)")
    pr("=" * 80)

    if "full_esclip" in runs:
        lk = build_lookup(runs["full_esclip"])

        best_steps = []
        stopped_early_count = 0
        ran_full_count = 0
        loss_improvements = []
        step_distribution = Counter()

        for vname, r in lk.items():
            es = r.get("early_stopping_info")
            if es is None:
                ran_full_count += 1
                continue

            bs = es.get("best_step")
            if bs is not None:
                best_steps.append(bs)
                step_distribution[bs] += 1

            if es.get("stopped_early", False):
                stopped_early_count += 1
            else:
                ran_full_count += 1

            lh = es.get("loss_history", [])
            if lh and len(lh) >= 2:
                first_loss = lh[0][1] if isinstance(lh[0], (list, tuple)) else lh[0]
                best_loss = es.get("best_loss")
                if best_loss is not None:
                    loss_improvements.append(first_loss - best_loss)

        pr(f"  Total videos with ES info: {len(best_steps)}")
        pr(f"  Stopped early: {stopped_early_count}")
        pr(f"  Ran full 50 steps: {ran_full_count}")

        if best_steps:
            bs_arr = np.array(best_steps)
            pr(f"\n  Best step distribution:")
            pr(f"    Mean best_step:   {bs_arr.mean():.1f}")
            pr(f"    Median best_step: {np.median(bs_arr):.1f}")
            pr(f"    best_step=0 (pretrained best): {step_distribution.get(0, 0)} "
               f"({100*step_distribution.get(0, 0)/len(best_steps):.1f}%)")

            pr(f"\n  Top 10 most common best_step values:")
            for step, count in step_distribution.most_common(10):
                pr(f"    step={step:3d}: {count:4d} videos ({100*count/len(best_steps):.1f}%)")

        if loss_improvements:
            pr(f"\n  Anchor loss improvement (step0 -> best):")
            pr(f"    Mean loss improvement:    {np.mean(loss_improvements):.6f}")
            pr(f"    Videos where loss improved: {sum(1 for x in loss_improvements if x > 1e-6)}")
            pr(f"    Videos where loss stayed ~same: {sum(1 for x in loss_improvements if abs(x) <= 1e-6)}")

    # ========================================================================
    pr()
    pr("=" * 80)
    pr("PHASE 1b2: FULL BASELINE (no ES) - TRAINING LOSS ANALYSIS")
    pr("=" * 80)

    if "full_base" in runs:
        lk = build_lookup(runs["full_base"])

        final_losses = [r.get("final_loss") for r in lk.values()
                        if r.get("final_loss") is not None]
        train_steps = [r.get("num_train_steps", 50) for r in lk.values()]

        pr(f"  Videos: {len(final_losses)}")
        if final_losses:
            pr(f"  Final loss: mean={np.mean(final_losses):.6f}, "
               f"std={np.std(final_losses):.6f}, "
               f"min={np.min(final_losses):.6f}, max={np.max(final_losses):.6f}")
        pr(f"  Train steps: {Counter(train_steps).most_common(3)}")

        deltas_loss_psnr = []
        for vname, r in lk.items():
            if vname not in notta_lk:
                continue
            fl = r.get("final_loss")
            tp = r.get("psnr")
            np_ = notta_lk[vname].get("psnr")
            if fl is not None and tp is not None and np_ is not None:
                deltas_loss_psnr.append((fl, tp - np_))

        if deltas_loss_psnr:
            losses, dpsnrs = zip(*deltas_loss_psnr)
            corr = np.corrcoef(losses, dpsnrs)[0, 1]
            pr(f"\n  Correlation (final_loss vs dPSNR): r={corr:.4f}")
            pr(f"  (negative r means lower loss -> better PSNR improvement)")

    # ========================================================================
    pr()
    pr("=" * 80)
    pr("PHASE 1c: ADASTEER DELTA NORMS vs QUALITY")
    pr("=" * 80)

    for method_key, method_label in [
        ("adasteer_base", "AdaSteer (baseline)"),
        ("adasteer_esclip", "AdaSteer (ES+CLIP)"),
    ]:
        if method_key not in runs:
            continue
        lk = build_lookup(runs[method_key])

        norms_and_dpsnr = []
        for vname, r in lk.items():
            if vname not in notta_lk:
                continue
            dn = r.get("delta_norms")
            tp = r.get("psnr")
            np_ = notta_lk[vname].get("psnr")
            if dn is not None and tp is not None and np_ is not None:
                if isinstance(dn, list):
                    total_norm = float(np.sqrt(sum(x ** 2 for x in dn)))
                else:
                    total_norm = float(dn)
                norms_and_dpsnr.append((total_norm, tp - np_))

        if norms_and_dpsnr:
            norms, dpsnrs = zip(*norms_and_dpsnr)
            norms = np.array(norms)
            dpsnrs = np.array(dpsnrs)
            corr = np.corrcoef(norms, dpsnrs)[0, 1]

            pr(f"\n--- {method_label} ---")
            pr(f"  Videos: {len(norms)}")
            pr(f"  Delta L2 norm: mean={norms.mean():.4f}, std={norms.std():.4f}, "
               f"min={norms.min():.4f}, max={norms.max():.4f}")
            pr(f"  Correlation (norm vs dPSNR): r={corr:.4f}")
            pr(f"  (negative r means larger deltas -> worse PSNR)")

            for q_label, lo, hi in [("Q1 (smallest deltas)", 0, 25),
                                     ("Q2", 25, 50), ("Q3", 50, 75),
                                     ("Q4 (largest deltas)", 75, 100)]:
                lo_v = np.percentile(norms, lo)
                hi_v = np.percentile(norms, hi)
                mask = (norms >= lo_v) & (norms <= hi_v)
                if mask.sum() > 0:
                    pr(f"    {q_label}: mean dPSNR={dpsnrs[mask].mean():+.4f}, "
                       f"norm range=[{lo_v:.4f}, {hi_v:.4f}]")

    # ========================================================================
    pr()
    pr("=" * 80)
    pr("PHASE 1d: LORA ANALYSIS")
    pr("=" * 80)

    if "lora_base" in runs:
        lk = build_lookup(runs["lora_base"])

        final_losses = [r.get("final_loss") for r in lk.values()
                        if r.get("final_loss") is not None]
        psnrs = [r.get("psnr") for r in lk.values() if r.get("psnr") is not None]

        pr(f"  Videos: {len(final_losses)}")
        if final_losses:
            pr(f"  Final loss: mean={np.mean(final_losses):.6f}, "
               f"std={np.std(final_losses):.6f}")
        if psnrs:
            pr(f"  PSNR: mean={np.mean(psnrs):.4f}, std={np.std(psnrs):.4f}")

        improved_vids = []
        for vname, r in lk.items():
            if vname not in notta_lk:
                continue
            tp = r.get("psnr")
            np_ = notta_lk[vname].get("psnr")
            if tp is not None and np_ is not None and tp > np_:
                improved_vids.append((vname, tp - np_))

        pr(f"\n  Videos where LoRA improved over no-TTA: {len(improved_vids)}")
        if improved_vids:
            improved_vids.sort(key=lambda x: -x[1])
            pr(f"  Top 5 improved:")
            for vname, d in improved_vids[:5]:
                pr(f"    {vname}: +{d:.4f} dB")

    # ========================================================================
    pr()
    pr("=" * 80)
    pr("PHASE 1e: TRAINING TIME vs QUALITY (ES impact)")
    pr("=" * 80)

    if "full_esclip" in runs:
        lk_es = build_lookup(runs["full_esclip"])
        lk_base = build_lookup(runs.get("full_base", {"results": []}))

        es_times = [r.get("train_time", 0) for r in lk_es.values()]
        base_times = [r.get("train_time", 0) for r in lk_base.values()]

        if base_times:
            pr(f"  Full baseline train time: mean={np.mean(base_times):.1f}s, "
               f"std={np.std(base_times):.1f}s")
        pr(f"  Full ES+CLIP train time:  mean={np.mean(es_times):.1f}s, "
           f"std={np.std(es_times):.1f}s")
        if base_times:
            pr(f"  Time savings: {(1 - np.mean(es_times) / np.mean(base_times)) * 100:.1f}%")

        actual_steps = [r.get("num_train_steps", 50) for r in lk_es.values()]
        pr(f"\n  Actual steps (ES+CLIP): mean={np.mean(actual_steps):.1f}, "
           f"median={np.median(actual_steps):.0f}, "
           f"min={np.min(actual_steps)}, max={np.max(actual_steps)}")
        step_counts = Counter(actual_steps)
        pr(f"  Step count distribution (top 10):")
        for step, count in step_counts.most_common(10):
            pr(f"    {step:3d} steps: {count:4d} videos ({100 * count / len(actual_steps):.1f}%)")

    # ========================================================================
    pr()
    pr("=" * 80)
    pr("PHASE 1f: PER-VIDEO PSNR DISTRIBUTION (understanding variance)")
    pr("=" * 80)

    notta_psnrs = np.array([r.get("psnr") for r in notta_lk.values()
                            if r.get("psnr") is not None])
    pr(f"  No-TTA PSNR distribution:")
    pr(f"    mean={notta_psnrs.mean():.4f}, std={notta_psnrs.std():.4f}")
    pr(f"    min={notta_psnrs.min():.4f}, max={notta_psnrs.max():.4f}")
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        pr(f"    P{pct:02d}: {np.percentile(notta_psnrs, pct):.4f}")

    low_thresh = np.percentile(notta_psnrs, 10)
    high_thresh = np.percentile(notta_psnrs, 90)
    low_psnr_vids = [(vname, r.get("psnr")) for vname, r in notta_lk.items()
                      if r.get("psnr") is not None and r["psnr"] < low_thresh]
    high_psnr_vids = [(vname, r.get("psnr")) for vname, r in notta_lk.items()
                       if r.get("psnr") is not None and r["psnr"] > high_thresh]

    pr(f"\n  Bottom 10% no-TTA videos ({len(low_psnr_vids)} videos, PSNR<{low_thresh:.2f}):")
    pr(f"    Mean PSNR: {np.mean([p for _, p in low_psnr_vids]):.4f}")

    pr(f"\n  Did TTA methods help bottom-10% more than top-10%?")
    for method_key, method_label in [
        ("full_base", "Full (baseline)"),
        ("full_esclip", "Full (ES+CLIP)"),
        ("adasteer_base", "AdaSteer (baseline)"),
    ]:
        if method_key not in runs:
            continue
        lk = build_lookup(runs[method_key])
        low_deltas = []
        high_deltas = []
        for vname, notta_psnr in low_psnr_vids:
            if vname in lk and lk[vname].get("psnr") is not None:
                low_deltas.append(lk[vname]["psnr"] - notta_psnr)
        for vname, notta_psnr in high_psnr_vids:
            if vname in lk and lk[vname].get("psnr") is not None:
                high_deltas.append(lk[vname]["psnr"] - notta_psnr)

        if low_deltas and high_deltas:
            pr(f"    {method_label}: bottom-10% dPSNR={np.mean(low_deltas):+.4f}, "
               f"top-10% dPSNR={np.mean(high_deltas):+.4f}")

    pr()
    pr("=" * 80)
    pr("DONE - Phase 1 diagnostics complete")
    pr("=" * 80)

    out.close()
    print(f"Report written to: {args.output_file}")


if __name__ == "__main__":
    main()
