#!/usr/bin/env python3
"""Quick status check for Phase 2 runs. Prints job status, video counts, and key metrics."""
import json
import os
import glob
import sys

RESULTS_BASE = "sweep_experiment/results"
SLURM_LOG_DIR = "sweep_experiment/slurm_log"

EXPECTED_RUNS = {
    "phase2_full_lr": ["FULL_LR5e6", "FULL_LR1e6", "FULL_LR5e7"],
    "phase2_lora_rescue": ["LORA_R4_LR1e5", "LORA_R4_LR5e6", "LORA_R4_LR1e6",
                           "LORA_R2_LR1e5", "LORA_R2_LR5e6"],
    "phase2_adasteer_rescue": ["ADA_LR1e3", "ADA_LR5e4", "ADA_LR1e4", "ADA_LR5e5"],
}

NO_TTA_REF = os.path.join(RESULTS_BASE, "ucf500_notta/NOTTA/summary.json")

def load_summary(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def get_metric(summary, key):
    """Get metric from summary, falling back to per-video average."""
    val = summary.get(key)
    if val is not None:
        return val
    # Compute from per-video results
    results = summary.get("results", [])
    vals = [r[key] for r in results if r.get("success") and r.get(key) is not None]
    if vals:
        return sum(vals) / len(vals)
    # Try avg_ prefix
    val = summary.get("avg_" + key)
    return val

def main():
    out_path = "sweep_experiment/phase2_status_report.txt"
    out = open(out_path, "w")
    def pr(msg=""):
        print(msg)
        print(msg, file=out)

    notta = load_summary(NO_TTA_REF)
    if notta:
        n_psnr = get_metric(notta, "psnr")
        n_ssim = get_metric(notta, "ssim")
        n_lpips = get_metric(notta, "lpips")
        n_fvd = notta.get("fvd")
        n_fid = notta.get("fid")
        pr("No-TTA baseline: PSNR=%.4f, SSIM=%.4f, LPIPS=%.4f, FVD=%.2f, FID=%.2f, n=%s" % (
            n_psnr or 0, n_ssim or 0, n_lpips or 0, n_fvd or 0, n_fid or 0,
            notta.get("num_successful")))
    else:
        pr("No-TTA baseline: NOT FOUND")

    pr()
    pr(f"{'Run ID':<20s} {'Status':<10s} {'n_ok':>5s} {'PSNR':>8s} {'dPSNR':>8s} "
       f"{'SSIM':>7s} {'LPIPS':>7s} {'FVD':>9s} {'FID':>9s} {'AvgTrain':>9s} "
       f"{'AvgSteps':>9s} {'ES_early':>9s}")
    pr("-" * 130)

    notta_psnr = get_metric(notta, "psnr") if notta else None

    for series, run_ids in EXPECTED_RUNS.items():
        for run_id in run_ids:
            summary_path = os.path.join(RESULTS_BASE, series, run_id, "summary.json")
            s = load_summary(summary_path)

            if s is None:
                # Check for SLURM logs
                logs = glob.glob(os.path.join(SLURM_LOG_DIR, f"*{series}*{run_id}*.out"))
                if not logs:
                    logs = glob.glob(os.path.join(SLURM_LOG_DIR, f"*{run_id}*.out"))

                if logs:
                    log_path = sorted(logs)[-1]
                    try:
                        with open(log_path) as f:
                            lines = f.readlines()
                        last_lines = [l.strip() for l in lines[-10:] if l.strip()]
                        err_line = last_lines[-1] if last_lines else "?"
                    except Exception:
                        err_line = "?"
                    pr(f"{run_id:<20s} {'FAILED':<10s}  Log: {log_path}")
                    pr(f"{'':20s}            Last line: {err_line[:80]}")
                else:
                    pr(f"{run_id:<20s} {'NO OUTPUT':<10s}  (no summary.json or SLURM log found)")
                continue

            n_ok = s.get("num_successful", "?")
            psnr = get_metric(s, "psnr")
            ssim = get_metric(s, "ssim")
            lpips = get_metric(s, "lpips")
            fvd = s.get("fvd")
            fid = s.get("fid")
            avg_train = s.get("avg_train_time", 0)
            method = s.get("method", "?")

            # Compute avg steps and ES early-stop count
            results = s.get("results", [])
            successful = [r for r in results if r.get("success")]
            es_stopped = 0
            total_steps_list = []
            for r in successful:
                es_info = r.get("early_stopping_info")
                if es_info and es_info.get("stopped_early"):
                    es_stopped += 1
                steps = r.get("num_train_steps")
                if steps is not None:
                    total_steps_list.append(steps)

            avg_steps = f"{sum(total_steps_list)/len(total_steps_list):.1f}" if total_steps_list else "?"
            es_pct = f"{100*es_stopped/len(successful):.0f}%" if successful else "?"

            dpsnr = "%+.4f" % (psnr - notta_psnr) if (psnr is not None and notta_psnr is not None) else "?"

            psnr_s = "%8.4f" % psnr if psnr is not None else "    None"
            ssim_s = "%7.4f" % ssim if ssim is not None else "   None"
            lpips_s = "%7.4f" % lpips if lpips is not None else "   None"
            fvd_s = "%9.2f" % fvd if fvd is not None else "     None"
            fid_s = "%9.2f" % fid if fid is not None else "     None"

            pr("%s %s %5s %s %8s %s %s %s %s %8.1fs %9s %9s" % (
                run_id.ljust(20), "OK".ljust(10), str(n_ok),
                psnr_s, dpsnr, ssim_s, lpips_s, fvd_s, fid_s,
                avg_train, avg_steps, es_pct))

    pr()
    pr("Legend: dPSNR = PSNR - NoTTA_PSNR (positive = improvement)")
    pr(f"        ES_early = % of videos where early stopping triggered")
    pr(f"        AvgSteps = mean training steps actually run")

    out.close()
    print(f"\nReport also saved to: {out_path}")

if __name__ == "__main__":
    main()
