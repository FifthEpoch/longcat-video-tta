#!/usr/bin/env python3
"""Does TTA produce a REAL population effect vs NO-TTA on a given series?

The whole routing/gate premise only works when a metric has a real per-population
effect (E[Δ] != 0). On short-horizon in-domain Panda, PSNR/VBench/FVD are all
null (deployable TTA ≈ NO-TTA) and the per-video "headroom" is a max-over-noise
artifact. This script asks the same question in a NEW regime (e.g. long-horizon,
where drift accumulates), across PSNR/SSIM/LPIPS + the 7 VBench dims, with paired
bootstrap CIs and a sign-flip test — and for each metric reports whether the
binary TTA/no-TTA GATE could ever be meaningful there.

For each metric the paired per-video delta is d = TTA − NO-TTA (sign-oriented so
"better" is positive; for LPIPS that means NO-TTA − TTA). We report:
  - mean d [95% bootstrap CI], sign-flip p            -> is there a real effect?
  - binary-gate ceilings: E[relu(d)] (gate vs always-NO-TTA) and E[relu(-d)]
    (gate vs always-TTA), and the NOISE FLOOR E|d|/2.
  - artifact flag: when E[d] CI includes 0, the gate-vs-TTA ceiling == E|d|/2 =
    pure max-over-noise, so no gate (however simple) can win.

FVD is distribution-level (not per-video), so it is reported as a point delta
from each method's merged_summary.json (paired FVD CIs need the feature-level
bootstrap in sweep_experiment/scripts/fvd_bootstrap_ci.py).

Usage (cluster):
  # long-horizon Panda (drift regime)
  python3 scripts/analyze_population_effect.py \
    --series-root sweep_experiment/results/panda_longctx_1000v \
    --notta-run NOTTA --tta-run ADA_S10 \
    --out sweep_experiment/reports/per_video_analysis/popeffect_panda_longctx.json
  # contrast: short-horizon
  python3 scripts/analyze_population_effect.py \
    --series-root sweep_experiment/results/panda_1000v_standard \
    --notta-run NOTTA --tta-run ADA --out .../popeffect_panda_std.json

VBench uses the gen-only scores when VBENCH_SUBDIR is set, e.g.
  VBENCH_SUBDIR=vbench_results_geneval python3 scripts/analyze_population_effect.py ...
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.analyze_per_video_tta_gain import (  # noqa: E402
    load_per_video_metrics,
    _canonical_video_id,
)
from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    load_per_video_vbench,
    VBENCH_DIMS,
)

NAN = float("nan")
# (metric_name, higher_is_better, source)  source: "pixel" or "vbench"
PIXEL_METRICS = [("psnr", True), ("ssim", True), ("lpips", False)]


def paired_bootstrap_ci(d: np.ndarray, n_boot: int, seed: int, ci: float = 95.0
                        ) -> Tuple[float, float, float]:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return NAN, NAN, NAN
    rng = np.random.default_rng(seed)
    n = d.size
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo = (100 - ci) / 2
    return float(d.mean()), float(np.percentile(means, lo)), float(np.percentile(means, 100 - lo))


def sign_flip_p(d: np.ndarray, n_perm: int, seed: int) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return NAN
    obs = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    n = d.size
    cnt = sum(abs(float((d * (rng.integers(0, 2, n) * 2 - 1)).mean())) >= obs
              for _ in range(n_perm))
    return (cnt + 1) / (n_perm + 1)


def _mean_ci(a: np.ndarray, n_boot: int, seed: int) -> Tuple[float, float, float]:
    return paired_bootstrap_ci(a, n_boot, seed)


def _read_fvd(run_dir: Path) -> Optional[float]:
    ms = run_dir / "merged_summary.json"
    if not ms.exists():
        return None
    try:
        blob = json.loads(ms.read_text())
    except Exception:  # noqa: BLE001
        return None
    v = blob.get("fvd")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def build_paired(notta: Dict[str, dict], tta: Dict[str, dict], key: str
                 ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    ids = sorted(set(notta) & set(tta))
    nv, tv, kept = [], [], []
    for vid in ids:
        a = notta[vid].get(key)
        b = tta[vid].get(key)
        try:
            a = float(a); b = float(b)
        except (TypeError, ValueError):
            continue
        if math.isfinite(a) and math.isfinite(b):
            nv.append(a); tv.append(b); kept.append(vid)
    return np.array(nv), np.array(tv), kept


def analyze_metric(name: str, higher_better: bool, notta_v: np.ndarray,
                   tta_v: np.ndarray, n_boot: int, seed: int) -> dict:
    # sign-orient so "better" is positive
    d = (tta_v - notta_v) if higher_better else (notta_v - tta_v)
    m, lo, hi = paired_bootstrap_ci(d, n_boot, seed)
    p = sign_flip_p(d, n_boot, seed + 1)
    relu_pos = float(np.mean(np.clip(d, 0, None)))   # gate vs always-NO-TTA
    relu_neg = float(np.mean(np.clip(-d, 0, None)))  # gate vs always-TTA(fixed)
    noise_floor = float(np.mean(np.abs(d)) / 2.0)
    pop_null = (lo < 0 < hi)
    real_effect = (not pop_null) and (p < 0.05)
    gate_artifact = pop_null and abs(relu_neg - noise_floor) < max(1e-9, 0.15 * noise_floor)
    return {
        "n": int(d.size),
        "notta_mean": float(np.mean(notta_v)),
        "tta_mean": float(np.mean(tta_v)),
        "delta_mean": m, "delta_ci": [lo, hi], "signflip_p": p,
        "real_population_effect": bool(real_effect),
        "gate_vs_notta_ceiling": relu_pos,
        "gate_vs_tta_ceiling": relu_neg,
        "noise_floor_abs_d_over_2": noise_floor,
        "gate_is_maxovernoise_artifact": bool(gate_artifact),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series-root", required=True)
    ap.add_argument("--notta-run", default="NOTTA")
    ap.add_argument("--tta-run", required=True, help="e.g. ADA_S10, ADA, LORA_R8")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    series = Path(args.series_root)
    notta_dir = series / args.notta_run
    tta_dir = series / args.tta_run
    for d in (notta_dir, tta_dir):
        if not d.exists():
            print(f"ERROR: run dir not found: {d}", file=sys.stderr)
            return 2

    notta_px = load_per_video_metrics(notta_dir)
    tta_px = load_per_video_metrics(tta_dir)
    notta_vb = load_per_video_vbench(notta_dir)
    tta_vb = load_per_video_vbench(tta_dir)

    # merge vbench dims into the per-video dicts (canonical ids already aligned)
    def merge(px, vb):
        out = {}
        for vid, m in px.items():
            out[vid] = dict(m)
        for vid, dims in vb.items():
            out.setdefault(vid, {}).update(dims)
        return out

    notta = merge(notta_px, notta_vb)
    tta = merge(tta_px, tta_vb)

    print(f"series      : {series}")
    print(f"NO-TTA run  : {args.notta_run}  ({len(notta_px)} pixel, {len(notta_vb)} vbench ids)")
    print(f"TTA run     : {args.tta_run}  ({len(tta_px)} pixel, {len(tta_vb)} vbench ids)")

    metrics: List[Tuple[str, bool]] = list(PIXEL_METRICS) + [(d, True) for d in VBENCH_DIMS]
    results: Dict[str, dict] = {}

    print(f"\n{'metric':<24}{'N':>5}  {'NO-TTA':>9} {'TTA':>9}  "
          f"{'Δ(better+)':>11}  {'95% CI':>22}  {'p':>6}  effect  gate")
    for i, (name, hb) in enumerate(metrics):
        nv, tv, kept = build_paired(notta, tta, name)
        if len(kept) < 10:
            print(f"{name:<24}{len(kept):>5}  (insufficient paired coverage)")
            continue
        r = analyze_metric(name, hb, nv, tv, args.n_boot, 42 + i)
        results[name] = r
        eff = "REAL" if r["real_population_effect"] else "null"
        gate = ("ARTIFACT" if r["gate_is_maxovernoise_artifact"]
                else ("meaningful" if r["real_population_effect"] else "?"))
        lo, hi = r["delta_ci"]
        print(f"{name:<24}{r['n']:>5}  {r['notta_mean']:>9.4f} {r['tta_mean']:>9.4f}  "
              f"{r['delta_mean']:>+11.4f}  [{lo:>+9.4f},{hi:>+9.4f}]  "
              f"{r['signflip_p']:>6.3f}  {eff:<6} {gate}")

    fvd_notta = _read_fvd(notta_dir)
    fvd_tta = _read_fvd(tta_dir)
    print("\nFVD (population, from merged_summary.json; point delta, not paired CI):")
    if fvd_notta is not None and fvd_tta is not None:
        print(f"  NO-TTA {fvd_notta:.2f}  |  {args.tta_run} {fvd_tta:.2f}  |  "
              f"Δ {fvd_tta - fvd_notta:+.2f} ({'TTA better' if fvd_tta < fvd_notta else 'TTA worse'})")
    else:
        print(f"  NO-TTA {fvd_notta}  |  {args.tta_run} {fvd_tta}  (missing merged_summary.fvd)")

    # verdict
    real_dims = [m for m, r in results.items() if r["real_population_effect"]]
    print("\n=== VERDICT ===")
    if real_dims:
        print(f"  Real population effect (CI excludes 0 AND sign-flip p<0.05) on: "
              f"{', '.join(real_dims)}")
        print(f"  -> the binary TTA/no-TTA gate is MEANINGFUL for these metrics "
              f"(always-TTA already differs from NO-TTA); worth routing.")
    else:
        print("  NO metric shows a real population effect (all CIs include 0). "
              "Binary gate cannot beat always-TTA/NO-TTA here (ceilings are noise floor).")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "series_root": str(series),
            "notta_run": args.notta_run,
            "tta_run": args.tta_run,
            "metrics": results,
            "fvd": {"notta": fvd_notta, "tta": fvd_tta,
                    "delta": (fvd_tta - fvd_notta) if (fvd_notta is not None and fvd_tta is not None) else None},
            "real_effect_metrics": real_dims,
        }, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
