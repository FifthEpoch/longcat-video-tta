#!/usr/bin/env python3
"""Bootstrap confidence intervals for FVD and paired ΔFVD between policies.

FVD is a *distribution-level* metric (one number over a set of videos), so it
cannot go through the per-video paired bootstrap used for PSNR/VBench in
scripts/router_significance_analysis.py. This script instead bootstraps at the
I3D-feature level:

  * Extract 400-d I3D features per generated clip (gen-only window) for each
    policy, keyed by canonical video_id, reusing sweep_experiment/scripts/
    eval_fvd.py (same DFoT protocol, same GT-cache reference).
  * Restrict every policy to the SHARED matched video_id set.
  * For B bootstrap draws, resample matched video_ids WITH REPLACEMENT (same
    indices across all policies), recompute each policy's FVD vs the fixed GT
    cache from sufficient stats, and record (a) the absolute FVD and (b) the
    PAIRED difference FVD(policy) − FVD(baseline). The paired difference is the
    honest test of "does this policy change FVD vs NO-TTA?" — the (large,
    high-dim) FVD bias mostly cancels because both use the same N and reference.

Verdict: if the ΔFVD 95% CI excludes 0, the policy's FVD differs from NO-TTA
beyond sampling noise; if it includes 0, FVD is "null" (no distinguishable
distribution shift), mirroring the PSNR/VBench null result.

Reference distribution is held fixed (the GT cache), the standard convention for
FVD CIs; this slightly under-states total variance but is exact for the paired
ΔFVD comparison across policies (identical reference cancels).

Usage (GPU; run via sbatch/run_fvd_bootstrap_ci.sbatch):
    python3 sweep_experiment/scripts/fvd_bootstrap_ci.py \
      --gt-cache gt_caches/panda_ood_budget_1000v_preview_longcat.npz \
      --num-cond-frames 14 --num-gen-frames 14 \
      --baseline always_notta \
      --policy always_notta:sweep_experiment/reports/budget_oracle_fvd_1000v_preview/matched/always_notta/videos \
      --policy fixed:sweep_experiment/reports/budget_oracle_fvd_1000v_preview/matched/fixed_S10_LR5e3/videos \
      --policy oracle:sweep_experiment/reports/budget_oracle_fvd_1000v_preview/oracle_best_psnr/videos \
      --n-boot 2000 \
      --output-dir sweep_experiment/reports/per_video_analysis/2026-07-31/fvd_bootstrap
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sweep_experiment.scripts.eval_fvd import (  # noqa: E402
    _canonical_video_id_from_path,
    _load_i3d,
    compute_frechet_from_sufficient_stats,
    extract_i3d_features,
    load_video_as_tensor,
)


def _extract_policy_features(
    videos_dir: Path,
    i3d_model,
    *,
    device: str,
    batch_size: int,
    num_cond_frames: int,
    num_gen_frames: int,
) -> Dict[str, np.ndarray]:
    """Return {canonical_video_id: 400-d I3D feature} for one policy dir."""
    paths = sorted(videos_dir.glob("*.mp4"))
    if not paths:
        raise SystemExit(f"[error] no mp4s under {videos_dir}")
    vids: List[str] = []
    tensors: List = []
    seen: set = set()
    for p in paths:
        vid = _canonical_video_id_from_path(p)
        if vid in seen:  # keep first per id (matched dirs are 1 clip per id)
            continue
        t = load_video_as_tensor(
            str(p),
            num_cond_frames=num_cond_frames,
            num_gen_frames=num_gen_frames,
        )
        if t is None:
            continue
        seen.add(vid)
        vids.append(vid)
        tensors.append(t)
    feats = extract_i3d_features(tensors, i3d_model, device, batch_size)
    return {vid: feats[i] for i, vid in enumerate(vids)}


def _fvd_from_features(
    feat_mat: np.ndarray, ref_sum: np.ndarray, ref_cov: np.ndarray, ref_count: int
) -> float:
    gen_sum = feat_mat.sum(axis=0)
    gen_cov = feat_mat.T @ feat_mat
    return compute_frechet_from_sufficient_stats(
        gen_sum, gen_cov, feat_mat.shape[0], ref_sum, ref_cov, ref_count
    )


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q))


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap CI for FVD / paired ΔFVD")
    ap.add_argument("--gt-cache", type=str, required=True)
    ap.add_argument(
        "--policy", action="append", required=True,
        metavar="NAME:DIR",
        help="Repeatable. NAME:path/to/videos (dir of gen mp4s).",
    )
    ap.add_argument("--baseline", type=str, default="always_notta",
                    help="Policy NAME used as the ΔFVD reference (default always_notta).")
    ap.add_argument("--num-cond-frames", type=int, default=14)
    ap.add_argument("--num-gen-frames", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    policies: List[Tuple[str, Path]] = []
    for spec in args.policy:
        if ":" not in spec:
            raise SystemExit(f"[error] --policy must be NAME:DIR, got {spec!r}")
        name, path = spec.split(":", 1)
        policies.append((name, Path(path)))
    names = [n for n, _ in policies]
    if args.baseline not in names:
        raise SystemExit(f"[error] baseline {args.baseline!r} not among policies {names}")

    if not Path(args.gt_cache).exists():
        raise SystemExit(f"[error] GT cache not found: {args.gt_cache}")
    cache = np.load(args.gt_cache, allow_pickle=True)
    ref_sum = cache["ref_fvd_sum"].astype(np.float64)
    ref_cov = cache["ref_fvd_cov"].astype(np.float64)
    ref_count = int(cache["ref_fvd_count"])
    print(f"GT cache reference: {ref_count} videos", file=sys.stderr)

    t0 = time.time()
    print("Loading I3D...", file=sys.stderr)
    i3d = _load_i3d(args.device)

    feats_by_policy: Dict[str, Dict[str, np.ndarray]] = {}
    for name, vdir in policies:
        print(f"Extracting I3D features: {name} <- {vdir}", file=sys.stderr)
        feats_by_policy[name] = _extract_policy_features(
            vdir, i3d, device=args.device, batch_size=args.batch_size,
            num_cond_frames=args.num_cond_frames, num_gen_frames=args.num_gen_frames,
        )
        print(f"  {name}: {len(feats_by_policy[name])} clips", file=sys.stderr)

    # matched set = ids present in every policy
    matched = set(feats_by_policy[names[0]])
    for n in names[1:]:
        matched &= set(feats_by_policy[n])
    matched_ids = sorted(matched)
    n = len(matched_ids)
    if n < 30:
        raise SystemExit(f"[error] only {n} matched video ids across policies.")
    print(f"Matched video ids across all policies: {n}", file=sys.stderr)

    # stack per policy in the matched order
    F: Dict[str, np.ndarray] = {
        name: np.stack([feats_by_policy[name][v] for v in matched_ids], axis=0)
        for name in names
    }

    # point FVD per policy (matched N)
    point_fvd = {
        name: _fvd_from_features(F[name], ref_sum, ref_cov, ref_count) for name in names
    }
    for name in names:
        print(f"  point FVD[{name}] = {point_fvd[name]:.4f} (N={n})", file=sys.stderr)

    # ---- bootstrap over matched ids -------------------------------------
    rng = np.random.default_rng(args.seed)
    boot_fvd: Dict[str, np.ndarray] = {name: np.empty(args.n_boot) for name in names}
    for b in range(args.n_boot):
        idx = rng.integers(0, n, n)
        for name in names:
            boot_fvd[name][b] = _fvd_from_features(
                F[name][idx], ref_sum, ref_cov, ref_count
            )

    base = args.baseline
    results = []
    for name in names:
        abs_mean = float(boot_fvd[name].mean())
        abs_lo, abs_hi = _pct(boot_fvd[name], 2.5), _pct(boot_fvd[name], 97.5)
        if name == base:
            d_mean = d_lo = d_hi = 0.0
            excludes0 = False
        else:
            diff = boot_fvd[name] - boot_fvd[base]  # paired (same resample idx)
            d_mean = float(diff.mean())
            d_lo, d_hi = _pct(diff, 2.5), _pct(diff, 97.5)
            excludes0 = (d_lo > 0) or (d_hi < 0)
        results.append({
            "policy": name,
            "point_fvd": round(point_fvd[name], 4),
            "boot_fvd_mean": round(abs_mean, 4),
            "boot_fvd_ci": [round(abs_lo, 4), round(abs_hi, 4)],
            "d_vs_baseline_mean": round(d_mean, 4),
            "d_vs_baseline_ci": [round(d_lo, 4), round(d_hi, 4)],
            "d_ci_excludes_0": bool(excludes0),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "gt_cache": args.gt_cache,
        "baseline": base,
        "n_matched": n,
        "n_boot": args.n_boot,
        "num_cond_frames": args.num_cond_frames,
        "num_gen_frames": args.num_gen_frames,
        "elapsed_seconds": round(time.time() - t0, 1),
        "results": results,
    }
    (args.output_dir / "fvd_bootstrap.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# FVD bootstrap CI — paired ΔFVD vs NO-TTA",
        "",
        f"**GT cache:** `{Path(args.gt_cache).name}`  ·  **Matched N:** {n}  ·  "
        f"**Bootstrap:** {args.n_boot}  ·  **Baseline:** `{base}`  ·  "
        f"gen window = [{args.num_cond_frames} cond | {args.num_gen_frames} gen].",
        "",
        "`ΔFVD vs NO-TTA` is a PAIRED bootstrap (same resampled video ids for both "
        "policies), so the FVD estimator bias cancels. FVD lower is better.",
        "",
        "| Policy | point FVD | bootstrap FVD [95% CI] | ΔFVD vs NO-TTA [95% CI] | CI excludes 0? |",
        "|---|---:|---|---|:--:|",
    ]
    for r in results:
        d = "—" if r["policy"] == base else (
            f"{r['d_vs_baseline_mean']:+.4f} "
            f"[{r['d_vs_baseline_ci'][0]:+.4f}, {r['d_vs_baseline_ci'][1]:+.4f}]"
        )
        excl = "—" if r["policy"] == base else ("**yes**" if r["d_ci_excludes_0"] else "no (null)")
        lines.append(
            f"| {r['policy']} | {r['point_fvd']:.3f} | "
            f"{r['boot_fvd_mean']:.3f} [{r['boot_fvd_ci'][0]:.3f}, {r['boot_fvd_ci'][1]:.3f}] | "
            f"{d} | {excl} |"
        )
    lines += [
        "",
        "## Read",
        "",
        "- **ΔFVD CI includes 0** ⇒ that policy's FVD is statistically "
        "indistinguishable from NO-TTA (FVD is *null*, like PSNR/VBench).",
        "- **ΔFVD CI excludes 0 and is positive** ⇒ the policy makes FVD *worse* "
        "(further from the reference distribution).",
        "",
    ]
    report = args.output_dir / "fvd_bootstrap_summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
