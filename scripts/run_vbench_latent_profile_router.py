#!/usr/bin/env python3
"""Compare VBench budget routers with vs without rich VAE latent profiles.

Runs OOF ridge config pickers @ pilot N=200:

  baseline_exp7     — Phase-0 + probe PSNR/SSIM (prior best ~12.8%)
  vae_profile_probe — VAE latent profile (~130-d) + probe only
  vae_profile_full  — Phase-0 + VAE profile + probe

Requires ``vae_latent_profile_features.csv`` from
``extract_vae_latent_profile_features.py``.

Usage:
  python3 scripts/run_vbench_latent_profile_router.py --run-all
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import List, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.budget_routing_common import labeled_mask, load_pilot_bundle  # noqa: E402
from scripts.run_budget_routing_experiments import build_probe_features  # noqa: E402
from scripts.run_vbench_gain_prediction_experiments import (  # noqa: E402
    PROBE2,
    _policy_from_budget_task,
    _row,
)
from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
    run_budget_config_task,
)

EXPERIMENTS = (
    "baseline_exp7",
    "vae_profile_probe",
    "vae_profile_full",
)
BASELINE_BEST = 12.8


def _run_ridge_gain_router(
    name: str,
    X: np.ndarray,
    bundle: dict,
    Y: np.ndarray,
    fixed_vb: np.ndarray,
    grid: Sequence[str],
    *,
    seed: int,
    n_folds: int,
) -> dict:
    Y_gain = Y - fixed_vb[:, np.newaxis]
    with tempfile.TemporaryDirectory() as tmp:
        res = run_budget_config_task(
            video_ids=bundle["video_ids"],
            X=X,
            Y=Y_gain,
            fixed_vb=np.zeros(len(fixed_vb)),
            notta_vb=np.full(len(fixed_vb), np.nan),
            grid_runs=grid,
            output_dir=Path(tmp),
            seed=seed,
            n_folds=n_folds,
        )
    pol = _policy_from_budget_task(res)
    mask = labeled_mask(fixed_vb, Y)
    return _row(
        name,
        pol,
        n=int(mask.sum()),
        extra={"n_features": int(X.shape[1]), "target": "delta_vbench_total"},
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--series-root",
        type=Path,
        default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot",
    )
    ap.add_argument(
        "--feature-date",
        type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-06",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    profile_csv = args.feature_date / "vae_latent_profile_features.csv"
    if not profile_csv.is_file():
        print(f"[error] missing {profile_csv} — run extract_vae_latent_profile first", file=sys.stderr)
        return 2

    out = args.output_dir or (
        args.feature_date / "vae_latent_profile_router"
    )
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_pilot_bundle(args.series_root, args.feature_date)
    impute = compute_impute(bundle["video_ids"], bundle["features"], bundle["feat_names"])
    X_base = build_feature_matrix(
        bundle["video_ids"], bundle["features"], bundle["feat_names"], impute=impute,
    )
    X_probe, feat_names = build_probe_features(
        bundle, bundle["video_ids"], X_base, bundle["feat_names"], probe_runs=PROBE2,
    )
    Y = bundle["Y_total"]
    fixed_vb = bundle["fixed_vb"]
    grid = bundle["grid_runs"]

    vae_idx = [i for i, n in enumerate(feat_names) if n.startswith("vae_")]
    probe_idx = [i for i, n in enumerate(feat_names) if n.startswith("probe_")]
    base_idx = [i for i in range(len(feat_names)) if i not in set(vae_idx) and i not in set(probe_idx)]

    if not vae_idx:
        print("[error] no vae_* columns in feature bundle — check profile CSV join", file=sys.stderr)
        return 2

    X_vae_block = X_probe[:, vae_idx]
    X_probe_only = X_probe[:, probe_idx]
    X_phase0 = X_probe[:, base_idx] if base_idx else X_base

    configs = {
        "baseline_exp7": np.column_stack([X_phase0, X_probe_only]),
        "vae_profile_probe": np.column_stack([X_vae_block, X_probe_only]),
        "vae_profile_full": X_probe,
    }
    if not np.any(np.isfinite(X_vae_block)):
        print("[warn] vae block all NaN — profile CSV may not overlap video_ids", file=sys.stderr)

    results = {}
    for name, X in configs.items():
        print(f"[run] {name}  X.shape={X.shape}", file=sys.stderr)
        row = _run_ridge_gain_router(
            name, X, bundle, Y, fixed_vb, grid,
            seed=args.seed, n_folds=args.n_folds,
        )
        results[name] = row
        (out / f"{name}.json").write_text(
            json.dumps({"row": row}, indent=2), encoding="utf-8",
        )
        print(f"  captured={row.get('captured_pct')}", file=sys.stderr)

    best_name = max(
        EXPERIMENTS,
        key=lambda n: results[n].get("captured_pct") or -1.0,
    )
    best_cap = results[best_name].get("captured_pct") or float("nan")
    delta = best_cap - BASELINE_BEST if math.isfinite(best_cap) else float("nan")

    lines = [
        "# VAE latent profile router @ N=200",
        "",
        f"Profile CSV: `{profile_csv}`",
        "",
        "| Experiment | Features | Captured % | Match % | vs exp7 (12.8%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in EXPERIMENTS:
        r = results[name]
        cap = r.get("captured_pct")
        cap_s = f"{cap:.1f}" if cap is not None else "—"
        mr = r.get("match_rate")
        mr_s = f"{100 * mr:.1f}" if mr is not None else "—"
        d = (cap - BASELINE_BEST) if cap is not None else float("nan")
        d_s = f"{d:+.1f}" if math.isfinite(d) else "—"
        lines.append(
            f"| `{name}` | {r.get('n_features', '—')} | {cap_s} | {mr_s} | {d_s} |"
        )

    lines += [
        "",
        f"**Best:** `{best_name}` at **{best_cap:.1f}%** ({delta:+.1f} pp vs exp7).",
        "",
        "Interpretation: if `vae_profile_*` does not beat baseline by >2pp, richer VAE",
        "pooling alone is unlikely to unlock routing; next step is learned verifiers",
        "on probe outputs (VideoScore / VideoAlign).",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}/summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
