#!/usr/bin/env python3
"""Deploy-strict VBench config router @ pilot N=200 — VAE inference embedding ONLY.

**Router input x(v):** pooled statistics of the LongCat-VAE latent tensor produced
by ``encode_video`` on the TTA-visible pixel window — the same encode every
LongCat inference / AdaSteer pass already performs. Loaded from
``vae_latent_profile_features.csv`` (~130 dims).

**Explicitly excluded from router inputs:**
  * video_features.csv (CLIP, DINO, cuts, Laplacian, …)
  * AdaSteer / NOTTA metrics, probe PSNR/SSIM, Tier-3 LoRA probes
  * diffusion-OOD DiT forwards, bpp/FFT, motion, loss-variance, VAE decode rec-error

**Offline labels only (not deployed as features):** measured VBench total per
12-config grid from the pilot sweep — used once to fit ridge weights.

Usage:
  python3 scripts/run_deploy_strict_router_experiments.py
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.budget_routing_common import labeled_mask, load_pilot_bundle  # noqa: E402
from scripts.run_budget_routing_experiments import _policy_from_budget_task  # noqa: E402
from scripts.run_vbench_gain_prediction_experiments import _row  # noqa: E402
from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
    run_budget_config_task,
)

EXPERIMENT_NAME = "vae_inference_embedding"
EXPERIMENT_DESC = (
    "~130-d LongCat-VAE latent profile from encode_video on input video only "
    "(inference-path cache; no other feature CSVs)"
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
        print(
            f"[error] missing {profile_csv}\n"
            "  Run: bash sweep_experiment/sbatch/submit_vae_latent_profile_pilot.sh",
            file=sys.stderr,
        )
        return 2

    bundle = load_pilot_bundle(
        args.series_root,
        args.feature_date,
        feature_sources={
            "include_video_features": False,
            "vae_latent_profile": True,
            "fast_pixel": False,
            "vae_recerr": False,
            "ood": False,
            "tier3": False,
            "motion": False,
            "loss_var": False,
            "flow": False,
        },
    )
    feat_names = bundle["feat_names"]
    if not feat_names or not all(n.startswith("vae_") for n in feat_names):
        print(
            f"[error] expected vae_* columns only, got: {feat_names[:8]}...",
            file=sys.stderr,
        )
        return 2

    video_ids = bundle["video_ids"]
    impute = compute_impute(video_ids, bundle["features"], feat_names)
    X = build_feature_matrix(video_ids, bundle["features"], feat_names, impute=impute)
    if not np.any(np.isfinite(X)):
        print("[error] VAE feature matrix all NaN — check video_id overlap", file=sys.stderr)
        return 2

    print(
        f"[run] {EXPERIMENT_NAME}: {len(feat_names)} features from {profile_csv.name}",
        file=sys.stderr,
    )

    with tempfile.TemporaryDirectory() as tmp:
        res = run_budget_config_task(
            video_ids=video_ids,
            X=X,
            Y=bundle["Y_total"],
            fixed_vb=bundle["fixed_vb"],
            notta_vb=np.full(len(video_ids), np.nan),
            grid_runs=bundle["grid_runs"],
            output_dir=Path(tmp),
            seed=args.seed,
            n_folds=args.n_folds,
        )
    pol = _policy_from_budget_task(res)
    mask = labeled_mask(bundle["fixed_vb"], bundle["Y_total"])
    row = _row(
        EXPERIMENT_NAME,
        pol,
        n=int(mask.sum()),
        extra={
            "n_features": int(X.shape[1]),
            "description": EXPERIMENT_DESC,
            "feature_csv": str(profile_csv),
            "feature_prefix": "vae_",
        },
    )

    out = args.output_dir or (args.feature_date / "deploy_strict_router")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{EXPERIMENT_NAME}.json").write_text(
        json.dumps({"row": row, "deploy_rule": EXPERIMENT_DESC}, indent=2),
        encoding="utf-8",
    )

    cap = row.get("captured_pct")
    cap_s = f"{cap:.1f}" if cap is not None else "—"
    mr = row.get("match_rate")
    mr_s = f"{100 * mr:.1f}" if mr is not None else "—"
    pg = row.get("policy_gain")
    pg_s = f"{pg:+.4f}" if pg is not None else "—"

    summary = "\n".join([
        "# Deploy-strict router @ N=200 — VAE inference embedding only",
        "",
        "**Router input:** `vae_latent_profile_features.csv` only (~130-d). "
        "Produced by `extract_vae_latent_profile_features.py` → same `encode_video` "
        "as LongCat inference. **No** CLIP/DINO/bpp/OOD/Tier-3/probe/TTA metrics.",
        "",
        f"Feature CSV: `{profile_csv}`",
        "",
        "| Experiment | # feat | Captured % | Match % | Δ vs fixed S10 |",
        "|---|---:|---:|---:|---:|",
        f"| `{EXPERIMENT_NAME}` | {row.get('n_features')} | {cap_s} | {mr_s} | {pg_s} |",
        "",
        "**Deploy pipeline:** encode video once (cache latent profile) → ridge pick "
        "config → one AdaSteer run.",
        "",
    ])
    (out / "summary.md").write_text(summary, encoding="utf-8")
    print(f"  captured={cap_s}% match={mr_s}% delta_fixed={pg_s}", file=sys.stderr)
    print(f"Wrote {out}/summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
