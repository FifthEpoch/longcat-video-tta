#!/usr/bin/env python3
"""Deploy router experiments @ pilot N=200 — structured feature blocks.

Feature space is built as **ordered blocks** (concatenated for ridge):

  Block A — ``video_caption`` (9): cuts + CLIP caption–video + DINO + Laplacian + RGB
  Block B — ``diffusion_ood`` (~20): frozen base DiT flow-matching loss @ 3 timesteps
            + score norms + latent summaries (Slide 2 / ``compute_diffusion_ood_score.py``)
  Block C — ``vae_inference`` (~130): LongCat ``encode_video`` latent profile

**Never used:** Tier-3 LoRA probes, AdaSteer/NOTTA metrics, probe PSNR/SSIM, bpp/FFT aux.

Experiments (``--run-all``):
  video_caption_only       — Block A
  diffusion_ood_only       — Block B
  video_caption_ood        — A + B  (**headline when OOD allowed**)
  vae_inference_embedding  — Block C only (prior deploy baseline @ 9.7%)
  video_caption_ood_vae    — A + B + C (full structured stack)

Usage:
  python3 scripts/run_deploy_strict_router_experiments.py --run-all
  python3 scripts/run_deploy_strict_router_experiments.py --experiment video_caption_ood
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.budget_routing_common import (  # noqa: E402
    DEPLOY_BLOCK_OOD,
    DEPLOY_BLOCK_VAE,
    DEPLOY_BLOCK_VIDEO_CORE,
    build_deploy_feature_keep,
    labeled_mask,
    load_pilot_bundle,
)
from scripts.run_budget_routing_experiments import _policy_from_budget_task  # noqa: E402
from scripts.run_vbench_gain_prediction_experiments import _row  # noqa: E402
from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
    run_budget_config_task,
)

EXPERIMENT_SPECS: Dict[str, dict] = {
    "video_caption_only": {
        "video_core": True,
        "ood": False,
        "vae_latent_profile": False,
        "description": "Block A: video + caption stats only (9-d)",
    },
    "diffusion_ood_only": {
        "video_core": False,
        "ood": True,
        "vae_latent_profile": False,
        "description": "Block B: diffusion-OOD only (~20-d)",
    },
    "video_caption_ood": {
        "video_core": True,
        "ood": True,
        "vae_latent_profile": False,
        "description": "Blocks A+B: video/caption + diffusion-OOD (~29-d)",
    },
    "vae_inference_embedding": {
        "video_core": False,
        "ood": False,
        "vae_latent_profile": True,
        "description": "Block C: VAE encode profile only (~130-d)",
    },
    "video_caption_ood_vae": {
        "video_core": True,
        "ood": True,
        "vae_latent_profile": True,
        "description": "Blocks A+B+C: full structured stack (~159-d)",
    },
}

ALL_EXPERIMENTS = tuple(EXPERIMENT_SPECS.keys())


def _feature_sources(spec: dict) -> dict:
    return {
        "include_video_features": False,
        "video_core": spec["video_core"],
        "ood": spec["ood"],
        "vae_latent_profile": spec["vae_latent_profile"],
        "fast_pixel": False,
        "vae_recerr": False,
        "tier3": False,
        "motion": False,
        "loss_var": False,
        "flow": False,
    }


def _required_csvs(feature_date: Path, spec: dict) -> List[Path]:
    paths: List[Path] = []
    if spec["video_core"]:
        paths.append(feature_date / "video_features.csv")
    if spec["ood"]:
        paths.append(feature_date / "diffusion_ood_scores.csv")
    if spec["vae_latent_profile"]:
        paths.append(feature_date / "vae_latent_profile_features.csv")
    return paths


def _load_bundle(
    series_root: Path,
    feature_date: Path,
    spec: dict,
) -> Tuple[dict, List[str], Dict[str, List[str]]]:
    _, block_map = build_deploy_feature_keep(
        feature_date,
        video_core=spec["video_core"],
        ood=spec["ood"],
        vae_latent_profile=spec["vae_latent_profile"],
    )
    bundle = load_pilot_bundle(
        series_root,
        feature_date,
        feature_sources=_feature_sources(spec),
    )
    return bundle, bundle["feat_names"], block_map


def _run_experiment(
    name: str,
    bundle: dict,
    feat_names: List[str],
    block_map: Dict[str, List[str]],
    spec: dict,
    *,
    seed: int,
    n_folds: int,
) -> dict:
    video_ids = bundle["video_ids"]
    impute = compute_impute(video_ids, bundle["features"], feat_names)
    X = build_feature_matrix(video_ids, bundle["features"], feat_names, impute=impute)
    if not np.any(np.isfinite(X)):
        raise ValueError(f"{name}: feature matrix all NaN")

    with tempfile.TemporaryDirectory() as tmp:
        res = run_budget_config_task(
            video_ids=video_ids,
            X=X,
            Y=bundle["Y_total"],
            fixed_vb=bundle["fixed_vb"],
            notta_vb=np.full(len(video_ids), np.nan),
            grid_runs=bundle["grid_runs"],
            output_dir=Path(tmp),
            seed=seed,
            n_folds=n_folds,
        )
    pol = _policy_from_budget_task(res)
    mask = labeled_mask(bundle["fixed_vb"], bundle["Y_total"])
    block_dims = {k: len(v) for k, v in block_map.items()}
    return _row(
        name,
        pol,
        n=int(mask.sum()),
        extra={
            "n_features": int(X.shape[1]),
            "description": spec["description"],
            "feature_blocks": block_map,
            "block_dims": block_dims,
        },
    )


def _block_table_md(block_map: Dict[str, List[str]]) -> List[str]:
    lines = [
        "### Feature blocks (concatenated in this order)",
        "",
        "| Block | # dims | Source |",
        "|-------|-------:|--------|",
    ]
    block_meta = {
        DEPLOY_BLOCK_VIDEO_CORE: "`video_features.csv` — cuts, CLIP, DINO, Laplacian, RGB",
        DEPLOY_BLOCK_OOD: "`diffusion_ood_scores.csv` — frozen base DiT @ t∈{100,500,900}",
        DEPLOY_BLOCK_VAE: "`vae_latent_profile_features.csv` — LongCat `encode_video` pools",
    }
    for block, cols in block_map.items():
        lines.append(f"| `{block}` | {len(cols)} | {block_meta.get(block, '—')} |")
    lines.append("")
    return lines


def _write_summary(
    out: Path,
    feature_date: Path,
    results: Dict[str, dict],
    block_map: Dict[str, List[str]],
) -> None:
    lines = [
        "# Deploy router @ N=200 — structured feature blocks",
        "",
        "**Block A** `video_caption` · **Block B** `diffusion_ood` · **Block C** `vae_inference`",
        "",
        "No Tier-3 / probe / TTA eval metrics. Offline labels = pilot 12-config VBench only.",
        "",
        f"Feature dir: `{feature_date}`",
        "",
    ]
    lines.extend(_block_table_md(block_map))
    lines += [
        "| Experiment | Blocks | # feat | Captured % | Match % | Δ vs fixed S10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    block_label = {
        "video_caption_only": "A",
        "diffusion_ood_only": "B",
        "video_caption_ood": "A+B",
        "vae_inference_embedding": "C",
        "video_caption_ood_vae": "A+B+C",
    }
    for name in ALL_EXPERIMENTS:
        if name not in results:
            continue
        r = results[name]
        cap = r.get("captured_pct")
        cap_s = f"{cap:.1f}" if cap is not None else "—"
        mr = r.get("match_rate")
        mr_s = f"{100 * mr:.1f}" if mr is not None else "—"
        pg = r.get("policy_gain")
        pg_s = f"{pg:+.4f}" if pg is not None else "—"
        lines.append(
            f"| `{name}` | {block_label[name]} | {r.get('n_features')} "
            f"| {cap_s} | {mr_s} | {pg_s} |"
        )
    scored = [n for n in ALL_EXPERIMENTS if n in results]
    if scored:
        best = max(scored, key=lambda n: results[n].get("captured_pct") or -1.0)
        best_cap = results[best].get("captured_pct")
        cap_str = f"{best_cap:.1f}" if best_cap is not None else "—"
        lines += [
            "",
            f"**Best in suite:** `{best}` @ **{cap_str}%** captured.",
            "",
        ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


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
    ap.add_argument("--run-all", action="store_true")
    ap.add_argument("--experiment", action="append", default=[])
    args = ap.parse_args()

    needed = ALL_EXPERIMENTS if args.run_all else tuple(args.experiment)
    if not needed:
        ap.error("pass --run-all or --experiment <name>")

    for name in needed:
        if name not in EXPERIMENT_SPECS:
            print(f"[error] unknown experiment {name}", file=sys.stderr)
            return 2
        for path in _required_csvs(args.feature_date, EXPERIMENT_SPECS[name]):
            if not path.is_file():
                print(f"[error] missing {path}", file=sys.stderr)
                return 2

    out = args.output_dir or (args.feature_date / "deploy_strict_router")
    out.mkdir(parents=True, exist_ok=True)

    _, ref_block_map = build_deploy_feature_keep(
        args.feature_date,
        video_core=True,
        ood=True,
        vae_latent_profile=(args.feature_date / "vae_latent_profile_features.csv").is_file(),
    )

    results: Dict[str, dict] = {}
    for name in needed:
        spec = EXPERIMENT_SPECS[name]
        print(f"[run] {name}: {spec['description']}", file=sys.stderr)
        bundle, feat_names, block_map = _load_bundle(
            args.series_root, args.feature_date, spec,
        )
        row = _run_experiment(
            name, bundle, feat_names, block_map, spec,
            seed=args.seed, n_folds=args.n_folds,
        )
        results[name] = row
        (out / f"{name}.json").write_text(
            json.dumps({"row": row, "spec": spec, "feature_blocks": block_map}, indent=2),
            encoding="utf-8",
        )
        print(
            f"  n_feat={row.get('n_features')} captured={row.get('captured_pct')}",
            file=sys.stderr,
        )

    _write_summary(out, args.feature_date, results, ref_block_map or block_map)
    print(f"Wrote {out}/summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
