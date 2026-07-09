#!/usr/bin/env python3
"""Cross-metric evaluation of deploy routers @ pilot N=200.

For each routing policy (VBench ridge pickers + baselines), look up the
**already-generated** per-video outputs from ``panda_ood_budget_pilot`` and
report mean PSNR / SSIM / LPIPS on the router-selected config per video.

VBench routing is trained on VBench total only; this script answers whether
selected configs also move reconstruction / perceptual metrics.

FVD/FID are **population** distributional metrics: per-video lookup is invalid.
The script (a) reports merged_summary FVD/FID for fixed grid configs + NOTTA,
(b) builds symlink policy dirs for router picks (for ``eval_fvd.py`` when mp4s
exist), and (c) optionally runs FVD eval with ``--run-fvd``.

Usage:
  python3 scripts/analyze_deploy_router_aux_metrics.py

  python3 scripts/analyze_deploy_router_aux_metrics.py \\
      --series-root sweep_experiment/results/panda_ood_budget_pilot \\
      --feature-date sweep_experiment/reports/per_video_analysis/2026-07-06 \\
      --output-dir sweep_experiment/reports/per_video_analysis/2026-07-06/deploy_router_aux_metrics \\
      --run-fvd --gt-cache gt_caches/panda_1000_longcat.npz
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    FIXED_ADA_RUN_ID,
    NOTTA_RUN_ID,
    discover_runs,
    load_merged_summary,
    load_run_all_metrics,
)
from scripts.budget_routing_common import (  # noqa: E402
    labeled_mask,
    load_metric_matrix,
    load_pilot_bundle,
)
from scripts.correlate_tta_gain_with_features import spearman_rho  # noqa: E402
from scripts.run_deploy_strict_router_experiments import (  # noqa: E402
    EXPERIMENT_SPECS,
    _load_bundle,
)
from scripts.train_vbench_headroom_router import (  # noqa: E402
    build_feature_matrix,
    compute_impute,
    run_budget_config_task,
)
from scripts.caption_utils import canonical_video_id  # noqa: E402
from sweep_experiment.scripts.build_oracle_policy_dirs import (  # noqa: E402
    find_mp4,
    _load_chunk_summary_order,
)

ROUTER_EXPERIMENTS = ("video_caption_only", "vae_inference_embedding")

METRIC_SPECS: Tuple[Tuple[str, bool], ...] = (
    ("psnr", True),
    ("ssim", True),
    ("lpips", False),
)


def _fmt(v: Optional[float], *, decimals: int = 3) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"


def _fmt_delta(v: Optional[float], *, decimals: int = 3, pct_base: Optional[float] = None) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    s = f"{v:+.{decimals}f}"
    if pct_base is not None and abs(pct_base) > 1e-9:
        s += f" ({100 * v / pct_base:+.2f}%)"
    return s


def load_notta_metrics(
    series_root: Path,
    video_ids: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    runs = discover_runs(series_root)
    if NOTTA_RUN_ID not in runs:
        return {}
    per_vid = load_run_all_metrics(runs[NOTTA_RUN_ID])
    return {vid: per_vid[vid] for vid in video_ids if vid in per_vid}


def mean_notta_metric(
    notta: Dict[str, Dict[str, float]],
    video_ids: Sequence[str],
    metric: str,
    valid: np.ndarray,
) -> float:
    vals = [
        notta[video_ids[i]][metric]
        for i in range(len(video_ids))
        if valid[i] and video_ids[i] in notta and metric in notta[video_ids[i]]
    ]
    return float(np.mean(vals)) if vals else float("nan")


def mean_from_picks(
    picks: np.ndarray,
    M: np.ndarray,
    *,
    valid: np.ndarray,
) -> float:
    vals: List[float] = []
    n, k = M.shape
    for i in range(n):
        if not valid[i]:
            continue
        j = int(picks[i])
        if j < 0 or j >= k:
            continue
        v = M[i, j]
        if np.isfinite(v):
            vals.append(float(v))
    return float(np.mean(vals)) if vals else float("nan")


def mean_oracle(M: np.ndarray, *, valid: np.ndarray, higher_is_better: bool) -> float:
    vals: List[float] = []
    for i in range(M.shape[0]):
        if not valid[i]:
            continue
        row = M[i]
        if np.all(np.isnan(row)):
            continue
        v = float(np.nanmax(row) if higher_is_better else np.nanmin(row))
        if np.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def metric_headroom_capture(
    policy_mean: float,
    fixed_mean: float,
    oracle_mean: float,
) -> Optional[float]:
    headroom = oracle_mean - fixed_mean
    if not (np.isfinite(headroom) and abs(headroom) > 1e-9):
        return None
    return (policy_mean - fixed_mean) / headroom


def per_video_metric_values(
    picks: np.ndarray,
    M: np.ndarray,
    *,
    valid: np.ndarray,
) -> np.ndarray:
    out = np.full(M.shape[0], np.nan, dtype=float)
    for i in range(M.shape[0]):
        if not valid[i]:
            continue
        j = int(picks[i])
        if 0 <= j < M.shape[1]:
            v = M[i, j]
            if np.isfinite(v):
                out[i] = float(v)
    return out


def run_router_oof_picks(
    name: str,
    bundle: dict,
    feat_names: List[str],
    *,
    output_dir: Path,
    seed: int,
    n_folds: int,
) -> Tuple[np.ndarray, dict]:
    video_ids = bundle["video_ids"]
    impute = compute_impute(video_ids, bundle["features"], feat_names)
    X = build_feature_matrix(video_ids, bundle["features"], feat_names, impute=impute)
    res = run_budget_config_task(
        video_ids=video_ids,
        X=X,
        Y=bundle["Y_total"],
        fixed_vb=bundle["fixed_vb"],
        notta_vb=np.full(len(video_ids), np.nan),
        grid_runs=bundle["grid_runs"],
        output_dir=output_dir / name,
        seed=seed,
        n_folds=n_folds,
    )
    picks = np.full(len(video_ids), -1, dtype=int)
    grid = bundle["grid_runs"]
    csv_path = output_dir / name / "budget_config_oof_predictions.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            rid = row.get("picked_run", "")
            if vid in video_ids and rid in grid:
                picks[video_ids.index(vid)] = grid.index(rid)
    return picks, res


def _index_grid_videos(series_root: Path, run_id: str) -> Dict[str, Path]:
    """Map canonical video_id -> generated mp4 for one grid config."""
    run_dir = series_root / run_id
    if not run_dir.is_dir():
        return {}

    out: Dict[str, Path] = {}
    chunk_dirs = sorted(run_dir.glob("chunk_*/"))
    if not chunk_dirs:
        chunk_dirs = [run_dir]

    for chunk_dir in chunk_dirs:
        videos_dir = chunk_dir / "videos"
        summary_path = chunk_dir / "summary.json"
        if not summary_path.exists():
            continue
        with summary_path.open(encoding="utf-8") as f:
            summary = json.load(f)
        idx_by_name = _load_chunk_summary_order(summary)

        for rec in summary.get("per_video_results", summary.get("results", [])):
            if not rec.get("success", False):
                continue
            vname = rec.get("video_name", "")
            vid = canonical_video_id(vname)
            if not vid:
                continue
            mp4 = find_mp4(videos_dir, vname, idx_by_name)
            if mp4 is None:
                op = rec.get("output_path")
                if op:
                    p = Path(op)
                    if p.exists() and p.suffix.lower() == ".mp4":
                        mp4 = p
            if mp4 is not None and mp4.exists():
                out[vid] = mp4.resolve()
    return out


def build_router_fvd_dir(
    *,
    policy_name: str,
    picks: np.ndarray,
    video_ids: Sequence[str],
    grid_runs: Sequence[str],
    series_root: Path,
    output_root: Path,
    valid: np.ndarray,
) -> Tuple[int, int]:
    video_index = {rid: _index_grid_videos(series_root, rid) for rid in grid_runs}
    out_dir = output_root / policy_name
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    linked = 0
    skipped = 0
    manifest: List[dict] = []
    for i, vid in enumerate(video_ids):
        if not valid[i]:
            skipped += 1
            continue
        j = int(picks[i])
        if j < 0 or j >= len(grid_runs):
            skipped += 1
            continue
        rid = grid_runs[j]
        src = video_index.get(rid, {}).get(vid)
        if src is None:
            skipped += 1
            continue
        dst = videos_dir / f"{vid}.mp4"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(src, dst)
        linked += 1
        manifest.append({"video_id": vid, "picked_run": rid, "source_mp4": str(src)})

    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "policy": policy_name,
                "series_root": str(series_root),
                "linked_videos": linked,
                "skipped_videos": skipped,
                "entries": manifest,
            },
            f,
            indent=2,
        )
    return linked, skipped


def run_fvd_eval(
    *,
    gen_dir: Path,
    gt_cache: Path,
    out_json: Path,
    device: str = "cuda",
) -> Optional[dict]:
    eval_script = _REPO / "sweep_experiment/scripts/eval_fvd.py"
    if not eval_script.is_file():
        return None
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(eval_script),
        "--gen-dir",
        str(gen_dir),
        "--gt-cache",
        str(gt_cache),
        "--num-cond-frames",
        "14",
        "--num-gen-frames",
        "14",
        "--min-videos",
        "50",
        "--output",
        str(out_json),
        "--device",
        device,
        "--force",
    ]
    rc = subprocess.call(cmd, cwd=str(_REPO))
    if rc != 0 or not out_json.exists():
        return None
    return json.loads(out_json.read_text(encoding="utf-8"))


def population_fvd_fid_table(series_root: Path, grid_runs: Sequence[str]) -> List[dict]:
    runs = discover_runs(series_root)
    rows: List[dict] = []
    for label, rid in (
        ("NOTTA", NOTTA_RUN_ID),
        ("Fixed AdaSteer (S10_LR5e3)", FIXED_ADA_RUN_ID),
    ):
        if rid in runs:
            m = load_merged_summary(runs[rid])
            rows.append(
                {
                    "policy": label,
                    "run_id": rid,
                    "fvd": m.get("fvd"),
                    "fid": m.get("fid"),
                    "n": m.get("num_successful") or m.get("num_videos"),
                }
            )
    for rid in grid_runs:
        if rid not in runs:
            continue
        m = load_merged_summary(runs[rid])
        rows.append(
            {
                "policy": f"grid:{rid}",
                "run_id": rid,
                "fvd": m.get("fvd"),
                "fid": m.get("fid"),
                "n": m.get("num_successful") or m.get("num_videos"),
            }
        )
    return rows


def analyze_policy_on_metrics(
    name: str,
    picks: np.ndarray,
    bundle: dict,
    metric_mats: Dict[str, np.ndarray],
    notta: Dict[str, Dict[str, float]],
    *,
    valid: np.ndarray,
    vb_policy: Optional[dict] = None,
) -> dict:
    video_ids = bundle["video_ids"]
    grid = bundle["grid_runs"]
    fixed_j = grid.index(bundle["fixed_run"])
    fixed_picks = np.full(len(video_ids), fixed_j, dtype=int)

    out: dict = {"name": name, "metrics": {}, "n_valid": int(valid.sum())}
    if vb_policy is not None:
        out["vbench"] = {
            "mean_policy": vb_policy["policy"].get("mean_policy_vbench"),
            "mean_fixed": vb_policy["policy"].get("mean_fixed_vbench"),
            "captured_pct": 100 * vb_policy["policy"].get("fraction_oracle_captured", float("nan")),
            "match_rate_pct": 100 * vb_policy.get("oof_oracle_match_rate", float("nan")),
        }

    for metric, higher in METRIC_SPECS:
        M = metric_mats[metric]
        fixed_mean = mean_from_picks(fixed_picks, M, valid=valid)
        policy_mean = mean_from_picks(picks, M, valid=valid)
        oracle_mean = mean_oracle(M, valid=valid, higher_is_better=higher)
        notta_mean = mean_notta_metric(notta, video_ids, metric, valid)
        cap = metric_headroom_capture(policy_mean, fixed_mean, oracle_mean)
        out["metrics"][metric] = {
            "policy_mean": policy_mean,
            "fixed_mean": fixed_mean,
            "oracle_mean": oracle_mean,
            "notta_mean": notta_mean,
            "delta_vs_fixed": policy_mean - fixed_mean,
            "delta_vs_notta": policy_mean - notta_mean if np.isfinite(notta_mean) else float("nan"),
            "oracle_headroom": oracle_mean - fixed_mean,
            "captured_fraction": cap,
            "captured_pct": 100 * cap if cap is not None else float("nan"),
        }

    if vb_policy is not None:
        Y = bundle["Y_total"]
        vb_per = per_video_metric_values(picks, Y, valid=valid)
        fixed_vb = bundle["fixed_vb"]
        psnr_per = per_video_metric_values(picks, metric_mats["psnr"], valid=valid)
        fixed_psnr = per_video_metric_values(fixed_picks, metric_mats["psnr"], valid=valid)
        vb_gain = vb_per - fixed_vb
        psnr_gain = psnr_per - fixed_psnr
        mask = valid & np.isfinite(vb_gain) & np.isfinite(psnr_gain)
        rho = spearman_rho(vb_gain[mask], psnr_gain[mask]) if mask.sum() >= 10 else None
        out["cross_metric"] = {
            "spearman_vbench_gain_vs_psnr_gain": rho,
            "n_pairs": int(mask.sum()),
        }
    return out


def write_markdown_report(
    out_path: Path,
    results: List[dict],
    pop_fvd: List[dict],
    fvd_results: Dict[str, dict],
    *,
    series_root: Path,
) -> None:
    lines = [
        "# Deploy router — cross-metric analysis @ N=200",
        "",
        f"**Series:** `{series_root}`",
        "",
        "Per-video metrics are looked up from **existing** grid outputs for the",
        "OOF router-selected config. VBench routing optimizes VBench total only.",
        "",
        "## PSNR / SSIM / LPIPS (per-video mean on selected config)",
        "",
        "| Policy | VB captured % | PSNR (dB) | Δ vs fixed | PSNR cap % | SSIM | Δ vs fixed | LPIPS | Δ vs fixed | ρ(VB gain, ΔPSNR) |",
        "|--------|-------------:|----------:|-----------:|-----------:|-----:|-----------:|------:|-----------:|------------------:|",
    ]
    for r in results:
        vb = r.get("vbench") or {}
        m = r["metrics"]
        psnr = m["psnr"]
        ssim = m["ssim"]
        lpips = m["lpips"]
        rho = (r.get("cross_metric") or {}).get("spearman_vbench_gain_vs_psnr_gain")
        rho_s = f"{rho:.2f}" if rho is not None else "—"
        lines.append(
            f"| {r['name']} "
            f"| {_fmt(vb.get('captured_pct'), decimals=1)} "
            f"| {_fmt(psnr['policy_mean'])} "
            f"| {_fmt_delta(psnr['delta_vs_fixed'], decimals=3)} "
            f"| {_fmt(psnr.get('captured_pct'), decimals=1)} "
            f"| {_fmt(ssim['policy_mean'], decimals=4)} "
            f"| {_fmt_delta(ssim['delta_vs_fixed'], decimals=4)} "
            f"| {_fmt(lpips['policy_mean'], decimals=4)} "
            f"| {_fmt_delta(lpips['delta_vs_fixed'], decimals=4)} "
            f"| {rho_s} |"
        )

    lines += [
        "",
        "**Capt % (PSNR/SSIM/LPIPS):** fraction of oracle−fixed headroom recovered on that metric.",
        "",
        "## Population FVD / FID (merged_summary per run — not per-video mix)",
        "",
        "Router policies mix configs per video; FVD/FID requires a symlink dir + ``eval_fvd.py``.",
        "",
        "| Source | FVD ↓ | FID ↓ | N |",
        "|--------|------:|------:|--:|",
    ]
    for row in pop_fvd:
        if row["policy"].startswith("grid:"):
            continue
        lines.append(
            f"| {row['policy']} "
            f"| {_fmt(row.get('fvd'), decimals=1)} "
            f"| {_fmt(row.get('fid'), decimals=1)} "
            f"| {row.get('n', '—')} |"
        )

    if fvd_results:
        lines += ["", "### Router policy FVD (symlink eval)", ""]
        lines += ["| Policy | FVD ↓ | FID ↓ | linked |", "|--------|------:|------:|-------:|"]
        for name, blob in fvd_results.items():
            lines.append(
                f"| {name} "
                f"| {_fmt(blob.get('fvd'), decimals=1)} "
                f"| {_fmt(blob.get('fid'), decimals=1)} "
                f"| {blob.get('num_valid_pairs', '—')} |"
            )
    else:
        lines += [
            "",
            "> **FVD for router picks:** run with ``--run-fvd`` after mp4s exist",
            "> (``NO_SAVE_VIDEOS=0`` on budget pilot). Symlink dirs under ``fvd_policies/``.",
        ]

    lines += [
        "",
        "## Interpretation checklist",
        "",
        "1. **VBench routing can decouple from PSNR** — check ρ and PSNR captured %.",
        "2. **LPIPS may move opposite PSNR** when configs trade sharpness vs fidelity.",
        "3. **FVD is the honest distributional check** — compare router symlink FVD to fixed row.",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


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
    ap.add_argument(
        "--experiments",
        nargs="*",
        default=list(ROUTER_EXPERIMENTS),
        help="Deploy router experiment names",
    )
    ap.add_argument("--run-fvd", action="store_true")
    ap.add_argument(
        "--gt-cache",
        type=Path,
        default=_REPO / "gt_caches/panda_1000_longcat.npz",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not args.series_root.is_dir():
        print(f"[error] series root missing: {args.series_root}", file=sys.stderr)
        return 2

    out = args.output_dir or (args.feature_date / "deploy_router_aux_metrics")
    out.mkdir(parents=True, exist_ok=True)

    bundle = load_pilot_bundle(args.series_root, args.feature_date)
    video_ids = bundle["video_ids"]
    grid = bundle["grid_runs"]
    valid = labeled_mask(bundle["fixed_vb"], bundle["Y_total"])
    runs = discover_runs(args.series_root)

    metric_mats = {
        "psnr": bundle["psnr"],
        "ssim": bundle["ssim"],
        "lpips": load_metric_matrix(runs, grid, video_ids, "lpips"),
    }
    notta = load_notta_metrics(args.series_root, video_ids)

    fixed_j = grid.index(bundle["fixed_run"])
    baseline_picks = {
        "fixed_S10_LR5e3": np.full(len(video_ids), fixed_j, dtype=int),
        "oracle_vbench": np.nanargmax(bundle["Y_total"], axis=1).astype(int),
        "oracle_psnr": np.nanargmax(bundle["psnr"], axis=1).astype(int),
    }

    all_results: List[dict] = []

    for bname, picks in baseline_picks.items():
        all_results.append(
            analyze_policy_on_metrics(
                bname, picks, bundle, metric_mats, notta, valid=valid,
            )
        )

    fvd_policy_dir = out / "fvd_policies"
    fvd_results: Dict[str, dict] = {}

    for exp in args.experiments:
        if exp not in EXPERIMENT_SPECS:
            print(f"[warn] unknown experiment {exp}", file=sys.stderr)
            continue
        spec = EXPERIMENT_SPECS[exp]
        bundle_exp, feat_names, _ = _load_bundle(args.series_root, args.feature_date, spec)
        picks, res = run_router_oof_picks(
            exp,
            bundle_exp,
            feat_names,
            output_dir=out / "router_runs",
            seed=args.seed,
            n_folds=args.n_folds,
        )
        label = {
            "video_caption_only": "router_video_caption (Block A)",
            "vae_inference_embedding": "router_vae_pooled (Block C)",
        }.get(exp, exp)
        all_results.append(
            analyze_policy_on_metrics(
                label,
                picks,
                bundle,
                metric_mats,
                notta,
                valid=valid,
                vb_policy=res,
            )
        )
        linked, skipped = build_router_fvd_dir(
            policy_name=exp,
            picks=picks,
            video_ids=video_ids,
            grid_runs=grid,
            series_root=args.series_root,
            output_root=fvd_policy_dir,
            valid=valid,
        )
        print(f"[fvd-dir] {exp}: linked={linked} skipped={skipped}", file=sys.stderr)
        if args.run_fvd and linked >= 50 and args.gt_cache.exists():
            fvd_json = fvd_policy_dir / exp / "fvd.json"
            blob = run_fvd_eval(
                gen_dir=fvd_policy_dir / exp / "videos",
                gt_cache=args.gt_cache,
                out_json=fvd_json,
                device=args.device,
            )
            if blob:
                fvd_results[label] = blob

    pop_fvd = population_fvd_fid_table(args.series_root, grid)

    payload = {
        "series_root": str(args.series_root),
        "n_videos": int(valid.sum()),
        "policies": all_results,
        "population_fvd_fid": pop_fvd,
        "router_fvd": fvd_results,
    }
    (out / "results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_markdown_report(
        out / "summary.md",
        all_results,
        pop_fvd,
        fvd_results,
        series_root=args.series_root,
    )
    print(f"Wrote {out}/summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
