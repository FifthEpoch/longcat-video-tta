#!/usr/bin/env python3
"""Align VBench-targeted vs PSNR-targeted deploy routers @ pilot N=200.

Both use the same 9-d Block A features; only the ridge label matrix differs.
Compares OOF config picks per video: agreement rate, oracle overlap, and
realized metrics when routers disagree.

Usage:
  python3 scripts/analyze_router_objective_alignment.py

  python3 scripts/analyze_router_objective_alignment.py \\
      --vb-picks-csv sweep_experiment/reports/per_video_analysis/2026-07-06/deploy_router_aux_metrics/router_runs/video_caption_only/budget_config_oof_predictions.csv \\
      --psnr-picks-csv sweep_experiment/reports/per_video_analysis/2026-07-06/deploy_psnr_router/budget_config_oof_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.budget_routing_common import labeled_mask, load_pilot_bundle  # noqa: E402
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


def _load_picks_csv(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row.get("video_id", "").strip()
            rid = row.get("picked_run", "").strip()
            if vid and rid:
                out[vid] = rid
    return out


def _picks_array(
    video_ids: Sequence[str],
    picks_by_vid: Dict[str, str],
    grid: Sequence[str],
) -> np.ndarray:
    arr = np.full(len(video_ids), -1, dtype=int)
    for i, vid in enumerate(video_ids):
        rid = picks_by_vid.get(vid)
        if rid and rid in grid:
            arr[i] = grid.index(rid)
    return arr


def _oracle_picks(M: np.ndarray, valid: np.ndarray) -> np.ndarray:
    picks = np.full(M.shape[0], -1, dtype=int)
    for i in range(M.shape[0]):
        if not valid[i]:
            continue
        row = M[i]
        if np.any(np.isfinite(row)):
            picks[i] = int(np.nanargmax(row))
    return picks


def _realized(M: np.ndarray, picks: np.ndarray, i: int) -> float:
    j = int(picks[i])
    if j < 0 or j >= M.shape[1]:
        return float("nan")
    v = M[i, j]
    return float(v) if np.isfinite(v) else float("nan")


def _fmt(v: Optional[float], *, d: int = 4) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:.{d}f}"


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "—"
    return f"{100 * num / den:.1f}%"


def run_router(
    bundle: dict,
    feat_names: List[str],
    Y: np.ndarray,
    fixed_y: np.ndarray,
    *,
    out_dir: Path,
    seed: int,
    n_folds: int,
) -> Tuple[Dict[str, str], Path]:
    video_ids = bundle["video_ids"]
    impute = compute_impute(video_ids, bundle["features"], feat_names)
    X = build_feature_matrix(video_ids, bundle["features"], feat_names, impute=impute)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_budget_config_task(
        video_ids=video_ids,
        X=X,
        Y=Y,
        fixed_vb=fixed_y,
        notta_vb=np.full(len(video_ids), np.nan),
        grid_runs=bundle["grid_runs"],
        output_dir=out_dir,
        seed=seed,
        n_folds=n_folds,
    )
    csv_path = out_dir / "budget_config_oof_predictions.csv"
    return _load_picks_csv(csv_path), csv_path


def build_pair_crosstab(
    vb_pick: np.ndarray,
    psnr_pick: np.ndarray,
    grid: Sequence[str],
    valid: np.ndarray,
) -> List[Tuple[str, str, int]]:
    counts: Counter = Counter()
    for i in range(len(vb_pick)):
        if not valid[i]:
            continue
        vj, pj = int(vb_pick[i]), int(psnr_pick[i])
        if vj < 0 or pj < 0:
            continue
        counts[(grid[vj], grid[pj])] += 1
    return sorted(counts.items(), key=lambda x: -x[2])


def analyze(
    *,
    video_ids: List[str],
    grid: List[str],
    vb_pick: np.ndarray,
    psnr_pick: np.ndarray,
    Y_vb: np.ndarray,
    psnr: np.ndarray,
    fixed_vb: np.ndarray,
    fixed_psnr: np.ndarray,
    valid: np.ndarray,
) -> dict:
    oracle_vb = _oracle_picks(Y_vb, valid)
    oracle_psnr = _oracle_picks(psnr, valid)

    n = int(valid.sum())
    agree = disagree = 0
    vb_wins_vb_on_disagree = psnr_wins_psnr_on_disagree = 0
    both_oracle_same = oracle_agree_count = 0

    rows: List[dict] = []
    for i, vid in enumerate(video_ids):
        if not valid[i]:
            continue
        vj, pj = int(vb_pick[i]), int(psnr_pick[i])
        oj_vb, oj_psnr = int(oracle_vb[i]), int(oracle_psnr[i])
        same_pick = vj == pj and vj >= 0
        oracle_same = oj_vb == oj_psnr and oj_vb >= 0
        if oracle_same:
            oracle_agree_count += 1

        vb_real = _realized(Y_vb, vb_pick, i)
        psnr_real_vb = _realized(Y_vb, psnr_pick, i)
        psnr_real = _realized(psnr, psnr_pick, i)
        vb_real_psnr = _realized(psnr, vb_pick, i)

        if same_pick:
            agree += 1
        else:
            disagree += 1
            if np.isfinite(vb_real) and np.isfinite(psnr_real_vb):
                if vb_real >= psnr_real_vb:
                    vb_wins_vb_on_disagree += 1
            if np.isfinite(psnr_real) and np.isfinite(vb_real_psnr):
                if psnr_real >= vb_real_psnr:
                    psnr_wins_psnr_on_disagree += 1

        rows.append({
            "video_id": vid,
            "vb_pick": grid[vj] if 0 <= vj < len(grid) else "",
            "psnr_pick": grid[pj] if 0 <= pj < len(grid) else "",
            "oracle_vb": grid[oj_vb] if 0 <= oj_vb < len(grid) else "",
            "oracle_psnr": grid[oj_psnr] if 0 <= oj_psnr < len(grid) else "",
            "picks_agree": int(same_pick),
            "oracles_agree": int(oracle_same),
            "vb_realized_vbench": vb_real,
            "psnr_pick_realized_vbench": psnr_real_vb,
            "psnr_realized_psnr": psnr_real,
            "vb_pick_realized_psnr": vb_real_psnr,
        })

    # Pick distribution overlap (Jaccard on sets of chosen configs)
    vb_set = {grid[int(vb_pick[i])] for i in range(len(video_ids))
              if valid[i] and 0 <= vb_pick[i] < len(grid)}
    psnr_set = {grid[int(psnr_pick[i])] for i in range(len(video_ids))
                if valid[i] and 0 <= psnr_pick[i] < len(grid)}
    jaccard = len(vb_set & psnr_set) / len(vb_set | psnr_set) if vb_set | psnr_set else float("nan")

    # Spearman: per-video rank of predicted winner vs other objective realized
    vb_from_vb = np.array([r["vb_realized_vbench"] for r in rows])
    vb_from_psnr = np.array([r["psnr_pick_realized_vbench"] for r in rows])
    psnr_from_psnr = np.array([r["psnr_realized_psnr"] for r in rows])
    psnr_from_vb = np.array([r["vb_pick_realized_psnr"] for r in rows])

    mask_vb = np.isfinite(vb_from_vb) & np.isfinite(vb_from_psnr)
    mask_psnr = np.isfinite(psnr_from_psnr) & np.isfinite(psnr_from_vb)

    crosstab = build_pair_crosstab(vb_pick, psnr_pick, grid, valid)
    top_pairs = crosstab[:12]

    # When oracles agree, router agreement
    oracle_same_router_agree = sum(
        1 for r in rows if r["oracles_agree"] and r["picks_agree"]
    )

    return {
        "n_videos": n,
        "pick_agree_count": agree,
        "pick_disagree_count": disagree,
        "oracle_agree_count": oracle_agree_count,
        "oracle_same_router_agree_count": oracle_same_router_agree,
        "pick_agreement_rate": agree / n if n else float("nan"),
        "pick_disagreement_rate": disagree / n if n else float("nan"),
        "oracle_agreement_rate": oracle_agree_count / n if n else float("nan"),
        "when_oracles_agree_router_agree_rate": (
            oracle_same_router_agree / oracle_agree_count if oracle_agree_count else float("nan")
        ),
        "vb_router_matches_vb_oracle_rate": float(
            np.mean((vb_pick[valid] == oracle_vb[valid]).astype(float))
        ),
        "psnr_router_matches_psnr_oracle_rate": float(
            np.mean((psnr_pick[valid] == oracle_psnr[valid]).astype(float))
        ),
        "when_disagree_vb_router_wins_vb_count": vb_wins_vb_on_disagree,
        "when_disagree_psnr_router_wins_psnr_count": psnr_wins_psnr_on_disagree,
        "config_jaccard": jaccard,
        "vb_configs_used": sorted(vb_set),
        "psnr_configs_used": sorted(psnr_set),
        "spearman_vb_realized_vb_pick_vs_psnr_pick": (
            spearman_rho(vb_from_vb[mask_vb], vb_from_psnr[mask_vb]) if mask_vb.sum() >= 10 else None
        ),
        "spearman_psnr_realized_psnr_pick_vs_vb_pick": (
            spearman_rho(psnr_from_psnr[mask_psnr], psnr_from_vb[mask_psnr]) if mask_psnr.sum() >= 10 else None
        ),
        "top_pick_pairs": [
            {"vb_pick": a, "psnr_pick": b, "count": c} for (a, b), c in top_pairs
        ],
        "per_video": rows,
    }


def write_summary(out: Path, report: dict, grid: List[str]) -> None:
    n = report["n_videos"]
    dis = report["when_disagree_vb_router_wins_vb_count"]
    dis_n = report["pick_disagree_count"]
    dis_psnr = report["when_disagree_psnr_router_wins_psnr_count"]

    lines = [
        "# Router objective alignment — VBench vs PSNR (9-d Block A) @ N=200",
        "",
        "Same features, different ridge labels. OOF config picks compared per video.",
        "",
        "## Pick overlap",
        "",
        f"| Metric | Value |",
        f"|--------|------:|",
        f"| Videos (labeled) | {n} |",
        f"| **Pick agreement** (same config) | **{_pct(report['pick_agree_count'], n)}** ({report['pick_agree_count']}/{n}) |",
        f"| Oracle agreement (VB oracle = PSNR oracle) | {_pct(report['oracle_agree_count'], n)} |",
        f"| When oracles agree → routers agree | {_pct(report['oracle_same_router_agree_count'], report['oracle_agree_count'])} |",
        f"| VB router matches VB oracle | {_pct(int(report['vb_router_matches_vb_oracle_rate'] * n), n)} |",
        f"| PSNR router matches PSNR oracle | {_pct(int(report['psnr_router_matches_psnr_oracle_rate'] * n), n)} |",
        f"| Config set Jaccard (configs ever picked) | {_fmt(report['config_jaccard'], d=3)} |",
        "",
        "## When routers **disagree** ({} videos)".format(dis_n),
        "",
        f"- VB router pick has **≥** PSNR-router pick on **realized VBench**: "
        f"{dis}/{dis_n} ({_pct(dis, dis_n)})",
        f"- PSNR router pick has **≥** VB-router pick on **realized PSNR**: "
        f"{dis_psnr}/{dis_n} ({_pct(dis_psnr, dis_n)})",
        "",
        "## Rank correlation of realized metrics (per-video)",
        "",
        f"- ρ(VBench from VB pick, VBench from PSNR pick): "
        f"{_fmt(report.get('spearman_vb_realized_vb_pick_vs_psnr_pick'), d=3)}",
        f"- ρ(PSNR from PSNR pick, PSNR from VB pick): "
        f"{_fmt(report.get('spearman_psnr_realized_psnr_pick_vs_vb_pick'), d=3)}",
        "",
        "## Top (VB pick → PSNR pick) pairs",
        "",
        "| VB pick | PSNR pick | Count |",
        "|---------|-----------|------:|",
    ]
    for p in report["top_pick_pairs"]:
        lines.append(f"| `{p['vb_pick']}` | `{p['psnr_pick']}` | {p['count']} |")

    lines += [
        "",
        "## Configs used",
        "",
        f"- **VB router:** {', '.join(f'`{c}`' for c in report['vb_configs_used'])}",
        f"- **PSNR router:** {', '.join(f'`{c}`' for c in report['psnr_configs_used'])}",
        "",
        "## Interpretation",
        "",
        "- **Low pick agreement** + **low oracle agreement** → objectives pull to different configs.",
        "- **High ρ on cross-realized metrics** → disagreements are small in metric space.",
        "- **When disagree:** each router should win on its own metric (sanity check).",
        "",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def write_per_video_csv(out: Path, rows: List[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with (out / "per_video_alignment.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


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
    ap.add_argument("--vb-picks-csv", type=Path, default=None)
    ap.add_argument("--psnr-picks-csv", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--rerun-routers", action="store_true")
    args = ap.parse_args()

    if not args.series_root.is_dir():
        print(f"[error] missing series: {args.series_root}", file=sys.stderr)
        return 2

    out = args.output_dir or (args.feature_date / "router_objective_alignment")
    out.mkdir(parents=True, exist_ok=True)

    spec = EXPERIMENT_SPECS["video_caption_only"]
    bundle, feat_names, _ = _load_bundle(args.series_root, args.feature_date, spec)
    video_ids = bundle["video_ids"]
    grid = list(bundle["grid_runs"])
    Y_vb = bundle["Y_total"]
    psnr = bundle["psnr"]
    fixed_j = grid.index(bundle["fixed_run"])
    fixed_vb = bundle["fixed_vb"]
    fixed_psnr = np.array([
        psnr[i, fixed_j] if np.isfinite(psnr[i, fixed_j]) else float("nan")
        for i in range(len(video_ids))
    ])
    valid = labeled_mask(fixed_vb, Y_vb) & labeled_mask(fixed_psnr, psnr)

    vb_csv = args.vb_picks_csv
    psnr_csv = args.psnr_picks_csv

    if args.rerun_routers or vb_csv is None or not vb_csv.is_file():
        vb_picks, vb_csv = run_router(
            bundle, feat_names, Y_vb, fixed_vb,
            out_dir=out / "_vb_router", seed=args.seed, n_folds=args.n_folds,
        )
    else:
        vb_picks = _load_picks_csv(vb_csv)

    if args.rerun_routers or psnr_csv is None or not psnr_csv.is_file():
        psnr_picks, psnr_csv = run_router(
            bundle, feat_names, psnr, fixed_psnr,
            out_dir=out / "_psnr_router", seed=args.seed, n_folds=args.n_folds,
        )
    else:
        psnr_picks = _load_picks_csv(psnr_csv)

    vb_arr = _picks_array(video_ids, vb_picks, grid)
    psnr_arr = _picks_array(video_ids, psnr_picks, grid)

    report = analyze(
        video_ids=video_ids,
        grid=grid,
        vb_pick=vb_arr,
        psnr_pick=psnr_arr,
        Y_vb=Y_vb,
        psnr=psnr,
        fixed_vb=fixed_vb,
        fixed_psnr=fixed_psnr,
        valid=valid,
    )
    report["sources"] = {"vb_picks_csv": str(vb_csv), "psnr_picks_csv": str(psnr_csv)}

    per_video = report.pop("per_video")
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_per_video_csv(out, per_video)
    write_summary(out, report, grid)

    print(
        f"Pick agreement: {100 * report['pick_agreement_rate']:.1f}% "
        f"(oracle agree {100 * report['oracle_agreement_rate']:.1f}%)",
        file=sys.stderr,
    )
    print(f"Wrote {out}/summary.md", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
