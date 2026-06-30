#!/usr/bin/env python3
"""Oracle routing analysis for VBench++ (and SSIM/LPIPS) — mirrors PSNR oracles.

**Method oracle:** per video pick NOTTA / AdaSteer / LoRA / retrieval by max PSNR
(same rule as ``run_phase1_oracle_fvd`` / Slide 4), then report mean VBench++
(and SSIM/LPIPS) of the chosen output.

**Budget config oracle:** per video pick best AdaSteer step×LR grid config by PSNR
(same as ``analyze_adasteer_budget_oracle.py``), report oracle SSIM/LPIPS and
VBench++ when per-video VBench exists under grid runs.

**VBench-total oracle (upper bound):** pick method by max ΔVBench total vs NOTTA.

Population FVD/FID for method oracle come from ``phase1_oracle_fvd`` JSON;
budget oracle FVD from ``run_budget_oracle_fvd.py`` (requires saved mp4s).

Example (cluster):
    python3 scripts/analyze_oracle_vbench.py \\
        --vbench-gains-csv sweep_experiment/reports/per_video_analysis/2026-06-28/vbench_agreement/per_video_vbench_gains.csv \\
        --method-fvd-json sweep_experiment/reports/phase1_oracle_fvd/oracle_best_psnr/fvd.json \\
        --budget-series-root sweep_experiment/results/panda_ood_budget_pilot \\
        --output-dir sweep_experiment/reports/per_video_analysis/2026-06-28/oracle_vbench
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.analyze_adasteer_budget_oracle import (  # noqa: E402
    NOTTA_RUN_ID,
    build_video_table,
    compute_oracle_metric_means,
    discover_runs,
    load_merged_summary,
    load_run_all_metrics,
    load_run_psnr,
    oracle_winner,
    parse_run_hparams,
)
from scripts.analyze_per_video_vbench_agreement import VBENCH_DIMS  # noqa: E402
from scripts.per_video_metric_store import (  # noqa: E402
    load_gains_csv,
    load_or_build_wide_table,
    vbench_total,
)

DIM_SHORT = {
    "subject_consistency": "Subj",
    "background_consistency": "BG",
    "aesthetic_quality": "Aes",
    "motion_smoothness": "Motn",
    "dynamic_degree": "Dyn",
    "imaging_quality": "IQ",
    "temporal_flickering": "Flick",
}

METHOD_ORACLE_CANDIDATES = (
    "NOTTA",
    "ADA",
    "LORA_R8_TTA",
    "K5_SIM",
    "K5_RAND",
    "K10_SIM",
    "K10_RAND",
)


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _fmt_delta(x: Optional[float], nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}"


def _mean(arr: List[float]) -> Optional[float]:
    if not arr:
        return None
    return float(np.mean(arr))


def analyze_method_oracle(
    wide_rows: Dict[str, Dict[str, float]],
    video_ids: Sequence[str],
    candidates: Sequence[str],
    *,
    baseline: str = "NOTTA",
) -> dict:
    """Oracle by max absolute PSNR; report VBench of winner vs NOTTA."""
    pick_counts: Dict[str, int] = {c: 0 for c in candidates}
    oracle_psnr: List[float] = []
    oracle_ssim: List[float] = []
    oracle_lpips: List[float] = []
    oracle_vb_tot: List[float] = []
    oracle_vb_dims: Dict[str, List[float]] = {d: [] for d in VBENCH_DIMS}
    notta_vb_tot: List[float] = []
    delta_vb_tot: List[float] = []

    for vid in video_ids:
        row = wide_rows.get(vid, {})
        pick_row: Dict[str, float] = {}
        for m in candidates:
            p = row.get(f"{m}_psnr")
            if p is not None and not math.isnan(p):
                pick_row[m] = float(p)
        w = oracle_winner(pick_row, candidates)
        if w is None:
            continue
        pick_counts[w] = pick_counts.get(w, 0) + 1
        oracle_psnr.append(pick_row[w])
        for spec, lst in (("ssim", oracle_ssim), ("lpips", oracle_lpips)):
            v = row.get(f"{w}_{spec}")
            if v is not None and not math.isnan(v):
                lst.append(float(v))
        vt = vbench_total(row, w)
        if not math.isnan(vt):
            oracle_vb_tot.append(vt)
        for d in VBENCH_DIMS:
            v = row.get(f"{w}_{d}")
            if v is not None and not math.isnan(v):
                oracle_vb_dims[d].append(float(v))
        nvt = vbench_total(row, baseline)
        if not math.isnan(nvt):
            notta_vb_tot.append(nvt)
        if not math.isnan(vt) and not math.isnan(nvt):
            delta_vb_tot.append(vt - nvt)

    always: Dict[str, dict] = {}
    for m in candidates:
        psnrs, vbts = [], []
        for vid in video_ids:
            row = wide_rows.get(vid, {})
            p = row.get(f"{m}_psnr")
            if p is not None and not math.isnan(p):
                psnrs.append(float(p))
            vt = vbench_total(row, m)
            if not math.isnan(vt):
                vbts.append(vt)
        always[m] = {"psnr": _mean(psnrs), "vbench_total": _mean(vbts)}

    return {
        "n": len(oracle_psnr),
        "pick_counts": pick_counts,
        "oracle_psnr": _mean(oracle_psnr),
        "oracle_ssim": _mean(oracle_ssim),
        "oracle_lpips": _mean(oracle_lpips),
        "oracle_vbench_total": _mean(oracle_vb_tot),
        "oracle_vbench_dims": {d: _mean(v) for d, v in oracle_vb_dims.items()},
        "delta_vbench_total_vs_notta": _mean(delta_vb_tot),
        "always": always,
    }


def analyze_vbench_total_oracle(
    wide_rows: Dict[str, Dict[str, float]],
    video_ids: Sequence[str],
    candidates: Sequence[str],
    *,
    baseline: str = "NOTTA",
) -> dict:
    """Upper bound: pick method with highest VBench total (not PSNR)."""
    oracle_vb: List[float] = []
    oracle_psnr: List[float] = []
    for vid in video_ids:
        row = wide_rows.get(vid, {})
        best_m, best_v = None, float("-inf")
        for m in candidates:
            if m == baseline:
                continue
            vt = vbench_total(row, m)
            if not math.isnan(vt) and vt > best_v:
                best_v, best_m = vt, m
        if best_m is None:
            continue
        oracle_vb.append(best_v)
        p = row.get(f"{best_m}_psnr")
        if p is not None and not math.isnan(p):
            oracle_psnr.append(float(p))
    return {
        "oracle_vbench_total": _mean(oracle_vb),
        "oracle_psnr": _mean(oracle_psnr),
        "n": len(oracle_vb),
    }


def load_run_vbench(run_dir: Path) -> Dict[str, Dict[str, float]]:
    return load_per_video_vbench(run_dir)


def analyze_budget_oracle_vbench(
    series_root: Path,
    *,
    baseline_series: Optional[Path] = None,
    grid_runs: Optional[Sequence[str]] = None,
) -> dict:
    runs = discover_runs(series_root)
    if baseline_series and baseline_series.is_dir():
        for name in ("NOTTA", NOTTA_RUN_ID):
            p = baseline_series / name
            if p.is_dir() and name not in runs:
                runs[name] = p
    if grid_runs is None:
        grid_runs = sorted(r for r in runs if r.startswith("S"))
    else:
        grid_runs = [r for r in grid_runs if r in runs]

    run_ids, table = build_video_table(runs)
    grid = [r for r in grid_runs if r in run_ids]
    vids = sorted(table.keys())

    metrics_by_run = {rid: load_run_all_metrics(runs[rid]) for rid in grid if rid in runs}
    oracle_basic = compute_oracle_metric_means(metrics_by_run, grid, vids)

    vb_by_run: Dict[str, Dict[str, Dict[str, float]]] = {}
    for rid in grid:
        if rid in runs:
            vb_by_run[rid] = load_run_vbench(runs[rid])

    oracle_vb_dims: Dict[str, List[float]] = {d: [] for d in VBENCH_DIMS}
    oracle_vb_tot: List[float] = []
    winners: Dict[str, int] = {}
    for vid in vids:
        row = table[vid]
        w = oracle_winner(row, grid)
        if w is None:
            continue
        winners[w] = winners.get(w, 0) + 1
        dm = vb_by_run.get(w, {}).get(vid, {})
        if not dm:
            continue
        vals = [dm.get(d) for d in VBENCH_DIMS]
        if all(v is not None for v in vals):
            oracle_vb_tot.append(float(np.mean(vals)))
            for d in VBENCH_DIMS:
                oracle_vb_dims[d].append(float(dm[d]))

    pop_fvd: Dict[str, Optional[float]] = {}
    for rid in grid:
        merged = load_merged_summary(runs[rid])
        pop_fvd[rid] = merged.get("fvd") if merged else None

    return {
        "series_root": str(series_root),
        "n_videos": len(vids),
        "grid_runs": grid,
        "oracle_psnr_ssim_lpips": oracle_basic,
        "oracle_vbench_total": _mean(oracle_vb_tot),
        "oracle_vbench_dims": {d: _mean(v) for d, v in oracle_vb_dims.items()},
        "n_vbench_videos": len(oracle_vb_tot),
        "winner_counts": winners,
        "population_fvd_by_config": pop_fvd,
        "vbench_available": len(oracle_vb_tot) > 0,
    }


def build_report(
    method_stats: dict,
    vbench_oracle_stats: dict,
    budget_stats: Optional[dict],
    *,
    method_fvd_json: Optional[Path],
    budget_fvd_json: Optional[Path],
    candidates: Sequence[str],
) -> str:
    lines: List[str] = []
    lines.append("# Oracle analysis — VBench++ & companion metrics")
    lines.append("")
    lines.append("## 1. Method oracle (pick max PSNR per video)")
    lines.append("")
    lines.append(f"- **Candidates:** {', '.join(f'`{c}`' for c in candidates)}")
    lines.append(f"- **N:** {method_stats['n']}")
    lines.append("")
    lines.append("| Policy | Mean PSNR | Mean VBench total | Δ total vs NOTTA |")
    lines.append("|---|---:|---:|---:|")
    base = method_stats["always"].get("NOTTA", {})
    lines.append(
        f"| Always NOTTA | {_fmt(base.get('psnr'))} | {_fmt(base.get('vbench_total'))} | — |"
    )
    for m in candidates:
        if m == "NOTTA":
            continue
        a = method_stats["always"].get(m, {})
        dvt = None
        if base.get("vbench_total") and a.get("vbench_total"):
            dvt = a["vbench_total"] - base["vbench_total"]
        lines.append(
            f"| Always `{m}` | {_fmt(a.get('psnr'))} | {_fmt(a.get('vbench_total'))} | "
            f"{_fmt_delta(dvt)} |"
        )
    lines.append(
        f"| **Oracle (best PSNR/video)** | **{_fmt(method_stats.get('oracle_psnr'))}** | "
        f"**{_fmt(method_stats.get('oracle_vbench_total'))}** | "
        f"**{_fmt_delta(method_stats.get('delta_vbench_total_vs_notta'))}** |"
    )
    lines.append("")

    lines.append("### Oracle VBench++ by dimension (absolute scores on routed output)")
    lines.append("")
    lines.append("| Dim | Oracle mean |")
    lines.append("|---|---:|")
    for d in VBENCH_DIMS:
        lines.append(
            f"| {DIM_SHORT[d]} | {_fmt(method_stats['oracle_vbench_dims'].get(d))} |"
        )
    lines.append("")

    lines.append("### Oracle pick frequency (by PSNR)")
    lines.append("")
    lines.append("| Method | Picks | % |")
    lines.append("|---|---:|---:|")
    n = method_stats["n"] or 1
    for m in candidates:
        c = method_stats["pick_counts"].get(m, 0)
        lines.append(f"| `{m}` | {c} | {100.0 * c / n:.1f}% |")
    lines.append("")

    lines.append("## 2. VBench-total oracle (upper bound — pick max VBench total)")
    lines.append("")
    lines.append(
        f"| Policy | Mean VBench total | Mean PSNR | N |"
    )
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| **Oracle (best VBench total/video)** | "
        f"**{_fmt(vbench_oracle_stats.get('oracle_vbench_total'))}** | "
        f"{_fmt(vbench_oracle_stats.get('oracle_psnr'))} | {vbench_oracle_stats.get('n')} |"
    )
    lines.append("")

    if method_fvd_json and method_fvd_json.exists():
        blob = json.loads(method_fvd_json.read_text())
        lines.append("### Method-oracle FVD (population, symlink eval)")
        lines.append("")
        lines.append(
            f"- **FVD:** {_fmt(blob.get('fvd'), 1)}  **FID:** {_fmt(blob.get('fid'), 1)}  "
            f"**(N={blob.get('num_valid_pairs', '—')})**"
        )
        lines.append(
            "- Compare to always-NOTTA FVD from same protocol (~155.9 in job 11061632)."
        )
        lines.append("")
    else:
        lines.append(
            "> Method-oracle FVD: run ``sweep_experiment/scripts/run_phase1_oracle_fvd.py`` "
            "and pass ``--method-fvd-json .../oracle_best_psnr/fvd.json``"
        )
        lines.append("")

    lines.append("## 3. Budget config oracle (AdaSteer step×LR grid)")
    lines.append("")
    if budget_stats is None:
        lines.append("*No ``--budget-series-root`` provided.*")
    else:
        lines.append(f"- **Series:** `{budget_stats['series_root']}`")
        lines.append(f"- **Grid configs:** {', '.join(f'`{r}`' for r in budget_stats['grid_runs'])}")
        lines.append(f"- **Videos:** {budget_stats['n_videos']}")
        ob = budget_stats["oracle_psnr_ssim_lpips"]
        lines.append("")
        lines.append("| Policy | PSNR | SSIM | LPIPS |")
        lines.append("|---|---:|---:|---:|")
        lines.append(
            f"| **Oracle (best grid PSNR/video)** | {_fmt(ob.get('psnr'))} | "
            f"{_fmt(ob.get('ssim'), 4)} | {_fmt(ob.get('lpips'), 4)} |"
        )
        if budget_stats["vbench_available"]:
            lines.append("")
            lines.append(f"### Oracle VBench++ (N={budget_stats['n_vbench_videos']} with per-video VBench)")
            lines.append("")
            lines.append(f"- **VBench total:** {_fmt(budget_stats.get('oracle_vbench_total'))}")
            for d in VBENCH_DIMS:
                lines.append(
                    f"- **{DIM_SHORT[d]}:** {_fmt(budget_stats['oracle_vbench_dims'].get(d))}"
                )
        else:
            lines.append("")
            lines.append(
                "> **Per-video VBench not available** for budget grid "
                "(runs used ``COMPUTE_VBENCH=0``). Population FVD per config below; "
                "oracle FVD requires ``run_budget_oracle_fvd.py`` with saved mp4s."
            )
        lines.append("")
        lines.append("### Population FVD by grid config (merged_summary)")
        lines.append("")
        lines.append("| Config | FVD |")
        lines.append("|---|---:|")
        for rid, fvd in sorted(budget_stats["population_fvd_by_config"].items()):
            lines.append(f"| `{rid}` | {_fmt(fvd, 1)} |")
        lines.append("")

        if budget_fvd_json and budget_fvd_json.exists():
            blob = json.loads(budget_fvd_json.read_text())
            lines.append(
                f"### Budget-oracle FVD (symlink eval): **FVD={_fmt(blob.get('fvd'), 1)}** "
                f"**FID={_fmt(blob.get('fid'), 1)}**"
            )
            lines.append("")
        else:
            lines.append(
                "> **Budget-oracle FVD not computed yet.** Pilot/1000v budget runs used "
                "``NO_SAVE_VIDEOS=1`` → job 11457714 failed. Re-run best configs with "
                "``NO_SAVE_VIDEOS=0`` then:"
            )
            lines.append("> ```bash")
            lines.append("> python sweep_experiment/scripts/run_budget_oracle_fvd.py \\")
            lines.append(f">     --series-root {budget_stats['series_root']} \\")
            lines.append(">     --gt-cache gt_caches/panda_1000_longcat.npz")
            lines.append("> ```")
            lines.append("")

    lines.append("## 4. Reading guide")
    lines.append("")
    lines.append(
        "- **PSNR oracle ≠ VBench oracle:** routing by PSNR does not maximize VBench total; "
        "compare §1 vs §2 for the gap."
    )
    lines.append(
        "- **Budget oracle PSNR ~+0.85 dB (200v pilot) / ~+1.0 dB (Q5)** — FVD ceiling unknown "
        "until ``run_budget_oracle_fvd.py`` succeeds."
    )
    lines.append(
        "- **Method oracle FVD** was **149.57** vs NOTTA **155.94** (−6.37) under 14+14 frame protocol."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vbench-gains-csv", type=Path, required=True)
    ap.add_argument("--ood-csv", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--methods", nargs="*", default=list(METHOD_ORACLE_CANDIDATES))
    ap.add_argument("--budget-series-root", type=Path, default=None)
    ap.add_argument("--baseline-series-root", type=Path, default=None)
    ap.add_argument("--budget-grid-runs", nargs="*", default=None)
    ap.add_argument("--method-fvd-json", type=Path, default=None)
    ap.add_argument("--budget-fvd-json", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    wide = load_or_build_wide_table(
        gains_csv=args.vbench_gains_csv,
        ood_csv=args.ood_csv,
        cache_dir=args.cache_dir,
    )
    candidates = [m for m in args.methods if m in wide.methods]
    if "NOTTA" not in candidates and "NOTTA" in wide.methods:
        candidates = ["NOTTA"] + candidates

    method_stats = analyze_method_oracle(wide.rows, wide.video_ids, candidates)
    vb_oracle = analyze_vbench_total_oracle(wide.rows, wide.video_ids, candidates)

    budget_stats = None
    if args.budget_series_root:
        budget_stats = analyze_budget_oracle_vbench(
            args.budget_series_root,
            baseline_series=args.baseline_series_root,
            grid_runs=args.budget_grid_runs,
        )

    report = build_report(
        method_stats,
        vb_oracle,
        budget_stats,
        method_fvd_json=args.method_fvd_json,
        budget_fvd_json=args.budget_fvd_json,
        candidates=candidates,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_md = args.output_dir / "oracle_vbench_summary.md"
    out_md.write_text(report, encoding="utf-8")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
