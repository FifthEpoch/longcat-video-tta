#!/usr/bin/env python3
"""Aggregate routing experiment JSONs + bootstrap CIs from OOF/per-video policies."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.budget_routing_common import bootstrap_captured, load_pilot_bundle, labeled_mask  # noqa: E402
from scripts.train_vbench_headroom_router import eval_config_pick_policy  # noqa: E402


def load_oof_csv(path: Path) -> Dict[str, dict]:
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[r["video_id"]] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-dir",
        type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-05/budget_routing_experiments",
    )
    ap.add_argument("--series-root", type=Path, default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot")
    ap.add_argument(
        "--feature-date",
        type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-06",
    )
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    bundle = load_pilot_bundle(args.series_root, args.feature_date)
    Y = bundle["Y_total"]
    fixed = bundle["fixed_vb"]
    grid = bundle["grid_runs"]
    vids = bundle["video_ids"]
    mask = labeled_mask(fixed, Y)
    vid_l = [vids[i] for i in range(len(vids)) if mask[i]]
    Yl = Y[mask]
    fl = fixed[mask]

    rows: List[dict] = []
    for jp in sorted(args.input_dir.glob("*.json")):
        if jp.name.startswith("_"):
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        if data.get("policy", data).get("skipped"):
            continue
        name = jp.stem
        policy = data.get("policy") or data

        # Try OOF csv from baseline tmp
        oof_path = args.input_dir / "_tmp_baseline/budget_config_oof_predictions.csv"
        if name == "baseline_linear_total" and oof_path.is_file():
            oof = load_oof_csv(oof_path)
            pv, ov, fv = [], [], []
            for v in vid_l:
                if v not in oof:
                    continue
                r = oof[v]
                pv.append(float(r["policy_vbench"]))
                ov.append(float(r["oracle_vbench"]))
                fv.append(float(r["fixed_vbench"]))
            cap, dlo, dhi, clo, chi = bootstrap_captured(
                np.array(pv), np.array(ov), np.array(fv), n_boot=args.n_boot, seed=args.seed,
            )
            rows.append({
                "experiment": name,
                "captured_pct": 100 * cap,
                "delta_ci_lo": dlo,
                "delta_ci_hi": dhi,
                "captured_ci_lo": 100 * clo,
                "captured_ci_hi": 100 * chi,
            })
            continue

        # Point estimate only from JSON
        cap = policy.get("fraction_oracle_captured")
        rows.append({
            "experiment": name,
            "captured_pct": None if cap is None else 100 * cap,
            "delta_ci_lo": None,
            "delta_ci_hi": None,
            "captured_ci_lo": None,
            "captured_ci_hi": None,
        })

    out_csv = args.input_dir / "routing_experiments_bootstrap.csv"
    if rows:
        fields = sorted({k for r in rows for k in r.keys()})
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    md = args.input_dir / "routing_experiments_bootstrap.md"
    lines = ["# Routing experiments — bootstrap CIs", ""]
    for r in rows:
        if r.get("captured_ci_lo") is not None:
            lines.append(
                f"- **{r['experiment']}**: captured **{r['captured_pct']:.1f}%** "
                f"[{r['captured_ci_lo']:.1f}%, {r['captured_ci_hi']:.1f}%], "
                f"Δ [{r['delta_ci_lo']:+.4f}, {r['delta_ci_hi']:+.4f}]"
            )
        else:
            cp = r.get("captured_pct")
            cp_s = f"{cp:.1f}" if cp is not None else "—"
            lines.append(f"- **{r['experiment']}**: captured **{cp_s}%** (point only)")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Merge per-task JSON rows into summary if array jobs were used
    summary_rows: List[dict] = []
    for jp in sorted(args.input_dir.glob("*.json")):
        if jp.name.startswith("_"):
            continue
        data = json.loads(jp.read_text(encoding="utf-8"))
        if "row" in data:
            summary_rows.append(data["row"])
        elif data.get("skipped"):
            summary_rows.append(data)
    if summary_rows:
        from scripts.run_budget_routing_experiments import write_summary

        write_summary(summary_rows, args.input_dir)

    print(f"Wrote {out_csv} and {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
