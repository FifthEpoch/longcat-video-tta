#!/usr/bin/env python3
"""Cross-metric table: FVD + PSNR + 7 VBench dims + overall, per policy.

Same row items as the router-FVD table (always_notta, fixed, the 4 routers,
oracle), one column per metric, all on the SAME matched N pool. Per-video
realized PSNR/VBench come from each policy's per-video config choice:

  * always_notta        -> NOTTA's own clip
  * fixed_S10_LR5e3     -> the fixed config
  * router_*            -> the config chosen by the OOF router (router_manifest.json;
                          may be NO-TTA for 13-action)
  * oracle_best_psnr    -> per-video best-PSNR config (oracle manifest winner_run);
                          its VBench = that PSNR-winning config's VBench (NOT
                          VBench-optimized)

FVD is read from each policy's fvd.json. VBench dims are the generated-only
scores when VBENCH_SUBDIR=vbench_results_geneval is exported.

Usage:
    VBENCH_SUBDIR=vbench_results_geneval python3 scripts/build_cross_metric_policy_table.py \
      --series-root sweep_experiment/results/panda_ood_budget_1000v_preview \
      --fvd-root sweep_experiment/reports/budget_oracle_fvd_1000v_preview \
      --output sweep_experiment/reports/per_video_analysis/2026-07-28/cross_metric_policy_table.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.analyze_adasteer_budget_oracle import NOTTA_RUN_ID  # noqa: E402
from scripts.analyze_adasteer_budget_vbench_oracle import (  # noqa: E402
    discover_runs,
    vbench_total_score,
)
from scripts.analyze_per_video_vbench_agreement import (  # noqa: E402
    VBENCH_DIMS,
    load_per_video_vbench,
)
from scripts.analyze_per_video_tta_gain import load_per_video_metrics  # noqa: E402
from scripts.budget_routing_common import load_pilot_bundle  # noqa: E402

FIXED_RUN = "S10_LR5e3"
DIMS = list(VBENCH_DIMS)
# dims naturally in [0,1] except imaging_quality (0-100, MUSIQ)
_NORM = {d: (0.01 if d == "imaging_quality" else 1.0) for d in DIMS}


def _read_fvd(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("fvd")
    except Exception:  # noqa: BLE001
        return None


def _router_picks(manifest: Path) -> Dict[str, str]:
    blob = json.loads(manifest.read_text(encoding="utf-8"))
    return {e["video_id"]: e["chosen_run"] for e in blob.get("picks", [])}


def _oracle_picks(manifest: Path) -> Dict[str, str]:
    blob = json.loads(manifest.read_text(encoding="utf-8"))
    out: Dict[str, str] = {}
    for e in blob.get("entries", []):
        vid = e.get("video_id")
        run = e.get("winner_run")
        if vid and run:
            out[vid] = run
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Cross-metric per-policy table")
    ap.add_argument("--series-root", type=Path, required=True)
    ap.add_argument(
        "--feature-date", type=Path,
        default=_REPO / "sweep_experiment/reports/per_video_analysis/2026-07-12",
    )
    ap.add_argument("--fvd-root", type=Path, required=True,
                    help="budget_oracle_fvd_1000v_preview root")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    bundle = load_pilot_bundle(args.series_root, args.feature_date,
                               require_vbench=True)
    vids: List[str] = bundle["video_ids"]
    grid_runs: List[str] = bundle["grid_runs"]
    run_idx: Dict[str, int] = bundle["run_idx"]
    psnr = np.array(bundle["psnr"], dtype=float)          # n x k
    Y_dim = {d: np.array(bundle["Y_dim"][d], dtype=float) for d in DIMS}  # n x k
    vid_i = {v: i for i, v in enumerate(vids)}

    runs = discover_runs(args.series_root)
    notta_dir = runs.get(NOTTA_RUN_ID)
    notta_psnr_map = load_per_video_metrics(notta_dir) if notta_dir else {}
    notta_vb_map = load_per_video_vbench(notta_dir) if notta_dir else {}

    fvd_root = args.fvd_root
    routers_root = fvd_root / "routers"
    matched = fvd_root / "matched"

    # policy -> (per-video chosen run dict, fvd)
    policies: List[dict] = []

    def add_router(pol: str) -> None:
        man = routers_root / pol / "router_manifest.json"
        if not man.is_file():
            print(f"[warn] missing {man}", file=sys.stderr)
            return
        policies.append({
            "name": pol,
            "picks": _router_picks(man),
            "fvd": _read_fvd(routers_root / pol / "fvd.json"),
        })

    # always_notta
    policies.append({
        "name": "always_notta",
        "picks": {v: NOTTA_RUN_ID for v in vids},
        "fvd": _read_fvd(matched / "always_notta" / "fvd.json"),
    })
    # fixed
    policies.append({
        "name": f"fixed_{FIXED_RUN}",
        "picks": {v: FIXED_RUN for v in vids},
        "fvd": _read_fvd(matched / f"fixed_{FIXED_RUN}" / "fvd.json"),
    })
    # routers
    for pol in ("router_psnr_ABC_13act", "router_psnr_ABC_12act",
                "router_vbench_ABC_13act", "router_vbench_ABC_12act"):
        add_router(pol)
    # oracle (best-PSNR config per video); FVD from common dir if present
    oracle_man = fvd_root / "oracle_best_psnr" / "manifest.json"
    oracle_fvd = _read_fvd(matched / "oracle_best_psnr_common" / "fvd.json")
    if oracle_fvd is None:
        oracle_fvd = _read_fvd(fvd_root / "oracle_best_psnr" / "fvd.json")
    if oracle_man.is_file():
        policies.append({
            "name": "oracle_best_psnr",
            "picks": _oracle_picks(oracle_man),
            "fvd": oracle_fvd,
        })

    # canonical set = intersection of every policy's video ids that ALSO have a
    # NOTTA metric anchor (so all columns are on the same matched pool)
    canon = set(vids)
    for p in policies:
        canon &= set(p["picks"].keys())
    canon &= set(notta_psnr_map.keys())
    canon = sorted(canon)
    n = len(canon)
    print(f"[info] canonical matched set N={n} "
          f"(policies={[p['name'] for p in policies]})", file=sys.stderr)

    def realized(pol_picks: Dict[str, str]):
        ps: List[float] = []
        dim_vals: Dict[str, List[float]] = {d: [] for d in DIMS}
        for v in canon:
            run = pol_picks.get(v)
            if run is None:
                continue
            if run == NOTTA_RUN_ID:
                pv = notta_psnr_map.get(v, {}).get("psnr")
                if pv is not None:
                    ps.append(float(pv))
                vb = notta_vb_map.get(v, {})
                for d in DIMS:
                    val = vb.get(d)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        dim_vals[d].append(float(val))
            else:
                j = run_idx.get(run)
                if j is None:
                    continue
                i = vid_i[v]
                if not np.isnan(psnr[i, j]):
                    ps.append(float(psnr[i, j]))
                for d in DIMS:
                    val = Y_dim[d][i, j]
                    if not np.isnan(val):
                        dim_vals[d].append(float(val))
        mean_ps = float(np.mean(ps)) if ps else float("nan")
        mean_dims = {d: (float(np.mean(vs)) if vs else float("nan"))
                     for d, vs in dim_vals.items()}
        norm_overall = np.mean([mean_dims[d] * _NORM[d] for d in DIMS])
        raw_overall = np.mean([mean_dims[d] for d in DIMS])
        return mean_ps, mean_dims, float(norm_overall), float(raw_overall)

    rows = []
    for p in policies:
        mean_ps, mean_dims, norm_overall, raw_overall = realized(p["picks"])
        rows.append({
            "name": p["name"], "fvd": p["fvd"], "psnr": mean_ps,
            "dims": mean_dims, "vb_norm": norm_overall, "vb_raw": raw_overall,
        })

    # ---- render ----
    short = {
        "subject_consistency": "Subj", "background_consistency": "Bg",
        "aesthetic_quality": "Aes", "motion_smoothness": "Motion",
        "dynamic_degree": "Dyn", "imaging_quality": "Imaging",
        "temporal_flickering": "Temp",
    }
    hdr = (["Policy", "FVD↓", "PSNR↑"] + [f"{short[d]}↑" for d in DIMS]
           + ["VB-mean(norm)↑"])
    lines = [
        "# Cross-metric per-policy comparison (matched N=" + str(n) + ")",
        "",
        "Same row items as the router-FVD table. Per-video realized PSNR/VBench "
        "from each policy's per-video config choice; FVD from each policy's "
        "`fvd.json`. VBench dims are generated-only "
        "(`VBENCH_SUBDIR=vbench_results_geneval`).",
        "",
        "> Dims in native units: Subj/Bg/Aes/Motion/Dyn/Temp in [0,1], "
        "**Imaging in [0,100]** (MUSIQ). `VB-mean(norm)` = mean of the 7 dims "
        "with Imaging scaled to [0,1]. ↓ lower better, ↑ higher better.",
        "",
        "| " + " | ".join(hdr) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(hdr) - 1)) + "|",
    ]

    def fmt(x, nd=3):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "—"
        return f"{x:.{nd}f}"

    for r in rows:
        cells = [
            f"`{r['name']}`",
            fmt(r["fvd"], 2),
            fmt(r["psnr"], 3),
        ]
        for d in DIMS:
            nd = 2 if d == "imaging_quality" else 4
            cells.append(fmt(r["dims"][d], nd))
        cells.append(fmt(r["vb_norm"], 4))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    (args.output.with_suffix(".json")).write_text(
        json.dumps({"n": n, "rows": rows}, indent=2), encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"\nWrote {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
