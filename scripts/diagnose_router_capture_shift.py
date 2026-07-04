#!/usr/bin/env python3
"""Decompose budget_config router 'captured' shift (e.g. 44% @ N=143 → 9% @ N=200).

Reads two router OOF CSVs (old vs new) and/or recomputes oracle headroom on video
subsets. Run on cluster where pilot VBench JSONs live.

Example:
    python3 scripts/diagnose_router_capture_shift.py \\
        --old-oof sweep_experiment/reports/per_video_analysis/2026-07-03/vbench_headroom_router/budget_config_oof_predictions.csv \\
        --new-oof sweep_experiment/reports/per_video_analysis/2026-07-04/vbench_headroom_router/budget_config_oof_predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.train_vbench_headroom_router import (  # noqa: E402
    eval_config_pick_policy,
    load_budget_score_matrix,
)


def load_oof(path: Path) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows[r["video_id"]] = r
    return rows


def ffloat(x: str) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def subset_stats(rows: Sequence[dict]) -> dict:
    pv = np.array([ffloat(r["policy_vbench"]) for r in rows])
    ov = np.array([ffloat(r["oracle_vbench"]) for r in rows])
    fv = np.array([ffloat(r["fixed_vbench"]) for r in rows])
    ok = np.isfinite(pv) & np.isfinite(ov) & np.isfinite(fv)
    pv, ov, fv = pv[ok], ov[ok], fv[ok]
    if len(pv) == 0:
        return {"n": 0}
    d = pv - fv
    h = ov - fv
    cap = d.mean() / h.mean() if abs(h.mean()) > 1e-9 else float("nan")
    return {
        "n": int(len(pv)),
        "fixed": float(fv.mean()),
        "policy": float(pv.mean()),
        "oracle": float(ov.mean()),
        "headroom": float(h.mean()),
        "policy_gain": float(d.mean()),
        "captured": float(cap),
    }


def fmt(st: dict) -> str:
    if not st.get("n"):
        return "n=0"
    cap = st["captured"]
    cap_s = f"{100 * cap:.1f}%" if cap == cap else "—"
    return (
        f"n={st['n']}  fixed={st['fixed']:.4f}  policy={st['policy']:.4f}  "
        f"oracle={st['oracle']:.4f}  headroom={st['headroom']:.4f}  "
        f"policy_gain={st['policy_gain']:+.4f}  captured={cap_s}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--old-oof", type=Path, required=True)
    ap.add_argument("--new-oof", type=Path, required=True)
    ap.add_argument(
        "--series-root",
        type=Path,
        default=_REPO / "sweep_experiment/results/panda_ood_budget_pilot",
    )
    args = ap.parse_args()

    if not args.old_oof.is_file():
        print(f"[error] missing {args.old_oof}", file=sys.stderr)
        return 2
    if not args.new_oof.is_file():
        print(f"[error] missing {args.new_oof}", file=sys.stderr)
        return 2

    old = load_oof(args.old_oof)
    new = load_oof(args.new_oof)
    old_ids: Set[str] = set(old)
    new_ids: Set[str] = set(new)
    added = sorted(new_ids - old_ids)
    removed = sorted(old_ids - new_ids)
    overlap = sorted(old_ids & new_ids)

    print("=== OOF video sets ===")
    print(f"old OOF: {len(old_ids)}  new OOF: {len(new_ids)}  overlap: {len(overlap)}")
    print(f"added in new (were excluded from old router): {len(added)}")
    print(f"dropped in new: {len(removed)}")

    old_rows = list(old.values())
    new_rows = list(new.values())
    st_old = subset_stats(old_rows)
    st_new = subset_stats(new_rows)
    print("\n=== Router OOF (point estimates from saved CSVs) ===")
    print(f"OLD {args.old_oof.parent.name}: {fmt(st_old)}")
    print(f"NEW {args.new_oof.parent.name}: {fmt(st_new)}")

    # Evaluate NEW router picks on OLD-only vs ADDED-only subsets (same OOF file)
    added_rows = [new[v] for v in added if v in new]
    overlap_rows = [new[v] for v in overlap if v in new]
    print("\n=== NEW OOF split by old coverage ===")
    print(f"overlap (old 143): {fmt(subset_stats(overlap_rows))}")
    print(f"added (old missing): {fmt(subset_stats(added_rows))}")

    # Oracle-only on full 200 from fresh Y matrix (no router)
    vids = sorted(new_ids)
    Y, fixed, _, grid, n_labeled = load_budget_score_matrix(args.series_root, vids)
    oracle_idx = np.nanargmax(Y, axis=1)
    pol_oracle = eval_config_pick_policy(
        oracle_idx, Y, fixed, grid,
    )
    print("\n=== Perfect oracle on current labels (upper bound, not router) ===")
    print(
        f"n_labeled={n_labeled}  headroom={pol_oracle['oracle_headroom']:.4f}  "
        f"fixed={pol_oracle['mean_fixed_vbench']:.4f}  "
        f"oracle={pol_oracle['mean_oracle_vbench']:.4f}"
    )

    # Oracle headroom on overlap vs added using Y matrix + new OOF oracle columns
    idx = {v: i for i, v in enumerate(vids)}
    def headroom_for(ids: Sequence[str]) -> Tuple[float, float, int]:
        hs: List[float] = []
        for vid in ids:
            i = idx.get(vid)
            if i is None:
                continue
            row = Y[i]
            fi = fixed[i]
            if np.isnan(fi) or np.all(np.isnan(row)):
                continue
            ov = float(np.nanmax(row))
            hs.append(ov - fi)
        if not hs:
            return float("nan"), float("nan"), 0
        a = np.asarray(hs)
        return float(a.mean()), float(np.median(a)), len(a)

    m_o, med_o, n_o = headroom_for(overlap)
    m_a, med_a, n_a = headroom_for(added)
    print("\n=== Per-video oracle headroom (oracle−fixed) from current Y ===")
    print(f"overlap (old train set): mean={m_o:+.4f}  median={med_o:+.4f}  n={n_o}")
    print(f"added (was unlabeled):   mean={m_a:+.4f}  median={med_a:+.4f}  n={n_a}")

    print("\n=== Interpretation ===")
    print(
        "- 'captured' = mean(policy−fixed) / mean(oracle−fixed). "
        "Both numerator and denominator must use the same video set and labels."
    )
    print(
        "- ~45% used N≈143 with inflated oracle headroom (~1.01 on contaminated total); "
        "~9% is N=200 with headroom ~0.14 — not the same experiment."
    )
    print(
        "- ~8% quintile-adaptive is a separate hand-crafted policy (+0.011 vs fixed), "
        "not the learned router."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
