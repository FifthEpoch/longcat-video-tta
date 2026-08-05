#!/usr/bin/env python3
"""Can the CHEAP initial-TTA loss predict per-video performance?

Two questions, both answered fully offline (no GPU) from existing summary.json
records of the budget grid + NOTTA:

  Q1 (regression): does a cheap 2-step probe (the S2 config, whose ``final_loss``
      IS the loss after 2 TTA steps) predict the per-video PSNR gain of the best
      fixed config vs no-TTA? -> Spearman/Pearson corr (bootstrap CI) + OOF ridge.

  Q2 (binary gate — the "route TTA vs no-TTA, then apply best fixed config"
      idea): (a) CEILING: is the 2-action oracle headroom mean(relu(fixed_gain))
      non-zero with a bootstrap CI excluding 0?  (b) PREDICTABILITY: can the probe
      predict the binary "TTA helps this video" label better than chance (AUC),
      and does an OOF gate beat always-no-TTA / always-fixed with CI?

The "initial-loss probe" features come from a cheap short config (default the
shortest present, e.g. S2_*):
    final_loss           loss after the probe's few steps
    base_loss            loss before adaptation (final_base_loss)
    loss_reduction       base_loss - final_loss  (how much the probe reduced loss)
    rel_reduction        loss_reduction / base_loss
    delta_norm           ||steering delta|| after the probe
    grad_norm_mean/first mean / first raw grad norm
    initial_loss         early_stopping_info.initial_loss (if ES logged)

Targets use PSNR (cleanest, and where the ~0.35 dB oracle headroom lives):
    fixed_gain = psnr[fixed] - psnr[notta]   (per video)
    binary_helps = 1[fixed_gain > 0]

Usage (cluster):
  cd /scratch/wc3013/longcat-video-tta
  python3 scripts/analyze_initial_loss_prediction.py \
    --series-root sweep_experiment/results/panda_ood_budget_1000v_preview \
    --notta-run NOTTA \
    --out sweep_experiment/reports/per_video_analysis/initial_loss_prediction_1000v.json

By default --probe-run and --fixed-run are auto-selected (shortest config as the
probe; best-mean-PSNR config as fixed). Override with --probe-run / --fixed-run.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

NAN = float("nan")
PROBE_FEATURES = (
    "final_loss",
    "base_loss",
    "loss_reduction",
    "rel_reduction",
    "delta_norm",
    "grad_norm_mean",
    "grad_norm_first",
    "initial_loss",
)


def _f(v) -> float:
    if v is None or v == "":
        return NAN
    try:
        return float(v)
    except (TypeError, ValueError):
        return NAN


def discover_summaries(run_dir: Path) -> List[Path]:
    paths = sorted(Path(p) for p in glob.glob(str(run_dir / "**" / "summary.json"), recursive=True))
    top = run_dir / "summary.json"
    if top.exists() and top not in paths:
        paths.append(top)
    return paths


def load_records(run_dir: Path) -> Dict[str, dict]:
    """video_name -> first successful per-video result across all summary.json."""
    recs: Dict[str, dict] = {}
    for p in discover_summaries(run_dir):
        try:
            s = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        for r in s.get("results", []):
            if not r.get("success", False):
                continue
            vid = r.get("video_name") or r.get("video") or r.get("video_id")
            if not vid:
                continue
            recs.setdefault(str(vid), r)
    return recs


def probe_features(rec: dict) -> Dict[str, float]:
    fl = _f(rec.get("final_loss"))
    bl = _f(rec.get("final_base_loss"))
    dn = _f(rec.get("delta_norm"))
    gn = rec.get("raw_grad_norms") or []
    gn = [_f(x) for x in gn if _f(x) == _f(x)]
    es = rec.get("early_stopping_info") or {}
    init = _f(es.get("initial_loss"))
    red = bl - fl if (bl == bl and fl == fl) else NAN
    rel = (red / bl) if (red == red and bl == bl and abs(bl) > 1e-12) else NAN
    return {
        "final_loss": fl,
        "base_loss": bl,
        "loss_reduction": red,
        "rel_reduction": rel,
        "delta_norm": dn,
        "grad_norm_mean": float(np.mean(gn)) if gn else NAN,
        "grad_norm_first": float(gn[0]) if gn else NAN,
        "initial_loss": init,
    }


def discover_grid_runs(series_root: Path, notta_run: str) -> List[str]:
    runs = []
    for child in sorted(series_root.iterdir()):
        if not child.is_dir():
            continue
        if child.name == notta_run:
            continue
        if discover_summaries(child):
            runs.append(child.name)
    return runs


def _steps_of(run_id: str) -> int:
    m = re.match(r"S(\d+)_", run_id)
    return int(m.group(1)) if m else 10**6


# ---------------------------------------------------------------------------
# stats helpers (numpy-only, dependency-light)
# ---------------------------------------------------------------------------

def rankdata(a: np.ndarray) -> np.ndarray:
    order = a.argsort(kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    a_sorted = a[order]
    i = 0
    while i < len(a_sorted):
        j = i
        while j + 1 < len(a_sorted) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return NAN
    xx, yy = x[m], y[m]
    if xx.std() < 1e-12 or yy.std() < 1e-12:
        return NAN
    return float(np.corrcoef(xx, yy)[0, 1])


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return NAN
    return pearson(rankdata(x[m]), rankdata(y[m]))


def boot_ci_stat(fn, *arrays, n_boot=5000, seed=42) -> Tuple[float, float, float]:
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n = len(arrays[0])
    base = fn(*arrays)
    if n < 3:
        return base, NAN, NAN
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        v = fn(*[a[ix] for a in arrays])
        if v == v:
            vals.append(v)
    if not vals:
        return base, NAN, NAN
    return base, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    m = np.isfinite(scores) & np.isfinite(labels)
    scores, labels = scores[m], labels[m]
    pos = labels == 1
    neg = labels == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return NAN
    r = rankdata(scores)
    auc = (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2.0) / (pos.sum() * neg.sum())
    return float(auc)


def kfold(n: int, k: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def ridge_oof(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, k: int = 5, seed: int = 0) -> np.ndarray:
    """Return OOF predictions. Standardize on train, impute nan->train mean, center y."""
    n, d = X.shape
    pred = np.full(n, NAN)
    for fold in kfold(n, k, seed):
        te = fold
        tr = np.setdiff1d(np.arange(n), te)
        Xtr, Xte = X[tr].copy(), X[te].copy()
        mu = np.nanmean(Xtr, axis=0)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        for j in range(d):
            Xtr[~np.isfinite(Xtr[:, j]), j] = mu[j]
            Xte[~np.isfinite(Xte[:, j]), j] = mu[j]
        sd = Xtr.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        Xtr = (Xtr - mu) / sd
        Xte = (Xte - mu) / sd
        ytr = y[tr]
        ymu = ytr.mean()
        A = Xtr.T @ Xtr + alpha * np.eye(d)
        w = np.linalg.solve(A, Xtr.T @ (ytr - ymu))
        pred[te] = Xte @ w + ymu
    return pred


def build_feature_matrix(feats: List[Dict[str, float]], names=PROBE_FEATURES) -> np.ndarray:
    X = np.full((len(feats), len(names)), NAN)
    for i, fd in enumerate(feats):
        for j, nm in enumerate(names):
            X[i, j] = fd.get(nm, NAN)
    keep = [j for j in range(len(names)) if np.isfinite(X[:, j]).sum() >= 3]
    return X[:, keep], [names[j] for j in keep]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series-root", required=True)
    ap.add_argument("--notta-run", default="NOTTA")
    ap.add_argument("--probe-run", default=None, help="cheap probe config (default: shortest present)")
    ap.add_argument("--fixed-run", default=None, help="fixed config (default: best mean PSNR)")
    ap.add_argument("--metric", default="psnr", choices=["psnr", "ssim"])
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    series_root = Path(args.series_root)
    notta_dir = series_root / args.notta_run
    if not notta_dir.exists():
        print(f"ERROR: notta run not found: {notta_dir}", file=sys.stderr)
        return 2

    grid_runs = discover_grid_runs(series_root, args.notta_run)
    if not grid_runs:
        print(f"ERROR: no grid runs under {series_root}", file=sys.stderr)
        return 2

    higher_better = args.metric != "lpips"
    notta_recs = load_records(notta_dir)
    grid_recs = {rid: load_records(series_root / rid) for rid in grid_runs}

    # mean metric per config over videos it shares with notta -> pick fixed
    common_all = set(notta_recs)
    for rid in grid_runs:
        common_all &= set(grid_recs[rid])
    video_ids = sorted(common_all)
    if len(video_ids) < 10:
        print(f"ERROR: only {len(video_ids)} videos common to ALL configs+notta; "
              f"check run dirs.", file=sys.stderr)
        # fall back to notta ∩ (fixed, probe) later; still try
    notta_metric = np.array([_f(notta_recs[v].get(args.metric)) for v in video_ids])

    def mean_gain(rid: str) -> float:
        g = np.array([_f(grid_recs[rid][v].get(args.metric)) for v in video_ids]) - notta_metric
        g = g[np.isfinite(g)]
        return float(g.mean()) if len(g) else NAN

    fixed_run = args.fixed_run
    if fixed_run is None:
        gains = {rid: mean_gain(rid) for rid in grid_runs}
        fixed_run = max(gains, key=lambda r: gains[r] if gains[r] == gains[r] else -1e9)
    probe_run = args.probe_run
    if probe_run is None:
        probe_run = min(grid_runs, key=_steps_of)

    print(f"videos (common to all {len(grid_runs)} configs + notta): {len(video_ids)}")
    print(f"probe-run (cheap initial-loss features): {probe_run}  ({_steps_of(probe_run)} steps)")
    print(f"fixed-run (best mean {args.metric}):       {fixed_run}  "
          f"(mean gain {mean_gain(fixed_run):+.3f})")

    # per-video arrays on the video_ids order
    fixed_metric = np.array([_f(grid_recs[fixed_run][v].get(args.metric)) for v in video_ids])
    fixed_gain = fixed_metric - notta_metric
    if not higher_better:
        fixed_gain = -fixed_gain
    probe_feats = [probe_features(grid_recs[probe_run][v]) for v in video_ids]
    X, feat_names = build_feature_matrix(probe_feats)

    # config-oracle (max over all 12) gain for reference (max-over-noise ceiling)
    grid_metric = np.full((len(video_ids), len(grid_runs)), NAN)
    for j, rid in enumerate(grid_runs):
        grid_metric[:, j] = [_f(grid_recs[rid][v].get(args.metric)) for v in video_ids]
    oracle12 = np.nanmax(grid_metric, axis=1) if higher_better else np.nanmin(grid_metric, axis=1)
    oracle12_gain = (oracle12 - notta_metric) if higher_better else (notta_metric - oracle12)

    m = np.isfinite(fixed_gain)
    fg = fixed_gain[m]
    Xm = X[m]
    feats_report = {}
    print("\n=== Q1: does the cheap probe predict per-video fixed-config PSNR gain? ===")
    print(f"{'feature':<18}{'spearman [95% CI]':<30}{'pearson [95% CI]'}")
    for j, nm in enumerate(feat_names):
        xj = Xm[:, j]
        sp, sl, sh = boot_ci_stat(spearman, xj, fg, n_boot=args.n_boot)
        pr, pl, ph = boot_ci_stat(pearson, xj, fg, n_boot=args.n_boot)
        sig = "*" if (sl == sl and (sl > 0 or sh < 0)) else " "
        print(f"{nm:<18}{sp:+.3f} [{sl:+.3f},{sh:+.3f}]{sig:<4}{pr:+.3f} [{pl:+.3f},{ph:+.3f}]")
        feats_report[nm] = {"spearman": sp, "spearman_ci": [sl, sh],
                            "pearson": pr, "pearson_ci": [pl, ph]}

    # OOF ridge regression of gain on probe features
    oof = ridge_oof(Xm, fg, alpha=args.alpha)
    r_oof, rl, rh = boot_ci_stat(pearson, oof, fg, n_boot=args.n_boot)
    print(f"\nOOF ridge (probe features -> fixed gain): corr(pred, actual) "
          f"{r_oof:+.3f} [{rl:+.3f},{rh:+.3f}]  (N={len(fg)})")

    # ---- Q2: binary gate ceiling + predictability ----
    print("\n=== Q2: binary TTA/no-TTA gate (apply best fixed config when TTA) ===")
    relu_pos = np.clip(fg, 0, None)      # gain captured if we apply fixed only when it helps
    relu_neg = np.clip(-fg, 0, None)     # loss avoided if we skip fixed when it hurts
    # ceiling vs always-no-TTA: mean(relu(fixed_gain)); vs always-fixed: mean(relu(-fixed_gain))
    c1, c1l, c1h = boot_ci_stat(lambda a: a.mean(), relu_pos, n_boot=args.n_boot)
    c2, c2l, c2h = boot_ci_stat(lambda a: a.mean(), relu_neg, n_boot=args.n_boot)
    fg_mean, fgl, fgh = boot_ci_stat(lambda a: a.mean(), fg, n_boot=args.n_boot)
    frac_help = float((fg > 0).mean())
    print(f"fraction of videos where fixed > no-TTA : {frac_help:.1%}")
    print(f"always-fixed gain vs no-TTA             : {fg_mean:+.4f} [{fgl:+.4f},{fgh:+.4f}] dB")
    print(f"PERFECT-gate headroom vs always-no-TTA  : {c1:+.4f} [{c1l:+.4f},{c1h:+.4f}] dB "
          f"({'REAL' if c1l > 0 else 'null'})")
    print(f"PERFECT-gate headroom vs always-fixed   : {c2:+.4f} [{c2l:+.4f},{c2h:+.4f}] dB "
          f"({'REAL' if c2l > 0 else 'null'})")
    o12, o12l, o12h = boot_ci_stat(lambda a: a.mean(), oracle12_gain[np.isfinite(oracle12_gain)],
                                   n_boot=args.n_boot)
    print(f"[ref] 12-config oracle (max-over-noise) : {o12:+.4f} [{o12l:+.4f},{o12h:+.4f}] dB")

    labels = (fg > 0).astype(float)
    # predictability: AUC of each probe feature + OOF ridge-probe
    print("\nbinary-label predictability (AUC vs 0.5 = chance):")
    for j, nm in enumerate(feat_names):
        a = auc_score(Xm[:, j], labels)
        # AUC symmetric; report max(a, 1-a) direction-agnostic with the raw too
        print(f"  {nm:<18} AUC {a:.3f}  (dir-agnostic {max(a, 1-a):.3f})")
    gate_oof = ridge_oof(Xm, labels, alpha=args.alpha)
    auc_oof = auc_score(gate_oof, labels)
    print(f"  {'OOF ridge-probe':<18} AUC {auc_oof:.3f}")

    # realized OOF gate policy: apply fixed when predicted-helps (gate_oof>0.5), else no-TTA
    apply = gate_oof > 0.5
    policy_gain_vs_notta = np.where(apply, fg, 0.0)  # 0 gain if we skip (stay at no-TTA)
    pg, pgl, pgh = boot_ci_stat(lambda a: a.mean(), policy_gain_vs_notta, n_boot=args.n_boot)
    print(f"\nOOF gate policy gain vs always-no-TTA   : {pg:+.4f} [{pgl:+.4f},{pgh:+.4f}] dB "
          f"({'beats no-TTA' if pgl > 0 else 'null'})")
    # vs always-fixed: policy - fixed_gain (skipping when we predict hurt)
    policy_vs_fixed = policy_gain_vs_notta - fg
    pvf, pvfl, pvfh = boot_ci_stat(lambda a: a.mean(), policy_vs_fixed, n_boot=args.n_boot)
    print(f"OOF gate policy gain vs always-fixed    : {pvf:+.4f} [{pvfl:+.4f},{pvfh:+.4f}] dB "
          f"({'beats fixed' if pvfl > 0 else 'null'})")

    verdict = []
    if any(feats_report[n]["spearman_ci"][0] > 0 or feats_report[n]["spearman_ci"][1] < 0
           for n in feats_report):
        verdict.append("Q1: at least one probe feature has a CI-significant monotonic link to gain.")
    else:
        verdict.append("Q1: NO probe feature shows a CI-significant link to per-video gain.")
    verdict.append(
        f"Q2 ceiling: perfect gate {'CAN' if c1l > 0 else 'CANNOT'} beat no-TTA and "
        f"{'CAN' if c2l > 0 else 'CANNOT'} beat always-fixed (CI vs 0)."
    )
    verdict.append(
        f"Q2 deployable: OOF gate {'beats' if pgl > 0 else 'does NOT beat'} no-TTA; "
        f"{'beats' if pvfl > 0 else 'does NOT beat'} always-fixed."
    )
    print("\n=== VERDICT ===")
    for v in verdict:
        print("  - " + v)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "series_root": str(series_root),
            "metric": args.metric,
            "n_videos": len(video_ids),
            "probe_run": probe_run,
            "probe_steps": _steps_of(probe_run),
            "fixed_run": fixed_run,
            "feat_names": feat_names,
            "q1_feature_corr": feats_report,
            "q1_oof_ridge_corr": {"corr": r_oof, "ci": [rl, rh], "n": int(len(fg))},
            "q2_frac_help": frac_help,
            "q2_always_fixed_gain": {"mean": fg_mean, "ci": [fgl, fgh]},
            "q2_perfect_gate_vs_notta": {"mean": c1, "ci": [c1l, c1h]},
            "q2_perfect_gate_vs_fixed": {"mean": c2, "ci": [c2l, c2h]},
            "q2_oracle12_gain": {"mean": o12, "ci": [o12l, o12h]},
            "q2_gate_auc_oof": auc_oof,
            "q2_oof_gate_vs_notta": {"mean": pg, "ci": [pgl, pgh]},
            "q2_oof_gate_vs_fixed": {"mean": pvf, "ci": [pvfl, pvfh]},
            "verdict": verdict,
        }
        out.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
