#!/usr/bin/env python3
"""Best-of-k seed headroom + routability probe (offline, CPU).

Reads a best-of-k seed run (bestofk_experiment/scripts/run_bestofk_seeds.py) and
answers, with the SAME CI / null machinery as router_significance_analysis:

  (A) Is there REAL headroom from seed selection? oracle best-of-k PSNR minus
      the deployed single-seed reference (candidate 0), paired bootstrap 95% CI
      + sign-flip p, plus vs the expected random single seed (mean over seeds).
      Unlike the AdaSteer config grid, candidates here are genuinely different
      videos, so positive, tight headroom = a real lever.

  (B) Is that headroom ROUTABLE with a deploy-legitimate (GT-free) selector?
      - descriptive: mean within-video Spearman corr between each cheap signal
        (seam continuity, motion, sharpness) and PSNR across the k seeds.
      - a leakage-free OOF ridge selector (fold by VIDEO) predicts per-candidate
        PSNR from the GT-free signals, argmax per video -> realized PSNR.
        Reports Δ vs reference with bootstrap CI + sign-flip p, a shuffle-null
        (permute the selector's chosen seed across videos) and a random-seed
        null, and match% vs the 1/k chance floor.

Verdict distinguishes: real+routable (headroom>0 AND selector beats shuffle),
real-but-unroutable (headroom>0 but selector ~ shuffle -> need a better signal,
e.g. model likelihood), or no headroom.

Usage:
    python3 scripts/analyze_bestofk_headroom.py \
      --summary bestofk_experiment/results/panda_1000v_k8/summary.json \
      --output-dir sweep_experiment/reports/per_video_analysis/2026-07-31/bestofk
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.train_vbench_headroom_router import (  # noqa: E402
    kfold_indices,
    ridge_fit,
    ridge_predict,
    standardize_train_test,
)

_SIGNALS = ["sig_seam_l2", "sig_temporal_l2", "sig_sharpness"]
_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


# --- self-contained stat helpers (same definitions as router_significance) ---
def paired_bootstrap_ci(d: np.ndarray, *, n_boot: int, seed: int, ci: float = 95.0
                        ) -> Tuple[float, float, float]:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = d.size
    means = np.array([d[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return (float(d.mean()),
            float(np.percentile(means, (100 - ci) / 2)),
            float(np.percentile(means, 100 - (100 - ci) / 2)))


def sign_flip_p(d: np.ndarray, *, n_perm: int, seed: int) -> float:
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    obs = abs(float(d.mean()))
    rng = np.random.default_rng(seed)
    n = d.size
    cnt = sum(abs(float((d * (rng.integers(0, 2, n) * 2 - 1)).mean())) >= obs
              for _ in range(n_perm))
    return (cnt + 1) / (n_perm + 1)


def _one_sided_p(null: np.ndarray, observed: float) -> float:
    null = null[np.isfinite(null)]
    if null.size == 0 or not math.isfinite(observed):
        return float("nan")
    return float((np.sum(null >= observed) + 1) / (null.size + 1))


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    if np.any(counts > 1):
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        ranks = avg[inv]
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3 or np.all(a[m] == a[m][0]) or np.all(b[m] == b[m][0]):
        return float("nan")
    ra, rb = _rankdata(a[m]), _rankdata(b[m])
    ra -= ra.mean(); rb -= rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def _gather_results(summary_path: Path) -> List[dict]:
    """Accept a single summary.json or a directory of chunk_*/summary.json."""
    if summary_path.is_dir():
        files = sorted(summary_path.glob("**/summary.json"))
        if not files:
            raise SystemExit(f"[error] no summary.json under {summary_path}")
        out: List[dict] = []
        seen: set = set()
        for fp in files:
            for r in json.loads(fp.read_text()).get("results", []):
                key = r.get("video_name")
                if r.get("success") and r.get("candidates") and key not in seen:
                    seen.add(key)
                    out.append(r)
        return out
    return [r for r in json.loads(summary_path.read_text()).get("results", [])
            if r.get("success") and r.get("candidates")]


def _load(summary_path: Path) -> Tuple[List[str], np.ndarray, Dict[str, np.ndarray], int]:
    results = _gather_results(summary_path)
    if not results:
        raise SystemExit(f"[error] no successful candidates in {summary_path}")
    k = max(len(r["candidates"]) for r in results)
    vids: List[str] = []
    psnr_rows: List[np.ndarray] = []
    sig_rows: Dict[str, List[np.ndarray]] = {s: [] for s in _SIGNALS}
    for r in results:
        cands = sorted(r["candidates"], key=lambda c: c.get("seed_index", 0))
        if len(cands) < k:
            continue
        pr = np.array([c.get("psnr") if c.get("psnr") is not None else np.nan for c in cands], float)
        if np.all(np.isnan(pr)):
            continue
        vids.append(r["video_name"])
        psnr_rows.append(pr[:k])
        for s in _SIGNALS:
            sig_rows[s].append(
                np.array([c.get(s, np.nan) if c.get(s) is not None else np.nan for c in cands][:k], float)
            )
    P = np.vstack(psnr_rows)
    S = {s: np.vstack(sig_rows[s]) for s in _SIGNALS}
    return vids, P, S, k


def _oof_selector(P: np.ndarray, feats: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold-by-video OOF ridge predicting per-candidate PSNR; return picked seed idx.

    P: [n,k] psnr. feats: [n,k,d] GT-free signals. Picks argmax predicted per video.
    """
    n, k = P.shape
    d = feats.shape[2]
    folds = kfold_indices(n, n_folds, seed)
    pick = np.zeros(n, dtype=int)
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        Xtr = feats[train_idx].reshape(-1, d)
        ytr = P[train_idx].reshape(-1)
        m = np.isfinite(ytr) & np.all(np.isfinite(Xtr), axis=1)
        Xte = feats[test_idx].reshape(-1, d)
        if m.sum() < 10:
            for r, ti in enumerate(test_idx):
                pick[ti] = 0
            continue
        Xtr_s, Xte_s, _, _ = standardize_train_test(Xtr[m], Xte)
        w = ridge_fit(Xtr_s, ytr[m], 1.0)
        pred = ridge_predict(Xte_s, w).reshape(len(test_idx), k)
        for r, ti in enumerate(test_idx):
            row = pred[r]
            pick[ti] = int(np.nanargmax(row)) if np.any(np.isfinite(row)) else 0
    return pick


def _realized(P: np.ndarray, pick: np.ndarray) -> np.ndarray:
    return np.array([P[i, pick[i]] if np.isfinite(P[i, pick[i]]) else np.nan
                     for i in range(len(pick))])


def _shuffle_null(P: np.ndarray, pick: np.ndarray, ref: np.ndarray,
                  n_draw: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(pick)
    out = np.empty(n_draw)
    for b in range(n_draw):
        perm = rng.permutation(pick)
        real = _realized(P, perm)
        out[b] = float(np.nanmean(real - ref))
    return out


def _random_null(P: np.ndarray, ref: np.ndarray, n_draw: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, k = P.shape
    out = np.empty(n_draw)
    for b in range(n_draw):
        pick = rng.integers(0, k, n)
        real = _realized(P, pick)
        out[b] = float(np.nanmean(real - ref))
    return out


def _f(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _fd(x, nd=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}"


def _fp(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser(description="Best-of-k headroom + routability probe")
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--n-perm", type=int, default=10000)
    args = ap.parse_args()

    vids, P, S, k = _load(args.summary)
    n = len(vids)
    if n < 30:
        raise SystemExit(f"[error] only {n} usable videos in {args.summary}")

    ref = P[:, 0]                       # deployed single-seed reference (candidate 0)
    mean_seed = np.nanmean(P, axis=1)   # expected random single seed
    oracle = np.nanmax(P, axis=1)

    # ---- (A) oracle headroom ----
    h_ref_m, h_ref_lo, h_ref_hi = paired_bootstrap_ci(oracle - ref, n_boot=args.n_boot, seed=args.seed)
    h_ref_p = sign_flip_p(oracle - ref, n_perm=args.n_perm, seed=args.seed + 1)
    h_mean_m, h_mean_lo, h_mean_hi = paired_bootstrap_ci(oracle - mean_seed, n_boot=args.n_boot, seed=args.seed + 2)
    frac_better = float(np.mean(oracle > ref + 1e-9))

    # ---- (B1) descriptive: within-video Spearman(signal, psnr) ----
    sig_corr: Dict[str, float] = {}
    for s in _SIGNALS:
        cs = [_spearman(S[s][i], P[i]) for i in range(n)]
        cs = [c for c in cs if math.isfinite(c)]
        sig_corr[s] = float(np.mean(cs)) if cs else float("nan")

    # ---- (B2) leakage-free OOF ridge selector on GT-free signals ----
    feats = np.stack([S[s] for s in _SIGNALS], axis=2)  # [n,k,d]
    # impute NaNs with per-signal global mean
    for j, s in enumerate(_SIGNALS):
        col = feats[:, :, j]
        mu = np.nanmean(col)
        col[~np.isfinite(col)] = mu
        feats[:, :, j] = col

    pick = _oof_selector(P, feats, args.n_folds, args.seed)
    realized = _realized(P, pick)
    d_ref = realized - ref
    sel_m, sel_lo, sel_hi = paired_bootstrap_ci(d_ref, n_boot=args.n_boot, seed=args.seed + 3)
    sel_p = sign_flip_p(d_ref, n_perm=args.n_perm, seed=args.seed + 4)
    shuf = _shuffle_null(P, pick, ref, args.n_perm, args.seed + 5)
    rnd = _random_null(P, ref, args.n_perm, args.seed + 6)
    p_shuf = _one_sided_p(shuf, sel_m)
    p_rand = _one_sided_p(rnd, sel_m)
    argmax_true = np.array([int(np.nanargmax(P[i])) for i in range(n)])
    match = float(np.mean(pick == argmax_true))
    captured = sel_m / h_ref_m if abs(h_ref_m) > 1e-9 else float("nan")

    # ---- verdict ----
    real_headroom = (h_ref_lo > 0) and (h_ref_p < 0.05)
    routable = (sel_lo > 0) and (sel_p < 0.05) and (math.isfinite(p_shuf) and p_shuf < 0.05)
    if real_headroom and routable:
        verdict = "REAL headroom AND routable with a GT-free selector (deployable win)"
    elif real_headroom:
        verdict = ("REAL headroom but NOT routable with these cheap signals "
                   "(selector ~ shuffle) — try a stronger signal (model likelihood)")
    else:
        verdict = "no real best-of-k headroom on this pool"

    payload = {
        "summary": str(args.summary), "n_videos": n, "k": k,
        "oracle_headroom_vs_ref": {"mean": h_ref_m, "ci": [h_ref_lo, h_ref_hi], "signflip_p": h_ref_p},
        "oracle_headroom_vs_meanseed": {"mean": h_mean_m, "ci": [h_mean_lo, h_mean_hi]},
        "frac_videos_with_better_seed": frac_better,
        "signal_within_video_spearman": sig_corr,
        "oof_selector": {
            "d_vs_ref_mean": sel_m, "d_vs_ref_ci": [sel_lo, sel_hi], "signflip_p": sel_p,
            "p_vs_shuffle": p_shuf, "p_vs_random": p_rand,
            "shuffle_null_hi95": float(np.percentile(shuf, 97.5)),
            "random_null_hi95": float(np.percentile(rnd, 97.5)),
            "match_rate": match, "match_chance": 1.0 / k,
            "captured_of_oracle": captured,
        },
        "verdict": verdict,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "bestofk_headroom.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Best-of-k seed headroom + routability probe",
        "",
        f"**Run:** `{args.summary}`  ·  **N videos:** {n}  ·  **k seeds:** {k}  ·  "
        f"bootstrap={args.n_boot}, null={args.n_perm}.",
        "",
        "## (A) Is there REAL seed-selection headroom?",
        "",
        "| Comparison | mean Δ | 95% CI | sign-flip p |",
        "|---|---:|---|---:|",
        f"| oracle best-of-k − reference (seed0) | {_fd(h_ref_m)} dB | "
        f"[{_fd(h_ref_lo)}, {_fd(h_ref_hi)}] | {_fp(h_ref_p)} |",
        f"| oracle best-of-k − mean random seed | {_fd(h_mean_m)} dB | "
        f"[{_fd(h_mean_lo)}, {_fd(h_mean_hi)}] | — |",
        "",
        f"Fraction of videos where a better-than-seed0 candidate exists: **{frac_better:.1%}**.",
        "",
        "## (B) Is the headroom ROUTABLE with a deploy-legitimate (GT-free) selector?",
        "",
        "Within-video Spearman corr between each cheap signal and PSNR across the k "
        "seeds (positive ⇒ signal ranks better seeds higher; deploy-legitimate):",
        "",
        "| Signal | mean within-video Spearman(signal, PSNR) |",
        "|---|---:|",
    ]
    for s in _SIGNALS:
        lines.append(f"| `{s}` | {_fd(sig_corr[s], 3)} |")
    lines += [
        "",
        "Leakage-free OOF ridge selector (fold by video) on the GT-free signals:",
        "",
        "| Metric | value |",
        "|---|---:|",
        f"| Δ vs reference (seed0) | {_fd(sel_m)} dB [{_fd(sel_lo)}, {_fd(sel_hi)}] |",
        f"| sign-flip p (Δ>0) | {_fp(sel_p)} |",
        f"| p vs shuffle-picks null | {_fp(p_shuf)} |",
        f"| p vs random-seed null | {_fp(p_rand)} |",
        f"| match% (argmax hit) / chance | {_fp(match)} / {_fp(1.0 / k)} |",
        f"| captured of oracle headroom | {_fp(captured)} |",
        "",
        f"## Verdict\n\n**{verdict}**",
        "",
    ]
    report = args.output_dir / "bestofk_headroom_summary.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report}")
    print(f"[headroom vs seed0] {_fd(h_ref_m)} dB CI[{_fd(h_ref_lo)},{_fd(h_ref_hi)}] "
          f"p={_fp(h_ref_p)}; selector Δ={_fd(sel_m)} p_shuf={_fp(p_shuf)} -> {verdict}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
