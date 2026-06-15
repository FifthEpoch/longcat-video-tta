#!/usr/bin/env python3
"""Characterize per-video oracle routing winners (NOTTA / AdaSteer / LoRA).

Oracle policy: per video, pick the method with highest *absolute* PSNR among
NOTTA, AdaSteer (ADA), and LoRA_R8_TTA; population mean oracle PSNR is the
oracle routing uplift (+0.226 dB vs always-NOTTA on Panda 999-video intersection).

Reads ``per_video_gains.csv`` (required) and optionally joins feature batteries
from the Phase-0 gating pipeline:

  * ``video_features.csv``       — Tier-1 cuts / CLIP / DINO / flow / bpp / FFT / VAE
  * ``diffusion_ood_scores.csv`` — diffusion flow-matching MSE + latent stats
  * ``tier3_probe_features.csv`` — grad-norm + single-step loss-drop probes

When auxiliary CSVs are absent locally, pass cluster paths via CLI flags.
See ``sweep_experiment/reports/RUNBOOK_friday_morning_2026-06-12.md`` for
expected cluster locations under
``sweep_experiment/reports/per_video_analysis/2026-06-09/``.

Emits under ``--output-dir``:
  * ``oracle_winner_characteristics.md``  — slide-ready tables + narrative
  * ``oracle_winner_feature_stats.csv``   — per-feature per-bucket means/medians
  * ``oracle_winner_anova.csv``           — one-way ANOVA F + eta² across buckets

Usage:
    python scripts/analyze_oracle_winner_characteristics.py

    python scripts/analyze_oracle_winner_characteristics.py \\
        --gains-csv sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv \\
        --features-csv sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv \\
        --ood-csv sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv \\
        --tier3-csv sweep_experiment/reports/per_video_analysis/2026-06-09/tier3_probe_features.csv \\
        --output-dir sweep_experiment/reports/per_video_analysis/2026-06-09/oracle_winner_analysis
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_GAINS = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/per_video_gains.csv"
)
DEFAULT_FEATURES = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv"
)
DEFAULT_OOD = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv"
)
DEFAULT_TIER3 = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/tier3_probe_features.csv"
)
DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "sweep_experiment/reports/per_video_analysis/2026-06-09/oracle_winner_analysis"
)

BASELINE = "NOTTA"
ADA = "ADA"
LORA = "LORA_R8_TTA"
WINNER_LABELS = {
    BASELINE: "NOTTA",
    ADA: "AdaSteer",
    LORA: "LoRA",
}
BUCKETS: Tuple[str, ...] = (BASELINE, ADA, LORA)

# Columns always present in per_video_gains.csv (baseline features).
GAINS_FEATURES: Tuple[str, ...] = (
    "mean_flow",
    "caption_len_words",
    "caption_len_chars",
    "NOTTA_psnr",
)

NON_FEATURE_COLS = frozenset(
    {
        "video_id",
        "caption",
        "n_frames_used",
        "n_visible_frames",
        "n_gen_target_frames",
        "tta_visible_range",
        "gen_target_range",
        "seed",
        "clip_model",
        "dino_model",
        "hist_bins_per_channel",
        "hist_bhattacharyya_thresh",
        "flow_model",
        "input_size_h",
        "input_size_w",
    }
)

KEY_OOD_COLS: Tuple[str, ...] = (
    "mean_diffusion_loss_caption",
    "mean_diffusion_loss_uncond",
    "delta_caption_minus_uncond",
    "latent_norm_mean",
    "latent_norm_std",
    "latent_kurtosis",
)

KEY_VIDEO_FEATURES: Tuple[str, ...] = (
    "cut_count_pyscenedetect",
    "cut_count_histogram",
    "cut_density_per_frame",
    "clip_text_image_sim_mean",
    "clip_text_image_sim_var",
    "dino_temporal_l2_mean",
    "laplacian_variance_mean",
    "rgb_histogram_entropy_mean",
    "bpp_h264",
    "hf_energy_ratio_3d",
    "rec_err_l1",
)

KEY_TIER3_COLS: Tuple[str, ...] = (
    "mean_grad_norm_lora",
    "mean_loss_drop_pct",
)


def _coerce(v) -> float:
    if v is None or v == "":
        return float("nan")
    try:
        x = float(v)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(x) or math.isinf(x):
        return float("nan")
    return x


def _f(row: dict, key: str) -> float:
    return _coerce(row.get(key))


def load_csv_rows(path: Path) -> List[dict]:
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))


def oracle_winner(row: dict) -> str:
    psnrs = {
        BASELINE: _f(row, f"{BASELINE}_psnr"),
        ADA: _f(row, f"{ADA}_psnr"),
        LORA: _f(row, f"{LORA}_psnr"),
    }
    return max(psnrs, key=lambda k: psnrs[k])


def discover_numeric_cols(rows: List[dict], known: Sequence[str]) -> List[str]:
    present = [c for c in known if c in rows[0]]
    if present:
        return present
    cols: List[str] = []
    for k in rows[0].keys():
        if k in NON_FEATURE_COLS:
            continue
        v = _coerce(rows[0].get(k))
        if not math.isnan(v):
            cols.append(k)
    return cols


def join_rows(
    gains: List[dict],
    aux: Optional[List[dict]],
    aux_cols: Sequence[str],
) -> List[dict]:
    aux_map: Dict[str, dict] = {}
    if aux:
        for r in aux:
            vid = (r.get("video_id") or "").strip()
            if vid:
                aux_map[vid] = r
    out: List[dict] = []
    for g in gains:
        vid = (g.get("video_id") or "").strip()
        if not vid:
            continue
        merged = dict(g)
        if vid in aux_map:
            for c in aux_cols:
                if c in aux_map[vid]:
                    merged[c] = aux_map[vid][c]
        out.append(merged)
    return out


def _stats(arr: Sequence[float]) -> Tuple[int, float, float]:
    a = np.asarray([x for x in arr if not np.isnan(x)], dtype=float)
    if a.size == 0:
        return 0, float("nan"), float("nan")
    return int(a.size), float(np.mean(a)), float(np.median(a))


def one_way_anova(groups: Dict[str, List[float]]) -> Tuple[float, float, int]:
    """Return (F, eta_squared, n_total) for 3-group one-way ANOVA."""
    all_vals: List[float] = []
    group_arrays: List[np.ndarray] = []
    for b in BUCKETS:
        vals = [x for x in groups.get(b, []) if not np.isnan(x)]
        if vals:
            group_arrays.append(np.asarray(vals, dtype=float))
            all_vals.extend(vals)
    if len(group_arrays) < 2 or len(all_vals) < 6:
        return float("nan"), float("nan"), len(all_vals)
    grand_mean = float(np.mean(all_vals))
    n_total = len(all_vals)
    k = len(group_arrays)
    ss_between = sum(
        arr.size * (float(arr.mean()) - grand_mean) ** 2 for arr in group_arrays
    )
    ss_within = sum(float(np.sum((arr - arr.mean()) ** 2)) for arr in group_arrays)
    if ss_within <= 0:
        return float("nan"), float("nan"), n_total
    df_between = k - 1
    df_within = n_total - k
    if df_within <= 0:
        return float("nan"), float("nan"), n_total
    f_stat = (ss_between / df_between) / (ss_within / df_within)
    eta_sq = ss_between / (ss_between + ss_within)
    return float(f_stat), float(eta_sq), n_total


def quintile_bin_indices(values: np.ndarray, n_bins: int = 5) -> List[np.ndarray]:
    """Return list of index arrays per quintile (low → high)."""
    mask = ~np.isnan(values)
    if not mask.any():
        return []
    v = values[mask]
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n_bins + 1)))
    if edges.size < 2:
        return [np.where(mask)[0]]
    bins: List[np.ndarray] = []
    for qi in range(edges.size - 1):
        lo, hi = edges[qi], edges[qi + 1]
        if qi == 0:
            idx = np.where(mask & (values <= hi))[0]
        elif qi == edges.size - 2:
            idx = np.where(mask & (values >= lo))[0]
        else:
            idx = np.where(mask & (values >= lo) & (values < hi))[0]
        bins.append(idx)
    return bins


def winner_fraction_table(rows: List[dict], feature: str) -> List[str]:
    lines = [
        f"### Oracle winner mix by `{feature}` quintile",
        "",
        "| Quintile | N | NOTTA % | AdaSteer % | LoRA % |",
        "|---|---:|---:|---:|---:|",
    ]
    vals = np.asarray([_f(r, feature) for r in rows], dtype=float)
    bin_indices = quintile_bin_indices(vals)
    for qi, idx in enumerate(bin_indices):
        n = len(idx)
        if n == 0:
            lines.append(f"| Q{qi + 1} | 0 | — | — | — |")
            continue
        counts = {b: 0 for b in BUCKETS}
        for i in idx:
            counts[oracle_winner(rows[i])] += 1
        lines.append(
            f"| Q{qi + 1} | {n} | "
            f"{100 * counts[BASELINE] / n:.1f}% | "
            f"{100 * counts[ADA] / n:.1f}% | "
            f"{100 * counts[LORA] / n:.1f}% |"
        )
    lines.append("")
    return lines


def build_report(
    rows: List[dict],
    feature_cols: List[str],
    loaded_sources: Dict[str, bool],
) -> Tuple[str, List[dict], List[dict]]:
    n = len(rows)
    lines: List[str] = [
        "# Oracle winner characteristics (Panda 1000v, N=999)",
        "",
        "## Oracle definition (confirmed)",
        "",
        "Per video, oracle routing picks the method with the **highest absolute PSNR** "
        "among NOTTA, AdaSteer (`ADA`), and LoRA_R8_TTA. Population oracle PSNR is "
        "the mean of those per-video best PSNR values (not the mean of per-video ΔPSNR).",
        "",
    ]

    notta_psnr = [_f(r, f"{BASELINE}_psnr") for r in rows]
    ada_psnr = [_f(r, f"{ADA}_psnr") for r in rows]
    lora_psnr = [_f(r, f"{LORA}_psnr") for r in rows]
    ada_d = [_f(r, f"{ADA}_dpsnr") for r in rows]
    lora_d = [_f(r, f"{LORA}_dpsnr") for r in rows]

    oracle_psnr: List[float] = []
    oracle_gain: List[float] = []
    winners: Dict[str, int] = {b: 0 for b in BUCKETS}

    for r in rows:
        w = oracle_winner(r)
        winners[w] += 1
        p = {
            BASELINE: _f(r, f"{BASELINE}_psnr"),
            ADA: _f(r, f"{ADA}_psnr"),
            LORA: _f(r, f"{LORA}_psnr"),
        }[w]
        oracle_psnr.append(p)
        oracle_gain.append(p - _f(r, f"{BASELINE}_psnr"))

    def mean_psnr(arr: Sequence[float]) -> float:
        return float(np.mean(np.asarray(arr, dtype=float)))

    lines += [
        "| Policy | Mean PSNR | Δ vs always-NOTTA |",
        "|---|---:|---:|",
        f"| Always NOTTA | {mean_psnr(notta_psnr):.3f} dB | 0.000 dB |",
        f"| Always AdaSteer | {mean_psnr(ada_psnr):.3f} dB | "
        f"{mean_psnr(ada_psnr) - mean_psnr(notta_psnr):+.3f} dB |",
        f"| Always LoRA | {mean_psnr(lora_psnr):.3f} dB | "
        f"{mean_psnr(lora_psnr) - mean_psnr(notta_psnr):+.3f} dB |",
        f"| **Oracle (best PSNR)** | **{mean_psnr(oracle_psnr):.3f} dB** | "
        f"**{mean_psnr(oracle_psnr) - mean_psnr(notta_psnr):+.3f} dB** |",
        "",
        f"**Oracle picks:** NOTTA {winners[BASELINE]} ({100*winners[BASELINE]/n:.1f}%) · "
        f"AdaSteer {winners[ADA]} ({100*winners[ADA]/n:.1f}%) · "
        f"LoRA {winners[LORA]} ({100*winners[LORA]/n:.1f}%)",
        "",
        f"Oracle ΔPSNR vs NOTTA: mean {mean_psnr(oracle_gain):.3f} dB, "
        f"median {float(np.median(oracle_gain)):.3f} dB.",
        "",
        "## Win magnitude: AdaSteer vs LoRA (head-to-head on ΔPSNR)",
        "",
        "When a method *wins* oracle (absolute PSNR), it also tends to win on ΔPSNR. "
        "Head-to-head ΔPSNR comparison quantifies margin sizes when one TTA method "
        "beats the other.",
        "",
    ]

    lora_beats_ada_idx = [i for i in range(n) if lora_d[i] > ada_d[i]]
    ada_beats_lora_idx = [i for i in range(n) if ada_d[i] > lora_d[i]]

    def margin_stats(indices: List[int], delta_key: str, margin_fn) -> Tuple[int, float, float]:
        deltas = [_f(rows[i], delta_key) for i in indices]
        margins = [margin_fn(i) for i in indices]
        cnt, m_d, med_d = _stats(deltas)
        _, m_m, med_m = _stats(margins)
        return cnt, m_m, med_m

    _, ada_win_mean, ada_win_med = margin_stats(
        ada_beats_lora_idx,
        f"{ADA}_dpsnr",
        lambda i: ada_d[i] - lora_d[i],
    )
    _, lora_win_mean, lora_win_med = margin_stats(
        lora_beats_ada_idx,
        f"{LORA}_dpsnr",
        lambda i: lora_d[i] - ada_d[i],
    )

    ada_oracle_wins = [r for r in rows if oracle_winner(r) == ADA]
    lora_oracle_wins = [r for r in rows if oracle_winner(r) == LORA]

    _, ada_oracle_d_mean, ada_oracle_d_med = _stats(
        [_f(r, f"{ADA}_dpsnr") for r in ada_oracle_wins]
    )
    _, lora_oracle_d_mean, lora_oracle_d_med = _stats(
        [_f(r, f"{LORA}_dpsnr") for r in lora_oracle_wins]
    )

    lines += [
        "| Comparison | N | Mean gain (dB) | Median gain (dB) |",
        "|---|---:|---:|---:|",
        f"| AdaSteer beats LoRA (ΔPSNR) | {len(ada_beats_lora_idx)} | "
        f"{ada_win_mean:.3f} | {ada_win_med:.3f} |",
        f"| LoRA beats AdaSteer (ΔPSNR) | {len(lora_beats_ada_idx)} | "
        f"{lora_win_mean:.3f} | {lora_win_med:.3f} |",
        f"| AdaSteer oracle wins → Ada ΔPSNR | {len(ada_oracle_wins)} | "
        f"{ada_oracle_d_mean:.3f} | {ada_oracle_d_med:.3f} |",
        f"| LoRA oracle wins → LoRA ΔPSNR | {len(lora_oracle_wins)} | "
        f"{lora_oracle_d_mean:.3f} | {lora_oracle_d_med:.3f} |",
        "",
        "**Takeaway:** AdaSteer wins are larger in magnitude — when AdaSteer beats LoRA "
        f"on ΔPSNR, mean margin {ada_win_mean:.3f} dB (median {ada_win_med:.3f}) vs "
        f"LoRA wins mean {lora_win_mean:.3f} dB (median {lora_win_med:.3f}). "
        f"Oracle-win ΔPSNR: AdaSteer mean {ada_oracle_d_mean:.3f} dB vs LoRA "
        f"{lora_oracle_d_mean:.3f} dB.",
        "",
    ]

    # Per-bucket feature means
    bucket_rows: Dict[str, List[dict]] = {b: [] for b in BUCKETS}
    for r in rows:
        bucket_rows[oracle_winner(r)].append(r)

    lines += [
        "## Feature means by oracle winner bucket",
        "",
    ]

    if not loaded_sources.get("video_features"):
        lines.append(
            "_Note: `video_features.csv` not loaded — cluster path: "
            "`sweep_experiment/reports/per_video_analysis/2026-06-09/video_features.csv`_"
        )
        lines.append("")
    if not loaded_sources.get("ood"):
        lines.append(
            "_Note: `diffusion_ood_scores.csv` not loaded — cluster path: "
            "`sweep_experiment/reports/per_video_analysis/2026-06-09/diffusion_ood_scores.csv`_"
        )
        lines.append("")
    if not loaded_sources.get("tier3"):
        lines.append(
            "_Note: `tier3_probe_features.csv` not loaded — cluster path: "
            "`sweep_experiment/reports/per_video_analysis/2026-06-09/tier3_probe_features.csv`_"
        )
        lines.append("")

    feature_stats_rows: List[dict] = []
    anova_rows: List[dict] = []

    for feat in feature_cols:
        for bucket in BUCKETS:
            vals = [_f(r, feat) for r in bucket_rows[bucket]]
            cnt, mean_v, med_v = _stats(vals)
            feature_stats_rows.append(
                {
                    "feature": feat,
                    "bucket": WINNER_LABELS[bucket],
                    "n": cnt,
                    "mean": mean_v,
                    "median": med_v,
                }
            )
        groups = {
            b: [_f(r, feat) for r in bucket_rows[b]] for b in BUCKETS
        }
        f_stat, eta_sq, an_n = one_way_anova(groups)
        anova_rows.append(
            {
                "feature": feat,
                "F": f_stat,
                "eta_squared": eta_sq,
                "n": an_n,
            }
        )

    # Table: key features only for slides (top by eta²)
    ranked = sorted(
        [a for a in anova_rows if not math.isnan(a["eta_squared"])],
        key=lambda x: x["eta_squared"],
        reverse=True,
    )
    slide_features = [a["feature"] for a in ranked[:12]]
    if not slide_features:
        slide_features = [f for f in feature_cols if f in GAINS_FEATURES]

    lines.append("| Feature | NOTTA mean | AdaSteer mean | LoRA mean | η² |")
    lines.append("|---|---:|---:|---:|---:|")
    for feat in slide_features:
        means = {}
        for bucket in BUCKETS:
            vals = [_f(r, feat) for r in bucket_rows[bucket]]
            means[bucket] = _stats(vals)[1]
        eta = next((a["eta_squared"] for a in anova_rows if a["feature"] == feat), float("nan"))
        lines.append(
            f"| `{feat}` | "
            f"{means[BASELINE]:.3g} | {means[ADA]:.3g} | {means[LORA]:.3g} | "
            f"{eta:.3f} |"
        )
    lines.append("")

    # OOD hypothesis section
    lines += [
        "## OOD hypothesis (exploratory)",
        "",
        "Illustrative hypotheses (not forced): LoRA on moderately OOD, NOTTA on "
        "extremely OOD, AdaSteer on in-distribution. Test via diffusion OOD quintiles "
        "when `diffusion_ood_scores.csv` is available.",
        "",
    ]

    ood_col = "mean_diffusion_loss_caption"
    if ood_col in feature_cols and any(
        not math.isnan(_f(r, ood_col)) for r in rows
    ):
        ood_vals = np.asarray([_f(r, ood_col) for r in rows], dtype=float)
        bin_indices = quintile_bin_indices(ood_vals)
        if bin_indices:
            lines.extend(winner_fraction_table(rows, ood_col))
            q1_idx = bin_indices[0]
            q5_idx = bin_indices[-1]
            def notta_pct(indices: np.ndarray) -> float:
                if len(indices) == 0:
                    return float("nan")
                return 100 * sum(
                    1 for i in indices if oracle_winner(rows[i]) == BASELINE
                ) / len(indices)
            lines.append(
                f"NOTTA oracle-win share: lowest-OOD quintile "
                f"{notta_pct(q1_idx):.1f}% vs highest-OOD quintile "
                f"{notta_pct(q5_idx):.1f}%."
            )
            lines.append("")
    else:
        lines.append(
            "_OOD quintile stratification skipped — `mean_diffusion_loss_caption` "
            "not available. Re-run with `--ood-csv` after cluster Stage 1b._"
        )
        lines.append("")

    # Per-bucket narrative bullets
    lines += [
        "## Per-bucket characterization (from available features)",
        "",
    ]
    for bucket in BUCKETS:
        label = WINNER_LABELS[bucket]
        br = bucket_rows[bucket]
        lines.append(f"### {label} wins ({len(br)} videos)")
        bullets: List[str] = []
        for feat in ("NOTTA_psnr", "mean_flow", "caption_len_words"):
            if feat in feature_cols or feat in GAINS_FEATURES:
                cnt, mean_v, med_v = _stats([_f(r, feat) for r in br])
                if cnt > 0:
                    bullets.append(f"{feat}: mean {mean_v:.3g}, median {med_v:.3g}")
        if ood_col in feature_cols:
            cnt, mean_v, med_v = _stats([_f(r, ood_col) for r in br])
            if cnt > 0 and not math.isnan(mean_v):
                bullets.append(f"{ood_col}: mean {mean_v:.3g}, median {med_v:.3g}")
        if bullets:
            for b in bullets:
                lines.append(f"- {b}")
        lines.append("")

    lines += [
        "## Top features differing across winner buckets (ANOVA η²)",
        "",
        "| Feature | F | η² | N |",
        "|---|---:|---:|---:|",
    ]
    for a in ranked[:15]:
        lines.append(
            f"| `{a['feature']}` | {a['F']:.2f} | {a['eta_squared']:.3f} | {a['n']} |"
        )
    lines.append("")

    lines += [
        "## Why oracle (+0.226 dB) >> always-AdaSteer (+0.008 dB)",
        "",
        f"- Oracle picks NOTTA on {winners[BASELINE]} videos ({100*winners[BASELINE]/n:.1f}%) "
        "where TTA hurts; always-AdaSteer forces TTA on all of them.",
        f"- Oracle picks LoRA on {winners[LORA]} videos where AdaSteer is suboptimal.",
        f"- Skip-Ada-if-Δ≤0 policy recovers most uplift (~+0.213 dB) because "
        "AdaSteer ΔPSNR is ≤0 on roughly half of videos.",
        "",
    ]

    return "\n".join(lines), feature_stats_rows, anova_rows


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Oracle winner bucket characterization for Panda slides"
    )
    ap.add_argument("--gains-csv", type=Path, default=DEFAULT_GAINS)
    ap.add_argument(
        "--features-csv",
        type=Path,
        default=DEFAULT_FEATURES,
        help="Tier-1 video_features.csv (optional; cluster Stage 1a)",
    )
    ap.add_argument(
        "--ood-csv",
        type=Path,
        default=DEFAULT_OOD,
        help="diffusion_ood_scores.csv (optional; cluster Stage 1b)",
    )
    ap.add_argument(
        "--tier3-csv",
        type=Path,
        default=DEFAULT_TIER3,
        help="tier3_probe_features.csv (optional; cluster Stage 1c)",
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--require-aux",
        action="store_true",
        help="Fail if optional feature/OOD/tier3 CSVs are missing",
    )
    args = ap.parse_args()

    if not args.gains_csv.exists():
        print(f"[error] gains CSV not found: {args.gains_csv}", file=sys.stderr)
        return 2

    gains_rows = load_csv_rows(args.gains_csv)
    loaded_sources: Dict[str, bool] = {
        "video_features": False,
        "ood": False,
        "tier3": False,
    }
    feature_cols: List[str] = list(GAINS_FEATURES)

    aux_video: Optional[List[dict]] = None
    if args.features_csv.exists():
        aux_video = load_csv_rows(args.features_csv)
        loaded_sources["video_features"] = True
        vf_cols = discover_numeric_cols(aux_video, KEY_VIDEO_FEATURES)
        feature_cols.extend([c for c in vf_cols if c not in feature_cols])
    elif args.require_aux:
        print(f"[error] features CSV not found: {args.features_csv}", file=sys.stderr)
        return 2

    aux_ood: Optional[List[dict]] = None
    if args.ood_csv.exists():
        aux_ood = load_csv_rows(args.ood_csv)
        loaded_sources["ood"] = True
        ood_cols = discover_numeric_cols(aux_ood, KEY_OOD_COLS)
        feature_cols.extend([c for c in ood_cols if c not in feature_cols])
    elif args.require_aux:
        print(f"[error] OOD CSV not found: {args.ood_csv}", file=sys.stderr)
        return 2

    aux_tier3: Optional[List[dict]] = None
    if args.tier3_csv.exists():
        aux_tier3 = load_csv_rows(args.tier3_csv)
        loaded_sources["tier3"] = True
        t3_cols = discover_numeric_cols(aux_tier3, KEY_TIER3_COLS)
        feature_cols.extend([c for c in t3_cols if c not in feature_cols])
    elif args.require_aux:
        print(f"[error] tier3 CSV not found: {args.tier3_csv}", file=sys.stderr)
        return 2

    rows = join_rows(gains_rows, aux_video, feature_cols)
    if aux_ood:
        ood_map = {(r.get("video_id") or "").strip(): r for r in aux_ood}
        for r in rows:
            vid = (r.get("video_id") or "").strip()
            if vid in ood_map:
                for k, v in ood_map[vid].items():
                    if k not in NON_FEATURE_COLS and k != "video_id":
                        r[k] = v
    if aux_tier3:
        t3_map = {(r.get("video_id") or "").strip(): r for r in aux_tier3}
        for r in rows:
            vid = (r.get("video_id") or "").strip()
            if vid in t3_map:
                for k, v in t3_map[vid].items():
                    if k not in NON_FEATURE_COLS and k != "video_id":
                        r[k] = v

    feature_cols = sorted(set(feature_cols))

    report, stats_rows, anova_rows = build_report(rows, feature_cols, loaded_sources)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "oracle_winner_characteristics.md"
    report_path.write_text(report, encoding="utf-8")
    write_csv(
        out_dir / "oracle_winner_feature_stats.csv",
        stats_rows,
        ["feature", "bucket", "n", "mean", "median"],
    )
    write_csv(
        out_dir / "oracle_winner_anova.csv",
        anova_rows,
        ["feature", "F", "eta_squared", "n"],
    )

    print(report)
    print(f"\nWrote {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
