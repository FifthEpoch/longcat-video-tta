#!/usr/bin/env python3
"""Audit old vs new run pairs to diagnose large metric regressions.

Compares two summary.json files on common videos and reports:
- metric deltas on common successful samples
- caption drift rate (exact string comparison)
- key config differences (from config.json and/or summary fields)
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _safe_mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def _fmt(v: Optional[float], nd: int = 6) -> str:
    return "nan" if v is None else f"{v:.{nd}f}"


def _normalize_name(name: str) -> str:
    return Path(name).stem


def _results_by_video(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in summary.get("results", []):
        if not isinstance(row, dict):
            continue
        name = row.get("video_name")
        if not name:
            vp = row.get("video_path")
            if isinstance(vp, str):
                name = Path(vp).name
        if not isinstance(name, str) or not name:
            continue
        out[_normalize_name(name)] = row
    return out


def _load_config_for_summary(summary_path: Path, summary_obj: Dict[str, Any]) -> Dict[str, Any]:
    cfg_path = summary_path.parent / "config.json"
    if cfg_path.exists():
        return _load_json(cfg_path)
    exp_cfg = summary_obj.get("exp_config")
    if isinstance(exp_cfg, dict):
        return exp_cfg
    return {}


def _collect_config_subset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "method",
        "series_name",
        "run_id",
        "checkpoint_dir",
        "data_dir",
        "max_videos",
        "seed",
        "num_cond_frames",
        "num_frames",
        "gen_start_frame",
        "tta_total_frames",
        "tta_context_frames",
        "num_inference_steps",
        "guidance_scale",
        "resolution",
        "num_steps",
        "learning_rate",
        "warmup_steps",
        "delta_steps",
        "delta_lr",
        "delta_target",
        "delta_target_blocks",
        "num_groups",
        "lora_rank",
        "lora_alpha",
        "lora_target_blocks",
        "target_modules",
        "es_disable",
        "es_check_every",
        "es_patience",
        "es_anchor_sigmas",
        "es_noise_draws",
        "es_holdout_fraction",
        "clip_gate_enabled",
        "clip_gate_backend",
        "clip_gate_threshold",
        "clip_gate_log_only",
        "clip_gate_skip_tta",
    ]
    return {k: cfg.get(k) for k in keys if k in cfg}


def _print_config_diffs(old_cfg: Dict[str, Any], new_cfg: Dict[str, Any]) -> None:
    old_sub = _collect_config_subset(old_cfg)
    new_sub = _collect_config_subset(new_cfg)
    all_keys = sorted(set(old_sub.keys()) | set(new_sub.keys()))
    changed = [k for k in all_keys if old_sub.get(k) != new_sub.get(k)]

    print("\n[Config diff]")
    if not changed:
        print("  no differences in tracked keys")
        return
    for k in changed:
        print(f"  {k}: old={old_sub.get(k)!r} | new={new_sub.get(k)!r}")


def _analyze_pair(old_summary_path: Path, new_summary_path: Path, show_examples: int) -> int:
    old_sum = _load_json(old_summary_path)
    new_sum = _load_json(new_summary_path)
    old_cfg = _load_config_for_summary(old_summary_path, old_sum)
    new_cfg = _load_config_for_summary(new_summary_path, new_sum)

    old_rows = _results_by_video(old_sum)
    new_rows = _results_by_video(new_sum)
    common_ids = sorted(set(old_rows.keys()) & set(new_rows.keys()))

    print("=" * 100)
    print("OLD:", old_summary_path)
    print("NEW:", new_summary_path)
    print("common_videos:", len(common_ids))

    if not common_ids:
        print("No common video ids; cannot compare.")
        return 1

    # Compare only samples where both runs reported success.
    common_ok: List[Tuple[str, Dict[str, Any], Dict[str, Any]]] = []
    for vid in common_ids:
        o = old_rows[vid]
        n = new_rows[vid]
        if o.get("success") and n.get("success"):
            common_ok.append((vid, o, n))

    print("common_successful:", len(common_ok))
    if not common_ok:
        print("No common successful videos; cannot compute metric deltas.")
        _print_config_diffs(old_cfg, new_cfg)
        return 1

    d_psnr = []
    d_ssim = []
    d_lpips_improve = []
    caption_same = 0
    caption_diff_examples: List[Tuple[str, str, str]] = []

    for vid, o, n in common_ok:
        o_psnr, n_psnr = o.get("psnr"), n.get("psnr")
        o_ssim, n_ssim = o.get("ssim"), n.get("ssim")
        o_lpips, n_lpips = o.get("lpips"), n.get("lpips")

        if o_psnr is not None and n_psnr is not None:
            d_psnr.append(float(n_psnr) - float(o_psnr))
        if o_ssim is not None and n_ssim is not None:
            d_ssim.append(float(n_ssim) - float(o_ssim))
        if o_lpips is not None and n_lpips is not None:
            # positive means LPIPS improved in NEW
            d_lpips_improve.append(float(o_lpips) - float(n_lpips))

        o_cap = str(o.get("caption", "")).strip()
        n_cap = str(n.get("caption", "")).strip()
        if o_cap == n_cap:
            caption_same += 1
        elif len(caption_diff_examples) < show_examples:
            caption_diff_examples.append((vid, o_cap, n_cap))

    print("\n[Common-video metric deltas: NEW - OLD]")
    print("  mean_delta_psnr:", _fmt(_safe_mean(d_psnr)))
    print("  mean_delta_ssim:", _fmt(_safe_mean(d_ssim)))
    print("  mean_delta_lpips_improve(+):", _fmt(_safe_mean(d_lpips_improve)))

    same_rate = caption_same / len(common_ok) if common_ok else 0.0
    print("\n[Caption drift]")
    print(f"  exact_match_count: {caption_same}/{len(common_ok)}")
    print(f"  exact_match_rate:  {same_rate:.4f}")
    if caption_diff_examples:
        print("  examples (video, old_caption, new_caption):")
        for vid, oc, nc in caption_diff_examples:
            print(f"    - {vid}")
            print(f"      old: {oc[:220]}")
            print(f"      new: {nc[:220]}")

    _print_config_diffs(old_cfg, new_cfg)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Audit regression between two run summaries.")
    p.add_argument("--old-summary", required=True, type=Path, help="Path to historical summary.json")
    p.add_argument("--new-summary", required=True, type=Path, help="Path to rerun summary.json")
    p.add_argument("--show-caption-diffs", type=int, default=5, help="Number of caption mismatch examples")
    args = p.parse_args()

    code = _analyze_pair(args.old_summary, args.new_summary, args.show_caption_diffs)
    raise SystemExit(code)


if __name__ == "__main__":
    main()

