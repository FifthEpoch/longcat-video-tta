"""GT-free drift verifier for Wan I2V chunked search.

Copied from delta_experiment/scripts/diag_longhorizon_drift.py
(gen_free_signals / verifier_score). LOWER composite = closer to the
first-second-after-cond reference + smaller seam. cand0 = NOTTA seed.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

VERIFIER_SIGNALS = ["sharpness", "colorfulness", "contrast", "temporal_motion"]


def _to_gray(frames: np.ndarray) -> np.ndarray:
    return 0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]


def _laplacian_var(gray_t: np.ndarray) -> float:
    g = gray_t
    lap = (
        4.0 * g
        - np.roll(g, 1, axis=0) - np.roll(g, -1, axis=0)
        - np.roll(g, 1, axis=1) - np.roll(g, -1, axis=1)
    )
    lap = lap[1:-1, 1:-1]
    return float(np.var(lap))


def _colorfulness(frame: np.ndarray) -> float:
    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    return (
        (float(np.std(rg)) ** 2 + float(np.std(yb)) ** 2) ** 0.5
        + 0.3 * (float(np.mean(rg)) ** 2 + float(np.mean(yb)) ** 2) ** 0.5
    )


def gen_free_signals(gen_only: np.ndarray, last_cond_frame: np.ndarray) -> Dict[str, float]:
    """GT-free signals on generated frames [T,H,W,3] float[0,1]."""
    gen_only = np.clip(gen_only.astype(np.float32), 0.0, 1.0)
    t = gen_only.shape[0]
    gray = _to_gray(gen_only)
    sharp = float(np.mean([_laplacian_var(gray[i]) for i in range(t)]))
    colorful = float(np.mean([_colorfulness(gen_only[i]) for i in range(t)]))
    contrast = float(np.mean([np.std(gray[i]) for i in range(t)]))
    if t >= 2:
        motion = float(np.mean(np.abs(gen_only[1:] - gen_only[:-1])))
    else:
        motion = float("nan")
    seam_jump = float(np.mean(np.abs(gen_only[0] - np.clip(last_cond_frame, 0, 1))))
    seam_ratio = (
        seam_jump / (motion + 1e-6) if motion == motion and motion > 0 else float("nan")
    )
    return {
        "sharpness": sharp,
        "colorfulness": colorful,
        "contrast": contrast,
        "temporal_motion": motion,
        "seam_jump": seam_jump,
        "seam_ratio": seam_ratio,
    }


def reference_signals(frames_01: np.ndarray) -> Dict[str, float]:
    """Reference from first-1s-after-cond (or any real-ish window)."""
    return gen_free_signals(frames_01, frames_01[0])


def _rel_dev(cur: float, ref: float) -> float:
    if cur is None or ref is None or cur != cur or ref != ref:
        return float("nan")
    return abs(float(cur) - float(ref)) / (abs(float(ref)) + 1e-6)


def signal_devs(free: Dict[str, float], ref: Dict[str, float]) -> Dict[str, float]:
    """Per-signal |cur-ref| / (|ref|+eps). NaN if either side is missing."""
    return {k: _rel_dev(free.get(k), ref.get(k)) for k in VERIFIER_SIGNALS}


def seam_term(free: Dict[str, float], ref: Dict[str, float]) -> float:
    seam_jump = free.get("seam_jump")
    ref_motion = ref.get("temporal_motion")
    if (
        seam_jump is None or seam_jump != seam_jump
        or ref_motion is None or ref_motion != ref_motion
    ):
        return float("nan")
    return float(seam_jump) / (float(ref_motion) + 1e-6)


def score_breakdown(
    free: Dict[str, float],
    ref: Dict[str, float],
    seam_weight: float = 1.0,
) -> Dict[str, float]:
    """Per-term verifier loss. score = sum(devs) + seam_weight * seam_term."""
    devs = signal_devs(free, ref)
    seam = seam_term(free, ref)
    parts = [v for v in devs.values() if v == v]
    if seam == seam:
        parts.append(float(seam_weight) * seam)
    return {
        **{f"dev_{k}": float(v) if v == v else float("nan") for k, v in devs.items()},
        "seam_term": float(seam) if seam == seam else float("nan"),
        "score": float(sum(parts)) if parts else float("nan"),
    }


def verifier_score(
    free: Dict[str, float],
    ref: Dict[str, float],
    seam_weight: float = 1.0,
) -> float:
    """Lower = closer to ref. Two-sided deviation (does not reward freeze)."""
    return float(score_breakdown(free, ref, seam_weight=seam_weight)["score"])


def motion_pick_score(
    free: Dict[str, float],
    ref: Dict[str, float],
    seam_weight: float = 0.25,
) -> float:
    """Higher = more motion. Small seam penalty so a jump-cut does not win.

    One-sided. Do not regularize toward the first-second reference — that
    is what made the I2V-32 composite prefer freeze.
    """
    motion = free.get("temporal_motion")
    if motion is None or motion != motion:
        return float("-inf")
    seam = seam_term(free, ref)
    penalty = float(seam_weight) * seam if seam == seam else 0.0
    return float(motion) - penalty
