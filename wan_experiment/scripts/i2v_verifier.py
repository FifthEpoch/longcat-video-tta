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


def verifier_score(
    free: Dict[str, float],
    ref: Dict[str, float],
    seam_weight: float = 1.0,
) -> float:
    """Lower = closer to ref. Two-sided deviation (does not reward freeze)."""
    s = 0.0
    for k in VERIFIER_SIGNALS:
        rv, cv = ref.get(k), free.get(k)
        if rv is None or cv is None or rv != rv or cv != cv:
            continue
        s += abs(cv - rv) / (abs(rv) + 1e-6)
    seam_jump = free.get("seam_jump")
    ref_motion = ref.get("temporal_motion")
    if (
        seam_jump is not None and seam_jump == seam_jump
        and ref_motion is not None and ref_motion == ref_motion
    ):
        s += seam_weight * seam_jump / (ref_motion + 1e-6)
    return float(s)
