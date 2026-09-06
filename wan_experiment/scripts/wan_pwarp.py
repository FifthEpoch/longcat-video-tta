"""Slide the guessed picture after pass 1 (pred-only, not extra).

Ordinary white extras. Direction = leftover mean flow.
Magnitude = integer latent pixels per 3-latent strip (default 1).
Holes = edge repeat. No wrap. No noise painted into pred.
"""
from __future__ import annotations

import math

DEFAULT_STEP = 1
_EPS = 1e-6


def _shift_replicate(pred, dy: int, dx: int):
    """Translate [B, T, C, H, W]. New edge = repeat. No wrap."""
    import torch.nn.functional as F

    if dy == 0 and dx == 0:
        return pred
    bsz, n_t, c, h, w = pred.shape
    pad_top = max(int(dy), 0)
    pad_bot = max(-int(dy), 0)
    pad_left = max(int(dx), 0)
    pad_right = max(-int(dx), 0)
    flat = pred.reshape(bsz * n_t, c, h, w)
    padded = F.pad(flat, (pad_left, pad_right, pad_top, pad_bot), mode="replicate")
    ys = pad_top - int(dy)
    xs = pad_left - int(dx)
    out = padded[:, :, ys:ys + h, xs:xs + w]
    return out.reshape(bsz, n_t, c, h, w)


class PWarpState:
    """Frozen leftover direction + integer slide of pred after pass 1."""

    def __init__(
        self,
        vy_lat: float,
        vx_lat: float,
        step: int = DEFAULT_STEP,
        enabled: bool = True,
        flow_log: dict | None = None,
    ):
        self.vy = float(vy_lat)
        self.vx = float(vx_lat)
        self.step = int(step)
        self.enabled = bool(enabled)
        self.n_shifts = 0
        self.last_dy = 0
        self.last_dx = 0
        self.last_log: dict = {}
        self.flow_log = dict(flow_log or {})

    def _block_shift(self) -> tuple[int, int]:
        ay, ax = abs(self.vy), abs(self.vx)
        if ay + ax < _EPS:
            return 0, 0
        mag = self.step if self.step > 0 else 0
        if mag <= 0:
            return 0, 0
        if ay >= ax:
            return int(math.copysign(mag, self.vy)), 0
        return 0, int(math.copysign(mag, self.vx))

    def pred_fn(self, denoised_pred, rng=None, index: int = 0):
        """Shift pred after pass 1. Caller should invoke only at index==0."""
        del rng
        if (not self.enabled) or int(index) != 0:
            return denoised_pred
        dy, dx = self._block_shift()
        out = _shift_replicate(denoised_pred, dy, dx)
        self.n_shifts += 1
        self.last_dy = int(dy)
        self.last_dx = int(dx)
        self.last_log = {
            "pwarp": True,
            "n_shifts": int(self.n_shifts),
            "dy": int(dy),
            "dx": int(dx),
            "step": int(self.step),
            "vy_lat": self.vy,
            "vx_lat": self.vx,
            **self.flow_log,
        }
        return out
