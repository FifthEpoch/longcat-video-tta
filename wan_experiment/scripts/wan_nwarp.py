"""Leftover-flow HIWYN extras for Self Forcing (extra-only, not pred).

Pass 1 stays ordinary white. Later extras are spatially white per
frame, transported by a frozen leftover mean velocity, holes
resampled (no wrap), mixed with plain snow at gamma. Particle field
carries across 3-latent blocks and 21-latent chunks.
"""
from __future__ import annotations

import math

import numpy as np

# Wan VAE: 8× spatial, 4× temporal (one latent ≈ 4 pixel frames).
_SPATIAL = 8.0
_TEMPORAL = 4.0
DEFAULT_GAMMA = 0.5


def leftover_mean_flow_px(frames: np.ndarray) -> tuple[float, float, dict]:
    """Mean optical flow of a leftover clip [T,H,W,C] in [0, 1].

    Returns (vy, vx) in pixels per pixel-frame, plus a small log.
    Farneback if OpenCV is present; else a phase-correlation fallback.
    """
    frames = np.clip(np.asarray(frames, dtype=np.float32), 0.0, 1.0)
    if frames.ndim != 4 or frames.shape[0] < 2:
        return 0.0, 0.0, {"backend": "empty", "n_pairs": 0}
    grays = (
        0.299 * frames[..., 0] + 0.587 * frames[..., 1] + 0.114 * frames[..., 2]
    )
    # Every other frame is enough for a leftover mean.
    idx = list(range(0, grays.shape[0], 2))
    if len(idx) < 2:
        idx = list(range(grays.shape[0]))
    try:
        import cv2
    except Exception:
        return _phase_mean_flow(grays[idx])
    vys, vxs = [], []
    for a, b in zip(idx[:-1], idx[1:]):
        g0 = np.clip(grays[a] * 255.0, 0, 255).astype(np.uint8)
        g1 = np.clip(grays[b] * 255.0, 0, 255).astype(np.uint8)
        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0,
        )
        # Farneback is (x, y) in the last dim.
        vxs.append(float(np.mean(flow[..., 0])) / float(b - a))
        vys.append(float(np.mean(flow[..., 1])) / float(b - a))
    vy = float(np.mean(vys)) if vys else 0.0
    vx = float(np.mean(vxs)) if vxs else 0.0
    return vy, vx, {
        "backend": "farneback",
        "n_pairs": len(vys),
        "vy_px": vy,
        "vx_px": vx,
    }


def _phase_mean_flow(grays: np.ndarray) -> tuple[float, float, dict]:
    vys, vxs = [], []
    for a, b in zip(range(len(grays) - 1), range(1, len(grays))):
        f0 = np.fft.rfft2(grays[a])
        f1 = np.fft.rfft2(grays[b])
        r = f0 * np.conj(f1)
        r /= np.maximum(np.abs(r), 1e-8)
        corr = np.fft.irfft2(r, s=grays[a].shape)
        peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
        dy = float(peak[0])
        dx = float(peak[1])
        h, w = corr.shape
        if dy > h / 2.0:
            dy -= h
        if dx > w / 2.0:
            dx -= w
        vys.append(dy / float(b - a))
        vxs.append(dx / float(b - a))
    vy = float(np.mean(vys)) if vys else 0.0
    vx = float(np.mean(vxs)) if vxs else 0.0
    return vy, vx, {
        "backend": "phase",
        "n_pairs": len(vys),
        "vy_px": vy,
        "vx_px": vx,
    }


def leftover_vel_latent(vy_px: float, vx_px: float) -> tuple[float, float]:
    """Pixels / pixel-frame → latent pixels / latent frame."""
    return (
        float(vy_px) * _TEMPORAL / _SPATIAL,
        float(vx_px) * _TEMPORAL / _SPATIAL,
    )


def _shift_fill(field, dy: int, dx: int, rng):
    """Integer translate. Holes = fresh Gaussian. No wrap."""
    import torch

    if dy == 0 and dx == 0:
        return field
    c, h, w = field.shape
    out = torch.randn(
        field.shape, device=field.device, dtype=field.dtype, generator=rng,
    )
    ys0 = max(0, -int(dy))
    ys1 = min(h, h - int(dy))
    yd0 = max(0, int(dy))
    yd1 = min(h, h + int(dy))
    xs0 = max(0, -int(dx))
    xs1 = min(w, w - int(dx))
    xd0 = max(0, int(dx))
    xd1 = min(w, w + int(dx))
    if ys1 > ys0 and xs1 > xs0 and yd1 > yd0 and xd1 > xd0:
        out[:, yd0:yd1, xd0:xd1] = field[:, ys0:ys1, xs0:xs1]
    return out


def _gamma_mix(warped, white, gamma: float):
    g = float(gamma)
    g = min(1.0, max(0.0, g))
    denom = math.sqrt((1.0 - g) ** 2 + g ** 2)
    return ((1.0 - g) * warped + g * white) / denom


class NWarpState:
    """Frozen leftover velocity + carried snow field."""

    def __init__(
        self,
        vy_lat: float,
        vx_lat: float,
        gamma: float = DEFAULT_GAMMA,
        enabled: bool = True,
        flow_log: dict | None = None,
    ):
        self.vy = float(vy_lat)
        self.vx = float(vx_lat)
        self.gamma = float(gamma)
        self.enabled = bool(enabled)
        self.y_acc = 0.0
        self.x_acc = 0.0
        self.field = None
        self.n_extras = 0
        self.last_log: dict = {}
        self.flow_log = dict(flow_log or {})

    def extra_fn(self, denoised_pred, rng, index: int):
        """Build extra with the same layout as pred [B, T, C, H, W]."""
        import torch

        del index
        bsz, n_t, c, h, w = denoised_pred.shape
        if not self.enabled:
            return torch.randn(
                denoised_pred.shape,
                device=denoised_pred.device,
                dtype=denoised_pred.dtype,
                generator=rng,
            )
        frames = []
        dy_sum = 0
        dx_sum = 0
        for _ in range(int(n_t)):
            if self.field is None:
                self.field = torch.randn(
                    [c, h, w],
                    device=denoised_pred.device,
                    dtype=denoised_pred.dtype,
                    generator=rng,
                )
            else:
                prev_iy = int(math.floor(self.y_acc))
                prev_ix = int(math.floor(self.x_acc))
                self.y_acc += self.vy
                self.x_acc += self.vx
                dy = int(math.floor(self.y_acc)) - prev_iy
                dx = int(math.floor(self.x_acc)) - prev_ix
                dy_sum += dy
                dx_sum += dx
                self.field = _shift_fill(self.field, dy, dx, rng)
            frames.append(self.field)
        warped = torch.stack(frames, dim=0).unsqueeze(0)
        if bsz != 1:
            warped = warped.expand(bsz, -1, -1, -1, -1).contiguous()
        white = torch.randn(
            warped.shape,
            device=warped.device,
            dtype=warped.dtype,
            generator=rng,
        )
        extra = _gamma_mix(warped, white, self.gamma)
        self.n_extras += 1
        self.last_log = {
            "nwarp": True,
            "n_extras": int(self.n_extras),
            "vy_lat": self.vy,
            "vx_lat": self.vx,
            "y_acc": self.y_acc,
            "x_acc": self.x_acc,
            "dy": int(dy_sum),
            "dx": int(dx_sum),
            "gamma": self.gamma,
            **self.flow_log,
        }
        return extra
