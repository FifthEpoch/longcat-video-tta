"""AdaSteer on Wan / Self-Forcing: one δ on the timestep embed (or residual).

LongCat paper AdaSteer is t' = t + δ into frozen per-block adaLN.
Wan analogue: hook CausalWanModel.time_embedding (output dim), then
frozen time_projection supplies the 6×dim adaLN modulation.

Loss is the student's few-step x0 reconstruction on observed latents
only. No future GT. No LoRA. No TTC.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW


ADASTEER_METHODS = ("ada_fixed", "ada_stream", "ada_resid")
DEFAULT_STEPS = 10
DEFAULT_LR = 5e-3
DEFAULT_BLEND = 0.5
DEFAULT_REFIT_STEPS = 5


def _clone_cond(conditional_dict: dict) -> dict:
    out = {}
    for key, val in conditional_dict.items():
        out[key] = val.detach().clone() if torch.is_tensor(val) else val
    return out


def _drop_kv_caches(pipeline) -> None:
    """Caches allocated under inference_mode cannot be updated outside it."""
    for name in ("kv_cache1", "kv_cache2", "crossattn_cache"):
        if hasattr(pipeline, name):
            setattr(pipeline, name, None)


def _causal_model(pipeline) -> nn.Module:
    gen = getattr(pipeline, "generator", None)
    model = getattr(gen, "model", gen)
    if model is None:
        raise RuntimeError("pipeline.generator.model missing")
    if hasattr(model, "get_base_model"):
        try:
            model = model.get_base_model()
        except Exception:
            pass
    inner = getattr(model, "module", None)
    if inner is not None and hasattr(inner, "time_embedding"):
        model = inner
    if not hasattr(model, "time_embedding"):
        names = [n for n, _ in model.named_modules() if "time" in n.lower()]
        raise RuntimeError(f"no time_embedding on generator.model; time modules={names[:12]}")
    return model


class WanAdaSteer:
    def __init__(self, pipeline, placement: str = "time_embed"):
        self.pipeline = pipeline
        self.placement = placement
        self.model = _causal_model(pipeline)
        self._hooks: list = []
        if placement == "time_embed":
            last = self.model.time_embedding[-1]
            dim = int(getattr(last, "out_features", 0) or self.model.dim)
            self.inject_dim = dim
            self._block_ids: list[int] = []
        elif placement == "residual":
            dim = int(self.model.dim)
            n = len(self.model.blocks)
            lo = int(round(0.55 * n))
            hi = int(round(0.80 * n))
            self._block_ids = list(range(lo, max(lo + 1, hi)))
            self.inject_dim = dim
        else:
            raise ValueError(f"unknown placement {placement}")
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        self.delta = nn.Parameter(torch.zeros(self.inject_dim, device=device, dtype=dtype))
        self.delta0: torch.Tensor | None = None
        self.fit_log: dict[str, Any] = {}

    def apply(self) -> None:
        self.remove()
        delta = self.delta
        if self.placement == "time_embed":
            def _hook(_m, _inp, out):
                return out + delta.to(dtype=out.dtype)

            self._hooks.append(self.model.time_embedding.register_forward_hook(_hook))
            return
        def _mk():
            def _h(_m, _inp, out):
                if isinstance(out, tuple):
                    return (out[0] + delta.to(dtype=out[0].dtype),) + tuple(out[1:])
                return out + delta.to(dtype=out.dtype)
            return _h

        for bi in self._block_ids:
            self._hooks.append(self.model.blocks[bi].register_forward_hook(_mk()))

    def remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def snapshot(self) -> None:
        self.delta0 = self.delta.detach().clone()

    def optimize(
        self,
        clean_latents: torch.Tensor,
        conditional_dict: dict,
        *,
        steps: int,
        lr: float,
        device,
        tag: str,
    ) -> dict[str, Any]:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.delta.requires_grad_(True)
        self.apply()
        opt = AdamW([self.delta], lr=float(lr), betas=(0.9, 0.999), eps=1e-15)
        losses: list[float] = []
        raw_norms: list[float] = []
        # Never fit under the runner's inference_mode. Prefix / KV
        # caches created there are inference tensors; inplace updates
        # then raise. Drop caches, clone inputs, then fit.
        with torch.inference_mode(False), torch.enable_grad():
            clean = clean_latents.detach().clone()
            cond = _clone_cond(conditional_dict)
            _drop_kv_caches(self.pipeline)
            for _ in range(int(steps)):
                opt.zero_grad(set_to_none=True)
                loss = _student_x0_loss(
                    self.pipeline, clean, cond, device,
                )
                if not loss.requires_grad:
                    raise RuntimeError(
                        "AdaSteer loss has no grad_fn after leaving "
                        "inference_mode. generator forward is detached."
                    )
                loss.backward()
                raw = float(torch.nn.utils.clip_grad_norm_([self.delta], float("inf")).item())
                raw_norms.append(raw)
                if raw > 1.0 and self.delta.grad is not None:
                    self.delta.grad.mul_(1.0 / (raw + 1e-6))
                opt.step()
                losses.append(float(loss.detach().item()))
        info = {
            "tag": tag,
            "placement": self.placement,
            "steps": int(steps),
            "lr": float(lr),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
            "delta_norm": float(self.delta.detach().float().norm().item()),
            "grad_norm_max": max(raw_norms) if raw_norms else None,
            "inject_dim": int(self.inject_dim),
            "residual_blocks": list(self._block_ids),
        }
        self.fit_log = info
        print(
            f"  adasteer {tag} place={self.placement} "
            f"loss {info['loss_first']}->{info['loss_last']} "
            f"|δ|={info['delta_norm']:.4g}",
            flush=True,
        )
        return info

    def stream_update(
        self,
        window_latents: torch.Tensor,
        conditional_dict: dict,
        *,
        steps: int,
        lr: float,
        blend: float,
        device,
        chunk: int,
    ) -> dict[str, Any]:
        info = self.optimize(
            window_latents, conditional_dict,
            steps=steps, lr=lr, device=device, tag=f"stream_c{chunk}",
        )
        if self.delta0 is not None:
            lam = float(blend)
            with torch.no_grad():
                self.delta.copy_((1.0 - lam) * self.delta + lam * self.delta0)
            info["blend"] = lam
            info["delta_norm_after_blend"] = float(
                self.delta.detach().float().norm().item()
            )
        self.apply()
        return info


def _student_x0_loss(pipeline, clean, conditional_dict, device) -> torch.Tensor:
    raw = pipeline.denoising_step_list
    if hasattr(raw, "detach"):
        steps = [float(s) for s in raw.detach().cpu().tolist()]
    else:
        steps = [float(s) for s in list(raw)]
    if not steps:
        raise RuntimeError("empty denoising_step_list")
    tval = steps[int(torch.randint(0, len(steps), (1,)).item())]
    n = int(clean.shape[1])
    noise = torch.randn_like(clean)
    t_flat = torch.ones([n], device=device, dtype=torch.long) * int(tval)
    noisy = pipeline.scheduler.add_noise(
        clean.flatten(0, 1), noise.flatten(0, 1), t_flat,
    ).unflatten(0, clean.shape[:2])
    _drop_kv_caches(pipeline)
    pipeline._initialize_kv_cache(1, clean.dtype, device)
    pipeline._initialize_crossattn_cache(1, clean.dtype, device)
    timestep = torch.ones([1, n], device=device, dtype=torch.float32) * float(tval)
    _, pred = pipeline.generator(
        noisy_image_or_video=noisy,
        conditional_dict=conditional_dict,
        timestep=timestep,
        kv_cache=pipeline.kv_cache1,
        crossattn_cache=pipeline.crossattn_cache,
        current_start=0,
    )
    return (pred.float() - clean.float()).pow(2).mean()


def fit_for_method(
    pipeline,
    method: str,
    prefix_latents: torch.Tensor,
    conditional_dict: dict,
    device,
    *,
    steps: int = DEFAULT_STEPS,
    lr: float = DEFAULT_LR,
) -> WanAdaSteer:
    placement = "residual" if method == "ada_resid" else "time_embed"
    ctl = WanAdaSteer(pipeline, placement=placement)
    ctl.optimize(
        prefix_latents, conditional_dict,
        steps=steps, lr=lr, device=device, tag="prefix",
    )
    ctl.snapshot()
    ctl.apply()
    return ctl
