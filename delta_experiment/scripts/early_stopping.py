#!/usr/bin/env python3
"""
Conditioning-aware early stopping for diffusion model TTA.

Uses an "anchor loss" computed at fixed timesteps and fixed noise seeds on
held-out validation latents. The stopper monitors the anchor loss and
restores the best checkpoint when patience is exhausted.

Key design choices:
  - Conditioning latents are kept clean (timestep=0) and passed via
    ``num_cond_latents`` so the DiT's attention treats them correctly.
  - Fixed noise draws and fixed sigma values remove stochastic variance
    from the anchor loss so the curve reflects model changes only.
  - Best checkpoint is stored in-memory via user-supplied ``save_fn`` /
    ``restore_fn`` pairs (deepcopy of trainable params).
"""

import argparse
import copy
import hashlib
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common import compute_flow_matching_loss_conditioned_fixed


# ============================================================================
# Public argument helpers
# ============================================================================

def add_early_stopping_args(parser: argparse.ArgumentParser):
    """Add early-stopping CLI arguments to an argparse parser."""
    g = parser.add_argument_group("Early stopping")
    g.add_argument("--es-disable", action="store_true", default=False,
                    help="Disable early stopping entirely.")
    g.add_argument("--es-check-every", type=int, default=5,
                    help="Evaluate anchor loss every N training steps.")
    g.add_argument("--es-patience", type=int, default=3,
                    help="Stop after this many checks without improvement.")
    g.add_argument("--es-anchor-sigmas", type=str, default="0.25,0.5,0.75",
                    help="Comma-separated sigma values for anchor loss.")
    g.add_argument("--es-noise-draws", type=int, default=2,
                    help="Number of noise draws per anchor sigma.")
    g.add_argument("--es-strategy", type=str, default="patience",
                    choices=["patience", "first_rise"],
                    help="Stopping strategy.")
    g.add_argument("--es-holdout-fraction", type=float, default=0.25,
                    help="Fraction of non-context conditioning frames held "
                         "out for anchor loss (used by split_tta_latents).")


def build_early_stopper_from_args(args) -> Optional["AnchoredEarlyStopper"]:
    """Build an AnchoredEarlyStopper from parsed CLI args, or None if disabled."""
    if getattr(args, "es_disable", False):
        return None
    anchor_sigmas = [float(x) for x in args.es_anchor_sigmas.split(",")]
    return AnchoredEarlyStopper(
        check_every=args.es_check_every,
        patience=args.es_patience,
        anchor_sigmas=anchor_sigmas,
        noise_draws=args.es_noise_draws,
        strategy=args.es_strategy,
    )


# ============================================================================
# AnchoredEarlyStopper
# ============================================================================

class AnchoredEarlyStopper:
    """Conditioning-aware early stopper for diffusion model TTA.

    Workflow per video
    ------------------
    1. ``setup(...)`` – receive pre-split latents, cache deterministic noise.
    2. In the training loop call ``step(current_step, save_fn)`` every
       ``check_every`` steps.  It returns ``(should_stop, info_dict)``.
    3. After the loop call ``restore(restore_fn)`` to load the best
       checkpoint.

    The anchor loss is evaluated by passing ``[cond_clean | val_noisy]``
    to the DiT with ``num_cond_latents = T_cond`` and per-token timesteps
    ``[0,...,0, sigma*1000,...,sigma*1000]``, exactly matching LongCat
    inference.
    """

    def __init__(
        self,
        check_every: int = 5,
        patience: int = 3,
        anchor_sigmas: Optional[List[float]] = None,
        noise_draws: int = 2,
        strategy: str = "patience",
    ):
        self.check_every = check_every
        self.patience = patience
        self.anchor_sigmas = anchor_sigmas or [0.25, 0.5, 0.75]
        self.noise_draws = noise_draws
        self.strategy = strategy

        # Internal state – reset in setup()
        self._reset()

    def _reset(self):
        self.model = None
        self.cond_latents = None
        self.val_latents = None
        self.prompt_embeds = None
        self.prompt_mask = None
        self.device = None
        self.dtype = None
        self.forward_fn = None

        # Fixed noise draws (generated once per video in setup)
        self.fixed_noises: List[torch.Tensor] = []

        self.best_loss = float("inf")
        self.best_state = None
        self.checks_without_improvement = 0
        self.step_count = 0
        self.stopped_early = False
        self.best_step = 0
        self.loss_history: List[Tuple[int, float]] = []

    # ------------------------------------------------------------------
    # setup – called once per video before the training loop
    # ------------------------------------------------------------------
    def setup(
        self,
        model: nn.Module,
        cond_latents: torch.Tensor,
        val_latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        forward_fn: Optional[Callable] = None,
        video_id: str = "",
        save_fn: Optional[Callable] = None,
    ):
        """Prepare the stopper for a new video.

        Parameters
        ----------
        model         : the trainable module (dit, wrapper, etc.)
        cond_latents  : clean context latents [B, C, T_cond, H, W]
        val_latents   : held-out latents [B, C, T_val, H, W]
        forward_fn    : optional custom forward;
                        signature ``(hidden_states, timestep, num_cond_latents) -> pred``
        video_id      : string for deterministic noise seeding
        save_fn       : callable returning in-memory state snapshot
        """
        self._reset()
        self.model = model
        self.cond_latents = cond_latents
        self.val_latents = val_latents
        self.prompt_embeds = prompt_embeds
        self.prompt_mask = prompt_mask
        self.device = device
        self.dtype = dtype
        self.forward_fn = forward_fn

        # Generate deterministic noise draws seeded from video_id
        seed_base = int(hashlib.md5(video_id.encode()).hexdigest()[:8], 16) % (2**31)
        self.fixed_noises = []
        for draw_idx in range(self.noise_draws):
            gen = torch.Generator(device=device)
            gen.manual_seed(seed_base + draw_idx)
            noise = torch.randn(
                val_latents.shape, generator=gen,
                device=device, dtype=val_latents.dtype,
            )
            self.fixed_noises.append(noise)

        # Snapshot initial state
        if save_fn is not None:
            self.best_state = save_fn()
        else:
            self.best_state = self._default_snapshot()

        # Evaluate initial anchor loss
        self.best_loss = self._compute_anchor_loss()
        self.loss_history.append((0, self.best_loss))

    # ------------------------------------------------------------------
    # step – called during training
    # ------------------------------------------------------------------
    def step(
        self,
        current_step: int,
        save_fn: Optional[Callable] = None,
    ) -> Tuple[bool, dict]:
        """Check whether to stop. Call every training step.

        Parameters
        ----------
        save_fn : callable returning in-memory state snapshot
                  (e.g. ``lambda: copy.deepcopy(wrapper.delta.data)``)

        Returns ``(should_stop, info_dict)``.
        """
        self.step_count = current_step

        if current_step == 0 or current_step % self.check_every != 0:
            return False, {}

        loss = self._compute_anchor_loss()
        self.loss_history.append((current_step, loss))

        improved = loss < self.best_loss
        if improved:
            self.best_loss = loss
            self.best_step = current_step
            if save_fn is not None:
                self.best_state = save_fn()
            else:
                self.best_state = self._default_snapshot()
            self.checks_without_improvement = 0
        else:
            self.checks_without_improvement += 1

        info = {
            "anchor_loss": loss,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "checks_without_improvement": self.checks_without_improvement,
        }

        should_stop = False
        if self.strategy == "patience":
            should_stop = self.checks_without_improvement >= self.patience
        elif self.strategy == "first_rise":
            should_stop = not improved and current_step > 0

        if should_stop:
            self.stopped_early = True

        return should_stop, info

    # ------------------------------------------------------------------
    # restore – called after the training loop
    # ------------------------------------------------------------------
    def restore(self, restore_fn: Optional[Callable] = None):
        """Restore the best checkpoint found during training.

        Parameters
        ----------
        restore_fn : callable that takes the saved state and applies it.
                     e.g. ``lambda s: wrapper.delta.data.copy_(s)``
                     If None, uses model.load_state_dict.
        """
        if self.best_state is None:
            return

        if restore_fn is not None:
            restore_fn(self.best_state)
        elif self.model is not None:
            self.model.load_state_dict(self.best_state, strict=False)

    # ------------------------------------------------------------------
    # state property
    # ------------------------------------------------------------------
    @property
    def state(self) -> Optional[dict]:
        """Return a summary dict for logging."""
        if not self.loss_history:
            return None
        return {
            "stopped_early": self.stopped_early,
            "best_step": self.best_step,
            "best_loss": self.best_loss,
            "total_checks": len(self.loss_history),
            "loss_history": self.loss_history,
        }

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _default_snapshot(self) -> dict:
        """Deep-copy the model's trainable parameters."""
        if self.model is None:
            return {}
        return {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
            if v.requires_grad or k in {
                n for n, p in self.model.named_parameters() if p.requires_grad
            }
        }

    def _compute_anchor_loss(self) -> float:
        """Evaluate conditioning-aware anchor loss on held-out val latents."""
        if self.val_latents is None or self.model is None:
            return float("inf")

        was_training = self.model.training
        self.model.eval()

        loss = compute_flow_matching_loss_conditioned_fixed(
            dit=self.model,
            cond_latents=self.cond_latents,
            target_latents=self.val_latents,
            prompt_embeds=self.prompt_embeds,
            prompt_mask=self.prompt_mask,
            fixed_sigmas=self.anchor_sigmas,
            fixed_noises=self.fixed_noises,
            device=self.device,
            dtype=self.dtype,
            forward_fn=self.forward_fn,
        )

        if was_training:
            self.model.train()

        return loss
