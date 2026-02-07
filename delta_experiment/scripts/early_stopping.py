#!/usr/bin/env python3
"""
Validation-free early stopping for diffusion model TTA.

Uses an "anchor loss" computed at fixed timesteps and noise seeds on a
held-out fraction of conditioning frames. The stopper monitors the anchor
loss and restores the best checkpoint when patience is exhausted.
"""

import argparse
import copy
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from common import compute_flow_matching_loss_fixed


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
    g.add_argument("--es-anchor-timesteps", type=str, default="0.25,0.5,0.75",
                    help="Comma-separated sigma values for anchor loss.")
    g.add_argument("--es-noise-draws", type=int, default=2,
                    help="Number of noise draws per anchor sigma.")
    g.add_argument("--es-strategy", type=str, default="patience",
                    choices=["patience", "first_rise"],
                    help="Stopping strategy.")
    g.add_argument("--es-holdout-fraction", type=float, default=0.25,
                    help="Fraction of conditioning frames held out for anchor loss.")


def build_early_stopper_from_args(args) -> Optional["AnchoredEarlyStopper"]:
    """Build an AnchoredEarlyStopper from parsed CLI args, or None if disabled."""
    if getattr(args, "es_disable", False):
        return None
    anchor_timesteps = [float(x) for x in args.es_anchor_timesteps.split(",")]
    return AnchoredEarlyStopper(
        check_every=args.es_check_every,
        patience=args.es_patience,
        anchor_timesteps=anchor_timesteps,
        noise_draws=args.es_noise_draws,
        strategy=args.es_strategy,
        holdout_fraction=args.es_holdout_fraction,
    )


# ============================================================================
# AnchoredEarlyStopper
# ============================================================================

class AnchoredEarlyStopper:
    """Validation-free early stopper for diffusion model TTA.

    Workflow per video:
        1. ``setup(model, latents, ...)`` – split conditioning latents into
           train / holdout, snapshot initial weights.
        2. In the training loop, call ``step(current_step)`` every
           ``check_every`` steps.  It returns ``(should_stop, info_dict)``.
        3. After the loop, call ``restore()`` to load the best checkpoint.
    """

    def __init__(
        self,
        check_every: int = 5,
        patience: int = 3,
        anchor_timesteps: Optional[List[float]] = None,
        noise_draws: int = 2,
        strategy: str = "patience",
        holdout_fraction: float = 0.25,
    ):
        self.check_every = check_every
        self.patience = patience
        self.anchor_timesteps = anchor_timesteps or [0.25, 0.5, 0.75]
        self.noise_draws = noise_draws
        self.strategy = strategy
        self.holdout_fraction = holdout_fraction

        # Internal state – reset in setup()
        self._reset()

    def _reset(self):
        self.model = None
        self.holdout_latents = None
        self.train_latents = None
        self.prompt_embeds = None
        self.prompt_mask = None
        self.device = None
        self.dtype = None
        self.patch_size = None

        self.best_loss = float("inf")
        self.best_state = None
        self.checks_without_improvement = 0
        self.step_count = 0
        self.stopped_early = False
        self.best_step = 0
        self.loss_history: List[Tuple[int, float]] = []
        self.forward_fn = None

    # ------------------------------------------------------------------
    # setup – called once per video before the training loop
    # ------------------------------------------------------------------
    def setup(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_mask: torch.Tensor,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        patch_size: Optional[tuple] = None,
        forward_fn: Optional[Callable] = None,
    ):
        """Prepare the stopper for a new video.

        Parameters
        ----------
        model : the trainable module (dit, delta params, etc.)
        latents : full conditioning latents [B, C, T, H, W]
        prompt_embeds : text embeddings
        prompt_mask : text attention mask
        forward_fn : optional custom forward function for computing anchor loss
        """
        self._reset()
        self.model = model
        self.device = device
        self.dtype = dtype
        self.patch_size = patch_size
        self.forward_fn = forward_fn

        # Split latents into train / holdout along temporal axis
        T = latents.shape[2]
        n_holdout = max(1, int(T * self.holdout_fraction))
        n_train = T - n_holdout

        self.train_latents = latents[:, :, :n_train].contiguous()
        self.holdout_latents = latents[:, :, n_train:].contiguous()
        self.prompt_embeds = prompt_embeds
        self.prompt_mask = prompt_mask

        # Snapshot initial state
        self.best_state = self._snapshot_state()
        self.best_loss = self._compute_anchor_loss()
        self.loss_history.append((0, self.best_loss))

    # ------------------------------------------------------------------
    # step – called during training
    # ------------------------------------------------------------------
    def step(self, current_step: int) -> Tuple[bool, dict]:
        """Check whether to stop. Call every training step.

        Returns (should_stop, info_dict).
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
            self.best_state = self._snapshot_state()
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
        restore_fn : optional callable that takes the saved state dict
                     and restores it. If None, uses model.load_state_dict.
        """
        if self.best_state is None:
            return

        if restore_fn is not None:
            restore_fn(self.best_state)
        elif self.model is not None:
            self.model.load_state_dict(self.best_state)

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
    def _snapshot_state(self) -> dict:
        """Deep-copy the model's trainable state dict."""
        if self.model is None:
            return {}
        return {
            k: v.detach().clone()
            for k, v in self.model.state_dict().items()
            if v.requires_grad or k in [
                n for n, p in self.model.named_parameters() if p.requires_grad
            ]
        }

    def _compute_anchor_loss(self) -> float:
        """Evaluate anchor loss on the held-out latents."""
        if self.holdout_latents is None or self.model is None:
            return float("inf")

        was_training = self.model.training
        self.model.eval()

        loss = compute_flow_matching_loss_fixed(
            dit=self.model,
            latents=self.holdout_latents,
            prompt_embeds=self.prompt_embeds,
            prompt_mask=self.prompt_mask,
            fixed_sigmas=self.anchor_timesteps,
            noise_draws=self.noise_draws,
            device=self.device,
            dtype=self.dtype,
            forward_fn=self.forward_fn,
        )

        if was_training:
            self.model.train()

        return loss
