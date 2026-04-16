"""
TinyLoRA: Ultra-low-rank adaptation for LongCat-Video DiT.

Implements the TinyLoRA method from "Learning to Reason in 13 Parameters"
(Morris et al., 2026) adapted for video diffusion transformer test-time
adaptation.

TinyLoRA uses SVD-based decomposition to create extremely parameter-efficient
adapters:
  1. Compute SVD of each target weight matrix: W = U S V^T
  2. Keep top-r singular vectors frozen: U_r [d_out, r], V_r [d_in, r]
  3. Train a tiny vector v in R^r per module (or shared via weight tying)
  4. Weight update: dW = (alpha/r) * U_r @ diag(v) @ V_r^T

The efficient forward avoids materializing dW:
  y = W @ x + (alpha/r) * U_r @ (v * (V_r^T @ x))

Reference: https://arxiv.org/abs/2602.04118
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Target module presets for LongCatSingleStreamBlock
#
# Per block, the Linear modules are:
#   attn.qkv             [hidden, 3*hidden]   self-attention QKV
#   attn.proj             [hidden, hidden]     self-attention output
#   cross_attn.q_linear   [hidden, hidden]     cross-attention query
#   cross_attn.kv_linear  [hidden, 2*hidden]   cross-attention KV
#   cross_attn.proj       [hidden, hidden]     cross-attention output
#   ffn.w1                [hidden, ffn_dim]    SwiGLU gate
#   ffn.w2                [ffn_dim, hidden]    SwiGLU down-proj
#   ffn.w3                [hidden, ffn_dim]    SwiGLU up-proj
# ============================================================================

DEFAULT_TARGETS = ["attn.qkv", "attn.proj"]

ALL_ATTENTION_TARGETS = [
    "attn.qkv",
    "attn.proj",
    "cross_attn.q_linear",
    "cross_attn.kv_linear",
    "cross_attn.proj",
]

ALL_TARGETS = ALL_ATTENTION_TARGETS + ["ffn.w1", "ffn.w2", "ffn.w3"]

TARGET_PRESETS = {
    "qkv_proj": ["attn.qkv", "attn.proj"],
    "self_attn": ["attn.qkv", "attn.proj"],
    "all_attn": ALL_ATTENTION_TARGETS,
    "all": ALL_TARGETS,
}


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TinyLoRAConfig:
    """Configuration for TinyLoRA adaptation.

    Args:
        svd_rank: Frozen SVD rank. Paper recommends r=2 based on ablations.
        alpha: Scaling factor for the adapter output (scaling = alpha / r).
        n_tie: Weight tying factor — groups of n_tie consecutive adapted
               layers share a single trainable v vector.
        target_modules: Dotted attribute paths to nn.Linear layers within
                        each LongCatSingleStreamBlock.
    """

    svd_rank: int = 2
    alpha: float = 1.0
    n_tie: int = 1
    target_modules: List[str] = field(
        default_factory=lambda: list(DEFAULT_TARGETS)
    )


# ============================================================================
# Core TinyLoRA layer
# ============================================================================


class TinyLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with a TinyLoRA adapter.

    Forward computation:
        y = W @ x + bias + (alpha/r) * U_r @ diag(v) @ V_r^T @ x

    Only v (R^r) is trainable. U_r and V_r^T are frozen singular vectors
    obtained from the SVD of the original weight matrix.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        svd_rank: int = 2,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.svd_rank = svd_rank
        self.scaling = alpha / svd_rank

        for p in self.original_layer.parameters():
            p.requires_grad = False

        dtype = original_layer.weight.dtype
        W = original_layer.weight.data.float()

        U, S, V = torch.svd_lowrank(W, q=svd_rank, niter=4)
        # U: [d_out, r]  S: [r]  V: [d_in, r]
        self.register_buffer("frozen_U", U.to(dtype))
        self.register_buffer("frozen_Vt", V.t().contiguous().to(dtype))  # [r, d_in]

        device = original_layer.weight.device
        self.v = nn.Parameter(torch.zeros(svd_rank, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.original_layer(x)
        # V_r^T @ x  →  x @ V_r  →  F.linear(x, Vt) where Vt is [r, d_in]
        h = F.linear(x, self.frozen_Vt)  # [..., d_in] → [..., r]
        h = h * self.v  # element-wise scale by trainable v
        # U_r @ h  →  h @ U_r^T  →  F.linear(h, U) where U is [d_out, r]
        lora_out = F.linear(h, self.frozen_U)  # [..., r] → [..., d_out]
        return result + lora_out * self.scaling

    @property
    def weight(self):
        return self.original_layer.weight

    @property
    def bias(self):
        return self.original_layer.bias

    def extra_repr(self) -> str:
        return (
            f"in={self.original_layer.in_features}, "
            f"out={self.original_layer.out_features}, "
            f"svd_rank={self.svd_rank}, scaling={self.scaling:.4f}"
        )


# ============================================================================
# Module-tree navigation helpers
# ============================================================================


def _resolve_submodule(module: nn.Module, path: str) -> nn.Module:
    """Navigate a dotted path (e.g. 'attn.qkv') to reach a submodule."""
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def _set_submodule(module: nn.Module, path: str, value: nn.Module):
    """Set the submodule at a dotted path."""
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = value
    else:
        setattr(parent, last, value)


# ============================================================================
# Injection / removal
# ============================================================================

# Each entry: (TinyLoRALinear instance, parent block, dotted path within block)
InjectedInfo = Tuple[TinyLoRALinear, nn.Module, str]


def inject_tinylora_into_dit(
    dit: nn.Module,
    config: TinyLoRAConfig,
) -> List[InjectedInfo]:
    """Inject TinyLoRA adapters into all target linear layers of a LongCat DiT.

    Modifies ``dit.blocks`` in-place by replacing target ``nn.Linear``
    modules with ``TinyLoRALinear`` wrappers.  SVD is computed once per
    layer during injection.

    Returns a list of (tinylora_layer, parent_block, path) tuples that
    can be used for restoration via ``remove_tinylora_from_dit``.
    """
    injected: List[InjectedInfo] = []

    for block in dit.blocks:
        for target_path in config.target_modules:
            try:
                original = _resolve_submodule(block, target_path)
            except (AttributeError, IndexError, TypeError):
                continue

            if not isinstance(original, nn.Linear):
                continue

            lora_layer = TinyLoRALinear(
                original, svd_rank=config.svd_rank, alpha=config.alpha
            )
            _set_submodule(block, target_path, lora_layer)
            injected.append((lora_layer, block, target_path))

    return injected


def remove_tinylora_from_dit(injected: List[InjectedInfo]):
    """Undo ``inject_tinylora_into_dit`` — restore original nn.Linear layers."""
    for lora_layer, block, target_path in injected:
        _set_submodule(block, target_path, lora_layer.original_layer)


def apply_weight_tying(
    injected: List[InjectedInfo],
    n_tie: int,
    svd_rank: int,
    dtype: torch.dtype = torch.bfloat16,
):
    """Share a single trainable v vector across groups of n_tie modules.

    Consecutive adapted layers are grouped.  Within each group, all
    ``TinyLoRALinear.v`` attributes point to the same ``nn.Parameter``.
    """
    if n_tie <= 1:
        return

    modules = [entry[0] for entry in injected]
    device = modules[0].v.device if modules else "cpu"
    for group_start in range(0, len(modules), n_tie):
        group = modules[group_start : group_start + n_tie]
        shared_v = nn.Parameter(torch.zeros(svd_rank, dtype=dtype, device=device))
        for m in group:
            m.v = shared_v


# ============================================================================
# Wrapper
# ============================================================================


class TinyLoRAWrapper(nn.Module):
    """Wraps a LongCat DiT with TinyLoRA adapters for test-time adaptation.

    The wrapper injects ``TinyLoRALinear`` layers into the DiT's transformer
    blocks *once* (computing SVDs at init time).  For per-video TTA, call
    ``reset_v()`` before each video to zero the trainable parameters.

    Typical usage::

        wrapper = TinyLoRAWrapper(dit, config)
        for video in videos:
            wrapper.reset_v()
            optimizer = AdamW(wrapper.get_trainable_params(), lr=1e-3)
            for step in range(num_steps):
                loss = compute_flow_matching_loss(wrapper, ...)
                loss.backward(); optimizer.step()
        wrapper.remove()   # restores the original DiT
    """

    def __init__(self, dit: nn.Module, config: TinyLoRAConfig):
        super().__init__()
        self.dit = dit
        self._config = config

        for p in self.dit.parameters():
            p.requires_grad = False

        self._injected = inject_tinylora_into_dit(dit, config)

        if not self._injected:
            raise ValueError(
                f"No layers were injected. Check target_modules="
                f"{config.target_modules} against the DiT block structure."
            )

        if config.n_tie > 1:
            dtype = self._injected[0][0].frozen_U.dtype
            apply_weight_tying(
                self._injected, config.n_tie, config.svd_rank, dtype
            )

    # -- Forward (delegates to the modified DiT) --

    @property
    def config(self):
        """Proxy ``dit.config`` so callers (e.g. flow-matching loss) can
        access ``patch_size``, ``adaln_tembed_dim``, etc."""
        return self.dit.config

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_attention_mask=None,
        num_cond_latents=0,
        **kwargs,
    ):
        return self.dit(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            num_cond_latents=num_cond_latents,
            **kwargs,
        )

    # -- Parameter management --

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return a deduplicated list of trainable v parameters."""
        seen: set = set()
        params: List[nn.Parameter] = []
        for lora_mod, _, _ in self._injected:
            pid = id(lora_mod.v)
            if pid not in seen:
                seen.add(pid)
                params.append(lora_mod.v)
        return params

    def reset_v(self):
        """Zero all trainable v vectors (call before each new video)."""
        for p in self.get_trainable_params():
            p.data.zero_()

    def remove(self):
        """Restore the original ``nn.Linear`` layers in the DiT."""
        remove_tinylora_from_dit(self._injected)
        self._injected.clear()

    def param_summary(self) -> Dict[str, int]:
        """Return a dict summarising parameter counts."""
        params = self.get_trainable_params()
        return {
            "total_model_params": sum(p.numel() for p in self.dit.parameters()),
            "tinylora_trainable": sum(p.numel() for p in params),
            "num_adapted_layers": len(self._injected),
            "num_v_vectors": len(params),
            "svd_rank": self._config.svd_rank,
            "n_tie": self._config.n_tie,
        }
