"""
Feed-Forward Network (FFN) variants.

Modern large models have moved beyond the original Transformer FFN
(two linear layers + ReLU).  This module provides:

  - :class:`FFN`         — Plain two-layer FFN with any activation.
  - :class:`GatedFFN`    — GLU-family: SwiGLU (LLaMA), GeGLU, ReGLU, …
  - :class:`MoEFFN`      — Sparse Mixture-of-Experts FFN (Top-K routing).

All classes accept ``embed_dim`` and produce ``(B, T, embed_dim)`` outputs,
making them drop-in replacements for each other.
"""

from __future__ import annotations

from typing import Callable, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Activation aliases
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, Callable] = {
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "swish": F.silu,
    "mish": F.mish,
}


def _get_activation(name: str) -> Callable:
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}. Available: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]


# ---------------------------------------------------------------------------
# Plain FFN
# ---------------------------------------------------------------------------


class FFN(nn.Module):
    """
    Standard two-layer feed-forward network.

    ``FFN(x) = W2 · act(W1 · x + b1) + b2``

    Args:
        embed_dim:   Input / output dimensionality.
        ffn_dim:     Hidden layer width.  Defaults to ``4 * embed_dim``.
        activation:  Activation name — one of ``"relu"``, ``"gelu"``,
                     ``"silu"`` / ``"swish"``, ``"mish"``.
        dropout:     Dropout after the activation.
        bias:        Whether linear layers have bias terms.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        ffn_dim = ffn_dim or embed_dim * 4
        self.w1 = nn.Linear(embed_dim, ffn_dim, bias=bias)
        self.w2 = nn.Linear(ffn_dim, embed_dim, bias=bias)
        self.act = _get_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(self.act(self.w1(x))))


# ---------------------------------------------------------------------------
# Gated FFN (SwiGLU / GeGLU / …)
# ---------------------------------------------------------------------------


class GatedFFN(nn.Module):
    """
    Gated Linear Unit FFN variants: SwiGLU, GeGLU, ReGLU, …

    ``GatedFFN(x) = W2 · (act(W1 · x) ⊙ V · x)``

    where ⊙ is element-wise multiplication.  The gate branch (V·x) acts as
    a soft feature selector.

    Choosing ``activation``:
      - ``"silu"``  → **SwiGLU** (LLaMA, Mistral, Qwen, Phi, …)
      - ``"gelu"``  → **GeGLU** (PaLM, T5 v1.1, …)
      - ``"relu"``  → **ReGLU**

    Args:
        embed_dim:   Input / output dimensionality.
        ffn_dim:     Hidden dimensionality of *each* of the two sub-projections.
                     Defaults to ``int(embed_dim * 8 / 3)``, which keeps the
                     parameter count close to a plain 4× FFN.
        activation:  Activation applied to the gate path.
        dropout:     Dropout between the gating and the output projection.
        bias:        Whether projections have bias terms.
        packed_gates: If True, use a single ``nn.Linear(embed_dim, 2 * ffn_dim)``
                     for gate + up projections, then ``chunk(2, dim=-1)``.
                     Reduces kernel launch overhead and improves memory locality.
                     Defaults to False for backward compatibility.
        gradient_checkpointing: If True, wrap the forward pass with
                     ``torch.utils.checkpoint.checkpoint`` to reduce activation
                     memory by ~50-70% at the cost of ~20-30% extra compute.
                     Defaults to False for backward compatibility.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = False,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        # Default keeps total params ≈ 4× FFN
        ffn_dim = ffn_dim or int(embed_dim * 8 / 3)
        # Round to nearest multiple of 64 for tensor-core efficiency
        ffn_dim = (ffn_dim + 63) // 64 * 64

        self.gradient_checkpointing = gradient_checkpointing

        # Packed gates: Single linear for gate + up projections
        self.w13 = nn.Linear(embed_dim, 2 * ffn_dim, bias=bias)
        self.w2 = nn.Linear(ffn_dim, embed_dim, bias=bias)  # output
        self.act = _get_activation(activation)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward implementation for checkpointing support."""
        x13 = self.w13(x)
        x1, x3 = torch.chunk(x13, 2, dim=-1)
        return self.w2(self.drop(self.act(x1) * x3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        return self._forward_impl(x)


# ---------------------------------------------------------------------------
# Mixture-of-Experts FFN
# ---------------------------------------------------------------------------


class MoEFFN(nn.Module):
    """
    Sparse Top-K Mixture-of-Experts FFN.

    Each token is routed to the ``top_k`` highest-scoring experts.
    The final output is the weighted average of the selected experts'
    outputs.  Uses the auxiliary load-balancing loss of Switch Transformer
    (Fedus et al., 2021).

    Args:
        embed_dim:    Input / output dimensionality.
        ffn_dim:      Hidden width of each expert.
        num_experts:  Total number of expert FFNs.
        top_k:        Number of experts each token activates.
        expert_cls:   Expert class to use (:class:`FFN` or :class:`GatedFFN`).
        expert_kwargs: Extra kwargs forwarded to each expert's ``__init__``.
        router_bias:  Whether the router linear layer has a bias.

    Attributes:
        aux_loss:     Auxiliary load-balancing loss (scalar tensor).
                      Add ``λ * moe_layer.aux_loss`` to your training loss.
    """

    def __init__(
        self,
        embed_dim: int,
        ffn_dim: Optional[int] = None,
        num_experts: int = 8,
        top_k: int = 2,
        expert_cls=None,
        expert_kwargs: Optional[dict] = None,
        router_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        if expert_cls is None:
            expert_cls = GatedFFN
        expert_kwargs = expert_kwargs or {}

        self.experts = nn.ModuleList([expert_cls(embed_dim, ffn_dim, **expert_kwargs) for _ in range(num_experts)])
        self.router = nn.Linear(embed_dim, num_experts, bias=router_bias)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        x_flat = x.view(-1, D)  # (B*T, D)
        N = x_flat.shape[0]

        # Router
        logits = self.router(x_flat)  # (N, E)
        probs = F.softmax(logits, dim=-1)  # (N, E)
        top_probs, top_idx = probs.topk(self.top_k, dim=-1)  # (N, k)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)  # renorm

        # Auxiliary load-balancing loss (Switch Transformer style)
        # loss = E · mean(f_i) · mean(p_i)  where f_i = fraction of tokens to expert i
        with torch.no_grad():
            # One-hot top-1 for load counting
            top1 = top_idx[:, 0]
            f = torch.zeros(self.num_experts, device=x.device)
            f.scatter_add_(0, top1, torch.ones(N, device=x.device))
            f = f / N
        p = probs.mean(0)  # (E,)
        self.aux_loss = self.num_experts * (f * p).sum()

        # Dispatch
        out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = top_idx[:, k]  # (N,)
            weights = top_probs[:, k]  # (N,)

            for e in range(self.num_experts):
                token_mask = expert_indices == e  # (N,)
                if not token_mask.any():
                    continue
                tokens = x_flat[token_mask]  # (n_e, D)
                expert_out = self.experts[e](tokens)  # (n_e, D)
                out[token_mask] += weights[token_mask, None] * expert_out

        return out.view(B, T, D)
