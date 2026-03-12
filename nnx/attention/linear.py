"""
Linear Attention backend.

Wraps the ``flash_linear_attention`` library (fla-org/flash-linear-attention)
which provides O(T) recurrent / chunkwise-parallel linear attention kernels.

Several variants are supported:
  - ``"gla"``   — Gated Linear Attention (Yang et al., 2024)
  - ``"delta"`` — DeltaNet (Schlag et al., 2021 / Yang et al., 2024)
  - ``"based"`` — Based (Arora et al., 2024)
  - ``"retention"`` — RetNet-style retention

Requires: pip install flash-linear-attention
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

_VARIANTS = Literal["gla", "delta", "based", "retention"]


def _import_fla(variant: str):
    """Lazily import the requested FLA kernel."""
    try:
        if variant == "gla":
            from fla.ops.gla import chunk_gla

            return chunk_gla
        elif variant == "delta":
            from fla.ops.delta_rule import chunk_delta_rule

            return chunk_delta_rule
        elif variant == "based":
            from fla.ops.based import parallel_based

            return parallel_based
        elif variant == "retention":
            from fla.ops.retention import chunk_retention

            return chunk_retention
        else:
            raise ValueError(f"Unknown flash-linear-attention variant: {variant!r}")
    except ImportError as exc:
        raise ImportError(
            "LinearAttention requires the flash-linear-attention library. "
            "Install it with:\n"
            "  pip install flash-linear-attention\n"
            "or visit https://github.com/fla-org/flash-linear-attention",
        ) from exc


class LinearAttention(nn.Module):
    """
    Linear (sub-quadratic) attention using the Flash Linear Attention library.

    Unlike the quadratic backends, linear attention does not compute a full
    attention matrix, so the semantics of ``attention_mask`` are different:
    padding tokens are zeroed out *before* the recurrence rather than
    masked in logit space.

    Args:
        embed_dim:  Total embedding dimensionality.
        num_heads:  Number of attention heads.
        variant:    Which linear-attention kernel to use.
                    One of ``"gla"``, ``"delta"``, ``"based"``,
                    ``"retention"``.
        head_dim:   Per-head dimensionality (default: embed_dim // num_heads).
        bias:       Whether projection layers include a bias term.
        expand_k:   Key expansion ratio (GLA / DeltaNet).
        expand_v:   Value expansion ratio (GLA / DeltaNet).
        chunk_size: Chunk size for chunkwise-parallel kernels.

    Note:
        Because linear attention is inherently *causal* (it uses a recurrence), passing ``causal=False`` has no effect — the
        recurrence always processes left-to-right.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        variant: _VARIANTS = "gla",
        head_dim: Optional[int] = None,
        bias: bool = False,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        chunk_size: int = 64, ) -> None:

        assert torch.cuda.is_available(), "CUDA is not available"

        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.variant = variant
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.chunk_size = chunk_size

        k_dim = int(self.head_dim * expand_k)
        v_dim = int(self.head_dim * expand_v)
        inner_q = self.head_dim * num_heads
        inner_k = k_dim * num_heads
        inner_v = v_dim * num_heads

        self.q_proj = nn.Linear(embed_dim, inner_q, bias=bias)
        self.k_proj = nn.Linear(embed_dim, inner_k, bias=bias)
        self.v_proj = nn.Linear(embed_dim, inner_v, bias=bias)
        self.out_proj = nn.Linear(inner_v, embed_dim, bias=bias)

        # GLA / DeltaNet have a gate
        self._has_gate = variant in ("gla", "delta")
        if self._has_gate:
            self.g_proj = nn.Linear(embed_dim, inner_v, bias=bias)

        self._kernel = _import_fla(variant)
        self._k_dim = k_dim
        self._v_dim = v_dim

        if variant == "delta":
            self.to(dtype=torch.bfloat16)

    def _split(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True, ) -> torch.Tensor:
        """
        Args:
            query:          (B, T, D)
            key:            (B, T, D) — defaults to query (self-attention).
            value:          (B, T, D) — defaults to key.
            attention_mask: HF-style (B, T) mask.  Padding positions are
                            zeroed in q/k/v before the recurrence.
            causal:         Ignored — linear attention is always causal.

        Returns:
            (B, T, D)
        """
        if key is None:
            key = query
        if value is None:
            value = key

        q = self._split(self.q_proj(query), self.head_dim)
        k = self._split(self.k_proj(key), self._k_dim)
        v = self._split(self.v_proj(value), self._v_dim)

        # Apply HF-style mask by zeroing padding positions
        if attention_mask is not None:
            pad = (~attention_mask.bool()).unsqueeze(-1)  # (B, T, 1)
            q = q.masked_fill(pad.unsqueeze(1), 0.0)
            k = k.masked_fill(pad.unsqueeze(1), 0.0)
            v = v.masked_fill(pad.unsqueeze(1), 0.0)

        if self.variant == "gla":
            # GLA expects a gate in (B, H, T, V)
            g = self._split(torch.sigmoid(self.g_proj(query)), self._v_dim)
            out, _ = self._kernel(q, k, v, g)
        elif self.variant == "delta":
            # DeltaNet expects beta (gate) as the 4th positional argument
            beta = torch.sigmoid(self.g_proj(query))
            out, _ = self._kernel(q, k, v, beta)
        elif self.variant == "based":
            out = self._kernel(q, k, v)
        elif self.variant == "retention":
            out, _ = self._kernel(q, k, v)
        else:
            raise ValueError(self.variant)

        # (B, H, T, Dv) → (B, T, H*Dv)
        B, H, T, Dv = out.shape
        out = out.transpose(1, 2).contiguous().view(B, T, H * Dv)
        return self.out_proj(out)
