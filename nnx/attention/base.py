"""
Base class for all nnx attention modules.

Every concrete attention backend must implement `forward` with the
signature defined here so that higher-level blocks (TransformerLayer,
etc.) can swap backends transparently.
"""

from __future__ import annotations

import abc
from typing import Optional

import torch
import torch.nn as nn


class BaseAttention(nn.Module, abc.ABC):
    """
    Abstract base for all attention backends.

    Subclasses receive *pre-projected* Q, K, V tensors so the backend
    only needs to implement the actual attention computation.  Head
    splitting / merging is handled here.

    Args:
        embed_dim:   Total embedding dimensionality.
        num_heads:   Number of attention heads.
        dropout:     Dropout probability on attention weights (not
                     supported by all backends).
        bias:        Whether QKV and output projections have a bias.
        head_dim:    Dimensionality per head.  Defaults to
                     ``embed_dim // num_heads``.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        head_dim: Optional[int] = None, ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim or (embed_dim // num_heads)

        if self.head_dim * num_heads != embed_dim and head_dim is None:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}) "
                "when head_dim is not explicitly provided.",
            )

        inner_dim = self.head_dim * num_heads

        self.q_proj = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.out_proj = nn.Linear(inner_dim, embed_dim, bias=bias)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, D) → (B, H, T, Dh)"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, Dh) → (B, T, D)"""
        B, H, T, Dh = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * Dh)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def _attend(
        self,
        q: torch.Tensor,  # (B, H, Tq, Dh)
        k: torch.Tensor,  # (B, H, Tk, Dh)
        v: torch.Tensor,  # (B, H, Tk, Dh)
        attn_bias: Optional[torch.Tensor],  # broadcastable additive bias
        position_ids: Optional[torch.Tensor],  # (B, T) or (T,)
    ) -> torch.Tensor:  # (B, H, Tq, Dh)
        """Core attention computation — implemented by each backend."""

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        causal: bool = False, ) -> torch.Tensor:
        """
        Args:
            query:          (B, Tq, D) query tensor.
            key:            (B, Tk, D) key tensor.  Defaults to ``query``
                            (self-attention).
            value:          (B, Tk, D) value tensor.  Defaults to ``key``.
            attention_mask: HuggingFace-style mask — a tensor where
                            ``1``/``True`` marks *real* positions and
                            ``0``/``False`` marks *padding*.  Shape can be
                            (B, Tk) or any broadcastable additive bias.
            position_ids:   (B, T) or (T,) position indices. If None, positions
                            are assumed to be [0, 1, 2, ...]. Used for RoPE/ALiBi
                            to support chunked sequences.
            causal:         If True, apply a causal (autoregressive) mask
                            on top of ``attention_mask``.

        Returns:
            (B, Tq, D) attended output.
        """
        from nnx.utils.mask import hf_to_additive, make_causal_mask, combine_masks

        if key is None:
            key = query
        if value is None:
            value = key

        # Project
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        # Build additive bias
        bias_parts = []

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # HF-style (B, T) — convert
                bias_parts.append(hf_to_additive(attention_mask, dtype=q.dtype))
            else:
                # Already a pre-built additive bias
                bias_parts.append(attention_mask)

        if causal:
            bias_parts.append(make_causal_mask(q.shape[-2], device=q.device, dtype=q.dtype))

        attn_bias = combine_masks(*bias_parts)

        # Attend
        out = self._attend(q, k, v, attn_bias, position_ids)

        # Merge heads and project
        return self.out_proj(self._merge_heads(out))
