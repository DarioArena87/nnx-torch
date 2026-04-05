"""
Base class for all nnx attention modules.

Every concrete attention backend must implement `forward` with the
signature defined here so that higher-level blocks (TransformerLayer,
etc.) can swap backends transparently.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange


@dataclass
class AttentionOutput:
    """
    Output container for attention with optional KV cache support.

    HuggingFace-style interface for attention outputs.

    Args:
        hidden_states: (B, Tq, D) attended output.
        attention_weights: Optional attention weights tensor of shape
            (B, H, Tq, Tk). Only populated when the backend supports it.
        past_key_value: Optional tuple of (key_cache, value_cache) for
            autoregressive generation. Only populated when ``use_cache=True``.
    """

    hidden_states: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None
    past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None


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
        num_key_value_heads: Number of KV heads for Grouped Query Attention.
            When ``None`` or equal to ``num_heads``, uses standard MHA.
            When less than ``num_heads``, enables GQA.
        use_cache:   If True, return ``past_key_value`` in the output
            for autoregressive generation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        head_dim: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        use_cache: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = head_dim or (embed_dim // num_heads)
        self.num_key_value_heads = num_key_value_heads or num_heads
        self.use_cache = use_cache

        if self.head_dim * num_heads != embed_dim and head_dim is None:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads}) "
                "when head_dim is not explicitly provided.",
            )

        inner_dim = self.head_dim * num_heads
        kv_inner_dim = self.head_dim * self.num_key_value_heads

        # B1: GQA support — K/V project to smaller dimension when num_kv_heads < num_heads
        # B2: Hybrid QKV — separate Q projection, packed KV projection
        # This supports cross attention (key != query) while maintaining
        # most of the optimization benefit (2 kernel launches instead of 3)
        self.q_proj = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.kv_proj = nn.Linear(embed_dim, 2 * kv_inner_dim, bias=bias)

        self.out_proj = nn.Linear(inner_dim, embed_dim, bias=bias)

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------

    def _split_heads(self, x: torch.Tensor, num_heads: Optional[int] = None) -> torch.Tensor:
        """(B, T, D) → (B, H, T, Dh)"""
        h = num_heads or self.num_heads
        return rearrange(x, '... t (h d) -> ... h t d', h=h)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, Dh) → (B, T, D)"""
        return rearrange(x, '... h t d -> ... t (h d)')

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Repeat KV heads for GQA support.

        When ``num_key_value_heads < num_heads``, each KV head is repeated
        ``num_heads // num_key_value_heads`` times to match the query heads.
        """
        if self.num_key_value_heads == self.num_heads:
            return x
        n_rep = self.num_heads // self.num_key_value_heads
        return x.repeat_interleave(n_rep, dim=1)

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
        is_causal: bool = False,  # S1: native causal flag
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
        causal: bool = False,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: Optional[bool] = None,
    ) -> AttentionOutput | torch.Tensor:
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
            past_key_value: Optional tuple of (key_cache, value_cache) from
                            a previous forward pass for autoregressive generation.
                            Shape: ((B, H, T_prev, Dh), (B, H, T_prev, Dh)).
            use_cache:      Override the module-level ``use_cache`` setting.
                            If True, return ``past_key_value`` in the output.

        Returns:
            If ``use_cache=True``: ``AttentionOutput`` with ``hidden_states``
            and ``past_key_value``.
            If ``use_cache=False``: ``torch.Tensor`` of shape (B, Tq, D).
        """
        from nnx.utils.mask import hf_to_additive, make_causal_mask, combine_masks

        effective_use_cache = use_cache if use_cache is not None else self.use_cache

        if key is None:
            key = query
        if value is None:
            value = key

        # B2: Hybrid QKV projections — Q from query, KV from key
        # This correctly supports cross attention where key != query
        q = self.q_proj(query)
        kv = self.kv_proj(key)
        kv_dim = self.head_dim * self.num_key_value_heads
        k, v = kv.split([kv_dim, kv_dim], dim=-1)

        # Split heads
        q = self._split_heads(q, self.num_heads)
        k = self._split_heads(k, self.num_key_value_heads)
        v = self._split_heads(v, self.num_key_value_heads)

        # B5: KV Cache support — concatenate with past key/value
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)

        # Build additive bias
        bias_parts = []

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # HF-style (B, T) — convert
                bias_parts.append(hf_to_additive(attention_mask, dtype=q.dtype))
            else:
                # Already a pre-built additive bias
                bias_parts.append(attention_mask)

        # S1: Causal dispatch — only build mask when attn_bias exists
        # When causal=True and no other bias, let SDPA handle it natively
        needs_causal_mask = causal and (attention_mask is not None)

        if needs_causal_mask:
            bias_parts.append(make_causal_mask(q.shape[-2], device=q.device, dtype=q.dtype))

        attn_bias = combine_masks(*bias_parts)

        # B1: GQA — repeat KV heads to match query heads
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Attend — pass is_causal for S1 optimization
        out = self._attend(q, k, v, attn_bias, position_ids, is_causal=causal)

        # Merge heads and project
        output = self.out_proj(self._merge_heads(out))

        # B5: Return past_key_value if caching is enabled
        if effective_use_cache:
            present_key_value = (k, v)
            return AttentionOutput(hidden_states=output, past_key_value=present_key_value)

        return output
