"""
Attention with Linear Biases (ALiBi).

This attention backend adds a fixed, head-specific linear bias to the
attention logits, encouraging the model to attend more to nearby tokens.
Unlike absolute positional embeddings, ALiBi does not increase token
representation dimensionality and generalizes well to longer sequences.

The implementation uses :class:`nnx.layers.embedding.ALiBiEmbedding` to
generate the bias matrix.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F

from .base import BaseAttention
from ..layers.embedding import ALiBiEmbedding


class ALiBiAttention(BaseAttention):
    """
    Multi-head attention with ALiBi (Attention with Linear Biases).

    ALiBi adds a pre-computed, head-specific linear bias to the attention
    logits.  The bias is a function of the distance between query and key
    positions: each head has a slope, and the bias is ``-slope * |i - j|``.
    This encourages attention to nearby tokens without learnable position
    embeddings.

    Args:
        embed_dim:  Total embedding dimensionality.
        num_heads:  Number of attention heads.
        dropout:    Dropout probability on attention weights.
        bias:       Whether QKV and output projections have a bias.
        max_len:    Maximum sequence length for bias matrix pre-computation.
        head_dim:   Per-head dimensionality (default: embed_dim // num_heads).
        use_flex_attention: If True and PyTorch 2.5+ is available, use
                     ``torch.nn.attention.flex_attention`` with a ``score_mod``
                     function that applies ALiBi bias. This enables
                     FlashAttention-compatible ALiBi. Falls back to the
                     standard implementation when FlexAttention is not available.
                     Defaults to False for backward compatibility.

    Note:
        The number of heads determines the set of slopes used.  If
        ``num_heads`` is not a power of 2, the slopes are computed by
        interpolation as described in the ALiBi paper.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        max_len: int = 4096,
        head_dim: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        use_cache: bool = False,
        use_flex_attention: bool = False,
    ) -> None:
        super().__init__(
            embed_dim, num_heads, dropout, bias, head_dim,
            num_key_value_heads=num_key_value_heads,
            use_cache=use_cache,
        )
        self.alibi = ALiBiEmbedding(num_heads, max_len)
        self.use_flex_attention = use_flex_attention
        self._flex_attention: Optional[Callable] = None
        self._create_block_mask: Optional[Callable] = None

        if use_flex_attention:
            self._init_flex_attention()

    def _init_flex_attention(self) -> None:
        """Initialize FlexAttention components if available."""
        try:
            from torch.nn.attention.flex_attention import (
                flex_attention,
                create_block_mask,
            )
            self._flex_attention = flex_attention
            self._create_block_mask = create_block_mask
        except ImportError:
            # FlexAttention not available, fall back to standard implementation
            self.use_flex_attention = False
            self._flex_attention = None
            self._create_block_mask = None

    @staticmethod
    def _get_alibi_slopes(num_heads: int) -> torch.Tensor:
        """Compute per-head slopes as in the ALiBi paper."""
        n = 2 ** (-(2 ** -(math.log2(num_heads) - 3)))
        return torch.tensor([n ** (i + 1) for i in range(num_heads)])

    def _alibi_score_mod(
        self,
        score: torch.Tensor,
        b: int,
        h: int,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        """ALiBi score_mod function for FlexAttention."""
        slopes = self._get_alibi_slopes(self.num_heads).to(score.device)
        slope = slopes[h]
        return score + slope * (q_idx - kv_idx)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        # Use FlexAttention if enabled and available
        if self.use_flex_attention and self._flex_attention is not None:
            return self._attend_flex(
                q, k, v, attn_bias, position_ids, is_causal
            )

        # Standard implementation with materialized bias
        # Sequence lengths
        Tq = q.shape[2]
        Tk = k.shape[2]

        if position_ids is None:
            # Backward compatible: use sequential positions
            T = max(Tq, Tk)
            alibi_bias = self.alibi(T, k.device)
            alibi_bias = alibi_bias[:, :, :Tq, :Tk]
        else:
            # Use explicit position_ids
            alibi_bias = self.alibi.with_positions(position_ids, k.device)
            # Slice to match Q-K dimensions if needed
            alibi_bias = alibi_bias[:, :, :Tq, :Tk]

        # Combine with any provided attention bias
        if attn_bias is not None:
            combined_bias = attn_bias + alibi_bias
        else:
            combined_bias = alibi_bias

        # Scaled dot-product attention
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=combined_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )

    def _attend_flex(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        """FlexAttention-based ALiBi attention."""
        assert self._flex_attention is not None, "FlexAttention not initialized"

        # Create score_mod that combines ALiBi and optional additive bias
        def combined_score_mod(
            score: torch.Tensor,
            b: int,
            h: int,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
        ) -> torch.Tensor:
            # Apply ALiBi bias
            slopes = self._get_alibi_slopes(self.num_heads).to(score.device)
            slope = slopes[h]
            modified = score + slope * (q_idx - kv_idx)

            # Apply causal mask if needed
            if is_causal:
                modified = torch.where(
                    q_idx >= kv_idx, modified, float("-inf")
                )

            # Apply additive bias if provided
            if attn_bias is not None:
                bias_val = attn_bias[b, 0, 0, kv_idx]
                modified = modified + bias_val

            return modified

        # Create block mask for causal if needed
        block_mask = None
        if is_causal and self._create_block_mask is not None:
            Tq = q.shape[2]
            Tk = k.shape[2]

            def causal_mask(
                b: int, h: int, q_idx: torch.Tensor, kv_idx: torch.Tensor
            ) -> bool:
                return q_idx >= kv_idx

            block_mask = self._create_block_mask(
                causal_mask,
                B=q.shape[0],
                H=self.num_heads,
                Q_LEN=Tq,
                KV_LEN=Tk,
                device=q.device,
            )

        return self._flex_attention(
            q, k, v, score_mod=combined_score_mod, block_mask=block_mask
        )
