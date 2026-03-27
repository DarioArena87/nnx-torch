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

from typing import Optional

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
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, head_dim)
        self.alibi = ALiBiEmbedding(num_heads, max_len)

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
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
