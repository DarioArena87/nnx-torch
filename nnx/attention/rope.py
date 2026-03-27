"""
Rotary Position Embedding (RoPE) attention.

This attention backend applies rotary positional embeddings to queries
and keys before computing scaled dot-product attention.  RoPE injects
positional information by rotating Q/K vectors in complex space, which
preserves relative position relationships without adding extra parameters.

The implementation uses :class:`nnx.layers.embedding.RotaryEmbedding` to
perform the rotation.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseAttention
from ..layers.embedding import RotaryEmbedding


class RoPEAttention(BaseAttention):
    """
    Multi-head attention with Rotary Position Embedding (RoPE).

    RoPE is applied to queries and keys before the attention computation,
    encoding absolute positions as rotations in the embedding space.  This
    preserves the dot-product attention property while enabling the model
    to attend based on relative positions.

    Args:
        embed_dim:  Total embedding dimensionality.
        num_heads:  Number of attention heads.
        dropout:    Dropout probability on attention weights.
        bias:       Whether QKV and output projections have a bias.
        base:       The angle base for RoPE.  Default 10000 (LLaMA-3 uses 500000).
        max_len:    Maximum sequence length for pre-computed rotation cache.
        head_dim:   Per-head dimensionality (default: embed_dim // num_heads).

    Note:
        The head_dim must be even for RoPE to work correctly (since it
        operates on pairs of dimensions).  An error is raised if head_dim
        is odd.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        base: float = 10000.0,
        max_len: int = 4096,
        head_dim: Optional[int] = None,
    ) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, head_dim)

        if self.head_dim % 2 != 0:
            raise ValueError(
                f"RoPE requires an even head_dim, got {self.head_dim}. "
                "Set head_dim explicitly or adjust embed_dim/num_heads."
            )

        self.rope = RotaryEmbedding(self.head_dim, max_len, base)

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
            if Tq == Tk:
                # Self-attention: use the standard rotate_queries_keys
                q, k = self.rope.rotate_queries_keys(q, k)
            else:
                # Cross-attention: apply RoPE separately with offset=0
                # Rotate queries
                cos_q = self.rope.cos_cached[:Tq].unsqueeze(0).unsqueeze(0)
                sin_q = self.rope.sin_cached[:Tq].unsqueeze(0).unsqueeze(0)
                q = q * cos_q + self.rope._rotate_half(q) * sin_q
                # Rotate keys
                cos_k = self.rope.cos_cached[:Tk].unsqueeze(0).unsqueeze(0)
                sin_k = self.rope.sin_cached[:Tk].unsqueeze(0).unsqueeze(0)
                k = k * cos_k + self.rope._rotate_half(k) * sin_k
        else:
            # Use explicit position_ids
            q = self.rope.rotate_with_positions(q, position_ids)
            k = self.rope.rotate_with_positions(k, position_ids)

        # Scaled dot-product attention
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
