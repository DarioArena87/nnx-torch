"""
Scaled Dot-Product Attention (SDPA) backend.

Uses ``torch.nn.functional.scaled_dot_product_attention``, which
dispatches to FlashAttention-2, memory-efficient attention, or the
math kernel depending on what is available on the current device.

This is the recommended default backend — it requires no extra
dependencies and is fused on CUDA.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .base import BaseAttention


class SDPAttention(BaseAttention):
    """
    Multi-head attention backed by
    ``torch.nn.functional.scaled_dot_product_attention``.

    This backend supports:
      - Arbitrary additive attention biases (e.g. ALiBi, RoPE, padding).
      - Causal masking (via ``causal=True`` in ``forward``).
      - Dropout during training.
      - FlashAttention-2 / memory-efficient kernels automatically when
        the attn_bias is ``None`` and the device supports it.

    Args:
        embed_dim:  Total embedding dimensionality.
        num_heads:  Number of attention heads.
        dropout:    Dropout probability on attention weights.
        bias:       Whether projection layers include a bias term.
        head_dim:   Per-head dimensionality (default: embed_dim // num_heads).
    """

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # torch.nn.functional.scaled_dot_product_attention accepts an
        # *additive* attn_mask (identical to our convention).
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
        )
