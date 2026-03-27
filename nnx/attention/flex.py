"""
Flex Attention backend.

``torch.nn.attention.flex_attention`` (introduced in PyTorch 2.3) lets
you express arbitrary attention patterns as a Python score-modification
function that gets compiled into a single efficient CUDA kernel via
``torch.compile``.

This backend is ideal when you need custom bias functions (e.g. ALiBi,
document-level masking, relative position biases) that cannot be easily
expressed as a dense additive tensor.

Requires: PyTorch >= 2.3, CUDA device.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor
from torch.nn.attention.flex_attention import AuxOutput, BlockMask

from .base import BaseAttention


def _import_flex() -> tuple[Callable[..., Tensor | tuple[Tensor, Tensor] | tuple[Tensor, AuxOutput]], Callable[..., BlockMask]]:
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask

        return flex_attention, create_block_mask
    except ImportError as exc:
        raise ImportError(
            "FlexAttention requires PyTorch >= 2.3.  Please upgrade: pip install --upgrade torch"
        ) from exc


class FlexAttention(BaseAttention):
    """
    Multi-head attention backed by ``torch.nn.attention.flex_attention``.

    The key feature of this backend is the ``score_mod`` callable: a
    function that receives the raw dot-product score and position indices
    and returns a modified score.  This allows custom bias patterns
    (e.g. ALiBi, Rotary, document boundaries) to be fused into the
    attention kernel.

    Args:
        embed_dim:   Total embedding dimensionality.
        num_heads:   Number of attention heads.
        dropout:     Dropout probability (passed to flex_attention).
        bias:        Whether projection layers include a bias term.
        head_dim:    Per-head dimensionality.
        score_mod:   Optional callable with signature
                     ``(score, b, h, q_idx, kv_idx) -> score``.
                     If ``None``, plain dot-product attention is used.
        block_mask:  Optional pre-built BlockMask (from
                     ``create_block_mask``).  Efficient for sparse patterns.

    Example — ALiBi-style score_mod::

        def alibi_score_mod(score, b, h, q_idx, kv_idx):
            bias = -torch.abs(q_idx - kv_idx).float() * (h + 1) * 0.5
            return score + bias

        attn = FlexAttention(512, 8, score_mod=alibi_score_mod)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        head_dim: Optional[int] = None,
        score_mod: Optional[Callable] = None,
        block_mask=None, ) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, head_dim)
        self.score_mod = score_mod
        self.block_mask = block_mask
        self._flex_attention, _ = _import_flex()

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_bias: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # flex_attention uses score_mod for custom biases; additive biases
        # from padding masks are composed on top if provided.

        score_mod = self.score_mod

        if attn_bias is not None:
            # Wrap any additive bias into the score_mod pipeline.
            _bias = attn_bias  # captured in closure

            def _score_mod_with_bias(score, b, h, q_idx, kv_idx):
                # attn_bias is (B, 1, 1, Tk) — index appropriately
                bias_val = _bias[b, 0, 0, kv_idx]
                modified = score + bias_val
                if score_mod is not None:
                    modified = score_mod(modified, b, h, q_idx, kv_idx)
                return modified

            effective_score_mod = _score_mod_with_bias
        else:
            effective_score_mod = score_mod

        return self._flex_attention(q, k, v, score_mod=effective_score_mod, block_mask=self.block_mask)
