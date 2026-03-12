"""
Mask utilities for nnx.

HuggingFace-style convention:
  - attention_mask: BoolTensor or {0,1} IntTensor of shape (B, T)
      1 / True  → real token, attend to it
      0 / False → padding, ignore it

Internally we convert to additive bias tensors:
  - 0.0   → attend
  - -inf  → block
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional


def hf_to_additive(
    attention_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert a HuggingFace-style boolean / 0-1 mask of shape (B, T)
    to a broadcastable additive bias of shape (B, 1, 1, T) ready
    to be added to raw attention logits.

    Args:
        attention_mask: (B, T) — 1/True = real, 0/False = padding.
        dtype: target dtype for the additive mask.

    Returns:
        (B, 1, 1, T) additive bias tensor.
    """
    # Ensure boolean
    mask = attention_mask.bool()  # (B, T)
    # 1 → 0.0, 0 → -inf
    additive = torch.zeros_like(mask, dtype=dtype)
    additive = additive.masked_fill(~mask, float("-inf"))
    return additive[:, None, None, :]  # (B, 1, 1, T)


def hf_to_additive_2d(
    attention_mask: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert a HuggingFace-style mask to a 2-D additive bias (B, 1, T_q, T_k).
    Useful when you have an explicit query mask and key mask and want
    to block padding in both dimensions.

    attention_mask: (B, T) applies equally to query and key positions.
    """
    additive_k = hf_to_additive(attention_mask, dtype=dtype)  # (B,1,1,T)
    return additive_k


def make_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Standard upper-triangular causal mask of shape (1, 1, T, T).
    0.0 on and below the diagonal, -inf above.
    """
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    return mask[None, None, :, :]  # (1, 1, T, T)


def combine_masks(*masks: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    Add together any number of broadcastable additive mask tensors,
    filtering out None values.
    """
    valid = [m for m in masks if m is not None]
    if not valid:
        return None
    out = valid[0]
    for m in valid[1:]:
        out = out + m
    return out


def pad_mask_to_4d(mask: torch.Tensor) -> torch.Tensor:
    """
    Accept masks of rank 2 (B,T), 3 (B,1,T) or 4 (B,H,Tq,Tk)
    and return a 4-D additive mask (B, 1, 1, T) or pass-through.
    """
    if mask.dim() == 2:
        return hf_to_additive(mask, dtype=torch.float32)
    if mask.dim() == 3:
        return mask.unsqueeze(2)
    return mask
