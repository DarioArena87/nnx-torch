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
from collections import OrderedDict
from typing import Optional
from einops import rearrange

# ------------------------------------------------------------------
# Mask caching (M1 optimization)
# ------------------------------------------------------------------

_mask_cache: OrderedDict[str, torch.Tensor] = OrderedDict()
_MAX_MASK_CACHE_SIZE = 64


def _make_cache_key(size: int, device: torch.device, dtype: torch.dtype) -> str:
    """Create a unique cache key for a mask configuration."""
    return f"{size}_{device}_{dtype}"


def get_cached_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Retrieve or create a causal mask, caching it for future use.

    Args:
        seq_len: Sequence length (T).
        device: Target device for the mask tensor.
        dtype: Target dtype for the mask tensor.

    Returns:
        (1, 1, T, T) additive causal bias tensor.
    """
    key = _make_cache_key(seq_len, device, dtype)
    if key in _mask_cache:
        return _mask_cache[key]

    # Build the mask
    mask = torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=dtype),
        diagonal=1,
    )
    result = mask[None, None, :, :]  # (1, 1, T, T)

    # Evict oldest entries if cache is full
    while len(_mask_cache) >= _MAX_MASK_CACHE_SIZE:
        _mask_cache.popitem(last=False)

    _mask_cache[key] = result
    return result


def clear_mask_cache() -> None:
    """Clear all cached attention masks."""
    _mask_cache.clear()


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
    return rearrange(additive, '... t -> ... 1 1 t')  # (B, 1, 1, T)


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

    Uses the internal cache to avoid recreating masks for common sizes.
    """
    return get_cached_causal_mask(seq_len, device, dtype)


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
        return rearrange(mask, '... h t -> ... h 1 t')
    return mask
