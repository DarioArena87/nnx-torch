"""
Linear Attention backends using Flash Linear Attention (FLA) library.

This module provides separate attention classes for each FLA variant:
- GLAAttention: Gated Linear Attention
- DeltaAttention: DeltaNet
- BasedAttention: Based linear attention
- RetentionAttention: RetNet-style retention

All classes inherit from BaseAttention and follow the standard nnx interface.

Requires: pip install flash-linear-attention
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import warnings
from einops import rearrange

from .base import BaseAttention


# ----------------------------------------------------------------------
# Kernel imports (lazy)
# ----------------------------------------------------------------------

def _import_gla_kernel():
    """Import GLA kernel."""
    try:
        from fla.ops.gla import chunk_gla
        return chunk_gla
    except ImportError as exc:
        raise ImportError(
            "GLAAttention requires the flash-linear-attention library. "
            "Install it with:\n"
            "  pip install flash-linear-attention\n"
            "or visit https://github.com/fla-org/flash-linear-attention",
        ) from exc


def _import_delta_kernel():
    """Import DeltaNet kernel."""
    try:
        from fla.ops.delta_rule import chunk_delta_rule
        return chunk_delta_rule
    except ImportError as exc:
        raise ImportError(
            "DeltaAttention requires the flash-linear-attention library. "
            "Install it with:\n"
            "  pip install flash-linear-attention\n"
            "or visit https://github.com/fla-org/flash-linear-attention",
        ) from exc


def _import_based_kernel():
    """Import Based kernel."""
    try:
        from fla.ops.based import parallel_based
        return parallel_based
    except ImportError as exc:
        raise ImportError(
            "BasedAttention requires the flash-linear-attention library. "
            "Install it with:\n"
            "  pip install flash-linear-attention\n"
            "or visit https://github.com/fla-org/flash-linear-attention",
        ) from exc


def _import_retention_kernel():
    """Import Retention kernel."""
    try:
        from fla.ops.retention import chunk_retention
        return chunk_retention
    except ImportError as exc:
        raise ImportError(
            "RetentionAttention requires the flash-linear-attention library. "
            "Install it with:\n"
            "  pip install flash-linear-attention\n"
            "or visit https://github.com/fla-org/flash-linear-attention",
        ) from exc


# ----------------------------------------------------------------------
# Base class for FLA variants
# ----------------------------------------------------------------------

class _FLABaseAttention(BaseAttention):
    """
    Base class for FLA linear attention variants.
    
    Handles common setup for FLA kernels: asymmetric projections for
    keys/values via expand_k/expand_v parameters, and head splitting.
    
    Note: Linear attention is inherently causal (uses recurrence), so
    the causal parameter is ignored — the recurrence always processes
    left-to-right.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        head_dim: Optional[int] = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        **kwargs,
    ) -> None:
        # Validate CUDA availability early
        assert torch.cuda.is_available(), f"{self.__class__.__name__} requires CUDA"
        
        super().__init__(embed_dim, num_heads, dropout, bias, head_dim)
        
        # Asymmetric projection dimensions
        self._k_dim = int(self.head_dim * expand_k)
        self._v_dim = int(self.head_dim * expand_v)
        
        # Override inner dimensions for asymmetric projections
        inner_q = self.head_dim * num_heads
        inner_k = self._k_dim * num_heads
        inner_v = self._v_dim * num_heads
        
        # Recreate projection layers with correct dimensions
        self.q_proj = nn.Linear(embed_dim, inner_q, bias=bias)
        self.k_proj = nn.Linear(embed_dim, inner_k, bias=bias)
        self.v_proj = nn.Linear(embed_dim, inner_v, bias=bias)
        self.out_proj = nn.Linear(inner_v, embed_dim, bias=bias)
        
        # Initialize kernel (to be done in subclass)
        self._kernel = None
    
    def _split_heads(self, x: torch.Tensor, head_dim: int) -> torch.Tensor:
        """Split tensor into heads: (B, T, D) -> (B, H, T, Dh)"""
        return rearrange(x, '... t (h d) -> ... h t d', h=self.num_heads, d=head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, Tq, D)
            key: (B, Tk, D) defaults to query
            value: (B, Tk, D) defaults to key
            attention_mask: HF-style (B, Tk) boolean mask (1=real, 0=pad)
            position_ids: (B, T) or (T,) position indices (ignored by linear attention)
            causal: ignored — linear attention is always causal
        
        Returns:
            (B, Tq, D)
        """
        if key is None:
            key = query
        if value is None:
            value = key
        
        # Project and split heads
        q = self._split_heads(self.q_proj(query), self.head_dim)  # (B, H, Tq, Dh)
        k = self._split_heads(self.k_proj(key), self._k_dim)      # (B, H, Tk, Dh_k)
        v = self._split_heads(self.v_proj(value), self._v_dim)    # (B, H, Tk, Dh_v)
        
        # Apply mask by zeroing padding positions (linear attention semantics)
        # Only apply to k/v if query length differs (cross-attention)
        if attention_mask is not None:
            pad = (~attention_mask.bool()).unsqueeze(-1)  # (B, Tk, 1)
            # Always apply to k and v
            k = k.masked_fill(pad.unsqueeze(1), 0.0)
            v = v.masked_fill(pad.unsqueeze(1), 0.0)
            # Apply to q only if Tq == Tk (self-attention)
            if q.shape[2] == k.shape[2]:
                q = q.masked_fill(pad.unsqueeze(1), 0.0)
        
        # Call kernel (subclass implements _attend) with original query for gate computation if needed
        out = self._attend(q, k, v, query, position_ids)
        
        # Merge heads and project
        out = self._merge_heads(out)  # (B, T, H*Dv)
        return self.out_proj(out)
    
    @property
    def kernel(self):
        """Lazy-load kernel on first use."""
        if self._kernel is None:
            self._kernel = self._import_kernel()
        return self._kernel


# ----------------------------------------------------------------------
# Gated Linear Attention (GLA)
# ----------------------------------------------------------------------

class GLAAttention(_FLABaseAttention):
    """
    Gated Linear Attention (GLA).
    
    Uses the chunk_gla kernel from flash-linear-attention.
    
    Args:
        embed_dim: Total embedding dimensionality.
        num_heads: Number of attention heads.
        dropout: Dropout probability (not applied in kernel, used for training).
        bias: Whether projection layers include bias.
        head_dim: Per-head dimensionality (default: embed_dim // num_heads).
        expand_k: Key expansion ratio (default: 1.0).
        expand_v: Value expansion ratio (default: 1.0).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        head_dim: Optional[int] = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            embed_dim, num_heads, dropout, bias, head_dim,
            expand_k=expand_k, expand_v=expand_v, **kwargs
        )
        # GLA requires a gate projection; gate dimension must match query dimension
        inner_g = self.head_dim * num_heads
        self.g_proj = nn.Linear(embed_dim, inner_g, bias=bias)
    
    def _import_kernel(self):
        return _import_gla_kernel()
    
    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply GLA kernel.
        
        Args:
            q: (B, H, Tq, Dh)
            k: (B, H, Tk, Dh_k)
            v: (B, H, Tk, Dh_v)
            query: (B, Tq, D) original query for gate computation
            position_ids: ignored (present for interface compatibility)
        
        Returns:
            (B, H, Tq, Dh_v)
        """
        # GLA requires query and key/value lengths to be equal
        if q.shape[2] != k.shape[2]:
            raise ValueError(
                f"GLAAttention does not support cross-attention with different query/key lengths. "
                f"Query length: {q.shape[2]}, Key length: {k.shape[2]}"
            )
        
        # GLA expects gate g with shape (B, T, H, Dh) matching q and k
        B, Tq, _ = query.shape
        g = rearrange(torch.sigmoid(self.g_proj(query)), '... t (h d) -> ... t h d', h=self.num_heads, d=self.head_dim)
        
        # Transpose q, k, v from head-first (B, H, T, D) to seq-first (B, T, H, D)
        q_seq = rearrange(q, '... h t d -> ... t h d').contiguous()
        k_seq = rearrange(k, '... h t d -> ... t h d').contiguous()
        v_seq = rearrange(v, '... h t d -> ... t h d').contiguous()
        
        # Call kernel: chunk_gla(q, k, v, g) -> (out, latent_state)
        # Output shape: (B, T, H, Dv)
        out_seq, _ = self.kernel(q_seq, k_seq, v_seq, g)
        
        # Transpose back to head-first (B, H, T, Dv)
        out = rearrange(out_seq, '... t h d -> ... h t d').contiguous()
        return out


# ----------------------------------------------------------------------
# DeltaNet Attention
# ----------------------------------------------------------------------

class DeltaAttention(_FLABaseAttention):
    """
    DeltaNet attention using the chunk_delta_rule kernel.
    
    DeltaNet is a gated linear attention variant with a simplified gating
    mechanism. It uses the same asymmetric projection structure as GLA.
    
    Note: DeltaNet kernel requires bfloat16 precision. The module is
    automatically converted to bfloat16 upon initialization.
    
    Args:
        embed_dim: Total embedding dimensionality.
        num_heads: Number of attention heads.
        dropout: Dropout probability (not applied in kernel, used for training).
        bias: Whether projection layers include bias.
        head_dim: Per-head dimensionality (default: embed_dim // num_heads).
        expand_k: Key expansion ratio (default: 1.0).
        expand_v: Value expansion ratio (default: 1.0).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        head_dim: Optional[int] = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            embed_dim, num_heads, dropout, bias, head_dim,
            expand_k=expand_k, expand_v=expand_v, **kwargs
        )
        # DeltaNet requires a gate projection (beta) that outputs a scalar per head per position
        # So output features = num_heads
        self.g_proj = nn.Linear(embed_dim, num_heads, bias=bias)
        # DeltaNet requires bfloat16 precision
        self.to(dtype=torch.bfloat16)
    
    def _import_kernel(self):
        return _import_delta_kernel()
    
    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply DeltaNet kernel.
        
        Args:
            q: (B, H, Tq, Dh)
            k: (B, H, Tk, Dh_k)
            v: (B, H, Tk, Dh_v)
            query: (B, Tq, D) original query for gate computation
            position_ids: ignored (present for interface compatibility)
        
        Returns:
            (B, H, Tq, Dh_v) in bfloat16
        """
        # DeltaNet requires query and key/value lengths to be equal
        if q.shape[2] != k.shape[2]:
            raise ValueError(
                f"DeltaAttention does not support cross-attention with different query/key lengths. "
                f"Query length: {q.shape[2]}, Key length: {k.shape[2]}"
            )
        
        # DeltaNet expects beta with shape (B, H, T)
        B, Tq, _ = query.shape
        # Project to (B, Tq, H) and rearrange to (B, H, Tq)
        beta = rearrange(torch.sigmoid(self.g_proj(query)), '... t h -> ... h t').contiguous()
        
        # Transpose q, k, v from head-first (B, H, T, D) to seq-first (B, T, H, D)
        q_seq = rearrange(q, '... h t d -> ... t h d').contiguous()
        k_seq = rearrange(k, '... h t d -> ... t h d').contiguous()
        v_seq = rearrange(v, '... h t d -> ... t h d').contiguous()
        
        # Call kernel: chunk_delta_rule(q, k, v, beta) -> (out, latent_state)
        # head_first=False (default) because we passed seq-first tensors
        out_seq, _ = self.kernel(q_seq, k_seq, v_seq, beta)
        
        # Transpose back to head-first (B, H, T, Dv)
        out = rearrange(out_seq, '... t h d -> ... h t d').contiguous()
        return out


# ----------------------------------------------------------------------
# Based Linear Attention
# ----------------------------------------------------------------------

class BasedAttention(_FLABaseAttention):
    """
    Based linear attention using the parallel_based kernel.
    
    Based attention does not use a gate and has a simpler interface.
    
    Note: Based kernel requires all sequences to have the same length.
    Cross-attention with different query/key lengths is not supported.
    
    Args:
        embed_dim: Total embedding dimensionality.
        num_heads: Number of attention heads.
        dropout: Dropout probability (not applied in kernel, used for training).
        bias: Whether projection layers include bias.
        head_dim: Per-head dimensionality (default: embed_dim // num_heads).
        expand_k: Key expansion ratio (default: 1.0).
        expand_v: Value expansion ratio (default: 1.0).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        head_dim: Optional[int] = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            embed_dim, num_heads, dropout, bias, head_dim,
            expand_k=expand_k, expand_v=expand_v, **kwargs
        )
        # Based does not have a gate projection
    
    def _import_kernel(self):
        return _import_based_kernel()
    
    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply Based kernel.
        
        Args:
            q: (B, H, Tq, Dh)
            k: (B, H, Tk, Dh_k)
            v: (B, H, Tk, Dh_v)
            query: (B, Tq, D) original query (unused)
            position_ids: ignored (present for interface compatibility)
        
        Returns:
            (B, H, Tq, Dh_v)
        """
        # parallel_based expects all sequences to have same length
        if q.shape[2] != k.shape[2]:
            raise ValueError(
                f"BasedAttention does not support cross-attention with different query/key lengths. "
                f"Query length: {q.shape[2]}, Key length: {k.shape[2]}"
            )
        
        # Transpose q, k, v from head-first (B, H, T, D) to seq-first (B, T, H, D)
        q_seq = rearrange(q, '... h t d -> ... t h d').contiguous()
        k_seq = rearrange(k, '... h t d -> ... t h d').contiguous()
        v_seq = rearrange(v, '... h t d -> ... t h d').contiguous()
        
        # parallel_based returns output with shape (B, T, H, Dv)
        # head_first=False (default)
        out_seq = self.kernel(q_seq, k_seq, v_seq)
        
        # Transpose back to head-first (B, H, T, Dv)
        out = rearrange(out_seq, '... t h d -> ... h t d').contiguous()
        return out


# ----------------------------------------------------------------------
# Retention Attention
# ----------------------------------------------------------------------

class RetentionAttention(_FLABaseAttention):
    """
    Retention-style linear attention using the chunk_retention kernel.
    
    Retention attention is similar to GLA but without a learned gate.
    
    Note: Retention kernel requires all sequences to have the same length.
    Cross-attention with different query/key lengths is not supported.
    
    Args:
        embed_dim: Total embedding dimensionality.
        num_heads: Number of attention heads.
        dropout: Dropout probability (not applied in kernel, used for training).
        bias: Whether projection layers include bias.
        head_dim: Per-head dimensionality (default: embed_dim // num_heads).
        expand_k: Key expansion ratio (default: 1.0).
        expand_v: Value expansion ratio (default: 1.0).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
        head_dim: Optional[int] = None,
        expand_k: float = 1.0,
        expand_v: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(
            embed_dim, num_heads, dropout, bias, head_dim,
            expand_k=expand_k, expand_v=expand_v, **kwargs
        )
        # Retention does not have a gate projection
    
    def _import_kernel(self):
        return _import_retention_kernel()
    
    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query: torch.Tensor,
        position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Apply Retention kernel.
        
        Args:
            q: (B, H, Tq, Dh)
            k: (B, H, Tk, Dh_k)
            v: (B, H, Tk, Dh_v)
            query: (B, Tq, D) original query (unused)
            position_ids: ignored (present for interface compatibility)
        
        Returns:
            (B, H, Tq, Dh_v)
        """
        # chunk_retention requires all sequences to have same length
        if q.shape[2] != k.shape[2]:
            raise ValueError(
                f"RetentionAttention does not support cross-attention with different query/key lengths. "
                f"Query length: {q.shape[2]}, Key length: {k.shape[2]}"
            )
        
        # Transpose q, k, v from head-first (B, H, T, D) to seq-first (B, T, H, D)
        q_seq = rearrange(q, '... h t d -> ... t h d').contiguous()
        k_seq = rearrange(k, '... h t d -> ... t h d').contiguous()
        v_seq = rearrange(v, '... h t d -> ... t h d').contiguous()
        
        # chunk_retention returns (out, latent_state) with out shape (B, T, H, Dv)
        # head_first=False (default)
        out_seq, _ = self.kernel(q_seq, k_seq, v_seq)
        
        # Transpose back to head-first (B, H, T, Dv)
        out = rearrange(out_seq, '... t h d -> ... h t d').contiguous()
        return out