"""
Embedding and positional encoding layers.

  - :class:`TokenEmbedding`        — Simple learnable token embedding.
  - :class:`SinusoidalPositional`  — Fixed sinusoidal embeddings (Vaswani 2017).
  - :class:`LearnedPositional`     — Learnable positional embeddings (GPT-2).
  - :class:`RotaryEmbedding`       — RoPE (Su et al., 2021); used in LLaMA.
  - :class:`ALiBiEmbedding`        — ALiBi slopes (Press et al., 2021).
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Token embedding
# ---------------------------------------------------------------------------


class TokenEmbedding(nn.Module):
    """
    Learnable token embedding with optional output-scale by √embed_dim.

    Args:
        vocab_size:  Vocabulary size.
        embed_dim:   Embedding dimensionality.
        padding_idx: Padding token index (embedding is zeroed).
        scale:       If True, multiply embeddings by ``√embed_dim`` (as in
                     the original Transformer paper).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: Optional[int] = None,
        scale: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.scale = math.sqrt(embed_dim) if scale else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * self.scale


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------


class SinusoidalPositional(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    Args:
        embed_dim:   Embedding dimensionality.
        max_len:     Maximum sequence length.
        dropout:     Dropout applied to ``embedding + positional``.
    """

    def __init__(
        self,
        embed_dim: int,
        max_len: int = 4096,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)

        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: embed_dim // 2])
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) input embeddings.

        Returns:
            (B, T, D) embeddings + positional encoding.
        """
        return self.drop(x + self.pe[:, : x.size(1)])


# ---------------------------------------------------------------------------
# Learned positional encoding
# ---------------------------------------------------------------------------


class LearnedPositional(nn.Module):
    """
    Learnable absolute positional embedding (GPT-2 style).

    Args:
        max_len:   Maximum sequence length.
        embed_dim: Embedding dimensionality.
        dropout:   Dropout applied after addition.
    """

    def __init__(
        self,
        max_len: int,
        embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pe = nn.Embedding(max_len, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        B, T, D = x.shape
        positions = torch.arange(offset, offset + T, device=x.device)
        return self.drop(x + self.pe(positions).unsqueeze(0))


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al. (2021).

    Unlike additive positional encodings, RoPE is applied *inside* the
    attention module to Q and K vectors.  Call :meth:`rotate_queries_keys`
    before computing attention scores.

    Args:
        head_dim:  Dimensionality per attention head.
        max_len:   Maximum sequence length for pre-computation.
        base:      The angle base.  Default 10000.  LLaMA-3 uses 500000.
        dtype:     Dtype for the rotation matrices.
    """

    def __init__(
        self,
        head_dim: int,
        max_len: int = 4096,
        base: float = 10_000.0,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base

        # Pre-compute cos/sin cache
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_len, dtype)

    def _build_cache(self, seq_len: int, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)  # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (T, D)
        self.register_buffer("cos_cached", emb.cos().to(dtype))
        self.register_buffer("sin_cached", emb.sin().to(dtype))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Split last dim in half, negate second half and swap."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def rotate_queries_keys(
        self,
        q: torch.Tensor,  # (B, H, T, D)
        k: torch.Tensor,  # (B, H, T, D)
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to Q and K in-place equivalent.

        Args:
            q: (B, H, T, head_dim) query tensor.
            k: (B, H, T, head_dim) key tensor.
            offset: Token offset for KV-cache inference.

        Returns:
            Rotated (q, k).
        """
        T = q.shape[2]
        cos = self.cos_cached[offset : offset + T].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset : offset + T].unsqueeze(0).unsqueeze(0)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.rotate_queries_keys(q, k, offset=offset)

    def rotate_with_positions(
        self,
        x: torch.Tensor,           # (B, H, T, D)
        position_ids: torch.Tensor, # (B, T) or (T,)
    ) -> torch.Tensor:
        """
        Apply RoPE using explicit position indices.

        Supports chunked sequences where position_ids may be [2049, 2050, ...].

        Args:
            x: (B, H, T, head_dim) tensor to rotate.
            position_ids: (B, T) or (T,) position indices.

        Returns:
            Rotated tensor.
        """
        # Ensure position_ids is 2D (B, T) for consistent indexing
        if position_ids.dim() == 1:
            # (T,) -> (1, T) -> broadcast to (B, T)
            position_ids = position_ids.unsqueeze(0)
            # Will broadcast to batch dimension

        # Get cos/sin for each position: (B, T, D)
        cos = self.cos_cached[position_ids]  # (B, T, D)
        sin = self.sin_cached[position_ids]  # (B, T, D)

        # Reshape for broadcasting: (B, 1, T, D)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        return x * cos + self._rotate_half(x) * sin

    def _extend_cache(self, seq_len: int) -> None:
        """Extend the cos/sin cache to accommodate longer sequences."""
        dtype = self.cos_cached.dtype
        self._build_cache(seq_len, dtype)


# ---------------------------------------------------------------------------
# ALiBi — Attention with Linear Biases
# ---------------------------------------------------------------------------


class ALiBiEmbedding(nn.Module):
    """
    Attention with Linear Biases (ALiBi) — Press et al. (2021).

    ALiBi adds a fixed, head-specific linear bias to attention logits.
    Unlike additive position embeddings it does not increase the token
    representation but biases *attention* so that nearby tokens are
    preferred.

    This module produces additive bias tensors to be passed as
    ``attention_mask`` (or combined with them) to an attention backend.

    Args:
        num_heads: Number of attention heads.
        max_len:   Maximum sequence length for pre-computation.

    Usage::

        alibi = ALiBiEmbedding(num_heads=8, max_len=2048)
        bias = alibi(seq_len=512, device=x.device)  # (1, H, T, T)
        out = attn(x, attention_mask=bias)
    """

    def __init__(self, num_heads: int, max_len: int = 4096) -> None:
        super().__init__()
        slopes = self._get_slopes(num_heads)
        self.register_buffer("slopes", slopes)  # (H,)
        self.max_len = max_len

    @staticmethod
    def _get_slopes(n: int) -> torch.Tensor:
        """Compute per-head slopes as in the ALiBi paper."""

        def _slopes(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start**i) for i in range(n)]

        if math.log2(n).is_integer():
            return torch.tensor(_slopes(n), dtype=torch.float32)
        # nearest power-of-2 trick from the paper
        p = 2 ** math.floor(math.log2(n))
        slopes = _slopes(p) + _slopes(2 * p)[0::2][: n - p]
        return torch.tensor(slopes, dtype=torch.float32)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            (1, H, T, T) additive ALiBi bias matrix.
        """
        positions = torch.arange(seq_len, device=device)
        # relative distances: (T, T)
        dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (T, T)
        dist = dist.float()
        # (H, 1, 1) * (1, T, T) → (H, T, T)
        bias = -self.slopes.to(device)[:, None, None] * dist.abs().unsqueeze(0)
        return bias.unsqueeze(0)  # (1, H, T, T)

    def with_positions(
        self,
        position_ids: torch.Tensor,  # (B, T) or (T,)
        device: torch.device,
    ) -> torch.Tensor:
        """
        Compute ALiBi bias using explicit position indices.

        Supports chunked sequences where position_ids may be [2049, 2050, ...].

        Args:
            position_ids: (B, T) or (T,) position indices.
            device: Target device.

        Returns:
            (B, H, Tq, Tk) additive ALiBi bias matrix.
        """
        # Ensure position_ids is 2D: (B, T)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)  # (1, T)

        B, T = position_ids.shape

        # Get positions for queries and keys
        # position_ids: (B, T) representing absolute positions
        # We need relative distances between all query and key positions
        q_pos = rearrange(position_ids, '... t -> ... t 1')
        k_pos = rearrange(position_ids, '... t -> ... 1 t')
        dist = (q_pos - k_pos).abs()  # (B, T, T)

        # Apply slopes: (H, 1, 1) * (B, T, T) -> (B, H, T, T)
        bias = -self.slopes.to(device)[None, :, None, None] * rearrange(dist, '... t1 t2 -> ... 1 t1 t2').to(device)

        return bias  # (B, H, T, T)
