"""
Normalization layers.

Provides drop-in alternatives to ``torch.nn.LayerNorm``:

  - :class:`RMSNorm`        — Root Mean Square LayerNorm (Zhang & Sennrich, 2019).
                              Used in LLaMA, Mistral, Qwen, …
  - :class:`ScaleNorm`      — Normalization by a single learned scalar.
  - :class:`CosineNorm`     — Normalize to unit sphere (cosine similarity).
  - :class:`AdaptiveRMSNorm`— RMSNorm with conditioning from a side input
                              (useful for diffusion / cross-modal models).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleNorm(nn.Module):
    """
    Normalise by the L2 norm of the feature vector, then scale by a
    single learned scalar ``g``.  Proposed by Nguyen & Salazar (2019).

    Args:
        dim:  Feature dimensionality (used only for ``g`` initialisation).
        eps:  Numerical stability epsilon.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.tensor(dim**0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return self.scale * x / norm


class CosineNorm(nn.Module):
    """
    Normalise feature vectors to lie on the unit hypersphere.

    This is equivalent to ``ScaleNorm`` with a fixed scale of 1 and is
    useful when the downstream operation (e.g. a dot-product) benefits
    from bounded inputs.

    Args:
        eps: Numerical stability epsilon.
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, dim=-1, eps=self.eps)


class AdaptiveRMSNorm(nn.Module):
    """
    RMSNorm with a conditioning signal (modulation).

    The scale (and optionally shift) parameters are predicted from a
    side input ``cond`` via a small linear projection, rather than
    being fixed learned parameters.  This is the pattern used in
    DiT (Peebles & Xie, 2022) and many diffusion / flow models.

    Args:
        dim:        Feature dimensionality.
        cond_dim:   Dimensionality of the conditioning signal.
        eps:        Numerical stability epsilon.
        use_shift:  If True, also predict a shift (bias) from ``cond``.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-6,
        use_shift: bool = True,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.use_shift = use_shift
        out_features = dim * 2 if use_shift else dim
        self.proj = nn.Linear(cond_dim, out_features, bias=True)
        # Zero-init so the network starts as a plain RMSNorm
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, T, D) input.
            cond: (B, cond_dim) or (B, 1, cond_dim) conditioning signal.

        Returns:
            (B, T, D) normalised and modulated output.
        """
        if cond.dim() == 2:
            cond = cond.unsqueeze(1)  # (B, 1, cond_dim)

        params = self.proj(cond)  # (B, 1, D) or (B, 1, 2D)
        normed = self._norm(x.float()).to(x.dtype)

        if self.use_shift:
            scale, shift = params.chunk(2, dim=-1)
            return normed * (1 + scale) + shift
        else:
            return normed * (1 + params)
