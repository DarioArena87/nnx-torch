"""
High-level Transformer building blocks.

  - :class:`TransformerLayer`   — A single Transformer block (Pre-LN by default) with a pluggable attention backend and FFN variant.
  - :class:`TransformerStack`   — A stack of N :class:`TransformerLayer` s.
  - :class:`CrossAttentionLayer`— Encoder-decoder cross-attention block.

All blocks accept HuggingFace-style attention masks.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from nnx.attention.base import BaseAttention
from nnx.attention.sdpa import SDPAttention
from nnx.layers.feedforward import GatedFFN

# Norm type alias
_NormType = Literal["layernorm", "rmsnorm"]


def _make_norm(norm_type: _NormType, dim: int) -> nn.Module:
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return nn.RMSNorm(dim)
    else:
        raise ValueError(f"Unknown norm type: {norm_type!r}")


# ---------------------------------------------------------------------------
# Single Transformer Layer
# ---------------------------------------------------------------------------


class TransformerLayer(nn.Module):
    """
    A single Transformer encoder/decoder block.

    Supports:
      - Pre-LN (default) and Post-LN ordering.
      - Any attention backend that subclasses :class:`~nnx.attention.base.BaseAttention`.
      - Any FFN variant (:class:`~nnx.layers.feedforward.FFN`,
        :class:`~nnx.layers.feedforward.GatedFFN`,
        :class:`~nnx.layers.feedforward.MoEFFN`, …).
      - Optional causal masking.
      - HuggingFace-style attention masks.

    Args:
        embed_dim:      Model width.
        num_heads:      Number of attention heads.
        ffn_dim:        FFN hidden dimensionality.
        dropout:        Dropout rate applied after attention and FFN.
        attention_cls:  Attention class to use.  Must accept
                        ``(embed_dim, num_heads, **attention_kwargs)``
                        and expose the :class:`~nnx.attention.base.BaseAttention`
                        ``forward`` signature.
        attention_kwargs: Extra kwargs forwarded to ``attention_cls.__init__``.
        ffn_cls:        FFN class to use.
        ffn_kwargs:     Extra kwargs forwarded to ``ffn_cls.__init__``.
        norm_type:      ``"layernorm"`` or ``"rmsnorm"``.
        pre_norm:       If True (default), apply norm before sub-layers (Pre-LN).
                        If False, apply norm after (Post-LN, original Transformer).
        causal:         If True, automatically add a causal mask.

    Example — LLaMA-style block::

        from nnx.layers.transformer import TransformerLayer
        from nnx.layers.feedforward import GatedFFN

        layer = TransformerLayer(
            embed_dim=4096,
            num_heads=32,
            ffn_cls=GatedFFN,
            ffn_kwargs={"activation": "silu"},
            norm_type="rmsnorm",
            causal=True,
        )
        out = layer(x, attention_mask=mask)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_cls: Type[BaseAttention] = SDPAttention,
        attention_kwargs: Optional[dict] = None,
        ffn_cls=GatedFFN,
        ffn_kwargs: Optional[dict] = None,
        norm_type: _NormType = "rmsnorm",
        pre_norm: bool = True,
        causal: bool = False,
        gradient_checkpointing: bool = False, ) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.causal = causal
        self.gradient_checkpointing = gradient_checkpointing

        attn_kw = attention_kwargs or {}
        ffn_kw = ffn_kwargs or {}

        self.self_attn = attention_cls(embed_dim, num_heads, dropout=dropout, **attn_kw)
        self.ffn = ffn_cls(embed_dim, ffn_dim, **ffn_kw)

        self.norm1 = _make_norm(norm_type, embed_dim)
        self.norm2 = _make_norm(norm_type, embed_dim)
        self.drop = nn.Dropout(dropout)

    def _forward_impl(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Internal forward implementation for checkpointing support."""
        # --- Self-attention sub-layer ---
        residual = x
        if self.pre_norm:
            x = self.norm1(x)

        x = self.self_attn(x, key=key, value=value, attention_mask=attention_mask, position_ids=position_ids, causal=self.causal)
        x = self.drop(x) + residual

        if not self.pre_norm:
            x = self.norm1(x)

        # --- FFN sub-layer ---
        residual = x
        if self.pre_norm:
            x = self.norm2(x)

        x = self.drop(self.ffn(x)) + residual

        if not self.pre_norm:
            x = self.norm2(x)

        return x

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None, ) -> torch.Tensor:
        """
        Args:
            x:              (B, T, D) input.
            attention_mask: HF-style (B, T) or (B, 1, 1, T) additive mask.
                            1/True = real token, 0/False = padding.
            position_ids:   Optional (B, Tk) position ids for attention layer.
            key:            Optional (B, Tk, D) for cross-attention.
            value:          Optional (B, Tk, D) for cross-attention.

        Returns:
            (B, T, D)
        """
        if self.gradient_checkpointing and self.training:
            return checkpoint(
                self._forward_impl, x, attention_mask, position_ids, key, value, use_reentrant=False
            )
        return self._forward_impl(x, attention_mask, position_ids, key, value)


# ---------------------------------------------------------------------------
# Cross-Attention Layer
# ---------------------------------------------------------------------------


class CrossAttentionLayer(nn.Module):
    """
    Encoder-decoder cross-attention block (Pre-LN).

    A Transformer decoder block with *two* attention sub-layers:
      1. Masked self-attention over the target sequence.
      2. Cross-attention from target queries to encoder key/values.

    Args:
        embed_dim:          Model width.
        num_heads:          Number of attention heads.
        ffn_dim:            FFN hidden dimensionality.
        dropout:            Dropout rate.
        attention_cls:      Shared attention class for both sub-layers.
        attention_kwargs:   Extra kwargs for attention.
        ffn_cls:            FFN class.
        ffn_kwargs:         Extra kwargs for FFN.
        norm_type:          Normalisation variant.
        causal:             Whether decoder self-attention is causal.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_cls: Type[BaseAttention] = SDPAttention,
        attention_kwargs: Optional[dict] = None,
        ffn_cls=GatedFFN,
        ffn_kwargs: Optional[dict] = None,
        norm_type: _NormType = "rmsnorm",
        causal: bool = True, ) -> None:
        super().__init__()
        self.causal = causal

        kw = attention_kwargs or {}
        ffn_kw = ffn_kwargs or {}

        self.self_attn = attention_cls(embed_dim, num_heads, dropout=dropout, **kw)
        self.cross_attn = attention_cls(embed_dim, num_heads, dropout=dropout, **kw)
        self.ffn = ffn_cls(embed_dim, ffn_dim, **ffn_kw)

        self.norm1 = _make_norm(norm_type, embed_dim)
        self.norm2 = _make_norm(norm_type, embed_dim)
        self.norm3 = _make_norm(norm_type, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None, ) -> torch.Tensor:
        """
        Args:
            x:               (B, Tq, D) decoder input.
            encoder_out:     (B, Tk, D) encoder output.
            self_attn_mask:  HF-style (B, Tq) decoder padding mask.
            cross_attn_mask: HF-style (B, Tk) encoder padding mask.

        Returns:
            (B, Tq, D)
        """
        # 1. Masked self-attention
        residual = x
        x = self.self_attn(self.norm1(x), attention_mask=self_attn_mask, causal=self.causal)
        x = self.drop(x) + residual

        # 2. Cross-attention
        residual = x
        normed = self.norm2(x)
        x = self.cross_attn(normed, key=encoder_out, value=encoder_out, attention_mask=cross_attn_mask, causal=False)
        x = self.drop(x) + residual

        # 3. FFN
        residual = x
        x = self.drop(self.ffn(self.norm3(x))) + residual
        return x


# ---------------------------------------------------------------------------
# Transformer Stack
# ---------------------------------------------------------------------------


class TransformerStack(nn.Module):
    """
    A configurable stack of :class:`TransformerLayer` blocks.

    Args:
        n_layers:     Number of layers.
        embed_dim:    Model width.
        num_heads:    Number of attention heads.
        ffn_dim:      FFN hidden dimensionality.
        dropout:      Dropout rate.
        attention_cls: Attention class shared across all layers.
        attention_kwargs: Extra kwargs for attention init.
        ffn_cls:      FFN class.
        ffn_kwargs:   Extra kwargs for FFN init.
        norm_type:    ``"layernorm"`` or ``"rmsnorm"``.
        pre_norm:     Pre-LN (True) or Post-LN (False).
        causal:       Causal masking.
        final_norm:   If True, add a normalisation layer after the last block.
        gradient_checkpointing: If True, use activation checkpointing.
        use_nested_tensor: If True and padding mask is provided, convert to
            ``torch.nested.nested_tensor`` for optimized variable-length
            sequence processing. Falls back to regular tensors when
            NestedTensor is not available or not beneficial.

    Example — GPT-style decoder::

        from nnx.layers.transformer import TransformerStack

        model = TransformerStack(
            n_layers=12,
            embed_dim=768,
            num_heads=12,
            norm_type="layernorm",
            causal=True,
            final_norm=True,
        )
        out = model(x, attention_mask=mask)  # (B, T, 768)
    """

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        num_heads: int,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.0,
        attention_cls: Type[BaseAttention] = SDPAttention,
        attention_kwargs: Optional[dict] = None,
        ffn_cls=GatedFFN,
        ffn_kwargs: Optional[dict] = None,
        norm_type: _NormType = "rmsnorm",
        pre_norm: bool = True,
        causal: bool = False,
        final_norm: bool = True,
        gradient_checkpointing: bool = False,
        use_nested_tensor: bool = False,
    ) -> None:
        super().__init__()

        layer_kwargs = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
            attention_cls=attention_cls,
            attention_kwargs=attention_kwargs,
            ffn_cls=ffn_cls,
            ffn_kwargs=ffn_kwargs,
            norm_type=norm_type,
            pre_norm=pre_norm,
            causal=causal,
            gradient_checkpointing=gradient_checkpointing, )

        self.layers = nn.ModuleList([TransformerLayer(**layer_kwargs) for _ in range(n_layers)])
        self.final_norm = _make_norm(norm_type, embed_dim) if final_norm else None
        
        # T4: NestedTensor support
        self.use_nested_tensor = use_nested_tensor
        self._nested_tensor_available = self._check_nested_tensor_support()

    @staticmethod
    def _check_nested_tensor_support() -> bool:
        """T4: Check if NestedTensor is available in this PyTorch version."""
        try:
            # NestedTensor support was added in PyTorch 1.13+
            # and improved in 2.0+
            nt = torch.nested.nested_tensor([torch.randn(3, 4)])
            del nt
            return True
        except (AttributeError, RuntimeError):
            return False

    def _to_nested_tensor(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        T4: Convert padded tensors to NestedTensor for variable-length sequences.
        
        Args:
            x: (B, T, D) padded input tensor.
            attention_mask: (B, T) boolean mask where 1=real, 0=padding.
        
        Returns:
            Tuple of (NestedTensor, None) - mask is not needed for NestedTensor.
        """
        if attention_mask is None or not self._nested_tensor_available:
            return x, attention_mask
        
        # Get sequence lengths from mask
        seq_lengths = attention_mask.sum(dim=1).long()  # (B,)
        
        # Extract individual sequences based on lengths
        sequences = []
        for i, length in enumerate(seq_lengths):
            if length > 0:
                sequences.append(x[i, :length])
        
        if not sequences:
            # All sequences are empty, return original
            return x, attention_mask
        
        # Create nested tensor
        nested_x = torch.nested.nested_tensor(sequences, layout=torch.jagged)
        
        return nested_x, None

    def _from_nested_tensor(
        self,
        nested_x: torch.Tensor,
        original_shape: torch.Size,
    ) -> torch.Tensor:
        """
        T4: Convert NestedTensor back to padded tensor.
        
        Args:
            nested_x: NestedTensor output.
            original_shape: Original (B, T, D) shape for padding.
        
        Returns:
            (B, T, D) padded tensor.
        """
        if not isinstance(nested_x, torch.Tensor) or nested_x.layout != torch.jagged:
            return nested_x
        
        # Convert jagged tensor back to padded
        B, T, D = original_shape
        padded = torch.zeros(B, T, D, device=nested_x.device, dtype=nested_x.dtype)
        
        for i in range(B):
            seq = nested_x[i]
            seq_len = seq.shape[0]
            if seq_len > 0:
                padded[i, :seq_len] = seq
        
        return padded

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, T, D) input embeddings.
            attention_mask: HF-style (B, T) mask.
            position_ids: (B, T) position ids.

        Returns:
            (B, T, D) output.
        """
        # T4: Convert to NestedTensor if enabled
        original_shape = x.shape
        use_nested = (
            self.use_nested_tensor
            and self._nested_tensor_available
            and attention_mask is not None
        )
        
        if use_nested:
            x, attention_mask = self._to_nested_tensor(x, attention_mask)
        
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, position_ids=position_ids)

        if self.final_norm is not None:
            x = self.final_norm(x)
        
        # T4: Convert back from NestedTensor if needed
        if use_nested:
            x = self._from_nested_tensor(x, original_shape)

        return x
