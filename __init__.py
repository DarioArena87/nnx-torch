"""
nnx — Modern Neural Network Layers for PyTorch
===============================================

A supplement to ``torch.nn`` with:

  * Modern attention backends (SDPA, FlexAttention, LinearAttention, RWKV)
  * HuggingFace-style attention masks (1/True = real, 0/False = padding)
  * FFN variants (SwiGLU, GeGLU, MoE)
  * Normalization layers (RMSNorm, ScaleNorm, AdaptiveRMSNorm)
  * Positional encodings (RoPE, ALiBi, Sinusoidal, Learned)
  * Composable Transformer blocks and stacks
"""

from nnx import layers, utils

from nnx.attention import (
    build_attention,
)
from nnx.layers import (
    ScaleNorm,
    CosineNorm,
    AdaptiveRMSNorm,
    FFN,
    GatedFFN,
    MoEFFN,
    TokenEmbedding,
    SinusoidalPositional,
    LearnedPositional,
    RotaryEmbedding,
    ALiBiEmbedding,
    TransformerLayer,
    TransformerStack,
    CrossAttentionLayer,
)
from nnx.utils import hf_to_additive, make_causal_mask, combine_masks

__version__ = "0.1.0"
