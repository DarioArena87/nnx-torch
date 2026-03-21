from typing import Optional, Literal

from .base import BaseAttention
from .flex import FlexAttention
from .linear import (
    LinearAttention,
    GLAAttention,
    DeltaAttention,
    BasedAttention,
    RetentionAttention,
)
from .rwkv import RWKVTimeMixing, RWKV6TimeMixing
from .rope import RoPEAttention
from .alibi import ALiBiAttention
from .sdpa import SDPAttention


def build_attention(
    attn_type: Literal["sdpa", "rwkv", "flex", "linear", "rwkv6", "rope", "alibi", "gla", "delta", "based", "retention"],
    embed_dim: int,
    num_heads: int = 1,
    dropout: float = 0.0,
    bias: bool = True,
    head_dim: Optional[int] = None,
    **kwargs,
):
    match attn_type:
        case "sdpa":
            return SDPAttention(embed_dim, num_heads, dropout, bias)
        case "rwkv":
            return RWKVTimeMixing(embed_dim, head_size=head_dim, **kwargs)
        case "flex":
            return FlexAttention(embed_dim, num_heads, dropout, bias)
        case "linear":
            # Backward compatibility: use variant parameter if provided
            variant = kwargs.pop("variant", "gla")
            return LinearAttention(embed_dim, num_heads, head_dim=head_dim, variant=variant, **kwargs)
        case "gla":
            return GLAAttention(embed_dim, num_heads, dropout, bias, head_dim, **kwargs)
        case "delta":
            return DeltaAttention(embed_dim, num_heads, dropout, bias, head_dim, **kwargs)
        case "based":
            return BasedAttention(embed_dim, num_heads, dropout, bias, head_dim, **kwargs)
        case "retention":
            return RetentionAttention(embed_dim, num_heads, dropout, bias, head_dim, **kwargs)
        case "rwkv6":
            return RWKV6TimeMixing(embed_dim, num_heads, **kwargs)
        case "rope":
            return RoPEAttention(embed_dim, num_heads, dropout, bias, head_dim=head_dim, **kwargs)
        case "alibi":
            return ALiBiAttention(embed_dim, num_heads, dropout, bias, head_dim=head_dim, **kwargs)
        case _:
            raise ValueError(f"Unknown attention type: {attn_type}")
