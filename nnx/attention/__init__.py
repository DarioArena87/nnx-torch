from typing import Optional, Literal

from .base import BaseAttention
from .flex import FlexAttention
from .linear import LinearAttention
from .rwkv import RWKVTimeMixing, RWKV6TimeMixing
from .sdpa import SDPAttention


def build_attention(
    attn_type: Literal["sdpa", "rwkv", "flex", "linear", "rwkv6"],
    embed_dim: int,
    num_heads: int = 1,
    dropout: float = 0.0,
    bias: bool = True,
    head_dim: Optional[int] = None,
):
    match attn_type:
        case "sdpa":
            return SDPAttention(embed_dim, num_heads, dropout, bias)
        case "rwkv":
            return RWKVTimeMixing(embed_dim, num_heads, head_size=head_dim)
        case "flex":
            return FlexAttention(embed_dim, num_heads, dropout, bias)
        case "linear":
            return LinearAttention(embed_dim, num_heads, head_dim=head_dim)
        case "rwkv6":
            return RWKV6TimeMixing(embed_dim, num_heads)
