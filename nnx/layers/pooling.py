from __future__ import annotations

import torch
from .normalization import RMSNorm
from torch import nn as nn, Tensor
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Pooling for fixed-size latent representation
# ---------------------------------------------------------------------------
class LatentPooler(nn.Module):
    """Pools variable-length sequence hidden states into a fixed-size latent vector.

    Uses attention-weighted pooling where a learnable query attends over the
    sequence to produce a weighted sum, followed by a projection to latent_dim.
    """

    def __init__(self, d_model: int, latent_dim: int, dropout: float = 0.0):
        """Initialize the pooler.

        Args:
            d_model: Input hidden dimension.
            latent_dim: Output latent dimension.
            dropout: Dropout probability.
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.attn = nn.Linear(d_model, 1)
        self.proj = nn.Linear(d_model, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = RMSNorm(d_model)

    def forward(self, hidden_states: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        """Pool sequence to fixed-size latent representation.

        Args:
            hidden_states: Tensor of shape (batch, seq_len, d_model).
            attention_mask: Tensor of shape (batch, seq_len), 1 for real tokens.

        Returns:
            Tensor of shape (batch, latent_dim).
        """
        # Compute attention scores using learnable query: (batch, seq_len)
        # Expand query to match batch size: (batch, d_model)
        query = self.query.unsqueeze(0).expand(hidden_states.shape[0], -1)
        # Combine query attention with learned attention: (batch, seq_len)
        query_scores = torch.bmm(hidden_states, query.unsqueeze(-1)).squeeze(-1)
        learned_scores = self.attn(hidden_states).squeeze(-1)
        attn_scores = query_scores + learned_scores

        if attention_mask is not None:
            # Mask out padding tokens with large negative value
            attn_scores.masked_fill_(attention_mask == 0, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, seq_len)

        # Weighted sum: (batch, d_model)
        pooled = torch.sum(hidden_states * attn_weights.unsqueeze(-1), dim=1)

        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)
        output = self.proj(pooled)  # (batch, latent_dim)

        return output
