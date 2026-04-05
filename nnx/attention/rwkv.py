"""
RWKV Attention backend.

Implements the RWKV "WKV" time-mixing mechanism in pure PyTorch with
optional CUDA kernels from the official RWKV repository.

Two variants are provided:

  ``RWKVTimeMixing``
      The core WKV time-mixing layer (RWKV-4 style).  This is the
      recurrent attention replacement — O(T) in time and memory.

  ``RWKV6TimeMixing``
      The data-dependent (DDlerp) time-mixing from RWKV-6 / Eagle-7B.

Both layers follow the same ``BaseAttention``-compatible ``forward``
interface, though the QKV semantics differ from standard attention.

References:
    RWKV-4: https://arxiv.org/abs/2305.13048
    RWKV-6: https://arxiv.org/abs/2404.05892
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


class RWKVTimeMixing(nn.Module):
    """
    RWKV-4 time-mixing (WKV) layer.

    This is a *drop-in replacement for self-attention* in a Transformer
    block.  The receptance, key, and value vectors are computed from the
    input and the previous token (via learnable interpolation), and the
    output is produced by a recurrent WKV operator.

    Args:
        embed_dim:  Model width.
        layer_id:   Index of this layer (used to initialise time-decay).
        n_layers:   Total number of layers (used for decay init).
        head_size:  Head dimension for grouped WKV (RWKV-5+).
                    If None, defaults to a single-head WKV.
        use_recurrent_kernel: If True and ``fast_linear_attention`` package
            is available, use optimized CUDA kernels. Falls back to Python
            loop when not available.
        chunk_size: Size of chunks for chunked processing. Default: 64.

    Note on ``attention_mask``:
        The WKV recurrence processes the sequence left-to-right.
        Padding tokens are zeroed in r/k/v before the recurrence.
        Causal masking is intrinsic — future tokens cannot attend to
        past tokens by construction.
    """

    def __init__(
        self,
        embed_dim: int,
        layer_id: int = 0,
        n_layers: int = 12,
        head_size: Optional[int] = None,
        use_recurrent_kernel: bool = False,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.layer_id = layer_id
        self.head_size = head_size or embed_dim

        # --- Learnable time-shift mix ratios ---
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, embed_dim))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, embed_dim))

        # --- Learnable time-decay and first-token bonus ---
        # Initialisation follows the RWKV-4 paper
        with torch.no_grad():
            ratio_0_to_1 = layer_id / max(n_layers - 1, 1)
            ratio_1_to_almost_0 = 1.0 - (layer_id / n_layers)

            decay = torch.ones(embed_dim)
            for i in range(embed_dim):
                decay[i] = -5 + 8 * (i / (embed_dim - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay)

            zigzag = (torch.arange(embed_dim) % 3 - 1) * ratio_1_to_almost_0
            self.time_first = nn.Parameter(torch.ones(embed_dim) * torch.log(torch.tensor(0.3)) + zigzag)

        # --- Projections ---
        self.key = nn.Linear(embed_dim, embed_dim, bias=False)
        self.value = nn.Linear(embed_dim, embed_dim, bias=False)
        self.receptance = nn.Linear(embed_dim, embed_dim, bias=False)
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln_x = nn.LayerNorm(embed_dim)
        
        # W1: FLA kernel integration
        self.use_recurrent_kernel = use_recurrent_kernel
        self.chunk_size = chunk_size
        self._fla_kernel = None
        
        if use_recurrent_kernel:
            self._try_import_fla_kernel()

    def _try_import_fla_kernel(self) -> None:
        """W1: Try to import FLA RWKV kernel, fall back gracefully."""
        try:
            from fla.ops.rwkv import rwkv4
            self._fla_kernel = rwkv4
        except ImportError:
            # FLA not available, will use Python fallback
            self._fla_kernel = None

    def _wkv_chunked(
        self,
        w: torch.Tensor,
        u: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        W1: Chunked WKV computation for better memory efficiency.
        
        Processes the sequence in chunks and accumulates state.
        """
        B, T, D = k.shape
        chunk_size = self.chunk_size
        out = torch.zeros_like(v)
        
        # Initialize state
        aa = torch.zeros(B, D, device=k.device, dtype=k.dtype)
        bb = torch.zeros(B, D, device=k.device, dtype=k.dtype)
        pp = torch.full((B, D), -1e38, device=k.device, dtype=k.dtype)
        
        exp_w = torch.exp(-torch.exp(w))
        
        num_chunks = (T + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, T)
            
            for t in range(start, end):
                kt = k[:, t, :]
                vt = v[:, t, :]
                
                qt = torch.maximum(pp, kt + u)
                e1 = torch.exp(pp - qt)
                e2 = torch.exp(kt + u - qt)
                wkv = (e1 * aa + e2 * vt) / (e1 * bb + e2)
                
                pp_new = torch.maximum(pp + exp_w.log(), kt)
                e1 = torch.exp(pp - pp_new)
                e2 = torch.exp(kt - pp_new)
                aa = e1 * aa + e2 * vt
                bb = e1 * bb + e2
                pp = pp_new
                
                out[:, t, :] = wkv
        
        return out

    def _wkv_pytorch(
        self,
        w: torch.Tensor,  # time-decay  (D,)
        u: torch.Tensor,  # time-first  (D,)
        k: torch.Tensor,  # (B, T, D)
        v: torch.Tensor,  # (B, T, D)
    ) -> torch.Tensor:
        """Pure-PyTorch WKV recurrence — correct but not fused."""
        B, T, D = k.shape
        out = torch.zeros_like(v)
        aa = torch.zeros(B, D, device=k.device, dtype=k.dtype)
        bb = torch.zeros(B, D, device=k.device, dtype=k.dtype)
        pp = torch.full((B, D), -1e38, device=k.device, dtype=k.dtype)

        exp_w = torch.exp(-torch.exp(w))  # per-channel decay (D,)

        for t in range(T):
            kt = k[:, t, :]  # (B, D)
            vt = v[:, t, :]  # (B, D)

            # "first" step: include u bonus
            qt = torch.maximum(pp, kt + u)
            e1 = torch.exp(pp - qt)
            e2 = torch.exp(kt + u - qt)
            wkv = (e1 * aa + e2 * vt) / (e1 * bb + e2)

            # Update state
            pp_new = torch.maximum(pp + exp_w.log(), kt)
            e1 = torch.exp(pp - pp_new)
            e2 = torch.exp(kt - pp_new)
            aa = e1 * aa + e2 * vt
            bb = e1 * bb + e2
            pp = pp_new

            out[:, t, :] = wkv

        return out

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, T, D) input.
            attention_mask: HF-style (B, T) mask — 1=real, 0=padding.
            causal:         Ignored (RWKV is always causal).

        Returns:
            (B, T, D)
        """
        B, T, D = x.shape

        # Time-shift: mix current token with the previous one
        x_shifted = F.pad(x, (0, 0, 1, -1))  # shift right by 1
        xk = x * self.time_mix_k + x_shifted * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + x_shifted * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + x_shifted * (1 - self.time_mix_r)

        k = self.key(xk)
        v = self.value(xv)
        r = torch.sigmoid(self.receptance(xr))

        # Zero-out padding positions
        if attention_mask is not None:
            pad = attention_mask.bool().unsqueeze(-1)  # (B, T, 1)
            k = k * pad
            v = v * pad

        # W1: Use FLA kernel if available, otherwise use Python implementation
        if self.use_recurrent_kernel and self._fla_kernel is not None:
            # FLA kernel expects (B, T, H, D) format
            k_ = k.unsqueeze(2)  # (B, T, 1, D)
            v_ = v.unsqueeze(2)  # (B, T, 1, D)
            wkv = self._fla_kernel(k_, v_, self.time_decay, self.time_first)
            wkv = wkv.squeeze(2)  # (B, T, D)
        elif self.chunk_size > 0 and T > self.chunk_size:
            # Use chunked Python implementation for long sequences
            wkv = self._wkv_chunked(self.time_decay, self.time_first, k, v)
        else:
            wkv = self._wkv_pytorch(self.time_decay, self.time_first, k, v)
        
        rwkv = r * self.ln_x(wkv)
        return self.output(rwkv)


class RWKV6TimeMixing(nn.Module):
    """
    RWKV-6 / Eagle data-dependent time-mixing (DDlerp).

    An upgraded version of :class:`RWKVTimeMixing` where the interpolation
    ratios are computed dynamically from the input (Lerpable parameters
    become input-dependent via a small linear layer), enabling the model
    to vary its short vs. long context blending on a per-token basis.

    Args:
        embed_dim:  Model width.
        layer_id:   Index of this layer.
        n_layers:   Total number of layers.
        n_heads:    Number of WKV heads (RWKV-6 uses multi-head WKV).
        use_recurrent_kernel: If True and ``fast_linear_attention`` package
            is available, use optimized CUDA kernels. Falls back to Python
            loop when not available.
        chunk_size: Size of chunks for chunked processing. Default: 64.
    """

    def __init__(
        self,
        embed_dim: int,
        layer_id: int = 0,
        n_layers: int = 12,
        n_heads: int = 1,
        use_recurrent_kernel: bool = False,
        chunk_size: int = 64,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        # DDlerp: a small MLP maps x_mean → per-channel mix ratios
        self.ddlerp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.Tanh(),
        )

        self.time_mix_r = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.5)
        self.time_mix_w = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.5)
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.5)
        self.time_mix_g = nn.Parameter(torch.ones(1, 1, embed_dim) * 0.5)

        # Decay — initialised similarly to RWKV-4
        with torch.no_grad():
            ratio = layer_id / max(n_layers - 1, 1)
            decay = torch.ones(n_heads, self.head_dim)
            for h in range(n_heads):
                for i in range(self.head_dim):
                    decay[h, i] = -5 + 8 * (i / (self.head_dim - 1)) ** (0.7 + 1.3 * ratio)
            self.time_decay = nn.Parameter(decay)

        self.r_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.w_proj = nn.Linear(embed_dim, embed_dim, bias=False)  # per-token decay
        self.output = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln_x = nn.GroupNorm(n_heads, embed_dim, eps=64e-5)
        
        # W1: FLA kernel integration
        self.use_recurrent_kernel = use_recurrent_kernel
        self.chunk_size = chunk_size
        self._fla_kernel = None
        
        if use_recurrent_kernel:
            self._try_import_fla_kernel()

    def _try_import_fla_kernel(self) -> None:
        """W1: Try to import FLA RWKV6 kernel, fall back gracefully."""
        try:
            from fla.ops.rwkv6 import rwkv6
            self._fla_kernel = rwkv6
        except ImportError:
            # FLA not available, will use Python fallback
            self._fla_kernel = None

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal: bool = True,
    ) -> torch.Tensor:
        B, T, D = x.shape
        x_shifted = F.pad(x, (0, 0, 1, -1))

        # Data-dependent interpolation coefficients
        dd = torch.sigmoid(self.ddlerp(x_shifted))  # (B, T, D)

        def lerp(mix, base):
            return x * (mix + dd * (1 - mix)) + x_shifted * (1 - mix - dd * (1 - mix))

        xr = lerp(self.time_mix_r, x)
        xk = lerp(self.time_mix_k, x)
        xv = lerp(self.time_mix_v, x)
        xg = lerp(self.time_mix_g, x)
        xw = lerp(self.time_mix_w, x)

        r = self.r_proj(xr)
        k = self.k_proj(xk)
        v = self.v_proj(xv)
        g = F.silu(self.g_proj(xg))

        # Per-token log-decay: softplus so decay stays negative
        w = -F.softplus(-(self.w_proj(xw))) - 0.5  # (B, T, D)

        # Zero padding
        if attention_mask is not None:
            pad = attention_mask.bool().unsqueeze(-1)
            k, v = k * pad, v * pad

        # W1: Use FLA kernel if available, otherwise use Python loop
        H, Dh = self.n_heads, self.head_dim
        
        if self.use_recurrent_kernel and self._fla_kernel is not None:
            # FLA kernel expects (B, T, H, D) format
            r_ = rearrange(r, '... t (h d) -> ... t h d', h=H, d=Dh)
            k_ = rearrange(k, '... t (h d) -> ... t h d', h=H, d=Dh)
            v_ = rearrange(v, '... t (h d) -> ... t h d', h=H, d=Dh)
            w_ = rearrange(w, '... t (h d) -> ... t h d', h=H, d=Dh)
            
            # Call FLA RWKV6 kernel
            out_seq = self._fla_kernel(r_, k_, v_, w_, self.time_decay)
            out = rearrange(out_seq, 'b t h d -> b t (h d)', h=H, d=Dh)
        else:
            # Simple sequential WKV-6 (full CUDA kernel in the fla library)
            r_ = rearrange(r, '... t (h d) -> ... t h d', h=H, d=Dh)
            k_ = rearrange(k, '... t (h d) -> ... t h d', h=H, d=Dh)
            v_ = rearrange(v, '... t (h d) -> ... t h d', h=H, d=Dh)
            w_ = rearrange(w, '... t (h d) -> ... t h d', h=H, d=Dh)

            out = torch.zeros_like(v_)
            state = torch.zeros(B, H, Dh, Dh, device=x.device, dtype=x.dtype)

            for t in range(T):
                kk = k_[:, t, :, :, None]  # (B,H,Dh,1)
                vv = v_[:, t, :, None, :]  # (B,H,1,Dh)
                rr = r_[:, t, :, :, None]  # (B,H,Dh,1)
                ww = torch.exp(w_[:, t, :, :])  # (B,H,Dh) decay per dim
                state = state * ww.unsqueeze(-1) + kk * vv
                out[:, t] = (state @ rr).squeeze(-1)  # (B,H,Dh)

            out = rearrange(out, 'b t h d -> b t (h d)', h=H, d=Dh)
        
        out = rearrange(self.ln_x(rearrange(out, 'b t d -> (b t) d')), '(b t) d -> b t d', b=B, t=T)
        return self.output(out * g)
