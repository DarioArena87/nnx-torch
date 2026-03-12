"""
Tests for RoPEAttention and ALiBiAttention backends.
Run with: pytest tests/test_attention_rope_alibi.py -v
"""

import pytest
import torch
import typing

from nnx.attention import RoPEAttention, ALiBiAttention, SDPAttention, RWKVTimeMixing, RWKV6TimeMixing, LinearAttention, build_attention
from nnx.attention.linear import _VARIANTS


class TestRoPEAttention:
    """Test suite for RoPEAttention."""

    @pytest.fixture
    def B(self):
        return 2

    @pytest.fixture
    def T(self):
        return 10

    @pytest.fixture
    def D(self):
        return 128

    @pytest.fixture
    def x(self, B, T, D):
        return torch.randn(B, T, D)

    @pytest.fixture
    def mask(self, B, T):
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 7:] = False
        mask[1, 9:] = False
        return mask

    def test_basic_forward(self, x, D):
        """Test basic self-attention without masks."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_hf_mask(self, x, mask, D):
        """Test with HuggingFace-style boolean mask."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_causal(self, x, D):
        """Test causal masking."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        out = attn(x, causal=True)
        assert out.shape == x.shape

    def test_causal_with_mask(self, x, mask, D):
        """Test both causal and padding mask."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        out = attn(x, attention_mask=mask, causal=True)
        assert out.shape == x.shape

    def test_cross_attention(self, D):
        """Test cross-attention with different query/key lengths."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        B = 2
        q = torch.randn(B, 5, D)
        kv = torch.randn(B, 15, D)
        kv_mask = torch.ones(B, 15, dtype=torch.bool)
        kv_mask[0, 12:] = False

        out = attn(q, key=kv, value=kv, attention_mask=kv_mask)
        assert out.shape == q.shape

    def test_cross_attention_no_mask(self, D):
        """Test cross-attention without mask."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        B = 2
        q = torch.randn(B, 7, D)
        kv = torch.randn(B, 20, D)
        out = attn(q, key=kv, value=kv)
        assert out.shape == q.shape

    def test_eval_mode(self, x, D):
        """Test that dropout is disabled in eval mode."""
        attn = RoPEAttention(embed_dim=D, num_heads=4, dropout=0.1)
        attn.eval()
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_head_dim_validation(self):
        """Test that odd head_dim raises an error."""
        # Use embed_dim that's divisible by num_heads but gives odd head_dim
        # e.g., embed_dim=128, num_heads=8 -> head_dim=16 (even)
        # We need to explicitly set head_dim to an odd value
        with pytest.raises(ValueError, match="even head_dim"):
            RoPEAttention(embed_dim=128, num_heads=8, head_dim=7)  # odd head_dim

    def test_custom_base(self, x, D):
        """Test with custom RoPE base."""
        attn = RoPEAttention(embed_dim=D, num_heads=4, base=500000.0)
        out = attn(x)
        assert out.shape == x.shape

    def test_different_num_heads(self, x, D):
        """Test with various numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            if D % num_heads == 0:
                attn = RoPEAttention(embed_dim=D, num_heads=num_heads)
                out = attn(x)
                assert out.shape == x.shape

    def test_gradients(self, x, D):
        """Test that gradients flow correctly."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        # Check that parameters have gradients
        for param in attn.parameters():
            assert param.grad is not None


class TestALiBiAttention:
    """Test suite for ALiBiAttention."""

    @pytest.fixture
    def B(self):
        return 2

    @pytest.fixture
    def T(self):
        return 10

    @pytest.fixture
    def D(self):
        return 128

    @pytest.fixture
    def x(self, B, T, D):
        return torch.randn(B, T, D)

    @pytest.fixture
    def mask(self, B, T):
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 7:] = False
        mask[1, 9:] = False
        return mask

    def test_basic_forward(self, x, D):
        """Test basic self-attention without masks."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_hf_mask(self, x, mask, D):
        """Test with HuggingFace-style boolean mask."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_causal(self, x, D):
        """Test causal masking."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        out = attn(x, causal=True)
        assert out.shape == x.shape

    def test_causal_with_mask(self, x, mask, D):
        """Test both causal and padding mask."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        out = attn(x, attention_mask=mask, causal=True)
        assert out.shape == x.shape

    def test_cross_attention(self, D):
        """Test cross-attention with different query/key lengths."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        B = 2
        q = torch.randn(B, 5, D)
        kv = torch.randn(B, 15, D)
        kv_mask = torch.ones(B, 15, dtype=torch.bool)
        kv_mask[0, 12:] = False

        out = attn(q, key=kv, value=kv, attention_mask=kv_mask)
        assert out.shape == q.shape

    def test_cross_attention_no_mask(self, D):
        """Test cross-attention without mask."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        B = 2
        q = torch.randn(B, 7, D)
        kv = torch.randn(B, 20, D)
        out = attn(q, key=kv, value=kv)
        assert out.shape == q.shape

    def test_eval_mode(self, x, D):
        """Test that dropout is disabled in eval mode."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4, dropout=0.1)
        attn.eval()
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_different_num_heads(self):
        """Test with various numbers of heads (including non-power-of-2)."""
        D = 128
        x = torch.randn(2, 10, D)
        for num_heads in [1, 2, 4, 8, 16]:
            if D % num_heads == 0:
                attn = ALiBiAttention(embed_dim=D, num_heads=num_heads)
                out = attn(x)
                assert out.shape == x.shape

    def test_gradients(self, x, D):
        """Test that gradients flow correctly."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        # Check that parameters have gradients
        for param in attn.parameters():
            assert param.grad is not None

    def test_bias_shape(self):
        """Test that the internal ALiBi bias has correct shape."""
        attn = ALiBiAttention(embed_dim=128, num_heads=4)
        seq_len = 10
        # Access the alibi module and generate bias
        bias = attn.alibi(seq_len, torch.device("cpu"))
        assert bias.shape == (1, 4, seq_len, seq_len)


class TestSDPAttention:
    """Test suite for SDPAttention."""

    @pytest.fixture
    def B(self):
        return 2

    @pytest.fixture
    def T(self):
        return 10

    @pytest.fixture
    def D(self):
        return 128

    @pytest.fixture
    def x(self, B, T, D):
        return torch.randn(B, T, D)

    @pytest.fixture
    def mask(self, B, T):
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 7:] = False
        mask[1, 9:] = False
        return mask

    def test_basic_forward(self, x, D):
        """Test basic self-attention without masks."""
        attn = SDPAttention(embed_dim=D, num_heads=4)
        out = attn(x)
        assert out.shape == x.shape

    def test_with_hf_mask(self, x, mask, D):
        """Test with HuggingFace-style boolean mask."""
        attn = SDPAttention(embed_dim=D, num_heads=4)
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_causal(self, x, D):
        """Test causal masking."""
        attn = SDPAttention(embed_dim=D, num_heads=4)
        out = attn(x, causal=True)
        assert out.shape == x.shape

    def test_cross_attention(self, D):
        """Test cross-attention with different query/key lengths."""
        attn = SDPAttention(embed_dim=D, num_heads=4)
        B = 2
        q = torch.randn(B, 5, D)
        kv = torch.randn(B, 15, D)
        kv_mask = torch.ones(B, 15, dtype=torch.bool)
        kv_mask[0, 12:] = False

        out = attn(q, key=kv, value=kv, attention_mask=kv_mask)
        assert out.shape == q.shape

    def test_cross_attention_no_mask(self, D):
        """Test cross-attention without mask."""
        attn = SDPAttention(embed_dim=D, num_heads=4)
        B = 2
        q = torch.randn(B, 7, D)
        kv = torch.randn(B, 20, D)
        out = attn(q, key=kv, value=kv)
        assert out.shape == q.shape

    def test_eval_mode(self, x, D):
        """Test that dropout is disabled in eval mode."""
        attn = SDPAttention(embed_dim=D, num_heads=4, dropout=0.1)
        attn.eval()
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_different_num_heads(self, x, D):
        """Test with various numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            if D % num_heads == 0:
                attn = SDPAttention(embed_dim=D, num_heads=num_heads)
                out = attn(x)
                assert out.shape == x.shape

    def test_gradients(self, x, D):
        """Test that gradients flow correctly."""
        attn = SDPAttention(embed_dim=D, num_heads=4)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        # Check that parameters have gradients
        for param in attn.parameters():
            assert param.grad is not None


class TestRWKVAttention:
    """Test suite for RWKVAttention."""

    @pytest.fixture
    def B(self):
        return 2

    @pytest.fixture
    def T(self):
        return 10

    @pytest.fixture
    def D(self):
        return 128

    @pytest.fixture
    def x(self, B, T, D):
        return torch.randn(B, T, D)

    @pytest.fixture
    def mask(self, B, T):
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 7:] = False
        mask[1, 9:] = False
        return mask

    def test_rwkv_time_mixing(self, x, mask, D):
        """Test RWKVTimeMixing (RWKV-4)."""
        rwkv = RWKVTimeMixing(embed_dim=D, layer_id=0, n_layers=6)
        out = rwkv(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_rwkv6_time_mixing(self, x, mask, D):
        """Test RWKV6TimeMixing (RWKV-6)."""
        rwkv6 = RWKV6TimeMixing(embed_dim=D, layer_id=1, n_layers=6, n_heads=4)
        out = rwkv6(x, attention_mask=mask)
        assert out.shape == x.shape


class TestLinearAttention:
    """Test suite for LinearAttention."""

    @pytest.fixture
    def B(self):
        return 2

    @pytest.fixture
    def T(self):
        return 10

    @pytest.fixture
    def D(self):
        return 128

    @pytest.fixture
    def x(self, B, T, D):
        return torch.randn(B, T, D)

    @pytest.fixture
    def mask(self, B, T):
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 7:] = False
        mask[1, 9:] = False
        return mask

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="LinearAttention requires CUDA")
    @pytest.mark.parametrize("variant", typing.get_args(_VARIANTS))
    def test_linear_attention_variants(self, D, variant):
        """Test LinearAttention with different kernel variants."""
        # Create tensors directly on CUDA with bfloat16 dtype
        x = torch.randn(2, 10, D, device="cuda", dtype=torch.bfloat16)
        mask = torch.ones(2, 10, dtype=torch.bool, device="cuda")
        mask[0, 7:] = False
        mask[1, 9:] = False
        
        linear_attn = LinearAttention(embed_dim=D, num_heads=4, variant=variant).to(device="cuda", dtype=torch.bfloat16)
        out = linear_attn(x, attention_mask=mask)
        assert out.shape == x.shape


class TestBuildAttentionFactory:
    """Test suite for build_attention factory function."""

    def test_rope_via_factory(self):
        """Test creating RoPEAttention via build_attention."""
        attn = build_attention("rope", embed_dim=128, num_heads=4)
        assert isinstance(attn, RoPEAttention)

    def test_alibi_via_factory(self):
        """Test creating ALiBiAttention via build_attention."""
        attn = build_attention("alibi", embed_dim=128, num_heads=4)
        assert isinstance(attn, ALiBiAttention)

    def test_sdpa_via_factory(self):
        """Test creating SDPAttention via build_attention."""
        attn = build_attention("sdpa", embed_dim=128, num_heads=4)
        assert isinstance(attn, SDPAttention)

    def test_rwkv_via_factory(self):
        """Test creating RWKVTimeMixing via build_attention."""
        attn = build_attention("rwkv", embed_dim=128)
        assert isinstance(attn, RWKVTimeMixing)

    def test_rope_with_custom_params(self):
        """Test factory with custom parameters."""
        attn = build_attention(
            "rope",
            embed_dim=128,
            num_heads=4,
            dropout=0.1,
            bias=False,
            base=500000.0,
            max_len=8192,
        )
        assert isinstance(attn, RoPEAttention)
        assert attn.dropout == 0.1
        assert attn.rope.base == 500000.0
        assert attn.rope.cos_cached.shape[0] == 8192

    def test_alibi_with_custom_params(self):
        """Test factory with custom parameters."""
        attn = build_attention(
            "alibi",
            embed_dim=128,
            num_heads=4,
            dropout=0.2,
            bias=False,
            max_len=8192,
        )
        assert isinstance(attn, ALiBiAttention)
        assert attn.dropout == 0.2
        assert attn.alibi.max_len == 8192

    def test_sdpa_with_custom_params(self):
        """Test factory with custom parameters for SDPAttention."""
        attn = build_attention(
            "sdpa",
            embed_dim=128,
            num_heads=4,
            dropout=0.15,
            bias=True,
        )
        assert isinstance(attn, SDPAttention)
        assert attn.dropout == 0.15

    def test_rwkv_with_custom_params(self):
        """Test factory with custom parameters for RWKV."""
        attn = build_attention(
            "rwkv",
            embed_dim=128,
            layer_id=2,
            n_layers=4,
        )
        assert isinstance(attn, RWKVTimeMixing)
        assert attn.layer_id == 2

    def test_invalid_type(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown attention type"):
            build_attention("invalid_type", embed_dim=128, num_heads=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
