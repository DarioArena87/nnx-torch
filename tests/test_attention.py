"""
Tests for attention modules.
Run with: pytest tests/test_attention.py -v
"""

import pytest
import torch
import warnings

from nnx.attention import (
    RoPEAttention,
    ALiBiAttention,
    SDPAttention,
    RWKVTimeMixing,
    RWKV6TimeMixing,
    GLAAttention,
    DeltaAttention,
    BasedAttention,
    RetentionAttention,
    build_attention,
)


# ============================================================================
# RoPEAttention Tests
# ============================================================================

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

    def test_with_explicit_position_ids(self, x, D):
        """Test with explicit position_ids (non-sequential)."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        B, T = x.shape[:2]
        # Create non-sequential position IDs (e.g., chunked sequences)
        position_ids = torch.arange(1000, 1000 + T).unsqueeze(0).expand(B, -1)
        out = attn(x, position_ids=position_ids)
        assert out.shape == x.shape

    def test_position_ids_vs_sequential(self, D):
        """Test that explicit position_ids produce different results vs sequential."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        x = torch.randn(2, 10, D)
        
        # Without position_ids (sequential positions 0,1,2,...)
        out_seq = attn(x)
        
        # With explicit position_ids starting at offset
        position_ids = torch.arange(100, 110).unsqueeze(0).expand(2, -1)
        out_pos = attn(x, position_ids=position_ids)
        
        # Results should differ due to different rotations
        assert not torch.allclose(out_seq, out_pos)

    def test_batch_varying_position_ids(self, D):
        """Test with different position_ids for each batch element."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        x = torch.randn(2, 10, D)
        # Different starting positions for each batch
        position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]])
        out = attn(x, position_ids=position_ids)
        assert out.shape == x.shape

    def test_position_ids_single_token(self, D):
        """Test with single token sequence and position_ids."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        x = torch.randn(2, 1, D)
        position_ids = torch.tensor([[100], [200]])
        out = attn(x, position_ids=position_ids)
        assert out.shape == x.shape

    def test_position_ids_large_offset(self, D):
        """Test with very large position offset (e.g., 1e6)."""
        # Increase max_len to accommodate large position IDs
        attn = RoPEAttention(embed_dim=D, num_heads=4, max_len=1_000_010)
        x = torch.randn(2, 10, D)
        position_ids = torch.arange(1_000_000, 1_000_010).unsqueeze(0).expand(2, -1)
        out = attn(x, position_ids=position_ids)
        assert out.shape == x.shape
        # Verify no NaN/Inf
        assert torch.all(torch.isfinite(out))

    def test_position_ids_with_causal(self, D):
        """Test position_ids combined with causal masking."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        x = torch.randn(2, 10, D)
        position_ids = torch.arange(100, 110).unsqueeze(0).expand(2, -1)
        out = attn(x, causal=True, position_ids=position_ids)
        assert out.shape == x.shape

    def test_position_ids_with_padding_mask(self, D):
        """Test position_ids with padding mask."""
        attn = RoPEAttention(embed_dim=D, num_heads=4)
        x = torch.randn(2, 10, D)
        mask = torch.ones(2, 10, dtype=torch.bool)
        mask[0, 7:] = False
        mask[1, 9:] = False
        position_ids = torch.arange(100, 110).unsqueeze(0).expand(2, -1)
        out = attn(x, attention_mask=mask, position_ids=position_ids)
        assert out.shape == x.shape


# ============================================================================
# ALiBiAttention Tests
# ============================================================================

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

    def test_with_explicit_position_ids(self, x, D):
        """Test with explicit position_ids (non-sequential)."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        B, T = x.shape[:2]
        # Create non-sequential position IDs (e.g., chunked sequences)
        position_ids = torch.arange(1000, 1000 + T).unsqueeze(0).expand(B, -1)
        out = attn(x, position_ids=position_ids)
        assert out.shape == x.shape

    def test_position_ids_offset_equivalence(self, D):
        """Test that ALiBi with offset position_ids gives same result as sequential.
        
        ALiBi only depends on relative distances, so absolute position offset
        should not affect the output (for self-attention with same positions).
        """
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        x = torch.randn(2, 10, D)
        
        # Without position_ids (sequential positions 0,1,2,...)
        out_seq = attn(x)
        
        # With explicit position_ids starting at offset
        position_ids = torch.arange(100, 110).unsqueeze(0).expand(2, -1)
        out_pos = attn(x, position_ids=position_ids)
        
        # Results should be identical because relative distances are the same
        assert torch.allclose(out_seq, out_pos)

    def test_batch_varying_position_ids(self, D):
        """Test with different position_ids for each batch element."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        x = torch.randn(2, 10, D)
        # Different starting positions for each batch
        position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]])
        out = attn(x, position_ids=position_ids)
        assert out.shape == x.shape

    def test_position_ids_cross_attention(self, D):
        """Test position_ids with cross-attention (different Q and K lengths)."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        B = 2
        q = torch.randn(B, 5, D)
        kv = torch.randn(B, 15, D)
        # Position IDs for both Q and K (they share the same position space)
        position_ids = torch.arange(50, 50 + 15).unsqueeze(0).expand(B, -1)
        out = attn(q, key=kv, value=kv, position_ids=position_ids)
        assert out.shape == q.shape

    def test_position_ids_single_token(self, D):
        """Test with single token sequence and position_ids."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        x = torch.randn(2, 1, D)
        position_ids = torch.tensor([[100], [200]])
        out = attn(x, position_ids=position_ids)
        assert out.shape == x.shape

    def test_position_ids_large_offset(self, D):
        """Test with very large position offset (e.g., 1e6)."""
        attn = ALiBiAttention(embed_dim=D, num_heads=4)
        x = torch.randn(2, 10, D)
        # Large offset should still work (ALiBi uses absolute positions)
        position_ids = torch.arange(1_000_000, 1_000_010).unsqueeze(0).expand(2, -1)
        out = attn(x, position_ids=position_ids)
        assert out.shape == x.shape
        # Verify no NaN/Inf
        assert torch.all(torch.isfinite(out))


# ============================================================================
# SDPAttention Tests
# ============================================================================

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


# ============================================================================
# RWKV Attention Tests
# ============================================================================

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


# ============================================================================
# Linear Attention Tests (Comprehensive per-variant testing)
# ============================================================================

class TestGLAAttention:
    """Test suite for GLAAttention (Gated Linear Attention)."""

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
        # GLA works on CUDA
        return torch.randn(B, T, D, dtype=torch.float32).cuda()

    @pytest.fixture
    def mask(self, B, T):
        mask = torch.ones(B, T, dtype=torch.bool).cuda()
        mask[0, 7:] = False
        mask[1, 9:] = False
        return mask

    def test_basic_forward(self, x, D):
        """Test basic self-attention without masks."""
        attn = GLAAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x)
        assert out.shape == x.shape

    def test_with_hf_mask(self, x, mask, D):
        """Test with HuggingFace-style boolean mask."""
        attn = GLAAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_causal(self, x, D):
        """Test causal masking (always on for linear attention)."""
        attn = GLAAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x, causal=True)
        assert out.shape == x.shape

    def test_cross_attention_same_length(self, D):
        """Test cross-attention with same query/key lengths."""
        attn = GLAAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 10, D).cuda()
        kv = torch.randn(B, 10, D).cuda()
        out = attn(q, key=kv, value=kv)
        assert out.shape == q.shape

    def test_cross_attention_different_lengths_raises(self, D):
        """Test that cross-attention with different lengths raises an error."""
        attn = GLAAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 5, D).cuda()
        kv = torch.randn(B, 15, D).cuda()
        with pytest.raises(ValueError, match="does not support cross-attention with different query/key lengths"):
            attn(q, key=kv, value=kv)

    def test_cross_attention_with_mask(self, D):
        """Test cross-attention with mask (same length)."""
        attn = GLAAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 10, D).cuda()
        kv = torch.randn(B, 10, D).cuda()
        kv_mask = torch.ones(B, 10, dtype=torch.bool).cuda()
        kv_mask[0, 8:] = False
        out = attn(q, key=kv, value=kv, attention_mask=kv_mask)
        assert out.shape == q.shape

    def test_eval_mode(self, x, D):
        """Test that behavior is deterministic in eval mode."""
        attn = GLAAttention(embed_dim=D, num_heads=4).cuda()
        attn.eval()
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_different_num_heads(self, D):
        """Test with various numbers of heads."""
        x = torch.randn(2, 10, D).cuda()
        for num_heads in [1, 2, 4, 8]:
            if D % num_heads == 0:
                attn = GLAAttention(embed_dim=D, num_heads=num_heads).cuda()
                out = attn(x)
                assert out.shape == x.shape

    def test_gradients(self, x, D):
        """Test that gradients flow correctly."""
        attn = GLAAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x)
        loss = out.sum()
        loss.backward()
        # Check that parameters have gradients
        for param in attn.parameters():
            assert param.grad is not None

    def test_expand_v_only(self, D):
        """Test with expanded value dimension (expand_v only, expand_k=1.0)."""
        # GLA requires expand_k=1.0 but can have expand_v != 1.0
        attn = GLAAttention(embed_dim=D, num_heads=4, expand_k=1.0, expand_v=2.0).cuda()
        x = torch.randn(2, 10, D).cuda()
        out = attn(x)
        assert out.shape == x.shape

    def test_no_warnings(self, x, D):
        """Test that no warnings are raised during forward pass."""
        attn = GLAAttention(embed_dim=D, num_heads=4).cuda()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = attn(x)
            # Check no warnings
            assert len(w) == 0, f"Unexpected warnings: {[str(warning.message) for warning in w]}"


class TestDeltaAttention:
    """Test suite for DeltaAttention (DeltaNet)."""

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
        # DeltaNet requires bfloat16
        return torch.randn(B, T, D, dtype=torch.bfloat16).cuda()

    @pytest.fixture
    def mask(self, B, T):
        mask = torch.ones(B, T, dtype=torch.bool).cuda()
        mask[0, 7:] = False
        mask[1, 9:] = False
        return mask

    def test_basic_forward(self, x, D):
        """Test basic self-attention without masks."""
        attn = DeltaAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x)
        assert out.shape == x.shape

    def test_with_hf_mask(self, x, mask, D):
        """Test with HuggingFace-style boolean mask."""
        attn = DeltaAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_causal(self, x, D):
        """Test causal masking."""
        attn = DeltaAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x, causal=True)
        assert out.shape == x.shape

    def test_cross_attention_same_length(self, D):
        """Test cross-attention with same query/key lengths."""
        attn = DeltaAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 10, D, dtype=torch.bfloat16).cuda()
        kv = torch.randn(B, 10, D, dtype=torch.bfloat16).cuda()
        out = attn(q, key=kv, value=kv)
        assert out.shape == q.shape

    def test_cross_attention_different_lengths_raises(self, D):
        """Test that cross-attention with different lengths raises an error."""
        attn = DeltaAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 5, D, dtype=torch.bfloat16).cuda()
        kv = torch.randn(B, 15, D, dtype=torch.bfloat16).cuda()
        with pytest.raises(ValueError, match="does not support cross-attention with different query/key lengths"):
            attn(q, key=kv, value=kv)

    def test_cross_attention_with_mask(self, D):
        """Test cross-attention with mask (same length)."""
        attn = DeltaAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 10, D, dtype=torch.bfloat16).cuda()
        kv = torch.randn(B, 10, D, dtype=torch.bfloat16).cuda()
        kv_mask = torch.ones(B, 10, dtype=torch.bool).cuda()
        kv_mask[0, 8:] = False
        out = attn(q, key=kv, value=kv, attention_mask=kv_mask)
        assert out.shape == q.shape

    def test_eval_mode(self, x, D):
        """Test that behavior is deterministic in eval mode."""
        attn = DeltaAttention(embed_dim=D, num_heads=4).cuda()
        attn.eval()
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_different_num_heads(self, D):
        """Test with various numbers of heads."""
        x = torch.randn(2, 10, D, dtype=torch.bfloat16).cuda()
        for num_heads in [1, 2, 4, 8]:
            if D % num_heads == 0:
                attn = DeltaAttention(embed_dim=D, num_heads=num_heads).cuda()
                out = attn(x)
                assert out.shape == x.shape

    def test_gradients(self, x, D):
        """Test that gradients flow correctly."""
        attn = DeltaAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x)
        loss = out.sum()
        loss.backward()
        for param in attn.parameters():
            assert param.grad is not None

    def test_expand_v_only(self, D):
        """Test with expanded value dimension (expand_v only, expand_k=1.0)."""
        # DeltaNet requires expand_k=1.0 but can have expand_v != 1.0
        attn = DeltaAttention(embed_dim=D, num_heads=4, expand_k=1.0, expand_v=2.0).cuda()
        x = torch.randn(2, 10, D, dtype=torch.bfloat16).cuda()
        out = attn(x)
        assert out.shape == x.shape

    def test_no_warnings(self, x, D):
        """Test that no warnings are raised during forward pass."""
        attn = DeltaAttention(embed_dim=D, num_heads=4).cuda()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = attn(x)
            assert len(w) == 0, f"Unexpected warnings: {[str(warning.message) for warning in w]}"


class TestBasedAttention:
    """Test suite for BasedAttention."""

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
        return torch.randn(B, T, D).cuda()

    @pytest.fixture
    def mask(self, B, T):
        mask = torch.ones(B, T, dtype=torch.bool).cuda()
        mask[0, 7:] = False
        mask[1, 9:] = False
        return mask

    def test_basic_forward(self, x, D):
        """Test basic self-attention without masks."""
        attn = BasedAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x)
        assert out.shape == x.shape

    def test_with_hf_mask(self, x, mask, D):
        """Test with HuggingFace-style boolean mask."""
        attn = BasedAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_causal(self, x, D):
        """Test causal masking."""
        attn = BasedAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x, causal=True)
        assert out.shape == x.shape

    def test_cross_attention_same_length(self, D):
        """Test cross-attention with same query/key lengths."""
        attn = BasedAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 10, D).cuda()
        kv = torch.randn(B, 10, D).cuda()
        out = attn(q, key=kv, value=kv)
        assert out.shape == q.shape

    def test_cross_attention_no_mask(self, D):
        """Test cross-attention without mask (same length)."""
        attn = BasedAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 10, D).cuda()
        kv = torch.randn(B, 10, D).cuda()
        out = attn(q, key=kv, value=kv)
        assert out.shape == q.shape

    def test_eval_mode(self, x, D):
        """Test that behavior is deterministic in eval mode."""
        attn = BasedAttention(embed_dim=D, num_heads=4).cuda()
        attn.eval()
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_different_num_heads_small(self):
        """Test with various numbers of heads (using smaller size to avoid OOM)."""
        D = 64  # Smaller
        x = torch.randn(1, 8, D).cuda()  # Smaller batch and seq
        for num_heads in [1, 2, 4]:
            if D % num_heads == 0:
                attn = BasedAttention(embed_dim=D, num_heads=num_heads).cuda()
                out = attn(x)
                assert out.shape == x.shape

    def test_gradients_small(self):
        """Test that gradients flow correctly (with smaller size)."""
        attn = BasedAttention(embed_dim=64, num_heads=4).cuda()
        x = torch.randn(1, 8, 64).cuda()
        out = attn(x)
        loss = out.sum()
        loss.backward()
        for param in attn.parameters():
            assert param.grad is not None

    def test_expand_kv(self, D):
        """Test with expanded key/value dimensions."""
        attn = BasedAttention(embed_dim=D, num_heads=4, expand_k=2.0, expand_v=2.0).cuda()
        x = torch.randn(2, 10, D).cuda()
        out = attn(x)
        assert out.shape == x.shape

    def test_no_warnings(self, x, D):
        """Test that no warnings are raised during forward pass."""
        attn = BasedAttention(embed_dim=D, num_heads=4).cuda()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = attn(x)
            assert len(w) == 0, f"Unexpected warnings: {[str(warning.message) for warning in w]}"

    def test_cross_attention_different_lengths_raises(self, D):
        """Test that cross-attention with different lengths raises an error."""
        attn = BasedAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 5, D).cuda()
        kv = torch.randn(B, 15, D).cuda()
        with pytest.raises(ValueError, match="does not support cross-attention with different query/key lengths"):
            attn(q, key=kv, value=kv)


class TestRetentionAttention:
    """Test suite for RetentionAttention."""

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
        return torch.randn(B, T, D).cuda()

    @pytest.fixture
    def mask(self, B, T):
        mask = torch.ones(B, T, dtype=torch.bool).cuda()
        mask[0, 7:] = False
        mask[1, 9:] = False
        return mask

    def test_basic_forward(self, x, D):
        """Test basic self-attention without masks."""
        attn = RetentionAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x)
        assert out.shape == x.shape

    def test_with_hf_mask(self, x, mask, D):
        """Test with HuggingFace-style boolean mask."""
        attn = RetentionAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_causal(self, x, D):
        """Test causal masking."""
        attn = RetentionAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x, causal=True)
        assert out.shape == x.shape

    def test_cross_attention_same_length(self, D):
        """Test cross-attention with same query/key lengths."""
        attn = RetentionAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 10, D).cuda()
        kv = torch.randn(B, 10, D).cuda()
        out = attn(q, key=kv, value=kv)
        assert out.shape == q.shape

    def test_cross_attention_no_mask(self, D):
        """Test cross-attention without mask (same length)."""
        attn = RetentionAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 10, D).cuda()
        kv = torch.randn(B, 10, D).cuda()
        out = attn(q, key=kv, value=kv)
        assert out.shape == q.shape

    def test_eval_mode(self, x, D):
        """Test that behavior is deterministic in eval mode."""
        attn = RetentionAttention(embed_dim=D, num_heads=4).cuda()
        attn.eval()
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_different_num_heads(self, D):
        """Test with various numbers of heads."""
        x = torch.randn(2, 10, D).cuda()
        for num_heads in [1, 2, 4, 8]:
            if D % num_heads == 0:
                attn = RetentionAttention(embed_dim=D, num_heads=num_heads).cuda()
                out = attn(x)
                assert out.shape == x.shape

    def test_gradients(self, x, D):
        """Test that gradients flow correctly."""
        attn = RetentionAttention(embed_dim=D, num_heads=4).cuda()
        out = attn(x)
        loss = out.sum()
        loss.backward()
        for param in attn.parameters():
            assert param.grad is not None

    def test_expand_kv(self, D):
        """Test with expanded key/value dimensions."""
        attn = RetentionAttention(embed_dim=D, num_heads=4, expand_k=2.0, expand_v=2.0).cuda()
        x = torch.randn(2, 10, D).cuda()
        out = attn(x)
        assert out.shape == x.shape

    def test_no_warnings(self, x, D):
        """Test that no warnings are raised during forward pass."""
        attn = RetentionAttention(embed_dim=D, num_heads=4).cuda()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = attn(x)
            assert len(w) == 0, f"Unexpected warnings: {[str(warning.message) for warning in w]}"

    def test_cross_attention_different_lengths_raises(self, D):
        """Test that cross-attention with different lengths raises an error."""
        attn = RetentionAttention(embed_dim=D, num_heads=4).cuda()
        B = 2
        q = torch.randn(B, 5, D).cuda()
        kv = torch.randn(B, 15, D).cuda()
        with pytest.raises(ValueError, match="does not support cross-attention with different query/key lengths"):
            attn(q, key=kv, value=kv)

# ============================================================================
# build_attention factory tests
# ============================================================================

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

    def test_gla_via_factory(self):
        """Test creating GLAAttention via build_attention."""
        attn = build_attention("gla", embed_dim=128, num_heads=4)
        assert isinstance(attn, GLAAttention)

    def test_delta_via_factory(self):
        """Test creating DeltaAttention via build_attention."""
        attn = build_attention("delta", embed_dim=128, num_heads=4)
        assert isinstance(attn, DeltaAttention)

    def test_based_via_factory(self):
        """Test creating BasedAttention via build_attention."""
        attn = build_attention("based", embed_dim=128, num_heads=4)
        assert isinstance(attn, BasedAttention)

    def test_retention_via_factory(self):
        """Test creating RetentionAttention via build_attention."""
        attn = build_attention("retention", embed_dim=128, num_heads=4)
        assert isinstance(attn, RetentionAttention)

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

    def test_gla_with_custom_params(self):
        """Test factory with custom parameters for GLA."""
        attn = build_attention(
            "gla",
            embed_dim=128,
            num_heads=4,
            expand_k=1.0,  # GLA requires expand_k=1.0
            expand_v=2.0,
        )
        assert isinstance(attn, GLAAttention)
        assert attn._k_dim == 32  # head_dim=32, expand_k=1
        assert attn._v_dim == 32 * 2

    def test_delta_with_custom_params(self):
        """Test factory with custom parameters for Delta."""
        attn = build_attention(
            "delta",
            embed_dim=128,
            num_heads=4,
            expand_k=1.0,  # Delta requires expand_k=1.0
            expand_v=1.5,
        )
        assert isinstance(attn, DeltaAttention)
        assert attn._k_dim == 32  # head_dim=32, expand_k=1
        assert attn._v_dim == int(32 * 1.5)

    def test_invalid_type(self):
        """Test that invalid type raises error."""
        with pytest.raises(ValueError, match="Unknown attention type"):
            build_attention("invalid_type", embed_dim=128, num_heads=4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
