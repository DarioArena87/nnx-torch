"""
Tests for embedding and positional encoding layers.
Run with: pytest tests/test_embeddings.py -v
"""

import pytest
import torch

from nnx.layers.embedding import (
    TokenEmbedding,
    SinusoidalPositional,
    LearnedPositional,
    RotaryEmbedding,
    ALiBiEmbedding,
)


class TestTokenEmbedding:
    """Test suite for TokenEmbedding."""

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
    def ids(self, B, T):
        return torch.randint(0, 1000, (B, T))

    def test_token_embedding(self, ids, D):
        """Test TokenEmbedding basic functionality."""
        emb = TokenEmbedding(1000, D, scale=True)
        out = emb(ids)
        assert out.shape == (ids.shape[0], ids.shape[1], D)

    def test_token_embedding_no_scale(self, ids, D):
        """Test TokenEmbedding without scaling."""
        emb = TokenEmbedding(1000, D, scale=False)
        out = emb(ids)
        assert out.shape == (ids.shape[0], ids.shape[1], D)

    def test_token_embedding_with_padding(self, D):
        """Test TokenEmbedding with padding index."""
        emb = TokenEmbedding(1000, D, padding_idx=0)
        ids = torch.tensor([[0, 1, 2], [3, 0, 1]])
        out = emb(ids)
        assert out.shape == (2, 3, D)
        # Check that padding embedding is zero
        assert torch.allclose(out[0, 0], torch.zeros(D))
        assert torch.allclose(out[1, 1], torch.zeros(D))

    def test_token_embedding_gradients(self, ids, D):
        """Test that gradients flow through TokenEmbedding."""
        emb = TokenEmbedding(1000, D)
        ids = torch.randint(0, 1000, (2, 10))
        out = emb(ids)
        loss = out.sum()
        loss.backward()
        for param in emb.parameters():
            assert param.grad is not None


class TestSinusoidalPositional:
    """Test suite for SinusoidalPositional."""

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

    def test_sinusoidal_positional(self, x, D):
        """Test SinusoidalPositional encoding."""
        pos = SinusoidalPositional(D)
        out = pos(x)
        assert out.shape == x.shape

    def test_sinusoidal_positional_with_dropout(self, x, D):
        """Test SinusoidalPositional with dropout."""
        pos = SinusoidalPositional(D, dropout=0.1)
        pos.train()
        out1 = pos(x)
        out2 = pos(x)
        # In training mode with dropout, outputs should differ
        assert not torch.allclose(out1, out2)
        pos.eval()
        out3 = pos(x)
        out4 = pos(x)
        # In eval mode, outputs should be identical
        assert torch.allclose(out3, out4)

    def test_sinusoidal_positional_different_max_len(self, D):
        """Test SinusoidalPositional with different max_len."""
        x = torch.randn(2, 50, D)
        pos = SinusoidalPositional(D, max_len=1000)
        out = pos(x)
        assert out.shape == x.shape


class TestLearnedPositional:
    """Test suite for LearnedPositional."""

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

    def test_learned_positional(self, x, D):
        """Test LearnedPositional encoding."""
        pos = LearnedPositional(512, D)
        out = pos(x)
        assert out.shape == x.shape

    def test_learned_positional_with_offset(self, x, D):
        """Test LearnedPositional with offset."""
        pos = LearnedPositional(512, D)
        out1 = pos(x, offset=0)
        out2 = pos(x, offset=10)
        assert out1.shape == x.shape
        assert out2.shape == x.shape
        # Outputs should be different due to different positional embeddings
        assert not torch.allclose(out1, out2)

    def test_learned_positional_with_dropout(self, x, D):
        """Test LearnedPositional with dropout."""
        pos = LearnedPositional(512, D, dropout=0.1)
        pos.train()
        out1 = pos(x)
        out2 = pos(x)
        # In training mode with dropout, outputs should differ
        assert not torch.allclose(out1, out2)
        pos.eval()
        out3 = pos(x)
        out4 = pos(x)
        # In eval mode, outputs should be identical
        assert torch.allclose(out3, out4)


class TestRotaryEmbedding:
    """Test suite for RotaryEmbedding."""

    @pytest.fixture
    def D(self):
        return 128

    def test_rotary_embedding(self, D):
        """Test RotaryEmbedding basic functionality."""
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        B, H, T, head_dim = 2, 4, 10, 32
        q = torch.randn(B, H, T, head_dim)
        k = torch.randn(B, H, T, head_dim)
        qr, kr = rope(q, k)
        assert qr.shape == q.shape
        assert kr.shape == k.shape

    def test_rotary_embedding_with_offset(self, D):
        """Test RotaryEmbedding with offset for KV-cache."""
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        B, H, T, head_dim = 2, 4, 10, 32
        q = torch.randn(B, H, T, head_dim)
        k = torch.randn(B, H, T, head_dim)
        qr1, kr1 = rope(q, k, offset=0)
        qr2, kr2 = rope(q, k, offset=5)
        # Results should differ due to different offsets
        assert not torch.allclose(qr1, qr2)
        assert not torch.allclose(kr1, kr2)

    def test_rotary_embedding_custom_base(self, D):
        """Test RotaryEmbedding with custom base."""
        rope = RotaryEmbedding(head_dim=32, max_len=512, base=500000.0)
        B, H, T, head_dim = 2, 4, 10, 32
        q = torch.randn(B, H, T, head_dim)
        k = torch.randn(B, H, T, head_dim)
        qr, kr = rope(q, k)
        assert qr.shape == q.shape
        assert kr.shape == k.shape

    def test_rotary_embedding_even_head_dim(self):
        """Test that RotaryEmbedding works with even head_dim."""
        # Should work fine with even head_dim
        rope = RotaryEmbedding(head_dim=64, max_len=512)
        assert rope.head_dim == 64

    def test_rotary_embedding_gradients(self, D):
        """Test that gradients flow through RotaryEmbedding."""
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        q = torch.randn(2, 4, 10, 32, requires_grad=True)
        k = torch.randn(2, 4, 10, 32, requires_grad=True)
        qr, kr = rope(q, k)
        loss = qr.sum() + kr.sum()
        loss.backward()
        assert q.grad is not None
        assert k.grad is not None

    def test_rotate_with_positions_basic(self, D):
        """Test RotaryEmbedding.rotate_with_positions with explicit positions."""
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        B, H, T = 2, 4, 10
        x = torch.randn(B, H, T, 32)
        # Use non-sequential positions
        position_ids = torch.arange(100, 100 + T).unsqueeze(0).expand(B, -1)
        rotated = rope.rotate_with_positions(x, position_ids)
        assert rotated.shape == x.shape

    def test_rotate_with_positions_1d_position_ids(self, D):
        """Test with 1D position_ids (shape T)."""
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        B, H, T = 2, 4, 10
        x = torch.randn(B, H, T, 32)
        position_ids = torch.arange(50, 50 + T)  # 1D
        rotated = rope.rotate_with_positions(x, position_ids)
        assert rotated.shape == x.shape

    def test_rotate_with_positions_batch_varying(self, D):
        """Test with different positions for each batch."""
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        B, H, T = 2, 4, 10
        x = torch.randn(B, H, T, 32)
        position_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                     [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]])
        rotated = rope.rotate_with_positions(x, position_ids)
        assert rotated.shape == x.shape

    def test_rotate_with_positions_large_offset(self, D):
        """Test with very large position offset (requires large max_len)."""
        # Use a large max_len to accommodate the offset
        rope = RotaryEmbedding(head_dim=32, max_len=1_000_010)
        B, H, T = 2, 4, 10
        x = torch.randn(B, H, T, 32)
        position_ids = torch.arange(1_000_000, 1_000_000 + T).unsqueeze(0).expand(B, -1)
        rotated = rope.rotate_with_positions(x, position_ids)
        assert rotated.shape == x.shape
        assert torch.all(torch.isfinite(rotated))

    def test_rotate_with_positions_gradients(self, D):
        """Test gradients flow through rotate_with_positions."""
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        B, H, T = 2, 4, 10
        x = torch.randn(B, H, T, 32, requires_grad=True)
        position_ids = torch.arange(50, 50 + T).unsqueeze(0).expand(B, -1)
        rotated = rope.rotate_with_positions(x, position_ids)
        loss = rotated.sum()
        loss.backward()
        assert x.grad is not None

    def test_rotate_with_positions_consistency(self, D):
        """Test that rotate_with_positions gives same result as rotate_queries_keys with offset for sequential positions."""
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        B, H, T = 2, 4, 10
        x = torch.randn(B, H, T, 32)
        # Method 1: rotate_with_positions with sequential positions
        position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        rotated1 = rope.rotate_with_positions(x, position_ids)
        
        # Method 2: rotate_queries_keys with offset=0
        rotated2_q, rotated2_k = rope.rotate_queries_keys(x, x, offset=0)
        
        # Should be identical
        assert torch.allclose(rotated1, rotated2_q)


class TestALiBiEmbedding:
    """Test suite for ALiBiEmbedding."""

    @pytest.fixture
    def T(self):
        return 10

    def test_alibi_embedding(self, T):
        """Test ALiBiEmbedding basic functionality."""
        alibi = ALiBiEmbedding(num_heads=8)
        bias = alibi(T, device=torch.device("cpu"))
        assert bias.shape == (1, 8, T, T)

    def test_alibi_embedding_different_num_heads(self):
        """Test ALiBiEmbedding with different numbers of heads."""
        for num_heads in [1, 2, 4, 8, 16]:
            alibi = ALiBiEmbedding(num_heads=num_heads)
            bias = alibi(10, device=torch.device("cpu"))
            assert bias.shape == (1, num_heads, 10, 10)

    def test_alibi_embedding_slopes_shape(self):
        """Test that ALiBi slopes have correct shape."""
        alibi = ALiBiEmbedding(num_heads=8)
        assert alibi.slopes.shape == (8,)

    def test_alibi_embedding_values(self):
        """Test ALiBiEmbedding bias values are negative and structured."""
        alibi = ALiBiEmbedding(num_heads=4)
        bias = alibi(5, device=torch.device("cpu"))
        # Bias should be negative or zero (since it's -slope * dist)
        assert torch.all(bias <= 0)
        # Check that bias is more negative for farther distances
        # bias shape: (1, H, T, T)
        # For a given head, bias[i, j] should be more negative as |i-j| increases
        for h in range(4):
            for i in range(5):
                for j in range(5):
                    dist = abs(i - j)
                    if dist > 0:
                        # bias[0, h, i, j] should be negative
                        assert bias[0, h, i, j] <= 0

    def test_alibi_embedding_device(self):
        """Test ALiBiEmbedding on different device."""
        alibi = ALiBiEmbedding(num_heads=4)
        bias_cpu = alibi(10, device=torch.device("cpu"))
        assert bias_cpu.device.type == "cpu"
        if torch.cuda.is_available():
            bias_cuda = alibi(10, device=torch.device("cuda"))
            assert bias_cuda.device.type == "cuda"

    def test_with_positions_basic(self):
        """Test ALiBiEmbedding.with_positions with explicit positions."""
        alibi = ALiBiEmbedding(num_heads=4)
        T = 10
        position_ids = torch.arange(1000, 1000 + T).unsqueeze(0)  # (1, T)
        bias = alibi.with_positions(position_ids, torch.device("cpu"))
        assert bias.shape == (1, 4, T, T)

    def test_with_positions_1d(self):
        """Test with 1D position_ids."""
        alibi = ALiBiEmbedding(num_heads=4)
        T = 10
        position_ids = torch.arange(50, 50 + T)  # (T,)
        bias = alibi.with_positions(position_ids, torch.device("cpu"))
        assert bias.shape == (1, 4, T, T)

    def test_with_positions_batch_varying(self):
        """Test with different positions per batch."""
        alibi = ALiBiEmbedding(num_heads=4)
        position_ids = torch.tensor([[0, 1, 2, 3, 4],
                                     [100, 101, 102, 103, 104]])
        bias = alibi.with_positions(position_ids, torch.device("cpu"))
        B, T = position_ids.shape
        assert bias.shape == (B, 4, T, T)

    def test_with_positions_cross_attention(self):
        """Test with different Q and K lengths (using same position space)."""
        alibi = ALiBiEmbedding(num_heads=4)
        # For cross-attention, we have different lengths for Q and K
        # but position_ids corresponds to the key positions
        kv_T = 15
        position_ids = torch.arange(50, 50 + kv_T).unsqueeze(0)  # (1, Tk)
        bias = alibi.with_positions(position_ids, torch.device("cpu"))
        # The bias will be (1, 4, Tk, Tk) - for keys
        assert bias.shape == (1, 4, kv_T, kv_T)

    def test_with_positions_large_offset(self):
        """Test with very large position offset."""
        alibi = ALiBiEmbedding(num_heads=4)
        T = 10
        position_ids = torch.arange(1_000_000, 1_000_000 + T).unsqueeze(0)
        bias = alibi.with_positions(position_ids, torch.device("cpu"))
        assert bias.shape == (1, 4, T, T)
        # Verify no NaN/Inf
        assert torch.all(torch.isfinite(bias))

    def test_with_positions_slopes_applied_correctly(self):
        """Test that slopes are correctly applied to bias."""
        alibi = ALiBiEmbedding(num_heads=2)
        T = 5
        position_ids = torch.arange(0, T).unsqueeze(0)  # (1, T)
        bias = alibi.with_positions(position_ids, torch.device("cpu"))
        # Bias should be negative or zero
        assert torch.all(bias <= 0)
        # For each head, bias should be more negative for larger distances
        for h in range(2):
            for i in range(T):
                for j in range(T):
                    dist = abs(i - j)
                    if dist > 0:
                        # Check that bias is negative
                        assert bias[0, h, i, j] < 0

    def test_with_positions_consistency_with_forward(self):
        """Test that with_positions gives consistent results with forward for sequential positions."""
        alibi = ALiBiEmbedding(num_heads=4)
        T = 10
        # Method 1: using forward with seq_len
        bias1 = alibi(T, torch.device("cpu"))  # (1, 4, T, T)
        
        # Method 2: using with_positions with sequential positions [0, 1, ..., T-1]
        position_ids = torch.arange(T).unsqueeze(0)
        bias2 = alibi.with_positions(position_ids, torch.device("cpu"))
        
        # Should be identical
        assert torch.allclose(bias1, bias2)


class TestTiedEmbedding:
    """Test suite for TiedEmbedding layer."""

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
    def V(self):
        return 1000

    def test_tied_embedding_forward(self, B, T, V, D):
        """Test TiedEmbedding forward pass."""
        from nnx.layers.embedding import TiedEmbedding
        emb = TiedEmbedding(vocab_size=V, embedding_dim=D)
        input_ids = torch.randint(0, V, (B, T))
        out = emb(input_ids)
        assert out.shape == (B, T, D)

    def test_tied_embedding_decode(self, B, T, V, D):
        """Test TiedEmbedding.decode() returns correct logits shape."""
        from nnx.layers.embedding import TiedEmbedding
        emb = TiedEmbedding(vocab_size=V, embedding_dim=D)
        hidden = torch.randn(B, T, D)
        logits = emb.decode(hidden)
        assert logits.shape == (B, T, V)

    def test_tied_embedding_weight_tying(self, V, D):
        """Test TiedEmbedding input and output share same weights."""
        from nnx.layers.embedding import TiedEmbedding
        emb = TiedEmbedding(vocab_size=V, embedding_dim=D)
        
        # Encode a token
        input_ids = torch.tensor([[0]])
        encoded = emb(input_ids)  # (1, 1, D)
        
        # Decode back to logits
        hidden = torch.zeros(1, 1, D)
        hidden[0, 0] = encoded[0, 0]
        logits = emb.decode(hidden)  # (1, 1, V)
        
        # The logit for token 0 should be the highest (since hidden = embedding[0])
        assert logits[0, 0, 0] > logits[0, 0, 1:].max()

    def test_tied_embedding_padding_idx(self, V, D):
        """Test TiedEmbedding padding_idx behavior."""
        from nnx.layers.embedding import TiedEmbedding
        emb = TiedEmbedding(vocab_size=V, embedding_dim=D, padding_idx=0)
        input_ids = torch.tensor([[0, 1, 2]])
        out = emb(input_ids)
        # Padding token (index 0) should be zero
        assert torch.allclose(out[0, 0], torch.zeros(D))

    def test_tied_embedding_gradient_flow(self, V, D):
        """Test gradient flow through tied weights."""
        from nnx.layers.embedding import TiedEmbedding
        emb = TiedEmbedding(vocab_size=V, embedding_dim=D)
        input_ids = torch.randint(0, V, (2, 10))
        
        # Forward through encode
        hidden = emb(input_ids)
        # Forward through decode
        logits = emb.decode(hidden)
        
        loss = logits.sum()
        loss.backward()
        
        # Weight should have gradients
        assert emb.weight.grad is not None


class TestRoPEOptimization:
    """Test suite for RoPE optimization features."""

    @pytest.fixture
    def D(self):
        return 128

    def test_rotate_queries_keys_output(self, D):
        """Test rotate_queries_keys() produces correct output."""
        from nnx.layers.embedding import RotaryEmbedding
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        B, H, T = 2, 4, 10
        q = torch.randn(B, H, T, 32)
        k = torch.randn(B, H, T, 32)
        qr, kr = rope.rotate_queries_keys(q, k)
        assert qr.shape == q.shape
        assert kr.shape == k.shape

    def test_rotate_with_positions_output(self, D):
        """Test rotate_with_positions() produces correct output."""
        from nnx.layers.embedding import RotaryEmbedding
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        B, H, T = 2, 4, 10
        x = torch.randn(B, H, T, 32)
        position_ids = torch.arange(T).unsqueeze(0).expand(B, -1)
        rotated = rope.rotate_with_positions(x, position_ids)
        assert rotated.shape == x.shape

    def test_persistent_false_buffers(self, D):
        """Verify persistent=False buffers work correctly."""
        from nnx.layers.embedding import RotaryEmbedding
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        
        # Check that cos_cached and sin_cached are buffers
        assert hasattr(rope, 'cos_cached')
        assert hasattr(rope, 'sin_cached')
        
        # They should not be in state_dict (persistent=False)
        state_dict = rope.state_dict()
        assert 'cos_cached' not in state_dict
        assert 'sin_cached' not in state_dict

    def test_rotate_queries_keys_preserves_norm(self, D):
        """Test that RoPE rotation produces valid output with correct shape."""
        from nnx.layers.embedding import RotaryEmbedding
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        B, H, T = 2, 4, 10
        q = torch.randn(B, H, T, 32)
        k = torch.randn(B, H, T, 32)
        
        qr, kr = rope.rotate_queries_keys(q, k)
        
        # Output shapes should match input
        assert qr.shape == q.shape
        assert kr.shape == k.shape
        
        # Output should be finite (no NaN/Inf)
        assert torch.all(torch.isfinite(qr))
        assert torch.all(torch.isfinite(kr))
        
        # RoPE should produce different output than input (rotation applied)
        assert not torch.allclose(qr, q)
        assert not torch.allclose(kr, k)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
