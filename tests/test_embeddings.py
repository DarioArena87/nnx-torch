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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
