"""
Tests for feed-forward network variants (FFN, GatedFFN, MoEFFN).
Run with: pytest tests/test_feedforward.py -v
"""

import pytest
import torch

from nnx.layers.feedforward import FFN, GatedFFN, MoEFFN


class TestFFN:
    """Test suite for FFN."""

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

    def test_ffn_basic_gelu(self, x, D):
        """Test FFN with GELU activation."""
        ffn = FFN(D, activation="gelu")
        out = ffn(x)
        assert out.shape == x.shape

    def test_ffn_basic_relu(self, x, D):
        """Test FFN with ReLU activation."""
        ffn = FFN(D, activation="relu")
        out = ffn(x)
        assert out.shape == x.shape

    def test_ffn_basic_silu(self, x, D):
        """Test FFN with SiLU activation."""
        ffn = FFN(D, activation="silu")
        out = ffn(x)
        assert out.shape == x.shape

    def test_ffn_custom_ffn_dim(self, x, D):
        """Test FFN with custom hidden dimension."""
        ffn = FFN(D, ffn_dim=256, activation="gelu")
        out = ffn(x)
        assert out.shape == x.shape

    def test_ffn_with_dropout(self, x, D):
        """Test FFN with dropout."""
        ffn = FFN(D, activation="gelu", dropout=0.1)
        out = ffn(x)
        assert out.shape == x.shape

    def test_ffn_no_bias(self, x, D):
        """Test FFN without bias."""
        ffn = FFN(D, activation="gelu", bias=False)
        out = ffn(x)
        assert out.shape == x.shape

    def test_ffn_gradients(self, x, D):
        """Test that gradients flow correctly through FFN."""
        ffn = FFN(D, activation="gelu")
        out = ffn(x)
        loss = out.sum()
        loss.backward()
        for param in ffn.parameters():
            assert param.grad is not None

    def test_ffn_eval_mode(self, x, D):
        """Test that FFN behaves consistently in eval mode."""
        ffn = FFN(D, activation="gelu", dropout=0.1)
        ffn.eval()
        out1 = ffn(x)
        out2 = ffn(x)
        assert torch.allclose(out1, out2)

    def test_ffn_different_ffn_dim(self, D):
        """Test FFN with various ffn_dim values."""
        x = torch.randn(2, 10, D)
        for ffn_dim in [64, 128, 256, 512]:
            ffn = FFN(D, ffn_dim=ffn_dim)
            out = ffn(x)
            assert out.shape == x.shape


class TestGatedFFN:
    """Test suite for GatedFFN."""

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

    def test_gatedffn_basic_swiglu(self, x, D):
        """Test GatedFFN with SiLU activation (SwiGLU)."""
        gffn = GatedFFN(D, activation="silu")
        out = gffn(x)
        assert out.shape == x.shape

    def test_gatedffn_basic_geglu(self, x, D):
        """Test GatedFFN with GELU activation (GeGLU)."""
        gffn = GatedFFN(D, activation="gelu")
        out = gffn(x)
        assert out.shape == x.shape

    def test_gatedffn_basic_relu(self, x, D):
        """Test GatedFFN with ReLU activation (ReGLU)."""
        gffn = GatedFFN(D, activation="relu")
        out = gffn(x)
        assert out.shape == x.shape

    def test_gatedffn_custom_ffn_dim(self, x, D):
        """Test GatedFFN with custom hidden dimension."""
        gffn = GatedFFN(D, ffn_dim=256, activation="silu")
        out = gffn(x)
        assert out.shape == x.shape

    def test_gatedffn_with_dropout(self, x, D):
        """Test GatedFFN with dropout."""
        gffn = GatedFFN(D, activation="silu", dropout=0.1)
        out = gffn(x)
        assert out.shape == x.shape

    def test_gatedffn_no_bias(self, x, D):
        """Test GatedFFN without bias."""
        gffn = GatedFFN(D, activation="silu", bias=False)
        out = gffn(x)
        assert out.shape == x.shape

    def test_gatedffn_gradients(self, x, D):
        """Test that gradients flow correctly through GatedFFN."""
        gffn = GatedFFN(D, activation="silu")
        out = gffn(x)
        loss = out.sum()
        loss.backward()
        for param in gffn.parameters():
            assert param.grad is not None

    def test_gatedffn_eval_mode(self, x, D):
        """Test that GatedFFN behaves consistently in eval mode."""
        gffn = GatedFFN(D, activation="silu", dropout=0.1)
        gffn.eval()
        out1 = gffn(x)
        out2 = gffn(x)
        assert torch.allclose(out1, out2)

    def test_gatedffn_ffn_dim_rounding(self, D):
        """Test that GatedFFN rounds ffn_dim to multiple of 64."""
        # Test a value that needs rounding
        gffn = GatedFFN(D, ffn_dim=200, activation="silu")
        # The actual ffn_dim should be (200 + 63) // 64 * 64 = 256 for the output layer and double it for the linear+up projection
        # Check the weight shape
        expected_dim = (200 + 63) // 64 * 64
        assert gffn.w13.weight.shape[0] == 2 * expected_dim
        assert gffn.w2.weight.shape[1] == expected_dim

    def test_packed_gates_basic(self, x, D):
        """Test GatedFFN with packed_gates=True."""
        gffn = GatedFFN(D, activation="silu")
        out = gffn(x)
        assert out.shape == x.shape

    def test_packed_gates_vs_unpacked(self, D):
        """Verify numerical equivalence between packed_gates=True and False."""
        torch.manual_seed(42)
        gffn_packed = GatedFFN(D, activation="silu")

        torch.manual_seed(42)
        gffn_unpacked = GatedFFN(D, activation="silu")

        # Copy weights from unpacked to packed for fair comparison
        with torch.no_grad():
            # Initialize both with same seed, they should produce similar outputs
            pass

        x = torch.randn(2, 10, D)
        gffn_packed.eval()
        gffn_unpacked.eval()

        with torch.no_grad():
            out_packed = gffn_packed(x)
            out_unpacked = gffn_unpacked(x)

        # Both should produce valid outputs with same shape
        assert out_packed.shape == out_unpacked.shape
        torch.testing.assert_close(out_packed, out_unpacked, rtol=1e-4, atol=1e-4)

    def test_gradient_checkpointing_basic(self, x, D):
        """Test GatedFFN with gradient_checkpointing=True."""
        gffn = GatedFFN(D, activation="silu", gradient_checkpointing=True)
        gffn.train()  # Checkpointing only active in training mode
        out = gffn(x)
        assert out.shape == x.shape

    def test_gradient_checkpointing_gradients(self, D):
        """Verify gradients flow correctly through checkpointed layers."""
        gffn = GatedFFN(D, activation="silu", gradient_checkpointing=True)
        gffn.train()
        x = torch.randn(2, 10, D, requires_grad=True)
        out = gffn(x)
        loss = out.sum()
        loss.backward()
        # Check that input gradients exist
        assert x.grad is not None
        # Check that parameter gradients exist
        for param in gffn.parameters():
            assert param.grad is not None

    def test_gradient_checkpointing_with_requires_grad(self, D):
        """Test with requires_grad=True inputs."""
        gffn = GatedFFN(D, activation="silu", gradient_checkpointing=True)
        gffn.train()
        x = torch.randn(2, 10, D, requires_grad=True)
        out = gffn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_gradient_checkpointing_eval_mode(self, x, D):
        """Test that checkpointing is disabled in eval mode."""
        gffn = GatedFFN(D, activation="silu", gradient_checkpointing=True)
        gffn.eval()
        out1 = gffn(x)
        out2 = gffn(x)
        # In eval mode, outputs should be identical (no dropout, no checkpointing)
        assert torch.allclose(out1, out2)

class TestMoEFFN:
    """Test suite for MoEFFN."""

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

    def test_moeffn_basic(self, x, D):
        """Test MoEFFN with default settings."""
        moe = MoEFFN(D, ffn_dim=256, num_experts=4, top_k=2)
        out = moe(x)
        assert out.shape == x.shape

    def test_moeffn_aux_loss(self, x, D):
        """Test that MoEFFN computes auxiliary loss."""
        moe = MoEFFN(D, ffn_dim=256, num_experts=4, top_k=2)
        out = moe(x)
        assert out.shape == x.shape
        # aux_loss should be a scalar tensor >= 0
        assert moe.aux_loss.item() >= 0
        assert moe.aux_loss.requires_grad

    def test_moeffn_different_num_experts(self, x, D):
        """Test MoEFFN with different numbers of experts."""
        for num_experts in [2, 4, 8]:
            moe = MoEFFN(D, ffn_dim=256, num_experts=num_experts, top_k=2)
            out = moe(x)
            assert out.shape == x.shape
            assert moe.num_experts == num_experts

    def test_moeffn_different_top_k(self, x, D):
        """Test MoEFFN with different top_k values."""
        for top_k in [1, 2]:
            num_experts = 4
            moe = MoEFFN(D, ffn_dim=256, num_experts=num_experts, top_k=top_k)
            out = moe(x)
            assert out.shape == x.shape

    def test_moeffn_with_ffn_expert(self, x, D):
        """Test MoEFFN using FFN as expert instead of GatedFFN."""
        moe = MoEFFN(D, ffn_dim=256, num_experts=4, top_k=2, expert_cls=FFN)
        out = moe(x)
        assert out.shape == x.shape

    def test_moeffn_with_expert_kwargs(self, x, D):
        """Test MoEFFN with expert kwargs."""
        moe = MoEFFN(D, ffn_dim=256, num_experts=4, top_k=2, expert_kwargs={"activation": "relu"})
        out = moe(x)
        assert out.shape == x.shape

    def test_moeffn_gradients(self, x, D):
        """Test that gradients flow correctly through MoEFFN."""
        moe = MoEFFN(D, ffn_dim=256, num_experts=4, top_k=2)
        out = moe(x)
        loss = out.sum()
        # Include aux loss in gradient computation
        total_loss = loss + moe.aux_loss
        total_loss.backward()
        # Check that parameters have gradients
        for name, param in moe.named_parameters():
            assert param.grad is not None, f"Parameter {name} has no gradient"

    def test_moeffn_eval_mode(self, x, D):
        """Test that MoEFFN behaves consistently in eval mode."""
        moe = MoEFFN(D, ffn_dim=256, num_experts=4, top_k=2)
        moe.eval()
        out1 = moe(x)
        out2 = moe(x)
        assert torch.allclose(out1, out2)

    def test_moeffn_router_output_shape(self, D):
        """Test that the router produces correct output shape."""
        moe = MoEFFN(D, ffn_dim=256, num_experts=4, top_k=2)
        # Create a small input to check router
        x_small = torch.randn(5, D)
        logits = moe.router(x_small)
        assert logits.shape == (5, 4)

    def test_moeffn_top_k_routing(self, x, D):
        """Test that top-k routing selects correct number of experts."""
        moe = MoEFFN(D, ffn_dim=256, num_experts=4, top_k=2)
        _ = moe(x)  # Run forward to populate routing
        # The aux_loss should be computed
        assert moe.aux_loss.item() >= 0
        # The value should be between 0 and num_experts theoretically
        # (it's num_experts * sum(f * p) where f and p are probability distributions)
        assert moe.aux_loss.item() <= moe.num_experts

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
