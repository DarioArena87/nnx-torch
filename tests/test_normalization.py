"""
Tests for normalization layers (ScaleNorm, CosineNorm, AdaptiveRMSNorm).
Run with: pytest tests/test_normalization.py -v
"""

import pytest
import torch

from nnx.layers.normalization import ScaleNorm, CosineNorm, AdaptiveRMSNorm


class TestNormalizationLayers:
    """Test suite for normalization layers."""

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
    def cond(self, B, D):
        return torch.randn(B, D)

    def test_scalenorm_basic(self, x, D):
        """Test ScaleNorm basic forward pass."""
        norm = ScaleNorm(D)
        out = norm(x)
        assert out.shape == x.shape

    def test_scalenorm_different_dims(self):
        """Test ScaleNorm with various dimensions."""
        x = torch.randn(2, 10, 64)
        norm = ScaleNorm(64)
        out = norm(x)
        assert out.shape == x.shape

    def test_scalenorm_scale_parameter(self, x, D):
        """Test that ScaleNorm has a single scale parameter."""
        norm = ScaleNorm(D)
        assert hasattr(norm, 'scale')
        assert norm.scale.shape == ()
        assert norm.scale.requires_grad

    def test_scalenorm_gradients(self, x, D):
        """Test that gradients flow correctly through ScaleNorm."""
        norm = ScaleNorm(D)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        assert norm.scale.grad is not None

    def test_cosinenorm_basic(self, x):
        """Test CosineNorm basic forward pass."""
        norm = CosineNorm()
        out = norm(x)
        assert out.shape == x.shape

    def test_cosinenorm_unit_norm(self, x):
        """Test that CosineNorm produces unit-norm outputs."""
        norm = CosineNorm()
        out = norm(x)
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_cosinenorm_eps(self):
        """Test CosineNorm with different epsilon values."""
        x = torch.randn(2, 10, 128)
        norm = CosineNorm(eps=1e-6)
        out = norm(x)
        assert out.shape == x.shape

    def test_cosinenorm_gradients(self, x):
        """Test that gradients flow correctly through CosineNorm."""
        norm = CosineNorm()
        x_req_grad = x.clone().requires_grad_(True)
        out = norm(x_req_grad)
        loss = out.sum()
        loss.backward()
        assert x_req_grad.grad is not None

    def test_adaptivermsnorm_basic(self, x, cond, D):
        """Test AdaptiveRMSNorm basic forward pass."""
        norm = AdaptiveRMSNorm(dim=D, cond_dim=D)
        out = norm(x, cond)
        assert out.shape == x.shape

    def test_adaptivermsnorm_with_cond_dim(self, x, D):
        """Test AdaptiveRMSNorm with different conditioning dimension."""
        cond = torch.randn(2, 64)
        norm = AdaptiveRMSNorm(dim=D, cond_dim=64)
        out = norm(x, cond)
        assert out.shape == x.shape

    def test_adaptivermsnorm_no_shift(self, x, D):
        """Test AdaptiveRMSNorm without shift parameter."""
        cond = torch.randn(2, 64)
        norm = AdaptiveRMSNorm(dim=D, cond_dim=64, use_shift=False)
        out = norm(x, cond)
        assert out.shape == x.shape

    def test_adaptivermsnorm_cond_3d(self, x, D):
        """Test AdaptiveRMSNorm with 3D conditioning tensor."""
        cond = torch.randn(2, 1, 64)
        norm = AdaptiveRMSNorm(dim=D, cond_dim=64)
        out = norm(x, cond)
        assert out.shape == x.shape

    def test_adaptivermsnorm_zero_init(self, x, D):
        """Test that AdaptiveRMSNorm starts as identity-like (zero-initialized)."""
        norm = AdaptiveRMSNorm(dim=D, cond_dim=64)
        # With zero-initialized projection, output should be close to normalized input
        cond = torch.zeros(2, 64)
        out = norm(x, cond)
        # Should preserve the normalized scale but with zero shift
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        expected = x / (rms + 1e-6)
        assert torch.allclose(out, expected, atol=1e-5)

    def test_adaptivermsnorm_gradients(self, x, D):
        """Test that gradients flow correctly through AdaptiveRMSNorm."""
        norm = AdaptiveRMSNorm(dim=D, cond_dim=64)
        cond = torch.randn(2, 64)
        out = norm(x, cond)
        loss = out.sum()
        loss.backward()
        for param in norm.parameters():
            assert param.grad is not None

    def test_adaptivermsnorm_parameters_count(self, D):
        """Test AdaptiveRMSNorm parameter count with and without shift."""
        norm_with_shift = AdaptiveRMSNorm(dim=D, cond_dim=64, use_shift=True)
        params_with_shift = sum(p.numel() for p in norm_with_shift.parameters())
        # proj: (64, 128*2=256) + bias: 256 = 64*256 + 256 = 16384 + 256 = 16640
        expected_with_shift = 64 * 256 + 256
        assert params_with_shift == expected_with_shift

        norm_without_shift = AdaptiveRMSNorm(dim=D, cond_dim=64, use_shift=False)
        params_without_shift = sum(p.numel() for p in norm_without_shift.parameters())
        expected_without_shift = 64 * 128 + 128
        assert params_without_shift == expected_without_shift

    def test_scalenorm_eval_mode(self, x, D):
        """Test that ScaleNorm behaves consistently in eval mode."""
        norm = ScaleNorm(D)
        norm.eval()
        out1 = norm(x)
        out2 = norm(x)
        assert torch.allclose(out1, out2)

    def test_cosinenorm_eval_mode(self, x):
        """Test that CosineNorm behaves consistently in eval mode."""
        norm = CosineNorm()
        norm.eval()
        out1 = norm(x)
        out2 = norm(x)
        assert torch.allclose(out1, out2)

    def test_adaptivermsnorm_eval_mode(self, x, D):
        """Test that AdaptiveRMSNorm behaves consistently in eval mode."""
        norm = AdaptiveRMSNorm(dim=D, cond_dim=64)
        cond = torch.randn(2, 64)
        norm.eval()
        out1 = norm(x, cond)
        out2 = norm(x, cond)
        assert torch.allclose(out1, out2)

    def test_normalization_output_range(self, x):
        """Test that CosineNorm output values are bounded."""
        norm = CosineNorm()
        out = norm(x)
        # Values should be in [-1, 1] for unit-normalized vectors
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0)


class TestRMSNormFallback:
    """Test suite for RMSNorm fallback implementation."""

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

    def test_rmsnorm_forward(self, x, D):
        """Test RMSNorm forward pass."""
        from nnx.layers.normalization import RMSNorm
        norm = RMSNorm(D)
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_gradient_computation(self, x, D):
        """Test RMSNorm gradient computation."""
        from nnx.layers.normalization import RMSNorm
        norm = RMSNorm(D)
        out = norm(x)
        loss = out.sum()
        loss.backward()
        # Check that weight has gradients
        assert norm.weight.grad is not None

    def test_rmsnorm_formula(self, D):
        """Verify behavior matches expected RMSNorm formula."""
        from nnx.layers.normalization import RMSNorm
        torch.manual_seed(42)
        norm = RMSNorm(D)
        norm.eval()
        
        x = torch.randn(2, 10, D)
        with torch.no_grad():
            out = norm(x)
        
        # Manually compute RMSNorm
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        expected = x / (rms + 1e-6) * norm.weight
        
        torch.testing.assert_close(out, expected, rtol=1e-4, atol=1e-4)

    def test_rmsnorm_weight_shape(self, D):
        """Test that RMSNorm has correct weight shape."""
        from nnx.layers.normalization import RMSNorm
        norm = RMSNorm(D)
        assert norm.weight.shape == (D,)

    def test_rmsnorm_eps(self, D):
        """Test RMSNorm with different epsilon values."""
        from nnx.layers.normalization import RMSNorm
        norm = RMSNorm(D, eps=1e-5)
        x = torch.randn(2, 10, D)
        out = norm(x)
        assert out.shape == x.shape
        assert torch.all(torch.isfinite(out))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])