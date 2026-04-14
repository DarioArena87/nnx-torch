"""
Tests for LatentPooler layer.
Run with: pytest tests/test_pooling.py -v
"""

import pytest
import torch

from nnx.layers.pooling import LatentPooler


class TestLatentPooler:
    """Test suite for LatentPooler."""

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
    def L(self):
        return 64

    @pytest.fixture
    def x(self, B, T, D):
        return torch.randn(B, T, D)

    def test_pooler_basic_output_shape(self, x, D, L):
        """Test LatentPooler produces correct output shape."""
        pooler = LatentPooler(d_model=D, latent_dim=L)
        out = pooler(x)
        assert out.shape == (x.shape[0], L)

    def test_pooler_different_latent_dim(self, B, T, D):
        """Test LatentPooler with various latent_dim values."""
        x = torch.randn(B, T, D)
        for latent_dim in [32, 64, 128, 256, 512]:
            pooler = LatentPooler(d_model=D, latent_dim=latent_dim)
            out = pooler(x)
            assert out.shape == (B, latent_dim)

    def test_pooler_different_d_model(self, B, T, L):
        """Test LatentPooler with various d_model values."""
        for d_model in [64, 128, 256, 512]:
            x = torch.randn(B, T, d_model)
            pooler = LatentPooler(d_model=d_model, latent_dim=L)
            out = pooler(x)
            assert out.shape == (B, L)

    def test_pooler_without_mask(self, B, T, D, L):
        """Test LatentPooler without attention mask."""
        x = torch.randn(B, T, D)
        pooler = LatentPooler(d_model=D, latent_dim=L)
        out = pooler(x, attention_mask=None)
        assert out.shape == (B, L)

    def test_pooler_with_mask(self, B, T, D, L):
        """Test LatentPooler with attention mask."""
        x = torch.randn(B, T, D)
        # Create a mask where last 3 tokens are padding
        mask = torch.ones(B, T, dtype=torch.long)
        mask[:, -3:] = 0
        pooler = LatentPooler(d_model=D, latent_dim=L)
        out = pooler(x, attention_mask=mask)
        assert out.shape == (B, L)

    def test_pooler_with_partial_mask(self, B, T, D, L):
        """Test LatentPooler with varying amounts of padding in mask."""
        x = torch.randn(B, T, D)
        # Test different mask configurations
        for padding_count in [0, 1, 3, 5, T - 1]:
            mask = torch.ones(B, T, dtype=torch.long)
            if padding_count > 0:
                mask[:, -padding_count:] = 0
            pooler = LatentPooler(d_model=D, latent_dim=L)
            out = pooler(x, attention_mask=mask)
            assert out.shape == (B, L)

    def test_pooler_with_dropout_train(self, B, T, D, L):
        """Test LatentPooler with dropout in training mode."""
        x = torch.randn(B, T, D)
        pooler = LatentPooler(d_model=D, latent_dim=L, dropout=0.1)
        pooler.train()
        # Should have some stochasticity from dropout
        out = pooler(x)
        assert out.shape == (B, L)

    def test_pooler_with_dropout_eval(self, x, D, L):
        """Test LatentPooler with dropout in eval mode."""
        pooler = LatentPooler(d_model=D, latent_dim=L, dropout=0.1)
        pooler.eval()
        out1 = pooler(x)
        out2 = pooler(x)
        # Should be deterministic in eval mode
        assert torch.allclose(out1, out2)
        assert out1.shape == out2.shape == (x.shape[0], L)

    def test_pooler_zero_dropout(self, x, D, L):
        """Test LatentPooler with zero dropout is deterministic."""
        pooler = LatentPooler(d_model=D, latent_dim=L, dropout=0.0)
        out1 = pooler(x)
        out2 = pooler(x)
        assert torch.allclose(out1, out2)

    def test_pooler_gradients(self, x, D, L):
        """Test that gradients flow correctly through LatentPooler."""
        pooler = LatentPooler(d_model=D, latent_dim=L)
        out = pooler(x)
        loss = out.sum()
        loss.backward()
        # Check all parameters have gradients
        assert pooler.query.grad is not None
        assert pooler.attn.weight.grad is not None
        assert pooler.attn.bias.grad is not None
        assert pooler.proj.weight.grad is not None
        assert pooler.proj.bias.grad is not None

    def test_pooler_gradients_with_mask(self, B, T, D, L):
        """Test that gradients flow correctly with attention mask."""
        x = torch.randn(B, T, D, requires_grad=True)
        mask = torch.ones(B, T, dtype=torch.long)
        mask[:, -2:] = 0
        pooler = LatentPooler(d_model=D, latent_dim=L)
        out = pooler(x, attention_mask=mask)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        for param in pooler.parameters():
            assert param.grad is not None

    def test_pooler_different_batch_sizes(self, T, D, L):
        """Test LatentPooler with various batch sizes."""
        for batch_size in [1, 2, 4, 8, 16]:
            x = torch.randn(batch_size, T, D)
            pooler = LatentPooler(d_model=D, latent_dim=L)
            out = pooler(x)
            assert out.shape == (batch_size, L)

    def test_pooler_different_sequence_lengths(self, B, D, L):
        """Test LatentPooler with various sequence lengths."""
        for seq_len in [1, 2, 5, 10, 20, 50, 100]:
            x = torch.randn(B, seq_len, D)
            mask = torch.ones(B, seq_len, dtype=torch.long)
            # Add some padding for longer sequences
            if seq_len > 5:
                mask[:, -2:] = 0
            pooler = LatentPooler(d_model=D, latent_dim=L)
            out = pooler(x, attention_mask=mask)
            assert out.shape == (B, L)

    def test_pooler_single_token_sequence(self, B, D, L):
        """Test LatentPooler with single token sequence."""
        x = torch.randn(B, 1, D)
        pooler = LatentPooler(d_model=D, latent_dim=L)
        out = pooler(x)
        assert out.shape == (B, L)

    def test_pooler_batch_size_one(self, T, D, L):
        """Test LatentPooler with batch size of 1."""
        x = torch.randn(1, T, D)
        mask = torch.ones(1, T, dtype=torch.long)
        pooler = LatentPooler(d_model=D, latent_dim=L)
        out = pooler(x, attention_mask=mask)
        assert out.shape == (1, L)

    def test_pooler_all_padding_mask(self, B, T, D, L):
        """Test LatentPooler when entire sequence is padding (edge case)."""
        x = torch.randn(B, T, D)
        mask = torch.zeros(B, T, dtype=torch.long)
        pooler = LatentPooler(d_model=D, latent_dim=L)
        out = pooler(x, attention_mask=mask)
        # Should still produce output even with all padding
        assert out.shape == (B, L)
        # All attention should go to first position due to softmax with -inf
        # (or numerically it should be well-behaved)

    def test_pooler_consistency_across_calls(self, x, D, L):
        """Test that LatentPooler produces consistent results for same input."""
        pooler = LatentPooler(d_model=D, latent_dim=L)
        out1 = pooler(x)
        out2 = pooler(x)
        assert torch.allclose(out1, out2)

    def test_pooler_different_inputs_produce_different_outputs(self, B, T, D, L):
        """Test that different inputs produce different outputs."""
        x1 = torch.randn(B, T, D)
        x2 = torch.randn(B, T, D)
        pooler = LatentPooler(d_model=D, latent_dim=L)
        out1 = pooler(x1)
        out2 = pooler(x2)
        # With high probability, different random inputs produce different outputs
        assert not torch.allclose(out1, out2)

    def test_pooler_parameters_registered(self, D, L):
        """Test that all parameters are properly registered."""
        pooler = LatentPooler(d_model=D, latent_dim=L)
        params = dict(pooler.named_parameters())
        assert "query" in params
        assert "attn.weight" in params
        assert "attn.bias" in params
        assert "proj.weight" in params
        assert "proj.bias" in params

    def test_pooler_parameter_shapes(self, D, L):
        """Test that parameters have correct shapes."""
        pooler = LatentPooler(d_model=D, latent_dim=L)
        assert pooler.query.shape == (D,)
        assert pooler.attn.weight.shape == (1, D)
        assert pooler.attn.bias.shape == (1,)
        assert pooler.proj.weight.shape == (L, D)
        assert pooler.proj.bias.shape == (L,)

    def test_pooler_device_transfer(self, D, L):
        """Test that LatentPooler can be moved to different devices."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        pooler = LatentPooler(d_model=D, latent_dim=L)
        pooler = pooler.cuda()
        x = torch.randn(2, 10, D).cuda()
        mask = torch.ones(2, 10, dtype=torch.long).cuda()
        out = pooler(x, attention_mask=mask)
        assert out.is_cuda
        assert out.shape == (2, L)

    def test_pooler_multiple_forward_passes(self, x, D, L):
        """Test that LatentPooler works correctly over multiple forward passes."""
        pooler = LatentPooler(d_model=D, latent_dim=L)
        # Multiple forward passes should not cause issues
        for _ in range(10):
            out = pooler(x)
            assert out.shape == (x.shape[0], L)

    def test_pooler_gradient_accumulation(self, B, T, D, L):
        """Test that gradient accumulation works correctly."""
        pooler = LatentPooler(d_model=D, latent_dim=L)
        optimizer = torch.optim.SGD(pooler.parameters(), lr=0.01)
        
        x1 = torch.randn(B, T, D)
        x2 = torch.randn(B, T, D)
        
        # Forward pass 1
        out1 = pooler(x1)
        loss1 = out1.sum()
        loss1.backward()
        
        # Forward pass 2
        out2 = pooler(x2)
        loss2 = out2.sum()
        loss2.backward()
        
        # Gradient accumulation should have accumulated gradients
        optimizer.step()
        optimizer.zero_grad()
        # After optimization, gradients should be cleared
        for param in pooler.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    def test_pooler_equivalence_mask_ones_vs_none(self, B, T, D, L):
        """Test that mask of all ones produces same result as no mask."""
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.long)
        pooler = LatentPooler(d_model=D, latent_dim=L)
        
        out_with_mask = pooler(x, attention_mask=mask)
        out_without_mask = pooler(x, attention_mask=None)
        
        assert torch.allclose(out_with_mask, out_without_mask)

    def test_pooler_known_values(self, D, L):
        """Test LatentPooler with deterministic inputs for reproducibility."""
        torch.manual_seed(42)
        pooler = LatentPooler(d_model=D, latent_dim=L)
        
        # Set all weights to a known state
        torch.nn.init.constant_(pooler.query, 0.0)
        torch.nn.init.constant_(pooler.attn.weight, 0.0)
        torch.nn.init.constant_(pooler.attn.bias, 0.0)
        torch.nn.init.constant_(pooler.proj.weight, 0.0)
        torch.nn.init.constant_(pooler.proj.bias, 0.0)
        
        x = torch.randn(2, 5, D)
        out = pooler(x)
        
        # With zero weights, output should be all zeros
        assert torch.allclose(out, torch.zeros(2, L), atol=1e-6)
