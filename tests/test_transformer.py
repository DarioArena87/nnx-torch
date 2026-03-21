"""
Tests for Transformer components (TransformerLayer, CrossAttentionLayer, TransformerStack).
Run with: pytest tests/test_transformer.py -v
"""

import pytest
import torch

from nnx.layers.transformer import TransformerLayer, CrossAttentionLayer, TransformerStack
from nnx.layers.feedforward import FFN, GatedFFN
from nnx.attention import SDPAttention, RoPEAttention, ALiBiAttention


class TestTransformerLayer:
    """Test suite for TransformerLayer."""

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
        mask[0, 7:] = False  # first seq padded from pos 7
        mask[1, 9:] = False  # second seq padded from pos 9
        return mask

    def test_encoder_with_rmsnorm(self, x, mask, D):
        """Test TransformerLayer as encoder (rmsnorm, SwiGLU, SDPA)."""
        layer = TransformerLayer(embed_dim=D, num_heads=4, causal=False, norm_type="rmsnorm")
        out = layer(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_decoder_with_layernorm(self, x, mask, D):
        """Test TransformerLayer as causal decoder (layernorm, FFN)."""
        layer = TransformerLayer(
            embed_dim=D,
            num_heads=4,
            causal=True,
            norm_type="layernorm",
            ffn_cls=FFN,
        )
        out = layer(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_with_geglu_ffn(self, x, D):
        """Test TransformerLayer with GeGLU FFN."""
        layer = TransformerLayer(
            embed_dim=D,
            num_heads=4,
            causal=True,
            attention_cls=SDPAttention,
            ffn_cls=GatedFFN,
            ffn_kwargs={"activation": "gelu"},
        )
        out = layer(x)
        assert out.shape == x.shape

    def test_with_rope_attention(self, x, D):
        """Test TransformerLayer with RoPEAttention."""
        layer = TransformerLayer(
            embed_dim=D,
            num_heads=4,
            attention_cls=RoPEAttention,
            causal=True,
        )
        out = layer(x)
        assert out.shape == x.shape

    def test_with_alibi_attention(self, x, D):
        """Test TransformerLayer with ALiBiAttention."""
        layer = TransformerLayer(
            embed_dim=D,
            num_heads=4,
            attention_cls=ALiBiAttention,
            causal=True,
        )
        out = layer(x)
        assert out.shape == x.shape

    def test_with_dropout(self, x, D):
        """Test TransformerLayer with dropout."""
        layer = TransformerLayer(embed_dim=D, num_heads=4, dropout=0.1)
        out = layer(x)
        assert out.shape == x.shape

    def test_pre_norm_vs_post_norm(self, x, D):
        """Test both pre-norm (default) and post-norm."""
        # Pre-norm (default)
        layer_pre = TransformerLayer(embed_dim=D, num_heads=4, pre_norm=True)
        out_pre = layer_pre(x)
        assert out_pre.shape == x.shape

        # Post-norm
        layer_post = TransformerLayer(embed_dim=D, num_heads=4, pre_norm=False)
        out_post = layer_post(x)
        assert out_post.shape == x.shape

    def test_cross_attention(self, D):
        """Test TransformerLayer with cross-attention (different key/value)."""
        layer = TransformerLayer(embed_dim=D, num_heads=4)
        B = 2
        x = torch.randn(B, 5, D)
        kv = torch.randn(B, 15, D)
        kv_mask = torch.ones(B, 15, dtype=torch.bool)
        kv_mask[0, 12:] = False

        out = layer(x, key=kv, value=kv, attention_mask=kv_mask)
        assert out.shape == x.shape

    def test_eval_mode_deterministic(self, x, D):
        """Test that TransformerLayer is deterministic in eval mode."""
        layer = TransformerLayer(embed_dim=D, num_heads=4, dropout=0.1)
        layer.eval()
        out1 = layer(x)
        out2 = layer(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_gradients_flow(self, x, D):
        """Test that gradients flow correctly through TransformerLayer."""
        layer = TransformerLayer(embed_dim=D, num_heads=4)
        out = layer(x)
        loss = out.sum()
        loss.backward()
        for param in layer.parameters():
            assert param.grad is not None, f"Parameter {param.name if hasattr(param, 'name') else 'unknown'} has no gradient"

    def test_causal_mask(self, x, D):
        """Test TransformerLayer with causal masking."""
        layer = TransformerLayer(embed_dim=D, num_heads=4, causal=True)
        out = layer(x)
        assert out.shape == x.shape

    def test_causal_with_padding_mask(self, x, mask, D):
        """Test TransformerLayer with both causal and padding mask."""
        layer = TransformerLayer(embed_dim=D, num_heads=4, causal=True)
        out = layer(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_different_num_heads(self, x, D):
        """Test TransformerLayer with various numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            if D % num_heads == 0:
                layer = TransformerLayer(embed_dim=D, num_heads=num_heads)
                out = layer(x)
                assert out.shape == x.shape

    def test_custom_ffn_dim(self, x, D):
        """Test TransformerLayer with custom FFN dimension."""
        layer = TransformerLayer(embed_dim=D, num_heads=4, ffn_dim=256)
        out = layer(x)
        assert out.shape == x.shape


class TestCrossAttentionLayer:
    """Test suite for CrossAttentionLayer."""

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
    def encoder_out(self, B, D):
        T_enc = 20
        return torch.randn(B, T_enc, D)

    @pytest.fixture
    def cross_attn_mask(self, B):
        T_enc = 20
        mask = torch.ones(B, T_enc, dtype=torch.bool)
        mask[0, 15:] = False
        mask[1, 17:] = False
        return mask

    def test_basic_cross_attention(self, x, encoder_out, cross_attn_mask, D):
        """Test CrossAttentionLayer basic forward pass."""
        cross = CrossAttentionLayer(embed_dim=D, num_heads=4)
        out = cross(x, encoder_out=encoder_out, cross_attn_mask=cross_attn_mask)
        assert out.shape == x.shape

    def test_with_causal_self_attention(self, x, encoder_out, D):
        """Test CrossAttentionLayer with causal self-attention."""
        cross = CrossAttentionLayer(embed_dim=D, num_heads=4, causal=True)
        out = cross(x, encoder_out=encoder_out)
        assert out.shape == x.shape

    def test_with_dropout(self, x, encoder_out, D):
        """Test CrossAttentionLayer with dropout."""
        cross = CrossAttentionLayer(embed_dim=D, num_heads=4, dropout=0.1)
        out = cross(x, encoder_out=encoder_out)
        assert out.shape == x.shape

    def test_with_custom_attention_cls(self, x, encoder_out, D):
        """Test CrossAttentionLayer with custom attention class."""
        cross = CrossAttentionLayer(
            embed_dim=D,
            num_heads=4,
            attention_cls=RoPEAttention,
        )
        out = cross(x, encoder_out=encoder_out)
        assert out.shape == x.shape

    def test_eval_mode(self, x, encoder_out, D):
        """Test that CrossAttentionLayer is deterministic in eval mode."""
        cross = CrossAttentionLayer(embed_dim=D, num_heads=4, dropout=0.1)
        cross.eval()
        out1 = cross(x, encoder_out=encoder_out)
        out2 = cross(x, encoder_out=encoder_out)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_gradients_flow(self, x, encoder_out, D):
        """Test that gradients flow correctly through CrossAttentionLayer."""
        cross = CrossAttentionLayer(embed_dim=D, num_heads=4)
        out = cross(x, encoder_out=encoder_out)
        loss = out.sum()
        loss.backward()
        for param in cross.parameters():
            assert param.grad is not None, f"Parameter has no gradient"

    def test_different_num_heads(self, x, encoder_out, D):
        """Test CrossAttentionLayer with various numbers of heads."""
        for num_heads in [1, 2, 4, 8]:
            if D % num_heads == 0:
                cross = CrossAttentionLayer(embed_dim=D, num_heads=num_heads)
                out = cross(x, encoder_out=encoder_out)
                assert out.shape == x.shape

    def test_with_layernorm(self, x, encoder_out, D):
        """Test CrossAttentionLayer with LayerNorm."""
        cross = CrossAttentionLayer(embed_dim=D, num_heads=4, norm_type="layernorm")
        out = cross(x, encoder_out=encoder_out)
        assert out.shape == x.shape

    def test_with_gatedffn(self, x, encoder_out, D):
        """Test CrossAttentionLayer with GatedFFN."""
        cross = CrossAttentionLayer(
            embed_dim=D,
            num_heads=4,
            ffn_cls=GatedFFN,
            ffn_kwargs={"activation": "silu"},
        )
        out = cross(x, encoder_out=encoder_out)
        assert out.shape == x.shape


class TestTransformerStack:
    """Test suite for TransformerStack."""

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
        mask[0, 7:] = False  # first seq padded from pos 7
        mask[1, 9:] = False  # second seq padded from pos 9
        return mask

    def test_basic_stack(self, x, mask, D):
        """Test TransformerStack with default settings."""
        stack = TransformerStack(
            n_layers=4,
            embed_dim=D,
            num_heads=4,
            ffn_cls=GatedFFN,
            norm_type="rmsnorm",
            causal=True,
            final_norm=True,
        )
        out = stack(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_stack_without_final_norm(self, x, D):
        """Test TransformerStack without final normalization."""
        stack = TransformerStack(
            n_layers=3,
            embed_dim=D,
            num_heads=4,
            final_norm=False,
        )
        out = stack(x)
        assert out.shape == x.shape

    def test_encoder_stack(self, x, mask, D):
        """Test TransformerStack as encoder (non-causal)."""
        stack = TransformerStack(
            n_layers=2,
            embed_dim=D,
            num_heads=4,
            causal=False,
            norm_type="layernorm",
        )
        out = stack(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_with_custom_attention(self, x, D):
        """Test TransformerStack with RoPEAttention."""
        stack = TransformerStack(
            n_layers=2,
            embed_dim=D,
            num_heads=4,
            attention_cls=RoPEAttention,
            causal=True,
        )
        out = stack(x)
        assert out.shape == x.shape

    def test_with_custom_ffn(self, x, D):
        """Test TransformerStack with FFN instead of GatedFFN."""
        stack = TransformerStack(
            n_layers=2,
            embed_dim=D,
            num_heads=4,
            ffn_cls=FFN,
            ffn_kwargs={"activation": "gelu"},
        )
        out = stack(x)
        assert out.shape == x.shape

    def test_with_dropout(self, x, D):
        """Test TransformerStack with dropout."""
        stack = TransformerStack(
            n_layers=2,
            embed_dim=D,
            num_heads=4,
            dropout=0.1,
        )
        out = stack(x)
        assert out.shape == x.shape

    def test_eval_mode(self, x, D):
        """Test that TransformerStack is deterministic in eval mode."""
        stack = TransformerStack(
            n_layers=2,
            embed_dim=D,
            num_heads=4,
            dropout=0.1,
        )
        stack.eval()
        out1 = stack(x)
        out2 = stack(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_gradients_flow(self, x, D):
        """Test that gradients flow correctly through TransformerStack."""
        stack = TransformerStack(
            n_layers=2,
            embed_dim=D,
            num_heads=4,
        )
        out = stack(x)
        loss = out.sum()
        loss.backward()
        for param in stack.parameters():
            assert param.grad is not None, f"Parameter has no gradient"

    def test_different_num_layers(self, x, D):
        """Test TransformerStack with different numbers of layers."""
        for n_layers in [1, 2, 4]:
            stack = TransformerStack(
                n_layers=n_layers,
                embed_dim=D,
                num_heads=4,
            )
            out = stack(x)
            assert out.shape == x.shape

    def test_parameter_count(self, x, D):
        """Test TransformerStack parameter count is reasonable."""
        stack = TransformerStack(
            n_layers=4,
            embed_dim=D,
            num_heads=4,
        )
        param_count = sum(p.numel() for p in stack.parameters())
        # Should have parameters for 4 layers + optional final_norm
        assert param_count > 0
        # Rough sanity check: should be in the order of millions for typical configs
        assert param_count < 10_000_000  # less than 10M params

    def test_with_causal_and_mask(self, x, mask, D):
        """Test TransformerStack with causal masking and padding mask."""
        stack = TransformerStack(
            n_layers=2,
            embed_dim=D,
            num_heads=4,
            causal=True,
        )
        out = stack(x, attention_mask=mask)
        assert out.shape == x.shape

    def test_pre_norm_vs_post_norm(self, x, D):
        """Test both pre-norm and post-norm configurations."""
        stack_pre = TransformerStack(
            n_layers=2,
            embed_dim=D,
            num_heads=4,
            pre_norm=True,
        )
        out_pre = stack_pre(x)
        assert out_pre.shape == x.shape

        stack_post = TransformerStack(
            n_layers=2,
            embed_dim=D,
            num_heads=4,
            pre_norm=False,
        )
        out_post = stack_post(x)
        assert out_post.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
