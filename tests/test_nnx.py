"""
Tests for the nnx library.
Run with: python test_nnx.py
"""

import traceback
import typing

import torch


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def ok(name):
    print(f"  ✓  {name}")


def fail(name, err = None):
    print(f"  ✗  {name}")
    traceback.print_exc()
    raise err if err is not None else Exception(f"Test {name} failed")


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------
def test_mask_utilities():
    section("Mask utilities")

    from nnx.utils.mask import hf_to_additive, make_causal_mask, combine_masks

    B, T = 2, 10
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 7:] = False  # first seq padded from pos 7
    mask[1, 9:] = False  # second seq padded from pos 9

    try:
        additive = hf_to_additive(mask)
        assert additive.shape == (B, 1, 1, T)
        assert additive[0, 0, 0, 7] == float("-inf")
        assert additive[0, 0, 0, 6] == 0.0
        ok("hf_to_additive shape and values")
    except Exception as e:
        fail("hf_to_additive", e)

    try:
        causal = make_causal_mask(T, device=torch.device("cpu"))
        assert causal.shape == (1, 1, T, T)
        assert causal[0, 0, 0, 1] == float("-inf")
        assert causal[0, 0, 1, 0] == 0.0
        ok("make_causal_mask")
    except Exception as e:
        fail("make_causal_mask", e)

    try:
        combined = combine_masks(additive, causal)
        assert combined.shape == (B, 1, T, T)
        ok("combine_masks")
    except Exception as e:
        fail("combine_masks", e)


# ---------------------------------------------------------------------------
# Normalization layers
# ---------------------------------------------------------------------------
def test_normalization_layers():
    section("Normalization layers")

    from nnx.layers.normalization import ScaleNorm, CosineNorm, AdaptiveRMSNorm

    B, T, D = 2, 10, 128
    x = torch.randn(B, T, D)

    try:
        norm = ScaleNorm(128)
        out = norm(x)
        assert out.shape == x.shape
        ok("ScaleNorm")
    except Exception as e:
        fail("ScaleNorm", e)

    try:
        norm = CosineNorm()
        out = norm(x)
        assert out.shape == x.shape
        # unit-norm check
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
        ok("CosineNorm — unit-norm check")
    except Exception as e:
        fail("CosineNorm", e)

    try:
        norm = AdaptiveRMSNorm(128, cond_dim=64)
        cond = torch.randn(B, 64)
        out = norm(x, cond)
        assert out.shape == x.shape
        ok("AdaptiveRMSNorm")
    except Exception as e:
        fail("AdaptiveRMSNorm", e)


# ---------------------------------------------------------------------------
# Feed-forward networks
# ---------------------------------------------------------------------------
def test_ff_networks():
    section("Feed-forward networks")

    from nnx.layers.feedforward import FFN, GatedFFN, MoEFFN

    B, T, D = 2, 10, 128
    x = torch.randn(B, T, D)

    try:
        ffn = FFN(D, activation="gelu")
        out = ffn(x)
        assert out.shape == x.shape
        ok("FFN (gelu)")
    except Exception as e:
        fail("FFN", e)

    try:
        ffn = GatedFFN(D, activation="silu")
        out = ffn(x)
        assert out.shape == x.shape
        ok("GatedFFN / SwiGLU")
    except Exception as e:
        fail("GatedFFN", e)

    try:
        moe = MoEFFN(D, ffn_dim=256, num_experts=4, top_k=2)
        out = moe(x)
        assert out.shape == x.shape
        assert moe.aux_loss.item() >= 0
        ok("MoEFFN (4 experts, top-2)")
    except Exception as e:
        fail("MoEFFN", e)


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def test_embedding_layers():
    section("Embeddings and positional encodings")

    from nnx.layers.embedding import (
        TokenEmbedding,
        SinusoidalPositional,
        LearnedPositional,
        RotaryEmbedding,
        ALiBiEmbedding,
    )

    B, T, D = 2, 10, 128
    x = torch.randn(B, T, D)

    try:
        emb = TokenEmbedding(1000, 128, scale=True)
        ids = torch.randint(0, 1000, (B, T))
        out = emb(ids)
        assert out.shape == (B, T, 128)
        ok("TokenEmbedding")
    except Exception as e:
        fail("TokenEmbedding", e)

    try:
        pos = SinusoidalPositional(128)
        out = pos(x)
        assert out.shape == x.shape
        ok("SinusoidalPositional")
    except Exception as e:
        fail("SinusoidalPositional", e)

    try:
        pos = LearnedPositional(512, 128)
        out = pos(x)
        assert out.shape == x.shape
        ok("LearnedPositional")
    except Exception as e:
        fail("LearnedPositional", e)

    try:
        rope = RotaryEmbedding(head_dim=32, max_len=512)
        q = torch.randn(B, 4, T, 32)
        k = torch.randn(B, 4, T, 32)
        qr, kr = rope(q, k)
        assert qr.shape == q.shape
        assert kr.shape == k.shape
        ok("RotaryEmbedding")
    except Exception as e:
        fail("RotaryEmbedding", e)

    try:
        alibi = ALiBiEmbedding(num_heads=8)
        bias = alibi(T, device=torch.device("cpu"))
        assert bias.shape == (1, 8, T, T)
        ok("ALiBiEmbedding")
    except Exception as e:
        fail("ALiBiEmbedding", e)


# ---------------------------------------------------------------------------
# SDPA Attention backend
# ---------------------------------------------------------------------------
def test_sdpa_attention():
    section("SDPAttention (SDPA backend)")

    from nnx.attention import SDPAttention

    B, T, D = 2, 10, 128
    x = torch.randn(B, T, D)

    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 7:] = False  # first seq padded from pos 7
    mask[1, 9:] = False  # second seq padded from pos 9

    try:
        attn = SDPAttention(embed_dim=D, num_heads=4)
        out = attn(x)
        assert out.shape == x.shape
        ok("SDPAttention — self-attention")
    except Exception as e:
        fail("SDPAttention self-attention", e)

    try:
        attn = SDPAttention(embed_dim=D, num_heads=4)
        out = attn(x, attention_mask=mask)
        assert out.shape == x.shape
        ok("SDPAttention — with HF-style mask")
    except Exception as e:
        fail("SDPAttention with mask", e)

    try:
        attn = SDPAttention(embed_dim=D, num_heads=4)
        out = attn(x, causal=True)
        assert out.shape == x.shape
        ok("SDPAttention — causal=True")
    except Exception as e:
        fail("SDPAttention causal", e)

    try:
        # Cross-attention: different T for query and key/value
        attn = SDPAttention(embed_dim=D, num_heads=4)
        q = torch.randn(B, 5, D)
        kv = torch.randn(B, 15, D)
        kv_mask = torch.ones(B, 15, dtype=torch.bool)
        out = attn(q, key=kv, value=kv, attention_mask=kv_mask)
        assert out.shape == (B, 5, D)
        ok("SDPAttention — cross-attention")
    except Exception as e:
        fail("SDPAttention cross-attention", e)


# ---------------------------------------------------------------------------
# RWKV Attention
# ---------------------------------------------------------------------------
def test_rwkw_attention():
    section("RWKV Attention")

    from nnx.attention import RWKVTimeMixing, RWKV6TimeMixing

    B, T, D = 2, 10, 128
    x = torch.randn(B, T, D)

    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 7:] = False  # first seq padded from pos 7
    mask[1, 9:] = False  # second seq padded from pos 9

    try:
        rwkv = RWKVTimeMixing(embed_dim=D, layer_id=0, n_layers=6)
        out = rwkv(x, attention_mask=mask)
        assert out.shape == x.shape
        ok("RWKVTimeMixing (RWKV-4)")
    except Exception as e:
        fail("RWKVTimeMixing", e)

    try:
        rwkv6 = RWKV6TimeMixing(embed_dim=D, layer_id=1, n_layers=6, n_heads=4)
        out = rwkv6(x, attention_mask=mask)
        assert out.shape == x.shape
        ok("RWKV6TimeMixing (RWKV-6)")
    except Exception as e:
        fail("RWKV6TimeMixing", e)


def test_linear_attention():
    section("Linear Attention")
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)

    from nnx.attention import LinearAttention
    from nnx.attention.linear import _VARIANTS

    B, T, D = 2, 10, 128
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 7:] = False
    mask[1, 9:] = False

    for variant in typing.get_args(_VARIANTS):
        try:
            linear_attn = LinearAttention(embed_dim=D, num_heads=4, variant=variant)
            out = linear_attn(x, attention_mask=mask)
            assert out.shape == x.shape
            ok(f"Linear attention ({variant})")
        except Exception as e:
            fail(f"LinearAttention ({variant})", e)



# ---------------------------------------------------------------------------
# TransformerLayer
# ---------------------------------------------------------------------------
def test_transformer_layer():
    section("TransformerLayer")

    from nnx.layers.transformer import TransformerLayer
    from nnx.layers.feedforward import FFN, GatedFFN
    from nnx.attention import SDPAttention

    B, T, D = 2, 10, 128
    x = torch.randn(B, T, D)

    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 7:] = False  # first seq padded from pos 7
    mask[1, 9:] = False  # second seq padded from pos 9

    try:
        layer = TransformerLayer(embed_dim=D, num_heads=4, causal=False, norm_type="rmsnorm")
        out = layer(x, attention_mask=mask)
        assert out.shape == x.shape
        ok("TransformerLayer — encoder (rmsnorm, SwiGLU, SDPA)")
    except Exception:
        fail("TransformerLayer encoder", None)

    try:
        layer = TransformerLayer(
            embed_dim=D,
            num_heads=4,
            causal=True,
            norm_type="layernorm",
            ffn_cls=FFN,
        )
        out = layer(x, attention_mask=mask)
        assert out.shape == x.shape
        ok("TransformerLayer — causal decoder (layernorm, FFN)")
    except Exception:
        fail("TransformerLayer decoder", None)

    try:
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
        ok("TransformerLayer — GeGLU FFN")
    except Exception:
        fail("TransformerLayer GeGLU", None)


# ---------------------------------------------------------------------------
# CrossAttentionLayer
# ---------------------------------------------------------------------------
def test_cross_attention_layer():
    section("CrossAttentionLayer")

    from nnx.layers.transformer import CrossAttentionLayer

    B, T, D = 2, 10, 128
    x = torch.randn(B, T, D)

    try:
        cross = CrossAttentionLayer(embed_dim=D, num_heads=4)
        enc = torch.randn(B, 20, D)
        enc_mask = torch.ones(B, 20, dtype=torch.bool)
        enc_mask[0, 15:] = False
        out = cross(x, encoder_out=enc, cross_attn_mask=enc_mask)
        assert out.shape == x.shape
        ok("CrossAttentionLayer")
    except Exception:
        fail("CrossAttentionLayer", None)


# ---------------------------------------------------------------------------
# TransformerStack
# ---------------------------------------------------------------------------
def test_transformer_stack():
    section("TransformerStack")

    from nnx.layers.transformer import TransformerStack
    from nnx.layers.feedforward import GatedFFN

    B, T, D = 2, 10, 128
    x = torch.randn(B, T, D)

    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 7:] = False  # first seq padded from pos 7
    mask[1, 9:] = False  # second seq padded from pos 9

    try:
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
        param_count = sum(p.numel() for p in stack.parameters())
        ok(f"TransformerStack (4 layers, {param_count:,} params)")
    except Exception:
        fail("TransformerStack", None)


# ---------------------------------------------------------------------------
# build_attention factory
# ---------------------------------------------------------------------------
def test_attention_factory():
    section("build_attention factory")

    from nnx.attention import build_attention

    B, T, D = 2, 10, 128
    x = torch.randn(B, T, D)

    try:
        attn = build_attention("sdpa", embed_dim=D, num_heads=4)
        out = attn(x)
        assert out.shape == x.shape
        ok("build_attention('sdpa')")
    except Exception:
        fail("build_attention sdpa", None)

    try:
        attn = build_attention("rwkv", embed_dim=D)
        out = attn(x)
        assert out.shape == x.shape
        ok("build_attention('rwkv')")
    except Exception:
        fail("build_attention rwkv", None)


# ---------------------------------------------------------------------------
# Top-level import
# ---------------------------------------------------------------------------
def test_top_level_import():
    section("Top-level nnx import")

    try:
        import nnx

        stack = nnx.TransformerStack(
            n_layers=2,
            embed_dim=64,
            num_heads=4,
            ffn_cls=nnx.GatedFFN,
            norm_type="rmsnorm",
            causal=True,
        )
        xx = torch.randn(1, 8, 64)
        mm = torch.ones(1, 8, dtype=torch.bool)
        out = stack(xx, attention_mask=mm)
        assert out.shape == xx.shape
        ok("nnx top-level import and full forward pass")
    except Exception:
        fail("nnx top-level", None)


print("\n" + "=" * 60)
print("  All tests complete.")
print("=" * 60 + "\n")
