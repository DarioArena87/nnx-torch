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
