"""
Tests for mask utilities.
Run with: pytest tests/test_mask_utils.py -v
"""

import pytest
import torch

from nnx.utils.mask import hf_to_additive, make_causal_mask, combine_masks


class TestHfToAdditive:
    """Test suite for hf_to_additive function."""

    @pytest.fixture
    def B(self):
        return 2

    @pytest.fixture
    def T(self):
        return 10

    @pytest.fixture
    def mask(self, B, T):
        """Create a HuggingFace-style boolean mask."""
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 7:] = False  # first seq padded from pos 7
        mask[1, 9:] = False  # second seq padded from pos 9
        return mask

    def test_hf_to_additive(self, mask, B, T):
        """Test hf_to_additive shape and values."""
        additive = hf_to_additive(mask)
        assert additive.shape == (B, 1, 1, T)
        assert additive[0, 0, 0, 7] == float("-inf")
        assert additive[0, 0, 0, 6] == 0.0


class TestMakeCausalMask:
    """Test suite for make_causal_mask function."""

    @pytest.fixture
    def T(self):
        return 10

    def test_make_causal_mask(self, T):
        """Test make_causal_mask shape and values."""
        causal = make_causal_mask(T, device=torch.device("cpu"))
        assert causal.shape == (1, 1, T, T)
        assert causal[0, 0, 0, 1] == float("-inf")
        assert causal[0, 0, 1, 0] == 0.0


class TestCombineMasks:
    """Test suite for combine_masks function."""

    @pytest.fixture
    def B(self):
        return 2

    @pytest.fixture
    def T(self):
        return 10

    @pytest.fixture
    def mask(self, B, T):
        """Create a HuggingFace-style boolean mask."""
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[0, 7:] = False  # first seq padded from pos 7
        mask[1, 9:] = False  # second seq padded from pos 9
        return mask

    def test_combine_masks(self, mask, B, T):
        """Test combine_masks shape."""
        additive = hf_to_additive(mask)
        causal = make_causal_mask(T, device=torch.device("cpu"))
        combined = combine_masks(additive, causal)
        assert combined.shape == (B, 1, T, T)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
