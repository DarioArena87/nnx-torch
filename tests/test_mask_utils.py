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


class TestMaskCaching:
    """Test suite for mask caching functionality."""

    def test_get_cached_causal_mask_returns_cached(self):
        """Test get_cached_causal_mask() returns cached mask on second call."""
        from nnx.utils.mask import get_cached_causal_mask, clear_mask_cache
        clear_mask_cache()  # Start fresh
        
        seq_len = 10
        device = torch.device("cpu")
        dtype = torch.float32
        
        # First call - creates new mask
        mask1 = get_cached_causal_mask(seq_len, device, dtype)
        # Second call - should return cached mask
        mask2 = get_cached_causal_mask(seq_len, device, dtype)
        
        # Should be the exact same object (cached)
        assert mask1 is mask2

    def test_clear_mask_cache(self):
        """Test clear_mask_cache() empties the cache."""
        from nnx.utils.mask import get_cached_causal_mask, clear_mask_cache, _mask_cache
        clear_mask_cache()
        
        # Create a mask to populate cache
        get_cached_causal_mask(10, torch.device("cpu"), torch.float32)
        assert len(_mask_cache) > 0
        
        # Clear cache
        clear_mask_cache()
        assert len(_mask_cache) == 0

    def test_cache_size_limit(self):
        """Test cache size limit (64 entries)."""
        from nnx.utils.mask import get_cached_causal_mask, clear_mask_cache, _mask_cache, _MAX_MASK_CACHE_SIZE
        clear_mask_cache()
        
        # Create more masks than the cache limit
        for i in range(_MAX_MASK_CACHE_SIZE + 10):
            get_cached_causal_mask(i + 1, torch.device("cpu"), torch.float32)
        
        # Cache should not exceed the limit
        assert len(_mask_cache) <= _MAX_MASK_CACHE_SIZE

    def test_mask_device_dtype_correct(self):
        """Test masks are device/dtype correct."""
        from nnx.utils.mask import get_cached_causal_mask, clear_mask_cache
        clear_mask_cache()
        
        seq_len = 8
        device = torch.device("cpu")
        dtype = torch.float64
        
        mask = get_cached_causal_mask(seq_len, device, dtype)
        assert mask.device.type == device.type
        assert mask.dtype == dtype
        assert mask.shape == (1, 1, seq_len, seq_len)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
