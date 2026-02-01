"""Tests for first_meet_pairs_for_outlet module."""

import pytest
from channel_heads.first_meet_pairs_for_outlet import _normalize_pair, _to_node_id_list
import numpy as np


class TestNormalizePair:
    """Test pair normalization function."""

    def test_already_normalized(self):
        """Test pair that's already in correct order."""
        assert _normalize_pair(3, 5) == (3, 5)

    def test_needs_swap(self):
        """Test pair that needs swapping."""
        assert _normalize_pair(5, 3) == (3, 5)

    def test_equal_values(self):
        """Test pair with equal values."""
        assert _normalize_pair(4, 4) == (4, 4)

    def test_large_values(self):
        """Test with large node IDs."""
        assert _normalize_pair(1000, 500) == (500, 1000)


class TestToNodeIdList:
    """Test node ID list conversion."""

    def test_boolean_mask(self):
        """Test conversion from boolean mask."""
        mask = np.array([False, True, False, True, True])
        result = _to_node_id_list(mask)
        assert result == [1, 3, 4]

    def test_index_array(self):
        """Test conversion from index array."""
        indices = np.array([5, 2, 8, 2])  # with duplicate
        result = _to_node_id_list(indices)
        assert result == [2, 5, 8]  # sorted and unique

    def test_empty_mask(self):
        """Test empty mask."""
        mask = np.array([False, False, False])
        result = _to_node_id_list(mask)
        assert result == []

    def test_all_true_mask(self):
        """Test all-true mask."""
        mask = np.array([True, True, True])
        result = _to_node_id_list(mask)
        assert result == [0, 1, 2]


# Note: Testing first_meet_pairs_for_outlet() fully requires
# mock StreamObject with .source, .target, .node_indices, and .streampoi()
# See improvement.md for fixture creation recommendations.
