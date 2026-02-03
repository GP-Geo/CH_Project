"""Tests for coupling_analysis module."""

import numpy as np
import pytest

from channel_heads.coupling_analysis import CouplingAnalyzer, PairTouchResult


class TestPairTouchResult:
    """Test PairTouchResult dataclass."""

    def test_creation(self):
        """Test creating a PairTouchResult."""
        result = PairTouchResult(
            touching=True, overlap_px=5, contact_px=10, size1_px=100, size2_px=120
        )
        assert result.touching is True
        assert result.overlap_px == 5
        assert result.contact_px == 10
        assert result.size1_px == 100
        assert result.size2_px == 120

    def test_no_touching(self):
        """Test non-touching pair result."""
        result = PairTouchResult(
            touching=False, overlap_px=0, contact_px=0, size1_px=100, size2_px=120
        )
        assert result.touching is False
        assert result.overlap_px == 0
        assert result.contact_px == 0


class TestCouplingAnalyzerInit:
    """Test CouplingAnalyzer initialization."""

    def test_init_with_mock_objects(self, simple_y_network):
        """Test initializing CouplingAnalyzer with mock objects."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        assert analyzer.fd is net["fd"]
        assert analyzer.s is net["s"]
        assert analyzer.dem is net["dem"]
        assert analyzer.connectivity == 8  # default

    def test_init_connectivity_4(self, simple_y_network):
        """Test initializing with connectivity=4."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"], connectivity=4)
        assert analyzer.connectivity == 4

    def test_init_invalid_connectivity(self, simple_y_network):
        """Test that invalid connectivity raises ValueError."""
        net = simple_y_network
        with pytest.raises(ValueError, match="connectivity must be 4 or 8"):
            CouplingAnalyzer(net["fd"], net["s"], net["dem"], connectivity=6)


class TestCouplingAnalyzerCache:
    """Test CouplingAnalyzer cache management."""

    def test_cache_starts_empty(self, simple_y_network):
        """Test that cache starts empty."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        assert analyzer.cache_size == 0

    def test_cache_populates_on_mask_access(self, simple_y_network):
        """Test that cache grows when masks are accessed."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        # Access mask for head 0
        _ = analyzer.influence_mask(0)
        assert analyzer.cache_size == 1

        # Access mask for head 1
        _ = analyzer.influence_mask(1)
        assert analyzer.cache_size == 2

        # Accessing same mask doesn't increase cache
        _ = analyzer.influence_mask(0)
        assert analyzer.cache_size == 2

    def test_clear_cache(self, simple_y_network):
        """Test clearing the cache."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        # Populate cache
        _ = analyzer.influence_mask(0)
        _ = analyzer.influence_mask(1)
        assert analyzer.cache_size == 2

        # Clear and verify
        n_cleared = analyzer.clear_cache()
        assert n_cleared == 2
        assert analyzer.cache_size == 0


class TestCouplingAnalyzerInfluenceMask:
    """Test influence mask computation."""

    def test_influence_mask_returns_bool_array(self, simple_y_network):
        """Test that influence_mask returns a boolean array."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        mask = analyzer.influence_mask(0)
        assert mask.dtype == bool
        assert mask.shape == net["grid_shape"]

    def test_influence_mask_contains_head_position(self, simple_y_network):
        """Test that the head's own position is in its influence mask."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        # Head 0 is at position (0, 2)
        mask = analyzer.influence_mask(0)
        assert mask[0, 2]

    def test_different_heads_have_different_masks(self, simple_y_network):
        """Test that different heads have different masks."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        mask0 = analyzer.influence_mask(0)
        mask1 = analyzer.influence_mask(1)

        # Masks should be different (different head positions)
        assert not np.array_equal(mask0, mask1)


class TestCouplingAnalyzerPairTouching:
    """Test pair touching detection."""

    def test_pair_touching_returns_result(self, simple_y_network):
        """Test that pair_touching returns a PairTouchResult."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        result = analyzer.pair_touching(0, 1)
        assert isinstance(result, PairTouchResult)
        assert isinstance(result.touching, bool)
        assert isinstance(result.overlap_px, int)
        assert isinstance(result.contact_px, int)
        assert isinstance(result.size1_px, int)
        assert isinstance(result.size2_px, int)

    def test_basin_sizes_positive(self, simple_y_network):
        """Test that basin sizes are positive."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        result = analyzer.pair_touching(0, 1)
        assert result.size1_px > 0
        assert result.size2_px > 0

    def test_touching_basins_detected(self, touching_basins_network):
        """Test that touching basins are correctly detected."""
        net = touching_basins_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        result = analyzer.pair_touching(0, 1)
        # In the touching_basins_network, heads 0 and 1 have adjacent basins
        # that should touch (either overlap or contact)
        assert result.touching or result.overlap_px > 0 or result.contact_px > 0


class TestCouplingAnalyzerEvaluatePairs:
    """Test evaluate_pairs_for_outlet method."""

    def test_evaluate_pairs_returns_dataframe(self, simple_y_network):
        """Test that evaluate_pairs_for_outlet returns a DataFrame."""
        import pandas as pd

        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        # Create a simple pairs dict
        pairs_at_confluence = {4: {(0, 1)}}  # confluence 4, heads 0 and 1

        df = analyzer.evaluate_pairs_for_outlet(6, pairs_at_confluence)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # one pair

    def test_evaluate_pairs_has_expected_columns(self, simple_y_network):
        """Test that result DataFrame has expected columns."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        pairs_at_confluence = {4: {(0, 1)}}
        df = analyzer.evaluate_pairs_for_outlet(6, pairs_at_confluence)

        expected_cols = [
            "outlet",
            "confluence",
            "head_1",
            "head_2",
            "touching",
            "overlap_px",
            "contact_px",
            "size1_px",
            "size2_px",
            "skipped_prefilter",
        ]
        assert list(df.columns) == expected_cols

    def test_evaluate_pairs_empty_input(self, simple_y_network):
        """Test with empty pairs dict."""
        import pandas as pd

        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        df = analyzer.evaluate_pairs_for_outlet(6, {})

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_evaluate_pairs_multiple_confluences(self, complex_network):
        """Test with multiple confluences."""
        net = complex_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        # Pairs at each confluence
        pairs_at_confluence = {
            4: {(0, 1)},  # first confluence
            5: {(2, 3)},  # second confluence
            6: {(0, 2), (0, 3), (1, 2), (1, 3)},  # main confluence
        }

        df = analyzer.evaluate_pairs_for_outlet(7, pairs_at_confluence)

        # Should have 2 + 4 = 6 pairs
        assert len(df) == 6

        # All outlets should be 7
        assert (df["outlet"] == 7).all()

        # Heads should be normalized (head_1 < head_2)
        assert (df["head_1"] < df["head_2"]).all()
