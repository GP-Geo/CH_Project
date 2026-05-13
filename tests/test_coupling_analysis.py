"""Tests for coupling_analysis module."""

import numpy as np
import pytest

from channel_heads.coupling_analysis import CouplingAnalyzer, PairTouchResult


class TestPairTouchResult:
    """Test PairTouchResult dataclass."""

    def test_creation(self):
        """Test creating a PairTouchResult."""
        result = PairTouchResult(touching=True, contact_px=10, size1_px=100, size2_px=120)
        assert result.touching is True
        assert result.contact_px == 10
        assert result.size1_px == 100
        assert result.size2_px == 120

    def test_no_touching(self):
        """Test non-touching pair result."""
        result = PairTouchResult(touching=False, contact_px=0, size1_px=100, size2_px=120)
        assert result.touching is False
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
        # that should touch (contact pixels > 0)
        assert result.touching or result.contact_px > 0


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
        """Test with multiple confluences (stream filter disabled to count all pairs)."""
        net = complex_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        # Pairs at each confluence
        pairs_at_confluence = {
            4: {(0, 1)},  # first confluence
            5: {(2, 3)},  # second confluence
            6: {(0, 2), (0, 3), (1, 2), (1, 3)},  # main confluence
        }

        df = analyzer.evaluate_pairs_for_outlet(7, pairs_at_confluence, use_stream_filter=False)

        # Should have 1 + 1 + 4 = 6 pairs (no stream filter applied)
        assert len(df) == 6

        # All outlets should be 7
        assert (df["outlet"] == 7).all()

        # Heads should be normalized (head_1 < head_2)
        assert (df["head_1"] < df["head_2"]).all()


class TestStreamFilter:
    """Tests for the stream-crossing gate in evaluate_pairs_for_outlet."""

    def test_crosses_stream_no_interior(self, simple_y_network):
        """Adjacent heads with no interior stream pixels return False."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        # Heads 0 and 1 at (0,2) and (0,7); interior (0,3)-(0,6) are not stream nodes
        assert analyzer._crosses_stream(0, 1) is False

    def test_crosses_stream_touching_basins(self, touching_basins_network):
        """Heads with no stream in between return False (basins can touch)."""
        net = touching_basins_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        # Heads 0 and 1 at (0,1) and (0,4); interior (0,2),(0,3) are not stream nodes
        assert analyzer._crosses_stream(0, 1) is False

    def test_stream_filter_drops_crossing_pairs(self, complex_network):
        """Pairs whose head-to-head vector crosses a stream node are dropped."""
        net = complex_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        # complex_network heads: 0=(0,1), 1=(0,3), 2=(0,8), 3=(0,10)
        # Cross-side pairs (0,2), (0,3), (1,3) pass through other head positions
        # (which are stream nodes), so they cross the stream.
        # Pairs (0,1) and (2,3) do not cross any stream interior pixel.
        pairs_at_confluence = {
            4: {(0, 1)},
            5: {(2, 3)},
            6: {(0, 2), (0, 3), (1, 2), (1, 3)},
        }

        df = analyzer.evaluate_pairs_for_outlet(7, pairs_at_confluence, use_stream_filter=True)

        # (0,2): line (0,1)→(0,8) interior hits (0,3) which is stream → dropped
        # (0,3): line (0,1)→(0,10) interior hits (0,3) and (0,8) → dropped
        # (1,3): line (0,3)→(0,10) interior hits (0,8) → dropped
        # (0,1): interior (0,2) not stream → kept
        # (2,3): interior (0,9) not stream → kept
        # (1,2): interior (0,4)–(0,7) none are stream → kept
        assert len(df) == 3
        remaining = set(zip(df["head_1"], df["head_2"]))
        assert remaining == {(0, 1), (2, 3), (1, 2)}

    def test_stream_filter_disabled(self, complex_network):
        """With use_stream_filter=False all pairs are kept (old behavior)."""
        net = complex_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        pairs_at_confluence = {
            4: {(0, 1)},
            5: {(2, 3)},
            6: {(0, 2), (0, 3), (1, 2), (1, 3)},
        }

        df = analyzer.evaluate_pairs_for_outlet(7, pairs_at_confluence, use_stream_filter=False)
        assert len(df) == 6

    def test_stream_filter_keeps_touching_pair(self, touching_basins_network):
        """Stream filter does not drop genuinely coupled pairs."""
        net = touching_basins_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        pairs_at_confluence = {2: {(0, 1)}}
        df = analyzer.evaluate_pairs_for_outlet(4, pairs_at_confluence, use_stream_filter=True)

        # Pair should survive the filter
        assert len(df) == 1

    def test_parallel_stream_filter_matches_sequential(self, complex_network):
        """Parallel stream filter produces same result as sequential."""
        import pandas as pd

        net = complex_network
        pairs_at_confluence = {
            4: {(0, 1)},
            5: {(2, 3)},
            6: {(0, 2), (0, 3), (1, 2), (1, 3)},
        }

        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        df_seq = analyzer.evaluate_pairs_for_outlet(7, pairs_at_confluence, use_stream_filter=True)

        analyzer.clear_cache()
        df_par = analyzer.evaluate_pairs_for_outlet_parallel(
            7, pairs_at_confluence, n_workers=4, use_stream_filter=True
        )

        df_seq_s = df_seq.sort_values(["head_1", "head_2"]).reset_index(drop=True)
        df_par_s = df_par.sort_values(["head_1", "head_2"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(df_seq_s, df_par_s)
