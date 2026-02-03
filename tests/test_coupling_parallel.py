"""Tests for parallel processing and spatial pre-filtering in coupling_analysis."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest

from channel_heads.coupling_analysis import CouplingAnalyzer


class TestPrefilterBasic:
    """Test spatial pre-filtering basics."""

    def test_heads_can_touch_close_heads(self, simple_y_network):
        """Heads close together should pass pre-filter."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(
            net["fd"], net["s"], net["dem"], threshold=300, prefilter_multiplier=2.0
        )

        # Heads 0 and 1 in simple_y_network are relatively close
        # Head 0: (0, 2), Head 1: (0, 7)
        # Distance = sqrt((0-0)^2 + (2-7)^2) = 5
        # Threshold distance = 2 * sqrt(300) = ~34.6
        # 5 < 34.6, so they should pass
        assert analyzer._heads_can_touch(0, 1) is True

    def test_heads_can_touch_with_small_threshold(self, simple_y_network):
        """With small threshold, distant heads fail pre-filter."""
        net = simple_y_network
        # Use very small threshold: 2 * sqrt(1) = 2 pixels
        analyzer = CouplingAnalyzer(
            net["fd"], net["s"], net["dem"], threshold=1, prefilter_multiplier=1.0
        )

        # Head 0: (0, 2), Head 1: (0, 7)
        # Distance = 5 pixels > 1 pixel threshold
        assert analyzer._heads_can_touch(0, 1) is False

    def test_prefilter_distance_calculation(self, simple_y_network):
        """Test that pre-filter distance is correctly calculated."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(
            net["fd"], net["s"], net["dem"], threshold=100, prefilter_multiplier=3.0
        )

        # Expected: 3.0 * sqrt(100) = 30
        assert analyzer._prefilter_distance == pytest.approx(30.0)


class TestPrefilterSkipping:
    """Test that pre-filtering correctly skips distant pairs."""

    def test_prefilter_skips_distant_pairs(self, simple_y_network):
        """Pairs beyond distance threshold should be skipped."""
        net = simple_y_network
        # Use small threshold to force skipping
        analyzer = CouplingAnalyzer(
            net["fd"], net["s"], net["dem"], threshold=1, prefilter_multiplier=1.0
        )

        pairs_at_confluence = {4: {(0, 1)}}
        df = analyzer.evaluate_pairs_for_outlet(6, pairs_at_confluence, use_prefilter=True)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["skipped_prefilter"] == True  # noqa: E712
        assert row["touching"] == False  # noqa: E712
        assert row["overlap_px"] == 0
        assert row["contact_px"] == 0
        assert pd.isna(row["size1_px"])
        assert pd.isna(row["size2_px"])

    def test_prefilter_evaluates_close_pairs(self, simple_y_network):
        """Close pairs should be fully evaluated, not skipped."""
        net = simple_y_network
        # Use large threshold so pairs pass pre-filter
        analyzer = CouplingAnalyzer(
            net["fd"], net["s"], net["dem"], threshold=300, prefilter_multiplier=2.0
        )

        pairs_at_confluence = {4: {(0, 1)}}
        df = analyzer.evaluate_pairs_for_outlet(6, pairs_at_confluence, use_prefilter=True)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["skipped_prefilter"] == False  # noqa: E712
        # Should have actual size values (not None)
        assert pd.notna(row["size1_px"])
        assert pd.notna(row["size2_px"])

    def test_prefilter_disabled(self, simple_y_network):
        """With use_prefilter=False, all pairs should be evaluated."""
        net = simple_y_network
        # Use small threshold but disable prefilter
        analyzer = CouplingAnalyzer(
            net["fd"], net["s"], net["dem"], threshold=1, prefilter_multiplier=1.0
        )

        pairs_at_confluence = {4: {(0, 1)}}
        df = analyzer.evaluate_pairs_for_outlet(6, pairs_at_confluence, use_prefilter=False)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["skipped_prefilter"] == False  # noqa: E712
        assert pd.notna(row["size1_px"])
        assert pd.notna(row["size2_px"])


class TestPrefilterNoFalseNegatives:
    """Test that pre-filter never skips pairs that would actually touch."""

    def test_prefilter_no_false_negatives(self, touching_basins_network):
        """Pre-filter should never skip pairs that would actually touch."""
        net = touching_basins_network
        # Use default threshold which should be generous enough
        analyzer = CouplingAnalyzer(
            net["fd"], net["s"], net["dem"], threshold=300, prefilter_multiplier=2.0
        )

        pairs_at_confluence = {2: {(0, 1)}}

        # Run with prefilter enabled
        df_prefilter = analyzer.evaluate_pairs_for_outlet(
            4, pairs_at_confluence, use_prefilter=True
        )

        # Run without prefilter
        analyzer.clear_cache()
        df_no_prefilter = analyzer.evaluate_pairs_for_outlet(
            4, pairs_at_confluence, use_prefilter=False
        )

        # The touching result should be the same
        assert df_prefilter.iloc[0]["touching"] == df_no_prefilter.iloc[0]["touching"]

        # If the pair was actually touching, it shouldn't have been skipped
        if df_no_prefilter.iloc[0]["touching"]:
            assert df_prefilter.iloc[0]["skipped_prefilter"] == False  # noqa: E712

    def test_prefilter_conservative_for_touching_basins(self, touching_basins_network):
        """Verify touching basins are found with prefilter enabled."""
        net = touching_basins_network
        analyzer = CouplingAnalyzer(
            net["fd"], net["s"], net["dem"], threshold=300, prefilter_multiplier=2.0
        )

        pairs_at_confluence = {2: {(0, 1)}}
        df = analyzer.evaluate_pairs_for_outlet(4, pairs_at_confluence, use_prefilter=True)

        # Pair should not be skipped
        assert df.iloc[0]["skipped_prefilter"] == False  # noqa: E712


class TestThreadSafeCache:
    """Test thread safety of the mask cache."""

    def test_cache_thread_safety(self, simple_y_network):
        """Ensure concurrent mask requests don't corrupt cache."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        errors = []
        results = {}
        lock = threading.Lock()

        def get_mask(head_id, iteration):
            try:
                mask = analyzer.influence_mask(head_id)
                # Store result for verification
                with lock:
                    key = (head_id, iteration)
                    results[key] = mask.sum()
            except Exception as e:
                with lock:
                    errors.append(e)

        # Spawn N threads requesting masks for same heads simultaneously
        threads = []
        n_iterations = 50
        for i in range(n_iterations):
            for head_id in [0, 1]:
                t = threading.Thread(target=get_mask, args=(head_id, i))
                threads.append(t)

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads
        for t in threads:
            t.join()

        # Check for errors
        assert len(errors) == 0, f"Thread errors: {errors}"

        # Verify cache integrity - all calls for same head should return same size
        for head_id in [0, 1]:
            sizes = [results[(head_id, i)] for i in range(n_iterations)]
            assert all(s == sizes[0] for s in sizes), f"Inconsistent results for head {head_id}"

    def test_cache_concurrent_writes(self, complex_network):
        """Test cache handles concurrent writes correctly."""
        net = complex_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        n_threads = 20
        n_heads = 4  # heads 0, 1, 2, 3
        results = {}
        lock = threading.Lock()

        def access_masks(thread_id):
            for head_id in range(n_heads):
                mask = analyzer.influence_mask(head_id)
                with lock:
                    if head_id not in results:
                        results[head_id] = []
                    results[head_id].append((thread_id, mask.sum()))

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(access_masks, i) for i in range(n_threads)]
            for f in futures:
                f.result()

        # All threads should get the same mask for each head
        for head_id in range(n_heads):
            sizes = [r[1] for r in results[head_id]]
            assert len(set(sizes)) == 1, f"Inconsistent mask sizes for head {head_id}: {sizes}"


class TestParallelMatchesSequential:
    """Test that parallel results match sequential results."""

    def test_parallel_matches_sequential(self, simple_y_network):
        """Parallel processing should produce identical results to sequential."""
        net = simple_y_network
        pairs_at_confluence = {4: {(0, 1)}}

        # Sequential
        analyzer_seq = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        df_seq = analyzer_seq.evaluate_pairs_for_outlet(6, pairs_at_confluence, use_prefilter=True)

        # Parallel
        analyzer_par = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        df_par = analyzer_par.evaluate_pairs_for_outlet_parallel(
            6, pairs_at_confluence, n_workers=4, use_prefilter=True
        )

        # Compare (ignoring row order since parallel may return in different order)
        df_seq_sorted = df_seq.sort_values(["confluence", "head_1", "head_2"]).reset_index(
            drop=True
        )
        df_par_sorted = df_par.sort_values(["confluence", "head_1", "head_2"]).reset_index(
            drop=True
        )

        pd.testing.assert_frame_equal(df_seq_sorted, df_par_sorted)

    def test_parallel_matches_sequential_complex(self, complex_network):
        """Parallel should match sequential for complex network."""
        net = complex_network
        pairs_at_confluence = {
            4: {(0, 1)},
            5: {(2, 3)},
            6: {(0, 2), (0, 3), (1, 2), (1, 3)},
        }

        # Sequential
        analyzer_seq = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        df_seq = analyzer_seq.evaluate_pairs_for_outlet(7, pairs_at_confluence, use_prefilter=True)

        # Parallel
        analyzer_par = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        df_par = analyzer_par.evaluate_pairs_for_outlet_parallel(
            7, pairs_at_confluence, n_workers=4, use_prefilter=True
        )

        # Compare
        df_seq_sorted = df_seq.sort_values(["confluence", "head_1", "head_2"]).reset_index(
            drop=True
        )
        df_par_sorted = df_par.sort_values(["confluence", "head_1", "head_2"]).reset_index(
            drop=True
        )

        pd.testing.assert_frame_equal(df_seq_sorted, df_par_sorted)

    def test_parallel_matches_sequential_no_prefilter(self, complex_network):
        """Parallel should match sequential with prefilter disabled."""
        net = complex_network
        pairs_at_confluence = {
            4: {(0, 1)},
            5: {(2, 3)},
        }

        # Sequential without prefilter
        analyzer_seq = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        df_seq = analyzer_seq.evaluate_pairs_for_outlet(7, pairs_at_confluence, use_prefilter=False)

        # Parallel without prefilter
        analyzer_par = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        df_par = analyzer_par.evaluate_pairs_for_outlet_parallel(
            7, pairs_at_confluence, n_workers=4, use_prefilter=False
        )

        # Compare
        df_seq_sorted = df_seq.sort_values(["confluence", "head_1", "head_2"]).reset_index(
            drop=True
        )
        df_par_sorted = df_par.sort_values(["confluence", "head_1", "head_2"]).reset_index(
            drop=True
        )

        pd.testing.assert_frame_equal(df_seq_sorted, df_par_sorted)


class TestParallelStress:
    """Stress tests for race conditions."""

    def test_parallel_stress(self, complex_network):
        """Run many parallel evaluations to expose race conditions."""
        net = complex_network
        pairs_at_confluence = {
            4: {(0, 1)},
            5: {(2, 3)},
            6: {(0, 2), (0, 3), (1, 2), (1, 3)},
        }

        results = []
        n_iterations = 50

        for _ in range(n_iterations):
            analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
            df = analyzer.evaluate_pairs_for_outlet_parallel(
                7, pairs_at_confluence, n_workers=8, use_prefilter=True
            )
            # Store hash of sorted results
            df_sorted = df.sort_values(["confluence", "head_1", "head_2"]).reset_index(drop=True)
            results.append(df_sorted.to_json())

        # All results should be identical
        assert len(set(results)) == 1, "Inconsistent results across stress test iterations"

    def test_parallel_stress_with_cache_clearing(self, complex_network):
        """Stress test with cache clearing between iterations."""
        net = complex_network
        pairs_at_confluence = {4: {(0, 1)}, 5: {(2, 3)}}

        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])
        results = []

        for _ in range(30):
            df = analyzer.evaluate_pairs_for_outlet_parallel(
                7, pairs_at_confluence, n_workers=4, use_prefilter=True
            )
            df_sorted = df.sort_values(["confluence", "head_1", "head_2"]).reset_index(drop=True)
            results.append(df_sorted.to_json())
            analyzer.clear_cache()

        # All results should be identical
        assert len(set(results)) == 1, "Inconsistent results after cache clearing"


class TestParallelEmptyInput:
    """Test parallel processing with edge cases."""

    def test_parallel_empty_pairs(self, simple_y_network):
        """Parallel should handle empty pairs dict."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        df = analyzer.evaluate_pairs_for_outlet_parallel(6, {}, n_workers=4)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert "skipped_prefilter" in df.columns

    def test_parallel_single_pair(self, simple_y_network):
        """Parallel should work with single pair."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        pairs_at_confluence = {4: {(0, 1)}}
        df = analyzer.evaluate_pairs_for_outlet_parallel(6, pairs_at_confluence, n_workers=1)

        assert len(df) == 1


class TestNewParameters:
    """Test new initialization parameters."""

    def test_invalid_threshold(self, simple_y_network):
        """Test that invalid threshold raises ValueError."""
        net = simple_y_network
        with pytest.raises(ValueError, match="threshold must be positive"):
            CouplingAnalyzer(net["fd"], net["s"], net["dem"], threshold=0)

        with pytest.raises(ValueError, match="threshold must be positive"):
            CouplingAnalyzer(net["fd"], net["s"], net["dem"], threshold=-100)

    def test_invalid_prefilter_multiplier(self, simple_y_network):
        """Test that invalid prefilter_multiplier raises ValueError."""
        net = simple_y_network
        with pytest.raises(ValueError, match="prefilter_multiplier must be positive"):
            CouplingAnalyzer(net["fd"], net["s"], net["dem"], prefilter_multiplier=0)

        with pytest.raises(ValueError, match="prefilter_multiplier must be positive"):
            CouplingAnalyzer(net["fd"], net["s"], net["dem"], prefilter_multiplier=-1.0)

    def test_default_parameters(self, simple_y_network):
        """Test default parameter values."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        assert analyzer.threshold == 300
        assert analyzer.prefilter_multiplier == 2.0

    def test_custom_parameters(self, simple_y_network):
        """Test custom parameter values."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(
            net["fd"], net["s"], net["dem"], threshold=500, prefilter_multiplier=3.0
        )

        assert analyzer.threshold == 500
        assert analyzer.prefilter_multiplier == 3.0


class TestOutputColumns:
    """Test output DataFrame has correct columns."""

    def test_sequential_has_skipped_column(self, simple_y_network):
        """Sequential evaluate_pairs_for_outlet should have skipped_prefilter column."""
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

    def test_parallel_has_skipped_column(self, simple_y_network):
        """Parallel evaluate_pairs_for_outlet_parallel should have skipped_prefilter column."""
        net = simple_y_network
        analyzer = CouplingAnalyzer(net["fd"], net["s"], net["dem"])

        pairs_at_confluence = {4: {(0, 1)}}
        df = analyzer.evaluate_pairs_for_outlet_parallel(6, pairs_at_confluence)

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
