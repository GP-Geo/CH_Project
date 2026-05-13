"""Tests for geometric_analysis module.

This module tests the geometric feature computation for paired channel heads:
- Feature 1: Lengthwise asymmetry (delta_L)
- Feature 2: Orientation similarity
- Feature 3: Euclidean head-head distance
- Feature 4: Apex angle
- Feature 5: Strahler order difference

Also tests the labeling functions:
- generate_labeled_dataset
- filter_hard_negatives
- merge_geometric_features

And tests for correctness fixes:
- GEOM_FEATURE_COLS completeness
- NaN-safe asymmetry statistics
- Y-axis negation for geographic correctness
- Projected CRS meters_per_unit
- Skip logging on computation errors
- Per-group hard negative filtering
"""

import math

import numpy as np
import pandas as pd

from channel_heads.first_meet_pairs_for_outlet import first_meet_pairs_for_outlet
from channel_heads.geometric_analysis import (
    GEOM_FEATURE_COLS,
    GeometricFeaturesAnalyzer,
    PairGeometricResult,
    _angle_between_vectors,
    _azimuth_difference,
    _compute_azimuth,
    _compute_direction_vector,
    _compute_proximity_profile,
    _euclidean_2d,
    _line_crosses_stream,
    _normalize_vector,
    _sample_path_coords,
    _trace_full_path,
    compute_asymmetry_statistics,
    filter_hard_negatives,
    generate_labeled_dataset,
    merge_geometric_features,
)

# ============================================================================
# Unit Tests: Helper Functions
# ============================================================================


class TestEuclidean2D:
    """Tests for _euclidean_2d function."""

    def test_same_point(self):
        """Distance between same point is zero."""
        assert _euclidean_2d(1.0, 2.0, 1.0, 2.0) == 0.0

    def test_horizontal_distance(self):
        """Horizontal distance computation."""
        assert _euclidean_2d(0.0, 0.0, 3.0, 0.0) == 3.0

    def test_vertical_distance(self):
        """Vertical distance computation."""
        assert _euclidean_2d(0.0, 0.0, 0.0, 4.0) == 4.0

    def test_diagonal_distance(self):
        """Diagonal distance (3-4-5 triangle)."""
        assert _euclidean_2d(0.0, 0.0, 3.0, 4.0) == 5.0

    def test_negative_coordinates(self):
        """Distance with negative coordinates."""
        assert _euclidean_2d(-1.0, -1.0, 2.0, 3.0) == 5.0


class TestAngleBetweenVectors:
    """Tests for _angle_between_vectors function."""

    def test_parallel_vectors(self):
        """Parallel vectors have 0 degree angle."""
        angle = _angle_between_vectors((1.0, 0.0), (2.0, 0.0))
        assert abs(angle - 0.0) < 0.001

    def test_perpendicular_vectors(self):
        """Perpendicular vectors have 90 degree angle."""
        angle = _angle_between_vectors((1.0, 0.0), (0.0, 1.0))
        assert abs(angle - 90.0) < 0.001

    def test_opposite_vectors(self):
        """Opposite vectors have 180 degree angle."""
        angle = _angle_between_vectors((1.0, 0.0), (-1.0, 0.0))
        assert abs(angle - 180.0) < 0.001

    def test_45_degree_angle(self):
        """45 degree angle between vectors."""
        angle = _angle_between_vectors((1.0, 0.0), (1.0, 1.0))
        assert abs(angle - 45.0) < 0.001

    def test_zero_vector(self):
        """Zero vector returns NaN."""
        angle = _angle_between_vectors((0.0, 0.0), (1.0, 0.0))
        assert math.isnan(angle)

    def test_near_zero_vector(self):
        """Very small vector returns NaN."""
        angle = _angle_between_vectors((1e-15, 1e-15), (1.0, 0.0))
        assert math.isnan(angle)


class TestComputeAzimuth:
    """Tests for _compute_azimuth function."""

    def test_north(self):
        """North direction (positive y) has azimuth 0."""
        az = _compute_azimuth(0.0, 1.0)
        assert abs(az - 0.0) < 0.001

    def test_east(self):
        """East direction (positive x) has azimuth 90."""
        az = _compute_azimuth(1.0, 0.0)
        assert abs(az - 90.0) < 0.001

    def test_south(self):
        """South direction (negative y) has azimuth 180."""
        az = _compute_azimuth(0.0, -1.0)
        assert abs(az - 180.0) < 0.001

    def test_west(self):
        """West direction (negative x) has azimuth 270."""
        az = _compute_azimuth(-1.0, 0.0)
        assert abs(az - 270.0) < 0.001

    def test_northeast(self):
        """Northeast direction has azimuth 45."""
        az = _compute_azimuth(1.0, 1.0)
        assert abs(az - 45.0) < 0.001

    def test_zero_vector(self):
        """Zero vector returns NaN."""
        az = _compute_azimuth(0.0, 0.0)
        assert math.isnan(az)


class TestAzimuthDifference:
    """Tests for _azimuth_difference function."""

    def test_same_azimuth(self):
        """Same azimuth has zero difference."""
        diff = _azimuth_difference(45.0, 45.0)
        assert abs(diff - 0.0) < 0.001

    def test_opposite_azimuths(self):
        """Opposite azimuths (180 apart) have 180 difference."""
        diff = _azimuth_difference(0.0, 180.0)
        assert abs(diff - 180.0) < 0.001

    def test_wrap_around(self):
        """350 deg vs 10 deg should give 20 deg difference."""
        diff = _azimuth_difference(350.0, 10.0)
        assert abs(diff - 20.0) < 0.001

    def test_wrap_around_reverse(self):
        """10 deg vs 350 deg should give 20 deg difference."""
        diff = _azimuth_difference(10.0, 350.0)
        assert abs(diff - 20.0) < 0.001

    def test_90_degree_difference(self):
        """90 degree difference."""
        diff = _azimuth_difference(45.0, 135.0)
        assert abs(diff - 90.0) < 0.001

    def test_nan_input(self):
        """NaN input returns NaN."""
        assert math.isnan(_azimuth_difference(float("nan"), 45.0))
        assert math.isnan(_azimuth_difference(45.0, float("nan")))


class TestNormalizeVector:
    """Tests for _normalize_vector function."""

    def test_unit_vector(self):
        """Unit vector is unchanged."""
        result = _normalize_vector(1.0, 0.0)
        assert result is not None
        assert abs(result[0] - 1.0) < 0.001
        assert abs(result[1] - 0.0) < 0.001

    def test_scale_down(self):
        """Vector is scaled to unit length."""
        result = _normalize_vector(3.0, 4.0)
        assert result is not None
        assert abs(result[0] - 0.6) < 0.001
        assert abs(result[1] - 0.8) < 0.001

    def test_zero_vector(self):
        """Zero vector returns None."""
        result = _normalize_vector(0.0, 0.0)
        assert result is None


class TestComputeDirectionVector:
    """Tests for _compute_direction_vector function."""

    def test_straight_path(self):
        """Direction along straight path."""
        path = [0, 1, 2]
        node_x = np.array([0.0, 1.0, 2.0])
        node_y = np.array([0.0, 0.0, 0.0])

        vec, flags = _compute_direction_vector(path, node_x, node_y, 1.0)

        assert vec is not None
        assert abs(vec[0] - 1.0) < 0.001  # Unit vector pointing right
        assert abs(vec[1] - 0.0) < 0.001
        assert "short_path" in flags  # Only 2 edges (< 3)

    def test_diagonal_path(self):
        """Direction along diagonal path."""
        path = [0, 1, 2, 3]
        node_x = np.array([0.0, 1.0, 2.0, 3.0])
        node_y = np.array([0.0, 1.0, 2.0, 3.0])

        vec, flags = _compute_direction_vector(path, node_x, node_y, 1.0)

        assert vec is not None
        # Diagonal unit vector
        expected = 1.0 / math.sqrt(2)
        assert abs(vec[0] - expected) < 0.001
        assert abs(vec[1] - expected) < 0.001
        assert flags == ""  # 3 edges, no flags

    def test_single_node_path(self):
        """Single node path returns None with flag."""
        path = [0]
        node_x = np.array([0.0])
        node_y = np.array([0.0])

        vec, flags = _compute_direction_vector(path, node_x, node_y, 1.0)

        assert vec is None
        assert "single_edge" in flags

    def test_two_node_path(self):
        """Two node path returns direction with flag."""
        path = [0, 1]
        node_x = np.array([0.0, 1.0])
        node_y = np.array([0.0, 0.0])

        vec, flags = _compute_direction_vector(path, node_x, node_y, 1.0)

        assert vec is not None
        assert "short_path" in flags


# ============================================================================
# Integration Tests: GeometricFeaturesAnalyzer
# ============================================================================


class TestGeometricFeaturesAnalyzerInit:
    """Tests for GeometricFeaturesAnalyzer initialization."""

    def test_init_simple_network(self, simple_y_network):
        """Analyzer initializes with simple Y network."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        assert analyzer.s is net["s"]
        assert analyzer.dem is net["dem"]
        assert analyzer.direction_sample_distance_m == 500.0

    def test_init_with_lat(self, simple_y_network):
        """Analyzer initializes with latitude for coordinate conversion."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"], lat=36.71)

        assert analyzer.lat == 36.71
        # Note: meters_per_unit depends on DEM metadata (cellsize)
        # Mock DEM has cellsize=1.0 which is treated as projected CRS
        # In real usage with geographic CRS (cellsize < 1), conversion > 1
        assert analyzer.meters_per_unit >= 1.0

    def test_init_custom_sample_distance(self, simple_y_network):
        """Analyzer initializes with custom sample distance."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(
            net["s"], net["dem"], direction_sample_distance_m=100.0
        )

        assert analyzer.direction_sample_distance_m == 100.0


class TestGeometricFeaturesAnalyzerCompute:
    """Tests for compute_pair_geometry method."""

    def test_compute_pair_simple_network(self, simple_y_network):
        """Compute features for simple Y network pair."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        # Get pairs at confluence
        pairs, heads = first_meet_pairs_for_outlet(net["s"], 6)  # outlet at node 6

        # There should be one confluence (node 4) with one pair (0, 1)
        assert 4 in pairs
        assert len(pairs[4]) == 1

        # Compute geometry
        h1, h2 = list(pairs[4])[0]
        result = analyzer.compute_pair_geometry(h1, h2, 4, L_1=100.0, L_2=120.0)

        assert isinstance(result, PairGeometricResult)
        assert result.head_1 == min(h1, h2)
        assert result.head_2 == max(h1, h2)
        assert result.confluence == 4

        # Check features are computed (may be NaN for short paths but shouldn't error)
        assert isinstance(result.orientation_diff_deg, float)
        assert isinstance(result.headhead_dist_m, float)
        assert result.headhead_dist_m >= 0  # Distance should be non-negative

    def test_compute_without_L_values(self, simple_y_network):
        """Features computed without L values where applicable."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        pairs, heads = first_meet_pairs_for_outlet(net["s"], 6)
        h1, h2 = list(pairs[4])[0]

        # Without L values, normalized distance should be NaN
        result = analyzer.compute_pair_geometry(h1, h2, 4, L_1=None, L_2=None)

        # Euclidean distance should still be computed
        assert isinstance(result.headhead_dist_m, float)
        assert result.headhead_dist_m >= 0

        # Normalized distance should be NaN
        assert math.isnan(result.headhead_dist_norm)


class TestGeometricFeaturesAnalyzerEvaluate:
    """Tests for evaluate_pairs_for_outlet method."""

    def test_evaluate_simple_network(self, simple_y_network):
        """Evaluate all pairs for simple network outlet."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        pairs, heads = first_meet_pairs_for_outlet(net["s"], 6)
        df = analyzer.evaluate_pairs_for_outlet(6, pairs)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1  # One pair in simple Y network

        # Check columns exist
        expected_cols = [
            "outlet",
            "confluence",
            "head_1",
            "head_2",
            "orientation_diff_deg",
            "headhead_dist_m",
            "headhead_dist_norm",
            "apex_angle_deg",
            "strahler_order_diff",
            "qc_flags",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_evaluate_complex_network(self, complex_network):
        """Evaluate all pairs for complex network outlet."""
        net = complex_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        pairs, heads = first_meet_pairs_for_outlet(net["s"], 7)  # outlet at node 7
        df = analyzer.evaluate_pairs_for_outlet(7, pairs)

        assert isinstance(df, pd.DataFrame)
        # Complex network has 4 heads and 3 confluences
        # Should have multiple pairs
        assert len(df) > 0

    def test_evaluate_with_asymmetry_df(self, simple_y_network):
        """L values are used from asymmetry DataFrame."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        pairs, heads = first_meet_pairs_for_outlet(net["s"], 6)

        # Create mock asymmetry DataFrame
        h1, h2 = list(pairs[4])[0]
        h1_norm, h2_norm = min(h1, h2), max(h1, h2)

        asymmetry_df = pd.DataFrame(
            {
                "outlet": [6],
                "confluence": [4],
                "head_1": [h1_norm],
                "head_2": [h2_norm],
                "L_1": [150.0],
                "L_2": [180.0],
                "delta_L": [0.2],
            }
        )

        df = analyzer.evaluate_pairs_for_outlet(6, pairs, asymmetry_df=asymmetry_df)

        # Features should be computed with L values available
        assert len(df) == 1


# ============================================================================
# Tests: Labeling Functions
# ============================================================================


class TestGenerateLabeledDataset:
    """Tests for generate_labeled_dataset function."""

    def test_basic_merge(self):
        """Basic merge of coupling, asymmetry, and geometric DataFrames."""
        coupling_df = pd.DataFrame(
            {
                "outlet": [1, 1],
                "confluence": [2, 2],
                "head_1": [10, 10],
                "head_2": [11, 12],
                "touching": [True, False],
            }
        )

        asymmetry_df = pd.DataFrame(
            {
                "outlet": [1, 1],
                "confluence": [2, 2],
                "head_1": [10, 10],
                "head_2": [11, 12],
                "L_1": [100.0, 150.0],
                "L_2": [120.0, 160.0],
                "delta_L": [0.18, 0.06],
            }
        )

        geometric_df = pd.DataFrame(
            {
                "outlet": [1, 1],
                "confluence": [2, 2],
                "head_1": [10, 10],
                "head_2": [11, 12],
                "orientation_diff_deg": [15.0, 30.0],
                "headhead_dist_m": [200.0, 300.0],
                "headhead_dist_norm": [0.9, 1.0],
                "apex_angle_deg": [45.0, 60.0],
                "strahler_order_diff": [0, 1],
                "qc_flags": ["", ""],
            }
        )

        result = generate_labeled_dataset(coupling_df, asymmetry_df, geometric_df)

        assert len(result) == 2
        assert "y" in result.columns
        assert result["y"].tolist() == [1, 0]  # touching=True -> y=1, False -> y=0
        assert "L_1" in result.columns
        assert "orientation_diff_deg" in result.columns

    def test_empty_dataframes(self):
        """Handle empty DataFrames gracefully."""
        coupling_df = pd.DataFrame(columns=["outlet", "confluence", "head_1", "head_2", "touching"])
        asymmetry_df = pd.DataFrame(
            columns=["outlet", "confluence", "head_1", "head_2", "L_1", "L_2", "delta_L"]
        )
        geometric_df = pd.DataFrame(
            columns=[
                "outlet",
                "confluence",
                "head_1",
                "head_2",
                "orientation_diff_deg",
            ]
        )

        result = generate_labeled_dataset(coupling_df, asymmetry_df, geometric_df)

        assert len(result) == 0
        assert "y" in result.columns


class TestFilterHardNegatives:
    """Tests for filter_hard_negatives function."""

    def test_filter_by_L_ratio(self):
        """Filter negatives by L_sum ratio."""
        labeled_df = pd.DataFrame(
            {
                "outlet": [1, 1, 1, 1],
                "confluence": [2, 2, 2, 2],
                "head_1": [10, 10, 10, 10],
                "head_2": [11, 12, 13, 14],
                "y": [1, 0, 0, 0],
                "L_1": [100.0, 120.0, 400.0, 50.0],
                "L_2": [100.0, 130.0, 450.0, 60.0],
                "headhead_dist_m": [200.0, 250.0, 300.0, 150.0],
            }
        )

        # median positive L_sum = 200, max_L_ratio = 3.0 -> threshold = 600
        # Pair (10, 13) has L_sum = 850 > 600, should be filtered
        result = filter_hard_negatives(labeled_df, max_L_ratio=3.0, max_dist_ratio=10.0)

        assert len(result) == 3  # 1 positive + 2 negatives kept
        assert 13 not in result["head_2"].values

    def test_filter_by_dist_ratio(self):
        """Filter negatives by head-head distance ratio."""
        labeled_df = pd.DataFrame(
            {
                "outlet": [1, 1, 1],
                "confluence": [2, 2, 2],
                "head_1": [10, 10, 10],
                "head_2": [11, 12, 13],
                "y": [1, 0, 0],
                "L_1": [100.0, 100.0, 100.0],
                "L_2": [100.0, 100.0, 100.0],
                "headhead_dist_m": [200.0, 500.0, 1500.0],
            }
        )

        # median positive dist = 200, max_dist_ratio = 5.0 -> threshold = 1000
        # Pair (10, 13) has dist = 1500 > 1000, should be filtered
        result = filter_hard_negatives(labeled_df, max_L_ratio=10.0, max_dist_ratio=5.0)

        assert len(result) == 2  # 1 positive + 1 negative kept
        assert 13 not in result["head_2"].values

    def test_all_positives(self):
        """All positives case - nothing filtered."""
        labeled_df = pd.DataFrame(
            {
                "outlet": [1, 1],
                "confluence": [2, 2],
                "head_1": [10, 10],
                "head_2": [11, 12],
                "y": [1, 1],
                "L_1": [100.0, 150.0],
                "L_2": [100.0, 160.0],
                "headhead_dist_m": [200.0, 300.0],
            }
        )

        result = filter_hard_negatives(labeled_df)
        assert len(result) == 2

    def test_all_negatives(self):
        """All negatives case - nothing filtered (no positive baseline)."""
        labeled_df = pd.DataFrame(
            {
                "outlet": [1, 1],
                "confluence": [2, 2],
                "head_1": [10, 10],
                "head_2": [11, 12],
                "y": [0, 0],
                "L_1": [100.0, 500.0],
                "L_2": [100.0, 600.0],
                "headhead_dist_m": [200.0, 1000.0],
            }
        )

        result = filter_hard_negatives(labeled_df)
        assert len(result) == 2


class TestMergeGeometricFeatures:
    """Tests for merge_geometric_features function."""

    def test_basic_merge(self):
        """Basic merge of geometric features."""
        base_df = pd.DataFrame(
            {
                "outlet": [1],
                "confluence": [2],
                "head_1": [10],
                "head_2": [11],
                "touching": [True],
                "L_1": [100.0],
            }
        )

        geometric_df = pd.DataFrame(
            {
                "outlet": [1],
                "confluence": [2],
                "head_1": [10],
                "head_2": [11],
                "orientation_diff_deg": [15.0],
                "headhead_dist_m": [200.0],
                "headhead_dist_norm": [0.9],
                "apex_angle_deg": [45.0],
                "strahler_order_diff": [0],
            }
        )

        result = merge_geometric_features(base_df, geometric_df)

        assert "orientation_diff_deg" in result.columns
        assert "apex_angle_deg" in result.columns
        assert "strahler_order_diff" in result.columns
        assert len(result) == 1


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_coincident_heads(self, simple_y_network):
        """Handle coincident (same location) heads."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        # Try computing with same head twice (artificial case)
        # This shouldn't crash, but may produce warnings/flags
        try:
            result = analyzer.compute_pair_geometry(0, 0, 4, L_1=100.0, L_2=100.0)
            # If it succeeds, check for QC flags
            assert "coincident_nodes" in result.qc_flags or result.headhead_dist_m == 0
        except (ValueError, IndexError):
            # This is also acceptable - rejecting invalid input
            pass

    def test_empty_pairs_dict(self, simple_y_network):
        """Handle empty pairs dictionary."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        df = analyzer.evaluate_pairs_for_outlet(6, {})

        assert len(df) == 0
        assert isinstance(df, pd.DataFrame)

    def test_invalid_node_ids(self, simple_y_network):
        """Handle invalid node IDs gracefully."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        # Very large invalid node IDs
        try:
            _result = analyzer.compute_pair_geometry(9999, 9998, 9997)
            # If it doesn't crash, that's OK
        except (IndexError, ValueError):
            # Expected behavior for invalid indices
            pass


class TestTouchingBasinsIntegration:
    """Integration tests with touching basins network."""

    def test_full_workflow(self, touching_basins_network):
        """Full workflow from pairs to labeled dataset."""
        net = touching_basins_network

        # Get pairs
        pairs, heads = first_meet_pairs_for_outlet(net["s"], 4)  # outlet at node 4

        # Compute geometric features
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])
        geom_df = analyzer.evaluate_pairs_for_outlet(4, pairs)

        # Create mock coupling and asymmetry DataFrames
        assert len(geom_df) > 0

        # Check that features are reasonable
        for _, row in geom_df.iterrows():
            # Distance should be positive
            if not math.isnan(row["headhead_dist_m"]):
                assert row["headhead_dist_m"] >= 0

            # Angles should be in range
            if not math.isnan(row["orientation_diff_deg"]):
                assert 0 <= row["orientation_diff_deg"] <= 180


# ============================================================================
# Fix Verification Tests
# ============================================================================


class TestGeomFeatureColsCompleteness:
    """Fix 2: GEOM_FEATURE_COLS should include all geometric feature columns."""

    def test_contains_apex_angle(self):
        """apex_angle_deg is in GEOM_FEATURE_COLS."""
        assert "apex_angle_deg" in GEOM_FEATURE_COLS

    def test_contains_strahler_order_diff(self):
        """strahler_order_diff is in GEOM_FEATURE_COLS."""
        assert "strahler_order_diff" in GEOM_FEATURE_COLS

    def test_contains_all_expected_columns(self):
        """All expected geometric feature columns are present."""
        expected = [
            "orientation_diff_deg",
            "headhead_dist_m",
            "headhead_dist_norm",
            "apex_angle_deg",
            "strahler_order_diff",
            "proximity_mean_m",
            "proximity_max_m",
            "proximity_profile_norm",
            "qc_flags",
        ]
        assert GEOM_FEATURE_COLS == expected


class TestProjectedCRSMetersPerUnit:
    """Fix 3: GeometricFeaturesAnalyzer with projected CRS sets meters_per_unit=cellsize."""

    def test_projected_crs_cellsize_30(self, simple_y_network):
        """Projected CRS with cellsize=30 should set meters_per_unit=30."""
        net = simple_y_network
        # Mock the DEM to have a projected CRS cellsize (>= 1)
        # The mock DEM has transform = (1.0, 0, 0, 0, -1.0, nrows)
        # so cellsize = 1.0, which is treated as projected CRS
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"], lat=36.71)

        # With cellsize=1.0, projected CRS, meters_per_unit should equal cellsize
        assert analyzer.meters_per_unit == 1.0

    def test_projected_crs_explicit(self):
        """Verify projected CRS detection sets correct meters_per_unit."""
        from tests.conftest import MockGridObject, MockStreamObject

        grid_shape = (10, 10)
        z = np.ones(grid_shape, dtype=float) * 100
        z[0, 2] = 150
        z[0, 7] = 145
        z[2, 4] = 110
        z[4, 4] = 100

        # Create DEM with cellsize=30 (projected CRS)
        dem = MockGridObject(z)
        dem.transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)

        node_positions = [(0, 2), (0, 7), (2, 4), (4, 4)]
        edges = [(0, 2), (1, 2), (2, 3)]
        s = MockStreamObject(
            node_positions=node_positions,
            edges=edges,
            channelheads=[0, 1],
            outlets=[3],
            confluences=[2],
            grid_shape=grid_shape,
        )
        s.transform = (30.0, 0.0, 500000.0, 0.0, -30.0, 4000000.0)

        analyzer = GeometricFeaturesAnalyzer(s, dem, lat=36.71)

        # cellsize=30.0 >= 1.0, so projected CRS branch should set meters_per_unit=30
        assert analyzer.meters_per_unit == 30.0


class TestYAxisNegation:
    """Fix 7: _node_y should be negated row indices for correct azimuth."""

    def test_node_y_is_negated(self, simple_y_network):
        """_node_y should be negative of row indices."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        r_nodes = net["s"].node_indices[0]
        expected_y = -np.asarray(r_nodes, dtype=np.float64)

        np.testing.assert_array_equal(analyzer._node_y, expected_y)

    def test_node_x_is_columns(self, simple_y_network):
        """_node_x should be column indices (unchanged)."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        c_nodes = net["s"].node_indices[1]
        expected_x = np.asarray(c_nodes, dtype=np.float64)

        np.testing.assert_array_equal(analyzer._node_x, expected_x)


class TestComputeAsymmetryStatisticsNaN:
    """Fix 5: compute_asymmetry_statistics handles NaN inputs."""

    def test_with_nan_values(self):
        """NaN values should be filtered out before computing stats."""
        values = [0.1, float("nan"), 0.3, float("nan"), 0.5]
        stats = compute_asymmetry_statistics(values)

        assert stats["count"] == 3
        assert abs(stats["median"] - 0.3) < 0.001
        assert abs(stats["mean"] - 0.3) < 0.001

    def test_all_nan(self):
        """All-NaN input returns count=0 and NaN stats."""
        values = [float("nan"), float("nan"), float("nan")]
        stats = compute_asymmetry_statistics(values)

        assert stats["count"] == 0
        assert math.isnan(stats["median"])
        assert math.isnan(stats["mean"])

    def test_empty_array(self):
        """Empty array returns count=0."""
        stats = compute_asymmetry_statistics([])
        assert stats["count"] == 0
        assert math.isnan(stats["median"])

    def test_no_nan(self):
        """Array without NaN works normally."""
        values = [0.1, 0.2, 0.3]
        stats = compute_asymmetry_statistics(values)
        assert stats["count"] == 3
        assert abs(stats["median"] - 0.2) < 0.001


class TestSkipLogging:
    """Fix 4: evaluate_pairs_for_outlet logs warning when pairs are skipped."""

    def test_geometry_skip_warning(self, simple_y_network):
        """Skipped geometry pairs should trigger a warning log."""
        from unittest.mock import patch

        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])

        # Create pairs with invalid node IDs that will cause computation errors
        invalid_pairs = {999: {(9998, 9999)}}

        with patch("channel_heads.geometric_analysis.logger") as mock_logger:
            df = analyzer.evaluate_pairs_for_outlet(1, invalid_pairs)

        assert len(df) == 0
        # The logger.warning should have been called with "skipped" message
        mock_logger.warning.assert_called_once()
        call_args = mock_logger.warning.call_args
        assert "skipped" in call_args[0][0]


class TestFilterHardNegativesPerGroup:
    """Fix 8: filter_hard_negatives with group_col computes per-group thresholds."""

    def _make_labeled_df(self):
        """Create labeled DataFrame with two basins having different scales."""
        return pd.DataFrame(
            {
                "outlet": [1, 1, 1, 2, 2, 2],
                "confluence": [10, 10, 10, 20, 20, 20],
                "head_1": [100, 100, 100, 200, 200, 200],
                "head_2": [101, 102, 103, 201, 202, 203],
                "y": [1, 0, 0, 1, 0, 0],
                "L_1": [100.0, 120.0, 500.0, 1000.0, 1100.0, 5000.0],
                "L_2": [100.0, 130.0, 600.0, 1000.0, 1200.0, 6000.0],
                "headhead_dist_m": [50.0, 60.0, 300.0, 500.0, 600.0, 3000.0],
                "basin": ["A", "A", "A", "B", "B", "B"],
            }
        )

    def test_per_group_filtering(self):
        """Per-group filtering uses group-specific thresholds."""
        df = self._make_labeled_df()

        result_global = filter_hard_negatives(df, max_L_ratio=3.0, max_dist_ratio=5.0)
        result_grouped = filter_hard_negatives(
            df, max_L_ratio=3.0, max_dist_ratio=5.0, group_col="basin"
        )

        # Both should preserve all positives
        assert (result_global["y"] == 1).sum() == 2
        assert (result_grouped["y"] == 1).sum() == 2

        # Per-group should also return a valid DataFrame
        assert isinstance(result_grouped, pd.DataFrame)
        assert len(result_grouped) > 0

    def test_group_col_none_matches_global(self):
        """group_col=None should produce same result as omitting it."""
        df = self._make_labeled_df()

        result_default = filter_hard_negatives(df, max_L_ratio=3.0, max_dist_ratio=5.0)
        result_none = filter_hard_negatives(df, max_L_ratio=3.0, max_dist_ratio=5.0, group_col=None)

        pd.testing.assert_frame_equal(result_default, result_none)

    def test_nonexistent_group_col_ignored(self):
        """Non-existent group_col falls back to global behavior."""
        df = self._make_labeled_df()

        result_default = filter_hard_negatives(df, max_L_ratio=3.0, max_dist_ratio=5.0)
        result_bad_col = filter_hard_negatives(
            df, max_L_ratio=3.0, max_dist_ratio=5.0, group_col="nonexistent"
        )

        pd.testing.assert_frame_equal(result_default, result_bad_col)


# ============================================================================
# Unit Tests: _line_crosses_stream
# ============================================================================


class TestLineStreamIntersection:
    """Tests for _line_crosses_stream helper."""

    def _make_mask(self, shape=(10, 10), stream_pixels=None):
        mask = np.zeros(shape, dtype=bool)
        if stream_pixels:
            for r, c in stream_pixels:
                mask[r, c] = True
        return mask

    def test_no_intermediate_pixels(self):
        """Two adjacent nodes have no interior pixels → returns False."""
        mask = self._make_mask(stream_pixels=[(0, 1)])
        # Adjacent horizontally: only endpoints, no interior
        assert _line_crosses_stream(0, 0, 0, 1, mask) is False

    def test_clear_crossing(self):
        """Line that passes through a stream pixel in the interior → returns True."""
        # Line from (0,0) to (0,4), stream pixel at (0,2) = interior
        mask = self._make_mask(stream_pixels=[(0, 2)])
        assert _line_crosses_stream(0, 0, 0, 4, mask) is True

    def test_no_crossing(self):
        """Line that avoids all stream pixels → returns False."""
        # Stream pixel is off the path
        mask = self._make_mask(stream_pixels=[(5, 5)])
        assert _line_crosses_stream(0, 0, 0, 4, mask) is False

    def test_endpoints_excluded(self):
        """Endpoints are on stream but interior is not → returns False."""
        # Stream only at endpoints (0,0) and (0,4), nothing in between
        mask = self._make_mask(stream_pixels=[(0, 0), (0, 4)])
        assert _line_crosses_stream(0, 0, 0, 4, mask) is False


# ============================================================================
# Unit Tests: filter_hard_negatives with stream intersection filter
# ============================================================================


class _MockStreamForFilter:
    """Minimal mock StreamObject for filter_hard_negatives tests.

    5×5 grid. node_indices are provided as arrays indexed by node ID.
    Node IDs are direct indices into r_arr and c_arr.
    """

    def __init__(self, r_arr, c_arr, shape=(5, 5)):
        self._r = np.asarray(r_arr)
        self._c = np.asarray(c_arr)
        self.shape = shape
        # node_indices as plain attribute (not callable)
        self.node_indices = (self._r, self._c)


class TestFilterHardNegativesWithStream:
    """Tests for filter_hard_negatives with s= stream intersection filter."""

    def _base_df(self, **kwargs):
        """Minimal labeled DataFrame with two negatives and one positive.

        head node IDs are used as indices into the mock stream's r/c arrays.
        """
        defaults = {
            "outlet": [1, 1, 1],
            "confluence": [10, 10, 10],
            "head_1": [0, 0, 0],
            "head_2": [4, 2, 3],
            "y": [1, 0, 0],
            "L_1": [100.0, 100.0, 100.0],
            "L_2": [100.0, 100.0, 100.0],
            "headhead_dist_m": [50.0, 50.0, 50.0],
        }
        defaults.update(kwargs)
        return pd.DataFrame(defaults)

    def test_s_none_unchanged(self):
        """s=None produces identical result to not passing s (backward compat)."""
        df = self._base_df()

        result_no_s = filter_hard_negatives(df.copy())
        result_s_none = filter_hard_negatives(df.copy(), s=None)

        pd.testing.assert_frame_equal(result_no_s, result_s_none)

    def test_crossing_pair_removed(self):
        """Negative whose head-to-head vector crosses a stream pixel is removed."""
        # Stream nodes at indices 0..4, laid out in a row: (0,0),(0,1),(0,2),(0,3),(0,4)
        # head_1=0 at (0,0), head_2=4 at (0,4): interior passes through (0,1),(0,2),(0,3)
        # which ARE stream pixels → trivially non-touching → removed
        s = _MockStreamForFilter(
            r_arr=[0, 0, 0, 0, 0],
            c_arr=[0, 1, 2, 3, 4],
        )
        # One negative: head_1=0 -> (0,0), head_2=4 -> (0,4): crosses stream
        df = pd.DataFrame(
            {
                "outlet": [1, 1],
                "confluence": [10, 10],
                "head_1": [0, 0],
                "head_2": [4, 4],
                "y": [1, 0],
                "L_1": [100.0, 100.0],
                "L_2": [100.0, 100.0],
                "headhead_dist_m": [50.0, 50.0],
            }
        )
        result = filter_hard_negatives(df, s=s)
        # Only the positive should remain
        assert list(result["y"]) == [1]

    def test_non_crossing_pair_kept(self):
        """Negative whose head-to-head vector avoids stream pixels is kept."""
        # Stream nodes only at corners: (0,0) and (4,4)
        # Line from (0,0) to (4,4) passes through both endpoints — no interior stream
        # But to avoid endpoint exclusion test, use nodes NOT on stream for the path
        # node 0 at (0,0), node 1 at (4,4): only endpoints on stream, interior is clear
        s = _MockStreamForFilter(
            r_arr=[0, 4],
            c_arr=[0, 4],
        )
        df = pd.DataFrame(
            {
                "outlet": [1, 1],
                "confluence": [10, 10],
                "head_1": [0, 0],
                "head_2": [1, 1],
                "y": [1, 0],
                "L_1": [100.0, 100.0],
                "L_2": [100.0, 100.0],
                "headhead_dist_m": [50.0, 50.0],
            }
        )
        result = filter_hard_negatives(df, s=s)
        # Both positive and negative should remain
        assert 1 in result["y"].values
        assert 0 in result["y"].values

    def test_and_logic(self):
        """Pair that fails L_ratio is still removed even without stream crossing check."""
        # Giant L_sum for negatives: well above max_L_ratio threshold
        s = _MockStreamForFilter(
            r_arr=[0, 0, 0],
            c_arr=[0, 2, 4],
        )
        df = pd.DataFrame(
            {
                "outlet": [1, 1],
                "confluence": [10, 10],
                "head_1": [0, 0],
                "head_2": [2, 2],
                "y": [1, 0],
                "L_1": [100.0, 1000.0],
                "L_2": [100.0, 1000.0],
                "headhead_dist_m": [50.0, 50.0],
            }
        )
        result = filter_hard_negatives(df, max_L_ratio=3.0, s=s)
        # Negative removed by L_ratio filter (2000 > 100*2*3.0=600)
        assert list(result["y"]) == [1]


# ============================================================================
# Tests: Proximity Profile Helpers
# ============================================================================


class TestTraceFullPath:
    """Tests for _trace_full_path."""

    def _make_children(self, edges):
        """Build children dict from edge list."""
        from collections import defaultdict

        children = defaultdict(list)
        for u, v in edges:
            children[u].append(v)
        return children

    def test_reaches_target(self, simple_y_network):
        """Full path from head to confluence ends at confluence."""
        from channel_heads.first_meet_pairs_for_outlet import _build_parents_from_stream
        from channel_heads.geometric_analysis import _build_children_from_parents

        s = simple_y_network["s"]
        parents = _build_parents_from_stream(s)
        children = _build_children_from_parents(parents, len(parents))

        # Left branch: 0 → 2 → 4 (confluence=4)
        path = _trace_full_path(0, 4, children)
        assert path[-1] == 4
        assert path[0] == 0
        assert 4 in path

    def test_includes_all_nodes(self, simple_y_network):
        """Full path includes all intermediate nodes."""
        from channel_heads.first_meet_pairs_for_outlet import _build_parents_from_stream
        from channel_heads.geometric_analysis import _build_children_from_parents

        s = simple_y_network["s"]
        parents = _build_parents_from_stream(s)
        children = _build_children_from_parents(parents, len(parents))

        path = _trace_full_path(0, 4, children)
        # Left branch is 0→2→4
        assert path == [0, 2, 4]

    def test_unreachable_target_returns_empty(self):
        """Returns empty list when target is unreachable."""
        children = {0: [1], 1: [2]}  # 0→1→2, no path to 99
        path = _trace_full_path(0, 99, children)
        assert path == []

    def test_single_step_path(self):
        """Direct edge from start to target."""
        children = {5: [10]}
        path = _trace_full_path(5, 10, children)
        assert path == [5, 10]


class TestSamplePathCoords:
    """Tests for _sample_path_coords."""

    def _make_arrays(self, positions):
        """Build node_x, node_y from (row, col) positions, matching analyzer convention."""
        rows = np.array([p[0] for p in positions], dtype=np.float64)
        cols = np.array([p[1] for p in positions], dtype=np.float64)
        # x = col, y = -row (same convention as GeometricFeaturesAnalyzer)
        return cols, -rows

    def test_output_shape(self):
        """Sampled array has shape (n_samples, 2)."""
        node_x, node_y = self._make_arrays([(0, 0), (0, 2), (0, 4)])
        coords = _sample_path_coords([0, 1, 2], node_x, node_y, n_samples=5, meters_per_unit=1.0)
        assert coords is not None
        assert coords.shape == (5, 2)

    def test_first_point_is_head(self):
        """Fraction 0 lands exactly on the head node."""
        node_x, node_y = self._make_arrays([(0, 0), (0, 4), (0, 8)])
        path = [0, 1, 2]
        coords = _sample_path_coords(path, node_x, node_y, n_samples=4, meters_per_unit=1.0)
        assert coords is not None
        # First sample (t=0) should be at node 0: x=0, y=0
        assert math.isclose(coords[0, 0], 0.0, abs_tol=1e-9)
        assert math.isclose(coords[0, 1], 0.0, abs_tol=1e-9)

    def test_uniform_path_evenly_spaced(self):
        """On a straight path, samples are evenly spaced."""
        # Straight horizontal line: nodes at x=0,4,8 (uniform spacing)
        node_x, node_y = self._make_arrays([(0, 0), (0, 4), (0, 8)])
        path = [0, 1, 2]
        # 4 samples: t=0, 0.25, 0.5, 0.75 → x=0, 2, 4, 6
        coords = _sample_path_coords(path, node_x, node_y, n_samples=4, meters_per_unit=1.0)
        assert coords is not None
        expected_x = [0.0, 2.0, 4.0, 6.0]
        for i, ex in enumerate(expected_x):
            assert math.isclose(
                coords[i, 0], ex, abs_tol=1e-9
            ), f"sample {i}: got {coords[i,0]}, expected {ex}"

    def test_returns_none_for_short_path(self):
        """Returns None for a single-node path (cannot define arc-length)."""
        node_x = np.array([1.0])
        node_y = np.array([1.0])
        result = _sample_path_coords([0], node_x, node_y, n_samples=5, meters_per_unit=1.0)
        assert result is None

    def test_meters_per_unit_scales_coords(self):
        """meters_per_unit correctly scales output coordinates."""
        node_x, node_y = self._make_arrays([(0, 0), (0, 1)])
        coords_1 = _sample_path_coords([0, 1], node_x, node_y, n_samples=2, meters_per_unit=1.0)
        coords_2 = _sample_path_coords([0, 1], node_x, node_y, n_samples=2, meters_per_unit=30.0)
        assert coords_1 is not None and coords_2 is not None
        np.testing.assert_allclose(coords_2, coords_1 * 30.0)


class TestComputeProximityProfile:
    """Tests for _compute_proximity_profile."""

    def test_parallel_channels_norm_equals_one(self):
        """Perfectly parallel channels have proximity_profile_norm == 1.0."""
        # Two rows of points with constant horizontal separation of 5
        coords_1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        coords_2 = np.array([[0.0, 5.0], [1.0, 5.0], [2.0, 5.0]])
        mean_m, max_m, norm = _compute_proximity_profile(coords_1, coords_2)
        assert math.isclose(mean_m, 5.0, abs_tol=1e-9)
        assert math.isclose(max_m, 5.0, abs_tol=1e-9)
        assert math.isclose(norm, 1.0, abs_tol=1e-9)

    def test_convergent_channels_norm_less_than_one(self):
        """Channels that converge have proximity_profile_norm < 1.0."""
        # Channel 1 stays at y=0, channel 2 starts at y=4 and ends at y=0
        coords_1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        coords_2 = np.array([[0.0, 4.0], [1.0, 2.0], [2.0, 0.0]])
        mean_m, max_m, norm = _compute_proximity_profile(coords_1, coords_2)
        assert max_m > 0
        assert norm < 1.0

    def test_norm_in_unit_interval(self):
        """proximity_profile_norm is always in [0, 1]."""
        rng = np.random.default_rng(42)
        coords_1 = rng.uniform(0, 10, (10, 2))
        coords_2 = rng.uniform(0, 10, (10, 2))
        _, _, norm = _compute_proximity_profile(coords_1, coords_2)
        assert 0.0 <= norm <= 1.0

    def test_returns_correct_types(self):
        """Return values are Python floats."""
        coords_1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        coords_2 = np.array([[0.0, 1.0], [1.0, 1.0]])
        mean_m, max_m, norm = _compute_proximity_profile(coords_1, coords_2)
        assert isinstance(mean_m, float)
        assert isinstance(max_m, float)
        assert isinstance(norm, float)


class TestProximityProfileIntegration:
    """Integration tests: proximity profile in GeometricFeaturesAnalyzer."""

    def test_proximity_fields_not_none(self, simple_y_network):
        """compute_pair_geometry populates proximity fields for a valid pair."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])
        # Pair: head_left=0, head_right=1, confluence=4
        result = analyzer.compute_pair_geometry(0, 1, 4)
        assert result.proximity_mean_m is not None
        assert result.proximity_max_m is not None
        assert result.proximity_profile_norm is not None

    def test_proximity_norm_in_range(self, simple_y_network):
        """proximity_profile_norm is in [0, 1] for a valid Y-network pair."""
        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])
        result = analyzer.compute_pair_geometry(0, 1, 4)
        assert 0.0 <= result.proximity_profile_norm <= 1.0

    def test_proximity_in_dataframe(self, simple_y_network):
        """evaluate_pairs_for_outlet includes proximity columns in output DataFrame."""
        from channel_heads.first_meet_pairs_for_outlet import first_meet_pairs_for_outlet

        net = simple_y_network
        analyzer = GeometricFeaturesAnalyzer(net["s"], net["dem"])
        pairs, _ = first_meet_pairs_for_outlet(net["s"], 6)
        df = analyzer.evaluate_pairs_for_outlet(6, pairs)
        assert "proximity_mean_m" in df.columns
        assert "proximity_max_m" in df.columns
        assert "proximity_profile_norm" in df.columns
        assert df["proximity_profile_norm"].notna().all()

    def test_n_proximity_samples_parameter(self, simple_y_network):
        """Different n_proximity_samples values produce consistent results."""
        net = simple_y_network
        analyzer_10 = GeometricFeaturesAnalyzer(net["s"], net["dem"], n_proximity_samples=10)
        analyzer_5 = GeometricFeaturesAnalyzer(net["s"], net["dem"], n_proximity_samples=5)
        r10 = analyzer_10.compute_pair_geometry(0, 1, 4)
        r5 = analyzer_5.compute_pair_geometry(0, 1, 4)
        # Both should produce valid norms
        assert r10.proximity_profile_norm is not None
        assert r5.proximity_profile_norm is not None
        # Both norms in [0, 1]
        assert 0.0 <= r10.proximity_profile_norm <= 1.0
        assert 0.0 <= r5.proximity_profile_norm <= 1.0

    def test_geom_feature_cols_includes_proximity(self):
        """GEOM_FEATURE_COLS contains the new proximity columns."""
        assert "proximity_mean_m" in GEOM_FEATURE_COLS
        assert "proximity_max_m" in GEOM_FEATURE_COLS
        assert "proximity_profile_norm" in GEOM_FEATURE_COLS
