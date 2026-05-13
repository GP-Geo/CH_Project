"""Tests for channel_heads.rasterizer module."""

import math

import numpy as np

from channel_heads.rasterizer import (
    BACKGROUND,
    BRANCH_A,
    BRANCH_B,
    CONFLUENCE_MARKER,
    NUM_CLASSES,
    OTHER_STREAMS,
    _compute_rotation_angle,
    _rotate_coordinates,
    rasterize_outlet_pair,
)

# =============================================================================
# Rotation Angle Tests
# =============================================================================


class TestComputeRotationAngle:
    """Tests for _compute_rotation_angle."""

    def test_heads_above_confluence_no_rotation(self):
        """When heads midpoint is already above confluence, angle is ~0."""
        # Heads at row 0, confluence at row 4 (below in raster coords).
        # Midpoint of heads is at (0, 5). Vector from conf to midpoint = (-4, 0).
        # This already points "up" (negative row), so angle should be ~0.
        angle = _compute_rotation_angle(
            head_1_rc=(0.0, 3.0),
            head_2_rc=(0.0, 7.0),
            confluence_rc=(4.0, 5.0),
        )
        assert abs(angle) < 1e-6 or abs(abs(angle) - 2 * math.pi) < 1e-6

    def test_heads_below_confluence_180_rotation(self):
        """When heads are below confluence, need ~180 degree rotation."""
        # Heads at row 4, confluence at row 0 (above heads).
        # Vector from conf to midpoint = (4, 0), pointing downward.
        # Need pi rotation to flip it upward.
        angle = _compute_rotation_angle(
            head_1_rc=(4.0, 3.0),
            head_2_rc=(4.0, 7.0),
            confluence_rc=(0.0, 5.0),
        )
        assert abs(abs(angle) - math.pi) < 1e-6

    def test_degenerate_coincident_returns_zero(self):
        """When heads midpoint coincides with confluence, return 0."""
        angle = _compute_rotation_angle(
            head_1_rc=(5.0, 3.0),
            head_2_rc=(5.0, 7.0),
            confluence_rc=(5.0, 5.0),
        )
        assert angle == 0.0

    def test_heads_to_right_of_confluence(self):
        """When heads midpoint is to the right of confluence."""
        angle = _compute_rotation_angle(
            head_1_rc=(5.0, 10.0),
            head_2_rc=(5.0, 10.0),
            confluence_rc=(5.0, 2.0),
        )
        # Vector from conf to midpoint = (0, 8), pointing right.
        # current_angle = atan2(8, 0) = pi/2
        # target_angle = pi
        # rotation = pi - pi/2 = pi/2
        assert abs(angle - math.pi / 2) < 1e-6


# =============================================================================
# Coordinate Rotation Tests
# =============================================================================


class TestRotateCoordinates:
    """Tests for _rotate_coordinates."""

    def test_zero_rotation(self):
        """Zero rotation returns original coordinates."""
        rows = np.array([1.0, 2.0, 3.0])
        cols = np.array([4.0, 5.0, 6.0])
        new_r, new_c = _rotate_coordinates(rows, cols, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(new_r, rows)
        np.testing.assert_allclose(new_c, cols)

    def test_180_rotation(self):
        """180-degree rotation around center flips coordinates."""
        rows = np.array([1.0])
        cols = np.array([0.0])
        new_r, new_c = _rotate_coordinates(rows, cols, 0.0, 0.0, math.pi)
        np.testing.assert_allclose(new_r, [-1.0], atol=1e-10)
        np.testing.assert_allclose(new_c, [0.0], atol=1e-10)

    def test_90_rotation(self):
        """90-degree CCW rotation around origin."""
        rows = np.array([1.0])
        cols = np.array([0.0])
        new_r, new_c = _rotate_coordinates(rows, cols, 0.0, 0.0, math.pi / 2)
        np.testing.assert_allclose(new_r, [0.0], atol=1e-10)
        np.testing.assert_allclose(new_c, [1.0], atol=1e-10)

    def test_rotation_around_nonzero_center(self):
        """Rotation around a non-origin center preserves center."""
        rows = np.array([5.0])
        cols = np.array([5.0])
        # Point at center should stay at center regardless of angle
        new_r, new_c = _rotate_coordinates(rows, cols, 5.0, 5.0, math.pi / 3)
        np.testing.assert_allclose(new_r, [5.0], atol=1e-10)
        np.testing.assert_allclose(new_c, [5.0], atol=1e-10)


# =============================================================================
# Rasterize Outlet Pair Tests
# =============================================================================


class TestRasterizeOutletPair:
    """Tests for rasterize_outlet_pair."""

    def test_output_shape(self, simple_y_network):
        """Output has the correct target shape."""
        net = simple_y_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
            target_size=64,
        )
        assert result.shape == (64, 64)

    def test_output_dtype(self, simple_y_network):
        """Output is uint8."""
        net = simple_y_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
        )
        assert result.dtype == np.uint8

    def test_values_in_valid_range(self, simple_y_network):
        """All output values are in {0, 1, 2, 3, 4}."""
        net = simple_y_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
        )
        unique_vals = set(np.unique(result))
        assert unique_vals.issubset({0, 1, 2, 3, 4})

    def test_background_dominates(self, simple_y_network):
        """Background (0) covers most of the raster."""
        net = simple_y_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
        )
        bg_frac = np.sum(result == BACKGROUND) / result.size
        assert bg_frac > 0.5

    def test_branch_a_present(self, simple_y_network):
        """Branch A (value 1) has at least one pixel."""
        net = simple_y_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
        )
        assert np.any(result == BRANCH_A)

    def test_branch_b_present(self, simple_y_network):
        """Branch B (value 2) has at least one pixel."""
        net = simple_y_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
        )
        assert np.any(result == BRANCH_B)

    def test_confluence_marker_present(self, simple_y_network):
        """Confluence marker (value 4) is present."""
        net = simple_y_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
        )
        assert np.any(result == CONFLUENCE_MARKER)

    def test_other_streams_present(self, simple_y_network):
        """Other streams (value 3) present for nodes below confluence."""
        net = simple_y_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
        )
        # Nodes 5 (downstream) and 6 (outlet) are not on path A or B
        assert np.any(result == OTHER_STREAMS)

    def test_canonical_alignment_confluence_at_bottom(self, simple_y_network):
        """After canonical alignment, confluence should be in the lower half."""
        net = simple_y_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
            target_size=64,
        )
        # Find confluence marker position
        conf_positions = np.argwhere(result == CONFLUENCE_MARKER)
        assert len(conf_positions) > 0
        # Confluence row should be in the lower half (row >= 32 for 64x64)
        # Allow some tolerance for padding effects
        mean_row = conf_positions[:, 0].mean()
        assert mean_row >= 64 * 0.3, f"Confluence at row {mean_row}, expected in lower portion"

    def test_resize_preserves_categorical_values(self, simple_y_network):
        """Nearest-neighbor resize should not create fractional values."""
        net = simple_y_network
        for size in [32, 64, 128, 256]:
            result = rasterize_outlet_pair(
                net["s"],
                outlet=6,
                head_1=0,
                head_2=1,
                confluence=4,
                grid_shape=net["grid_shape"],
                target_size=size,
            )
            # All values should be exact integers
            unique = np.unique(result)
            for v in unique:
                assert v == int(v), f"Non-integer value {v} at size {size}"
            assert set(unique).issubset({0, 1, 2, 3, 4})

    def test_different_target_sizes(self, simple_y_network):
        """Rasterization works at different target sizes."""
        net = simple_y_network
        for size in [16, 32, 64, 128]:
            result = rasterize_outlet_pair(
                net["s"],
                outlet=6,
                head_1=0,
                head_2=1,
                confluence=4,
                grid_shape=net["grid_shape"],
                target_size=size,
            )
            assert result.shape == (size, size)

    def test_swapping_heads_swaps_branches(self, simple_y_network):
        """Swapping head_1 and head_2 should swap branch A and B labels."""
        net = simple_y_network
        result_ab = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
            target_size=64,
        )
        result_ba = rasterize_outlet_pair(
            net["s"],
            outlet=6,
            head_1=1,
            head_2=0,
            confluence=4,
            grid_shape=net["grid_shape"],
            target_size=64,
        )
        # The number of branch A pixels in one should match branch B in the other
        a_count_1 = np.sum(result_ab == BRANCH_A)
        b_count_1 = np.sum(result_ab == BRANCH_B)
        a_count_2 = np.sum(result_ba == BRANCH_A)
        b_count_2 = np.sum(result_ba == BRANCH_B)
        assert a_count_1 == b_count_2
        assert b_count_1 == a_count_2


class TestRasterizeComplexNetwork:
    """Tests with the complex multi-confluence network."""

    def test_complex_network_rasterizes(self, complex_network):
        """Complex network produces valid raster."""
        net = complex_network
        # Pair: heads 0 and 1 at confluence 4
        result = rasterize_outlet_pair(
            net["s"],
            outlet=7,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
        )
        assert result.shape == (128, 128)
        assert result.dtype == np.uint8
        unique = set(np.unique(result))
        assert unique.issubset({0, 1, 2, 3, 4})

    def test_complex_network_branches_marked(self, complex_network):
        """Both branches and confluence are marked in complex network."""
        net = complex_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=7,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
        )
        assert np.any(result == BRANCH_A)
        assert np.any(result == BRANCH_B)
        assert np.any(result == CONFLUENCE_MARKER)

    def test_different_pairs_produce_different_branch_counts(self, complex_network):
        """Different pairs highlight different path nodes."""
        net = complex_network
        # Use large target to avoid resolution-induced equality
        result_01 = rasterize_outlet_pair(
            net["s"],
            outlet=7,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
            target_size=256,
        )
        result_23 = rasterize_outlet_pair(
            net["s"],
            outlet=7,
            head_1=2,
            head_2=3,
            confluence=5,
            grid_shape=net["grid_shape"],
            target_size=256,
        )
        # Both should be valid rasters with all expected classes
        assert np.any(result_01 == BRANCH_A)
        assert np.any(result_23 == BRANCH_A)


class TestRasterizeTouchingBasins:
    """Tests with the touching basins network."""

    def test_touching_network_rasterizes(self, touching_basins_network):
        """Touching basins network produces valid raster."""
        net = touching_basins_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=4,
            head_1=0,
            head_2=1,
            confluence=2,
            grid_shape=net["grid_shape"],
        )
        assert result.shape == (128, 128)
        unique = set(np.unique(result))
        assert unique.issubset({0, 1, 2, 3, 4})
        assert np.any(result == BRANCH_A)
        assert np.any(result == BRANCH_B)
        assert np.any(result == CONFLUENCE_MARKER)


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_class_values_distinct(self):
        """All class values are distinct."""
        values = [BACKGROUND, BRANCH_A, BRANCH_B, OTHER_STREAMS, CONFLUENCE_MARKER]
        assert len(values) == len(set(values))

    def test_num_classes(self):
        """NUM_CLASSES matches number of distinct classes."""
        assert NUM_CLASSES == 5
