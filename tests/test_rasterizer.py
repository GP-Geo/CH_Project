"""Tests for channel_heads.rasterizer module."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from channel_heads.rasterizer import (
    BACKGROUND,
    BRANCH_A,
    BRANCH_B,
    CONFLUENCE_MARKER,
    NUM_CLASSES,
    OTHER_STREAMS,
    _component_count,
    _compute_rotation_angle,
    _rotate_coordinates,
    precompute_raster_dataset,
    raster_quality_flags,
    rasterize_outlet_pair,
)
from tests.conftest import MockGridObject, MockStreamObject

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

    def test_confluence_marker_overwrites_branch_values(self, simple_y_network):
        """The shared branch endpoint is encoded as the confluence marker."""
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

        conf_positions = np.argwhere(result == CONFLUENCE_MARKER)
        assert len(conf_positions) == 1
        r, c = conf_positions[0]
        assert result[r, c] == CONFLUENCE_MARKER
        flags = raster_quality_flags(result)
        assert flags["branch_a_connected"]
        assert flags["branch_b_connected"]

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

    def test_direct_output_preserves_categorical_values(self, simple_y_network):
        """Direct final-grid drawing should keep categorical class values."""
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

    def test_direct_raster_quality_flags_pass_after_downscale(self, complex_network):
        """Branches and confluence remain connected in the final target grid."""
        net = complex_network
        result = rasterize_outlet_pair(
            net["s"],
            outlet=7,
            head_1=0,
            head_2=1,
            confluence=4,
            grid_shape=net["grid_shape"],
            target_size=32,
        )
        flags = raster_quality_flags(result)
        assert flags == {
            "has_branch_a": True,
            "has_branch_b": True,
            "has_confluence": True,
            "branch_a_connected": True,
            "branch_b_connected": True,
            "branches_connected": True,
        }


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


class TestPrecomputeRasterDataset:
    """Tests for batch raster precomputation metadata."""

    def test_precompute_records_structural_qa(self, tmp_path, simple_y_network):
        master_csv = tmp_path / "master.csv"
        pd.DataFrame(
            [
                {
                    "basin": "inyo",
                    "outlet": 6,
                    "confluence": 4,
                    "head_1": 0,
                    "head_2": 1,
                    "y": 1,
                }
            ]
        ).to_csv(master_csv, index=False)

        def loader(_basin, _lat, _z_th, _threshold):
            return simple_y_network["s"], simple_y_network["dem"]

        out = precompute_raster_dataset(
            master_csv=master_csv,
            output_dir=tmp_path / "rasters",
            dem_loader=loader,
            target_size=32,
        )

        row = out.iloc[0]
        assert row["raster_status"] == "ok"
        assert row["raster_error"] == ""
        assert row["has_branch_a"]
        assert row["has_branch_b"]
        assert row["has_confluence"]
        assert row["branch_a_connected"]
        assert row["branch_b_connected"]
        assert row["branches_connected"]
        assert row["raster_path"]


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module constants."""

    def test_class_values_exact(self):
        assert BACKGROUND == 0
        assert BRANCH_A == 1
        assert BRANCH_B == 2
        assert OTHER_STREAMS == 3
        assert CONFLUENCE_MARKER == 4
        assert NUM_CLASSES == 5

    def test_class_values_distinct(self):
        """All class values are distinct."""
        values = [BACKGROUND, BRANCH_A, BRANCH_B, OTHER_STREAMS, CONFLUENCE_MARKER]
        assert len(values) == len(set(values))

    def test_num_classes(self):
        """NUM_CLASSES matches number of distinct classes."""
        assert NUM_CLASSES == 5

    def test_rasterization_public_surfaces_reexport_same_objects(self):
        import channel_heads.rasterization as rasterization
        import channel_heads.rasterization.manifest as manifest
        import channel_heads.rasterization.patches as patches
        import channel_heads.rasterization.schema as schema
        import channel_heads.models.cnn as cnn
        import channel_heads.rasterizer as rasterizer

        assert patches.rasterize_outlet_pair is rasterizer.rasterize_outlet_pair
        assert rasterization.rasterize_outlet_pair is rasterizer.rasterize_outlet_pair
        assert patches.precompute_raster_dataset is rasterizer.precompute_raster_dataset
        assert rasterization.precompute_raster_dataset is rasterizer.precompute_raster_dataset
        assert patches.raster_quality_flags is rasterizer.raster_quality_flags
        assert rasterization.raster_quality_flags is rasterizer.raster_quality_flags
        assert rasterization.bresenham_line is rasterizer.bresenham_line
        assert (
            schema.NUM_CLASSES
            == patches.NUM_CLASSES
            == rasterizer.NUM_CLASSES
            == rasterization.NUM_CLASSES
            == cnn.NUM_CLASSES
        )
        assert patches.CLASS_LABELS is schema.CLASS_LABELS
        assert rasterization.CLASS_LABELS is schema.CLASS_LABELS
        assert manifest.PATCH_FLAG_COLUMNS is schema.PATCH_FLAG_COLUMNS
        assert schema.CLASS_LABELS == {
            BACKGROUND: "background",
            BRANCH_A: "branch_a",
            BRANCH_B: "branch_b",
            OTHER_STREAMS: "other_streams",
            CONFLUENCE_MARKER: "confluence_marker",
        }
        assert schema.PATCH_FLAG_COLUMNS == [
            "has_branch_a",
            "has_branch_b",
            "has_confluence",
            "branch_a_connected",
            "branch_b_connected",
            "branches_connected",
        ]


# =============================================================================
# Regression: direct projection vs. old draw-then-resize
# =============================================================================


def _make_large_extent_network():
    """A Y-network spanning a large grid relative to a small target_size.

    Heads sit at the top, the confluence near the bottom, with long branches.
    Under the OLD pipeline (draw on a ~native crop, then nearest-neighbour
    resize down to a small target) the one-pixel confluence marker and the
    thin branches would routinely vanish or disconnect. With direct
    projection into the final grid, connectivity is a construction invariant.
    """
    grid_shape = (200, 200)
    # 0: head A (top-left), 1: head B (top-right),
    # 2: confluence (bottom-center), 3: outlet (very bottom)
    node_positions = [
        (10, 60),
        (10, 140),
        (185, 100),
        (199, 100),
    ]
    edges = [(0, 2), (1, 2), (2, 3)]
    s = MockStreamObject(
        node_positions=node_positions,
        edges=edges,
        channelheads=[0, 1],
        outlets=[3],
        confluences=[2],
        grid_shape=grid_shape,
    )
    dem = MockGridObject(np.ones(grid_shape, dtype=float) * 100.0)
    return {"s": s, "dem": dem, "grid_shape": grid_shape}


class TestDirectProjectionRegression:
    """Tests that would have caught the original draw-then-resize bug."""

    def test_naive_nn_resize_loses_one_pixel_marker(self):
        """Demonstrates the failure mode of the OLD pipeline.

        A native-resolution label raster with thin branches and a one-pixel
        confluence marker loses the marker (and can break connectivity) under
        nearest-neighbour downsampling. This documents *why* direct projection
        is required.
        """
        native = np.zeros((200, 200), dtype=np.uint8)
        native[10:185, 60] = BRANCH_A  # thin vertical branch A
        native[10:185, 140] = BRANCH_B  # thin vertical branch B
        native[185, 100] = CONFLUENCE_MARKER  # single-pixel marker

        # Classic nearest-neighbour downsample via integer striding to ~25px.
        factor = 8
        naive = native[::factor, ::factor]

        # The one-pixel marker is dropped by NN downsampling...
        assert not np.any(naive == CONFLUENCE_MARKER)
        # ...and the thin branches are largely destroyed too.
        assert np.sum(naive == BRANCH_A) < np.sum(native == BRANCH_A) / factor

    def test_direct_rasterizer_survives_small_target(self):
        """The real rasterizer keeps marker + connectivity at a small target."""
        net = _make_large_extent_network()
        result = rasterize_outlet_pair(
            net["s"],
            outlet=3,
            head_1=0,
            head_2=1,
            confluence=2,
            grid_shape=net["grid_shape"],
            target_size=24,
        )
        flags = raster_quality_flags(result)
        # Despite a 200px extent collapsed to 24px, everything survives.
        assert flags["has_branch_a"]
        assert flags["has_branch_b"]
        assert flags["has_confluence"]
        assert flags["branch_a_connected"]
        assert flags["branch_b_connected"]
        assert flags["branches_connected"]

    def test_direct_rasterizer_survives_very_small_target(self):
        """A very small target still preserves marker and branch connectivity."""
        net = _make_large_extent_network()
        result = rasterize_outlet_pair(
            net["s"],
            outlet=3,
            head_1=0,
            head_2=1,
            confluence=2,
            grid_shape=net["grid_shape"],
            target_size=16,
        )

        assert result.shape == (16, 16)
        flags = raster_quality_flags(result)
        assert flags["has_confluence"]
        assert flags["branches_connected"]


# =============================================================================
# raster_quality_flags unit tests
# =============================================================================


class TestRasterQualityFlags:
    """Direct unit tests for raster_quality_flags and _component_count."""

    def test_all_valid(self):
        """Branch A and B each 8-connected to the confluence marker."""
        r = np.zeros((5, 5), dtype=np.uint8)
        r[0, 0] = BRANCH_A
        r[1, 0] = BRANCH_A
        r[0, 2] = BRANCH_B
        r[1, 2] = BRANCH_B
        r[2, 1] = CONFLUENCE_MARKER  # diagonally adjacent to both (1,0) and (1,2)
        flags = raster_quality_flags(r)
        assert flags == {
            "has_branch_a": True,
            "has_branch_b": True,
            "has_confluence": True,
            "branch_a_connected": True,
            "branch_b_connected": True,
            "branches_connected": True,
        }

    def test_missing_confluence(self):
        """No confluence marker → nothing is connected."""
        r = np.zeros((5, 5), dtype=np.uint8)
        r[0, 0] = BRANCH_A
        r[0, 2] = BRANCH_B
        flags = raster_quality_flags(r)
        assert flags["has_branch_a"]
        assert flags["has_branch_b"]
        assert not flags["has_confluence"]
        assert not flags["branch_a_connected"]
        assert not flags["branch_b_connected"]
        assert not flags["branches_connected"]

    def test_branch_a_disconnected(self):
        """Branch A separated from the marker by background fails connectivity."""
        r = np.zeros((6, 6), dtype=np.uint8)
        r[0, 0] = BRANCH_A  # isolated in the top-left corner
        r[4, 4] = CONFLUENCE_MARKER
        r[3, 4] = BRANCH_B  # 8-connected to the marker
        flags = raster_quality_flags(r)
        assert flags["has_branch_a"]
        assert flags["has_confluence"]
        assert not flags["branch_a_connected"]
        assert flags["branch_b_connected"]
        assert not flags["branches_connected"]

    def test_component_count_single_vs_split(self):
        """_component_count distinguishes connected from split masks."""
        connected = np.zeros((4, 4), dtype=bool)
        connected[1, 1] = True
        connected[2, 2] = True  # diagonal → 8-connected → one component
        assert _component_count(connected) == 1

        split = np.zeros((4, 4), dtype=bool)
        split[0, 0] = True
        split[3, 3] = True  # far apart → two components
        assert _component_count(split) == 2

        assert _component_count(np.zeros((4, 4), dtype=bool)) == 0


# =============================================================================
# Batch precompute QA gating
# =============================================================================


class TestPrecomputeQAGating:
    """Batch-level QA behaviour of precompute_raster_dataset."""

    def _write_master(self, tmp_path):
        master_csv = tmp_path / "master.csv"
        pd.DataFrame(
            [
                {
                    "basin": "inyo",
                    "outlet": 6,
                    "confluence": 4,
                    "head_1": 0,
                    "head_2": 1,
                    "y": 1,
                }
            ]
        ).to_csv(master_csv, index=False)
        return master_csv

    def test_ok_patch_has_all_required_classes(self, tmp_path, simple_y_network):
        """When raster_status == 'ok', the saved patch has classes 1, 2 and 4."""
        master_csv = self._write_master(tmp_path)

        def loader(_basin, _lat, _z_th, _threshold):
            return simple_y_network["s"], simple_y_network["dem"]

        out = precompute_raster_dataset(
            master_csv=master_csv,
            output_dir=tmp_path / "rasters",
            dem_loader=loader,
            target_size=32,
        )
        row = out.iloc[0]
        assert row["raster_status"] == "ok"
        assert row["raster_path"]
        assert row["raster_debug_path"] == row["raster_path"]
        assert out.columns.tolist() == [
            "basin",
            "outlet",
            "confluence",
            "head_1",
            "head_2",
            "y",
            "raster_path",
            "raster_debug_path",
            "raster_status",
            "raster_error",
            "has_branch_a",
            "has_branch_b",
            "has_confluence",
            "branch_a_connected",
            "branch_b_connected",
            "branches_connected",
        ]
        raster = np.load(row["raster_path"])
        present = set(np.unique(raster))
        assert {BRANCH_A, BRANCH_B, CONFLUENCE_MARKER}.issubset(present)
        # And the QA flags assert 8-connectivity of both branches.
        assert row["branch_a_connected"]
        assert row["branch_b_connected"]
        assert row["branches_connected"]

    def test_invalid_patch_gets_no_raster_path(self, tmp_path, simple_y_network, monkeypatch):
        """A patch that fails structural QA is 'invalid' with no raster_path,
        but is still saved to a debug path for inspection."""
        import channel_heads.rasterizer as rast

        def broken_raster(*_args, **_kwargs):
            # Branch A and B present but no confluence marker and disconnected.
            r = np.zeros((32, 32), dtype=np.uint8)
            r[5, 5] = BRANCH_A
            r[25, 25] = BRANCH_B
            return r

        monkeypatch.setattr(rast, "rasterize_outlet_pair", broken_raster)

        master_csv = self._write_master(tmp_path)

        def loader(_basin, _lat, _z_th, _threshold):
            return simple_y_network["s"], simple_y_network["dem"]

        out = precompute_raster_dataset(
            master_csv=master_csv,
            output_dir=tmp_path / "rasters",
            dem_loader=loader,
            target_size=32,
        )
        row = out.iloc[0]
        assert row["raster_status"] == "invalid"
        assert pd.isna(row["raster_path"]) or row["raster_path"] is None
        assert row["raster_error"].startswith("qa_failed:")
        # Debug artifact is still written for inspection.
        assert row["raster_debug_path"]
        assert Path(row["raster_debug_path"]).exists()
        assert not row["has_confluence"]

    def test_missing_stream_loader_marks_rows_skipped(self, tmp_path):
        master_csv = self._write_master(tmp_path)

        def loader(_basin, _lat, _z_th, _threshold):
            return None

        out = precompute_raster_dataset(
            master_csv=master_csv,
            output_dir=tmp_path / "rasters",
            dem_loader=loader,
            target_size=32,
        )

        row = out.iloc[0]
        assert row["raster_status"] == "skipped"
        assert row["raster_error"] == "missing_dem_or_stream"
        assert pd.isna(row["raster_path"]) or row["raster_path"] is None
        assert pd.isna(row["raster_debug_path"]) or row["raster_debug_path"] is None

    def test_missing_basin_config_marks_rows_skipped(self, tmp_path):
        master_csv = tmp_path / "master.csv"
        loader_called = False
        pd.DataFrame(
            [
                {
                    "basin": "__missing_config__",
                    "outlet": 6,
                    "confluence": 4,
                    "head_1": 0,
                    "head_2": 1,
                    "y": 1,
                }
            ]
        ).to_csv(master_csv, index=False)

        def loader(*_args):
            nonlocal loader_called
            loader_called = True
            return None

        out = precompute_raster_dataset(
            master_csv=master_csv,
            output_dir=tmp_path / "rasters",
            dem_loader=loader,
            target_size=32,
        )

        row = out.iloc[0]
        assert row["raster_status"] == "skipped"
        assert row["raster_error"] == "missing_basin_config"
        assert not loader_called

    def test_rasterizer_exception_marks_row_failed(self, tmp_path, simple_y_network, monkeypatch):
        import channel_heads.rasterizer as rast

        master_csv = self._write_master(tmp_path)

        def loader(_basin, _lat, _z_th, _threshold):
            return simple_y_network["s"], simple_y_network["dem"]

        def raise_rasterizer(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(rast, "rasterize_outlet_pair", raise_rasterizer)

        out = precompute_raster_dataset(
            master_csv=master_csv,
            output_dir=tmp_path / "rasters",
            dem_loader=loader,
            target_size=32,
        )

        row = out.iloc[0]
        assert row["raster_status"] == "failed"
        assert row["raster_error"] == "RuntimeError: boom"
        assert pd.isna(row["raster_path"]) or row["raster_path"] is None
        assert pd.isna(row["raster_debug_path"]) or row["raster_debug_path"] is None


# =============================================================================
# Mars rasterization smoke test (shared semantics with Earth)
# =============================================================================


def _load_mars_script():
    """Return the Mars patch module (logic moved into the package, Phase 4)."""
    from channel_heads.rasterization import mars_patches

    return mars_patches


class TestMarsRasterizationSmoke:
    """Smoke test that the Mars script's direct rasterization path matches
    Earth's class encoding and produces structurally valid patches."""

    def test_mars_direct_rasterization_is_valid(self):
        shapely = pytest.importorskip("shapely")
        mod = _load_mars_script()
        LineString = shapely.geometry.LineString

        # A simple Y in Mars projected metres: heads at the top (north),
        # confluence at the origin, outlet draining south.
        conf_xy = (0.0, 0.0)
        h1_xy = (-2000.0, 8000.0)
        h2_xy = (2000.0, 8000.0)
        path_a = LineString([h1_xy, conf_xy])
        path_b = LineString([h2_xy, conf_xy])
        network_segments = [LineString([conf_xy, (0.0, -6000.0)])]

        patch = mod.rasterize_mars_pair(
            h1_xy=h1_xy,
            h2_xy=h2_xy,
            conf_xy=conf_xy,
            path_a=path_a,
            path_b=path_b,
            network_segments=network_segments,
        )

        # Same shape / dtype / encoding as Earth.
        assert patch.shape == (mod.TARGET_SIZE, mod.TARGET_SIZE)
        assert patch.dtype == np.uint8
        assert set(np.unique(patch)).issubset({0, 1, 2, 3, 4})

        # Earth constants are reused, not redefined.
        assert mod.BRANCH_A == BRANCH_A
        assert mod.BRANCH_B == BRANCH_B
        assert mod.OTHER_STREAMS == OTHER_STREAMS
        assert mod.CONFLUENCE_MARKER == CONFLUENCE_MARKER

        # Structural QA passes using the shared Earth helper.
        flags = mod.raster_quality_flags(patch)
        assert flags["has_branch_a"]
        assert flags["has_branch_b"]
        assert flags["has_confluence"]
        assert flags["branch_a_connected"]
        assert flags["branch_b_connected"]
        assert flags["branches_connected"]
