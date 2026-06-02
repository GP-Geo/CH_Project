"""Tests for channel_heads.features.geometry — shared feature primitives.

The behavioral correctness of these four functions is already exercised in depth
by ``tests/test_geometric_analysis.py`` (via the ``_``-prefixed re-exports the
Earth analyzer still uses). These tests add direct coverage of the public names
plus identity checks proving ``features.geometry`` is the single source of truth
shared by the Earth module (and, by import, the Mars feature script).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from channel_heads.features import geometry as geo


class TestPrimitives:
    def test_angle_between_vectors(self):
        assert geo.angle_between_vectors((1.0, 0.0), (1.0, 0.0)) == pytest.approx(0.0)
        assert geo.angle_between_vectors((1.0, 0.0), (0.0, 1.0)) == pytest.approx(90.0)
        assert geo.angle_between_vectors((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(180.0)
        assert math.isnan(geo.angle_between_vectors((0.0, 0.0), (1.0, 0.0)))

    def test_compute_azimuth(self):
        assert geo.compute_azimuth(0.0, 1.0) == pytest.approx(0.0)  # north
        assert geo.compute_azimuth(1.0, 0.0) == pytest.approx(90.0)  # east
        assert geo.compute_azimuth(0.0, -1.0) == pytest.approx(180.0)  # south
        assert geo.compute_azimuth(-1.0, 0.0) == pytest.approx(270.0)  # west
        assert math.isnan(geo.compute_azimuth(0.0, 0.0))

    def test_azimuth_difference(self):
        assert geo.azimuth_difference(45.0, 45.0) == pytest.approx(0.0)
        assert geo.azimuth_difference(0.0, 180.0) == pytest.approx(180.0)
        assert geo.azimuth_difference(350.0, 10.0) == pytest.approx(20.0)  # wrap
        assert math.isnan(geo.azimuth_difference(float("nan"), 45.0))

    def test_proximity_profile(self):
        # parallel lines, constant separation -> norm == 1.0
        c1 = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
        c2 = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
        mean_m, max_m, norm = geo.compute_proximity_profile(c1, c2)
        assert mean_m == pytest.approx(1.0)
        assert max_m == pytest.approx(1.0)
        assert norm == pytest.approx(1.0)

    def test_proximity_profile_convergent(self):
        c1 = np.array([[0.0, 0.0], [1.0, 0.0]])
        c2 = np.array([[2.0, 0.0], [1.0, 0.0]])  # converge to same point
        _, _, norm = geo.compute_proximity_profile(c1, c2)
        assert 0.0 <= norm < 1.0


class TestPathHelpers:
    """Polyline direction/sampling helpers shared by the Mars feature script
    and notebooks/mars/."""

    def test_direction_straight_east(self):
        from channel_heads.features import line_direction_first_n_meters

        coords = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0]])
        vec, qc = line_direction_first_n_meters(coords, max_distance_m=500.0)
        assert vec is not None
        assert vec[0] == pytest.approx(1.0)
        assert vec[1] == pytest.approx(0.0)

    def test_direction_single_edge_flagged(self):
        from channel_heads.features import line_direction_first_n_meters

        vec, qc = line_direction_first_n_meters(np.array([[0.0, 0.0]]), 500.0)
        assert vec is None
        assert "single_edge" in qc

    def test_direction_short_path_qc(self):
        from channel_heads.features import line_direction_first_n_meters

        # only 2 edges (< MIN_EDGES_FOR_DIRECTION=3) -> "short_path" flag
        coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        vec, qc = line_direction_first_n_meters(coords, max_distance_m=500.0)
        assert vec is not None
        assert "short_path" in qc

    def test_sample_path_coords_count_and_endpoints(self):
        from channel_heads.features import sample_path_coords_along_line

        coords = np.array([[0.0, 0.0], [10.0, 0.0]])
        out = sample_path_coords_along_line(coords, n_samples=10)
        assert out.shape == (10, 2)
        # samples at fractions 0,0.1,...,0.9 (1.0 excluded)
        assert out[0, 0] == pytest.approx(0.0)
        assert out[-1, 0] == pytest.approx(9.0)

    def test_sample_path_degenerate_returns_none(self):
        from channel_heads.features import sample_path_coords_along_line

        assert sample_path_coords_along_line(np.array([[1.0, 1.0]]), 10) is None


class TestSingleSourceOfTruth:
    def test_geometric_analysis_reexports_same_objects(self):
        from channel_heads import geometric_analysis as ga

        assert ga._angle_between_vectors is geo.angle_between_vectors
        assert ga._compute_azimuth is geo.compute_azimuth
        assert ga._azimuth_difference is geo.azimuth_difference
        assert ga._compute_proximity_profile is geo.compute_proximity_profile

    def test_package_features_namespace(self):
        from channel_heads import features

        assert features.angle_between_vectors is geo.angle_between_vectors
        assert features.compute_proximity_profile is geo.compute_proximity_profile
