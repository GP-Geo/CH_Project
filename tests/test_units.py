"""Tests for channel_heads.units — the consolidated unit-conversion module.

These lock in the numeric behavior (unchanged by the Phase 2 consolidation) and
assert that ``geometric_analysis`` and ``dd_calibration`` re-export the *same*
objects, proving ``units`` is the single source of truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from channel_heads import units


class TestCoordinateConversions:
    def test_meters_per_degree_known_value(self):
        # At Inyo latitude ~36.7 -> ~99328 m/deg (geometric mean of lat/lon)
        assert units.compute_meters_per_degree(36.7) == pytest.approx(99328, abs=5)

    def test_meters_per_degree_equator_and_45(self):
        assert 110000 < units.compute_meters_per_degree(0) < 112000
        assert 90000 < units.compute_meters_per_degree(45) < 100000

    def test_meters_per_degree_sign_invariant(self):
        assert units.compute_meters_per_degree(36.7) == units.compute_meters_per_degree(-36.7)

    def test_pixel_size_meters_known_value(self):
        # 1 arc-second SRTM at 36.7 -> ~27.6 m
        assert units.compute_pixel_size_meters(36.7, 1 / 3600) == pytest.approx(27.6, abs=0.1)


class TestPixelSizeFromDem:
    def test_projected_dem_returns_cellsize(self):
        dem = type("DEM", (), {"cellsize": 30.0})()
        assert units.compute_pixel_size_m_from_dem(dem) == 30.0

    def test_geographic_dem_needs_lat(self):
        dem = type("DEM", (), {"cellsize": 1 / 3600})()
        with pytest.raises(ValueError):
            units.compute_pixel_size_m_from_dem(dem)

    def test_geographic_dem_with_lat(self):
        dem = type("DEM", (), {"cellsize": 1 / 3600})()
        got = units.compute_pixel_size_m_from_dem(dem, lat_deg=36.7)
        assert got == pytest.approx(units.compute_pixel_size_meters(36.7, 1 / 3600))


class TestAreaLengthDensity:
    def test_threshold_cells(self):
        assert units.compute_threshold_cells(1.0, 1000.0) == 1  # 1e6 / 1e6
        assert units.compute_threshold_cells(0.0001, 1000.0) == 1  # never below 1
        assert units.compute_threshold_cells(4.0, 1000.0) == 4

    def test_basin_area_km2(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[:4, :5] = True  # 20 px
        assert units.compute_basin_area_km2(mask, 100.0) == pytest.approx(20 * 100.0**2 / 1e6)

    def test_drainage_density(self):
        assert units.compute_drainage_density(10.0, 5.0) == pytest.approx(2.0)
        assert np.isnan(units.compute_drainage_density(10.0, 0.0))
        assert np.isnan(units.compute_drainage_density(10.0, -1.0))

    def test_stream_length_km(self):
        class _S:
            # two horizontal edges of length 3 and 4 px -> 7 px
            source_indices = (np.array([0, 0]), np.array([0, 3]))
            target_indices = (np.array([0, 0]), np.array([3, 7]))

        assert units.compute_stream_length_km(_S(), 100.0) == pytest.approx(0.7)

    def test_stream_length_empty(self):
        class _S:
            source_indices = (np.array([]), np.array([]))
            target_indices = (np.array([]), np.array([]))

        assert units.compute_stream_length_km(_S(), 100.0) == 0.0


class TestSingleSourceOfTruth:
    """Old call sites must re-export the identical objects from units."""

    def test_geometric_analysis_reexports(self):
        from channel_heads import geometric_analysis as ga

        assert ga.compute_meters_per_degree is units.compute_meters_per_degree
        assert ga.compute_pixel_size_meters is units.compute_pixel_size_meters

    def test_dd_calibration_reexports(self):
        from channel_heads import dd_calibration as dd

        assert dd.compute_pixel_size_m_from_dem is units.compute_pixel_size_m_from_dem
        assert dd.compute_threshold_cells is units.compute_threshold_cells
        assert dd.compute_stream_length_km is units.compute_stream_length_km
        assert dd.compute_basin_area_km2 is units.compute_basin_area_km2
        assert dd.compute_drainage_density is units.compute_drainage_density

    def test_package_level_exports_unchanged(self):
        import channel_heads as ch

        assert ch.compute_meters_per_degree is units.compute_meters_per_degree
        assert ch.compute_pixel_size_meters is units.compute_pixel_size_meters
