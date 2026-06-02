"""Unit tests for channel_heads.dd_calibration.

These cover the pure-math helpers (Dd_hull / Dd_true / basin area / stream
length), the first-order trimming behaviour, and the sweep cache/schema path.
The heavy DEM sweep itself is exercised end-to-end by the notebook, not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from channel_heads.dd_calibration import (
    DEFAULT_MARS_VALLEYS_GPKG,
    LOW_THRESHOLDS_KM2,
    PRACTICAL_THRESHOLDS_KM2,
    STATUS_OK,
    _basin_pixel_counts,
    _build_mars_adjacency,
    _horton_strahler_max,
    _strip_geometry_columns,
    basin_to_hull_area_ratio,
    compute_basin_area_km2,
    compute_convex_hull_area_km2,
    compute_drainage_density,
    compute_stream_length_km,
    compute_threshold_cells,
    length_retention_penalty,
    mars_network_strahler,
    mars_network_table,
    mask_boundary_xy,
    match_outlets_by_coords,
    run_practical_sweep,
    threshold_slug,
    trim_first_order,
    trim_to_min_order,
    validate_area_bin_grouping,
)
from channel_heads.stream_utils import line_pixels


class TestComputeDrainageDensity:
    def test_basic_ratio(self):
        assert compute_drainage_density(10.0, 5.0) == pytest.approx(2.0)

    @pytest.mark.parametrize("area", [0.0, -1.0, float("nan"), None])
    def test_non_positive_area_is_nan(self, area):
        assert np.isnan(compute_drainage_density(10.0, area))


class TestConvexHullArea:
    def test_unit_square_area(self):
        # A 10x10 px square at 100 m/px = 1000 m x 1000 m = 1 km^2.
        rows = np.array([0, 0, 10, 10], dtype=float)
        cols = np.array([0, 10, 10, 0], dtype=float)
        area, poly, status = compute_convex_hull_area_km2(rows, cols, 100.0)
        assert status == STATUS_OK
        assert area == pytest.approx(1.0)
        assert poly is not None and poly.shape[1] == 2

    def test_collinear_points_flagged(self):
        rows = np.array([0, 1, 2], dtype=float)
        cols = np.array([0, 1, 2], dtype=float)
        area, poly, status = compute_convex_hull_area_km2(rows, cols, 30.0)
        assert np.isnan(area)
        assert poly is None
        assert status != STATUS_OK


class TestBasinAreaAndLength:
    def test_basin_area_km2(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[:5, :4] = True  # 20 px
        assert compute_basin_area_km2(mask, 100.0) == pytest.approx(20 * 100.0**2 / 1e6)

    def test_stream_length_km(self):
        class _S:
            # one horizontal edge (3 px) + one vertical edge (4 px)
            source_indices = (np.array([0, 0]), np.array([0, 0]))
            target_indices = (np.array([0, 4]), np.array([3, 0]))

        # lengths: hypot(0,3)=3 and hypot(4,0)=4 -> 7 px * 100 m / 1000
        assert compute_stream_length_km(_S(), 100.0) == pytest.approx(0.7)

    def test_stream_length_empty(self):
        class _S:
            source_indices = (np.array([]), np.array([]))
            target_indices = (np.array([]), np.array([]))

        assert compute_stream_length_km(_S(), 100.0) == 0.0


class TestThresholdCells:
    def test_round_and_floor_at_one(self):
        assert compute_threshold_cells(1.0, 1000.0) == 1  # 1e6 / 1e6
        assert compute_threshold_cells(0.0001, 1000.0) == 1  # never below 1


class _FakeStream:
    """Minimal StreamObject stand-in for trim_first_order."""

    def __init__(self, order):
        self._order = np.asarray(order)
        self.subgraph_arg = None

    def streamorder(self, method="strahler"):
        assert method == "strahler"
        return self._order

    def subgraph(self, nal):
        self.subgraph_arg = np.asarray(nal)
        return ("subgraph", self.subgraph_arg)


class TestTrimFirstOrder:
    def test_returns_none_when_all_first_order(self):
        s = _FakeStream([1, 1, 1])
        assert trim_first_order(s) is None

    def test_keeps_order_ge_2(self):
        s = _FakeStream([1, 2, 1, 3])
        result = trim_first_order(s)
        assert result[0] == "subgraph"
        np.testing.assert_array_equal(s.subgraph_arg, np.array([False, True, False, True]))


class TestTrimToMinOrder:
    def test_min_order_le_1_returns_same_object(self):
        s = _FakeStream([1, 2, 3])
        assert trim_to_min_order(s, 1) is s

    def test_ge3_drops_first_and_second_order(self):
        s = _FakeStream([1, 2, 3, 2, 4])
        result = trim_to_min_order(s, 3)
        np.testing.assert_array_equal(s.subgraph_arg, np.array([False, False, True, False, True]))
        assert result[0] == "subgraph"

    def test_returns_none_when_nothing_survives(self):
        assert trim_to_min_order(_FakeStream([1, 2, 1, 2]), 3) is None


class TestMaskBoundaryXY:
    def test_square_mask_closed_polygon(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:15, 5:15] = True
        poly = mask_boundary_xy(mask, 100.0)
        assert poly is not None and poly.shape[1] == 2
        # y uses the -row convention, so all boundary y <= 0.
        assert np.all(poly[:, 1] <= 1e-9)
        # extent spans roughly the 10-px (=1000 m) square
        assert np.ptp(poly[:, 0]) == pytest.approx(1000.0, abs=200.0)

    def test_empty_mask_returns_none(self):
        assert mask_boundary_xy(np.zeros((5, 5), dtype=bool), 30.0) is None


class TestRunPracticalSweepCache:
    def test_loads_existing_cache_without_compute(self, tmp_path):
        cache = tmp_path / "cache.csv"
        df_in = pd.DataFrame(
            {
                "dem_name": ["x"],
                "threshold_km2": [1.0],
                "dd_hull_full": [3.0],
                "dd_true_full": [0.5],
                "status_full": [STATUS_OK],
            }
        )
        df_in.to_csv(cache, index=False)
        # basins=[] would normally yield an empty frame; the cache must win.
        out = run_practical_sweep(basins=[], cache_csv=cache, verbose=False)
        pd.testing.assert_frame_equal(out, df_in)

    def test_practical_thresholds_constant_is_sane(self):
        assert PRACTICAL_THRESHOLDS_KM2[0] < PRACTICAL_THRESHOLDS_KM2[-1]
        assert max(PRACTICAL_THRESHOLDS_KM2) <= 10
        assert PRACTICAL_THRESHOLDS_KM2 == sorted(PRACTICAL_THRESHOLDS_KM2)


class TestLowThresholdsAndSlug:
    def test_low_thresholds_monotonic_and_bounded(self):
        assert LOW_THRESHOLDS_KM2 == sorted(LOW_THRESHOLDS_KM2)
        assert max(LOW_THRESHOLDS_KM2) <= 10
        # ~85 m/px DEMs: anything below ~0.007 km^2 is the 1-pixel floor, so
        # the sweep should not include nominal values below that.
        assert min(LOW_THRESHOLDS_KM2) >= 0.01

    @pytest.mark.parametrize(
        "km2,expected",
        [(0.005, "0p005km2"), (0.01, "0p01km2"), (1.0, "1km2"), (10.0, "10km2"), (2.5, "2p5km2")],
    )
    def test_threshold_slug_is_filename_safe(self, km2, expected):
        slug = threshold_slug(km2)
        assert slug == expected
        assert "." not in slug and "/" not in slug


class TestHortonStrahlerMax:
    def test_empty(self):
        max_s, n_leaves, n_comp, is_tree = _horton_strahler_max({})
        assert (max_s, n_leaves, n_comp, is_tree) == (0, 0, 0, True)

    def test_y_tree_has_max_2(self):
        # Three leaves A, B, C joined at junction J.
        adj = {"A": {"J"}, "B": {"J"}, "C": {"J"}, "J": {"A", "B", "C"}}
        max_s, n_leaves, n_comp, is_tree = _horton_strahler_max(adj)
        assert max_s == 2
        assert n_leaves == 3
        assert n_comp == 1
        assert is_tree

    def test_double_y_has_max_3(self):
        # Two Y-junctions joined trunk-to-trunk:
        #   A,B -> J1; C,D -> J2; J1,J2 -> J3; J3 -> O.
        adj = {
            "A": {"J1"},
            "B": {"J1"},
            "C": {"J2"},
            "D": {"J2"},
            "J1": {"A", "B", "J3"},
            "J2": {"C", "D", "J3"},
            "J3": {"J1", "J2", "O"},
            "O": {"J3"},
        }
        max_s, n_leaves, n_comp, is_tree = _horton_strahler_max(adj)
        assert max_s == 3
        assert n_leaves == 5  # A, B, C, D, O
        assert is_tree

    def test_root_choice_does_not_change_max(self):
        # Same graph, two DFS starts (the function picks an arbitrary leaf
        # from `adj.keys()`; we just confirm rerunning with a re-ordered dict
        # gives the same max).
        adj1 = {"A": {"J"}, "J": {"A", "B"}, "B": {"J"}}
        adj2 = {"B": {"J"}, "J": {"A", "B"}, "A": {"J"}}
        assert _horton_strahler_max(adj1)[0] == _horton_strahler_max(adj2)[0]

    def test_disconnected_components(self):
        adj = {"A": {"B"}, "B": {"A"}, "C": {"D"}, "D": {"C"}}
        max_s, _, n_comp, _ = _horton_strahler_max(adj)
        assert max_s == 1
        assert n_comp == 2

    def test_cycle_flags_not_tree(self):
        # Triangle A-B-C-A: one back-edge is detected.
        adj = {"A": {"B", "C"}, "B": {"A", "C"}, "C": {"A", "B"}}
        max_s, _, _, is_tree = _horton_strahler_max(adj)
        assert not is_tree
        assert max_s >= 1


class TestBuildMarsAdjacency:
    def test_two_segments_share_endpoint(self):
        lines = [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.0, 0.0], [2.0, 0.0]])]
        adj = _build_mars_adjacency(lines)
        # Three nodes, two edges: (0,0)-(1,0) and (1,0)-(2,0).
        assert len(adj) == 3
        assert sum(len(v) for v in adj.values()) // 2 == 2

    def test_zero_length_segments_dropped(self):
        lines = [np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])]
        adj = _build_mars_adjacency(lines)
        assert len(adj) == 2

    def test_snap_collapses_near_coincident_points(self):
        # Two LineStrings whose endpoints differ at 4th decimal (~0.1 mm).
        lines = [np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[1.00001, 0.0], [2.0, 0.0]])]
        adj = _build_mars_adjacency(lines, snap_decimals=3)
        assert len(adj) == 3  # snapped to a shared (1.0, 0.0)


class TestBasinToHullAreaRatio:
    def test_scalar_matches_definition(self):
        # Dd_hull / Dd_true == basin_area / hull_area (both share L).
        assert basin_to_hull_area_ratio(2.0, 1.0) == pytest.approx(2.0)
        assert basin_to_hull_area_ratio(0.5, 1.0) == pytest.approx(0.5)

    def test_zero_or_nan_dd_true_returns_nan(self):
        assert np.isnan(basin_to_hull_area_ratio(1.0, 0.0))
        assert np.isnan(basin_to_hull_area_ratio(1.0, float("nan")))

    def test_array_broadcast(self):
        out = basin_to_hull_area_ratio(np.array([2.0, 1.0]), np.array([1.0, 1.0]))
        np.testing.assert_allclose(out, np.array([2.0, 1.0]))


class TestLengthRetentionPenalty:
    def test_above_threshold_zero(self):
        assert length_retention_penalty(60.0, min_retained_pct=30.0) == 0.0

    def test_at_threshold_zero(self):
        assert length_retention_penalty(30.0, min_retained_pct=30.0) == 0.0

    def test_below_threshold_quadratic(self):
        # deficit = (30 - 15)/30 = 0.5 -> penalty = 0.25
        assert length_retention_penalty(15.0, min_retained_pct=30.0) == pytest.approx(0.25)

    def test_zero_length_max_penalty(self):
        assert length_retention_penalty(0.0, min_retained_pct=30.0) == pytest.approx(1.0)

    def test_weight_scales(self):
        assert length_retention_penalty(0.0, min_retained_pct=30.0, weight=4.0) == pytest.approx(
            4.0
        )


class TestValidateAreaBinGrouping:
    def test_clean_dataset_no_warnings(self):
        # Two distinct outlets at distinct areas; no merges.
        df = pd.DataFrame(
            {
                "dem_name": ["A", "A"],
                "outlet_node": [1, 2],
                "basin_area_km2": [1.0, 5.0],
                "threshold_km2": [0.1, 0.1],
                "status_full": [STATUS_OK, STATUS_OK],
            }
        )
        res = validate_area_bin_grouping(df, area_bin_km2=0.1)
        assert res["merge_groups"].empty
        # Coord warning is expected since outlet_row/col not provided.

    def test_merge_detected(self):
        # Two outlets with identical area at the same threshold -- merged.
        df = pd.DataFrame(
            {
                "dem_name": ["A", "A"],
                "outlet_node": [1, 2],
                "basin_area_km2": [2.0, 2.0],
                "threshold_km2": [0.1, 0.1],
                "status_full": [STATUS_OK, STATUS_OK],
            }
        )
        res = validate_area_bin_grouping(df, area_bin_km2=0.1)
        assert len(res["merge_groups"]) == 1
        assert any("merge" in w for w in res["warnings"])


class TestMatchOutletsByCoords:
    def test_requires_coord_columns(self):
        df = pd.DataFrame({"dem_name": ["A"], "outlet_node": [1]})
        with pytest.raises(ValueError, match="outlet_row"):
            match_outlets_by_coords(df)

    def test_clusters_nearby_outlets(self):
        # Same DEM, three rows: two at (10,10) and (11,10) (within tol),
        # one at (50,50) (far).
        df = pd.DataFrame(
            {
                "dem_name": ["A", "A", "A"],
                "outlet_row": [10, 11, 50],
                "outlet_col": [10, 10, 50],
                "outlet_node": [1, 2, 3],
            }
        )
        out = match_outlets_by_coords(df, coord_tol_px=2)
        clusters = out["outlet_cluster"].tolist()
        assert clusters[0] == clusters[1]  # rows 0 and 1 share cluster
        assert clusters[0] != clusters[2]  # row 2 is its own cluster


@pytest.mark.skipif(
    not DEFAULT_MARS_VALLEYS_GPKG.exists(),
    reason="Mars valley GeoPackage not available",
)
class TestMarsNetworkStrahler:
    def test_one_row_per_network(self):
        tbl = mars_network_strahler()
        from channel_heads.dd_calibration import MARS_DD_STATS

        assert len(tbl) == 391
        # Sanity: Strahler must be >= 1 for non-empty networks (in MARS_DD_STATS).
        ok = tbl[tbl["status"] == STATUS_OK]
        assert (ok["max_strahler"] >= 1).all()
        assert int(MARS_DD_STATS["n"]) == len(ok)


class TestLinePixels:
    def test_endpoints_included(self):
        rr, cc = line_pixels(2, 3, 7, 9)
        assert (rr[0], cc[0]) == (2, 3)
        assert (rr[-1], cc[-1]) == (7, 9)

    @pytest.mark.parametrize(
        "r0,c0,r1,c1",
        [(0, 0, 0, 5), (0, 0, 5, 0), (0, 0, 5, 5), (3, 1, 0, 7), (7, 9, 2, 3), (4, 4, 4, 4)],
    )
    def test_length_is_chebyshev_plus_one(self, r0, c0, r1, c1):
        rr, cc = line_pixels(r0, c0, r1, c1)
        expected = max(abs(r1 - r0), abs(c1 - c0)) + 1
        assert rr.size == expected
        assert cc.size == expected

    def test_eight_connected(self):
        # Consecutive pixels never jump more than one cell in either axis.
        rr, cc = line_pixels(1, 2, 11, 5)
        assert np.all(np.abs(np.diff(rr)) <= 1)
        assert np.all(np.abs(np.diff(cc)) <= 1)

    def test_integer_dtype(self):
        rr, cc = line_pixels(0, 0, 4, 9)
        assert np.issubdtype(rr.dtype, np.integer)
        assert np.issubdtype(cc.dtype, np.integer)

    @pytest.mark.parametrize(
        "r0,c0,r1,c1",
        [
            (0, 0, 0, 8),
            (0, 0, 8, 0),
            (0, 0, 8, 8),
            (0, 0, 8, 3),
            (0, 0, 3, 8),
            (8, 3, 0, 0),
            (3, 8, 0, 0),
            (5, 5, 5, 5),
            (-4, 2, 6, -3),
        ],
    )
    def test_matches_skimage_when_available(self, r0, c0, r1, c1):
        skdraw = pytest.importorskip("skimage.draw")
        rr, cc = line_pixels(r0, c0, r1, c1)
        srr, scc = skdraw.line(r0, c0, r1, c1)
        np.testing.assert_array_equal(rr, srr)
        np.testing.assert_array_equal(cc, scc)


class TestBasinPixelCounts:
    def test_counts_match_manual(self):
        labels = np.array(
            [
                [0, 1, 1],
                [2, 2, 1],
                [2, 0, 3],
            ]
        )
        valid = np.ones_like(labels, dtype=bool)
        sizes = _basin_pixel_counts(labels, valid, n_outlets=3)
        np.testing.assert_array_equal(sizes, np.array([3, 3, 1]))

    def test_respects_valid_mask(self):
        labels = np.array([[1, 1], [2, 2]])
        valid = np.array([[True, False], [True, True]])
        sizes = _basin_pixel_counts(labels, valid, n_outlets=2)
        np.testing.assert_array_equal(sizes, np.array([1, 2]))

    def test_matches_per_label_loop(self):
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 5, size=(20, 20))  # labels 0..4 -> 4 outlets
        valid = rng.random((20, 20)) > 0.3
        n = 4
        sizes = _basin_pixel_counts(labels, valid, n_outlets=n)
        loop = np.array([int(((labels == i + 1) & valid).sum()) for i in range(n)])
        np.testing.assert_array_equal(sizes, loop)

    def test_missing_label_is_zero(self):
        labels = np.array([[0, 1], [1, 0]])
        valid = np.ones_like(labels, dtype=bool)
        sizes = _basin_pixel_counts(labels, valid, n_outlets=3)
        np.testing.assert_array_equal(sizes, np.array([2, 0, 0]))


class TestStripGeometryColumns:
    def test_strips_all_array_geometry_keys(self):
        record = {
            # tabular fields that MUST survive
            "basin_id": "x_o5",
            "dem_name": "x",
            "outlet_node": 5,
            "outlet_row": 10,
            "outlet_col": 12,
            "basin_area_km2": 3.2,
            "dd_hull_full": 1.1,
            "status_full": "ok",
            # array geometry fields that MUST be dropped
            "stream_xy_full": np.zeros((2, 2)),
            "stream_seg_full": np.zeros((3, 2, 2)),
            "stream_seg_trim": np.zeros((1, 2, 2)),
            "hull_xy_full": np.zeros((4, 2)),
            "removed_seg_trim": np.zeros((1, 2, 2)),
            "removed_seg_ge3": None,
            "outlet_xy": np.array([1.0, -2.0]),
            "basin_boundary_xy": np.zeros((5, 2)),
        }
        out = _strip_geometry_columns(record)
        # No array/None geometry leaked through.
        assert set(out) == {
            "basin_id",
            "dem_name",
            "outlet_node",
            "outlet_row",
            "outlet_col",
            "basin_area_km2",
            "dd_hull_full",
            "status_full",
        }
        # The look-alike tabular columns are preserved verbatim.
        assert out["outlet_node"] == 5
        assert out["outlet_col"] == 12
        assert out["basin_area_km2"] == 3.2

    def test_result_is_csv_serializable(self):
        record = {"a": 1, "stream_seg_full": np.zeros((2, 2, 2)), "outlet_xy": np.array([0.0])}
        out = _strip_geometry_columns(record)
        # A frame built from the stripped record writes to CSV without error.
        pd.DataFrame([out]).to_csv()


class TestValidateAreaBinGroupingRobustness:
    def test_non_finite_basin_area_does_not_crash(self):
        df = pd.DataFrame(
            {
                "dem_name": ["A", "A", "A"],
                "outlet_node": [1, 2, 3],
                "basin_area_km2": [1.0, float("nan"), float("inf")],
                "threshold_km2": [0.1, 0.1, 0.1],
                "status_full": [STATUS_OK, STATUS_OK, STATUS_OK],
            }
        )
        res = validate_area_bin_grouping(df, area_bin_km2=0.1)
        # No exception; the two non-finite rows are excluded with a warning.
        assert any("non-finite" in w for w in res["warnings"])
        assert res["merge_groups"].empty

    def test_finite_rows_still_evaluated_after_dropping(self):
        # One NaN row plus two finite rows that should merge (same area/thr).
        df = pd.DataFrame(
            {
                "dem_name": ["A", "A", "A"],
                "outlet_node": [1, 2, 3],
                "basin_area_km2": [2.0, 2.0, float("nan")],
                "threshold_km2": [0.1, 0.1, 0.1],
                "status_full": [STATUS_OK, STATUS_OK, STATUS_OK],
            }
        )
        res = validate_area_bin_grouping(df, area_bin_km2=0.1)
        assert len(res["merge_groups"]) == 1


@pytest.mark.skipif(
    not DEFAULT_MARS_VALLEYS_GPKG.exists(),
    reason="Mars valley GeoPackage not available",
)
class TestMarsNetworkTable:
    def test_reproduces_mars_dd_stats(self):
        from channel_heads.dd_calibration import MARS_DD_STATS

        tbl = mars_network_table()
        ok = tbl.loc[tbl["status"] == STATUS_OK, "dd_hull_km_km2"]
        assert len(ok) == int(MARS_DD_STATS["n"])
        assert ok.median() == pytest.approx(MARS_DD_STATS["median"], abs=1e-6)
        assert ok.quantile(0.25) == pytest.approx(MARS_DD_STATS["q1"], abs=1e-6)
        assert ok.quantile(0.75) == pytest.approx(MARS_DD_STATS["q3"], abs=1e-6)
