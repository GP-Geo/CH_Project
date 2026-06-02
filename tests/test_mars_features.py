"""Tests for channel_heads.features.mars_features.

Covers the pure helpers and an end-to-end feature computation on a hand-built
synthetic Y-network (a cheap correctness check of the 5 features, column order,
and the stream-crossing filter).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from channel_heads.features import mars_features as mf

gpd = pytest.importorskip("geopandas")
from shapely.geometry import LineString, Point  # noqa: E402


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_compute_strahler_orders_y_network():
    # 0,1 heads -> 2 confluence -> 3 outlet
    parents = [[], [], [0, 1], [2]]
    children = [[2], [2], [3], []]
    order = mf.compute_strahler_orders(4, parents, children)
    assert order.tolist() == [1, 1, 2, 2]


def test_build_mars_directed_adjacency_skips_selfloops_and_oob():
    seg = pd.DataFrame(
        {
            "upstream_node_id": [0, 1, 2, 2, 9],
            "downstream_node_id": [2, 2, 3, 2, 0],  # (2,2) self-loop, (9,0) oob
        }
    )
    parents, children = mf.build_mars_directed_adjacency(seg, n_nodes=4)
    assert parents[2] == [0, 1]
    assert children[2] == [3]
    assert children[0] == [2]


@pytest.mark.parametrize(
    "s,expected",
    [("0,2,5", [0, 2, 5]), ("", []), (None, []), ("3", [3]), ("1,,2", [1, 2])],
)
def test_parse_id_list(s, expected):
    assert mf.parse_id_list(s) == expected


def test_build_filtering_audit_arithmetic():
    audit = mf.build_filtering_audit(100, 90, 30, 70)
    row = audit.set_index("filter_name").loc["stream_crossing_filter"]
    assert row["n_removed"] == 30 and row["n_after"] == 70
    composite = audit.set_index("filter_name").loc["model_ready_total"]
    assert composite["n_after"] == 70


def test_crosses_other_channel_true_and_false():
    from shapely.strtree import STRtree

    # A segment cutting straight across the head-to-head line.
    blocker = LineString([(50, -100), (50, 100)])
    tree = STRtree([blocker])
    assert mf.crosses_other_channel((0, 0), (100, 0), tree, [blocker]) is True
    # No blocker between the heads.
    far = LineString([(0, 500), (100, 500)])
    tree2 = STRtree([far])
    assert mf.crosses_other_channel((0, 0), (100, 0), tree2, [far]) is False


# --------------------------------------------------------------------------- #
# End-to-end on a synthetic Y-network
# --------------------------------------------------------------------------- #
def _synthetic_inputs():
    crs = "EPSG:3857"
    nodes = gpd.GeoDataFrame(
        {
            "network_id": [0, 0, 0, 0],
            "node_id": [0, 1, 2, 3],
            "node_type": ["channel_head", "channel_head", "confluence", "outlet"],
        },
        geometry=[Point(0, 100), Point(100, 100), Point(50, 0), Point(50, -100)],
        crs=crs,
    )
    seg0 = LineString([(0, 100), (50, 0)])
    seg1 = LineString([(100, 100), (50, 0)])
    seg2 = LineString([(50, 0), (50, -100)])
    segments = gpd.GeoDataFrame(
        {
            "network_id": [0, 0, 0],
            "segment_id": [0, 1, 2],
            "upstream_node_id": [0, 1, 2],
            "downstream_node_id": [2, 2, 3],
        },
        geometry=[seg0, seg1, seg2],
        crs=crs,
    )
    networks = gpd.GeoDataFrame(
        {"network_id": [0], "outlet_node_id": [3]},
        geometry=[seg2],
        crs=crs,
    )
    L = seg0.length
    pairs = gpd.GeoDataFrame(
        {
            "network_id": [0],
            "pair_id": ["p"],
            "head_1_node_id": [0],
            "head_2_node_id": [1],
            "confluence_node_id": [2],
            "L_1": [L],
            "L_2": [L],
            "headhead_dist_m": [100.0],
            "n_segments_a": [1],
            "n_segments_b": [1],
            "path_a_node_ids": ["0,2"],
            "path_b_node_ids": ["1,2"],
        },
        geometry=[seg0],
        crs=crs,
    )
    pair_paths = gpd.GeoDataFrame(
        {"network_id": [0, 0], "pair_id": ["p", "p"], "branch": ["A", "B"]},
        geometry=[seg0, seg1],
        crs=crs,
    )
    return networks, nodes, segments, pairs, pair_paths


def test_build_feature_table_synthetic_y():
    networks, nodes, segments, pairs, pair_paths = _synthetic_inputs()
    df_all, df_ready, audit = mf.build_feature_table(
        networks, nodes, segments, pairs, pair_paths
    )

    assert list(df_all.columns) == mf.FINAL_COLUMNS
    assert len(df_all) == 1
    row = df_all.iloc[0]

    # apex angle between (-50,100) and (50,100): cos = 0.6 -> ~53.13 deg
    assert row["apex_angle_deg"] == pytest.approx(53.13, abs=0.1)
    # headhead_dist_norm = 100 / (L_1 + L_2)
    assert row["headhead_dist_norm"] == pytest.approx(100.0 / (2 * segments.geometry[0].length), rel=1e-6)
    # both branch parents are order-1 heads
    assert row["strahler_order_diff"] == 0.0
    assert np.isfinite(row["orientation_diff_deg"])
    assert np.isfinite(row["proximity_profile_norm"])
    # heads connect cleanly: no stream crossing -> pair is model-ready
    assert bool(row["stream_crossing_drop"]) is False
    assert len(df_ready) == 1

    composite = audit.set_index("filter_name").loc["model_ready_total"]
    assert composite["n_after"] == 1
