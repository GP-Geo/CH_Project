"""Tests for the pure (IO-free) helpers in channel_heads.mars.topology."""

from __future__ import annotations

import numpy as np
import pytest

from channel_heads.mars import topology


def test_cluster_endpoints_groups_within_tolerance():
    coords = np.array([[0.0, 0.0], [0.05, 0.0], [10.0, 10.0], [10.0, 10.02]])
    clusters = topology.cluster_endpoints(coords, snap_tol=0.2)
    # First two collapse to one node, last two to another.
    assert clusters[0] == clusters[1]
    assert clusters[2] == clusters[3]
    assert clusters[0] != clusters[2]


def test_cluster_endpoints_edge_cases():
    assert topology.cluster_endpoints(np.empty((0, 2)), 1.0).tolist() == []
    assert topology.cluster_endpoints(np.array([[1.0, 2.0]]), 1.0).tolist() == [0]


@pytest.mark.parametrize(
    "node,outlet,degree,expected",
    [
        (3, 3, 1, "outlet"),
        (1, 3, 1, "channel_head"),
        (2, 3, 4, "confluence"),
        (5, 3, 2, "connector"),
    ],
)
def test_classify_node_type(node, outlet, degree, expected):
    assert topology.classify_node_type(node, outlet, degree) == expected


def test_pick_outlet_endpoint_prefers_nearest_cluster_point():
    eps = np.array([[0.0, 0.0], [1.0, 0.0]])
    x, y, dist = topology.pick_outlet_endpoint(eps, centroid_xy=(0.1, 0.0), network_geom=None)
    assert (x, y) == (0.0, 0.0)
    assert dist == pytest.approx(0.1)
