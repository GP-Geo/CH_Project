"""Tests for channel_heads.pruning."""

from __future__ import annotations

import numpy as np
import pytest

from channel_heads.pruning import (
    apply_strategy,
    build_stream_graph,
    prune_by_order_gap,
)


class _FakeStream:
    """Minimal StreamObject stand-in for the pruning helpers.

    Builds source/target indices from an edge list, exposes Strahler orders
    via ``streamorder``, and records ``subgraph`` calls so tests can inspect
    which nodes were kept. ``subgraph`` returns a new ``_FakeStream`` of the
    surviving nodes so the helpers can be exercised end-to-end.

    Parameters
    ----------
    node_positions : list of (row, col) tuples — one per node, indexed by node id.
    edges          : list of (upstream_id, downstream_id) pairs.
    orders         : Strahler order per node (any iterable convertible to ndarray).
    """

    def __init__(
        self,
        node_positions: list[tuple[int, int]],
        edges: list[tuple[int, int]],
        orders: list[int],
    ):
        if len(node_positions) != len(orders):
            raise ValueError("node_positions and orders must have the same length")
        self._positions = node_positions
        self._edges = edges
        self._orders = np.asarray(orders, dtype=int)
        self.subgraph_call: np.ndarray | None = None

        rows = np.array([p[0] for p in node_positions], dtype=np.intp)
        cols = np.array([p[1] for p in node_positions], dtype=np.intp)
        self.node_indices = (rows, cols)

        if edges:
            sr = np.array([node_positions[u][0] for u, _ in edges], dtype=np.intp)
            sc = np.array([node_positions[u][1] for u, _ in edges], dtype=np.intp)
            tr = np.array([node_positions[v][0] for _, v in edges], dtype=np.intp)
            tc = np.array([node_positions[v][1] for _, v in edges], dtype=np.intp)
        else:
            sr = sc = tr = tc = np.array([], dtype=np.intp)
        self.source_indices = (sr, sc)
        self.target_indices = (tr, tc)

    def streamorder(self, method: str = "strahler") -> np.ndarray:
        assert method == "strahler"
        return self._orders

    def subgraph(self, keep: np.ndarray) -> "_FakeStream":
        keep = np.asarray(keep, dtype=bool)
        self.subgraph_call = keep
        old_to_new: dict[int, int] = {}
        new_positions: list[tuple[int, int]] = []
        for old_id, kept in enumerate(keep):
            if kept:
                old_to_new[old_id] = len(new_positions)
                new_positions.append(self._positions[old_id])
        new_edges = [
            (old_to_new[u], old_to_new[v])
            for u, v in self._edges
            if u in old_to_new and v in old_to_new
        ]
        new_orders = [int(self._orders[i]) for i, kept in enumerate(keep) if kept]
        return _FakeStream(new_positions, new_edges, new_orders)


# ---------------------------------------------------------------------------
# build_stream_graph
# ---------------------------------------------------------------------------


class TestBuildStreamGraph:
    def test_y_network_in_edges(self):
        # Node 0,1 are heads; node 2 is confluence; node 3 is outlet.
        s = _FakeStream(
            node_positions=[(0, 0), (0, 4), (2, 2), (4, 2)],
            edges=[(0, 2), (1, 2), (2, 3)],
            orders=[1, 1, 2, 2],
        )
        node_r, node_c, rc_to_id, in_edges = build_stream_graph(s)

        np.testing.assert_array_equal(node_r, [0, 0, 2, 4])
        np.testing.assert_array_equal(node_c, [0, 4, 2, 2])
        assert rc_to_id == {(0, 0): 0, (0, 4): 1, (2, 2): 2, (4, 2): 3}
        # Confluence (node 2) receives both heads; outlet (node 3) receives confluence.
        assert sorted(in_edges[2]) == [0, 1]
        assert in_edges[3] == [2]
        # Heads have no upstream.
        assert in_edges[0] == [] and in_edges[1] == []

    def test_empty_network(self):
        s = _FakeStream(node_positions=[], edges=[], orders=[])
        node_r, node_c, rc_to_id, in_edges = build_stream_graph(s)
        assert node_r.size == 0 and node_c.size == 0
        assert rc_to_id == {} and in_edges == []


# ---------------------------------------------------------------------------
# prune_by_order_gap
# ---------------------------------------------------------------------------


def _two_branch_with_small_tributary() -> _FakeStream:
    """A Y network with one big branch (order 3) and one small tributary (order 1).

    Topology::

        node 0 (order=3) ── trunk head
        node 1 (order=1) ── tiny tributary head
                  \\  /
                node 2 (order=3, confluence — receiving 3 + 1)
                  |
                node 3 (order=3, outlet)
    """
    return _FakeStream(
        node_positions=[(0, 0), (0, 4), (2, 2), (4, 2)],
        edges=[(0, 2), (1, 2), (2, 3)],
        orders=[3, 1, 3, 3],
    )


class TestPruneByOrderGap:
    def test_zero_or_negative_gap_returns_input(self):
        s = _two_branch_with_small_tributary()
        assert prune_by_order_gap(s, 0) is s
        assert prune_by_order_gap(s, -1) is s

    def test_none_input_returns_none(self):
        assert prune_by_order_gap(None, 3) is None

    def test_no_qualifying_junction_returns_input(self):
        # gap=10 — no tributary is that much smaller.
        s = _two_branch_with_small_tributary()
        assert prune_by_order_gap(s, 10) is s

    def test_drops_small_tributary_subtree(self):
        # gap=2 — tributary order 1 is 2 below receiving order 3 -> drop node 1.
        s = _two_branch_with_small_tributary()
        result = prune_by_order_gap(s, 2)
        # Original s.subgraph_call should mark node 1 as removed.
        assert s.subgraph_call is not None
        np.testing.assert_array_equal(s.subgraph_call, [True, False, True, True])
        # Returned object is a new (smaller) StreamObject, not the original.
        assert result is not s

    def test_drops_entire_tributary_subtree_not_just_root(self):
        # 5-node chain: head 0 -> mid 1 -> conf 2; trunk head 3 -> conf 2; conf 2 -> outlet 4.
        # head 0 + mid 1 are both small (order 1), trunk head 3 is order 3.
        s = _FakeStream(
            node_positions=[(0, 0), (1, 1), (2, 2), (0, 4), (4, 2)],
            edges=[(0, 1), (1, 2), (3, 2), (2, 4)],
            orders=[1, 1, 3, 3, 3],
        )
        prune_by_order_gap(s, 2)
        assert s.subgraph_call is not None
        # The tributary subtree (nodes 0 and 1) must both be dropped.
        np.testing.assert_array_equal(s.subgraph_call, [False, False, True, True, True])


# ---------------------------------------------------------------------------
# apply_strategy
# ---------------------------------------------------------------------------


class TestApplyStrategy:
    def test_none_input_returns_none(self):
        assert apply_strategy(None, 1, 4) is None

    def test_pre_remove_zero_only_runs_order_gap(self):
        # With pre_remove=0 the strip stage is skipped; gap=10 means no pruning.
        s = _two_branch_with_small_tributary()
        assert apply_strategy(s, 0, 10) is s

    def test_pre_remove_strips_low_order(self):
        # pre_remove=2 strips orders <=2; gap=10 means the order-gap stage is a no-op.
        s = _FakeStream(
            node_positions=[(0, 0), (0, 4), (2, 2), (4, 2)],
            edges=[(0, 2), (1, 2), (2, 3)],
            orders=[1, 2, 3, 3],
        )
        result = apply_strategy(s, pre_remove_max_order=2, order_gap_to_prune=10)
        # trim_to_min_order(s, 3) keeps only nodes with order >= 3.
        np.testing.assert_array_equal(s.subgraph_call, [False, False, True, True])
        assert result is not None

    def test_strip_then_gap_can_compose(self):
        # 1st-order tributary should be removed by the strip stage (pre_remove=1);
        # the surviving graph then sees no order gap >= 4, so it stays as-is.
        s = _two_branch_with_small_tributary()
        result = apply_strategy(s, pre_remove_max_order=1, order_gap_to_prune=4)
        # After strip, only nodes 0, 2, 3 remain (order >= 2).
        np.testing.assert_array_equal(s.subgraph_call, [True, False, True, True])
        assert result is not None

    def test_strip_removes_everything_returns_none(self):
        s = _FakeStream(
            node_positions=[(0, 0), (0, 1)],
            edges=[(0, 1)],
            orders=[1, 1],
        )
        assert apply_strategy(s, pre_remove_max_order=2, order_gap_to_prune=4) is None


# ---------------------------------------------------------------------------
# Regression: exported from channel_heads top-level namespace
# ---------------------------------------------------------------------------


def test_public_exports():
    import channel_heads as ch

    assert ch.apply_strategy is apply_strategy
    assert ch.prune_by_order_gap is prune_by_order_gap
    assert ch.build_stream_graph is build_stream_graph
