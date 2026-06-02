"""Characterization + unit tests for the shared first-meet pairing core.

The pairing algorithm was extracted from ``first_meet_pairs_for_outlet`` (Earth)
and ``extract_mars_first_meet_pairs`` (Mars) into
``channel_heads.pairing.dag``. ``_reference_first_meet`` below is a frozen,
standalone copy of the *original* propagation algorithm; the characterization
tests assert the new shared core reproduces it exactly on random DAGs, in both
"basin" (subset) and "whole network" (Mars) scopes.
"""

from __future__ import annotations

import random
from collections import defaultdict, deque

from channel_heads.pairing.dag import first_meet_pairs_on_dag, normalize_pair


# --------------------------------------------------------------------------
# Frozen reference: verbatim copy of the pre-refactor algorithm.
# --------------------------------------------------------------------------
def _reference_first_meet(parents, nodes, heads_set, confluences_set):
    nodes = set(nodes)
    children = {v: [] for v in nodes}
    for v in nodes:
        for p in parents[v]:
            if p in nodes:
                children[p].append(v)
    in_degree = {v: 0 for v in nodes}
    for v in nodes:
        for p in parents[v]:
            if p in nodes:
                in_degree[v] += 1
    queue = deque(v for v in nodes if in_degree[v] == 0)
    order = []
    while queue:
        v = queue.popleft()
        order.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    head_sets = {}
    pairs = defaultdict(set)
    for v in order:
        ps = [p for p in parents[v] if p in nodes]
        if not ps:
            head_sets[v] = frozenset({v}) if v in heads_set else frozenset()
        else:
            branch_sets = [head_sets[p] for p in ps]
            if v in confluences_set and len(branch_sets) >= 2:
                for i in range(len(branch_sets)):
                    Hi = branch_sets[i]
                    if not Hi:
                        continue
                    for j in range(i + 1, len(branch_sets)):
                        Hj = branch_sets[j]
                        if not Hj:
                            continue
                        for h in Hi:
                            for g in Hj:
                                a, b = (h, g) if h < g else (g, h)
                                pairs[v].add((a, b))
            merged = set()
            for H in branch_sets:
                merged.update(H)
            head_sets[v] = frozenset(merged)
    return pairs, head_sets


def _random_drainage_tree(n, rng):
    """Random tree rooted at node 0; each node i>0 flows downstream to j<i.

    Returns (parents, heads_set, confluences_set). parents[v] = upstream nodes.
    Heads = nodes with no parents; confluences = nodes with >= 2 parents.
    """
    parents = [[] for _ in range(n)]
    for i in range(1, n):
        down = rng.randrange(0, i)  # strictly smaller -> acyclic, single outlet
        parents[down].append(i)
    heads = {v for v in range(n) if not parents[v]}
    confs = {v for v in range(n) if len(parents[v]) >= 2}
    return parents, heads, confs


def _as_plain(pairs):
    return {k: set(v) for k, v in pairs.items() if v}


# --------------------------------------------------------------------------
# Unit tests on tiny explicit graphs
# --------------------------------------------------------------------------
class TestCoreBasics:
    def test_normalize_pair(self):
        assert normalize_pair(5, 3) == (3, 5)
        assert normalize_pair(3, 5) == (3, 5)

    def test_simple_y(self):
        # heads 0,1 -> confluence 2 -> outlet 3
        parents = [[], [], [0, 1], [2]]
        pairs, head_sets = first_meet_pairs_on_dag(
            parents, {0, 1, 2, 3}, heads_set={0, 1}, confluences_set={2}
        )
        assert _as_plain(pairs) == {2: {(0, 1)}}
        assert head_sets[3] == frozenset({0, 1})

    def test_two_confluences(self):
        # heads 0,1 meet at 2; head 3 joins at 4; 4 -> outlet 5
        # edges: 0->2,1->2,2->4,3->4,4->5
        parents = [[], [], [0, 1], [], [2, 3], [4]]
        pairs, _ = first_meet_pairs_on_dag(
            parents, set(range(6)), heads_set={0, 1, 3}, confluences_set={2, 4}
        )
        # at 2: (0,1); at 4: head 3 (branch via 3) x heads {0,1} (branch via 2)
        assert _as_plain(pairs) == {2: {(0, 1)}, 4: {(0, 3), (1, 3)}}


# --------------------------------------------------------------------------
# Characterization: new core == frozen original algorithm
# --------------------------------------------------------------------------
class TestCharacterizationVsOriginal:
    def test_random_whole_network_scope(self):
        """Mars-style: whole network in scope (nodes = all ids)."""
        for seed in range(60):
            rng = random.Random(seed)
            n = rng.randint(3, 40)
            parents, heads, confs = _random_drainage_tree(n, rng)
            nodes = set(range(n))
            got_pairs, got_hs = first_meet_pairs_on_dag(parents, nodes, heads, confs)
            ref_pairs, ref_hs = _reference_first_meet(parents, nodes, heads, confs)
            assert _as_plain(got_pairs) == _as_plain(ref_pairs), seed
            assert got_hs == ref_hs, seed

    def test_random_basin_subset_scope(self):
        """Earth-style: restrict to the basin upstream of a chosen outlet."""
        for seed in range(60):
            rng = random.Random(1000 + seed)
            n = rng.randint(5, 40)
            parents, _, _ = _random_drainage_tree(n, rng)

            # pick an outlet, collect its basin via reversed-graph DFS
            outlet = rng.randrange(n)
            seen = {outlet}
            stack = [outlet]
            while stack:
                v = stack.pop()
                for p in parents[v]:
                    if p not in seen:
                        seen.add(p)
                        stack.append(p)
            basin = seen
            heads = {v for v in basin if not [p for p in parents[v] if p in basin]}
            confs = {v for v in basin if len([p for p in parents[v] if p in basin]) >= 2}

            got_pairs, got_hs = first_meet_pairs_on_dag(parents, basin, heads, confs)
            ref_pairs, ref_hs = _reference_first_meet(parents, basin, heads, confs)
            assert _as_plain(got_pairs) == _as_plain(ref_pairs), seed
            assert got_hs == ref_hs, seed


# --------------------------------------------------------------------------
# Earth wrapper still produces the expected result via the shared core
# --------------------------------------------------------------------------
class TestMarsGraphHelpers:
    """Mars topology → pairing glue, shared by the script and notebooks/mars/."""

    def test_build_directed_adjacency_and_self_loop_skip(self):
        import pandas as pd

        from channel_heads.pairing import build_directed_adjacency

        # edges: 0->2, 1->2, 2->3, plus a degenerate self-loop 3->3 (skipped)
        segs = pd.DataFrame(
            {
                "upstream_node_id": [0, 1, 2, 3],
                "downstream_node_id": [2, 2, 3, 3],
                "segment_id": [10, 11, 12, 13],
            }
        )
        parents, children, edge_to_seg = build_directed_adjacency(segs, 4)
        assert parents[2] == [0, 1]
        assert parents[3] == [2]  # self-loop 3->3 skipped
        assert children[0] == [2] and children[2] == [3]
        assert edge_to_seg == {(0, 2): 10, (1, 2): 11, (2, 3): 12}

    def test_trace_downstream_path(self):
        from channel_heads.pairing import trace_downstream_path

        children = [[2], [2], [3], []]  # 0->2,1->2,2->3,3=outlet
        assert trace_downstream_path(0, 3, children, max_steps=5) == [0, 2, 3]
        assert trace_downstream_path(3, 0, children, max_steps=5) is None

    def test_chain_segment_geometries_reverses_as_needed(self):
        from shapely.geometry import LineString

        from channel_heads.pairing import chain_segment_geometries

        # path 0->1->2; seg 100 stored 0->1 (as-is), seg 101 stored 2->1 (reversed)
        edge_to_seg = {(0, 1): 100, (1, 2): 101}
        geom = {100: LineString([(0, 0), (1, 0)]), 101: LineString([(2, 0), (1, 0)])}
        start_by = {100: 0, 101: 2}
        end_by = {100: 1, 101: 1}
        line, seg_ids = chain_segment_geometries([0, 1, 2], edge_to_seg, geom, start_by, end_by)
        assert seg_ids == [100, 101]
        assert list(line.coords) == [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]

    def test_detect_crossed_segments(self):
        import geopandas as gpd
        from shapely.geometry import LineString

        from channel_heads.pairing import detect_crossed_segments

        # head-to-head line (0,0)->(4,0); segment 7 crosses it at x=2.
        # Small head buffer so the line's interior survives (the 2-unit gap to
        # each head exceeds the buffer).
        segs = gpd.GeoDataFrame({"segment_id": [7]}, geometry=[LineString([(2, -1), (2, 1)])])
        assert detect_crossed_segments((0.0, 0.0), (4.0, 0.0), segs, head_buffer_m=0.5) == [7]
        # far-away segment -> not crossed
        far = gpd.GeoDataFrame({"segment_id": [8]}, geometry=[LineString([(10, -1), (10, 1)])])
        assert detect_crossed_segments((0.0, 0.0), (4.0, 0.0), far, head_buffer_m=0.5) == []
        # a crossing within the head buffer is excluded (mirrors the filter)
        assert detect_crossed_segments((0.0, 0.0), (4.0, 0.0), segs, head_buffer_m=5.0) == []
        # coincident heads -> empty
        assert detect_crossed_segments((0.0, 0.0), (0.0, 0.0), segs) == []

    def test_chain_missing_edge_returns_none(self):
        from shapely.geometry import LineString

        from channel_heads.pairing import chain_segment_geometries

        line, seg_ids = chain_segment_geometries(
            [0, 1], {}, {0: LineString([(0, 0), (1, 0)])}, {}, {}
        )
        assert line is None and seg_ids == []


class TestEarthWrapper:
    def test_simple_y_network_fixture(self, simple_y_network):
        from channel_heads.first_meet_pairs_for_outlet import first_meet_pairs_for_outlet

        net = simple_y_network
        pairs, heads = first_meet_pairs_for_outlet(net["s"], net["outlets"][0])
        # one confluence with the single head pair; both heads in basin
        assert len(heads) == 2
        all_pairs = {p for s in pairs.values() for p in s}
        assert len(all_pairs) == 1
