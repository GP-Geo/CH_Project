"""Tests for first_meet_pairs_for_outlet module."""

import pytest
from channel_heads.first_meet_pairs_for_outlet import (
    _normalize_pair,
    _to_node_id_list,
    _build_parents_from_stream,
    _collect_basin_nodes_from_outlet,
    first_meet_pairs_for_outlet,
)
import numpy as np


class TestNormalizePair:
    """Test pair normalization function."""

    def test_already_normalized(self):
        """Test pair that's already in correct order."""
        assert _normalize_pair(3, 5) == (3, 5)

    def test_needs_swap(self):
        """Test pair that needs swapping."""
        assert _normalize_pair(5, 3) == (3, 5)

    def test_equal_values(self):
        """Test pair with equal values."""
        assert _normalize_pair(4, 4) == (4, 4)

    def test_large_values(self):
        """Test with large node IDs."""
        assert _normalize_pair(1000, 500) == (500, 1000)


class TestToNodeIdList:
    """Test node ID list conversion."""

    def test_boolean_mask(self):
        """Test conversion from boolean mask."""
        mask = np.array([False, True, False, True, True])
        result = _to_node_id_list(mask)
        assert result == [1, 3, 4]

    def test_index_array(self):
        """Test conversion from index array."""
        indices = np.array([5, 2, 8, 2])  # with duplicate
        result = _to_node_id_list(indices)
        assert result == [2, 5, 8]  # sorted and unique

    def test_empty_mask(self):
        """Test empty mask."""
        mask = np.array([False, False, False])
        result = _to_node_id_list(mask)
        assert result == []

    def test_all_true_mask(self):
        """Test all-true mask."""
        mask = np.array([True, True, True])
        result = _to_node_id_list(mask)
        assert result == [0, 1, 2]


class TestBuildParentsFromStream:
    """Test parent adjacency list construction."""

    def test_build_parents_simple_y(self, simple_y_network):
        """Test parent construction for simple Y network."""
        net = simple_y_network
        parents = _build_parents_from_stream(net["s"])

        # Should have 7 nodes (length matches node_positions)
        assert len(parents) == 7

        # Node 0 (head left) has no parents
        assert parents[0] == []

        # Node 1 (head right) has no parents
        assert parents[1] == []

        # Confluence node 4 should have parents 2 and 3 (intermediate nodes)
        assert set(parents[4]) == {2, 3}

    def test_build_parents_complex(self, complex_network):
        """Test parent construction for complex network."""
        net = complex_network
        parents = _build_parents_from_stream(net["s"])

        # Should have 8 nodes
        assert len(parents) == 8

        # Main confluence (node 6) has parents 4 and 5
        assert set(parents[6]) == {4, 5}

        # Outlet (node 7) has parent 6
        assert parents[7] == [6]


class TestCollectBasinNodes:
    """Test basin node collection."""

    def test_collect_basin_simple_y(self, simple_y_network):
        """Test collecting basin nodes for Y network."""
        net = simple_y_network
        parents = _build_parents_from_stream(net["s"])

        # Collect from outlet (node 6)
        basin = _collect_basin_nodes_from_outlet(parents, 6)

        # Should include all nodes (0-6)
        assert set(basin) == {0, 1, 2, 3, 4, 5, 6}

    def test_collect_basin_from_confluence(self, simple_y_network):
        """Test collecting basin from confluence (not outlet)."""
        net = simple_y_network
        parents = _build_parents_from_stream(net["s"])

        # Collect from confluence (node 4)
        basin = _collect_basin_nodes_from_outlet(parents, 4)

        # Should include upstream nodes: 0, 1, 2, 3, 4
        assert set(basin) == {0, 1, 2, 3, 4}

    def test_collect_basin_invalid_outlet(self, simple_y_network):
        """Test that invalid outlet raises IndexError."""
        net = simple_y_network
        parents = _build_parents_from_stream(net["s"])

        with pytest.raises(IndexError):
            _collect_basin_nodes_from_outlet(parents, 100)


class TestFirstMeetPairsForOutlet:
    """Integration tests for first_meet_pairs_for_outlet."""

    def test_simple_y_returns_tuple(self, simple_y_network):
        """Test that function returns (dict, list) tuple."""
        net = simple_y_network
        outlet_id = net["outlets"][0]

        pairs, basin_heads = first_meet_pairs_for_outlet(net["s"], outlet_id)

        assert isinstance(pairs, dict)
        assert isinstance(basin_heads, list)

    def test_simple_y_finds_heads(self, simple_y_network):
        """Test that all basin heads are found."""
        net = simple_y_network
        outlet_id = net["outlets"][0]

        pairs, basin_heads = first_meet_pairs_for_outlet(net["s"], outlet_id)

        # Should find both channel heads (0 and 1)
        assert set(basin_heads) == {0, 1}

    def test_simple_y_finds_pair(self, simple_y_network):
        """Test that head pair is found at confluence."""
        net = simple_y_network
        outlet_id = net["outlets"][0]

        pairs, basin_heads = first_meet_pairs_for_outlet(net["s"], outlet_id)

        # Confluence is node 4 in this network
        confluence_id = net["confluences"][0]

        # Should have one pair at the confluence
        assert confluence_id in pairs
        assert len(pairs[confluence_id]) == 1

        # The pair should be (0, 1) normalized
        pair = list(pairs[confluence_id])[0]
        assert pair == (0, 1)

    def test_complex_network_multiple_pairs(self, complex_network):
        """Test with multiple confluences."""
        net = complex_network
        outlet_id = net["outlets"][0]

        pairs, basin_heads = first_meet_pairs_for_outlet(net["s"], outlet_id)

        # Should find all 4 channel heads
        assert set(basin_heads) == {0, 1, 2, 3}

        # Should have pairs at all 3 confluences
        for conf_id in net["confluences"]:
            assert conf_id in pairs

    def test_complex_first_level_pairs(self, complex_network):
        """Test pairs at first level confluences."""
        net = complex_network
        outlet_id = net["outlets"][0]

        pairs, _ = first_meet_pairs_for_outlet(net["s"], outlet_id)

        # Confluence 4: heads 0 and 1 meet
        assert (0, 1) in pairs[4]

        # Confluence 5: heads 2 and 3 meet
        assert (2, 3) in pairs[5]

    def test_complex_main_confluence_pairs(self, complex_network):
        """Test pairs at main confluence (cross-branch)."""
        net = complex_network
        outlet_id = net["outlets"][0]

        pairs, _ = first_meet_pairs_for_outlet(net["s"], outlet_id)

        # Main confluence 6: heads from left branch (0,1) meet heads from right branch (2,3)
        main_conf_pairs = pairs[6]

        # Should have 4 cross-branch pairs
        expected = {(0, 2), (0, 3), (1, 2), (1, 3)}
        assert main_conf_pairs == expected

    def test_pairs_are_normalized(self, complex_network):
        """Test that all pairs are normalized (min, max)."""
        net = complex_network
        outlet_id = net["outlets"][0]

        pairs, _ = first_meet_pairs_for_outlet(net["s"], outlet_id)

        for conf_id, pair_set in pairs.items():
            for h1, h2 in pair_set:
                assert h1 < h2, f"Pair ({h1}, {h2}) at conf {conf_id} not normalized"


class TestFirstMeetPairsEdgeCases:
    """Test edge cases for first_meet_pairs_for_outlet."""

    def test_single_head_basin(self):
        """Test basin with single head (no pairs)."""
        from tests.conftest import MockStreamObject

        # Create a simple linear stream: head -> outlet
        node_positions = [(0, 0), (1, 0)]  # head, outlet
        edges = [(0, 1)]
        s = MockStreamObject(
            node_positions=node_positions,
            edges=edges,
            channelheads=[0],
            outlets=[1],
            confluences=[],
            grid_shape=(5, 5),
        )

        pairs, basin_heads = first_meet_pairs_for_outlet(s, outlet=1)

        # Should find the single head
        assert basin_heads == [0]

        # Should have no pairs (no confluences)
        assert len(pairs) == 0 or all(len(p) == 0 for p in pairs.values())

    def test_empty_edges(self):
        """Test handling of empty edge list."""
        from tests.conftest import MockStreamObject

        # Single node stream
        s = MockStreamObject(
            node_positions=[(0, 0)],
            edges=[],
            channelheads=[0],
            outlets=[0],
            confluences=[],
            grid_shape=(3, 3),
        )

        pairs, basin_heads = first_meet_pairs_for_outlet(s, outlet=0)

        # Should handle gracefully
        assert basin_heads == [0]
        assert len(pairs) == 0
