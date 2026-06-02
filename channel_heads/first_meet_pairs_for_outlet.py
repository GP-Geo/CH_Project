"""Compatibility shim for Earth first-meet pairing.

The implementation moved to :mod:`channel_heads.pairing.earth`. Keep this
module for notebooks, scripts, and user code that still import the historical
path.
"""

from __future__ import annotations

from .pairing.earth import (
    HeadId,
    HeadPair,
    NodeId,
    PairsAtConfluence,
    ParentsList,
    _build_children_from_parents,
    _build_parents_from_stream,
    _collect_basin_nodes_from_outlet,
    _normalize_pair,
    _to_node_id_list,
    _topological_sort_basin,
    first_meet_pairs_for_outlet,
)

__all__ = [
    "NodeId",
    "HeadId",
    "HeadPair",
    "PairsAtConfluence",
    "ParentsList",
    "first_meet_pairs_for_outlet",
    "_to_node_id_list",
    "_build_parents_from_stream",
    "_collect_basin_nodes_from_outlet",
    "_normalize_pair",
    "_build_children_from_parents",
    "_topological_sort_basin",
]
