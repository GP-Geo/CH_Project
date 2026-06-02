"""Graph-agnostic first-meet channel-head pairing.

Shared by the Earth (``StreamObject`` basin) and Mars (valley-network graph)
pipelines. See :mod:`channel_heads.pairing.dag`.
"""

from __future__ import annotations

from .dag import (
    build_children_from_parents,
    first_meet_pairs_on_dag,
    normalize_pair,
    topological_sort,
)
from .mars_graph import (
    build_directed_adjacency,
    chain_segment_geometries,
    detect_crossed_segments,
    trace_downstream_path,
)

__all__ = [
    "build_children_from_parents",
    "build_directed_adjacency",
    "chain_segment_geometries",
    "detect_crossed_segments",
    "first_meet_pairs_on_dag",
    "normalize_pair",
    "topological_sort",
    "trace_downstream_path",
]
