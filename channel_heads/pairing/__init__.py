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

# Mars graph helpers require geopandas/shapely. Keep them optional so the
# pure-Python DAG core (and its tests) import without the geo stack installed.
try:
    from .mars_graph import (
        build_directed_adjacency,
        chain_segment_geometries,
        detect_crossed_segments,
        trace_downstream_path,
    )
except ImportError:
    build_directed_adjacency = None
    chain_segment_geometries = None
    detect_crossed_segments = None
    trace_downstream_path = None

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
