"""First-meet channel-head pairing.

Shared DAG algorithms live in :mod:`channel_heads.pairing.dag`. Earth
TopoToolbox basin adaptation lives in :mod:`channel_heads.pairing.earth`; Mars
valley-network graph helpers live in :mod:`channel_heads.pairing.mars_graph`.
"""

from __future__ import annotations

from .dag import (
    build_children_from_parents,
    first_meet_pairs_on_dag,
    normalize_pair,
    topological_sort,
)
from .earth import first_meet_pairs_for_outlet

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
    "first_meet_pairs_for_outlet",
    "first_meet_pairs_on_dag",
    "normalize_pair",
    "topological_sort",
    "trace_downstream_path",
]
