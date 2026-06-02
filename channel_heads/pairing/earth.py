"""Earth/TopoToolbox first-meet adapter for drainage basins.

The graph-agnostic first-meet algorithm lives in :mod:`channel_heads.pairing.dag`
and is shared with the Mars pipeline. This module keeps the StreamObject-specific
parts: building parent adjacency from ``s.source`` / ``s.target``, collecting a
drainage basin upstream of an outlet, and reading points-of-interest via
``s.streampoi``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dag import build_children_from_parents as _build_children_from_parents
from .dag import first_meet_pairs_on_dag
from .dag import normalize_pair as _normalize_pair
from .dag import topological_sort as _topological_sort_basin

NodeId = int
HeadId = int
HeadPair = tuple[HeadId, HeadId]
PairsAtConfluence = dict[int, set[HeadPair]]
ParentsList = list[list[NodeId]]

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


def _to_node_id_list(x) -> list[int]:
    """Convert a TopoToolbox POI mask or node-id array to sorted unique ids."""

    arr = np.asarray(x)
    if arr.dtype == bool:
        idx = np.nonzero(arr)[0]
    else:
        idx = arr.ravel()
    return sorted({int(v) for v in idx})


def _build_parents_from_stream(s: Any) -> ParentsList:
    """Build ``parents[v]`` from a TopoToolbox ``StreamObject``.

    TopoToolbox exposes stream edges as ``s.source`` / ``s.target`` arrays whose
    ids index stream-network nodes. When ``node_indices`` is available, use it
    to preserve the full node-id domain even if the highest-id node has no edge.
    """

    if hasattr(s, "node_indices") and s.node_indices is not None:
        node_indices = s.node_indices() if callable(s.node_indices) else s.node_indices
        r, _ = node_indices
        n = int(len(r))
    else:
        n = int(max(int(np.max(s.source)), int(np.max(s.target))) + 1)

    src = s.source.ravel()
    tgt = s.target.ravel()
    if src.shape != tgt.shape:
        raise ValueError("s.source and s.target must have identical shape")

    parents_sets: list[set[int]] = [set() for _ in range(n)]
    for i, (u_raw, v_raw) in enumerate(zip(src, tgt)):
        u, v = int(u_raw), int(v_raw)
        if u == v:
            continue
        if not (0 <= u < n) or not (0 <= v < n):
            raise IndexError(f"edge {i} out of range: (u={u}, v={v}), n={n}")
        parents_sets[v].add(u)
    return [sorted(p) for p in parents_sets]


def _collect_basin_nodes_from_outlet(parents: ParentsList, outlet: int) -> list[int]:
    """Return all nodes upstream of an outlet, including the outlet itself."""

    n = len(parents)
    if outlet < 0 or outlet >= n:
        raise IndexError(f"outlet {outlet} out of range [0, {n})")

    seen = [False] * n
    stack = [outlet]
    seen[outlet] = True
    while stack:
        v = stack.pop()
        for p in parents[v]:
            if not seen[p]:
                seen[p] = True
                stack.append(p)
    return [i for i, ok in enumerate(seen) if ok]


def first_meet_pairs_for_outlet(
    s: Any,
    outlet: NodeId,
) -> tuple[PairsAtConfluence, list[int]]:
    """Compute first-meet channel-head pairs per confluence for one outlet.

    The Earth-specific work is limited to TopoToolbox stream extraction and
    basin scoping. Pair propagation is delegated to
    :func:`channel_heads.pairing.dag.first_meet_pairs_on_dag`.
    """

    parents = _build_parents_from_stream(s)
    basin_nodes = set(_collect_basin_nodes_from_outlet(parents, outlet))

    heads_set = set(_to_node_id_list(s.streampoi("channelheads"))) & basin_nodes
    confluences = set(_to_node_id_list(s.streampoi("confluences"))) & basin_nodes

    pairs_at_confluence, head_sets = first_meet_pairs_on_dag(
        parents, basin_nodes, heads_set, confluences
    )
    basin_heads = sorted(int(h) for h in head_sets.get(int(outlet), frozenset()))

    return pairs_at_confluence, basin_heads
