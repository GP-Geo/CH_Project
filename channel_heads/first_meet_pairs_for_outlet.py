"""Channel head pairing for first-meet detection on a TopoToolbox basin.

The graph-agnostic first-meet algorithm now lives in
:mod:`channel_heads.pairing.dag` and is shared with the Mars pipeline. This
module keeps the StreamObject-specific parts: building parent adjacency from
``s.source`` / ``s.target``, collecting a drainage basin upstream of an outlet,
and reading points-of-interest via ``s.streampoi``. The pairing math is
delegated to the shared core, so behavior is unchanged from previous versions.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .pairing.dag import build_children_from_parents as _build_children_from_parents
from .pairing.dag import first_meet_pairs_on_dag
from .pairing.dag import normalize_pair as _normalize_pair
from .pairing.dag import topological_sort as _topological_sort_basin

# Type aliases for clarity
NodeId = int
HeadId = int
HeadPair = tuple[HeadId, HeadId]  # normalized (min, max)
PairsAtConfluence = dict[int, set[HeadPair]]
ParentsList = list[list[NodeId]]

# Names re-exported from the shared pairing core for backward compatibility
# (imported by geometric_analysis, rasterizer, and the test suite).
__all__ = [
    "first_meet_pairs_for_outlet",
    "_to_node_id_list",
    "_build_parents_from_stream",
    "_collect_basin_nodes_from_outlet",
    "_normalize_pair",
    "_build_children_from_parents",
    "_topological_sort_basin",
]

# ------------------------------ helpers ------------------------------


def _to_node_id_list(x) -> list[int]:
    """Convert input to sorted unique list of node IDs where input is truthy."""

    arr = np.asarray(x)
    if arr.dtype == bool:
        # boolean mask over node ids
        idx = np.nonzero(arr)[0]
    else:
        idx = arr.ravel()
    # ensure Python ints
    ids = [int(v) for v in idx]
    # unique + sorted
    return sorted(set(ids))


def _build_parents_from_stream(s: Any) -> ParentsList:
    """parents[v] = list of upstream nodes u with edge (u -> v). Deterministic & duplicate-safe."""

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
    """Return all nodes upstream of an outlet (including the outlet itself).

    Performs depth-first traversal on the reversed graph (parent edges).

    Parameters
    ----------
    parents : ParentsList
        Adjacency list where parents[v] contains upstream neighbors of v.
    outlet : int
        Node ID of the outlet to start traversal from.

    Returns
    -------
    List[int]
        Sorted list of all node IDs in the basin.
    """
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


# ------------------------------ main routine ------------------------------


def first_meet_pairs_for_outlet(
    s: Any,  # StreamObject
    outlet: NodeId,
) -> tuple[PairsAtConfluence, list[int]]:
    """Compute first-meet channel-head pairs per confluence for a single outlet.

    This function identifies all pairs of channel heads that first meet at each
    confluence within the drainage basin of a specified outlet. A "first meet"
    occurs when two channel heads from different upstream branches converge
    at a confluence for the first time.

    The graph algorithm is delegated to
    :func:`channel_heads.pairing.dag.first_meet_pairs_on_dag`; this wrapper only
    builds the StreamObject basin scope. Uses an iterative topological-sort
    approach to avoid recursion depth issues on large networks.

    Parameters
    ----------
    s : StreamObject
        TopoToolbox StreamObject with `.source`, `.target`, `.node_indices`,
        and `.streampoi(key)` methods.
    outlet : int
        Node ID of the outlet to analyze.

    Returns
    -------
    pairs_at_confluence : Dict[int, Set[Tuple[int, int]]]
        Dictionary mapping confluence node IDs to sets of normalized head pairs.
        Each pair is (min_head_id, max_head_id).
    basin_heads : List[int]
        Sorted list of all channel head node IDs within the outlet's basin.

    Example
    -------
    >>> pairs, heads = first_meet_pairs_for_outlet(s, outlet_id=5)
    >>> for confluence, head_pairs in pairs.items():
    ...     print(f"Confluence {confluence}: {len(head_pairs)} pairs")
    """

    # 1) Build parent adjacency and restrict to basin
    parents = _build_parents_from_stream(s)
    basin_nodes = set(_collect_basin_nodes_from_outlet(parents, outlet))

    # 2) Get global POIs via streampoi, then restrict to basin
    heads_set = set(_to_node_id_list(s.streampoi("channelheads"))) & basin_nodes
    confluences = set(_to_node_id_list(s.streampoi("confluences"))) & basin_nodes

    # 3) Delegate the topological-sort + head-set propagation to the shared core
    pairs_at_confluence, head_sets = first_meet_pairs_on_dag(
        parents, basin_nodes, heads_set, confluences
    )

    # 4) Basin heads = the head set accumulated at the outlet
    basin_heads = sorted(int(h) for h in head_sets.get(int(outlet), frozenset()))

    return pairs_at_confluence, basin_heads
