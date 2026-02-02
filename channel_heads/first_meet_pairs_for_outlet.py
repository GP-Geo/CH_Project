"""Channel head pairing algorithm for first-meet detection.

This module implements the core algorithm for identifying pairs of channel heads
that first meet at each confluence in a stream network.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    pass

# Type aliases for clarity
NodeId = int
HeadId = int
HeadPair = tuple[HeadId, HeadId]  # normalized (min, max)
PairsAtConfluence = dict[int, set[HeadPair]]
ParentsList = list[list[NodeId]]

# ------------------------------ helpers ------------------------------


def _normalize_pair(h1: int, h2: int) -> HeadPair:
    return (h1, h2) if h1 < h2 else (h2, h1)


### streampoi returns always numpy arrays, so this is not strictly necessary,
# change this for simplicity
# reduced the function to only handle 1D numpy arrays


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
        r, c = s.node_indices
        n = int(len(r))
    else:
        n = int(max(int(np.max(s.source)), int(np.max(s.target))) + 1)

    src = s.source.ravel()
    tgt = s.target.ravel()
    if src.shape != tgt.shape:
        raise ValueError("s.source and s.target must have identical shape")

    parents_sets: list[set[int]] = [set() for _ in range(n)]
    for i in range(src.size):
        u = int(src[i])
        v = int(tgt[i])
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


def _build_children_from_parents(
    parents: ParentsList, basin_nodes: set[int]
) -> dict[int, list[int]]:
    """Build children adjacency (downstream direction) restricted to basin.

    Parameters
    ----------
    parents : ParentsList
        Adjacency list where parents[v] contains upstream neighbors of v.
    basin_nodes : Set[int]
        Set of node IDs in the basin.

    Returns
    -------
    Dict[int, List[int]]
        children[v] = list of downstream nodes (nodes that have v as parent).
    """
    children: dict[int, list[int]] = {v: [] for v in basin_nodes}
    for v in basin_nodes:
        for p in parents[v]:
            if p in basin_nodes:
                children[p].append(v)
    return children


def _topological_sort_basin(
    basin_nodes: set[int],
    parents: ParentsList,
    children: dict[int, list[int]],
) -> list[int]:
    """Topological sort of basin nodes from leaves (heads) to outlet.

    Uses Kahn's algorithm for iterative topological sorting.

    Parameters
    ----------
    basin_nodes : Set[int]
        Set of node IDs in the basin.
    parents : ParentsList
        Adjacency list where parents[v] contains upstream neighbors of v.
    children : Dict[int, List[int]]
        Adjacency list where children[v] contains downstream neighbors of v.

    Returns
    -------
    List[int]
        Nodes in topological order (upstream to downstream).
    """
    # Count incoming edges (from parents within basin) for each node
    in_degree = {v: 0 for v in basin_nodes}
    for v in basin_nodes:
        for p in parents[v]:
            if p in basin_nodes:
                in_degree[v] += 1

    # Start with nodes that have no parents in basin (channel heads and edge nodes)
    queue = [v for v in basin_nodes if in_degree[v] == 0]
    sorted_nodes: list[int] = []

    while queue:
        v = queue.pop(0)
        sorted_nodes.append(v)
        # For each child (downstream node), decrement in-degree
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)

    return sorted_nodes


def first_meet_pairs_for_outlet(
    s: Any,  # StreamObject
    outlet: NodeId,
) -> tuple[PairsAtConfluence, list[int]]:
    """Compute first-meet channel-head pairs per confluence for a single outlet.

    This function identifies all pairs of channel heads that first meet at each
    confluence within the drainage basin of a specified outlet. A "first meet"
    occurs when two channel heads from different upstream branches converge
    at a confluence for the first time.

    Uses an iterative topological-sort approach to avoid recursion depth issues
    on large networks.

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
    heads_all = _to_node_id_list(s.streampoi("channelheads"))
    confs_all = _to_node_id_list(s.streampoi("confluences"))

    heads_set = set(h for h in heads_all if h in basin_nodes)
    confluences = set(c for c in confs_all if c in basin_nodes)

    # 3) Build children adjacency and topological order
    children = _build_children_from_parents(parents, basin_nodes)
    sorted_nodes = _topological_sort_basin(basin_nodes, parents, children)

    # 4) Iteratively compute head sets from upstream to downstream
    # head_sets[v] = frozenset of channel heads upstream of v (including v if it's a head)
    head_sets: dict[int, frozenset[int]] = {}
    pairs_at_confluence: dict[int, set[HeadPair]] = defaultdict(set)

    for v in sorted_nodes:
        # Get parents within basin
        ps = [p for p in parents[v] if p in basin_nodes]

        if not ps:
            # Leaf node - is it a channel head?
            head_sets[v] = frozenset({v}) if v in heads_set else frozenset()
        else:
            # Get head sets from each parent branch
            branch_sets = [head_sets[p] for p in ps]

            # Emit first-meet pairs at confluences (only across distinct branches)
            if v in confluences and len(branch_sets) >= 2:
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
                                pairs_at_confluence[v].add(_normalize_pair(int(h), int(g)))

            # Merge all branch head sets
            merged: set[int] = set()
            for H in branch_sets:
                merged.update(H)
            head_sets[v] = frozenset(merged)

    # Get basin heads from the outlet's head set
    basin_heads = sorted(int(h) for h in head_sets.get(int(outlet), frozenset()))

    return pairs_at_confluence, basin_heads
