"""Channel head pairing algorithm for first-meet detection.

This module implements the core algorithm for identifying pairs of channel heads
that first meet at each confluence in a stream network.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Set, Tuple
from collections import defaultdict
import numpy as np
import numpy.typing as npt
from functools import lru_cache

if TYPE_CHECKING:
    from topotoolbox import StreamObject

# Type aliases for clarity
NodeId = int
HeadId = int
HeadPair = Tuple[HeadId, HeadId]  # normalized (min, max)
PairsAtConfluence = Dict[int, Set[HeadPair]]
ParentsList = List[List[NodeId]]

# ------------------------------ helpers ------------------------------


def _normalize_pair(h1: int, h2: int) -> HeadPair:
    return (h1, h2) if h1 < h2 else (h2, h1)


### streampoi returns always numpy arrays, so this is not strictly necessary,
# change this for simplicity
# reduced the function to only handle 1D numpy arrays


def _to_node_id_list(x) -> List[int]:
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

    parents_sets: List[Set[int]] = [set() for _ in range(n)]
    for i in range(src.size):
        u = int(src[i])
        v = int(tgt[i])
        if u == v:
            continue
        if not (0 <= u < n) or not (0 <= v < n):
            raise IndexError(f"edge {i} out of range: (u={u}, v={v}), n={n}")
        parents_sets[v].add(u)
    return [sorted(p) for p in parents_sets]


def _collect_basin_nodes_from_outlet(parents: ParentsList, outlet: int) -> List[int]:
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
) -> Tuple[PairsAtConfluence, List[int]]:
    """Compute first-meet channel-head pairs per confluence for a single outlet.

    This function identifies all pairs of channel heads that first meet at each
    confluence within the drainage basin of a specified outlet. A "first meet"
    occurs when two channel heads from different upstream branches converge
    at a confluence for the first time.

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

    heads = sorted(h for h in heads_all if h in basin_nodes)
    confluences = set(c for c in confs_all if c in basin_nodes)

    # 3) define memoized upstream head-set (as frozenset) *within the basin*
    pairs_at_confluence: Dict[int, Set[HeadPair]] = defaultdict(set)

    @lru_cache(maxsize=None)
    def head_set(v: int) -> frozenset:
        # Use only parents that are inside basin
        ps = [p for p in parents[v] if p in basin_nodes]
        if not ps:
            return frozenset({v}) if v in heads else frozenset()

        branch_sets = [head_set(p) for p in ps]

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

        # Return union of branch head-sets
        if not branch_sets:
            return frozenset()
        U: Set[int] = set()
        for H in branch_sets:
            if H:
                U.update(H)
        return frozenset(U)

    basin_heads = sorted(int(h) for h in head_set(int(outlet)))
    head_set.cache_clear()

    return pairs_at_confluence, basin_heads
