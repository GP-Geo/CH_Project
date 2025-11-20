from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import defaultdict, deque

NodeId = int
HeadId = int
HeadPair = Tuple[HeadId, HeadId]  # normalized (min, max)
HeadSet = List[HeadId]

# ------------------------------ helpers ------------------------------
def _normalize_pair(h1: int, h2: int) -> HeadPair:
    return (h1, h2) if h1 < h2 else (h2, h1)


def _to_node_id_list(x) -> List[int]:
    """Coerce streampoi output to a sorted unique Python list of ints.
    Accepts int, list/tuple, numpy array, or boolean mask aligned to node ids.
    """
    import numpy as np

    if x is None:
        return []
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


def _build_parents_from_stream(s) -> List[List[NodeId]]:
    """parents[v] = list of upstream nodes u with edge (u -> v). Deterministic & duplicate-safe."""
    import numpy as np

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
        u = int(src[i]); v = int(tgt[i])
        if u == v:
            continue
        if not (0 <= u < n) or not (0 <= v < n):
            raise IndexError(f"edge {i} out of range: (u={u}, v={v}), n={n}")
        parents_sets[v].add(u)
    return [sorted(p) for p in parents_sets]


def _collect_basin_nodes_from_outlet(parents: List[List[int]], outlet: int) -> List[int]:
    """Return all nodes upstream (including the outlet) reachable via `parents` starting at outlet.
    We do a simple DFS/BFS on the reversed graph.
    """
    n = len(parents)
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
def first_meet_pairs_for_outlet(s, outlet: NodeId):
    """Compute first-meet channel-head pairs per confluence for a single outlet.

    Parameters
    ----------
    s : StreamObject-like
        Must expose `.source`, `.target`, `.node_indices`, `.streampoi(key)`.
    outlet : int
        Node id for the outlet to analyze.

    Returns
    -------
    pairs_at_confluence : Dict[int, Set[Tuple[int,int]]]
        {confluence_node -> set of normalized head pairs}
    basin_heads : List[int]
        Sorted list of head node ids within the basin of `outlet`.
    """
    import numpy as np
    from functools import lru_cache

    # 1) parents and basin restriction
    parents = _build_parents_from_stream(s)
    basin_nodes = set(_collect_basin_nodes_from_outlet(parents, outlet))

    # 2) get global POIs via streampoi, then restrict to basin
    heads_all = _to_node_id_list(s.streampoi('channelheads'))
    confs_all = _to_node_id_list(s.streampoi('confluences'))

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
