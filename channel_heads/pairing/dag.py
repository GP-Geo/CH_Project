"""Generic first-meet channel-head pairing on a directed acyclic graph.

This is the graph-agnostic core shared by the Earth pipeline
(``channel_heads.first_meet_pairs_for_outlet``, operating on a TopoToolbox
``StreamObject`` basin) and the Mars pipeline
(``scripts/extract_mars_first_meet_pairs.py``, operating on a mapped
valley-network graph).

The algorithm is unchanged from the original Earth implementation — it has
simply been relocated and parameterized by the set of ``nodes`` in scope
(a drainage basin for Earth, a whole network for Mars). Outputs are sets of
normalized ``(min, max)`` head pairs, so they do not depend on iteration order.

A *first meet* of two channel heads happens at the confluence where their two
upstream branches first merge. Working from leaves to outlet in topological
order, each node carries the set of heads upstream of it; at a confluence we
emit every cross-branch head pair, then merge the branch head-sets.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

NodeId = int
HeadPair = tuple[int, int]  # normalized (min, max)
ParentsList = Any  # sequence indexable by node id -> list of parent node ids


def normalize_pair(h1: int, h2: int) -> HeadPair:
    """Return a head pair as ``(min, max)`` so pairs are order-independent."""
    return (h1, h2) if h1 < h2 else (h2, h1)


def build_children_from_parents(parents: ParentsList, nodes: set[int]) -> dict[int, list[int]]:
    """Build children adjacency (downstream direction) restricted to ``nodes``.

    ``children[v]`` = list of downstream nodes (nodes that have ``v`` as parent),
    keeping only edges whose endpoints are both in ``nodes``.
    """
    children: dict[int, list[int]] = {v: [] for v in nodes}
    for v in nodes:
        for p in parents[v]:
            if p in nodes:
                children[p].append(v)
    return children


def topological_sort(
    nodes: set[int],
    parents: ParentsList,
    children: dict[int, list[int]],
) -> list[int]:
    """Kahn topological sort of ``nodes`` from leaves (heads) to outlet.

    In-degree counts only parents within ``nodes`` (so the scope can be a
    sub-graph such as a single drainage basin).
    """
    in_degree = {v: 0 for v in nodes}
    for v in nodes:
        for p in parents[v]:
            if p in nodes:
                in_degree[v] += 1

    queue: deque[int] = deque(v for v in nodes if in_degree[v] == 0)
    sorted_nodes: list[int] = []
    while queue:
        v = queue.popleft()
        sorted_nodes.append(v)
        for c in children[v]:
            in_degree[c] -= 1
            if in_degree[c] == 0:
                queue.append(c)
    return sorted_nodes


def first_meet_pairs_on_dag(
    parents: ParentsList,
    nodes: set[int],
    heads_set: set[int],
    confluences_set: set[int],
) -> tuple[dict[int, set[HeadPair]], dict[int, frozenset[int]]]:
    """Compute first-meet head pairs per confluence over a DAG scope.

    Parameters
    ----------
    parents : sequence indexable by node id
        ``parents[v]`` is the list of upstream nodes ``u`` with edge ``u -> v``.
    nodes : set[int]
        The node ids in scope (a basin for Earth, a whole network for Mars).
    heads_set : set[int]
        Channel-head node ids (intersected with ``nodes`` by the caller, or not).
    confluences_set : set[int]
        Confluence node ids.

    Returns
    -------
    pairs_at_confluence : dict[int, set[(min, max)]]
        Normalized first-meet head pairs keyed by confluence node id.
        (A ``defaultdict(set)``, matching the historical return type.)
    head_sets : dict[int, frozenset[int]]
        ``head_sets[v]`` = frozenset of channel heads upstream of ``v``
        (including ``v`` itself if it is a head). Callers derive e.g. the
        outlet's full head list from this.
    """
    nodes = set(nodes)
    children = build_children_from_parents(parents, nodes)
    sorted_nodes = topological_sort(nodes, parents, children)

    head_sets: dict[int, frozenset[int]] = {}
    pairs_at_confluence: dict[int, set[HeadPair]] = defaultdict(set)

    for v in sorted_nodes:
        ps = [p for p in parents[v] if p in nodes]

        if not ps:
            head_sets[v] = frozenset({v}) if v in heads_set else frozenset()
        else:
            branch_sets = [head_sets[p] for p in ps]

            if v in confluences_set and len(branch_sets) >= 2:
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
                                pairs_at_confluence[v].add(normalize_pair(int(h), int(g)))

            merged: set[int] = set()
            for H in branch_sets:
                merged.update(H)
            head_sets[v] = frozenset(merged)

    return pairs_at_confluence, head_sets
