"""Strahler-order based pruning of TopoToolbox stream networks.

Two pruning operations live here, plus the joint ``apply_strategy`` that
combines them in the order used by the Mars-calibration regimes:

  1. ``prune_by_order_gap(s, order_gap)`` — remove tributary subtrees whose
     Strahler order is much lower than the channel they join (a small
     creek joining a big trunk is dropped together with everything
     upstream of it).
  2. ``apply_strategy(s, pre_remove_max_order, order_gap)`` — first strip
     all channels with Strahler order ``<= pre_remove_max_order`` (via
     ``dd_calibration.trim_to_min_order``), then run the order-gap pruner.

Both operations preserve the StreamObject ``s``'s topology API; they
return a new StreamObject built via ``s.subgraph(keep_mask)``. ``None``
is returned when the operation would leave no nodes.

These functions were lifted verbatim from
``notebooks/diagnostics/earth_network_pruning_experiments.ipynb`` so the
two Mars-calibration regimes (regA and regB) can call them from headless
scripts without depending on a notebook.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .dd_calibration import _node_rowcol, trim_to_min_order


def build_stream_graph(
    s: Any,
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int], list[list[int]]]:
    """Return ``(node_r, node_c, rc_to_id, in_edges)`` for a StreamObject.

    ``in_edges[i]`` is the list of upstream node IDs flowing into node ``i``
    (``source_indices → target_indices`` == upstream → downstream).
    """
    node_r, node_c = _node_rowcol(s)
    n = len(node_r)
    rc_to_id: dict[tuple[int, int], int] = {(int(node_r[i]), int(node_c[i])): i for i in range(n)}
    src_r, src_c = s.source_indices
    tgt_r, tgt_c = s.target_indices
    in_edges: list[list[int]] = [[] for _ in range(n)]
    for sr, sc, tr, tc in zip(
        np.asarray(src_r),
        np.asarray(src_c),
        np.asarray(tgt_r),
        np.asarray(tgt_c),
    ):
        sid = rc_to_id.get((int(sr), int(sc)))
        tid = rc_to_id.get((int(tr), int(tc)))
        if sid is not None and tid is not None:
            in_edges[tid].append(sid)
    return node_r, node_c, rc_to_id, in_edges


def prune_by_order_gap(s: Any, order_gap_to_prune: int) -> Any:
    """Prune tributary subtrees with a Strahler-order gap >= ``order_gap_to_prune``.

    At each junction, every upstream branch ``u`` whose Strahler order is at
    least ``order_gap_to_prune`` below the downstream node's order is marked
    as the root of a subtree to remove. The entire subtree upstream of each
    marked root is then dropped via ``s.subgraph``.

    Returns the original ``s`` when ``order_gap_to_prune <= 0`` or no junction
    qualifies. Returns ``None`` if the resulting subgraph is empty.
    """
    if s is None or order_gap_to_prune <= 0:
        return s

    strahler = np.asarray(s.streamorder(method="strahler"))
    node_r, _node_c, _rc_to_id, in_edges = build_stream_graph(s)
    n = len(node_r)

    pruned_roots: set[int] = set()
    for node_id in range(n):
        upstream = in_edges[node_id]
        if len(upstream) < 2:
            continue
        receiving_order = int(strahler[node_id])
        for up_id in upstream:
            if receiving_order - int(strahler[up_id]) >= order_gap_to_prune:
                pruned_roots.add(up_id)

    if not pruned_roots:
        return s

    nodes_to_remove: set[int] = set()
    for root in pruned_roots:
        stack = [root]
        while stack:
            curr = stack.pop()
            if curr in nodes_to_remove:
                continue
            nodes_to_remove.add(curr)
            for parent in in_edges[curr]:
                if parent not in nodes_to_remove:
                    stack.append(parent)

    keep = np.ones(n, dtype=bool)
    for nid in nodes_to_remove:
        keep[nid] = False
    return None if not keep.any() else s.subgraph(keep)


def apply_strategy(
    s: Any,
    pre_remove_max_order: int,
    order_gap_to_prune: int,
) -> Any:
    """Strip low-order channels then order-gap-prune the remainder.

    ``pre_remove_max_order=0`` keeps every channel; ``=1`` drops 1st-order
    only; ``=2`` drops 1st+2nd; etc. ``order_gap_to_prune`` is the
    receiving-vs-tributary Strahler gap threshold passed to
    :func:`prune_by_order_gap`.

    Returns ``None`` if the strip stage already removes every node.
    """
    if s is None:
        return None
    if pre_remove_max_order >= 1:
        s = trim_to_min_order(s, pre_remove_max_order + 1)
    return prune_by_order_gap(s, order_gap_to_prune)
