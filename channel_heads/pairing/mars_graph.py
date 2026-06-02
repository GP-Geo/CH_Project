"""Mars valley-network graph helpers for first-meet pairing.

The Mars pipeline reads a Phase-1 topology GeoPackage (directed
``upstream_node_id`` → ``downstream_node_id`` segments) and turns it into the
plain ``parents`` adjacency the graph-agnostic pairing core
(:func:`channel_heads.pairing.dag.first_meet_pairs_on_dag`) consumes, then
traces and chains segment geometries head→confluence.

These helpers were lifted verbatim from
``scripts/extract_mars_first_meet_pairs.py`` so both that batch script and the
``notebooks/mars/`` workflow call the same implementation instead of
duplicating it. They are pure graph/geometry functions — no file I/O — operating
on per-network frames whose ``node_id`` values are local (``0..n-1``).
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

_EPSILON = 1e-10
DEFAULT_HEAD_BUFFER_M = 5.0


def build_directed_adjacency(
    segments_df: pd.DataFrame, n_nodes: int
) -> tuple[list[list[int]], list[list[int]], dict[tuple[int, int], int]]:
    """Build parent/child adjacency lists and an (u,v)→segment_id lookup.

    Self-loop segments (``upstream_node_id == downstream_node_id``) are
    skipped — these are degenerate Phase-1 segments where flow direction
    is undefined.
    """
    parents: list[list[int]] = [[] for _ in range(n_nodes)]
    children: list[list[int]] = [[] for _ in range(n_nodes)]
    edge_to_seg: dict[tuple[int, int], int] = {}
    for _, row in segments_df.iterrows():
        u = int(row["upstream_node_id"])
        v = int(row["downstream_node_id"])
        if u == v or u < 0 or v < 0:
            continue
        parents[v].append(u)
        children[u].append(v)
        edge_to_seg[(u, v)] = int(row["segment_id"])
    return parents, children, edge_to_seg


def detect_crossed_segments(
    head_1_xy: tuple[float, float],
    head_2_xy: tuple[float, float],
    network_segments: gpd.GeoDataFrame,
    head_buffer_m: float = DEFAULT_HEAD_BUFFER_M,
) -> list[int]:
    """Return segment_ids this network's (buffered) head-to-head straight line
    intersects. Mirrors the Mars pair-feature stream-crossing filter exactly.

    The straight head-to-head line has small buffers around each head removed
    (so the head endpoints themselves don't count as crossings), then is tested
    against the network segments via an STR-tree.
    """
    p1 = Point(*head_1_xy)
    p2 = Point(*head_2_xy)
    straight = LineString([head_1_xy, head_2_xy])
    if straight.length < _EPSILON:
        return []
    interior = straight.difference(p1.buffer(head_buffer_m)).difference(p2.buffer(head_buffer_m))
    if interior.is_empty:
        return []
    geoms = list(network_segments.geometry)
    seg_ids = list(network_segments["segment_id"].astype(int))
    tree = STRtree(geoms)
    out: list[int] = []
    for i in tree.query(interior):
        if geoms[int(i)].intersects(interior):
            out.append(seg_ids[int(i)])
    return out


def trace_downstream_path(
    start: int, end: int, children: list[list[int]], max_steps: int
) -> list[int] | None:
    """Walk downstream from ``start`` until ``end`` is reached.

    Each node in the oriented Mars graph has at most one downstream
    child (the graph is a tree rooted at the outlet). Returns the list
    of node IDs ``[start, ..., end]`` or ``None`` if unreachable.
    """
    path = [start]
    cur = start
    for _ in range(max_steps):
        if cur == end:
            return path
        ch = children[cur]
        if not ch:
            return None
        cur = ch[0]
        path.append(cur)
    return None if cur != end else path


def chain_segment_geometries(
    node_path: list[int],
    edge_to_seg: dict[tuple[int, int], int],
    segment_geom_by_id: dict[int, LineString],
    start_node_by_seg: dict[int, int],
    end_node_by_seg: dict[int, int],
) -> tuple[LineString | None, list[int]]:
    """Chain segment LineStrings into one polyline along ``node_path``.

    Each segment LineString is reversed if its start coord corresponds to
    the downstream node of the edge being traversed (i.e. when the
    LineString orientation disagrees with the upstream→downstream walk).

    Returns ``(polyline, segment_ids)`` or ``(None, [])`` if any edge is
    missing.
    """
    if len(node_path) < 2:
        return None, []

    coords: list[tuple[float, float]] = []
    seg_ids: list[int] = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        seg_id = edge_to_seg.get((u, v))
        if seg_id is None:
            return None, []
        line = segment_geom_by_id[seg_id]
        seg_coords = list(line.coords)
        # Decide orientation: if segment's start_node_id == u, use as-is.
        if start_node_by_seg[seg_id] == u and end_node_by_seg[seg_id] == v:
            ordered = seg_coords
        elif start_node_by_seg[seg_id] == v and end_node_by_seg[seg_id] == u:
            ordered = list(reversed(seg_coords))
        else:
            # Fallback: pick orientation by coord proximity isn't necessary;
            # if neither matches the segment is malformed — skip pair.
            return None, []
        if coords:
            # Drop the duplicate shared endpoint between consecutive segments.
            ordered = ordered[1:]
        coords.extend(ordered)
        seg_ids.append(seg_id)
    if len(coords) < 2:
        return None, []
    return LineString(coords), seg_ids
