"""Earth/TopoToolbox path helpers for channel-head pair geometry.

This module is the canonical home for the low-level path-tracing and
coordinate helpers used by the Earth geometric feature analyzer
(:class:`channel_heads.geometric_analysis.GeometricFeaturesAnalyzer`) and by the
Earth patch rasterizer.

These helpers operate on TopoToolbox stream-network adjacency (parents /
children dicts) and node coordinates expressed in the Earth convention where
``x = column`` and ``y = -row`` (row indices increase downward in rasters, so
they are negated to make ``y`` increase northward for azimuth computation).

The implementation was extracted verbatim from
:mod:`channel_heads.geometric_analysis`; that module re-exports these names for
backward compatibility, so existing imports such as
``from channel_heads.geometric_analysis import _trace_full_path`` keep working.

References:
    Goren, L. and Shelef, E.: Channel concavity controls planform complexity
    of branching drainage networks, Earth Surf. Dynam., 12, 1347-1369,
    https://doi.org/10.5194/esurf-12-1347-2024, 2024.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
import numpy.typing as npt

# =============================================================================
# Type Aliases
# =============================================================================

NodeId = int
ParentsList = list[list[NodeId]]
ChildrenDict = dict[NodeId, list[NodeId]]
Coord2D = tuple[float, float]

# =============================================================================
# Constants
# =============================================================================

MIN_EDGES_FOR_DIRECTION = 3
EPSILON = 1e-10  # Small value for numerical stability


# =============================================================================
# Geometry Helpers
# =============================================================================


def _euclidean_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Compute 2D Euclidean distance between two points.

    Parameters
    ----------
    x1, y1 : float
        Coordinates of first point.
    x2, y2 : float
        Coordinates of second point.

    Returns
    -------
    float
        Euclidean distance.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def _normalize_vector(dx: float, dy: float) -> Coord2D | None:
    """Normalize a 2D vector to unit length.

    Parameters
    ----------
    dx, dy : float
        Vector components.

    Returns
    -------
    tuple[float, float] or None
        Unit vector, or None if input is zero.
    """
    mag = math.hypot(dx, dy)
    if mag < EPSILON:
        return None
    return (dx / mag, dy / mag)


# =============================================================================
# Graph Traversal Helpers
# =============================================================================


def _build_children_from_parents(parents: ParentsList, n_nodes: int) -> ChildrenDict:
    """Build children adjacency (downstream direction) from parents.

    Parameters
    ----------
    parents : ParentsList
        Adjacency list where parents[v] contains upstream neighbors of v.
    n_nodes : int
        Total number of nodes.

    Returns
    -------
    ChildrenDict
        children[v] = list of downstream neighbors.
    """
    children: ChildrenDict = defaultdict(list)
    for v in range(n_nodes):
        for p in parents[v]:
            children[p].append(v)
    return children


def _trace_path_downstream(
    start: NodeId,
    target: NodeId,
    children: ChildrenDict,
    node_x: npt.NDArray[np.float64],
    node_y: npt.NDArray[np.float64],
    max_distance_m: float,
    meters_per_unit: float,
) -> list[NodeId]:
    """Trace path from start node downstream toward target.

    Parameters
    ----------
    start : int
        Starting node ID (e.g., channel head).
    target : int
        Target node ID (e.g., confluence).
    children : ChildrenDict
        Children adjacency (downstream direction).
    node_x, node_y : np.ndarray
        Node coordinates in map units.
    max_distance_m : float
        Maximum distance to trace in meters.
    meters_per_unit : float
        Conversion factor from map units to meters.

    Returns
    -------
    list[int]
        List of node IDs from start toward target (including start).
    """
    path = [start]
    current = start
    accumulated_dist = 0.0

    # Set of visited nodes to avoid cycles
    visited = {start}

    while accumulated_dist < max_distance_m:
        # Get children of current node
        child_opts = [c for c in children.get(current, []) if c not in visited]

        if not child_opts:
            break

        # Pick child closest to target (or the only one)
        if len(child_opts) == 1:
            next_node = child_opts[0]
        else:
            # Pick child closest to target
            best_child = child_opts[0]
            best_dist = _euclidean_2d(
                node_x[best_child], node_y[best_child], node_x[target], node_y[target]
            )
            for c in child_opts[1:]:
                dist = _euclidean_2d(node_x[c], node_y[c], node_x[target], node_y[target])
                if dist < best_dist:
                    best_dist = dist
                    best_child = c
            next_node = best_child

        # Compute edge distance
        edge_dist = (
            _euclidean_2d(node_x[current], node_y[current], node_x[next_node], node_y[next_node])
            * meters_per_unit
        )

        accumulated_dist += edge_dist
        path.append(next_node)
        visited.add(next_node)
        current = next_node

        # Stop if we reached the target
        if current == target:
            break

    return path


def _compute_direction_vector(
    path: list[NodeId],
    node_x: npt.NDArray[np.float64],
    node_y: npt.NDArray[np.float64],
    meters_per_unit: float,
) -> tuple[Coord2D | None, str]:
    """Compute weighted average direction vector along a path.

    Direction points from first node toward last node in path.
    Edge vectors are weighted by their length.

    Parameters
    ----------
    path : list[int]
        Ordered list of node IDs.
    node_x, node_y : np.ndarray
        Node coordinates in map units.
    meters_per_unit : float
        Conversion factor for edge weighting.

    Returns
    -------
    tuple[Coord2D | None, str]
        (unit direction vector, qc_flags)
        Returns (None, flags) if path is too short.
    """
    qc_flags = []

    if len(path) < 2:
        qc_flags.append("single_edge")
        return None, ",".join(qc_flags)

    if len(path) < MIN_EDGES_FOR_DIRECTION + 1:
        qc_flags.append("short_path")

    # Accumulate weighted direction
    total_dx = 0.0
    total_dy = 0.0
    total_weight = 0.0

    for i in range(len(path) - 1):
        n1, n2 = path[i], path[i + 1]
        dx = node_x[n2] - node_x[n1]
        dy = node_y[n2] - node_y[n1]

        # Weight by edge length
        weight = math.sqrt(dx**2 + dy**2) * meters_per_unit

        if weight > EPSILON:
            total_dx += dx * weight
            total_dy += dy * weight
            total_weight += weight

    if total_weight < EPSILON:
        return None, ",".join(qc_flags)

    # Normalize
    avg_dx = total_dx / total_weight
    avg_dy = total_dy / total_weight

    unit_vec = _normalize_vector(avg_dx, avg_dy)
    return unit_vec, ",".join(qc_flags)


def _trace_full_path(
    start: NodeId,
    target: NodeId,
    children: ChildrenDict,
) -> list[NodeId]:
    """Trace the full downstream path from start to target.

    Unlike _trace_path_downstream, this has no distance limit and does not
    use a greedy heuristic — it simply follows the single downstream child
    at each node until the target is reached.

    Parameters
    ----------
    start : int
        Starting node ID (channel head).
    target : int
        Target node ID (confluence).
    children : ChildrenDict
        Downstream adjacency dict.

    Returns
    -------
    list[int]
        Ordered node sequence from start to target (inclusive).
        Returns an empty list if target is not reachable.
    """
    path = [start]
    current = start
    visited: set[NodeId] = {start}

    while current != target:
        opts = [c for c in children.get(current, []) if c not in visited]
        if not opts:
            return []  # target unreachable
        # At a channel head the path is always single-child until the confluence;
        # if there are somehow multiple, pick the one that is the target or first.
        next_node = target if target in opts else opts[0]
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path


def _sample_path_coords(
    path: list[NodeId],
    node_x: npt.NDArray[np.float64],
    node_y: npt.NDArray[np.float64],
    n_samples: int,
    meters_per_unit: float,
) -> npt.NDArray[np.float64] | None:
    """Sample n_samples evenly-spaced coordinate pairs along a path.

    Samples at arc-length fractions [0, 1/n_samples, 2/n_samples, …,
    (n_samples-1)/n_samples] (NOT including the endpoint at fraction 1.0,
    which is the confluence where both channels meet trivially at distance 0).

    Parameters
    ----------
    path : list[int]
        Ordered node sequence from head to confluence.
    node_x, node_y : np.ndarray
        Node coordinates in map units.
    n_samples : int
        Number of sample points.
    meters_per_unit : float
        Conversion factor (map units → metres), used only for arc-length
        computation; output coordinates are in map units × meters_per_unit.

    Returns
    -------
    np.ndarray of shape (n_samples, 2) or None
        Each row is (x_m, y_m) for one sampled point.
        Returns None if the path has fewer than 2 nodes.
    """
    if len(path) < 2:
        return None

    # Coordinates in metres
    xs = node_x[path] * meters_per_unit
    ys = node_y[path] * meters_per_unit

    # Cumulative arc-length along the path
    diffs = np.hypot(np.diff(xs), np.diff(ys))
    cum_len = np.concatenate([[0.0], np.cumsum(diffs)])
    total_len = cum_len[-1]

    if total_len < EPSILON:
        return None

    norm_len = cum_len / total_len  # normalised to [0, 1]

    # Target fractions: 0, 1/n, 2/n, …, (n-1)/n  (exclude the endpoint at 1.0)
    targets = np.arange(n_samples) / n_samples

    # Vectorized interpolation: find segment indices for all targets at once
    idxs = np.searchsorted(norm_len, targets, side="right") - 1
    idxs = np.clip(idxs, 0, len(path) - 2)

    lo = norm_len[idxs]
    hi = norm_len[idxs + 1]
    seg_len = hi - lo
    frac = np.where(seg_len > EPSILON, (targets - lo) / seg_len, 0.0)

    coords = np.column_stack(
        [
            xs[idxs] + frac * (xs[idxs + 1] - xs[idxs]),
            ys[idxs] + frac * (ys[idxs + 1] - ys[idxs]),
        ]
    )

    return coords


# =============================================================================
# Cellsize Detection (shared by both analyzers)
# =============================================================================


def _detect_cellsize(s: Any, dem: Any | None) -> float | None:
    """Detect cell size from StreamObject or DEM.

    Parameters
    ----------
    s : StreamObject
        TopoToolbox StreamObject.
    dem : GridObject or None
        Digital elevation model.

    Returns
    -------
    float or None
        Detected cell size, or None if not found.
    """
    cellsize = getattr(s, "cellsize", None)

    if cellsize is None and dem is not None:
        cellsize = getattr(dem, "cellsize", None)

        if cellsize is None:
            res = getattr(dem, "res", None)
            if res is not None:
                cellsize = abs(res[0]) if isinstance(res, (tuple, list)) else abs(res)

        if cellsize is None:
            transform = getattr(dem, "transform", None)
            if transform is not None:
                if hasattr(transform, "a"):
                    cellsize = abs(transform.a)
                elif isinstance(transform, (tuple, list)) and len(transform) >= 1:
                    cellsize = abs(transform[0])

    return cellsize


__all__ = [
    "NodeId",
    "ParentsList",
    "ChildrenDict",
    "Coord2D",
    "MIN_EDGES_FOR_DIRECTION",
    "EPSILON",
    "_euclidean_2d",
    "_normalize_vector",
    "_build_children_from_parents",
    "_trace_path_downstream",
    "_compute_direction_vector",
    "_trace_full_path",
    "_sample_path_coords",
    "_detect_cellsize",
]
