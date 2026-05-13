"""Geometric analysis for paired channel heads.

This module provides all geometric feature computation for channel head pairs
that share a first common downstream confluence:

Features:
    1. Lengthwise Asymmetry (delta_L): Normalized path length difference
       (Equation 4 from Goren & Shelef 2024)
    2. Orientation Similarity: Difference in initial downstream azimuths
    3. Euclidean Head-Head Distance: Planar distance, raw and normalized
    4. Strahler Order Difference: Difference in branch stream orders

Dataset Scope:
    Features are computed ONLY for pairs from `first_meet_pairs_for_outlet()`.
    - Positive pairs (y=1): `touching=True` from CouplingAnalyzer
    - Negative pairs (y=0): `touching=False` at the same confluence (hard negatives)

References:
    Goren, L. and Shelef, E.: Channel concavity controls planform complexity
    of branching drainage networks, Earth Surf. Dynam., 12, 1347-1369,
    https://doi.org/10.5194/esurf-12-1347-2024, 2024.
"""

from __future__ import annotations

import argparse
import logging
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.draw import line as skimage_line

from .config import resolve_dem_path
from .first_meet_pairs_for_outlet import _build_parents_from_stream, _normalize_pair
from .logging_config import get_logger

logger = get_logger(__name__)

# =============================================================================
# Type Aliases
# =============================================================================

NodeId = int
HeadId = int
HeadPair = tuple[HeadId, HeadId]
ParentsList = list[list[NodeId]]
ChildrenDict = dict[NodeId, list[NodeId]]
Coord2D = tuple[float, float]

# =============================================================================
# Constants
# =============================================================================

# Coordinate conversion
METERS_PER_DEGREE_LAT = 110540.0  # meters per degree of latitude (approximately constant)
METERS_PER_DEGREE_LON_EQUATOR = 111320.0  # meters per degree of longitude at equator

# Geometric feature computation
DEFAULT_DIRECTION_SAMPLE_DISTANCE_M = 500.0
MIN_EDGES_FOR_DIRECTION = 3
EPSILON = 1e-10  # Small value for numerical stability

# Geometric feature columns added by GeometricFeaturesAnalyzer and add_geometric_features_to_csv
GEOM_FEATURE_COLS: list[str] = [
    "orientation_diff_deg",
    "headhead_dist_m",
    "headhead_dist_norm",
    "apex_angle_deg",
    "strahler_order_diff",
    "proximity_mean_m",
    "proximity_max_m",
    "proximity_profile_norm",
    "qc_flags",
]

# Type alias for the stream loader used by add_geometric_features_to_csv
StreamLoaderFunc = Callable[[str, float, float], tuple[Any, Any] | None]


# =============================================================================
# Coordinate Conversion Helpers
# =============================================================================


def compute_meters_per_degree(lat_deg: float) -> float:
    """Compute meters per degree at a given latitude.

    For DEMs in geographic coordinates (lat/lon), this converts distances
    from degrees to meters.

    Parameters
    ----------
    lat_deg : float
        Latitude in degrees (positive for northern hemisphere).

    Returns
    -------
    float
        Approximate meters per degree (geometric mean of lat/lon directions).

    Notes
    -----
    The conversion uses:
    - 1 degree of latitude ~ 110,540 meters (approximately constant)
    - 1 degree of longitude ~ 111,320 * cos(latitude) meters

    We use the geometric mean for flow paths that can go in any direction.

    Examples
    --------
    >>> # At 36.7 latitude (Inyo Mountains)
    >>> m_per_deg = compute_meters_per_degree(36.7)
    >>> print(f"Meters per degree: {m_per_deg:.0f}")
    Meters per degree: 99287
    """
    lat_rad = math.radians(abs(lat_deg))

    # Meters per degree in each direction
    meters_per_deg_lon = METERS_PER_DEGREE_LON_EQUATOR * math.cos(lat_rad)
    meters_per_deg_lat = METERS_PER_DEGREE_LAT

    # Use geometric mean for flow paths in arbitrary directions
    return math.sqrt(meters_per_deg_lon * meters_per_deg_lat)


def compute_pixel_size_meters(lat_deg: float, cellsize_deg: float) -> float:
    """Compute approximate pixel size in meters for a geographic DEM.

    For DEMs in geographic coordinates (lat/lon), pixel size varies with latitude.
    This function computes the average linear pixel size in meters.

    Parameters
    ----------
    lat_deg : float
        Latitude in degrees (positive for northern hemisphere).
    cellsize_deg : float
        Cell size in degrees (e.g., 1/3600 for 1 arc-second SRTM).

    Returns
    -------
    float
        Approximate pixel size in meters.

    Examples
    --------
    >>> # 1 arc-second SRTM at 36.7 latitude (Inyo Mountains)
    >>> pixel_size = compute_pixel_size_meters(36.7, 1/3600)
    >>> print(f"Pixel size: {pixel_size:.1f} m")
    Pixel size: 27.6 m
    """
    return cellsize_deg * compute_meters_per_degree(lat_deg)


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


def _angle_between_vectors(v1: Coord2D, v2: Coord2D) -> float:
    """Compute angle between two 2D vectors in degrees.

    Parameters
    ----------
    v1 : tuple[float, float]
        First vector (dx, dy).
    v2 : tuple[float, float]
        Second vector (dx, dy).

    Returns
    -------
    float
        Angle in degrees [0, 180]. Returns NaN if either vector is zero.
    """
    dx1, dy1 = v1
    dx2, dy2 = v2

    mag1 = math.hypot(dx1, dy1)
    mag2 = math.hypot(dx2, dy2)

    if mag1 < EPSILON or mag2 < EPSILON:
        return float("nan")

    # Dot product
    dot = dx1 * dx2 + dy1 * dy2

    # Clamp to handle numerical errors
    cos_angle = max(-1.0, min(1.0, dot / (mag1 * mag2)))

    return math.degrees(math.acos(cos_angle))


def _compute_azimuth(dx: float, dy: float) -> float:
    """Compute azimuth from north, clockwise, in degrees.

    Parameters
    ----------
    dx : float
        Change in x (east-west direction).
    dy : float
        Change in y (north-south direction).

    Returns
    -------
    float
        Azimuth in degrees [0, 360). Returns NaN if vector is zero.
    """
    if abs(dx) < EPSILON and abs(dy) < EPSILON:
        return float("nan")

    # atan2(dx, dy) gives angle from north (y-axis), clockwise
    azimuth = math.degrees(math.atan2(dx, dy))

    # Wrap to [0, 360)
    if azimuth < 0:
        azimuth += 360.0

    return azimuth


def _azimuth_difference(az1: float, az2: float) -> float:
    """Compute absolute azimuth difference wrapped to [0, 180].

    Parameters
    ----------
    az1 : float
        First azimuth in degrees.
    az2 : float
        Second azimuth in degrees.

    Returns
    -------
    float
        Absolute difference in degrees [0, 180].
    """
    if math.isnan(az1) or math.isnan(az2):
        return float("nan")

    diff = abs(az1 - az2)

    # Wrap to [0, 180]
    if diff > 180:
        diff = 360 - diff

    return diff


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


# =============================================================================
# Proximity Profile Helpers
# =============================================================================


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

    coords = np.column_stack([
        xs[idxs] + frac * (xs[idxs + 1] - xs[idxs]),
        ys[idxs] + frac * (ys[idxs + 1] - ys[idxs]),
    ])

    return coords


def _compute_proximity_profile(
    coords_1: npt.NDArray[np.float64],
    coords_2: npt.NDArray[np.float64],
) -> tuple[float, float, float]:
    """Compute the proximity profile statistics between two sampled channel paths.

    Parameters
    ----------
    coords_1, coords_2 : np.ndarray of shape (n, 2)
        Sampled (x_m, y_m) coordinates along each channel path.

    Returns
    -------
    tuple (proximity_mean_m, proximity_max_m, proximity_profile_norm)
        proximity_mean_m : mean pairwise distance (metres)
        proximity_max_m  : max pairwise distance (metres)
        proximity_profile_norm : mean / max ∈ [0, 1]; 1.0 = parallel channels,
            <1.0 = convergent channels.
    """
    dists = np.hypot(coords_1[:, 0] - coords_2[:, 0], coords_1[:, 1] - coords_2[:, 1])
    mean_m = float(np.mean(dists))
    max_m = float(np.max(dists))
    norm = mean_m / max_m if max_m > EPSILON else float("nan")
    return mean_m, max_m, norm


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


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(slots=True)
class PairAsymmetryResult:
    """Result of lengthwise asymmetry computation for a channel head pair.

    Attributes
    ----------
    head_1 : int
        Node ID of first channel head.
    head_2 : int
        Node ID of second channel head.
    confluence : int
        Node ID of the confluence where the heads meet.
    L_1 : float
        Flow distance from head_1 to confluence (meters).
    L_2 : float
        Flow distance from head_2 to confluence (meters).
    delta_L : float
        Normalized lengthwise asymmetry: 2|L_1 - L_2| / (L_1 + L_2).
    distance_unit : str
        Unit of distance measurement ('meters').
    """

    head_1: int
    head_2: int
    confluence: int
    L_1: float
    L_2: float
    delta_L: float
    distance_unit: str = "meters"


@dataclass(slots=True)
class PairGeometricResult:
    """Geometric features for a channel head pair.

    Attributes
    ----------
    head_1 : int
        Node ID of first channel head (min ID).
    head_2 : int
        Node ID of second channel head (max ID).
    confluence : int
        Node ID of the confluence where heads meet.
    orientation_diff_deg : float
        Absolute difference in initial downstream azimuths [0, 180] degrees.
    headhead_dist_m : float
        Planar distance between channel heads (meters).
    headhead_dist_norm : float
        Head-head distance normalized by (L_1 + L_2), dimensionless.
    apex_angle_deg : float
        Angle at confluence between straight-line vectors to each head [0, 180].
        Uses planar positions only.
    strahler_order_diff : float
        |strahler_order(branch_1) - strahler_order(branch_2)| at the confluence.
        NaN when node_orders are not provided or branch parents are not found.
    proximity_mean_m : float or None
        Mean pairwise distance (metres) between n_proximity_samples equally-spaced
        points sampled along each channel path toward the confluence.
    proximity_max_m : float or None
        Maximum pairwise distance (metres) across the sampled points.
    proximity_profile_norm : float or None
        proximity_mean_m / proximity_max_m ∈ [0, 1].
        ≈ 1.0 for parallel channels; < 1.0 for strongly convergent channels.
        NaN when proximity_max_m ≈ 0.
    qc_flags : str
        Comma-separated quality control flags, or empty string.
    """

    head_1: int
    head_2: int
    confluence: int
    orientation_diff_deg: float
    headhead_dist_m: float
    headhead_dist_norm: float
    apex_angle_deg: float
    strahler_order_diff: float
    proximity_mean_m: float | None
    proximity_max_m: float | None
    proximity_profile_norm: float | None
    qc_flags: str


# =============================================================================
# Core Functions
# =============================================================================


def compute_delta_L(L_ij: float, L_ji: float) -> float:
    """Compute normalized lengthwise asymmetry from two path lengths.

    This is the core formula from Equation 4 of Goren & Shelef (2024):
        delta_L = 2|L_ij - L_ji| / (L_ij + L_ji)

    Parameters
    ----------
    L_ij : float
        Along-flow distance from channel head i to common junction.
    L_ji : float
        Along-flow distance from channel head j to common junction.

    Returns
    -------
    float
        Normalized lengthwise asymmetry, ranging from 0 (symmetric)
        to 2 (maximum asymmetry). Returns 0.0 if both lengths are zero.

    Examples
    --------
    >>> compute_delta_L(1000, 1000)  # Symmetric
    0.0
    >>> compute_delta_L(1000, 2000)  # Asymmetric
    0.6666666666666666
    >>> compute_delta_L(0, 1000)  # Maximum asymmetry
    2.0
    """
    total = L_ij + L_ji
    if total == 0:
        return 0.0
    return 2.0 * abs(L_ij - L_ji) / total


# =============================================================================
# Lengthwise Asymmetry Analyzer
# =============================================================================


class LengthwiseAsymmetryAnalyzer:
    """Compute lengthwise asymmetry for channel head pairs using TopoToolbox.

    This class uses TopoToolbox's built-in upstream_distance() function to
    compute flow path distances efficiently. The upstream_distance() returns
    the cumulative distance from each stream node to the outlet.

    For a channel head and confluence on the same flow path:
        L = upstream_dist[head] - upstream_dist[confluence]

    Parameters
    ----------
    s : StreamObject
        TopoToolbox StreamObject with stream network topology.
    dem : GridObject, optional
        Digital elevation model. Used to get cell size for distance conversion.
    lat : float, optional
        Latitude of the study area in degrees. Required for converting
        geographic coordinates to meters. If not provided, distances are
        returned in map units (which may be degrees for geographic DEMs).

    Attributes
    ----------
    s : StreamObject
        The stream network object.
    _upstream_dist : np.ndarray
        Precomputed upstream distances (node attribute list).
    _meters_per_unit : float
        Conversion factor from map units to meters.

    Example
    -------
    >>> import topotoolbox as tt3
    >>> dem = tt3.read_tif("dem.tif")
    >>> fd = tt3.FlowObject(dem)
    >>> s = tt3.StreamObject(fd, threshold=300)
    >>> # For Inyo Mountains at latitude 36.71
    >>> analyzer = LengthwiseAsymmetryAnalyzer(s, dem, lat=36.71)
    >>> result = analyzer.compute_pair_asymmetry(head_1=100, head_2=150, confluence=200)
    """

    def __init__(
        self,
        s: Any,  # StreamObject
        dem: Any | None = None,  # GridObject
        lat: float | None = None,  # Latitude in degrees
    ) -> None:
        self.s = s
        self.dem = dem
        self.lat = lat

        # Precompute upstream distances using TopoToolbox's built-in function
        # upstream_distance() returns cumulative distance from each node to the outlet
        self._upstream_dist: np.ndarray = s.upstream_distance()

        # Compute meters per unit conversion factor
        self._meters_per_unit: float = 1.0  # Default: assume already in meters
        self._detected_cellsize: float | None = None

        if lat is not None:
            cellsize = _detect_cellsize(s, dem)
            self._detected_cellsize = cellsize

            if cellsize is not None and cellsize < 1.0:
                # Cell size is in degrees (geographic coordinate system)
                # upstream_distance() returns distances in map units (degrees)
                # Convert degrees to meters
                self._meters_per_unit = compute_meters_per_degree(lat)
            elif cellsize is None:
                # Fallback: assume geographic CRS if lat is provided but cellsize unknown
                self._meters_per_unit = compute_meters_per_degree(lat)
            # else: cellsize >= 1.0 means it's likely already in meters (projected CRS)

    @property
    def meters_per_unit(self) -> float:
        """Conversion factor from map units to meters."""
        return self._meters_per_unit

    @property
    def detected_cellsize(self) -> float | None:
        """Detected cell size from DEM (for debugging)."""
        return self._detected_cellsize

    def compute_pair_asymmetry(
        self,
        head_1: int,
        head_2: int,
        confluence: int,
        use_meters: bool = True,  # deprecated: ignored, distances always in meters
    ) -> PairAsymmetryResult:
        """Compute lengthwise asymmetry for a pair of channel heads.

        Uses TopoToolbox's upstream_distance to compute flow path lengths.
        The distance from a head to a confluence is:
            L = upstream_dist[head] - upstream_dist[confluence]

        Parameters
        ----------
        head_1 : int
            Node ID of first channel head.
        head_2 : int
            Node ID of second channel head.
        confluence : int
            Node ID of the confluence where the heads meet.
        use_meters : bool
            Kept for API compatibility. TopoToolbox always uses map units.

        Returns
        -------
        PairAsymmetryResult
            Result object with path lengths and asymmetry value.

        Raises
        ------
        ValueError
            If node indices are out of bounds.
        """
        h1 = int(head_1)
        h2 = int(head_2)
        conf = int(confluence)

        # Validate node indices
        n_nodes = len(self._upstream_dist)
        if h1 >= n_nodes or h2 >= n_nodes or conf >= n_nodes:
            raise ValueError(
                f"Node index out of bounds. Network has {n_nodes} nodes, "
                f"but got head_1={h1}, head_2={h2}, confluence={conf}"
            )

        # Compute flow path lengths using upstream distance.
        # upstream_dist[node] = cumulative distance from each node toward the outlet.
        # For a head upstream of a confluence: L = upstream_dist[head] - upstream_dist[conf].
        # Negative values indicate a topology inconsistency (head downstream of confluence),
        # which should not occur in well-formed networks. We clamp to 0 and warn.
        L_1_raw = float(self._upstream_dist[h1] - self._upstream_dist[conf])
        L_2_raw = float(self._upstream_dist[h2] - self._upstream_dist[conf])
        if L_1_raw < 0 or L_2_raw < 0:
            logger.warning(
                "Negative path length for pair (%d, %d) at confluence %d "
                "(L_1=%.4g, L_2=%.4g). Head may be downstream of confluence. Clamping to 0.",
                h1, h2, conf, L_1_raw, L_2_raw,
            )
        L_1_raw = max(0.0, L_1_raw)
        L_2_raw = max(0.0, L_2_raw)

        # Convert to meters using the precomputed conversion factor
        L_1 = L_1_raw * self._meters_per_unit
        L_2 = L_2_raw * self._meters_per_unit

        # Compute asymmetry
        delta_L = compute_delta_L(L_1, L_2)

        return PairAsymmetryResult(
            head_1=h1,
            head_2=h2,
            confluence=conf,
            L_1=L_1,
            L_2=L_2,
            delta_L=delta_L,
            distance_unit="meters",
        )

    def evaluate_pairs_for_outlet(
        self,
        outlet: int,
        pairs_at_confluence: dict[int, set[HeadPair]],
        use_meters: bool = True,  # deprecated: ignored
    ) -> pd.DataFrame:
        """Compute lengthwise asymmetry for all pairs in an outlet's basin.

        Parameters
        ----------
        outlet : int
            Node ID of the outlet.
        pairs_at_confluence : Dict[int, Set[Tuple[int, int]]]
            Dictionary mapping confluence node IDs to sets of head pairs.
            This is the output from first_meet_pairs_for_outlet().
        use_meters : bool
            Deprecated, ignored. Distances are always returned in meters.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            - outlet: Outlet node ID
            - confluence: Confluence node ID
            - head_1, head_2: Channel head node IDs
            - L_1, L_2: Flow distances to confluence (meters)
            - delta_L: Lengthwise asymmetry value
            - distance_unit: 'meters'
        """
        rows = []
        out = int(outlet)
        n_skipped = 0

        for conf, pairs in pairs_at_confluence.items():
            if not pairs:
                continue

            for h1, h2 in pairs:
                try:
                    result = self.compute_pair_asymmetry(h1, h2, int(conf), use_meters)
                    # Normalize pair order to match coupling_analysis.py (min, max)
                    # This ensures the merge works correctly
                    head_min, head_max = min(h1, h2), max(h1, h2)
                    # Swap L values if we swapped heads
                    if h1 == head_min:
                        L_1_out, L_2_out = result.L_1, result.L_2
                    else:
                        L_1_out, L_2_out = result.L_2, result.L_1
                    rows.append(
                        {
                            "outlet": out,
                            "confluence": int(conf),
                            "head_1": head_min,
                            "head_2": head_max,
                            "L_1": L_1_out,
                            "L_2": L_2_out,
                            "delta_L": result.delta_L,
                            "distance_unit": result.distance_unit,
                        }
                    )
                except ValueError:
                    n_skipped += 1
                    continue

        if n_skipped > 0:
            logger.warning(
                "Outlet %d: skipped %d asymmetry pairs due to computation errors",
                out,
                n_skipped,
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "outlet",
                "confluence",
                "head_1",
                "head_2",
                "L_1",
                "L_2",
                "delta_L",
                "distance_unit",
            ],
        )

        if not df.empty:
            df.sort_values(["confluence", "head_1", "head_2"], inplace=True, ignore_index=True)

        return df

    def clear_cache(self) -> None:
        """Clear cache (no-op, kept for API compatibility)."""
        pass  # upstream_distance is computed once in __init__


# =============================================================================
# Stream-Crossing Helpers
# =============================================================================


def _line_crosses_stream(
    r1: int,
    c1: int,
    r2: int,
    c2: int,
    stream_mask: npt.NDArray[np.bool_],
) -> bool:
    """Return True if the rasterized line from (r1,c1) to (r2,c2) passes
    through any stream pixel, excluding the two endpoint pixels themselves.

    Uses Bresenham's line algorithm (skimage.draw.line) to enumerate
    intermediate pixels.
    """
    rr, cc = skimage_line(r1, c1, r2, c2)
    # Clip to valid array bounds
    valid = (rr >= 0) & (rr < stream_mask.shape[0]) & (cc >= 0) & (cc < stream_mask.shape[1])
    rr, cc = rr[valid], cc[valid]
    if len(rr) <= 2:
        # Only endpoints, no intermediate pixels
        return False
    # Exclude first and last pixel (the channel heads themselves)
    interior_rr, interior_cc = rr[1:-1], cc[1:-1]
    return bool(stream_mask[interior_rr, interior_cc].any())


def _build_stream_mask(
    s: Any,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray, npt.NDArray]:
    """Build a binary stream mask and return (mask, r_nodes, c_nodes).

    mask shape is inferred from s.shape if available, else from max node indices.
    """
    node_indices = (
        s.node_indices() if callable(getattr(s, "node_indices", None)) else s.node_indices
    )
    r_nodes, c_nodes = np.asarray(node_indices[0]), np.asarray(node_indices[1])

    shape = getattr(s, "shape", None)
    if shape is None:
        shape = (int(r_nodes.max()) + 1, int(c_nodes.max()) + 1)

    mask = np.zeros(shape, dtype=bool)
    mask[r_nodes, c_nodes] = True
    return mask, r_nodes, c_nodes


# =============================================================================
# Geometric Features Analyzer
# =============================================================================


class GeometricFeaturesAnalyzer:
    """Compute geometric features for channel head pairs.

    This analyzer computes features (orientation similarity, Euclidean distance,
    Strahler order difference) for pairs of channel heads that share a first
    common downstream confluence.

    Parameters
    ----------
    s : StreamObject
        TopoToolbox StreamObject with stream network topology.
    dem : GridObject, optional
        Digital elevation model. Used to get cell size for coordinate conversion.
    lat : float, optional
        Latitude of the study area in degrees. Required for converting
        geographic coordinates to meters.
    direction_sample_distance_m : float, optional
        Distance along path for direction estimation (default: 500 meters).
    node_orders : np.ndarray, optional
        Strahler order for each node. If provided, strahler_order_diff is computed.

    Example
    -------
    >>> geom_an = GeometricFeaturesAnalyzer(s, dem, lat=36.71)
    >>> result = geom_an.compute_pair_geometry(head_1, head_2, confluence, L_1, L_2)
    >>> print(f"Orientation diff: {result.orientation_diff_deg:.1f}")
    """

    def __init__(
        self,
        s: Any,  # StreamObject
        dem: Any | None = None,  # GridObject
        lat: float | None = None,
        direction_sample_distance_m: float = DEFAULT_DIRECTION_SAMPLE_DISTANCE_M,
        node_orders: npt.NDArray[np.float32] | None = None,
        n_proximity_samples: int = 10,
    ) -> None:
        self.s = s
        self.dem = dem
        self.lat = lat
        self.direction_sample_distance_m = direction_sample_distance_m
        self._node_orders = node_orders
        self.n_proximity_samples = n_proximity_samples

        # Build adjacency lists
        self._parents: ParentsList = _build_parents_from_stream(s)
        self._n_nodes = len(self._parents)
        self._children: ChildrenDict = _build_children_from_parents(self._parents, self._n_nodes)

        # Extract node coordinates
        node_indices = (
            s.node_indices() if callable(getattr(s, "node_indices", None)) else s.node_indices
        )
        self._r_nodes, self._c_nodes = node_indices

        # Convert row/col to x/y coordinates
        # Column = x (east), negated row = y (north)
        # Row indices increase downward in rasters, so negate to get
        # y increasing northward for correct azimuth computation
        self._node_x = np.asarray(self._c_nodes, dtype=np.float64)
        self._node_y = -np.asarray(self._r_nodes, dtype=np.float64)

        # Compute meters per unit conversion
        self._meters_per_unit: float = 1.0
        self._detected_cellsize: float | None = None

        if lat is not None:
            cellsize = _detect_cellsize(s, dem)
            self._detected_cellsize = cellsize

            if cellsize is not None and cellsize < 1.0:
                # Geographic CRS: node coords are in pixels, convert to meters
                self._meters_per_unit = compute_pixel_size_meters(lat, cellsize)
            elif cellsize is not None:
                # Projected CRS: node coords are pixel indices,
                # multiply by cellsize to get meters
                self._meters_per_unit = cellsize
            # else: cellsize is None, leave as 1.0 (pixel units)

    @property
    def meters_per_unit(self) -> float:
        """Conversion factor from map units to meters."""
        return self._meters_per_unit

    def _find_branch_parent(self, confluence: NodeId, target_head: NodeId) -> NodeId | None:
        """Find which parent of confluence is on the path to target_head.

        Parameters
        ----------
        confluence : int
            Confluence node ID.
        target_head : int
            Target channel head node ID.

        Returns
        -------
        int or None
            Parent node ID on path to target_head, or None if not found.
        """
        parent_list = self._parents[confluence]

        if len(parent_list) == 0:
            return None

        if len(parent_list) == 1:
            return parent_list[0]

        # Check which parent eventually leads to target_head
        # Do iterative DFS from each parent to find target
        for p in parent_list:
            if self._can_reach(p, target_head):
                return p

        return None

    def _can_reach(self, start: NodeId, target: NodeId) -> bool:
        """Check if target is reachable from start via upstream traversal (iterative)."""
        stack = [start]
        seen: set[NodeId] = {start}
        while stack:
            node = stack.pop()
            if node == target:
                return True
            for p in self._parents[node]:
                if p not in seen:
                    seen.add(p)
                    stack.append(p)
        return False

    def compute_pair_geometry(
        self,
        head_1: int,
        head_2: int,
        confluence: int,
        L_1: float | None = None,
        L_2: float | None = None,
    ) -> PairGeometricResult:
        """Compute all geometric features for a single pair.

        Parameters
        ----------
        head_1 : int
            Node ID of first channel head.
        head_2 : int
            Node ID of second channel head.
        confluence : int
            Node ID of the confluence where heads meet.
        L_1 : float, optional
            Flow distance from head_1 to confluence (meters).
            Required for normalized distance.
        L_2 : float, optional
            Flow distance from head_2 to confluence (meters).
            Required for normalized distance.

        Returns
        -------
        PairGeometricResult
            Result object with all computed features.
        """
        h1, h2 = int(head_1), int(head_2)
        conf = int(confluence)

        # Normalize pair order (min, max)
        h1_norm, h2_norm = _normalize_pair(h1, h2)
        if h1 != h1_norm:
            # Swap L values to match normalized order
            L_1, L_2 = L_2, L_1
            h1, h2 = h1_norm, h2_norm

        qc_flags: list[str] = []

        # Get node coordinates (in map units * meters_per_unit = meters)
        x1 = self._node_x[h1] * self._meters_per_unit
        y1 = self._node_y[h1] * self._meters_per_unit
        x2 = self._node_x[h2] * self._meters_per_unit
        y2 = self._node_y[h2] * self._meters_per_unit
        xc = self._node_x[conf] * self._meters_per_unit
        yc = self._node_y[conf] * self._meters_per_unit

        # -------------------------
        # Apex angle: angle AT the confluence between straight-line vectors to heads
        # -------------------------
        apex_angle_deg = _angle_between_vectors((x1 - xc, y1 - yc), (x2 - xc, y2 - yc))

        # -------------------------
        # Feature 2: Confluence Angle
        # -------------------------
        # Find parent branches for each head (needed for Strahler order)
        parent_1 = self._find_branch_parent(conf, h1)
        parent_2 = self._find_branch_parent(conf, h2)

        # -------------------------
        # Strahler order difference: |order(branch_1) - order(branch_2)|
        # -------------------------
        if self._node_orders is not None and parent_1 is not None and parent_2 is not None:
            strahler_order_diff = abs(
                float(self._node_orders[parent_1]) - float(self._node_orders[parent_2])
            )
        else:
            strahler_order_diff = float("nan")

        # -------------------------
        # Feature 3: Orientation Similarity
        # -------------------------
        # Trace downstream from each head
        path_down_1 = _trace_path_downstream(
            h1,
            conf,
            self._children,
            self._node_x,
            self._node_y,
            self.direction_sample_distance_m,
            self._meters_per_unit,
        )
        path_down_2 = _trace_path_downstream(
            h2,
            conf,
            self._children,
            self._node_x,
            self._node_y,
            self.direction_sample_distance_m,
            self._meters_per_unit,
        )

        vec_down_1, flags_down_1 = _compute_direction_vector(
            path_down_1, self._node_x, self._node_y, self._meters_per_unit
        )
        vec_down_2, flags_down_2 = _compute_direction_vector(
            path_down_2, self._node_x, self._node_y, self._meters_per_unit
        )

        if flags_down_1:
            qc_flags.append(f"orient1:{flags_down_1}")
        if flags_down_2:
            qc_flags.append(f"orient2:{flags_down_2}")

        if vec_down_1 is not None and vec_down_2 is not None:
            az_1 = _compute_azimuth(vec_down_1[0], vec_down_1[1])
            az_2 = _compute_azimuth(vec_down_2[0], vec_down_2[1])
            orientation_diff_deg = _azimuth_difference(az_1, az_2)
        else:
            orientation_diff_deg = float("nan")

        # -------------------------
        # Feature 4: Euclidean Distance
        # -------------------------
        headhead_dist_m = _euclidean_2d(x1, y1, x2, y2)

        if L_1 is not None and L_2 is not None:
            L_sum = L_1 + L_2
            if L_sum > EPSILON:
                headhead_dist_norm = headhead_dist_m / L_sum
            else:
                headhead_dist_norm = float("nan")
                qc_flags.append("zero_path_length")
        else:
            headhead_dist_norm = float("nan")

        # Check for coincident heads
        if headhead_dist_m < EPSILON:
            qc_flags.append("coincident_nodes")

        # -------------------------
        # Proximity Profile
        # -------------------------
        proximity_mean_m: float | None = None
        proximity_max_m: float | None = None
        proximity_profile_norm: float | None = None

        full_path_1 = _trace_full_path(h1, conf, self._children)
        full_path_2 = _trace_full_path(h2, conf, self._children)

        if full_path_1 and full_path_2:
            coords_1 = _sample_path_coords(
                full_path_1,
                self._node_x,
                self._node_y,
                self.n_proximity_samples,
                self._meters_per_unit,
            )
            coords_2 = _sample_path_coords(
                full_path_2,
                self._node_x,
                self._node_y,
                self.n_proximity_samples,
                self._meters_per_unit,
            )
            if coords_1 is not None and coords_2 is not None:
                proximity_mean_m, proximity_max_m, proximity_profile_norm = (
                    _compute_proximity_profile(coords_1, coords_2)
                )
            else:
                qc_flags.append("proximity_path_error")
        else:
            qc_flags.append("proximity_path_error")

        return PairGeometricResult(
            head_1=h1,
            head_2=h2,
            confluence=conf,
            orientation_diff_deg=orientation_diff_deg,
            headhead_dist_m=headhead_dist_m,
            headhead_dist_norm=headhead_dist_norm,
            apex_angle_deg=apex_angle_deg,
            strahler_order_diff=strahler_order_diff,
            proximity_mean_m=proximity_mean_m,
            proximity_max_m=proximity_max_m,
            proximity_profile_norm=proximity_profile_norm,
            qc_flags=",".join(qc_flags),
        )

    def evaluate_pairs_for_outlet(
        self,
        outlet: int,
        pairs_at_confluence: dict[int, set[HeadPair]],
        asymmetry_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute geometric features for all pairs in an outlet's basin.

        Parameters
        ----------
        outlet : int
            Node ID of the outlet.
        pairs_at_confluence : dict[int, set[tuple[int, int]]]
            Dictionary mapping confluence node IDs to sets of head pairs.
            This is the output from first_meet_pairs_for_outlet().
        asymmetry_df : pd.DataFrame, optional
            DataFrame with L_1, L_2 values from LengthwiseAsymmetryAnalyzer.
            If provided, path lengths are looked up instead of being None.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: outlet, confluence, head_1, head_2,
            orientation_diff_deg, headhead_dist_m, headhead_dist_norm,
            apex_angle_deg, strahler_order_diff, qc_flags
        """
        rows = []
        out = int(outlet)
        n_skipped = 0

        # Build lookup for L values if asymmetry_df provided
        L_lookup: dict[tuple[int, int, int], tuple[float, float]] = {}
        if asymmetry_df is not None and not asymmetry_df.empty:
            for _, row in asymmetry_df.iterrows():
                key = (int(row["confluence"]), int(row["head_1"]), int(row["head_2"]))
                L_lookup[key] = (float(row["L_1"]), float(row["L_2"]))

        for conf, pairs in pairs_at_confluence.items():
            if not pairs:
                continue

            for h1, h2 in pairs:
                h1_norm, h2_norm = _normalize_pair(int(h1), int(h2))

                # Look up L values
                L_1, L_2 = None, None
                key = (int(conf), h1_norm, h2_norm)
                if key in L_lookup:
                    L_1, L_2 = L_lookup[key]

                try:
                    result = self.compute_pair_geometry(h1_norm, h2_norm, int(conf), L_1, L_2)
                    rows.append(
                        {
                            "outlet": out,
                            "confluence": int(conf),
                            "head_1": result.head_1,
                            "head_2": result.head_2,
                            "orientation_diff_deg": result.orientation_diff_deg,
                            "headhead_dist_m": result.headhead_dist_m,
                            "headhead_dist_norm": result.headhead_dist_norm,
                            "apex_angle_deg": result.apex_angle_deg,
                            "strahler_order_diff": result.strahler_order_diff,
                            "proximity_mean_m": result.proximity_mean_m,
                            "proximity_max_m": result.proximity_max_m,
                            "proximity_profile_norm": result.proximity_profile_norm,
                            "qc_flags": result.qc_flags,
                        }
                    )
                except (ValueError, IndexError):
                    n_skipped += 1
                    continue

        if n_skipped > 0:
            logger.warning(
                "Outlet %d: skipped %d geometry pairs due to computation errors",
                out,
                n_skipped,
            )

        df = pd.DataFrame(
            rows,
            columns=[
                "outlet",
                "confluence",
                "head_1",
                "head_2",
                "orientation_diff_deg",
                "headhead_dist_m",
                "headhead_dist_norm",
                "apex_angle_deg",
                "strahler_order_diff",
                "proximity_mean_m",
                "proximity_max_m",
                "proximity_profile_norm",
                "qc_flags",
            ],
        )

        if not df.empty:
            df.sort_values(["confluence", "head_1", "head_2"], inplace=True, ignore_index=True)

        return df

    def clear_cache(self) -> None:
        """Clear cache (no-op, kept for API compatibility)."""
        pass


# =============================================================================
# Statistics & Merge Functions
# =============================================================================


def compute_asymmetry_statistics(delta_L_values: npt.ArrayLike) -> dict[str, float]:
    """Compute summary statistics for lengthwise asymmetry values.

    Parameters
    ----------
    delta_L_values : array-like
        Array of delta_L values.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'median', 'p25', 'p75', 'mean', 'std', 'count' keys.
        All float keys are NaN and count is 0 when input is empty or all-NaN.

    Example
    -------
    >>> stats = compute_asymmetry_statistics([0.1, 0.2, 0.3, 0.5, 0.8])
    >>> print(f"Median delta_L: {stats['median']:.2f}")
    """
    arr = np.asarray(delta_L_values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {
            "median": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "mean": np.nan,
            "std": np.nan,
            "count": 0,
        }

    return {
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": len(arr),
    }


def merge_coupling_and_asymmetry(
    coupling_df: pd.DataFrame,
    asymmetry_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge coupling analysis results with asymmetry analysis results.

    Parameters
    ----------
    coupling_df : pd.DataFrame
        DataFrame from CouplingAnalyzer.evaluate_pairs_for_outlet().
    asymmetry_df : pd.DataFrame
        DataFrame from LengthwiseAsymmetryAnalyzer.evaluate_pairs_for_outlet().

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all columns from both inputs.
    """
    # Merge on common keys
    merge_keys = ["outlet", "confluence", "head_1", "head_2"]

    # Select only asymmetry-specific columns to add
    asymmetry_cols = ["L_1", "L_2", "delta_L", "distance_unit"]
    asymmetry_subset = asymmetry_df[merge_keys + asymmetry_cols]

    merged = pd.merge(
        coupling_df,
        asymmetry_subset,
        on=merge_keys,
        how="left",
    )

    return merged


def merge_geometric_features(
    base_df: pd.DataFrame,
    geometric_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge geometric features into a base DataFrame.

    Parameters
    ----------
    base_df : pd.DataFrame
        Base DataFrame (e.g., from merge_coupling_and_asymmetry).
    geometric_df : pd.DataFrame
        From GeometricFeaturesAnalyzer with features 2-5.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with all columns.
    """
    merge_keys = ["outlet", "confluence", "head_1", "head_2"]

    # Use GEOM_FEATURE_COLS so new features are automatically included
    geom_cols = list(GEOM_FEATURE_COLS)

    # Add qc_flags if not already present with same name
    if "qc_flags" in geometric_df.columns and "qc_flags" not in base_df.columns:
        if "qc_flags" not in geom_cols:
            geom_cols.append("qc_flags")

    cols_to_merge = [c for c in geom_cols if c in geometric_df.columns]

    if not cols_to_merge:
        return base_df

    geom_subset = geometric_df[merge_keys + cols_to_merge]

    merged = pd.merge(
        base_df,
        geom_subset,
        on=merge_keys,
        how="left",
    )

    return merged


# =============================================================================
# Labeling Functions
# =============================================================================


def generate_labeled_dataset(
    coupling_df: pd.DataFrame,
    asymmetry_df: pd.DataFrame,
    geometric_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all features and add label column.

    Creates a labeled dataset for classification where:
    - y=1 (positive): pairs where touching=True (coupled)
    - y=0 (negative): pairs where touching=False (hard negatives)

    Parameters
    ----------
    coupling_df : pd.DataFrame
        From CouplingAnalyzer with 'touching' column.
    asymmetry_df : pd.DataFrame
        From LengthwiseAsymmetryAnalyzer with L_1, L_2, delta_L.
    geometric_df : pd.DataFrame
        From GeometricFeaturesAnalyzer with features 2-5.

    Returns
    -------
    pd.DataFrame
        Combined features with 'y' column (1=touching/positive, 0=not-touching/negative).
    """
    merge_keys = ["outlet", "confluence", "head_1", "head_2"]

    # Start with coupling (has the label)
    df = coupling_df.copy()

    # Add label column
    df["y"] = df["touching"].astype(int)

    # Merge asymmetry features
    if not asymmetry_df.empty:
        asymmetry_cols = ["L_1", "L_2", "delta_L"]
        cols_to_merge = [c for c in asymmetry_cols if c in asymmetry_df.columns]
        if cols_to_merge:
            asymmetry_subset = asymmetry_df[merge_keys + cols_to_merge]
            df = pd.merge(df, asymmetry_subset, on=merge_keys, how="left")

    # Merge geometric features (use GEOM_FEATURE_COLS so new features are included)
    if not geometric_df.empty:
        geom_cols = list(GEOM_FEATURE_COLS)
        cols_to_merge = [c for c in geom_cols if c in geometric_df.columns]
        if cols_to_merge:
            geom_subset = geometric_df[merge_keys + cols_to_merge]
            df = pd.merge(df, geom_subset, on=merge_keys, how="left")

    return df


def filter_hard_negatives(
    labeled_df: pd.DataFrame,
    max_L_ratio: float = 3.0,
    max_dist_ratio: float = 5.0,
    group_col: str | None = None,
    s: Any | None = None,
) -> pd.DataFrame:
    """Filter negatives to keep only 'hard' negatives matched to positives.

    Hard negatives are pairs at the same confluence with:
    - Similar L-scale (L_1 + L_2 within max_L_ratio of median positive)
    - Relatively close in planform (head-head distance within max_dist_ratio)
    - Head-to-head vector does NOT cross any stream pixel (geometric criterion)

    This avoids trivially far-away or trivially separated negatives. When ``s``
    is provided, pairs whose straight-line vector between the two channel heads
    crosses a stream pixel are removed as trivially non-touching.

    .. warning::
        **Data leakage risk:** The filtering thresholds are derived from the
        positives in ``labeled_df``. Call this function separately on each
        cross-validation fold's training set, not on the combined dataset, to
        prevent positives from the test fold from influencing the thresholds
        used to filter training negatives.

    Parameters
    ----------
    labeled_df : pd.DataFrame
        Labeled dataset with 'y', 'L_1', 'L_2', 'headhead_dist_m' columns.
    max_L_ratio : float, optional
        Maximum ratio of negative L_sum to median positive L_sum (default: 3.0).
    max_dist_ratio : float, optional
        Maximum ratio of negative distance to median positive distance (default: 5.0).
    group_col : str, optional
        Column name to group by (e.g., 'basin') for per-group threshold
        computation. If None, uses global thresholds across all data.
    s : StreamObject, optional
        TopoToolbox StreamObject. When provided, negatives whose head-to-head
        vector crosses a stream pixel are removed (geometric intersection filter).

    Returns
    -------
    pd.DataFrame
        Filtered dataset with positives and hard negatives only.
    """
    if labeled_df.empty:
        return labeled_df

    # Per-group filtering: compute thresholds within each group
    if group_col is not None and group_col in labeled_df.columns:
        parts = []
        for _, group_df in labeled_df.groupby(group_col):
            parts.append(filter_hard_negatives(group_df, max_L_ratio, max_dist_ratio, s=s))
        result = pd.concat(parts, ignore_index=True)
        result.sort_values(
            ["outlet", "confluence", "head_1", "head_2"],
            inplace=True,
            ignore_index=True,
        )
        return result

    # Separate positives and negatives
    positives = labeled_df[labeled_df["y"] == 1].copy()
    negatives = labeled_df[labeled_df["y"] == 0].copy()

    if positives.empty or negatives.empty:
        return labeled_df

    # Compute L_sum if L_1 and L_2 exist
    has_L = "L_1" in labeled_df.columns and "L_2" in labeled_df.columns
    has_dist = "headhead_dist_m" in labeled_df.columns

    if has_L:
        positives["L_sum"] = positives["L_1"] + positives["L_2"]
        negatives["L_sum"] = negatives["L_1"] + negatives["L_2"]

        median_pos_L = positives["L_sum"].median()

        if not pd.isna(median_pos_L) and median_pos_L > 0:
            L_threshold = median_pos_L * max_L_ratio
            negatives = negatives[(negatives["L_sum"].isna()) | (negatives["L_sum"] <= L_threshold)]

        # Clean up temp column
        positives.drop(columns=["L_sum"], inplace=True)
        if "L_sum" in negatives.columns:
            negatives.drop(columns=["L_sum"], inplace=True)

    if has_dist:
        median_pos_dist = positives["headhead_dist_m"].median()

        if not pd.isna(median_pos_dist) and median_pos_dist > 0:
            dist_threshold = median_pos_dist * max_dist_ratio
            negatives = negatives[
                (negatives["headhead_dist_m"].isna())
                | (negatives["headhead_dist_m"] <= dist_threshold)
            ]

    # Geometric stream-crossing filter: remove pairs whose head-to-head vector
    # crosses a stream pixel — these are trivially non-touching.
    # NOTE: Apply this AFTER train/test split to avoid cross-contamination of thresholds.
    if (
        s is not None
        and not negatives.empty
        and "head_1" in negatives.columns
        and "head_2" in negatives.columns
    ):
        stream_mask, r_nodes, c_nodes = _build_stream_mask(s)

        def _does_not_cross(h1: int, h2: int) -> bool:
            try:
                return not _line_crosses_stream(
                    int(r_nodes[h1]), int(c_nodes[h1]),
                    int(r_nodes[h2]), int(c_nodes[h2]),
                    stream_mask,
                )
            except (IndexError, KeyError):
                return True  # keep on error (conservative)

        keep_mask = [
            _does_not_cross(int(h1), int(h2))
            for h1, h2 in zip(negatives["head_1"], negatives["head_2"])
        ]
        negatives = negatives[keep_mask]

    # Combine
    result = pd.concat([positives, negatives], ignore_index=True)
    result.sort_values(
        ["outlet", "confluence", "head_1", "head_2"], inplace=True, ignore_index=True
    )

    return result


# =============================================================================
# CSV Enrichment Utility
# =============================================================================


def default_stream_loader(
    basin: str,
    lat: float,
    z_th: float,
    threshold: int = 300,
) -> tuple[Any, Any] | None:
    """Load DEM and create StreamObject for a basin.

    Parameters
    ----------
    basin : str
        Basin name (e.g., "inyo", "kammanasie").
    lat : float
        Latitude for coordinate conversion.
    z_th : float
        Elevation threshold for masking.
    threshold : int, optional
        Stream network area threshold in pixels (default: 300).

    Returns
    -------
    tuple[StreamObject, GridObject] or None
        (StreamObject, DEM GridObject) if successful, None if DEM not found.
    """
    try:
        import topotoolbox as tt3
    except ImportError:
        logger.error("topotoolbox not available - cannot load DEMs")
        return None

    dem_path = resolve_dem_path(basin)
    if dem_path is None or not Path(dem_path).exists():
        logger.warning(f"DEM not found for basin '{basin}'")
        return None

    try:
        dem = tt3.read_tif(str(dem_path))

        # Apply elevation threshold
        if z_th is not None and not np.isnan(z_th):
            dem.z[dem.z < z_th] = np.nan

        fd = tt3.FlowObject(dem)
        s = tt3.StreamObject(fd, threshold=threshold)

        return s, dem
    except Exception as e:
        logger.error(f"Failed to load DEM for basin '{basin}': {e}")
        return None


def _build_pairs_at_confluence(
    df_outlet: pd.DataFrame,
) -> dict[int, set[tuple[int, int]]]:
    """Build pairs_at_confluence dict from DataFrame rows.

    Parameters
    ----------
    df_outlet : pd.DataFrame
        DataFrame filtered to a single outlet, with columns:
        confluence, head_1, head_2.

    Returns
    -------
    dict[int, set[tuple[int, int]]]
        Mapping from confluence ID to set of (head_1, head_2) pairs.
    """
    pairs: dict[int, set[tuple[int, int]]] = {}
    for conf, h1, h2 in zip(
        df_outlet["confluence"].astype(int),
        df_outlet["head_1"].astype(int),
        df_outlet["head_2"].astype(int),
    ):
        h1_norm, h2_norm = _normalize_pair(h1, h2)
        if conf not in pairs:
            pairs[conf] = set()
        pairs[conf].add((h1_norm, h2_norm))
    return pairs


def _build_asymmetry_df(df_outlet: pd.DataFrame) -> pd.DataFrame:
    """Build asymmetry DataFrame from existing L_1, L_2 columns.

    Parameters
    ----------
    df_outlet : pd.DataFrame
        DataFrame with outlet, confluence, head_1, head_2, L_1, L_2 columns.

    Returns
    -------
    pd.DataFrame
        Subset with columns needed for asymmetry lookup.
    """
    required_cols = ["outlet", "confluence", "head_1", "head_2"]
    optional_cols = ["L_1", "L_2"]

    cols_present = required_cols + [c for c in optional_cols if c in df_outlet.columns]
    return df_outlet[cols_present].copy()


def _add_missing_stream_qc(df: pd.DataFrame, indices: pd.Index) -> pd.DataFrame:
    """Mark rows with missing_stream QC flag and NaN geometric features.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame being enriched.
    indices : pd.Index
        Indices of rows to mark.

    Returns
    -------
    pd.DataFrame
        DataFrame with updated rows.
    """
    for col in GEOM_FEATURE_COLS:
        if col == "qc_flags":
            # Append to existing qc_flags or set new
            if col not in df.columns:
                df[col] = ""
            df.loc[indices, col] = df.loc[indices, col].apply(
                lambda x: (f"{x},missing_stream" if x and not pd.isna(x) else "missing_stream")
            )
        else:
            if col not in df.columns:
                df[col] = np.nan
            df.loc[indices, col] = np.nan

    return df


def add_geometric_features_to_csv(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    stream_loader: StreamLoaderFunc | None = None,
    threshold: int = 300,
    verbose: bool = False,
) -> pd.DataFrame:
    """Add geometric features to an existing combined results CSV.

    Parameters
    ----------
    input_csv : str or Path
        Path to input CSV file with paired channel head data.
    output_csv : str or Path, optional
        Path to output CSV file. If None, returns DataFrame without saving.
    stream_loader : callable, optional
        Custom function to load StreamObject and DEM for a basin.
        Signature: (basin: str, lat: float, z_th: float) -> (s, dem) or None.
        If None, uses default_stream_loader.
    threshold : int, optional
        Stream threshold for default loader (default: 300).
    verbose : bool, optional
        Print progress information (default: False).

    Returns
    -------
    pd.DataFrame
        DataFrame with geometric features added.

    Raises
    ------
    FileNotFoundError
        If input CSV does not exist.
    ValueError
        If required columns are missing from input CSV.
    """
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    if verbose:
        # Configure only the package logger, not the root logger
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
            logger.addHandler(_h)

    # Load CSV
    logger.info(f"Loading CSV: {input_path}")
    df = pd.read_csv(input_path)

    # Validate required columns
    required_cols = ["outlet", "confluence", "head_1", "head_2"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Drop overlap_px if present
    if "overlap_px" in df.columns:
        logger.info("Dropping deprecated 'overlap_px' column")
        df = df.drop(columns=["overlap_px"])

    # Normalize head ordering early so merges match GeometricFeaturesAnalyzer output
    swap_mask = df["head_1"] > df["head_2"]
    if swap_mask.any():
        logger.info(f"Normalizing {swap_mask.sum()} rows with head_1 > head_2")
        df.loc[swap_mask, ["head_1", "head_2"]] = df.loc[swap_mask, ["head_2", "head_1"]].values
        if "L_1" in df.columns and "L_2" in df.columns:
            df.loc[swap_mask, ["L_1", "L_2"]] = df.loc[swap_mask, ["L_2", "L_1"]].values

    # Initialize geometric feature columns
    for col in GEOM_FEATURE_COLS:
        if col not in df.columns:
            df[col] = np.nan if col != "qc_flags" else ""

    # Use default loader if none provided
    if stream_loader is None:

        def stream_loader(basin: str, lat: float, z_th: float) -> tuple[Any, Any] | None:
            return default_stream_loader(basin, lat, z_th, threshold=threshold)

    # Check if basin column exists
    if "basin" not in df.columns:
        logger.warning("No 'basin' column found - treating all data as single basin")
        df["basin"] = "unknown"

    # Get lat and z_th columns if present
    has_lat = "lat" in df.columns
    has_z_th = "z_th" in df.columns

    # Group by basin
    basins = df["basin"].unique()
    logger.info(f"Processing {len(basins)} basin(s)")

    for basin in basins:
        basin_mask = df["basin"] == basin
        df_basin = df[basin_mask]

        # Get lat and z_th for this basin
        if has_lat:
            lat = df_basin["lat"].iloc[0]
        else:
            lat = 36.0  # Default latitude
            logger.warning(f"No 'lat' column - using default {lat} for {basin}")

        if has_z_th:
            z_th = df_basin["z_th"].iloc[0]
        else:
            z_th = 0.0  # No threshold
            logger.warning(f"No 'z_th' column - using default {z_th} for {basin}")

        # Load stream network for this basin
        logger.info(f"Loading stream network for basin '{basin}'")
        result = stream_loader(basin, lat, z_th)

        if result is None:
            # Mark all rows for this basin with missing_stream
            logger.warning(f"Could not load stream for basin '{basin}' - marking rows")
            df = _add_missing_stream_qc(df, df_basin.index)
            continue

        s, dem = result

        # Create analyzer for this basin
        analyzer = GeometricFeaturesAnalyzer(s, dem, lat=lat)

        # Process each outlet in this basin
        outlets = df_basin["outlet"].unique()
        logger.info(f"Processing {len(outlets)} outlet(s) in basin '{basin}'")

        for outlet in outlets:
            outlet_mask = basin_mask & (df["outlet"] == outlet)
            df_outlet = df[outlet_mask]

            # Build pairs_at_confluence
            pairs_at_confluence = _build_pairs_at_confluence(df_outlet)

            if not pairs_at_confluence:
                logger.debug(f"No pairs for outlet {outlet}")
                continue

            # Build asymmetry_df for L values
            asymmetry_df = _build_asymmetry_df(df_outlet)

            # Compute geometric features
            try:
                geom_df = analyzer.evaluate_pairs_for_outlet(
                    int(outlet), pairs_at_confluence, asymmetry_df=asymmetry_df
                )
            except Exception as e:
                logger.error(f"Error computing features for outlet {outlet}: {e}")
                df = _add_missing_stream_qc(df, df_outlet.index)
                continue

            # Merge geometric features back to main DataFrame
            if not geom_df.empty:
                merge_keys = ["outlet", "confluence", "head_1", "head_2"]
                geom_cols_present = [c for c in GEOM_FEATURE_COLS if c in geom_df.columns]
                if geom_cols_present:
                    geom_subset = geom_df[merge_keys + geom_cols_present]
                    df = df.set_index(merge_keys)
                    geom_subset = geom_subset.set_index(merge_keys)
                    df.update(geom_subset)
                    df = df.reset_index()

        # Clear analyzer cache between basins
        analyzer.clear_cache()

    # Save output if path provided
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved enriched CSV to: {output_path}")

    return df


def _add_geometric_features_cli() -> None:
    """CLI entry point for adding geometric features to CSV."""
    parser = argparse.ArgumentParser(
        description="Add geometric features to channel head pair results CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python -m channel_heads.geometric_analysis \\
        --input all_basins_combined_results.csv \\
        --output all_basins_combined_with_geom.csv \\
        --verbose
        """,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        default=300,
        help="Stream network threshold (default: 300)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()

    add_geometric_features_to_csv(
        input_csv=args.input,
        output_csv=args.output,
        threshold=args.threshold,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    _add_geometric_features_cli()


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Coordinate conversion
    "compute_meters_per_degree",
    "compute_pixel_size_meters",
    # Asymmetry
    "PairAsymmetryResult",
    "compute_delta_L",
    "LengthwiseAsymmetryAnalyzer",
    "compute_asymmetry_statistics",
    "merge_coupling_and_asymmetry",
    # Geometric features
    "PairGeometricResult",
    "GeometricFeaturesAnalyzer",
    # Labeling & merge
    "generate_labeled_dataset",
    "filter_hard_negatives",
    "merge_geometric_features",
    # CSV enrichment
    "add_geometric_features_to_csv",
    "default_stream_loader",
]
