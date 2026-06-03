"""Earth geometric feature generation for channel-head pairs.

This module is the canonical home for the Earth/TopoToolbox geometric feature
analyzer that computes, for each channel-head pair sharing a first common
downstream confluence:

    - orientation similarity (initial downstream azimuth difference),
    - Euclidean head-head distance (raw and normalized),
    - apex angle at the confluence,
    - Strahler order difference, and
    - the proximity profile.

The implementation was extracted verbatim from
:mod:`channel_heads.geometric_analysis`; that module re-exports these names for
backward compatibility, so existing imports such as
``from channel_heads.geometric_analysis import GeometricFeaturesAnalyzer`` keep
working.

References:
    Goren, L. and Shelef, E.: Channel concavity controls planform complexity
    of branching drainage networks, Earth Surf. Dynam., 12, 1347-1369,
    https://doi.org/10.5194/esurf-12-1347-2024, 2024.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..logging_config import get_logger
from ..pairing.earth import _build_parents_from_stream, _normalize_pair
from ..units import compute_pixel_size_meters
from .earth_paths import (
    EPSILON,
    ChildrenDict,
    NodeId,
    ParentsList,
    _build_children_from_parents,
    _compute_direction_vector,
    _detect_cellsize,
    _euclidean_2d,
    _sample_path_coords,
    _trace_full_path,
    _trace_path_downstream,
)
from .geometry import angle_between_vectors as _angle_between_vectors
from .geometry import azimuth_difference as _azimuth_difference
from .geometry import compute_azimuth as _compute_azimuth
from .geometry import compute_proximity_profile as _compute_proximity_profile

logger = get_logger(__name__)

# =============================================================================
# Type Aliases
# =============================================================================

HeadId = int
HeadPair = tuple[HeadId, HeadId]

# =============================================================================
# Constants
# =============================================================================

# Distance along path for downstream direction estimation (meters).
DEFAULT_DIRECTION_SAMPLE_DISTANCE_M = 500.0

# Geometric feature columns added by GeometricFeaturesAnalyzer and
# add_geometric_features_to_csv.
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


# =============================================================================
# Data Structures
# =============================================================================


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
# Merge Functions
# =============================================================================


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


__all__ = [
    "DEFAULT_DIRECTION_SAMPLE_DISTANCE_M",
    "GEOM_FEATURE_COLS",
    "PairGeometricResult",
    "GeometricFeaturesAnalyzer",
    "merge_geometric_features",
]
