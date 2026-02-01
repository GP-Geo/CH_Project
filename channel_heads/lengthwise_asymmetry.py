"""Lengthwise asymmetry metric from Goren & Shelef (2024).

This module implements the lengthwise asymmetry (ΔL) metric for quantifying
drainage network complexity, as described in the paper:

    Goren, L. and Shelef, E.: Channel concavity controls planform complexity
    of branching drainage networks, Earth Surf. Dynam., 12, 1347–1369,
    https://doi.org/10.5194/esurf-12-1347-2024, 2024.

The lengthwise asymmetry quantifies the difference in flow path lengths between
paired channel heads that diverge from a common divide and rejoin at a junction.

Formula (Equation 4 from the paper):
    ΔL_ij = 2|L_ij - L_ji| / (L_ij + L_ji)

Where:
    L_ij = along-flow distance from channel head i to common junction
    L_ji = along-flow distance from channel head j to common junction

Values range from 0 (perfect symmetry) to 2 (maximum asymmetry).

Implementation Notes:
    This module uses TopoToolbox's built-in distance functions:
    - upstream_distance(): cumulative distance from each node to the outlet

    For a head and confluence on the same flow path:
        L = upstream_dist[head] - upstream_dist[confluence]

Usage:
    from channel_heads.lengthwise_asymmetry import (
        LengthwiseAsymmetryAnalyzer,
        compute_delta_L,
    )

    # Using the analyzer class
    analyzer = LengthwiseAsymmetryAnalyzer(s, dem)
    result = analyzer.compute_pair_asymmetry(head_1, head_2, confluence)
    print(f"ΔL = {result['delta_L']:.2f}")

    # Using the simple function
    delta_L = compute_delta_L(L_ij=1500, L_ji=2000)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import pandas as pd

if TYPE_CHECKING:
    from topotoolbox import StreamObject, GridObject

# Type aliases
NodeId = int
HeadPair = Tuple[int, int]

# Constants for coordinate conversion
METERS_PER_DEGREE_LAT = 110540.0  # meters per degree of latitude (approximately constant)
METERS_PER_DEGREE_LON_EQUATOR = 111320.0  # meters per degree of longitude at equator


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
    - 1 degree of latitude ≈ 110,540 meters (approximately constant)
    - 1 degree of longitude ≈ 111,320 * cos(latitude) meters

    We use the geometric mean for flow paths that can go in any direction.

    Examples
    --------
    >>> # At 36.7° latitude (Inyo Mountains)
    >>> m_per_deg = compute_meters_per_degree(36.7)
    >>> print(f"Meters per degree: {m_per_deg:.0f}")
    Meters per degree: 99287
    """
    lat_rad = np.radians(abs(lat_deg))

    # Meters per degree in each direction
    meters_per_deg_lon = METERS_PER_DEGREE_LON_EQUATOR * np.cos(lat_rad)
    meters_per_deg_lat = METERS_PER_DEGREE_LAT

    # Use geometric mean for flow paths in arbitrary directions
    return np.sqrt(meters_per_deg_lon * meters_per_deg_lat)


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
    >>> # 1 arc-second SRTM at 36.7° latitude (Inyo Mountains)
    >>> pixel_size = compute_pixel_size_meters(36.7, 1/3600)
    >>> print(f"Pixel size: {pixel_size:.1f} m")
    Pixel size: 27.6 m
    """
    return cellsize_deg * compute_meters_per_degree(lat_deg)


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


def compute_delta_L(L_ij: float, L_ji: float) -> float:
    """Compute normalized lengthwise asymmetry from two path lengths.

    This is the core formula from Equation 4 of Goren & Shelef (2024):
        ΔL = 2|L_ij - L_ji| / (L_ij + L_ji)

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
    >>> # For Inyo Mountains at latitude 36.71°
    >>> analyzer = LengthwiseAsymmetryAnalyzer(s, dem, lat=36.71)
    >>> result = analyzer.compute_pair_asymmetry(head_1=100, head_2=150, confluence=200)
    """

    def __init__(
        self,
        s: Any,  # StreamObject
        dem: Optional[Any] = None,  # GridObject
        lat: Optional[float] = None,  # Latitude in degrees
    ) -> None:
        self.s = s
        self.dem = dem
        self.lat = lat

        # Precompute upstream distances using TopoToolbox's built-in function
        # upstream_distance() returns cumulative distance from each node to the outlet
        self._upstream_dist: np.ndarray = s.upstream_distance()

        # Compute meters per unit conversion factor
        self._meters_per_unit: float = 1.0  # Default: assume already in meters
        self._detected_cellsize: Optional[float] = None

        if lat is not None:
            # Get cell size from StreamObject (this is what upstream_distance uses)
            # The StreamObject.distance() method uses self.cellsize
            cellsize = getattr(s, 'cellsize', None)

            # Fallback: try to get from DEM if not on StreamObject
            if cellsize is None and dem is not None:
                cellsize = getattr(dem, 'cellsize', None)

                if cellsize is None:
                    res = getattr(dem, 'res', None)
                    if res is not None:
                        cellsize = abs(res[0]) if isinstance(res, (tuple, list)) else abs(res)

                if cellsize is None:
                    transform = getattr(dem, 'transform', None)
                    if transform is not None:
                        if hasattr(transform, 'a'):
                            cellsize = abs(transform.a)
                        elif isinstance(transform, (tuple, list)) and len(transform) >= 1:
                            cellsize = abs(transform[0])

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
    def detected_cellsize(self) -> Optional[float]:
        """Detected cell size from DEM (for debugging)."""
        return self._detected_cellsize

    def compute_pair_asymmetry(
        self,
        head_1: int,
        head_2: int,
        confluence: int,
        use_meters: bool = True,  # kept for API compatibility, always uses meters
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

        # Compute flow path lengths using upstream distance
        # upstream_dist[node] = distance from node to outlet
        # L = upstream_dist[head] - upstream_dist[confluence]
        # This works because the head is further from the outlet than the confluence
        L_1_raw = float(self._upstream_dist[h1] - self._upstream_dist[conf])
        L_2_raw = float(self._upstream_dist[h2] - self._upstream_dist[conf])

        # Sanity check: distances should be non-negative
        # (head should be further from outlet than confluence)
        if L_1_raw < 0:
            L_1_raw = 0.0
        if L_2_raw < 0:
            L_2_raw = 0.0

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
        pairs_at_confluence: Dict[int, Set[HeadPair]],
        use_meters: bool = True,
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
            Kept for API compatibility.

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
                    rows.append({
                        "outlet": out,
                        "confluence": int(conf),
                        "head_1": head_min,
                        "head_2": head_max,
                        "L_1": L_1_out,
                        "L_2": L_2_out,
                        "delta_L": result.delta_L,
                        "distance_unit": result.distance_unit,
                    })
                except ValueError:
                    # Skip pairs where computation fails
                    continue

        df = pd.DataFrame(rows, columns=[
            "outlet", "confluence", "head_1", "head_2",
            "L_1", "L_2", "delta_L", "distance_unit"
        ])

        if not df.empty:
            df.sort_values(["confluence", "head_1", "head_2"], inplace=True, ignore_index=True)

        return df

    def clear_cache(self) -> None:
        """Clear cache (no-op, kept for API compatibility)."""
        pass  # upstream_distance is computed once in __init__


def compute_asymmetry_statistics(delta_L_values: npt.ArrayLike) -> Dict[str, float]:
    """Compute summary statistics for lengthwise asymmetry values.

    Parameters
    ----------
    delta_L_values : array-like
        Array of ΔL values.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'median', 'p25', 'p75', 'mean', 'std' keys.

    Example
    -------
    >>> stats = compute_asymmetry_statistics([0.1, 0.2, 0.3, 0.5, 0.8])
    >>> print(f"Median ΔL: {stats['median']:.2f}")
    """
    arr = np.asarray(delta_L_values)
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

    Example
    -------
    >>> from channel_heads import CouplingAnalyzer, first_meet_pairs_for_outlet
    >>> from channel_heads.lengthwise_asymmetry import LengthwiseAsymmetryAnalyzer
    >>>
    >>> pairs, heads = first_meet_pairs_for_outlet(s, outlet)
    >>> coupling = CouplingAnalyzer(fd, s, dem).evaluate_pairs_for_outlet(outlet, pairs)
    >>> asymmetry = LengthwiseAsymmetryAnalyzer(s, dem).evaluate_pairs_for_outlet(outlet, pairs)
    >>> combined = merge_coupling_and_asymmetry(coupling, asymmetry)
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
