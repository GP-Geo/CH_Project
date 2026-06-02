"""Unit conversions — single source of truth.

Every length / area / threshold unit conversion used across the package lives
here. These functions were previously scattered across ``geometric_analysis.py``
and ``dd_calibration.py``; those modules now re-export from this module so the
public API is unchanged and behavior is identical (this is a pure consolidation,
no numerical change).

Covered conversions
-------------------
- meters per degree of latitude/longitude (``compute_meters_per_degree``)
- geographic pixel size in metres (``compute_pixel_size_meters``)
- DEM pixel size with geographic-vs-projected handling (``compute_pixel_size_m_from_dem``)
- km² → pixel cells (``compute_threshold_cells``)
- stream length in km (``compute_stream_length_km``)
- basin area in km² (``compute_basin_area_km2``)
- drainage density km/km² (``compute_drainage_density``)

Convex-hull area in km² (``compute_convex_hull_area_km2``) remains in
``dd_calibration.py`` because it is coupled to SciPy ``ConvexHull`` geometry and
status handling; it consumes ``pixel_size_m`` produced here.

Risk S1 — ``upstream_distance()`` units
---------------------------------------
``LengthwiseAsymmetryAnalyzer`` (in ``geometric_analysis.py``) assumes
TopoToolbox's ``s.upstream_distance()`` returns distances in **map units**
(arc-degrees for a geographic-CRS SRTM DEM) and multiplies by
``compute_meters_per_degree(lat)`` to get metres. If TopoToolbox instead returns
**pixel / edge counts**, the correct scale would be
``compute_pixel_size_meters(lat, cellsize)`` and ΔL values would currently be
off by a factor of ``cellsize_deg`` (≈ 1/3600 for 1-arc-second SRTM).

This has **not** been re-verified, so behavior is intentionally preserved. Use
``meters_per_unit_for_upstream_distance`` below as the single documented place to
resolve the scale once the unit convention is confirmed; see
``docs/ROADMAP_AND_RISKS.md`` (S1) for the verification recipe.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

# =============================================================================
# Constants — coordinate conversion
# =============================================================================

METERS_PER_DEGREE_LAT = 110540.0  # meters per degree of latitude (approximately constant)
METERS_PER_DEGREE_LON_EQUATOR = 111320.0  # meters per degree of longitude at equator


# =============================================================================
# Coordinate / pixel-size conversions
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


def compute_pixel_size_m_from_dem(dem: Any, lat_deg: float | None = None) -> float:
    """Resolve DEM pixel size to metres.

    Geographic-CRS DEMs (cellsize ~ 1 arc-second, i.e. cellsize < 1) are
    converted using `lat_deg`. Projected DEMs (cellsize >= 1) are assumed
    to be already in metres.
    """
    cellsize = getattr(dem, "cellsize", None)
    if cellsize is None:
        transform = getattr(dem, "transform", None)
        if transform is None:
            raise ValueError("DEM has no `cellsize` or `transform`")
        cellsize = abs(getattr(transform, "a", transform[0]))

    if cellsize >= 1.0:
        return float(cellsize)
    if lat_deg is None:
        raise ValueError(
            "Geographic-CRS DEM (cellsize < 1) requires `lat_deg` to convert to metres"
        )
    return compute_pixel_size_meters(float(lat_deg), float(cellsize))


def meters_per_unit_for_upstream_distance(
    lat_deg: float,
    cellsize_deg: float,
    *,
    assume_map_units: bool = True,
) -> float:
    """Scale factor to convert ``s.upstream_distance()`` output to metres (risk S1).

    The single documented place to resolve the ``upstream_distance()`` unit
    ambiguity. The default (``assume_map_units=True``) reproduces the current
    behaviour used by ``LengthwiseAsymmetryAnalyzer`` — treating the output as
    arc-degrees and scaling by metres-per-degree. If verification shows
    TopoToolbox returns pixel/edge counts, pass ``assume_map_units=False`` to
    scale by pixel size instead.

    This helper is not yet wired into the analyzers (behavior preserved); see
    ``docs/ROADMAP_AND_RISKS.md`` (S1).
    """
    if assume_map_units:
        return compute_meters_per_degree(lat_deg)
    return compute_pixel_size_meters(lat_deg, cellsize_deg)


# =============================================================================
# Area / length / density conversions
# =============================================================================


def compute_threshold_cells(threshold_km2: float, pixel_size_m: float) -> int:
    """Convert a contributing-area threshold in km^2 to pixel count."""
    cell_area_m2 = pixel_size_m**2
    cells = (threshold_km2 * 1_000_000.0) / cell_area_m2
    return max(1, int(round(cells)))


def compute_stream_length_km(s: Any, pixel_size_m: float) -> float:
    """Sum Euclidean edge lengths of a StreamObject and return km.

    `source_indices` / `target_indices` are (row, col) tuples pointing into
    the DEM grid — one entry per edge — not node-id arrays.
    """
    src_r, src_c = s.source_indices
    tgt_r, tgt_c = s.target_indices
    src_r = np.asarray(src_r)
    src_c = np.asarray(src_c)
    tgt_r = np.asarray(tgt_r)
    tgt_c = np.asarray(tgt_c)
    if src_r.size == 0:
        return 0.0
    edge_pixels = np.hypot(src_r - tgt_r, src_c - tgt_c)
    return float(edge_pixels.sum() * pixel_size_m / 1000.0)


def compute_basin_area_km2(basin_mask: npt.NDArray[np.bool_], pixel_size_m: float) -> float:
    n_px = int(basin_mask.sum())
    return n_px * (pixel_size_m**2) / 1_000_000.0


def compute_drainage_density(length_km: float, area_km2: float) -> float:
    """Drainage density = channel length / reference area (km / km^2).

    The reference area is the convex-hull area for ``Dd_hull`` or the
    DEM-derived basin area for ``Dd_true``. Returns NaN for a non-positive
    area so callers never divide by zero.
    """
    if area_km2 is None or not np.isfinite(area_km2) or area_km2 <= 0.0:
        return float("nan")
    return float(length_km) / float(area_km2)
