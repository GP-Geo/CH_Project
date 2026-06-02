"""Pure geometric feature primitives shared by the Earth and Mars pipelines.

Scalar / vector math behind the dimensionless pair features (orientation
difference, apex angle, proximity profile). These were duplicated between
``geometric_analysis.py`` (Earth) and ``build_mars_pair_features_5feat.py``
(Mars, explicit "ports"); the byte-identical primitives now live here.

Everything in this module is a pure function on plain coordinates / vectors —
no TopoToolbox ``StreamObject`` or GeoDataFrame dependency — so both pipelines
can call the same implementation.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

EPSILON = 1e-10  # Small value for numerical stability

Coord2D = tuple[float, float]


def angle_between_vectors(v1: Coord2D, v2: Coord2D) -> float:
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


def compute_azimuth(dx: float, dy: float) -> float:
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


def azimuth_difference(az1: float, az2: float) -> float:
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


def compute_proximity_profile(
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
