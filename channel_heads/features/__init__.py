"""Shared geometric feature primitives for the Earth and Mars pipelines.

See :mod:`channel_heads.features.geometry`.
"""

from __future__ import annotations

from .geometry import (
    angle_between_vectors,
    azimuth_difference,
    compute_azimuth,
    compute_proximity_profile,
)
from .mars_features import build_feature_table, build_mars_features
from .paths import (
    DIRECTION_SAMPLE_DISTANCE_M,
    MIN_EDGES_FOR_DIRECTION,
    N_PROXIMITY_SAMPLES,
    line_direction_first_n_meters,
    sample_path_coords_along_line,
)

__all__ = [
    "DIRECTION_SAMPLE_DISTANCE_M",
    "MIN_EDGES_FOR_DIRECTION",
    "N_PROXIMITY_SAMPLES",
    "angle_between_vectors",
    "azimuth_difference",
    "compute_azimuth",
    "compute_proximity_profile",
    "line_direction_first_n_meters",
    "sample_path_coords_along_line",
    # Mars Phase-3A feature generation
    "build_mars_features",
    "build_feature_table",
]
