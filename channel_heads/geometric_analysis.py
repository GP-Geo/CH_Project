"""Geometric analysis for paired channel heads (compatibility re-export surface).

This module historically provided all geometric feature computation for channel
head pairs that share a first common downstream confluence:

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

The implementations now live in the ``channel_heads.features`` and
``channel_heads.training`` packages and are re-exported here so existing imports
(`from channel_heads.geometric_analysis import ...`) and the
``python -m channel_heads.geometric_analysis`` CLI keep working:

    - lengthwise asymmetry      -> ``channel_heads.features.asymmetry``
    - Earth geometry analyzer   -> ``channel_heads.features.earth_geometry``
    - path / coordinate helpers -> ``channel_heads.features.earth_paths``
    - pure feature math         -> ``channel_heads.features.geometry``
    - labeling / hard negatives -> ``channel_heads.training.labeling``
    - CSV enrichment / loader   -> ``channel_heads.features.earth_enrichment``
    - unit conversions          -> ``channel_heads.units``

References:
    Goren, L. and Shelef, E.: Channel concavity controls planform complexity
    of branching drainage networks, Earth Surf. Dynam., 12, 1347-1369,
    https://doi.org/10.5194/esurf-12-1347-2024, 2024.
"""

from __future__ import annotations

from .features.asymmetry import (
    LengthwiseAsymmetryAnalyzer,
    PairAsymmetryResult,
    compute_asymmetry_statistics,
    compute_delta_L,
    merge_coupling_and_asymmetry,
)
from .features.earth_enrichment import (
    StreamLoaderFunc,  # noqa: F401  (re-exported for backward compatibility)
    _add_geometric_features_cli,
    _add_missing_stream_qc,  # noqa: F401  (re-exported for backward compatibility)
    _build_asymmetry_df,  # noqa: F401  (re-exported for backward compatibility)
    _build_pairs_at_confluence,  # noqa: F401  (re-exported for backward compatibility)
    add_geometric_features_to_csv,
    default_stream_loader,
)
from .features.earth_geometry import (
    DEFAULT_DIRECTION_SAMPLE_DISTANCE_M,  # noqa: F401  (re-exported for backward compatibility)
    GEOM_FEATURE_COLS,
    GeometricFeaturesAnalyzer,
    PairGeometricResult,
    merge_geometric_features,
)
from .features.earth_paths import EPSILON as EPSILON  # noqa: F401
from .features.earth_paths import MIN_EDGES_FOR_DIRECTION as MIN_EDGES_FOR_DIRECTION  # noqa: F401
from .features.earth_paths import (
    _build_children_from_parents as _build_children_from_parents,  # noqa: F401
)
from .features.earth_paths import (
    _compute_direction_vector as _compute_direction_vector,  # noqa: F401
)
from .features.earth_paths import _detect_cellsize as _detect_cellsize  # noqa: F401
from .features.earth_paths import _euclidean_2d as _euclidean_2d  # noqa: F401
from .features.earth_paths import _normalize_vector as _normalize_vector  # noqa: F401
from .features.earth_paths import _sample_path_coords as _sample_path_coords  # noqa: F401
from .features.earth_paths import _trace_full_path as _trace_full_path  # noqa: F401
from .features.earth_paths import _trace_path_downstream as _trace_path_downstream  # noqa: F401
from .features.geometry import angle_between_vectors as _angle_between_vectors  # noqa: F401
from .features.geometry import azimuth_difference as _azimuth_difference  # noqa: F401
from .features.geometry import compute_azimuth as _compute_azimuth  # noqa: F401
from .features.geometry import compute_proximity_profile as _compute_proximity_profile  # noqa: F401
from .training.labeling import (
    _build_stream_mask,  # noqa: F401  (re-exported for backward compatibility)
    _line_crosses_stream,  # noqa: F401  (re-exported; imported by tests/legacy users)
    filter_hard_negatives,
    generate_labeled_dataset,
)
from .units import compute_meters_per_degree, compute_pixel_size_meters

# =============================================================================
# Type Aliases (kept for backward compatibility)
# =============================================================================

NodeId = int
HeadId = int
HeadPair = tuple[HeadId, HeadId]
ParentsList = list[list[NodeId]]
ChildrenDict = dict[NodeId, list[NodeId]]
Coord2D = tuple[float, float]


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
    "GEOM_FEATURE_COLS",
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


if __name__ == "__main__":
    _add_geometric_features_cli()
