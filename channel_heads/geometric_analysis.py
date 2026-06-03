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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .config import resolve_dem_path
from .features.asymmetry import (
    LengthwiseAsymmetryAnalyzer,
    PairAsymmetryResult,
    compute_asymmetry_statistics,
    compute_delta_L,
    merge_coupling_and_asymmetry,
)
from .features.earth_geometry import (
    DEFAULT_DIRECTION_SAMPLE_DISTANCE_M,  # noqa: F401  (re-exported for backward compatibility)
    GEOM_FEATURE_COLS,
    GeometricFeaturesAnalyzer,
    PairGeometricResult,  # noqa: F401  (re-exported for backward compatibility)
    merge_geometric_features,  # noqa: F401  (re-exported for backward compatibility)
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
from .logging_config import get_logger
from .pairing.earth import _normalize_pair
from .stream_utils import line_pixels
from .units import compute_meters_per_degree, compute_pixel_size_meters

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
#
# ``DEFAULT_DIRECTION_SAMPLE_DISTANCE_M``, ``GEOM_FEATURE_COLS``,
# ``MIN_EDGES_FOR_DIRECTION`` and ``EPSILON`` now live in the
# ``channel_heads.features`` package and are imported above for backward
# compatibility.

# Type alias for the stream loader used by add_geometric_features_to_csv
StreamLoaderFunc = Callable[[str, float, float], tuple[Any, Any] | None]


# =============================================================================
# Earth geometry / path helpers (re-exported)
# =============================================================================
#
# The Earth geometry analyzer (``PairGeometricResult``,
# ``GeometricFeaturesAnalyzer``, ``GEOM_FEATURE_COLS``,
# ``merge_geometric_features``) now lives in
# ``channel_heads.features.earth_geometry``; the path / coordinate helpers live
# in ``channel_heads.features.earth_paths``. Both are imported at the top of this
# module and re-exported here for backward compatibility.


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
    rr, cc = line_pixels(r1, c1, r2, c2)
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
                    int(r_nodes[h1]),
                    int(c_nodes[h1]),
                    int(r_nodes[h2]),
                    int(c_nodes[h2]),
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
