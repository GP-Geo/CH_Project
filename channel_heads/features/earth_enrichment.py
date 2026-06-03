"""CSV enrichment and default Earth stream loading for channel-head pairs.

This module is the canonical home for the batch CSV-enrichment workflow that
adds geometric features to an existing combined-results CSV, plus the default
TopoToolbox stream loader it uses:

    - ``default_stream_loader`` loads a basin DEM and builds a StreamObject,
    - ``add_geometric_features_to_csv`` enriches a CSV with the geometric
      feature columns (per basin / per outlet), and
    - ``_add_geometric_features_cli`` is the argparse CLI entry point.

The implementation was extracted verbatim from
:mod:`channel_heads.geometric_analysis`; that module re-exports these names (and
keeps the ``python -m channel_heads.geometric_analysis`` CLI working) for
backward compatibility.

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
import pandas as pd

from ..config import resolve_dem_path
from ..logging_config import get_logger
from ..pairing.earth import _normalize_pair
from .earth_geometry import GEOM_FEATURE_COLS, GeometricFeaturesAnalyzer

logger = get_logger(__name__)

# Type alias for the stream loader used by add_geometric_features_to_csv
StreamLoaderFunc = Callable[[str, float, float], tuple[Any, Any] | None]


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


__all__ = [
    "StreamLoaderFunc",
    "default_stream_loader",
    "add_geometric_features_to_csv",
]


if __name__ == "__main__":
    _add_geometric_features_cli()
