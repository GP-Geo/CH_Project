"""Stream network rasterization for CNN-based spatial feature extraction.

This module converts stream network topology into fixed-size rasterized images
suitable for CNN input. Each image encodes a pair of channel heads and their
shared confluence within the context of the outlet's stream network.

Raster encoding (5 classes):
    0 = background (non-stream pixels)
    1 = branch A (head_1 → confluence path)
    2 = branch B (head_2 → confluence path)
    3 = other streams in the outlet
    4 = confluence marker

The raster is canonically aligned: centered on the confluence, rotated so the
confluence is at the bottom and the midpoint of the two heads is at the top,
then cropped and drawn directly into a fixed-size output grid.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features.earth_paths import _trace_full_path  # noqa: F401  (compat re-export)
from .logging_config import get_logger
from .rasterization.earth_patches import (
    _component_count,
    _compute_rotation_angle,
    _draw_edges_on_target_grid,
    _draw_path_on_target_grid,
    _get_rc,
    _project_to_target_grid,
    _rotate_coordinates,
    bresenham_line,
    raster_quality_flags,
    rasterize_outlet_pair,
)
from .rasterization.schema import (
    BACKGROUND,
    BRANCH_A,
    BRANCH_B,
    CONFLUENCE_MARKER,
    NUM_CLASSES,
    OTHER_STREAMS,
)

logger = get_logger(__name__)


# =============================================================================
# Batch Pre-computation
# =============================================================================


def precompute_raster_dataset(
    master_csv: Path,
    output_dir: Path,
    dem_loader: Callable[[str, float, float, int], tuple[Any, Any] | None],
    target_size: int = 128,
    threshold: int = 300,
) -> pd.DataFrame:
    """Pre-render raster patches for all pairs in the master dataset.

    Parameters
    ----------
    master_csv : Path
        Path to the master dataset CSV with columns: basin, outlet, head_1,
        head_2, confluence.
    output_dir : Path
        Directory to save .npy raster files. Organized as
        ``output_dir/{basin}/rasters/``.
    dem_loader : Callable
        Function ``(basin, lat, z_th, threshold) -> (StreamObject, GridObject)``
        or ``None`` if DEM not found. Same signature as
        ``geometric_analysis.default_stream_loader``.
    target_size : int
        Output image size.
    threshold : int
        Stream network threshold parameter.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added 'raster_path' column.
    """
    from .basin_config import get_basin_config

    df = pd.read_csv(master_csv)
    raster_paths: list[str | None] = [None] * len(df)
    raster_debug_paths: list[str | None] = [None] * len(df)
    raster_status: list[str] = ["pending"] * len(df)
    raster_errors: list[str] = [""] * len(df)
    qa_values: dict[str, list[bool]] = {
        "has_branch_a": [False] * len(df),
        "has_branch_b": [False] * len(df),
        "has_confluence": [False] * len(df),
        "branch_a_connected": [False] * len(df),
        "branch_b_connected": [False] * len(df),
        "branches_connected": [False] * len(df),
    }

    for basin_name, basin_df in df.groupby("basin"):
        basin_name = str(basin_name)
        logger.info("Rasterizing basin: %s (%d pairs)", basin_name, len(basin_df))

        # Load stream network
        try:
            config = get_basin_config(basin_name)
        except KeyError:
            logger.warning("No config for basin %s, skipping", basin_name)
            for row_idx in basin_df.index:
                raster_status[row_idx] = "skipped"
                raster_errors[row_idx] = "missing_basin_config"
            continue

        result = dem_loader(basin_name, config["lat"], config["z_th"], threshold)
        if result is None:
            logger.warning("DEM not found for basin %s, skipping", basin_name)
            for row_idx in basin_df.index:
                raster_status[row_idx] = "skipped"
                raster_errors[row_idx] = "missing_dem_or_stream"
            continue

        s, dem = result
        grid_shape = dem.shape if hasattr(dem, "shape") else dem.z.shape

        # Create output directory
        basin_raster_dir = output_dir / basin_name / "rasters"
        basin_raster_dir.mkdir(parents=True, exist_ok=True)

        for row_idx, row in basin_df.iterrows():
            outlet = int(row["outlet"])
            head_1 = int(row["head_1"])
            head_2 = int(row["head_2"])
            confluence = int(row["confluence"])

            fname = f"{outlet}_{head_1}_{head_2}.npy"
            fpath = basin_raster_dir / fname

            try:
                raster = rasterize_outlet_pair(
                    s,
                    outlet,
                    head_1,
                    head_2,
                    confluence,
                    grid_shape,
                    target_size=target_size,
                )
                flags = raster_quality_flags(raster)
                for key, value in flags.items():
                    qa_values[key][row_idx] = bool(value)

                np.save(fpath, raster)
                raster_debug_paths[row_idx] = str(fpath)
                if flags["branches_connected"]:
                    raster_paths[row_idx] = str(fpath)
                    raster_status[row_idx] = "ok"
                else:
                    raster_status[row_idx] = "invalid"
                    failed_flags = [key for key, value in flags.items() if not value]
                    raster_errors[row_idx] = "qa_failed:" + ",".join(failed_flags)
            except Exception as exc:
                raster_status[row_idx] = "failed"
                raster_errors[row_idx] = f"{type(exc).__name__}: {exc}"
                logger.exception(
                    "Failed to rasterize %s outlet=%d h1=%d h2=%d",
                    basin_name,
                    outlet,
                    head_1,
                    head_2,
                )

    df["raster_path"] = raster_paths
    df["raster_debug_path"] = raster_debug_paths
    df["raster_status"] = raster_status
    df["raster_error"] = raster_errors
    for key, values in qa_values.items():
        df[key] = values
    return df
