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

import pandas as pd

from .features.earth_paths import _trace_full_path  # noqa: F401  (compat re-export)
from .logging_config import get_logger
from .rasterization.earth_batch import _precompute_raster_dataset
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
    return _precompute_raster_dataset(
        master_csv=master_csv,
        output_dir=output_dir,
        dem_loader=dem_loader,
        target_size=target_size,
        threshold=threshold,
        rasterize_func=rasterize_outlet_pair,
        quality_func=raster_quality_flags,
        log=logger,
    )
