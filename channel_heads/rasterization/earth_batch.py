"""Earth batch raster precomputation for CNN patch datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from channel_heads.logging_config import get_logger
from channel_heads.rasterization.earth_patches import (
    raster_quality_flags,
    rasterize_outlet_pair,
)

logger = get_logger(__name__)

RasterizeFunc = Callable[..., npt.NDArray[np.uint8]]
QualityFunc = Callable[[npt.NDArray[np.uint8]], dict[str, bool]]


def _precompute_raster_dataset(
    master_csv: Path,
    output_dir: Path,
    dem_loader: Callable[[str, float, float, int], tuple[Any, Any] | None],
    target_size: int = 128,
    threshold: int = 300,
    *,
    rasterize_func: RasterizeFunc = rasterize_outlet_pair,
    quality_func: QualityFunc = raster_quality_flags,
    log: Any | None = None,
) -> pd.DataFrame:
    """Pre-render raster patches with injectable helpers for legacy shims."""
    from channel_heads.basin_config import get_basin_config

    log = logger if log is None else log

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
        log.info("Rasterizing basin: %s (%d pairs)", basin_name, len(basin_df))

        # Load stream network
        try:
            config = get_basin_config(basin_name)
        except KeyError:
            log.warning("No config for basin %s, skipping", basin_name)
            for row_idx in basin_df.index:
                raster_status[row_idx] = "skipped"
                raster_errors[row_idx] = "missing_basin_config"
            continue

        result = dem_loader(basin_name, config["lat"], config["z_th"], threshold)
        if result is None:
            log.warning("DEM not found for basin %s, skipping", basin_name)
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
                raster = rasterize_func(
                    s,
                    outlet,
                    head_1,
                    head_2,
                    confluence,
                    grid_shape,
                    target_size=target_size,
                )
                flags = quality_func(raster)
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
                log.exception(
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
    )


__all__ = ["precompute_raster_dataset"]
