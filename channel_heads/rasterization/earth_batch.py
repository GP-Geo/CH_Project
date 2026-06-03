"""Earth batch raster precomputation for CNN patch datasets."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
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

# A basin is split across worker processes only if it has at least this many
# pairs; smaller basins run as a single unit (one stream build) and rely on
# basin-level concurrency instead. Keeps redundant StreamObject rebuilds off
# the long tail of small basins while still parallelizing the dense ones.
MIN_ROWS_PER_CHUNK = 250


def _rasterize_one_pair(
    row_idx: int,
    row: Any,
    s: Any,
    grid_shape: tuple[int, int],
    basin_raster_dir: Path,
    target_size: int,
    rasterize_func: RasterizeFunc,
    quality_func: QualityFunc,
    basin_name: str,
    log: Any,
) -> dict[str, Any]:
    """Rasterize a single outlet pair, save the .npy, and return its result.

    Pure per-pair work: it only reads ``s`` and writes a distinct ``.npy``
    path. ``row`` may be a pandas Series or a plain dict (the parallel path
    passes dicts so rows are picklable across processes). The caller applies
    the returned record into the output lists by ``row_idx`` (preserving input
    order regardless of worker count).
    """
    outlet = int(row["outlet"])
    head_1 = int(row["head_1"])
    head_2 = int(row["head_2"])
    confluence = int(row["confluence"])

    fname = f"{outlet}_{head_1}_{head_2}.npy"
    fpath = basin_raster_dir / fname

    result: dict[str, Any] = {
        "row_idx": row_idx,
        "raster_path": None,
        "raster_debug_path": None,
        "raster_status": "pending",
        "raster_error": "",
        "flags": None,
    }
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
        np.save(fpath, raster)
        result["raster_debug_path"] = str(fpath)
        result["flags"] = flags
        if flags["branches_connected"]:
            result["raster_path"] = str(fpath)
            result["raster_status"] = "ok"
        else:
            result["raster_status"] = "invalid"
            failed_flags = [key for key, value in flags.items() if not value]
            result["raster_error"] = "qa_failed:" + ",".join(failed_flags)
    except Exception as exc:  # noqa: BLE001 — one bad pair shouldn't kill the basin
        result["raster_status"] = "failed"
        result["raster_error"] = f"{type(exc).__name__}: {exc}"
        log.exception(
            "Failed to rasterize %s outlet=%d h1=%d h2=%d",
            basin_name,
            outlet,
            head_1,
            head_2,
        )
    return result


def _skip_records(records, status, error):
    return [
        {
            "row_idx": row_idx,
            "raster_path": None,
            "raster_debug_path": None,
            "raster_status": status,
            "raster_error": error,
            "flags": None,
        }
        for row_idx, _row in records
    ]


def _rasterize_basin_unit(unit: tuple) -> list[dict[str, Any]]:
    """Process one (basin, row-chunk) unit in a worker process.

    Builds the basin StreamObject once for this unit, then rasterizes its
    chunk of pairs serially. Returns one result record per row (keyed by the
    original DataFrame index). Replicates the serial path's skip semantics for
    missing basin config / DEM so output is identical regardless of worker
    count. Module-level (not a closure) so it is importable under ``spawn``.
    """
    from channel_heads.basin_config import get_basin_config

    (
        basin_name,
        records,
        dem_loader,
        output_dir,
        target_size,
        threshold,
        rasterize_func,
        quality_func,
    ) = unit
    log = logger

    try:
        config = get_basin_config(basin_name)
    except KeyError:
        log.warning("No config for basin %s, skipping", basin_name)
        return _skip_records(records, "skipped", "missing_basin_config")

    loaded = dem_loader(basin_name, config["lat"], config["z_th"], threshold)
    if loaded is None:
        log.warning("DEM not found for basin %s, skipping", basin_name)
        return _skip_records(records, "skipped", "missing_dem_or_stream")

    s, dem = loaded
    grid_shape = dem.shape if hasattr(dem, "shape") else dem.z.shape
    basin_raster_dir = Path(output_dir) / basin_name / "rasters"
    basin_raster_dir.mkdir(parents=True, exist_ok=True)

    return [
        _rasterize_one_pair(
            row_idx, row, s, grid_shape, basin_raster_dir, target_size,
            rasterize_func, quality_func, basin_name, log,
        )
        for row_idx, row in records
    ]


def _build_units(df, dem_loader, output_dir, target_size, threshold,
                 rasterize_func, quality_func, workers):
    """Split the dataset into (basin, row-chunk) work units for the pool."""
    units = []
    for basin_name, basin_df in df.groupby("basin"):
        basin_name = str(basin_name)
        records = [(int(idx), row.to_dict()) for idx, row in basin_df.iterrows()]
        n_chunks = max(1, min(workers, -(-len(records) // MIN_ROWS_PER_CHUNK)))
        # Contiguous, near-equal chunks (order is irrelevant; results are
        # reassembled by row index).
        bounds = np.linspace(0, len(records), n_chunks + 1, dtype=int)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi > lo:
                units.append((
                    basin_name, records[lo:hi], dem_loader, output_dir,
                    target_size, threshold, rasterize_func, quality_func,
                ))
    return units


def _precompute_raster_dataset(
    master_csv: Path,
    output_dir: Path,
    dem_loader: Callable[[str, float, float, int], tuple[Any, Any] | None],
    target_size: int = 128,
    threshold: int = 300,
    *,
    n_workers: int = 1,
    rasterize_func: RasterizeFunc = rasterize_outlet_pair,
    quality_func: QualityFunc = raster_quality_flags,
    log: Any | None = None,
) -> pd.DataFrame:
    """Pre-render raster patches with injectable helpers for legacy shims.

    With ``n_workers == 1`` the per-basin loop runs serially in-process
    (bit-identical to the original implementation). With ``n_workers > 1`` the
    work is split into ``(basin, row-chunk)`` units and rendered across a
    ``ProcessPoolExecutor`` (``spawn`` context) — separate processes sidestep
    the GIL, so the CPU-bound rasterization actually scales with cores. Dense
    basins are chunked so they use multiple workers; output (patches + manifest
    rows, keyed by input index) is identical regardless of worker count.

    Parallel mode requires ``dem_loader`` / ``rasterize_func`` / ``quality_func``
    to be picklable (module-level functions or picklable callables — e.g.
    ``training.regime.RegimeStreamLoader``). Each chunk of a split basin rebuilds
    that basin's StreamObject, so peak memory scales with the number of workers
    concurrently holding a large basin.
    """
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

    def _apply(result: dict[str, Any]) -> None:
        row_idx = result["row_idx"]
        raster_paths[row_idx] = result["raster_path"]
        raster_debug_paths[row_idx] = result["raster_debug_path"]
        raster_status[row_idx] = result["raster_status"]
        raster_errors[row_idx] = result["raster_error"]
        flags = result["flags"]
        if flags is not None:
            for key, value in flags.items():
                qa_values[key][row_idx] = bool(value)

    workers = max(1, int(n_workers))

    if workers > 1:
        units = _build_units(
            df, dem_loader, output_dir, target_size, threshold,
            rasterize_func, quality_func, workers,
        )
        log.info(
            "Rasterizing %d basins as %d units across %d worker processes",
            df["basin"].nunique(),
            len(units),
            workers,
        )
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            for unit_results in executor.map(_rasterize_basin_unit, units):
                for result in unit_results:
                    _apply(result)
    else:
        for basin_name, basin_df in df.groupby("basin"):
            basin_name = str(basin_name)
            log.info("Rasterizing basin: %s (%d pairs)", basin_name, len(basin_df))

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
            basin_raster_dir = output_dir / basin_name / "rasters"
            basin_raster_dir.mkdir(parents=True, exist_ok=True)

            for row_idx, row in basin_df.iterrows():
                _apply(
                    _rasterize_one_pair(
                        row_idx, row, s, grid_shape, basin_raster_dir,
                        target_size, rasterize_func, quality_func, basin_name, log,
                    )
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
    *,
    n_workers: int = 1,
    rasterize_func: RasterizeFunc = rasterize_outlet_pair,
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
        ``features.earth_enrichment.default_stream_loader``. Must be picklable
        when ``n_workers > 1`` (use a module-level function or a picklable
        callable such as ``training.regime.RegimeStreamLoader``).
    target_size : int
        Output image size.
    threshold : int
        Stream network threshold parameter.
    n_workers : int
        Number of worker *processes* used to rasterize pairs. ``1`` (default)
        runs serially in-process and is bit-identical to the original behavior;
        ``> 1`` splits the work into (basin, chunk) units and renders them
        across a process pool (sidesteps the GIL).
    rasterize_func : callable
        Rasterization function to use (injectable for testing).

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
        n_workers=n_workers,
        rasterize_func=rasterize_func,
    )


__all__ = ["precompute_raster_dataset"]
