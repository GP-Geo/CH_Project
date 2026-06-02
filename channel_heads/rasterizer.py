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

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from .first_meet_pairs_for_outlet import (
    _build_children_from_parents,
    _build_parents_from_stream,
    _collect_basin_nodes_from_outlet,
)
from .geometric_analysis import _trace_full_path
from .logging_config import get_logger

logger = get_logger(__name__)

# Raster class values
BACKGROUND = 0
BRANCH_A = 1
BRANCH_B = 2
OTHER_STREAMS = 3
CONFLUENCE_MARKER = 4

NUM_CLASSES = 5


# =============================================================================
# Coordinate Helpers
# =============================================================================


def bresenham_line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Bresenham's line algorithm for 8-connected pixel paths.

    Returns all pixel coordinates (r, c) on the line from (r0, c0) to (r1, c1),
    inclusive of both endpoints.
    """
    pixels = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr

    while True:
        pixels.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 >= -dr:
            err -= dr
            c0 += sc
        if e2 <= dc:
            err += dc
            r0 += sr
    return pixels


def _project_to_target_grid(
    rot_r: npt.NDArray[np.float64],
    rot_c: npt.NDArray[np.float64],
    r_min: float,
    r_max: float,
    c_min: float,
    c_max: float,
    target_size: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Project rotated crop coordinates directly into the final raster grid."""
    r_span = max(float(r_max - r_min), 1e-9)
    c_span = max(float(c_max - c_min), 1e-9)
    scale = float(target_size - 1)
    out_r = (rot_r - r_min) / r_span * scale
    out_c = (rot_c - c_min) / c_span * scale
    return out_r, out_c


def _draw_path_on_target_grid(
    raster: npt.NDArray[np.uint8],
    path_nodes: list[int],
    node_to_idx: dict[int, int],
    out_r: npt.NDArray[np.float64],
    out_c: npt.NDArray[np.float64],
    value: int,
    protect: tuple[int, ...] = (),
) -> None:
    """Draw a connected path after projection into the final output grid."""
    h, w = raster.shape
    prev_ir, prev_ic = -1, -1

    node_pixels: set[tuple[int, int]] = set()
    node_positions: list[tuple[int, int]] = []
    for node_id in path_nodes:
        idx = node_to_idx.get(int(node_id))
        if idx is None:
            node_positions.append((-1, -1))
            continue
        ir = int(round(float(out_r[idx])))
        ic = int(round(float(out_c[idx])))
        node_positions.append((ir, ic))
        if 0 <= ir < h and 0 <= ic < w:
            node_pixels.add((ir, ic))

    for ir, ic in node_positions:
        if ir < 0:
            continue
        if prev_ir >= 0:
            for lr, lc in bresenham_line(prev_ir, prev_ic, ir, ic):
                if 0 <= lr < h and 0 <= lc < w:
                    is_node = (lr, lc) in node_pixels
                    if is_node or raster[lr, lc] not in protect:
                        raster[lr, lc] = value
        else:
            if 0 <= ir < h and 0 <= ic < w:
                raster[ir, ic] = value
        prev_ir, prev_ic = ir, ic


def _draw_edges_on_target_grid(
    raster: npt.NDArray[np.uint8],
    parents: list[list[int]],
    basin_node_set: set[int],
    node_to_idx: dict[int, int],
    out_r: npt.NDArray[np.float64],
    out_c: npt.NDArray[np.float64],
    value: int,
) -> None:
    """Draw all basin edges after projection into the final output grid."""
    h, w = raster.shape
    for node_id in basin_node_set:
        idx_n = node_to_idx.get(int(node_id))
        if idx_n is None:
            continue
        ir0 = int(round(float(out_r[idx_n])))
        ic0 = int(round(float(out_c[idx_n])))

        for parent in parents[node_id]:
            idx_p = node_to_idx.get(int(parent))
            if idx_p is None:
                continue
            ir1 = int(round(float(out_r[idx_p])))
            ic1 = int(round(float(out_c[idx_p])))

            for lr, lc in bresenham_line(ir0, ic0, ir1, ic1):
                if 0 <= lr < h and 0 <= lc < w:
                    raster[lr, lc] = value


def _component_count(mask: npt.NDArray[np.bool_]) -> int:
    """Count 8-connected components in a small binary mask."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return 0

    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    n_components = 0
    neighbors = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for r_raw, c_raw in coords:
        start = (int(r_raw), int(c_raw))
        if start in seen:
            continue
        n_components += 1
        stack = [start]
        seen.add(start)
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return n_components


def raster_quality_flags(raster: npt.NDArray[np.uint8]) -> dict[str, bool]:
    """Return core structural QA flags for a 5-class stream patch."""
    has_branch_a = bool(np.any(raster == BRANCH_A))
    has_branch_b = bool(np.any(raster == BRANCH_B))
    has_confluence = bool(np.any(raster == CONFLUENCE_MARKER))
    branch_a_connected = (
        has_branch_a
        and has_confluence
        and _component_count((raster == BRANCH_A) | (raster == CONFLUENCE_MARKER)) == 1
    )
    branch_b_connected = (
        has_branch_b
        and has_confluence
        and _component_count((raster == BRANCH_B) | (raster == CONFLUENCE_MARKER)) == 1
    )
    return {
        "has_branch_a": has_branch_a,
        "has_branch_b": has_branch_b,
        "has_confluence": has_confluence,
        "branch_a_connected": branch_a_connected,
        "branch_b_connected": branch_b_connected,
        "branches_connected": branch_a_connected and branch_b_connected,
    }


def _get_rc(
    stream_obj: Any,
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """Extract row and column arrays from StreamObject.node_indices."""
    ni = stream_obj.node_indices
    if callable(ni):
        r, c = ni()
    else:
        r, c = ni
    return np.asarray(r), np.asarray(c)


def _compute_rotation_angle(
    head_1_rc: tuple[float, float],
    head_2_rc: tuple[float, float],
    confluence_rc: tuple[float, float],
) -> float:
    """Compute rotation angle so confluence is at bottom, heads midpoint at top.

    In raster coordinates, row increases downward. We want the confluence at the
    bottom (high row) and the heads midpoint at the top (low row). The "upward"
    direction in raster space is negative row direction.

    Parameters
    ----------
    head_1_rc : tuple[float, float]
        (row, col) of head 1.
    head_2_rc : tuple[float, float]
        (row, col) of head 2.
    confluence_rc : tuple[float, float]
        (row, col) of the confluence.

    Returns
    -------
    float
        Rotation angle in radians.
    """
    mid_r = (head_1_rc[0] + head_2_rc[0]) / 2.0
    mid_c = (head_1_rc[1] + head_2_rc[1]) / 2.0

    # Vector from confluence to heads midpoint
    dr = mid_r - confluence_rc[0]
    dc = mid_c - confluence_rc[1]

    # Degenerate case: heads midpoint coincides with confluence
    if abs(dr) < 1e-9 and abs(dc) < 1e-9:
        return 0.0

    # We want this vector to point "upward" in raster space (negative row).
    # The target direction is (-1, 0) in (row, col) space.
    # angle = atan2(dc, dr) gives the current angle of the vector.
    # Target angle for "up" = atan2(0, -1) = pi.
    current_angle = math.atan2(dc, dr)
    target_angle = math.pi  # pointing in -row direction

    return target_angle - current_angle


def _rotate_coordinates(
    rows: npt.NDArray[np.float64],
    cols: npt.NDArray[np.float64],
    center_r: float,
    center_c: float,
    angle_rad: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rotate (row, col) coordinates around a center point.

    Parameters
    ----------
    rows, cols : np.ndarray
        Coordinate arrays.
    center_r, center_c : float
        Center of rotation.
    angle_rad : float
        Rotation angle in radians (counterclockwise).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Rotated (rows, cols).
    """
    dr = rows - center_r
    dc = cols - center_c
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_r = dr * cos_a - dc * sin_a + center_r
    new_c = dr * sin_a + dc * cos_a + center_c
    return new_r, new_c


# =============================================================================
# Core Rasterization
# =============================================================================


def rasterize_outlet_pair(
    s: Any,
    outlet: int,
    head_1: int,
    head_2: int,
    confluence: int,
    grid_shape: tuple[int, int],
    target_size: int = 128,
    padding_frac: float = 0.2,
) -> np.ndarray:
    """Rasterize an outlet's stream network with a specific pair highlighted.

    Creates a canonically aligned, fixed-size raster image encoding the stream
    network structure around a channel head pair.

    Parameters
    ----------
    s : StreamObject
        Stream network object.
    outlet : int
        Outlet node ID.
    head_1, head_2 : int
        Channel head node IDs of the pair.
    confluence : int
        Confluence node ID where the pair meets.
    grid_shape : tuple[int, int]
        Shape of the underlying DEM grid (rows, cols).
    target_size : int
        Output image size (square). Default 128.
    padding_frac : float
        Fractional padding around the bounding box of the pair paths.
        Default 0.2 (20% on each side).

    Returns
    -------
    np.ndarray
        Array of shape (target_size, target_size) with dtype uint8.
        Values: 0=background, 1=branch A, 2=branch B, 3=other streams,
        4=confluence marker.
    """
    r_nodes, c_nodes = _get_rc(s)

    # --- Step 1: Build graph and collect outlet basin nodes ---
    parents = _build_parents_from_stream(s)
    basin_node_list = _collect_basin_nodes_from_outlet(parents, outlet)
    basin_node_set = set(basin_node_list)
    children = _build_children_from_parents(parents, basin_node_set)

    # --- Step 2: Trace paths from each head to confluence ---
    path_a = _trace_full_path(head_1, confluence, children)
    path_b = _trace_full_path(head_2, confluence, children)

    path_a_set = set(path_a)
    path_b_set = set(path_b)

    # --- Step 3: Compute rotation angle ---
    conf_r, conf_c = float(r_nodes[confluence]), float(c_nodes[confluence])
    h1_r, h1_c = float(r_nodes[head_1]), float(c_nodes[head_1])
    h2_r, h2_c = float(r_nodes[head_2]), float(c_nodes[head_2])

    angle = _compute_rotation_angle(
        head_1_rc=(h1_r, h1_c),
        head_2_rc=(h2_r, h2_c),
        confluence_rc=(conf_r, conf_c),
    )

    # --- Step 4: Rotate all outlet stream node coordinates ---
    # Get coordinates of basin stream nodes only
    basin_node_arr = np.array(basin_node_list, dtype=np.intp)
    basin_r = r_nodes[basin_node_arr].astype(np.float64)
    basin_c = c_nodes[basin_node_arr].astype(np.float64)

    rot_r, rot_c = _rotate_coordinates(basin_r, basin_c, conf_r, conf_c, angle)

    # --- Step 5: Compute bounding box of pair paths + padding ---
    # Find rotated coords of path A and path B nodes
    path_ab_nodes = np.array(sorted(path_a_set | path_b_set), dtype=np.intp)
    path_r = r_nodes[path_ab_nodes].astype(np.float64)
    path_c = c_nodes[path_ab_nodes].astype(np.float64)
    path_rot_r, path_rot_c = _rotate_coordinates(path_r, path_c, conf_r, conf_c, angle)

    r_min, r_max = float(path_rot_r.min()), float(path_rot_r.max())
    c_min, c_max = float(path_rot_c.min()), float(path_rot_c.max())

    # Add padding
    r_span = r_max - r_min
    c_span = c_max - c_min
    # Ensure minimum span of 2 pixels to avoid degenerate crops
    r_span = max(r_span, 2.0)
    c_span = max(c_span, 2.0)

    pad_r = r_span * padding_frac
    pad_c = c_span * padding_frac
    r_min -= pad_r
    r_max += pad_r
    c_min -= pad_c
    c_max += pad_c

    # --- Step 6: Rasterize directly onto the final target grid ---
    # Previous versions drew into a native-size temporary crop and resized to
    # target_size. Nearest-neighbour downsampling can remove one-pixel markers
    # or disconnect thin branches. Projecting the rotated coordinates into the
    # final grid first makes connectivity a construction invariant.
    raster = np.zeros((target_size, target_size), dtype=np.uint8)

    # Build a mapping from node ID to its index in basin_node_arr
    node_to_idx = {int(n): i for i, n in enumerate(basin_node_arr)}
    out_r, out_c = _project_to_target_grid(rot_r, rot_c, r_min, r_max, c_min, c_max, target_size)

    # Draw all basin edges as OTHER_STREAMS (value 3) with connected lines
    _draw_edges_on_target_grid(
        raster,
        parents,
        basin_node_set,
        node_to_idx,
        out_r,
        out_c,
        OTHER_STREAMS,
    )

    # Overwrite branch A with connected path lines (value 1).
    # Protect: don't let A's interpolation overwrite B pixels (and vice versa).
    _draw_path_on_target_grid(
        raster,
        path_a,
        node_to_idx,
        out_r,
        out_c,
        BRANCH_A,
        protect=(BRANCH_B,),
    )

    # Overwrite branch B with connected path lines (value 2)
    _draw_path_on_target_grid(
        raster,
        path_b,
        node_to_idx,
        out_r,
        out_c,
        BRANCH_B,
        protect=(BRANCH_A,),
    )

    # Place confluence marker (value 4) — overwrites any branch value
    conf_idx = node_to_idx.get(confluence)
    if conf_idx is not None:
        ir = int(round(float(out_r[conf_idx])))
        ic = int(round(float(out_c[conf_idx])))
        if 0 <= ir < target_size and 0 <= ic < target_size:
            raster[ir, ic] = CONFLUENCE_MARKER

    return raster


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
