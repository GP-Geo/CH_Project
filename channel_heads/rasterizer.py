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
then cropped and resized to a fixed size.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from skimage.transform import resize as skimage_resize

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


def _bresenham_line(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
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


def _draw_path_on_raster(
    raster: npt.NDArray[np.uint8],
    path_nodes: list[int],
    node_to_idx: dict[int, int],
    rot_r: npt.NDArray[np.float64],
    rot_c: npt.NDArray[np.float64],
    r_min: float,
    c_min: float,
    value: int,
    protect: tuple[int, ...] = (),
) -> None:
    """Draw a connected path onto the raster using Bresenham lines.

    Draws lines between consecutive nodes along the path to ensure
    8-connectivity is preserved after rotation.

    Parameters
    ----------
    protect : tuple[int, ...]
        Raster values that Bresenham interpolation pixels should NOT
        overwrite. Actual path node pixels are always placed regardless.
        This prevents one branch's interpolation from cutting through
        another branch's line.
    """
    temp_h, temp_w = raster.shape
    prev_ir, prev_ic = -1, -1

    # Collect actual node pixel positions so we can force them
    node_pixels: set[tuple[int, int]] = set()
    node_positions: list[tuple[int, int]] = []
    for node_id in path_nodes:
        if node_id not in node_to_idx:
            node_positions.append((-1, -1))
            continue
        idx = node_to_idx[node_id]
        pr = rot_r[idx] - r_min
        pc = rot_c[idx] - c_min
        ir, ic = int(round(pr)), int(round(pc))
        node_positions.append((ir, ic))
        if 0 <= ir < temp_h and 0 <= ic < temp_w:
            node_pixels.add((ir, ic))

    # Draw Bresenham lines between consecutive nodes
    for ir, ic in node_positions:
        if ir < 0:
            continue

        if prev_ir >= 0:
            for lr, lc in _bresenham_line(prev_ir, prev_ic, ir, ic):
                if 0 <= lr < temp_h and 0 <= lc < temp_w:
                    is_node = (lr, lc) in node_pixels
                    if is_node or raster[lr, lc] not in protect:
                        raster[lr, lc] = value
        else:
            if 0 <= ir < temp_h and 0 <= ic < temp_w:
                raster[ir, ic] = value

        prev_ir, prev_ic = ir, ic


def _draw_edges_on_raster(
    raster: npt.NDArray[np.uint8],
    parents: list[list[int]],
    basin_node_set: set[int],
    node_to_idx: dict[int, int],
    rot_r: npt.NDArray[np.float64],
    rot_c: npt.NDArray[np.float64],
    r_min: float,
    c_min: float,
    value: int,
) -> None:
    """Draw all parent-child edges in the basin using Bresenham lines.

    For each node, draws lines to all its upstream parents (both must be
    in the basin) to ensure the stream network is 8-connected.

    Parameters
    ----------
    parents : list[list[int]]
        parents[v] = list of upstream nodes flowing into v.
    """
    temp_h, temp_w = raster.shape

    for node_id in basin_node_set:
        if node_id not in node_to_idx:
            continue
        idx_n = node_to_idx[node_id]
        ir0 = int(round(rot_r[idx_n] - r_min))
        ic0 = int(round(rot_c[idx_n] - c_min))

        for parent in parents[node_id]:
            if parent not in node_to_idx:
                continue
            idx_p = node_to_idx[parent]
            ir1 = int(round(rot_r[idx_p] - r_min))
            ic1 = int(round(rot_c[idx_p] - c_min))

            for lr, lc in _bresenham_line(ir0, ic0, ir1, ic1):
                if 0 <= lr < temp_h and 0 <= lc < temp_w:
                    raster[lr, lc] = value


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

    # --- Step 6: Rasterize onto temporary grid ---
    # Map rotated coordinates to pixel space in the temporary grid.
    # We draw Bresenham lines between consecutive/adjacent nodes to
    # ensure 8-connectivity is preserved after rotation.
    temp_h = max(int(np.ceil(r_max - r_min)) + 1, 1)
    temp_w = max(int(np.ceil(c_max - c_min)) + 1, 1)

    raster = np.zeros((temp_h, temp_w), dtype=np.uint8)

    # Build a mapping from node ID to its index in basin_node_arr
    node_to_idx = {int(n): i for i, n in enumerate(basin_node_arr)}

    # Draw all basin edges as OTHER_STREAMS (value 3) with connected lines
    _draw_edges_on_raster(
        raster,
        parents,
        basin_node_set,
        node_to_idx,
        rot_r,
        rot_c,
        r_min,
        c_min,
        OTHER_STREAMS,
    )

    # Overwrite branch A with connected path lines (value 1).
    # Protect: don't let A's interpolation overwrite B pixels (and vice versa).
    _draw_path_on_raster(
        raster,
        path_a,
        node_to_idx,
        rot_r,
        rot_c,
        r_min,
        c_min,
        BRANCH_A,
        protect=(BRANCH_B,),
    )

    # Overwrite branch B with connected path lines (value 2)
    _draw_path_on_raster(
        raster,
        path_b,
        node_to_idx,
        rot_r,
        rot_c,
        r_min,
        c_min,
        BRANCH_B,
        protect=(BRANCH_A,),
    )

    # Place confluence marker (value 4) — overwrites any branch value
    conf_idx = node_to_idx.get(confluence)
    if conf_idx is not None:
        pr = rot_r[conf_idx] - r_min
        pc = rot_c[conf_idx] - c_min
        ir, ic = int(round(pr)), int(round(pc))
        if 0 <= ir < temp_h and 0 <= ic < temp_w:
            raster[ir, ic] = CONFLUENCE_MARKER

    # --- Step 7: Resize to target size ---
    if raster.shape[0] == target_size and raster.shape[1] == target_size:
        return raster

    resized = skimage_resize(
        raster,
        (target_size, target_size),
        order=0,  # nearest-neighbor
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized.astype(np.uint8)


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

    for basin_name, basin_df in df.groupby("basin"):
        basin_name = str(basin_name)
        logger.info("Rasterizing basin: %s (%d pairs)", basin_name, len(basin_df))

        # Load stream network
        try:
            config = get_basin_config(basin_name)
        except KeyError:
            logger.warning("No config for basin %s, skipping", basin_name)
            continue

        result = dem_loader(basin_name, config["lat"], config["z_th"], threshold)
        if result is None:
            logger.warning("DEM not found for basin %s, skipping", basin_name)
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
                np.save(fpath, raster)
                raster_paths[row_idx] = str(fpath)
            except Exception:
                logger.exception(
                    "Failed to rasterize %s outlet=%d h1=%d h2=%d",
                    basin_name,
                    outlet,
                    head_1,
                    head_2,
                )

    df["raster_path"] = raster_paths
    return df
