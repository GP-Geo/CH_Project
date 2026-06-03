"""Shared raster drawing + coordinate primitives.

Low-level helpers used by the Mars patch rasterizer
(:mod:`channel_heads.rasterization.mars_patches`): Bresenham lines (from the
Earth patch implementation), projected-metres↔image-coord conversion,
confluence-centred rotation, and polyline drawing with branch protection.
"""

from __future__ import annotations

import math

import numpy as np

from channel_heads.rasterization.earth_patches import bresenham_line


def xy_to_rc(x_m: float, y_m: float, cell_size_m: float) -> tuple[float, float]:
    """Projected metres → (row, col) image coords. Row increases downward."""
    return (-y_m / cell_size_m, x_m / cell_size_m)


def linestring_to_rc(line, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """LineString coords → (rows, cols) arrays in image coords."""
    coords = np.asarray(line.coords, dtype=float)
    rs = -coords[:, 1] / cell_size_m
    cs = coords[:, 0] / cell_size_m
    return rs, cs


def compute_rotation_angle(
    h1_rc: tuple[float, float],
    h2_rc: tuple[float, float],
    conf_rc: tuple[float, float],
) -> float:
    """Angle that puts the head-midpoint above the confluence (image up).

    Port of ``channel_heads.rasterizer._compute_rotation_angle``.
    """
    mid_r = (h1_rc[0] + h2_rc[0]) / 2.0
    mid_c = (h1_rc[1] + h2_rc[1]) / 2.0
    dr = mid_r - conf_rc[0]
    dc = mid_c - conf_rc[1]
    if abs(dr) < 1e-9 and abs(dc) < 1e-9:
        return 0.0
    current_angle = math.atan2(dc, dr)
    target_angle = math.pi  # pointing in -row direction (image up)
    return target_angle - current_angle


def rotate_rc(
    rs: np.ndarray,
    cs: np.ndarray,
    cr: float,
    cc: float,
    angle_rad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate (rs, cs) about centre (cr, cc). Port of ``_rotate_coordinates``."""
    dr = rs - cr
    dc = cs - cc
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    new_r = dr * cos_a - dc * sin_a + cr
    new_c = dr * sin_a + dc * cos_a + cc
    return new_r, new_c


def draw_polyline(
    raster: np.ndarray,
    rs: np.ndarray,
    cs: np.ndarray,
    r_min: float,
    c_min: float,
    value: int,
    protect: tuple[int, ...] = (),
) -> None:
    """Draw a polyline onto ``raster`` via per-segment Bresenham lines.

    Vertex pixels are always written; interior pixels are protected from
    overwriting any value listed in ``protect`` (the branch-A vs branch-B
    protection mirroring Earth's ``_draw_path_on_raster``).
    """
    H, W = raster.shape
    if len(rs) < 2:
        ir = int(round(rs[0] - r_min))
        ic = int(round(cs[0] - c_min))
        if 0 <= ir < H and 0 <= ic < W:
            raster[ir, ic] = value
        return

    irs = np.rint(rs - r_min).astype(int)
    ics = np.rint(cs - c_min).astype(int)
    vertex_set: set[tuple[int, int]] = set()
    for ir, ic in zip(irs, ics):
        if 0 <= ir < H and 0 <= ic < W:
            vertex_set.add((int(ir), int(ic)))

    for i in range(len(rs) - 1):
        r0, c0 = int(irs[i]), int(ics[i])
        r1, c1 = int(irs[i + 1]), int(ics[i + 1])
        for lr, lc in bresenham_line(r0, c0, r1, c1):
            if 0 <= lr < H and 0 <= lc < W:
                is_vertex = (lr, lc) in vertex_set
                if is_vertex or raster[lr, lc] not in protect:
                    raster[lr, lc] = value


__all__ = [
    "bresenham_line",
    "xy_to_rc",
    "linestring_to_rc",
    "compute_rotation_angle",
    "rotate_rc",
    "draw_polyline",
]
