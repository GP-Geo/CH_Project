"""Polyline sampling/direction helpers for the Mars pair-feature pipeline.

These operate on already-projected (metre) polyline coordinate arrays — the
chained head→confluence branch geometries — to produce the inputs the
dimensionless pair features need: a "first N metres" direction vector
(orientation / apex angle) and arc-length-sampled coordinates (proximity
profile).

They are ports of the Earth ``geometric_analysis.py`` path helpers, adapted to
operate directly on Mars CRS coordinates (projected metres, so
``meters_per_unit = 1``). They were duplicated inside
``scripts/build_mars_pair_features_5feat.py``; the canonical Mars-coords
versions now live here so that script and the ``notebooks/mars/`` workflow call
the same implementation. The Earth versions in ``geometric_analysis.py`` (which
carry the unit conversion) are intentionally left separate.
"""

from __future__ import annotations

import math

import numpy as np

EPSILON = 1e-10
DIRECTION_SAMPLE_DISTANCE_M = 500.0
MIN_EDGES_FOR_DIRECTION = 3
N_PROXIMITY_SAMPLES = 10


def line_direction_first_n_meters(
    coords: np.ndarray, max_distance_m: float
) -> tuple[tuple[float, float] | None, list[str]]:
    """Compute weighted-average direction of the first ``max_distance_m``
    of a polyline.

    Port of geometric_analysis.py:_trace_path_downstream +
    _compute_direction_vector, adapted to operate on the chained LineString
    coords (Mars CRS is already projected metres, so meters_per_unit = 1).

    Each consecutive coord pair in the polyline is one "edge" of length
    `hypot(dx, dy)` in metres. Edges are accumulated, weighted by their
    length, until cumulative length >= max_distance_m. Matches Earth's
    "include the edge that crosses the threshold" behaviour.
    """
    qc: list[str] = []
    n = len(coords)
    if n < 2:
        return None, ["single_edge"]
    total_dx = 0.0
    total_dy = 0.0
    total_w = 0.0
    accumulated = 0.0
    n_edges_used = 0
    for i in range(n - 1):
        dx = float(coords[i + 1, 0] - coords[i, 0])
        dy = float(coords[i + 1, 1] - coords[i, 1])
        w = math.hypot(dx, dy)
        if w > EPSILON:
            total_dx += dx * w
            total_dy += dy * w
            total_w += w
            n_edges_used += 1
        accumulated += w
        if accumulated >= max_distance_m:
            break
    if n_edges_used < MIN_EDGES_FOR_DIRECTION:
        qc.append("short_path")
    if total_w < EPSILON:
        return None, qc
    avg_dx = total_dx / total_w
    avg_dy = total_dy / total_w
    mag = math.hypot(avg_dx, avg_dy)
    if mag < EPSILON:
        return None, qc
    return (avg_dx / mag, avg_dy / mag), qc


def sample_path_coords_along_line(coords: np.ndarray, n_samples: int) -> np.ndarray | None:
    """Port of geometric_analysis.py:_sample_path_coords.

    Samples at arc-length fractions [0, 1/n, ..., (n-1)/n] (excludes 1.0).
    """
    if len(coords) < 2:
        return None
    diffs = np.hypot(np.diff(coords[:, 0]), np.diff(coords[:, 1]))
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    total = float(cum[-1])
    if total < EPSILON:
        return None
    norm = cum / total
    targets = np.arange(n_samples) / n_samples
    idxs = np.searchsorted(norm, targets, side="right") - 1
    idxs = np.clip(idxs, 0, len(coords) - 2)
    lo = norm[idxs]
    hi = norm[idxs + 1]
    seg_len = hi - lo
    frac = np.where(seg_len > EPSILON, (targets - lo) / seg_len, 0.0)
    sx = coords[idxs, 0] + frac * (coords[idxs + 1, 0] - coords[idxs, 0])
    sy = coords[idxs, 1] + frac * (coords[idxs + 1, 1] - coords[idxs, 1])
    return np.column_stack([sx, sy])
