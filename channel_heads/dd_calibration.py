"""Stream-extraction threshold calibration against Martian drainage density.

This module supports choosing a TopoToolbox contributing-area threshold so the
terrestrial extracted-stream networks become directly comparable to mapped
Martian valley networks.

Key methodological point
------------------------
The Martian "drainage density" used here is a *mapped* proxy:

    Dd_hull = mapped_valley_length / convex_hull_area(mapped_valley_network)

It is therefore NOT a true watershed drainage density. For the terrestrial side
we compute both metrics so the comparison is apples-to-apples:

    Dd_true = stream_length / DEM-derived basin area
    Dd_hull = stream_length / convex hull area around the extracted streams

Threshold selection prioritises matching the Martian Dd_hull distribution.

The "basin" unit in this calibration is one drainage basin per outlet of the
StreamObject built at a given threshold (matching the per-network granularity
of the Martian shapefiles).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError

from .io.paths import DATA_DIR
from .logging_config import get_logger
from .units import (
    METERS_PER_DEGREE_LAT,
    METERS_PER_DEGREE_LON_EQUATOR,
    compute_basin_area_km2,
    compute_drainage_density,
    compute_pixel_size_m_from_dem,
    compute_stream_length_km,
    compute_threshold_cells,
)

logger = get_logger(__name__)


# =============================================================================
# Public defaults
# =============================================================================

DEFAULT_THRESHOLDS_KM2: list[float] = [0.5, 1, 2, 5, 10, 20, 50, 100]
"""Default contributing-area thresholds (km^2) sweep."""

PRACTICAL_THRESHOLDS_KM2: list[float] = [0.1, 0.2, 0.5, 1, 2, 3, 5, 7.5, 10]
"""Practical contributing-area thresholds (km^2).

Restricted to <= 10 km^2 and denser at small values: large thresholds
(e.g. ~150 km^2) leave too few usable terrestrial basins/pairs to be useful
for the pair-wise asymmetry project, so they are deliberately excluded.
"""

LOW_THRESHOLDS_KM2: list[float] = [0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
"""Very-low + practical contributing-area thresholds (km^2).

The study DEMs are ~85 m/px, so one pixel ~= 0.007 km^2. Any threshold below
roughly that floors to a single cell (``compute_threshold_cells`` clamps at
1), making 0.001/0.0025/0.005 indistinguishable from the fullest extractable
network at ~0.01 km^2. Lower nominal values are therefore *not* added: 0.01
km^2 already represents the most complete network the DEM can yield.
"""


def threshold_slug(km2: float) -> str:
    """Filename-safe threshold tag, e.g. 0.005 -> ``0p005km2``."""
    return f"{km2:g}".replace(".", "p") + "km2"


MARS_DD_STATS: dict[str, float] = {
    "n": 391,
    "median": 0.2238836162,
    "mean": 0.2418690910,
    "q1": 0.1611662654,
    "q3": 0.2986920715,
    "iqr": 0.1375258061,
    "std": 0.1189935099,
}
"""Mars Dd_hull target distribution (length km / hull km^2)."""


# Status values written into `per_basin_threshold_metrics.csv` to flag invalid rows.
STATUS_OK = "ok"
STATUS_NO_STREAM = "no_stream"
STATUS_ZERO_BASIN_AREA = "zero_basin_area"
STATUS_TOO_FEW_HULL_POINTS = "too_few_hull_points"
STATUS_INVALID_HULL = "invalid_hull"


@dataclass(slots=True)
class BasinMetrics:
    """Drainage-density metrics for one (basin, threshold) combination."""

    basin_id: str
    dem_name: str
    outlet_node: int
    threshold_km2: float
    threshold_cells: int
    pixel_size_m: float
    stream_length_km: float
    basin_area_km2: float
    hull_area_km2: float
    dd_true_km_km2: float
    dd_hull_km_km2: float
    n_stream_nodes: int
    n_basin_pixels: int
    status: str


# =============================================================================
# Geometry helpers
# =============================================================================


def _to_local_meters(
    rows: npt.NDArray[np.floating],
    cols: npt.NDArray[np.floating],
    pixel_size_m: float,
) -> npt.NDArray[np.floating]:
    """Return Nx2 (x_m, y_m) array of node positions in a local metric frame.

    Pixel-grid units multiplied by pixel size; the frame is local-equal-area
    enough for convex-hull area computation on a small region.
    """
    x_m = cols.astype(float) * pixel_size_m
    y_m = rows.astype(float) * pixel_size_m
    return np.column_stack([x_m, y_m])


def compute_convex_hull_area_km2(
    rows: npt.NDArray[np.floating],
    cols: npt.NDArray[np.floating],
    pixel_size_m: float,
) -> tuple[float, npt.NDArray[np.floating] | None, str]:
    """Convex hull area (km^2) around stream nodes.

    Returns (area_km2, hull_polygon_xy_m_or_None, status). Status is one of:
    STATUS_OK, STATUS_TOO_FEW_HULL_POINTS, STATUS_INVALID_HULL.
    """
    pts = _to_local_meters(rows, cols, pixel_size_m)
    # Need at least 3 unique non-collinear points.
    if len(np.unique(pts, axis=0)) < 3:
        return float("nan"), None, STATUS_TOO_FEW_HULL_POINTS
    try:
        hull = ConvexHull(pts)
    except QhullError:
        return float("nan"), None, STATUS_INVALID_HULL
    polygon = pts[hull.vertices]
    area_km2 = float(hull.volume) / 1_000_000.0  # 2D: hull.volume == area
    if area_km2 <= 0.0:
        return float("nan"), None, STATUS_INVALID_HULL
    return area_km2, polygon, STATUS_OK


# =============================================================================
# Per-basin metric computation
# =============================================================================


def _node_rowcol(s: Any) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    ni = s.node_indices() if callable(getattr(s, "node_indices", None)) else s.node_indices
    return np.asarray(ni[0], dtype=np.intp), np.asarray(ni[1], dtype=np.intp)


def evaluate_basin_metrics(
    *,
    dem_name: str,
    outlet_node: int,
    threshold_km2: float,
    threshold_cells: int,
    pixel_size_m: float,
    s_up: Any,
    basin_mask: npt.NDArray[np.bool_],
) -> BasinMetrics:
    """Compute Dd_true and Dd_hull for one basin/threshold combination."""
    basin_id = f"{dem_name}_o{int(outlet_node)}"
    basin_area_km2 = compute_basin_area_km2(basin_mask, pixel_size_m)
    rows, cols = _node_rowcol(s_up)
    n_nodes = int(rows.size)

    if n_nodes == 0:
        return BasinMetrics(
            basin_id=basin_id,
            dem_name=dem_name,
            outlet_node=int(outlet_node),
            threshold_km2=float(threshold_km2),
            threshold_cells=int(threshold_cells),
            pixel_size_m=float(pixel_size_m),
            stream_length_km=0.0,
            basin_area_km2=basin_area_km2,
            hull_area_km2=float("nan"),
            dd_true_km_km2=float("nan"),
            dd_hull_km_km2=float("nan"),
            n_stream_nodes=0,
            n_basin_pixels=int(basin_mask.sum()),
            status=STATUS_NO_STREAM,
        )

    length_km = compute_stream_length_km(s_up, pixel_size_m)
    hull_area_km2, _hull_poly, hull_status = compute_convex_hull_area_km2(
        rows.astype(float), cols.astype(float), pixel_size_m
    )

    if basin_area_km2 <= 0.0:
        return BasinMetrics(
            basin_id=basin_id,
            dem_name=dem_name,
            outlet_node=int(outlet_node),
            threshold_km2=float(threshold_km2),
            threshold_cells=int(threshold_cells),
            pixel_size_m=float(pixel_size_m),
            stream_length_km=length_km,
            basin_area_km2=basin_area_km2,
            hull_area_km2=hull_area_km2,
            dd_true_km_km2=float("nan"),
            dd_hull_km_km2=length_km / hull_area_km2 if hull_area_km2 > 0 else float("nan"),
            n_stream_nodes=n_nodes,
            n_basin_pixels=int(basin_mask.sum()),
            status=STATUS_ZERO_BASIN_AREA,
        )

    dd_true = length_km / basin_area_km2
    dd_hull = (
        length_km / hull_area_km2
        if hull_status == STATUS_OK and hull_area_km2 > 0
        else float("nan")
    )
    status = STATUS_OK if hull_status == STATUS_OK else hull_status

    return BasinMetrics(
        basin_id=basin_id,
        dem_name=dem_name,
        outlet_node=int(outlet_node),
        threshold_km2=float(threshold_km2),
        threshold_cells=int(threshold_cells),
        pixel_size_m=float(pixel_size_m),
        stream_length_km=length_km,
        basin_area_km2=basin_area_km2,
        hull_area_km2=hull_area_km2,
        dd_true_km_km2=dd_true,
        dd_hull_km_km2=dd_hull,
        n_stream_nodes=n_nodes,
        n_basin_pixels=int(basin_mask.sum()),
        status=status,
    )


# =============================================================================
# Summarisation & threshold selection
# =============================================================================


def _distribution_stats(values: pd.Series, prefix: str) -> dict[str, float]:
    """Compute count/mean/median/std/q1/q3/iqr/min/max for a numeric series."""
    if values.empty:
        nan = float("nan")
        return {
            f"{prefix}_mean": nan,
            f"{prefix}_median": nan,
            f"{prefix}_std": nan,
            f"{prefix}_q1": nan,
            f"{prefix}_q3": nan,
            f"{prefix}_iqr": nan,
            f"{prefix}_min": nan,
            f"{prefix}_max": nan,
        }
    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_median": float(values.median()),
        f"{prefix}_std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        f"{prefix}_q1": q1,
        f"{prefix}_q3": q3,
        f"{prefix}_iqr": q3 - q1,
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
    }


def summarize_threshold_metrics(
    per_basin_df: pd.DataFrame,
    mars_stats: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-threshold summary of Earth Dd_true and Dd_hull distributions.

    When `mars_stats` is provided, also emits `score_iqr`, `score_mean`,
    `score_median`, `score_normalized`, and `is_best_threshold` columns. The
    scores compare the Earth `Dd_hull` distribution to `mars_stats` exactly as
    in the original spec; with `mars_stats=None` (default), they are omitted.

    Set `mars_stats=MARS_DD_STATS` to restore the original Mars-matching
    behaviour.
    """
    valid = per_basin_df[per_basin_df["status"] == STATUS_OK].copy()

    rows = []
    for thr, grp in valid.groupby("threshold_km2"):
        dd_true = grp["dd_true_km_km2"].dropna()
        dd_hull = grp["dd_hull_km_km2"].dropna()
        if dd_true.empty:
            continue

        row: dict[str, float] = {"threshold_km2": float(thr), "count": int(dd_true.shape[0])}
        row.update(_distribution_stats(dd_true, "dd_true"))
        row.update(_distribution_stats(dd_hull, "dd_hull"))

        if mars_stats is not None and not dd_hull.empty:
            q1 = row["dd_hull_q1"]
            q3 = row["dd_hull_q3"]
            median = row["dd_hull_median"]
            mean = row["dd_hull_mean"]
            row["score_iqr"] = (
                abs(q1 - mars_stats["q1"])
                + abs(median - mars_stats["median"])
                + abs(q3 - mars_stats["q3"])
            )
            row["score_mean"] = abs(mean - mars_stats["mean"])
            row["score_median"] = abs(median - mars_stats["median"])
            row["score_normalized"] = (
                abs(q1 - mars_stats["q1"]) / mars_stats["q1"]
                + abs(median - mars_stats["median"]) / mars_stats["median"]
                + abs(q3 - mars_stats["q3"]) / mars_stats["q3"]
            )

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("threshold_km2").reset_index(drop=True)
    if summary.empty:
        if mars_stats is not None:
            summary["is_best_threshold"] = []
        return summary
    if mars_stats is not None and "score_iqr" in summary.columns:
        best = choose_best_threshold(summary)
        summary["is_best_threshold"] = summary["threshold_km2"] == best
    return summary


def choose_best_threshold(summary_df: pd.DataFrame, score_col: str = "score_iqr") -> float:
    """Pick the threshold minimising `score_col`."""
    if summary_df.empty:
        raise ValueError("summary_df is empty; cannot choose a best threshold")
    idx = summary_df[score_col].idxmin()
    return float(summary_df.loc[idx, "threshold_km2"])


# =============================================================================
# Misc utilities
# =============================================================================


def linear_index_fortran(row: int, col: int, n_rows: int) -> int:
    """Column-major (Fortran) linear index expected by FlowObject.drainagebasins."""
    return int(col) * int(n_rows) + int(row)


def _basin_pixel_counts(
    labels: npt.NDArray[np.integer],
    valid_dem: npt.NDArray[np.bool_],
    n_outlets: int,
) -> npt.NDArray[np.intp]:
    """Pixel count of each 1-based drainage-basin label within the valid DEM.

    Vectorized via a single ``np.bincount`` (label ``l`` -> ``outlet[l-1]``):
    returns an array of length ``n_outlets`` whose entry ``i`` is the size of
    the basin labelled ``i + 1``. Background label 0 is dropped. This avoids
    re-scanning the whole ``labels`` grid once per outlet.
    """
    valid_labels = labels[valid_dem].ravel().astype(np.int64)
    counts = np.bincount(valid_labels, minlength=n_outlets + 1)
    return counts[1 : n_outlets + 1]


def deg_to_local_meters(
    lon: npt.NDArray[np.floating],
    lat: npt.NDArray[np.floating],
    lat_center_deg: float,
) -> npt.NDArray[np.floating]:
    """Convert (lon, lat) arrays to local (x_m, y_m) anchored at the centroid."""
    lat_center_rad = math.radians(abs(lat_center_deg))
    m_per_deg_lon = METERS_PER_DEGREE_LON_EQUATOR * math.cos(lat_center_rad)
    m_per_deg_lat = METERS_PER_DEGREE_LAT
    lon_c = float(np.mean(lon))
    lat_c = float(np.mean(lat))
    x_m = (lon - lon_c) * m_per_deg_lon
    y_m = (lat - lat_c) * m_per_deg_lat
    return np.column_stack([x_m, y_m])


# =============================================================================
# Driver used by the CLI script (kept here so it's importable and testable)
# =============================================================================


def collect_basin_metrics_for_dem(
    *,
    dem_path: Path,
    basin_name: str,
    thresholds_km2: list[float],
    z_th: float | None,
    lat_deg: float,
    min_outlet_basin_pixels: int = 1,
) -> list[BasinMetrics]:
    """Sweep thresholds for one DEM. One row per (outlet basin, threshold).

    `min_outlet_basin_pixels` filters out trivial basins (mostly noise) without
    silently dropping otherwise-valid ones.
    """
    import topotoolbox as tt3  # imported lazily to keep the module light

    logger.info("Loading DEM %s", dem_path)
    dem = tt3.read_tif(str(dem_path))
    if z_th is not None:
        dem.z[dem.z < z_th] = np.nan

    pixel_size_m = compute_pixel_size_m_from_dem(dem, lat_deg=lat_deg)
    logger.info("  pixel_size = %.2f m, shape = %s", pixel_size_m, dem.z.shape)

    fd = tt3.FlowObject(dem)
    n_rows = dem.z.shape[0]
    valid_dem = ~np.isnan(dem.z)

    results: list[BasinMetrics] = []
    for thr_km2 in thresholds_km2:
        cells = compute_threshold_cells(thr_km2, pixel_size_m)
        logger.info("  threshold = %.3f km^2 -> %d cells; building StreamObject", thr_km2, cells)
        try:
            s = tt3.StreamObject(fd, threshold=cells)
        except Exception as exc:
            logger.warning("    failed to build StreamObject (%s); skipping", exc)
            continue

        # Pull outlet stream-node ids, then their (r,c) and column-major linear ids.
        outlet_mask = s.streampoi("outlets")
        outlet_node_ids = np.flatnonzero(outlet_mask)
        if outlet_node_ids.size == 0:
            logger.info("    no outlets at this threshold")
            continue

        rows_all, cols_all = _node_rowcol(s)
        outlet_lin = np.array(
            [linear_index_fortran(rows_all[o], cols_all[o], n_rows) for o in outlet_node_ids],
            dtype=np.int64,
        )

        # Single labelled basins grid -> one mask per outlet by label index.
        basins_grid = fd.drainagebasins(outlet_lin)
        labels = np.asarray(basins_grid.z)

        for i, out_id in enumerate(outlet_node_ids):
            label_value = i + 1  # drainagebasins labels are 1-based in outlet order
            basin_mask = (labels == label_value) & valid_dem
            n_basin_px = int(basin_mask.sum())
            if n_basin_px < min_outlet_basin_pixels:
                continue

            # Build subgraph upstream of this outlet only.
            up_mask = np.zeros(rows_all.shape[0], dtype=bool)
            up_mask[out_id] = True
            s_up = s.upstreamto(up_mask)

            metrics = evaluate_basin_metrics(
                dem_name=basin_name,
                outlet_node=int(out_id),
                threshold_km2=thr_km2,
                threshold_cells=cells,
                pixel_size_m=pixel_size_m,
                s_up=s_up,
                basin_mask=basin_mask,
            )
            results.append(metrics)

    return results


# =============================================================================
# Mars valley-network Dd_hull (mapped proxy reference distribution)
# =============================================================================

DEFAULT_MARS_VALLEYS_GPKG: Path = DATA_DIR / "final_valleys" / "final_valleys_fixed.gpkg"
"""Authoritative Mars mapped valley networks (391 networks, projected metres).

This file reproduces ``MARS_DD_STATS`` exactly when convex hulls are computed
per ``network_id``; the sibling ``*_concave_hulls_by_network.gpkg`` is empty.
"""


def _network_line_coords(geom: Any) -> list[npt.NDArray[np.floating]]:
    """Return a list of (N, 2) vertex arrays for a (Multi)LineString."""
    parts = geom.geoms if geom.geom_type == "MultiLineString" else [geom]
    return [np.asarray(line.coords)[:, :2] for line in parts]


def mars_network_geometry(
    gpkg_path: Path | str | None = None,
) -> dict[int, dict[str, Any]]:
    """Per-network Mars valley geometry + convex-hull Dd_hull.

    The GeoPackage CRS is a Mars equirectangular frame in metres, so hull
    area is taken directly from the planar convex hull (km^2). One entry per
    ``network_id``::

        {network_id: {"lines": [ (N,2) ... ],   # valley polylines, metres
                      "hull_xy": (M,2) | None,    # convex-hull polygon, metres
                      "length_km": float,
                      "hull_area_km2": float,
                      "dd_hull": float,
                      "status": str}}
    """
    import geopandas as gpd

    path = Path(gpkg_path) if gpkg_path is not None else DEFAULT_MARS_VALLEYS_GPKG
    gdf = gpd.read_file(path)

    out: dict[int, dict[str, Any]] = {}
    for nid, sub in gdf.groupby("network_id"):
        lines: list[npt.NDArray[np.floating]] = []
        for geom in sub.geometry:
            lines.extend(_network_line_coords(geom))
        if "Length(km)" in sub.columns:
            length_km = float(sub["Length(km)"].sum())
        elif "total_length_m" in sub.columns:
            length_km = float(sub["total_length_m"].sum()) / 1000.0
        else:
            length_km = float("nan")
        if not lines:
            out[int(nid)] = {
                "lines": [],
                "hull_xy": None,
                "length_km": length_km,
                "hull_area_km2": float("nan"),
                "dd_hull": float("nan"),
                "status": STATUS_NO_STREAM,
            }
            continue
        pts = np.vstack(lines)
        if len(np.unique(pts, axis=0)) < 3:
            status, hull_xy, hull_km2 = STATUS_TOO_FEW_HULL_POINTS, None, float("nan")
        else:
            try:
                hull = ConvexHull(pts)
                hull_xy = pts[hull.vertices]
                hull_km2 = float(hull.volume) / 1_000_000.0
                status = STATUS_OK if hull_km2 > 0 else STATUS_INVALID_HULL
                if hull_km2 <= 0:
                    hull_xy, hull_km2 = None, float("nan")
            except QhullError:
                status, hull_xy, hull_km2 = STATUS_INVALID_HULL, None, float("nan")
        out[int(nid)] = {
            "lines": lines,
            "hull_xy": hull_xy,
            "length_km": length_km,
            "hull_area_km2": hull_km2,
            "dd_hull": (
                length_km / hull_km2 if status == STATUS_OK and hull_km2 > 0 else float("nan")
            ),
            "status": status,
        }
    return out


def mars_network_table(gpkg_path: Path | str | None = None) -> pd.DataFrame:
    """Flat per-network Mars Dd_hull table (one row per ``network_id``)."""
    geom = mars_network_geometry(gpkg_path)
    rows = [
        {
            "network_id": nid,
            "length_km": g["length_km"],
            "hull_area_km2": g["hull_area_km2"],
            "dd_hull_km_km2": g["dd_hull"],
            "status": g["status"],
        }
        for nid, g in geom.items()
    ]
    return pd.DataFrame(rows).sort_values("network_id").reset_index(drop=True)


# =============================================================================
# Mars Strahler order (root-invariant Horton-Strahler max from polyline graph)
# =============================================================================


def _build_mars_adjacency(
    lines: list[npt.NDArray[np.floating]],
    snap_decimals: int = 3,
) -> dict[tuple[float, float], set[tuple[float, float]]]:
    """Undirected adjacency from a list of polylines.

    Vertices are snapped via rounding to ``snap_decimals`` decimals (mm in Mars
    equirectangular metres) so coincident endpoints from separate LineStrings
    collapse to one node. Zero-length segments are dropped.
    """
    from collections import defaultdict

    adj: dict[tuple[float, float], set[tuple[float, float]]] = defaultdict(set)
    for poly in lines:
        arr = np.asarray(poly, dtype=float)[:, :2]
        if arr.shape[0] < 2:
            continue
        snapped = [(round(p[0], snap_decimals), round(p[1], snap_decimals)) for p in arr]
        for u, v in zip(snapped[:-1], snapped[1:]):
            if u == v:
                continue
            adj[u].add(v)
            adj[v].add(u)
    return adj


def _strahler_rooted_at(
    adj: dict[Any, set[Any]],
    root: Any,
    component_nodes: set[Any],
) -> int:
    """Standard rooted Horton-Strahler order at ``root``.

    Iterative DFS records discovery order; reversing it yields a valid
    post-order on the spanning tree, on which the textbook rule is applied
    (single max child -> inherit; two or more equal-max children -> +1).
    """
    parent: dict[Any, Any] = {root: None}
    order: list[Any] = [root]
    visited: set[Any] = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v in visited or v not in component_nodes:
                continue
            visited.add(v)
            parent[v] = u
            order.append(v)
            stack.append(v)
    s_order: dict[Any, int] = {}
    for u in reversed(order):
        child_orders = [s_order[v] for v in adj[u] if parent.get(v) == u]
        if not child_orders:
            s_order[u] = 1
            continue
        mx = max(child_orders)
        s_order[u] = mx + 1 if child_orders.count(mx) >= 2 else mx
    # Root accumulates the whole tree, so its order is the max in the tree.
    return s_order[root]


def _horton_strahler_max(
    adj: dict[Any, set[Any]],
) -> tuple[int, int, int, bool]:
    """Max Horton-Strahler order of an undirected graph.

    Returns ``(max_strahler, n_leaves, n_components, is_tree)``. The
    Horton-Strahler number of an unrooted tree equals the maximum over leaf
    rootings (the textbook definition: rooting at a non-leaf can spuriously
    create branching at an interior node). For each component we therefore
    root at every degree-1 leaf and take the max. Cycles (n_edges > n-1) are
    flagged via ``is_tree=False``; the per-leaf computation still runs on
    the spanning tree, so a value is returned anyway.
    """
    if not adj:
        return 0, 0, 0, True

    # Find connected components and detect cycles per component.
    seen: set[Any] = set()
    components: list[set[Any]] = []
    is_tree = True
    for start in adj:
        if start in seen:
            continue
        comp: set[Any] = set()
        deg_sum = 0
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            deg_sum += len(adj[u])
            for v in adj[u]:
                if v not in seen:
                    stack.append(v)
        n_edges = deg_sum // 2
        if n_edges > len(comp) - 1:
            is_tree = False
        components.append(comp)

    overall_max = 0
    total_leaves = 0
    for comp in components:
        comp_leaves = [n for n in comp if len(adj[n]) == 1]
        total_leaves += len(comp_leaves)
        if not comp_leaves:
            # Pure cycle / isolated multigraph component: Strahler not well
            # defined; treat as a single channel.
            overall_max = max(overall_max, 1)
            continue
        for leaf_root in comp_leaves:
            m = _strahler_rooted_at(adj, leaf_root, comp)
            if m > overall_max:
                overall_max = m
    return overall_max, total_leaves, len(components), is_tree


def mars_network_strahler(
    gpkg_path: Path | str | None = None,
) -> pd.DataFrame:
    """Per-network max Horton-Strahler order for Mars valley networks.

    The Horton-Strahler max of an unrooted tree is invariant to root choice,
    so Martian flow direction is not required. One row per ``network_id``::

        network_id, max_strahler, n_nodes, n_edges, n_leaves,
        n_components, is_tree, length_km, hull_area_km2, dd_hull, status

    Geometry-derived fields (``length_km``, ``hull_area_km2``, ``dd_hull``,
    ``status``) are joined from :func:`mars_network_geometry` so a single
    table powers complexity-vs-Dd plots.
    """
    geom = mars_network_geometry(gpkg_path)
    rows: list[dict[str, Any]] = []
    for nid, g in geom.items():
        adj = _build_mars_adjacency(g["lines"]) if g["lines"] else {}
        max_s, n_leaves, n_comp, is_tree = _horton_strahler_max(adj)
        rows.append(
            {
                "network_id": int(nid),
                "max_strahler": int(max_s),
                "n_nodes": len(adj),
                "n_edges": sum(len(v) for v in adj.values()) // 2,
                "n_leaves": int(n_leaves),
                "n_components": int(n_comp),
                "is_tree": bool(is_tree),
                "length_km": float(g["length_km"]),
                "hull_area_km2": float(g["hull_area_km2"]),
                "dd_hull": float(g["dd_hull"]),
                "status": g["status"],
            }
        )
    return pd.DataFrame(rows).sort_values("network_id").reset_index(drop=True)


# =============================================================================
# Strahler first-order trimming + per-basin Dd_hull (full vs trimmed)
# =============================================================================


def trim_to_min_order(s: Any, min_order: int) -> Any:
    """Return the subnetwork keeping only Strahler order >= ``min_order``.

    ``min_order=2`` removes 1st-order links; ``min_order=3`` also removes
    2nd-order links. Returns ``None`` if nothing survives.
    """
    if min_order <= 1:
        return s
    order = np.asarray(s.streamorder(method="strahler"))
    keep = order >= min_order
    if not keep.any():
        return None
    return s.subgraph(keep)


def trim_first_order(s: Any) -> Any:
    """Backward-compatible alias: keep Strahler order >= 2.

    Returns ``None`` if the whole network was 1st-order.
    """
    return trim_to_min_order(s, 2)


def _basin_dd_hull(s_basin: Any, pixel_size_m: float) -> tuple[float, float, float, int, str]:
    """(length_km, hull_area_km2, dd_hull, n_nodes, status) for a subnetwork."""
    if s_basin is None:
        return 0.0, float("nan"), float("nan"), 0, STATUS_NO_STREAM
    rows, cols = _node_rowcol(s_basin)
    n = int(rows.size)
    if n == 0:
        return 0.0, float("nan"), float("nan"), 0, STATUS_NO_STREAM
    length_km = compute_stream_length_km(s_basin, pixel_size_m)
    hull_km2, _poly, hstatus = compute_convex_hull_area_km2(
        rows.astype(float), cols.astype(float), pixel_size_m
    )
    dd_hull = length_km / hull_km2 if hstatus == STATUS_OK and hull_km2 > 0 else float("nan")
    return length_km, hull_km2, dd_hull, n, hstatus


def _stream_segments_xy(s: Any, pixel_size_m: float) -> npt.NDArray[np.floating]:
    """Edge segments of a StreamObject as an (E, 2, 2) array in metres.

    Uses ``source_indices``/``target_indices`` (per-edge (row, col) tuples) so
    the drawn network follows true topology, not node-index order. y is
    negated so image rows increase downward when plotted.
    """
    sr, sc = s.source_indices
    tr, tc = s.target_indices
    sr = np.asarray(sr, float)
    sc = np.asarray(sc, float)
    tr = np.asarray(tr, float)
    tc = np.asarray(tc, float)
    if sr.size == 0:
        return np.empty((0, 2, 2), dtype=float)
    p0 = np.column_stack([sc * pixel_size_m, -sr * pixel_size_m])
    p1 = np.column_stack([tc * pixel_size_m, -tr * pixel_size_m])
    return np.stack([p0, p1], axis=1)


def mask_boundary_xy(
    mask: npt.NDArray[np.bool_], pixel_size_m: float
) -> npt.NDArray[np.floating] | None:
    """Largest outer boundary of a boolean basin mask, as (N, 2) metres.

    Coordinates use the same convention as ``_stream_segments_xy``
    (x = col * px, y = -row * px) so basin outlines overlay stream maps.
    """
    if not mask.any():
        return None
    padded = np.pad(mask.astype(float), 1)
    try:
        from skimage import measure

        contours = measure.find_contours(padded, 0.5)
        if not contours:
            return None
        biggest = max(contours, key=len)  # rows/cols in padded frame
        rr = biggest[:, 0] - 1.0
        cc = biggest[:, 1] - 1.0
    except ModuleNotFoundError:
        # Optional dependency fallback for presentation maps. Matplotlib's
        # contour paths are (x=col, y=row), opposite of skimage's row/col.
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        try:
            contour_set = ax.contour(padded, levels=[0.5])
            paths = (
                contour_set.get_paths()
                if hasattr(contour_set, "get_paths")
                else [
                    path
                    for collection in contour_set.collections
                    for path in collection.get_paths()
                ]
            )
        finally:
            plt.close(fig)
        if not paths:
            return None
        biggest_xy = max((path.vertices for path in paths), key=len)
        cc = biggest_xy[:, 0] - 1.0
        rr = biggest_xy[:, 1] - 1.0
    return np.column_stack([cc * pixel_size_m, -rr * pixel_size_m])


def _edge_keys(s: Any) -> set[tuple[int, int, int, int]]:
    """Hashable per-edge identity (src_row, src_col, tgt_row, tgt_col)."""
    sr, sc = s.source_indices
    tr, tc = s.target_indices
    return {
        (int(a), int(b), int(c), int(d))
        for a, b, c, d in zip(np.asarray(sr), np.asarray(sc), np.asarray(tr), np.asarray(tc))
    }


def _channelhead_count(s: Any) -> int:
    try:
        return int(np.count_nonzero(s.streampoi("channelheads")))
    except Exception:  # noqa: BLE001 - POI extraction is best-effort metadata
        return -1


def _max_strahler(s: Any) -> int:
    """Max Horton-Strahler order on a (sub)network; 0 for empty/None."""
    if s is None:
        return 0
    try:
        order = np.asarray(s.streamorder(method="strahler"))
    except Exception:  # noqa: BLE001 - missing topology is best-effort metadata
        return 0
    return int(order.max()) if order.size > 0 else 0


def _variant_payload(
    s_variant: Any,
    *,
    pixel_size_m: float,
    basin_area_km2: float,
    full_length_km: float | None,
    full_edge_keys: set[tuple[int, int, int, int]] | None,
    want_geom: bool,
) -> dict[str, Any]:
    """Metrics (+ optional geometry) for one network variant.

    ``full_edge_keys`` lets a pruned variant report the *removed* edges so
    maps can draw them in a distinct style.
    """
    length_km, hull_km2, dd_hull, n_nodes, status = _basin_dd_hull(s_variant, pixel_size_m)
    pct_len = (
        float("nan")
        if not full_length_km or full_length_km <= 0
        else 100.0 * length_km / full_length_km
    )
    payload: dict[str, Any] = {
        "length_km": length_km,
        "hull_area_km2": hull_km2,
        "dd_hull": dd_hull,
        "dd_true": compute_drainage_density(length_km, basin_area_km2),
        "n_nodes": n_nodes,
        "n_segments": 0 if s_variant is None else _edge_count(s_variant),
        "n_channelheads": -1 if s_variant is None else _channelhead_count(s_variant),
        "pct_length_retained": pct_len,
        "max_strahler": _max_strahler(s_variant),
        "status": STATUS_OK if status == STATUS_OK else status,
    }
    if want_geom:
        if s_variant is None:
            payload.update(stream_seg=None, hull_xy=None, removed_seg=None)
        else:
            rr, cc = _node_rowcol(s_variant)
            _, hull_poly, hs = compute_convex_hull_area_km2(
                rr.astype(float), cc.astype(float), pixel_size_m
            )
            payload["stream_seg"] = _stream_segments_xy(s_variant, pixel_size_m)
            payload["hull_xy"] = (
                np.column_stack([hull_poly[:, 0], -hull_poly[:, 1]])
                if hs == STATUS_OK and hull_poly is not None
                else None
            )
            payload["removed_seg"] = _removed_segments(s_variant, full_edge_keys, pixel_size_m)
    return payload


def _edge_count(s: Any) -> int:
    sr, _ = s.source_indices
    return int(np.asarray(sr).size)


def _removed_segments(
    s_variant: Any,
    full_edge_keys: set[tuple[int, int, int, int]] | None,
    pixel_size_m: float,
) -> npt.NDArray[np.floating] | None:
    """Edges present in the full network but absent from this variant."""
    if full_edge_keys is None:
        return None
    kept = _edge_keys(s_variant)
    removed = full_edge_keys - kept
    if not removed:
        return np.empty((0, 2, 2), dtype=float)
    arr = np.array(list(removed), dtype=float)  # (E, 4): sr, sc, tr, tc
    p0 = np.column_stack([arr[:, 1] * pixel_size_m, -arr[:, 0] * pixel_size_m])
    p1 = np.column_stack([arr[:, 3] * pixel_size_m, -arr[:, 2] * pixel_size_m])
    return np.stack([p0, p1], axis=1)


def outlet_pruning_variants(
    s_up: Any,
    *,
    pixel_size_m: float,
    basin_area_km2: float,
    min_orders: tuple[int, ...] = (2, 3),
    want_geom: bool = False,
) -> dict[str, dict[str, Any]]:
    """Full network + each Strahler-pruned variant for one outlet subgraph.

    Returns ``{"full": payload, "ge2": payload, "ge3": payload, ...}`` where
    each payload comes from :func:`_variant_payload`.
    """
    full_keys = _edge_keys(s_up) if want_geom else None
    full_len = compute_stream_length_km(s_up, pixel_size_m)
    variants: dict[str, dict[str, Any]] = {
        "full": _variant_payload(
            s_up,
            pixel_size_m=pixel_size_m,
            basin_area_km2=basin_area_km2,
            full_length_km=full_len,
            full_edge_keys=None,
            want_geom=want_geom,
        )
    }
    for mo in min_orders:
        variants[f"ge{mo}"] = _variant_payload(
            trim_to_min_order(s_up, mo),
            pixel_size_m=pixel_size_m,
            basin_area_km2=basin_area_km2,
            full_length_km=full_len,
            full_edge_keys=full_keys,
            want_geom=want_geom,
        )
    return variants


def collect_dd_hull_trim_for_dem(
    *,
    dem_path: Path,
    basin_name: str,
    threshold_km2: float,
    z_th: float | None,
    lat_deg: float,
    min_outlet_basin_pixels: int = 1,
    geometry_for_outlets: set[int] | None = None,
    max_outlets: int | None = None,
) -> list[dict[str, Any]]:
    """Per-outlet Dd_hull at one threshold, full network vs 1st-order-trimmed.

    One dict per outlet basin with ``dd_hull_full`` / ``dd_hull_trim`` (plus
    lengths, hull areas, node counts and per-variant status). When an outlet
    id is in ``geometry_for_outlets`` the dict also carries, in metres for
    to-scale plotting: ``stream_seg_full`` / ``stream_seg_trim`` ((E, 2, 2)
    true-topology edge segments), ``stream_xy_full`` (node cloud, for
    centring) and ``hull_xy_full`` (convex-hull polygon).
    """
    import topotoolbox as tt3

    dem = tt3.read_tif(str(dem_path))
    if z_th is not None:
        dem.z[dem.z < z_th] = np.nan
    pixel_size_m = compute_pixel_size_m_from_dem(dem, lat_deg=lat_deg)
    fd = tt3.FlowObject(dem)
    s = tt3.StreamObject(fd, threshold=compute_threshold_cells(threshold_km2, pixel_size_m))
    return _dd_trim_rows(
        s=s,
        fd=fd,
        basin_name=basin_name,
        threshold_km2=threshold_km2,
        pixel_size_m=pixel_size_m,
        n_rows=dem.z.shape[0],
        valid_dem=~np.isnan(dem.z),
        min_outlet_basin_pixels=min_outlet_basin_pixels,
        want_geom=geometry_for_outlets or set(),
        max_outlets=max_outlets,
    )


def _dd_trim_rows(
    *,
    s: Any,
    fd: Any,
    basin_name: str,
    threshold_km2: float,
    pixel_size_m: float,
    n_rows: int,
    valid_dem: npt.NDArray[np.bool_],
    min_outlet_basin_pixels: int,
    want_geom: set[int],
    max_outlets: int | None = None,
    min_orders: tuple[int, ...] = (2, 3),
) -> list[dict[str, Any]]:
    """Per-outlet full-vs-trimmed Dd rows for one built StreamObject ``s``.

    ``max_outlets`` keeps only the N largest basins (by pixel count). At very
    low thresholds a DEM can have thousands of outlets and millions of nodes;
    capping bounds runtime and focuses on substantial basins (the regime the
    low-threshold hypothesis is about).
    """
    outlet_node_ids = np.flatnonzero(s.streampoi("outlets"))
    if outlet_node_ids.size == 0:
        return []
    rows_all, cols_all = _node_rowcol(s)
    outlet_lin = np.array(
        [linear_index_fortran(rows_all[o], cols_all[o], n_rows) for o in outlet_node_ids],
        dtype=np.int64,
    )
    labels = np.asarray(fd.drainagebasins(outlet_lin).z)

    sizes = _basin_pixel_counts(labels, valid_dem, outlet_node_ids.size)
    keep_idx = np.flatnonzero(sizes >= min_outlet_basin_pixels)
    if max_outlets is not None and keep_idx.size > max_outlets:
        keep_idx = keep_idx[np.argsort(sizes[keep_idx])[::-1][:max_outlets]]

    out: list[dict[str, Any]] = []
    for i in keep_idx:
        i = int(i)
        out_id = int(outlet_node_ids[i])
        basin_mask = (labels == i + 1) & valid_dem
        up_mask = np.zeros(rows_all.shape[0], dtype=bool)
        up_mask[out_id] = True
        s_up = s.upstreamto(up_mask)

        area_km2 = compute_basin_area_km2(basin_mask, pixel_size_m)
        want = int(out_id) in want_geom
        v = outlet_pruning_variants(
            s_up,
            pixel_size_m=pixel_size_m,
            basin_area_km2=area_km2,
            min_orders=min_orders,
            want_geom=want,
        )

        rec: dict[str, Any] = {
            "basin_id": f"{basin_name}_o{int(out_id)}",
            "dem_name": basin_name,
            "outlet_node": int(out_id),
            # Outlet pixel position in the DEM grid -- threshold-independent
            # identity of a physical pour point. Used for cross-threshold
            # outlet matching (the StreamObject node id IS threshold-dependent).
            "outlet_row": int(rows_all[out_id]),
            "outlet_col": int(cols_all[out_id]),
            "threshold_km2": float(threshold_km2),
            "pixel_size_m": pixel_size_m,
            "basin_area_km2": area_km2,
            "n_channelheads_full": v["full"]["n_channelheads"],
            "n_segments_full": v["full"]["n_segments"],
        }
        # Legacy mapping: "trim" == ge2 for compatibility with older CSVs.
        # ge4+ get a generic ``geN`` suffix.
        suffix_for: dict[str, str] = {"full": "full"}
        for mo in min_orders:
            suffix_for[f"ge{mo}"] = "trim" if mo == 2 else f"ge{mo}"
        for vk, sfx in suffix_for.items():
            p = v[vk]
            rec[f"length_km_{sfx}"] = p["length_km"]
            rec[f"hull_area_km2_{sfx}"] = p["hull_area_km2"]
            rec[f"dd_hull_{sfx}"] = p["dd_hull"]
            rec[f"dd_true_{sfx}"] = p["dd_true"]
            rec[f"n_nodes_{sfx}"] = p["n_nodes"]
            rec[f"max_strahler_{sfx}"] = p["max_strahler"]
            rec[f"status_{sfx}"] = p["status"]
            if sfx != "full":
                rec[f"pct_length_{sfx}"] = p["pct_length_retained"]

        if want:
            rec["outlet_xy"] = np.array(
                [cols_all[out_id] * pixel_size_m, -rows_all[out_id] * pixel_size_m],
                dtype=float,
            )
            rec["basin_boundary_xy"] = mask_boundary_xy(basin_mask, pixel_size_m)
            for vk, sfx in suffix_for.items():
                rec[f"stream_seg_{sfx}"] = v[vk].get("stream_seg")
                rec[f"hull_xy_{sfx}"] = v[vk].get("hull_xy")
                if sfx != "full":
                    rec[f"removed_seg_{sfx}"] = v[vk].get("removed_seg")
        out.append(rec)
    return out


def collect_dd_trim_for_dem_multi(
    *,
    dem_path: Path,
    basin_name: str,
    thresholds_km2: list[float],
    z_th: float | None,
    lat_deg: float,
    min_outlet_basin_pixels: int = 1,
    max_outlets: int | None = None,
    min_orders: tuple[int, ...] = (2, 3),
) -> list[dict[str, Any]]:
    """Full-vs-trimmed Dd_hull/Dd_true for one DEM across many thresholds.

    Builds the DEM and ``FlowObject`` once and only the (cheap) StreamObject
    per threshold, so it is far faster than calling
    ``collect_dd_hull_trim_for_dem`` once per threshold.
    """
    import topotoolbox as tt3

    dem = tt3.read_tif(str(dem_path))
    if z_th is not None:
        dem.z[dem.z < z_th] = np.nan
    pixel_size_m = compute_pixel_size_m_from_dem(dem, lat_deg=lat_deg)
    fd = tt3.FlowObject(dem)
    n_rows = dem.z.shape[0]
    valid_dem = ~np.isnan(dem.z)

    rows: list[dict[str, Any]] = []
    for thr in thresholds_km2:
        cells = compute_threshold_cells(thr, pixel_size_m)
        try:
            s = tt3.StreamObject(fd, threshold=cells)
        except Exception as exc:  # noqa: BLE001 - log and skip a bad threshold
            logger.warning("[%s @ %s km^2] StreamObject failed: %s", basin_name, thr, exc)
            continue
        rows.extend(
            _dd_trim_rows(
                s=s,
                fd=fd,
                basin_name=basin_name,
                threshold_km2=float(thr),
                pixel_size_m=pixel_size_m,
                n_rows=n_rows,
                valid_dem=valid_dem,
                min_outlet_basin_pixels=min_outlet_basin_pixels,
                want_geom=set(),
                max_outlets=max_outlets,
                min_orders=min_orders,
            )
        )
    return rows


# Geometry columns hold numpy arrays / None; they are never written to the
# tabular sweep cache (they would serialize to unparseable repr strings).
# Matched as prefixes against every emitted column name, so this must cover
# the suffixed variants too (e.g. stream_seg_full, hull_xy_trim, removed_seg_ge3).
_GEOM_KEYS = (
    "stream_xy",
    "stream_seg",
    "hull_xy",
    "removed_seg",
    "outlet_xy",
    "basin_boundary",
)


def _strip_geometry_columns(record: dict[str, Any]) -> dict[str, Any]:
    """Drop array-valued geometry columns so a record is CSV-serializable.

    Geometry keys (and their ``_full``/``_trim``/``_geN`` suffixed variants)
    carry numpy arrays or ``None``; only the scalar/tabular fields belong in
    the flat sweep cache. Prefix matching keeps tabular look-alikes such as
    ``outlet_node``/``outlet_row``/``outlet_col`` and ``basin_area_km2``.
    """
    return {k: v for k, v in record.items() if not k.startswith(_GEOM_KEYS)}


def run_practical_sweep(
    *,
    basins: list[str],
    thresholds_km2: list[float] | None = None,
    cache_csv: Path | str | None = None,
    min_outlet_basin_pixels: int = 200,
    max_outlets: int | None = None,
    use_z_th: bool = True,
    min_orders: tuple[int, ...] = (2, 3),
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the practical-range full-vs-trimmed sweep across basins.

    Returns one tidy row per (basin, outlet, threshold) with ``dd_hull_full``,
    ``dd_true_full``, ``dd_hull_trim``, ``dd_true_trim`` and per-variant
    status. If ``cache_csv`` exists it is loaded instead of recomputed (the
    DEM sweep is heavy); otherwise the result is written there.
    """
    from .basin_config import LOCAL_TO_PAPER_BASIN, get_basin_config
    from .io.paths import EXAMPLE_DEMS

    thresholds_km2 = thresholds_km2 or PRACTICAL_THRESHOLDS_KM2

    if cache_csv is not None:
        cache_csv = Path(cache_csv)
        if cache_csv.exists():
            if verbose:
                print(f"Loading cached sweep: {cache_csv}")
            return pd.read_csv(cache_csv)

    all_rows: list[dict[str, Any]] = []
    for name in basins:
        cfg = get_basin_config(LOCAL_TO_PAPER_BASIN.get(name, name))
        z_th = cfg["z_th"] if use_z_th else None
        try:
            rows = collect_dd_trim_for_dem_multi(
                dem_path=EXAMPLE_DEMS[name],
                basin_name=name,
                thresholds_km2=thresholds_km2,
                z_th=z_th,
                lat_deg=float(cfg["lat"]),
                min_outlet_basin_pixels=min_outlet_basin_pixels,
                max_outlets=max_outlets,
                min_orders=min_orders,
            )
        except Exception as exc:  # noqa: BLE001 - skip a broken DEM, keep going
            print(f"  [{name}] failed: {exc}")
            continue
        if verbose:
            print(f"  {name}: {len(rows)} (outlet, threshold) rows")
        all_rows.extend(rows)

    df = pd.DataFrame([_strip_geometry_columns(r) for r in all_rows])
    if cache_csv is not None and not df.empty:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_csv, index=False)
        if verbose:
            print(f"Wrote {cache_csv} ({len(df)} rows)")
    return df


def single_basin_threshold_geometry(
    *,
    dem_path: Path,
    basin_name: str,
    thresholds_km2: list[float],
    z_th: float | None,
    lat_deg: float,
    outlet_rank: int = 0,
    with_pruning: bool = True,
) -> list[dict[str, Any]]:
    """Same selected basin drawn (and pruned) across several thresholds.

    The basin is selected by *size rank* among outlets at the finest
    threshold (``outlet_rank=0`` is the largest, ``1`` the second-largest, …)
    because raw outlet node IDs are not stable across thresholds. The pour
    point is threshold-invariant, so the basin mask/area is fixed once and
    only channel length changes with threshold.

    Each per-threshold dict carries ``threshold_km2``, ``basin_area_km2``,
    ``seed_outlet_node`` (resolved at the finest threshold, for reference),
    ``basin_boundary_xy``, ``outlet_xy`` and a ``variants`` dict
    (``full``/``ge2``/``ge3`` payloads with metrics + geometry from
    :func:`outlet_pruning_variants`).
    """
    import topotoolbox as tt3

    dem = tt3.read_tif(str(dem_path))
    if z_th is not None:
        dem.z[dem.z < z_th] = np.nan
    pixel_size_m = compute_pixel_size_m_from_dem(dem, lat_deg=lat_deg)
    fd = tt3.FlowObject(dem)
    n_rows = dem.z.shape[0]
    valid_dem = ~np.isnan(dem.z)

    thr_sorted = sorted(thresholds_km2)
    min_orders = (2, 3) if with_pruning else ()

    # Seed = pour point of the rank-th largest basin at the finest threshold.
    s0 = tt3.StreamObject(fd, threshold=compute_threshold_cells(thr_sorted[0], pixel_size_m))
    out0 = np.flatnonzero(s0.streampoi("outlets"))
    r0, c0 = _node_rowcol(s0)
    lin0 = np.array([linear_index_fortran(r0[o], c0[o], n_rows) for o in out0], dtype=np.int64)
    labels0 = np.asarray(fd.drainagebasins(lin0).z)
    sizes = _basin_pixel_counts(labels0, valid_dem, out0.size)
    order_by_size = np.argsort(sizes)[::-1]
    pick = int(order_by_size[min(outlet_rank, order_by_size.size - 1)])
    seed_rc = (int(r0[out0[pick]]), int(c0[out0[pick]]))
    seed_node = int(out0[pick])
    seed_lin = linear_index_fortran(seed_rc[0], seed_rc[1], n_rows)
    fixed_basin_mask = (np.asarray(fd.drainagebasins(np.array([seed_lin])).z) == 1) & valid_dem
    basin_area_km2 = compute_basin_area_km2(fixed_basin_mask, pixel_size_m)
    boundary_xy = mask_boundary_xy(fixed_basin_mask, pixel_size_m)

    series: list[dict[str, Any]] = []
    for thr in thr_sorted:
        s = tt3.StreamObject(fd, threshold=compute_threshold_cells(thr, pixel_size_m))
        outs = np.flatnonzero(s.streampoi("outlets"))
        if outs.size == 0:
            continue
        rr, cc = _node_rowcol(s)
        d2 = (rr[outs] - seed_rc[0]) ** 2 + (cc[outs] - seed_rc[1]) ** 2
        out_id = int(outs[int(np.argmin(d2))])
        up_mask = np.zeros(rr.shape[0], dtype=bool)
        up_mask[out_id] = True
        s_up = s.upstreamto(up_mask)
        variants = outlet_pruning_variants(
            s_up,
            pixel_size_m=pixel_size_m,
            basin_area_km2=basin_area_km2,
            min_orders=min_orders,
            want_geom=True,
        )
        series.append(
            {
                "threshold_km2": float(thr),
                "basin_area_km2": basin_area_km2,
                "seed_outlet_node": seed_node,
                "outlet_node": out_id,
                "basin_boundary_xy": boundary_xy,
                "outlet_xy": np.array(
                    [cc[out_id] * pixel_size_m, -rr[out_id] * pixel_size_m],
                    dtype=float,
                ),
                "variants": variants,
            }
        )
    return series


# =============================================================================
# Calibration validation, helpers, and cache augmentation
# =============================================================================
#
# Conceptual note: this notebook compares Earth networks (extracted via a
# contributing-area threshold + Strahler pruning) to Mars mapped valleys. To
# avoid circularity, the *calibration* phase tunes complexity and geometric
# representativeness (max Strahler, basin-to-hull representativeness, retained
# stream-length fraction) and the *evaluation* phase reports Dd_hull / Dd_true
# / stream length / hull area as outcomes -- never as direct optimisation
# targets. The helpers below support that split.


def basin_to_hull_area_ratio(
    dd_hull: float | npt.NDArray[np.floating],
    dd_true: float | npt.NDArray[np.floating],
) -> float | npt.NDArray[np.floating]:
    """Basin/hull area ratio derived from the two drainage densities.

    Because both ``Dd_hull`` and ``Dd_true`` share the same stream length::

        Dd_hull / Dd_true = (L / hull_area) / (L / basin_area) = basin_area / hull_area

    A value of 1.0 means the convex hull tracks the true basin perfectly; > 1
    means the hull is *smaller* than the basin (the network has not yet filled
    its catchment); < 1 means the hull spills beyond the basin (rare,
    typically a sign of mapping noise). This is the natural calibration
    target for terrestrial extraction: tune (threshold, prune-depth) until
    the ratio hovers near 1, *then* report Dd_hull as an outcome.
    """
    dd_true_arr = np.asarray(dd_true, dtype=float)
    dd_hull_arr = np.asarray(dd_hull, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(dd_true_arr > 0, dd_hull_arr / dd_true_arr, np.nan)
    if np.isscalar(dd_hull) and np.isscalar(dd_true):
        return float(ratio)
    return ratio


def validate_area_bin_grouping(
    df: pd.DataFrame,
    *,
    area_bin_km2: float = 0.1,
    coord_tol_px: int = 2,
) -> dict[str, Any]:
    """Diagnostic for `(dem_name, area_bin)` outlet grouping.

    Returns a dict with::

        {
          "summary":            pd.DataFrame  -- per-DEM counts
          "merge_groups":       pd.DataFrame  -- (dem, area_bin, threshold)
                                                 with >1 outlet at same thr
          "split_outlets":      pd.DataFrame  -- same physical outlet (coord
                                                 within `coord_tol_px`) that
                                                 falls into >1 area_bin
          "warnings":           list[str]
          "ok":                 bool
        }

    ``df`` must come from ``run_practical_sweep``. If the cache predates the
    outlet-coord patch (no ``outlet_row``/``outlet_col`` columns) the
    coordinate-based split test is skipped and a warning is emitted -- run
    :func:`augment_sweep_with_outlet_coords` first to enable it.
    """
    have_coords = {"outlet_row", "outlet_col"}.issubset(df.columns)
    valid = df[df["status_full"] == "ok"].copy()

    # `status_full == "ok"` is derived from hull status, not basin area, so a
    # row can still carry a NaN/inf basin_area_km2 (e.g. a hand-edited or
    # partially-augmented cache). Drop those before `.astype(int)`, which would
    # otherwise raise "cannot convert float NaN to integer".
    finite_area = np.isfinite(valid["basin_area_km2"].to_numpy(dtype=float))
    n_dropped = int((~finite_area).sum())
    if n_dropped:
        valid = valid[finite_area].copy()
    valid["area_bin"] = (valid["basin_area_km2"] / area_bin_km2).round().astype(int)

    # 1) Merges: more than one outlet at the same (dem, area_bin, threshold).
    per_thr_merge = (
        valid.groupby(["dem_name", "area_bin", "threshold_km2"])["outlet_node"]
        .nunique()
        .reset_index(name="n_outlets")
    )
    merge_groups = per_thr_merge[per_thr_merge["n_outlets"] > 1].copy()

    # 2) Splits: same physical outlet (coords within tolerance) landing in
    # multiple area_bins. Needs outlet_row/col.
    split_outlets = pd.DataFrame()
    if have_coords:
        split_rows = []
        for dem, sub in valid.groupby("dem_name"):
            coords = sub[
                ["outlet_row", "outlet_col", "area_bin", "outlet_node", "threshold_km2"]
            ].copy()
            # Cluster outlets by integer coord rounded to coord_tol_px.
            coords["cluster_r"] = (coords["outlet_row"] // max(coord_tol_px, 1)).astype(int)
            coords["cluster_c"] = (coords["outlet_col"] // max(coord_tol_px, 1)).astype(int)
            bins_per_cluster = (
                coords.groupby(["cluster_r", "cluster_c"])["area_bin"]
                .nunique()
                .reset_index(name="n_area_bins")
            )
            offenders = bins_per_cluster[bins_per_cluster["n_area_bins"] > 1]
            for _, off in offenders.iterrows():
                rows = coords[
                    (coords["cluster_r"] == off["cluster_r"])
                    & (coords["cluster_c"] == off["cluster_c"])
                ]
                split_rows.append(
                    {
                        "dem_name": dem,
                        "cluster_r": int(off["cluster_r"]),
                        "cluster_c": int(off["cluster_c"]),
                        "n_area_bins": int(off["n_area_bins"]),
                        "area_bins": tuple(sorted(rows["area_bin"].unique().tolist())),
                        "n_rows": int(len(rows)),
                    }
                )
        split_outlets = pd.DataFrame(split_rows)

    # 3) Per-DEM summary.
    by_dem = []
    total_grp_per_dem = (
        valid.groupby(["dem_name", "area_bin", "threshold_km2"]).size().reset_index()
    )
    for dem in valid["dem_name"].unique():
        d_total = total_grp_per_dem[total_grp_per_dem["dem_name"] == dem]
        d_merge = merge_groups[merge_groups["dem_name"] == dem]
        by_dem.append(
            {
                "dem_name": dem,
                "n_outlets": int(valid[valid["dem_name"] == dem]["outlet_node"].nunique()),
                "n_groups": int(len(d_total)),
                "n_merge_groups": int(len(d_merge)),
                "pct_merge_groups": (round(100 * len(d_merge) / max(len(d_total), 1), 1)),
            }
        )
    summary = pd.DataFrame(by_dem).sort_values("pct_merge_groups", ascending=False)

    warnings: list[str] = []
    if n_dropped:
        warnings.append(
            f"{n_dropped} row(s) with status_full=='ok' had non-finite "
            f"basin_area_km2 and were excluded from area-bin grouping."
        )
    if not have_coords:
        warnings.append(
            "outlet_row/outlet_col missing; coordinate-split test skipped. "
            "Run augment_sweep_with_outlet_coords() to enable it."
        )
    if not merge_groups.empty:
        warnings.append(
            f"{len(merge_groups)} (dem, area_bin, threshold) groups merge "
            f">=2 distinct outlets -- area-bin grouping is unsafe at "
            f"area_bin_km2={area_bin_km2:g}. Consider outlet-coordinate matching."
        )
    if not split_outlets.empty:
        warnings.append(
            f"{len(split_outlets)} physical outlets straddle a bin boundary "
            f"(coord-tol={coord_tol_px} px) -- the same outlet falls in "
            f"different area_bins at different thresholds."
        )

    return {
        "summary": summary.reset_index(drop=True),
        "merge_groups": merge_groups.reset_index(drop=True),
        "split_outlets": split_outlets.reset_index(drop=True),
        "warnings": warnings,
        "ok": bool(merge_groups.empty and split_outlets.empty),
    }


def augment_sweep_with_outlet_coords(
    df: pd.DataFrame,
    *,
    out_csv: Path | str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Backfill ``outlet_row`` / ``outlet_col`` into an old sweep cache.

    The cache CSV pre-dates the outlet-coord patch in :func:`_dd_trim_rows`;
    re-running the full sweep is expensive, so this helper rebuilds *only*
    the StreamObject at each ``(dem_name, threshold_km2)`` and resolves
    each row's ``outlet_node`` to grid coordinates. The pruning variants
    are untouched.

    Returns a copy of ``df`` with two new columns. If ``out_csv`` is given,
    the augmented frame is also written there.
    """
    if {"outlet_row", "outlet_col"}.issubset(df.columns):
        return df.copy()

    import topotoolbox as tt3

    from .basin_config import LOCAL_TO_PAPER_BASIN, get_basin_config
    from .io.paths import EXAMPLE_DEMS

    out = df.copy()
    out["outlet_row"] = -1
    out["outlet_col"] = -1

    for dem_name, dem_grp in out.groupby("dem_name"):
        dem_path = EXAMPLE_DEMS.get(dem_name)
        if dem_path is None or not dem_path.exists():
            if verbose:
                logger.warning("Skipping %s: DEM not found", dem_name)
            continue
        cfg = get_basin_config(LOCAL_TO_PAPER_BASIN.get(dem_name, dem_name))
        dem = tt3.read_tif(str(dem_path))
        if cfg.get("z_th") is not None:
            dem.z[dem.z < cfg["z_th"]] = np.nan
        pixel_size_m = compute_pixel_size_m_from_dem(dem, lat_deg=cfg["lat"])
        fd = tt3.FlowObject(dem)
        for thr_km2, thr_grp in dem_grp.groupby("threshold_km2"):
            cells = compute_threshold_cells(float(thr_km2), pixel_size_m)
            try:
                s = tt3.StreamObject(fd, threshold=cells)
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    logger.warning(
                        "  %s @ %s: StreamObject build failed (%s)", dem_name, thr_km2, exc
                    )
                continue
            rr, cc = _node_rowcol(s)
            for idx, row in thr_grp.iterrows():
                o = int(row["outlet_node"])
                if 0 <= o < rr.size:
                    out.at[idx, "outlet_row"] = int(rr[o])
                    out.at[idx, "outlet_col"] = int(cc[o])
        if verbose:
            print(f"  augmented {dem_name}: {len(dem_grp)} rows")

    if out_csv is not None:
        out_path = Path(out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_path, index=False)
        if verbose:
            print(f"Wrote {out_path}")
    return out


def match_outlets_by_coords(
    df: pd.DataFrame,
    *,
    coord_tol_px: int = 2,
) -> pd.DataFrame:
    """Cluster sweep rows into per-outlet groups using DEM-grid coordinates.

    Adds an ``outlet_cluster`` column (unique within each ``dem_name``).
    Outlets within ``coord_tol_px`` pixels of each other in row/col are
    treated as the same physical pour point across thresholds. Rows without
    coordinates get cluster id -1 (caller should drop or fall back).
    """
    if not {"outlet_row", "outlet_col"}.issubset(df.columns):
        raise ValueError(
            "match_outlets_by_coords needs outlet_row/outlet_col; "
            "call augment_sweep_with_outlet_coords first."
        )
    out = df.copy()
    out["outlet_cluster"] = -1
    next_id = 0
    for dem_name, sub in out.groupby("dem_name"):
        coord_idx = sub.index[(sub["outlet_row"] >= 0) & (sub["outlet_col"] >= 0)]
        if coord_idx.empty:
            continue
        rc = sub.loc[coord_idx, ["outlet_row", "outlet_col"]].to_numpy()
        # Union-find on the integer-grid Chebyshev distance <= coord_tol_px.
        parent = list(range(rc.shape[0]))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Use a coarse hash to limit pairwise checks.
        cell = (rc // max(coord_tol_px, 1)).astype(int)
        buckets: dict[tuple[int, int], list[int]] = {}
        for i, (cr, cc_) in enumerate(cell):
            buckets.setdefault((int(cr), int(cc_)), []).append(i)
        for (br, bc), members in buckets.items():
            neighbours = []
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    neighbours.extend(buckets.get((br + dr, bc + dc), []))
            for i in members:
                for j in neighbours:
                    if i >= j:
                        continue
                    if (
                        abs(rc[i, 0] - rc[j, 0]) <= coord_tol_px
                        and abs(rc[i, 1] - rc[j, 1]) <= coord_tol_px
                    ):
                        union(i, j)
        roots = {i: find(i) for i in range(rc.shape[0])}
        root_to_cluster: dict[int, int] = {}
        for i, root in roots.items():
            if root not in root_to_cluster:
                root_to_cluster[root] = next_id
                next_id += 1
            out.at[coord_idx[i], "outlet_cluster"] = root_to_cluster[root]
    return out


def length_retention_penalty(
    pct_length_retained: float | npt.NDArray[np.floating],
    *,
    min_retained_pct: float = 30.0,
    weight: float = 1.0,
) -> float | npt.NDArray[np.floating]:
    """Quadratic penalty for over-pruning.

    Returns 0 when ``pct_length_retained >= min_retained_pct``, otherwise
    grows as ``(min - pct) / min``-squared so a candidate that keeps too
    little of the original network is strongly disfavoured. ``weight``
    scales the result. Used as an additive term in the calibration loss.
    """
    pct = np.asarray(pct_length_retained, dtype=float)
    deficit = np.where(pct < min_retained_pct, (min_retained_pct - pct) / min_retained_pct, 0.0)
    penalty = weight * deficit**2
    if np.isscalar(pct_length_retained):
        return float(penalty)
    return penalty
