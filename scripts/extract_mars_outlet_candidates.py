#!/usr/bin/env python
"""Estimate outlet candidates for Martian valley networks from MOLA.

For every mapped valley network, the script:

  1. Loads valley-network polylines and the MOLA DEM, reprojecting valleys to
     the MOLA CRS if needed.
  2. Clips MOLA to the buffered total bounding box of all networks and writes
     ``mola_clipped_to_valley_networks.tif`` (or falls back to lazy windowed
     sampling from the source raster when the clip would be too large).
  3. Explodes MultiLineStrings to LineStrings, extracts segment endpoints,
     and clusters them with a configurable snap tolerance.
  4. Treats clusters touched by exactly one segment endpoint as terminal
     nodes (graph degree 1).
  5. Samples MOLA at each terminal node and picks the lowest valid elevation
     as that network's outlet candidate.
  6. Writes a GeoPackage with two layers (``outlet_candidates``,
     ``all_terminal_nodes``) plus a small validation PNG for random networks.

Run:
    python scripts/extract_mars_outlet_candidates.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window, from_bounds
from scipy.spatial import cKDTree
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALLEY_PATH = PROJECT_ROOT / "data/final_valleys/final_valleys_fixed.gpkg"
MOLA_PATH = PROJECT_ROOT / "data/Mars/Mars_DEM_reprojected.tif"
OUTPUT_DIR = PROJECT_ROOT / "data/Mars/outlet_candidates"
NETWORK_ID_COL = "network_id"
BUFFER_M = 10_000.0
SNAP_TOLERANCE_M = 200.0
OUTPUT_GPKG = "mars_vn_outlet_candidates.gpkg"

# Guard for the clip-and-save step. ~1.5 GB at int16; above this we skip
# materializing a clipped GeoTIFF and sample from the source raster instead.
MAX_CLIP_PIXELS = 750_000_000

VALIDATION_PLOT_N = 6
VALIDATION_PLOT_SEED = 0

log = logging.getLogger("mars_outlets")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_valley_networks(
    path: Path, target_crs, network_id_col: str
) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Valley network file has no CRS: {path}")
    if gdf.crs != target_crs:
        log.info("Reprojecting valley networks to MOLA CRS")
        gdf = gdf.to_crs(target_crs)
    else:
        log.info("Valley network CRS already matches MOLA CRS")

    if (
        network_id_col in gdf.columns
        and gdf[network_id_col].notna().all()
        and gdf[network_id_col].is_unique
    ):
        nid = gdf[network_id_col].astype(int).to_numpy()
        log.info("Using existing '%s' column as network_id", network_id_col)
    else:
        log.warning(
            "No valid unique '%s' column; falling back to feature index",
            network_id_col,
        )
        nid = np.arange(len(gdf), dtype=int)

    gdf = gdf.copy()
    gdf["network_id"] = nid
    return gdf


def clip_mola(
    src: rasterio.io.DatasetReader,
    bounds: tuple[float, float, float, float],
    buffer_m: float,
    output_path: Path,
    max_pixels: int,
) -> Optional[Path]:
    minx, miny, maxx, maxy = bounds
    bbox = (minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m)
    window = from_bounds(*bbox, transform=src.transform)
    window = window.round_offsets(op="floor").round_lengths(op="ceil")
    full = Window(0, 0, src.width, src.height)
    window = window.intersection(full)
    n_pixels = int(window.width) * int(window.height)
    log.info(
        "Clipped MOLA window: %d x %d px (%d total)",
        window.width,
        window.height,
        n_pixels,
    )
    if n_pixels > max_pixels:
        log.warning(
            "Clip would exceed %d pixels; skipping save and sampling from "
            "the source MOLA instead.",
            max_pixels,
        )
        return None

    new_transform = src.window_transform(window)
    profile = src.profile.copy()
    profile.update(
        {
            "height": int(window.height),
            "width": int(window.width),
            "transform": new_transform,
            "compress": "deflate",
            "tiled": True,
            "blockxsize": 512,
            "blockysize": 512,
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream the clip block-by-block to keep memory bounded.
    with rasterio.open(output_path, "w", **profile) as dst:
        block_h = 1024
        for row_off in range(0, int(window.height), block_h):
            h = min(block_h, int(window.height) - row_off)
            sub = Window(
                col_off=window.col_off,
                row_off=window.row_off + row_off,
                width=window.width,
                height=h,
            )
            dst.write(src.read(1, window=sub), 1, window=Window(0, row_off, window.width, h))
    log.info("Wrote clipped MOLA: %s", output_path)
    return output_path


def explode_to_linestrings(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            parts = [geom]
        elif geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        else:
            log.warning(
                "Skipping unexpected geometry %s for network_id=%s",
                geom.geom_type,
                row["network_id"],
            )
            continue
        for part in parts:
            if part.is_empty or len(part.coords) < 2:
                continue
            rows.append({"network_id": int(row["network_id"]), "geometry": part})
    return gpd.GeoDataFrame(rows, crs=gdf.crs)


def cluster_endpoints(coords: np.ndarray, snap_tol: float) -> np.ndarray:
    """Union-find cluster endpoints within ``snap_tol`` of each other."""
    n = len(coords)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.zeros(1, dtype=int)
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=snap_tol, output_type="ndarray")

    parent = np.arange(n)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[ra] = rb

    roots = np.fromiter((find(i) for i in range(n)), dtype=int, count=n)
    _, cluster_id = np.unique(roots, return_inverse=True)
    return cluster_id


def sample_raster(src: rasterio.io.DatasetReader, points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.array([], dtype=float)
    coords = [tuple(xy) for xy in points_xy]
    elevs = np.array([s[0] for s in src.sample(coords)], dtype=float)
    if src.nodata is not None:
        elevs[elevs == src.nodata] = np.nan
    return elevs


def process_networks(
    segments_gdf: gpd.GeoDataFrame,
    src: rasterio.io.DatasetReader,
    snap_tol: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_terminals: list[pd.DataFrame] = []
    outlets: list[pd.DataFrame] = []

    grouped = segments_gdf.groupby("network_id", sort=True)
    for nid, segs in tqdm(grouped, total=grouped.ngroups, desc="Networks"):
        endpoints: list[tuple[float, float]] = []
        for geom in segs.geometry:
            coords = list(geom.coords)
            endpoints.append(coords[0])
            endpoints.append(coords[-1])
        if not endpoints:
            continue
        coords_arr = np.asarray(endpoints, dtype=float)
        cluster_ids = cluster_endpoints(coords_arr, snap_tol)
        df = pd.DataFrame(
            {
                "cluster": cluster_ids,
                "x": coords_arr[:, 0],
                "y": coords_arr[:, 1],
            }
        )
        agg = (
            df.groupby("cluster")
            .agg(x=("x", "mean"), y=("y", "mean"), node_degree=("x", "size"))
            .reset_index(drop=True)
        )

        terminals = agg[agg["node_degree"] == 1].copy()
        if terminals.empty:
            # Fallback: closed network (loops). Take the lowest-degree clusters.
            min_deg = int(agg["node_degree"].min())
            terminals = agg[agg["node_degree"] == min_deg].copy()
            log.debug(
                "Network %s has no degree-1 nodes; falling back to degree=%d clusters",
                nid,
                min_deg,
            )

        elevs = sample_raster(src, terminals[["x", "y"]].to_numpy())
        terminals["mola_elev"] = elevs
        terminals["network_id"] = int(nid)
        terminals.reset_index(drop=True, inplace=True)

        valid = terminals.dropna(subset=["mola_elev"])
        if not valid.empty:
            outlet_idx = int(valid["mola_elev"].idxmin())
        else:
            outlet_idx = int(terminals.index[0])
            log.warning(
                "All terminal nodes for network %s sampled as nodata; "
                "selecting the first terminal as outlet",
                nid,
            )

        terminals["is_selected_outlet"] = np.arange(len(terminals)) == outlet_idx
        all_terminals.append(terminals)

        out_row = terminals.loc[[outlet_idx]].copy()
        out_row["n_terminal_nodes"] = int(len(terminals))
        outlets.append(out_row)

    terminals_df = pd.concat(all_terminals, ignore_index=True)
    outlets_df = pd.concat(outlets, ignore_index=True)
    return terminals_df, outlets_df


def make_geodataframe(df: pd.DataFrame, crs) -> gpd.GeoDataFrame:
    geom = gpd.points_from_xy(df["x"], df["y"])
    return gpd.GeoDataFrame(df.drop(columns=["x", "y"]), geometry=geom, crs=crs)


def validation_plot(
    valleys_gdf: gpd.GeoDataFrame,
    terminals_gdf: gpd.GeoDataFrame,
    outlets_gdf: gpd.GeoDataFrame,
    output_path: Path,
    n: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    nids = outlets_gdf["network_id"].unique()
    sample_n = min(n, len(nids))
    sample_nids = rng.choice(nids, size=sample_n, replace=False)
    cols = min(3, sample_n)
    rows = int(np.ceil(sample_n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.atleast_1d(axes).ravel()

    for ax, nid in zip(axes, sample_nids):
        vlines = valleys_gdf[valleys_gdf["network_id"] == nid]
        terms = terminals_gdf[terminals_gdf["network_id"] == nid]
        outlet = outlets_gdf[outlets_gdf["network_id"] == nid]
        vlines.plot(ax=ax, color="black", linewidth=0.6)
        terms.plot(ax=ax, color="tab:blue", markersize=18, label="terminal")
        outlet.plot(
            ax=ax,
            color="red",
            markersize=120,
            marker="*",
            label="selected outlet",
        )
        elev = outlet["mola_elev"].iloc[0]
        ax.set_title(
            f"network_id={int(nid)}  outlet_elev={elev:.0f} m",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        ax.legend(loc="best", fontsize=7)
    for ax in axes[sample_n:]:
        ax.set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.info("Wrote validation plot: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with rasterio.open(MOLA_PATH) as src:
        mola_crs = src.crs
        log.info(
            "Original MOLA: shape=%s, res=%s m, crs=%s",
            src.shape,
            src.res,
            src.crs.to_string() if src.crs else "<none>",
        )

    valleys_gdf = load_valley_networks(VALLEY_PATH, mola_crs, NETWORK_ID_COL)
    log.info("Loaded %d valley networks", len(valleys_gdf))

    clipped_path = OUTPUT_DIR / "mola_clipped_to_valley_networks.tif"
    with rasterio.open(MOLA_PATH) as src:
        bounds = tuple(valleys_gdf.total_bounds)
        clipped_out = clip_mola(src, bounds, BUFFER_M, clipped_path, MAX_CLIP_PIXELS)

    sample_path = clipped_out if clipped_out is not None else MOLA_PATH
    with rasterio.open(sample_path) as src:
        log.info(
            "Sampling from: %s (shape=%s)",
            Path(sample_path).name,
            src.shape,
        )
        segments_gdf = explode_to_linestrings(valleys_gdf)
        log.info("Exploded to %d LineString segments", len(segments_gdf))

        terminals_df, outlets_df = process_networks(
            segments_gdf, src, SNAP_TOLERANCE_M
        )

    log.info("Total terminal nodes: %d", len(terminals_df))
    log.info("Outlet candidates: %d", len(outlets_df))

    terminals_df["is_selected_outlet"] = terminals_df["is_selected_outlet"].astype(bool)
    outlets_df["method"] = "lowest_terminal_node_sampled_from_MOLA"
    outlets_df["confidence_auto"] = "unchecked"

    outlets_cols = [
        "network_id",
        "mola_elev",
        "node_degree",
        "n_terminal_nodes",
        "method",
        "confidence_auto",
        "x",
        "y",
    ]
    terminals_cols = [
        "network_id",
        "mola_elev",
        "node_degree",
        "is_selected_outlet",
        "x",
        "y",
    ]
    outlets_df = outlets_df[outlets_cols]
    terminals_df = terminals_df[terminals_cols]

    outlets_gdf = make_geodataframe(outlets_df, mola_crs)
    terminals_gdf = make_geodataframe(terminals_df, mola_crs)

    output_gpkg_path = OUTPUT_DIR / OUTPUT_GPKG
    if output_gpkg_path.exists():
        output_gpkg_path.unlink()
    outlets_gdf.to_file(output_gpkg_path, layer="outlet_candidates", driver="GPKG")
    terminals_gdf.to_file(
        output_gpkg_path, layer="all_terminal_nodes", driver="GPKG"
    )
    log.info("Wrote GeoPackage: %s", output_gpkg_path)

    plot_path = OUTPUT_DIR / "validation_plot.png"
    validation_plot(
        valleys_gdf,
        terminals_gdf,
        outlets_gdf,
        plot_path,
        n=VALIDATION_PLOT_N,
        seed=VALIDATION_PLOT_SEED,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
