"""Mars valley-network topology construction.

Moved verbatim (logic-preserving) from the former
``scripts/build_mars_network_topology.py`` so the Phase-1 Mars topology build is
a package function, not a standalone script.

For every mapped Martian valley network this module:

1. Loads valley-network polylines + the MOLA DEM, reprojecting valleys to the
   MOLA CRS and assigning a stable ``network_id``.
2. Explodes MultiLineStrings to LineStrings with stable ``segment_id``.
3. Clusters segment endpoints within ``snap_tolerance_m`` (cKDTree + union-find)
   into graph nodes; each LineString is an edge.
4. Classifies nodes by degree (1=channel head/outlet, >=3=confluence) and
   samples MOLA elevation.
5. Selects the outlet as the lowest-MOLA terminal node and snaps its geometry
   onto the valley vector.
6. Orients each segment by undirected graph distance to the outlet.
7. Assembles QGIS-ready layers and (optionally) a validation PNG.

Public entry point: :func:`build_topology` (in-memory layers) /
:func:`build_and_write_topology` (writes the GeoPackage + validation plot).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from channel_heads.io import paths
from channel_heads.logging_config import get_logger

# Defaults (were module-level constants in the script).
SNAP_TOLERANCE_M = 200.0
NETWORK_ID_COL = "network_id"
OUTLET_METHOD = "lowest_mola_terminal_node"
OUTLET_STATUS = "auto_unchecked"
FLOW_ORIENTATION_METHOD = "graph_distance_from_outlet"
VALIDATION_PLOT_N = 6
VALIDATION_PLOT_SEED = 0

log = get_logger("mars.topology")


# --------------------------------------------------------------------------- #
# Loading / explode
# --------------------------------------------------------------------------- #
def load_valley_networks(path, target_crs, network_id_col: str = NETWORK_ID_COL):
    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise ValueError(f"Valley network file has no CRS: {path}")
    if gdf.crs != target_crs:
        log.info("Reprojecting valley networks to MOLA CRS")
        gdf = gdf.to_crs(target_crs)

    if (
        network_id_col in gdf.columns
        and gdf[network_id_col].notna().all()
        and gdf[network_id_col].is_unique
    ):
        nid = gdf[network_id_col].astype(int).to_numpy()
    else:
        log.warning("No valid unique '%s'; using feature index", network_id_col)
        nid = np.arange(len(gdf), dtype=int)

    gdf = gdf.copy()
    gdf["network_id"] = nid
    gdf["original_feature_id"] = np.arange(len(gdf), dtype=int)
    return gdf


def explode_to_linestrings(gdf):
    import geopandas as gpd

    rows = []
    seg_counter = 0
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "LineString":
            parts = [geom]
        elif geom.geom_type == "MultiLineString":
            parts = list(geom.geoms)
        else:
            log.warning("Skipping geometry %s (network_id=%s)", geom.geom_type, row["network_id"])
            continue
        for part in parts:
            if part.is_empty or len(part.coords) < 2:
                continue
            rows.append(
                {
                    "network_id": int(row["network_id"]),
                    "original_feature_id": int(row["original_feature_id"]),
                    "segment_id": seg_counter,
                    "geometry": part,
                }
            )
            seg_counter += 1
    return gpd.GeoDataFrame(rows, crs=gdf.crs)


# --------------------------------------------------------------------------- #
# Pure topology helpers (unit-testable, no IO)
# --------------------------------------------------------------------------- #
def cluster_endpoints(coords: np.ndarray, snap_tol: float) -> np.ndarray:
    """Union-find cluster endpoints within ``snap_tol`` of each other."""
    from scipy.spatial import cKDTree

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


def classify_node_type(node_id: int, outlet_node_id: int, degree: int) -> str:
    if node_id == outlet_node_id:
        return "outlet"
    if degree == 1:
        return "channel_head"
    if degree >= 3:
        return "confluence"
    return "connector"


def sample_raster(src, points_xy: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.array([], dtype=float)
    coords = [tuple(xy) for xy in points_xy]
    elevs = np.array([s[0] for s in src.sample(coords)], dtype=float)
    if src.nodata is not None:
        elevs[elevs == src.nodata] = np.nan
    return elevs


def pick_outlet_endpoint(cluster_endpoints_xy, centroid_xy, network_geom):
    """Pick the point on the valley vector closest to the cluster centroid.

    Returns ``(snapped_x, snapped_y, snap_distance_m)``.
    """
    from shapely.geometry import Point
    from shapely.ops import nearest_points

    cx, cy = centroid_xy
    if len(cluster_endpoints_xy) > 0:
        dx = cluster_endpoints_xy[:, 0] - cx
        dy = cluster_endpoints_xy[:, 1] - cy
        d = np.hypot(dx, dy)
        pick = int(np.argmin(d))
        return (
            float(cluster_endpoints_xy[pick, 0]),
            float(cluster_endpoints_xy[pick, 1]),
            float(d[pick]),
        )
    centroid_pt = Point(cx, cy)
    snapped_pt, _ = nearest_points(network_geom, centroid_pt)
    return (float(snapped_pt.x), float(snapped_pt.y), float(centroid_pt.distance(snapped_pt)))


def process_network(nid: int, segs_in_net, src, snap_tol: float) -> dict | None:
    """Build the topology dict for a single network (or None if degenerate)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra
    from shapely.ops import unary_union

    segments: list[dict] = []
    endpoint_rows: list[tuple[int, str, float, float]] = []

    for _, row in segs_in_net.iterrows():
        coords = list(row.geometry.coords)
        if len(coords) < 2:
            continue
        seg_idx_local = len(segments)
        segments.append(
            {
                "segment_id": int(row["segment_id"]),
                "original_feature_id": int(row["original_feature_id"]),
                "geometry": row.geometry,
                "length_m": float(row.geometry.length),
            }
        )
        x0, y0 = coords[0][0], coords[0][1]
        x1, y1 = coords[-1][0], coords[-1][1]
        endpoint_rows.append((seg_idx_local, "start", float(x0), float(y0)))
        endpoint_rows.append((seg_idx_local, "end", float(x1), float(y1)))

    if not segments:
        return None

    ep_df = pd.DataFrame(endpoint_rows, columns=["seg_idx_local", "position", "x", "y"])
    coords_arr = ep_df[["x", "y"]].to_numpy()
    cluster_ids = cluster_endpoints(coords_arr, snap_tol)
    ep_df["cluster"] = cluster_ids

    n_nodes = int(cluster_ids.max()) + 1
    node_x = np.zeros(n_nodes, dtype=float)
    node_y = np.zeros(n_nodes, dtype=float)
    node_degree = np.zeros(n_nodes, dtype=int)
    node_connected_segs: list[set[int]] = [set() for _ in range(n_nodes)]

    for cid, group in ep_df.groupby("cluster"):
        cid_i = int(cid)
        node_x[cid_i] = float(group["x"].mean())
        node_y[cid_i] = float(group["y"].mean())
        seg_set = {int(v) for v in group["seg_idx_local"].tolist()}
        node_connected_segs[cid_i] = seg_set
        node_degree[cid_i] = len(seg_set)

    seg_start_node = np.full(len(segments), -1, dtype=int)
    seg_end_node = np.full(len(segments), -1, dtype=int)
    for _, row in ep_df.iterrows():
        si = int(row["seg_idx_local"])
        cid_i = int(row["cluster"])
        if row["position"] == "start":
            seg_start_node[si] = cid_i
        else:
            seg_end_node[si] = cid_i

    node_coords = np.column_stack([node_x, node_y])
    node_elev = sample_raster(src, node_coords)

    terminal_ids = np.where(node_degree == 1)[0]
    if len(terminal_ids) > 0:
        elev_terms = node_elev[terminal_ids]
        valid = ~np.isnan(elev_terms)
        if valid.any():
            ranked = np.where(valid, elev_terms, np.inf)
            outlet_node_id = int(terminal_ids[int(np.argmin(ranked))])
        else:
            outlet_node_id = int(terminal_ids[0])
            log.warning("Network %s: all terminals nodata; first terminal as outlet", nid)
    else:
        positive = node_degree[node_degree > 0]
        if positive.size == 0:
            return None
        min_deg = int(positive.min())
        cand_ids = np.where(node_degree == min_deg)[0]
        elev_cand = node_elev[cand_ids]
        valid = ~np.isnan(elev_cand)
        if valid.any():
            ranked = np.where(valid, elev_cand, np.inf)
            outlet_node_id = int(cand_ids[int(np.argmin(ranked))])
        else:
            outlet_node_id = int(cand_ids[0])

    cluster_eps = ep_df[ep_df["cluster"] == outlet_node_id]
    cluster_xy = cluster_eps[["x", "y"]].to_numpy()
    network_geom = unary_union([s["geometry"] for s in segments])
    centroid_xy = (float(node_x[outlet_node_id]), float(node_y[outlet_node_id]))
    snapped_x, snapped_y, snap_distance_m = pick_outlet_endpoint(cluster_xy, centroid_xy, network_geom)

    distances = np.full(n_nodes, np.nan, dtype=float)
    rows_g: list[int] = []
    cols_g: list[int] = []
    data_g: list[float] = []
    for si, seg in enumerate(segments):
        u = int(seg_start_node[si])
        v = int(seg_end_node[si])
        if u < 0 or v < 0 or u == v:
            continue
        rows_g.append(u)
        cols_g.append(v)
        data_g.append(seg["length_m"])
    if rows_g:
        adj = csr_matrix((data_g, (rows_g, cols_g)), shape=(n_nodes, n_nodes))
        dist_arr = np.asarray(dijkstra(adj, directed=False, indices=outlet_node_id), dtype=float)
        dist_arr[np.isinf(dist_arr)] = np.nan
        distances = dist_arr
    else:
        distances[outlet_node_id] = 0.0

    seg_upstream = np.full(len(segments), -1, dtype=int)
    seg_downstream = np.full(len(segments), -1, dtype=int)
    for si in range(len(segments)):
        u = int(seg_start_node[si])
        v = int(seg_end_node[si])
        du = distances[u] if u >= 0 else np.nan
        dv = distances[v] if v >= 0 else np.nan
        if np.isnan(du) and np.isnan(dv):
            seg_upstream[si], seg_downstream[si] = u, v
        elif np.isnan(du):
            seg_upstream[si], seg_downstream[si] = u, v
        elif np.isnan(dv):
            seg_upstream[si], seg_downstream[si] = v, u
        elif du >= dv:
            seg_upstream[si], seg_downstream[si] = u, v
        else:
            seg_upstream[si], seg_downstream[si] = v, u

    node_types = np.array(
        [classify_node_type(i, outlet_node_id, int(node_degree[i])) for i in range(n_nodes)],
        dtype=object,
    )

    return {
        "nid": int(nid),
        "segments": segments,
        "seg_start_node": seg_start_node,
        "seg_end_node": seg_end_node,
        "seg_upstream": seg_upstream,
        "seg_downstream": seg_downstream,
        "node_x": node_x,
        "node_y": node_y,
        "node_degree": node_degree,
        "node_elev": node_elev,
        "node_types": node_types,
        "node_distances": distances,
        "node_connected_segs": node_connected_segs,
        "outlet_node_id": int(outlet_node_id),
        "outlet_original_x": float(node_x[outlet_node_id]),
        "outlet_original_y": float(node_y[outlet_node_id]),
        "outlet_snapped_x": float(snapped_x),
        "outlet_snapped_y": float(snapped_y),
        "outlet_snap_distance_m": float(snap_distance_m),
        "network_geom": network_geom,
    }


def build_layers(results: list[dict], crs) -> dict:
    """Assemble the per-network results into the 7 GeoPackage layers."""
    import geopandas as gpd
    from shapely.geometry import LineString, MultiLineString, Point

    networks_rows, networks_geom = [], []
    segments_rows, segments_geom = [], []
    nodes_rows, nodes_geom = [], []
    terminal_rows, terminal_geom = [], []
    outlets_rows, outlets_geom = [], []
    heads_rows, heads_geom = [], []
    confluence_rows, confluence_geom = [], []

    for res in results:
        nid = res["nid"]
        segments = res["segments"]
        outlet_id = res["outlet_node_id"]
        node_x, node_y = res["node_x"], res["node_y"]
        node_degree, node_elev = res["node_degree"], res["node_elev"]
        node_types, node_distances = res["node_types"], res["node_distances"]
        node_connected_segs = res["node_connected_segs"]
        n_nodes = len(node_x)

        n_terminal = int((node_degree == 1).sum())
        n_confluence = int((node_degree >= 3).sum())
        n_channel_heads = max(n_terminal - 1, 0)
        total_length = float(sum(s["length_m"] for s in segments))
        networks_rows.append(
            {
                "network_id": int(nid),
                "total_length_m": total_length,
                "n_segments": int(len(segments)),
                "n_nodes": int(n_nodes),
                "n_terminal_nodes": n_terminal,
                "n_channel_heads": n_channel_heads,
                "n_confluences": n_confluence,
                "outlet_node_id": int(outlet_id),
                "outlet_mola_elev": (
                    float(node_elev[outlet_id]) if not np.isnan(node_elev[outlet_id]) else None
                ),
                "outlet_method": OUTLET_METHOD,
                "outlet_status": OUTLET_STATUS,
            }
        )
        net_geom = res["network_geom"]
        if isinstance(net_geom, LineString):
            net_geom = MultiLineString([net_geom])
        networks_geom.append(net_geom)

        for si, seg in enumerate(segments):
            segments_rows.append(
                {
                    "network_id": int(nid),
                    "segment_id": int(seg["segment_id"]),
                    "original_feature_id": int(seg["original_feature_id"]),
                    "start_node_id": int(res["seg_start_node"][si]),
                    "end_node_id": int(res["seg_end_node"][si]),
                    "upstream_node_id": int(res["seg_upstream"][si]),
                    "downstream_node_id": int(res["seg_downstream"][si]),
                    "length_m": float(seg["length_m"]),
                    "flow_orientation_method": FLOW_ORIENTATION_METHOD,
                }
            )
            segments_geom.append(seg["geometry"])

        head_counter = 0
        confluence_counter = 0
        for i in range(n_nodes):
            elev_v = float(node_elev[i]) if not np.isnan(node_elev[i]) else None
            dist_v = float(node_distances[i]) if not np.isnan(node_distances[i]) else None
            nodes_rows.append(
                {
                    "network_id": int(nid),
                    "node_id": int(i),
                    "node_degree": int(node_degree[i]),
                    "node_type": str(node_types[i]),
                    "mola_elev": elev_v,
                    "distance_to_outlet_m": dist_v,
                    "n_connected_segments": int(len(node_connected_segs[i])),
                }
            )
            nodes_geom.append(Point(float(node_x[i]), float(node_y[i])))

            if int(node_degree[i]) == 1:
                is_outlet = i == outlet_id
                is_head = not is_outlet
                terminal_rows.append(
                    {
                        "network_id": int(nid),
                        "node_id": int(i),
                        "is_outlet": bool(is_outlet),
                        "is_channel_head": bool(is_head),
                        "mola_elev": elev_v,
                        "distance_to_outlet_m": dist_v,
                    }
                )
                terminal_geom.append(Point(float(node_x[i]), float(node_y[i])))
                if is_head:
                    heads_rows.append(
                        {
                            "network_id": int(nid),
                            "head_id": int(head_counter),
                            "node_id": int(i),
                            "mola_elev": elev_v,
                            "distance_to_outlet_m": dist_v,
                        }
                    )
                    heads_geom.append(Point(float(node_x[i]), float(node_y[i])))
                    head_counter += 1
            if int(node_degree[i]) >= 3:
                confluence_rows.append(
                    {
                        "network_id": int(nid),
                        "confluence_id": int(confluence_counter),
                        "node_id": int(i),
                        "node_degree": int(node_degree[i]),
                        "mola_elev": elev_v,
                        "distance_to_outlet_m": dist_v,
                        "n_connected_segments": int(len(node_connected_segs[i])),
                    }
                )
                confluence_geom.append(Point(float(node_x[i]), float(node_y[i])))
                confluence_counter += 1

        outlets_rows.append(
            {
                "network_id": int(nid),
                "outlet_node_id": int(outlet_id),
                "mola_elev": (
                    float(node_elev[outlet_id]) if not np.isnan(node_elev[outlet_id]) else None
                ),
                "outlet_method": OUTLET_METHOD,
                "outlet_status": OUTLET_STATUS,
                "snap_distance_m": float(res["outlet_snap_distance_m"]),
                "original_x": float(res["outlet_original_x"]),
                "original_y": float(res["outlet_original_y"]),
                "snapped_x": float(res["outlet_snapped_x"]),
                "snapped_y": float(res["outlet_snapped_y"]),
            }
        )
        outlets_geom.append(Point(float(res["outlet_snapped_x"]), float(res["outlet_snapped_y"])))

    return {
        "mars_networks": gpd.GeoDataFrame(networks_rows, geometry=networks_geom, crs=crs),
        "mars_segments": gpd.GeoDataFrame(segments_rows, geometry=segments_geom, crs=crs),
        "mars_nodes": gpd.GeoDataFrame(nodes_rows, geometry=nodes_geom, crs=crs),
        "mars_terminal_nodes": gpd.GeoDataFrame(terminal_rows, geometry=terminal_geom, crs=crs),
        "mars_outlets": gpd.GeoDataFrame(outlets_rows, geometry=outlets_geom, crs=crs),
        "mars_channel_heads": gpd.GeoDataFrame(heads_rows, geometry=heads_geom, crs=crs),
        "mars_confluences": gpd.GeoDataFrame(confluence_rows, geometry=confluence_geom, crs=crs),
    }


def write_topology_geopackage(layers: dict, output_path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    for name, gdf in layers.items():
        if gdf.empty:
            log.warning("Layer '%s' is empty; skipping", name)
            continue
        gdf.to_file(output_path, layer=name, driver="GPKG")
        log.info("Wrote layer '%s' (%d rows)", name, len(gdf))
    return output_path


# --------------------------------------------------------------------------- #
# High-level entry points
# --------------------------------------------------------------------------- #
def build_topology(
    valleys_path=paths.MARS_VALLEYS,
    mola_path=paths.MARS_DEM,
    *,
    snap_tolerance_m: float = SNAP_TOLERANCE_M,
    network_id_col: str = NETWORK_ID_COL,
) -> dict:
    """Build the Mars topology layers in memory (no files written)."""
    import rasterio
    from tqdm import tqdm

    with rasterio.open(mola_path) as src:
        mola_crs = src.crs
    valleys_gdf = load_valley_networks(valleys_path, mola_crs, network_id_col)
    log.info("Loaded %d valley-network features", len(valleys_gdf))
    segments_gdf = explode_to_linestrings(valleys_gdf)
    log.info("Exploded to %d LineString segments", len(segments_gdf))

    results: list[dict] = []
    with rasterio.open(mola_path) as src:
        grouped = segments_gdf.groupby("network_id", sort=True)
        for nid, segs in tqdm(grouped, total=grouped.ngroups, desc="Networks"):
            res = process_network(int(nid), segs, src, snap_tolerance_m)
            if res is not None:
                results.append(res)
    log.info("Built topology for %d networks", len(results))
    return build_layers(results, valleys_gdf.crs)


def build_and_write_topology(
    valleys_path=paths.MARS_VALLEYS,
    mola_path=paths.MARS_DEM,
    output_path=paths.MARS_TOPOLOGY_GPKG,
    *,
    snap_tolerance_m: float = SNAP_TOLERANCE_M,
    make_validation_plot: bool = True,
) -> Path:
    """Full Phase-1 build: layers → GeoPackage (+ validation PNG). Returns path."""
    layers = build_topology(valleys_path, mola_path, snap_tolerance_m=snap_tolerance_m)
    out = write_topology_geopackage(layers, output_path)
    if make_validation_plot:
        plot_path = Path(output_path).parent / "validation_plot.png"
        try:
            validation_plot(layers, plot_path, n=VALIDATION_PLOT_N, seed=VALIDATION_PLOT_SEED)
        except Exception as exc:  # plotting is best-effort
            log.warning("Validation plot failed: %s", exc)
    return out


def validation_plot(layers: dict, output_path, n: int = VALIDATION_PLOT_N, seed: int = 0) -> None:
    import matplotlib.pyplot as plt

    networks = layers["mars_networks"]
    segments = layers["mars_segments"]
    nodes = layers["mars_nodes"]
    outlets = layers["mars_outlets"]
    heads = layers["mars_channel_heads"]
    confluences = layers["mars_confluences"]

    rng = np.random.default_rng(seed)
    nids = networks["network_id"].to_numpy()
    if len(nids) == 0:
        log.warning("No networks to plot")
        return
    sample_n = min(n, len(nids))
    sample_nids = rng.choice(nids, size=sample_n, replace=False)
    cols = min(3, sample_n)
    rows = int(np.ceil(sample_n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, nid in zip(axes, sample_nids):
        nid = int(nid)
        segs = segments[segments["network_id"] == nid]
        nds = nodes[nodes["network_id"] == nid]
        out = outlets[outlets["network_id"] == nid]
        hds = heads[heads["network_id"] == nid]
        conf = confluences[confluences["network_id"] == nid]
        segs.plot(ax=ax, color="black", linewidth=0.7)
        nds.plot(ax=ax, color="lightgray", markersize=10)
        if not conf.empty:
            conf.plot(ax=ax, color="tab:orange", markersize=35, marker="s")
        if not hds.empty:
            hds.plot(ax=ax, color="tab:blue", markersize=30)
        out.plot(ax=ax, color="red", markersize=120, marker="*")
        elev_v = out["mola_elev"].iloc[0]
        elev_txt = "n/a" if pd.isna(elev_v) else f"{elev_v:.0f} m"
        snap_v = out["snap_distance_m"].iloc[0]
        ax.set_title(f"network_id={nid}\nelev={elev_txt} snap={snap_v:.1f}m", fontsize=9)
        ax.set_aspect("equal")
    for ax in axes[sample_n:]:
        ax.set_visible(False)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    log.info("Wrote validation plot: %s", output_path)


__all__ = [
    "build_topology",
    "build_and_write_topology",
    "build_layers",
    "process_network",
    "cluster_endpoints",
    "classify_node_type",
    "pick_outlet_endpoint",
    "load_valley_networks",
    "explode_to_linestrings",
    "write_topology_geopackage",
    "validation_plot",
]
