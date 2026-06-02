"""Mars first-meet channel-head pair extraction (Phase 2B).

Moved from ``scripts/extract_mars_first_meet_pairs.py``. For every Martian
valley network this reads the Phase-1 topology GeoPackage and computes all
*first-meet* channel-head pairs at confluences using the shared, graph-agnostic
core in :mod:`channel_heads.pairing` (the same algorithm as the Earth pipeline),
operating on the Phase-1 directed graph (``upstream_node_id`` →
``downstream_node_id``).

Public entry point: :func:`extract_first_meet_pairs`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from channel_heads.io import paths
from channel_heads.logging_config import get_logger
from channel_heads.pairing import (
    build_directed_adjacency,
    chain_segment_geometries,
    first_meet_pairs_on_dag,
    trace_downstream_path,
)

VALIDATION_PLOT_N = 6
VALIDATION_PLOT_SEED = 0

log = get_logger("mars.pairs")


def load_topology_layers(gpkg_path):
    """Load mars_networks, mars_segments, mars_nodes from the Phase-1 GPKG."""
    import geopandas as gpd

    gpkg_path = Path(gpkg_path)
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Topology GeoPackage not found: {gpkg_path}")
    networks = gpd.read_file(gpkg_path, layer="mars_networks")
    segments = gpd.read_file(gpkg_path, layer="mars_segments")
    nodes = gpd.read_file(gpkg_path, layer="mars_nodes")
    log.info("Loaded topology: %d networks, %d segments, %d nodes", len(networks), len(segments), len(nodes))
    return networks, segments, nodes


def compute_first_meet_pairs(parents, children, heads_set, confluences_set):
    """Emit cross-branch first-meet pairs (delegates to pairing core)."""
    n = len(parents)
    pairs, _ = first_meet_pairs_on_dag(parents, set(range(n)), heads_set, confluences_set)
    return pairs


def process_network(nid, segs_in_net, nodes_in_net):
    """Compute pair rows + path rows for one network (or None)."""
    from shapely.geometry import MultiLineString

    if segs_in_net.empty:
        return None

    n_nodes = int(nodes_in_net["node_id"].max()) + 1
    parents, children, edge_to_seg = build_directed_adjacency(segs_in_net, n_nodes)

    nt_by_id = dict(zip(nodes_in_net["node_id"], nodes_in_net["node_type"]))
    heads_set = {int(k) for k, nt in nt_by_id.items() if nt == "channel_head"}
    confluences_set = {int(k) for k, nt in nt_by_id.items() if nt == "confluence"}
    if len(heads_set) < 2:
        return None

    pair_dict = compute_first_meet_pairs(parents, children, heads_set, confluences_set)
    if not pair_dict:
        return None

    segment_geom_by_id = dict(zip(segs_in_net["segment_id"], segs_in_net.geometry))
    start_node_by_seg = dict(zip(segs_in_net["segment_id"], segs_in_net["start_node_id"]))
    end_node_by_seg = dict(zip(segs_in_net["segment_id"], segs_in_net["end_node_id"]))
    node_xy = {
        int(r["node_id"]): (float(r.geometry.x), float(r.geometry.y))
        for _, r in nodes_in_net.iterrows()
    }

    pair_rows: list[dict] = []
    path_rows: list[dict] = []
    max_steps = n_nodes + 1
    pair_counter = 0

    for conf_id, pair_set in pair_dict.items():
        for h1, h2 in sorted(pair_set):
            path_a = trace_downstream_path(int(h1), int(conf_id), children, max_steps)
            path_b = trace_downstream_path(int(h2), int(conf_id), children, max_steps)
            if path_a is None or path_b is None:
                continue
            geom_a, segs_a = chain_segment_geometries(
                path_a, edge_to_seg, segment_geom_by_id, start_node_by_seg, end_node_by_seg
            )
            geom_b, segs_b = chain_segment_geometries(
                path_b, edge_to_seg, segment_geom_by_id, start_node_by_seg, end_node_by_seg
            )
            if geom_a is None or geom_b is None:
                continue
            L_1, L_2 = float(geom_a.length), float(geom_b.length)
            x1, y1 = node_xy[int(h1)]
            x2, y2 = node_xy[int(h2)]
            headhead_dist_m = float(np.hypot(x1 - x2, y1 - y2))
            pair_id = f"{int(nid)}_{int(conf_id)}_{int(h1)}_{int(h2)}"
            pair_rows.append(
                {
                    "pair_id": pair_id,
                    "network_id": int(nid),
                    "confluence_node_id": int(conf_id),
                    "head_1_node_id": int(h1),
                    "head_2_node_id": int(h2),
                    "L_1": L_1,
                    "L_2": L_2,
                    "L_sum": L_1 + L_2,
                    "headhead_dist_m": headhead_dist_m,
                    "n_segments_a": int(len(segs_a)),
                    "n_segments_b": int(len(segs_b)),
                    "path_a_node_ids": ",".join(str(n) for n in path_a),
                    "path_b_node_ids": ",".join(str(n) for n in path_b),
                    "path_a_segment_ids": ",".join(str(s) for s in segs_a),
                    "path_b_segment_ids": ",".join(str(s) for s in segs_b),
                    "geometry": MultiLineString([geom_a, geom_b]),
                }
            )
            for branch, head, conf, length, path, segs, geom in (
                ("A", h1, conf_id, L_1, path_a, segs_a, geom_a),
                ("B", h2, conf_id, L_2, path_b, segs_b, geom_b),
            ):
                path_rows.append(
                    {
                        "pair_id": pair_id,
                        "network_id": int(nid),
                        "branch": branch,
                        "head_node_id": int(head),
                        "confluence_node_id": int(conf),
                        "length_m": length,
                        "n_segments": int(len(segs)),
                        "path_node_ids": ",".join(str(n) for n in path),
                        "path_segment_ids": ",".join(str(s) for s in segs),
                        "geometry": geom,
                    }
                )
            pair_counter += 1

    if pair_counter == 0:
        return None
    return pair_rows, path_rows


def write_pair_layers(pair_rows, path_rows, crs, output_path) -> Path:
    import geopandas as gpd

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    gpd.GeoDataFrame(pair_rows, crs=crs).to_file(output_path, layer="mars_pairs", driver="GPKG")
    gpd.GeoDataFrame(path_rows, crs=crs).to_file(output_path, layer="mars_pair_paths", driver="GPKG")
    log.info("Wrote mars_pairs (%d) + mars_pair_paths (%d) -> %s", len(pair_rows), len(path_rows), output_path)
    return output_path


def extract_first_meet_pairs(
    topology_gpkg=paths.MARS_TOPOLOGY_GPKG,
    output_gpkg=paths.MARS_PAIRS_GPKG,
) -> Path:
    """Full Phase-2B extraction: topology GPKG → mars_vn_pairs.gpkg. Returns path."""
    from tqdm import tqdm

    networks, segments, nodes = load_topology_layers(topology_gpkg)
    grouped_segments = dict(tuple(segments.groupby("network_id", sort=True)))
    grouped_nodes = dict(tuple(nodes.groupby("network_id", sort=True)))

    all_pair_rows: list[dict] = []
    all_path_rows: list[dict] = []
    for nid in tqdm(sorted(grouped_segments.keys()), desc="Networks"):
        segs_in_net = grouped_segments[nid]
        nodes_in_net = grouped_nodes.get(int(nid))
        if nodes_in_net is None or nodes_in_net.empty:
            continue
        result = process_network(int(nid), segs_in_net, nodes_in_net)
        if result is None:
            continue
        pair_rows, path_rows = result
        all_pair_rows.extend(pair_rows)
        all_path_rows.extend(path_rows)

    log.info("Total pairs: %d  pair-paths: %d", len(all_pair_rows), len(all_path_rows))
    if all_pair_rows:
        df = pd.DataFrame(all_pair_rows)
        per_net = df.groupby("network_id").size()
        log.info("Pairs/network mean=%.2f median=%.0f max=%d", per_net.mean(), per_net.median(), per_net.max())
    return write_pair_layers(all_pair_rows, all_path_rows, segments.crs, output_gpkg)


__all__ = [
    "extract_first_meet_pairs",
    "process_network",
    "compute_first_meet_pairs",
    "load_topology_layers",
    "write_pair_layers",
]
