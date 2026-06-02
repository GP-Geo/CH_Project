#!/usr/bin/env python
"""Phase 2B — Mars first-meet channel-head pair extraction.

Reads the Phase-1 Mars topology GeoPackage and, for every Martian valley
network, computes all *first-meet* channel-head pairs at confluences using
the same algorithm as the Earth pipeline's
``channel_heads.first_meet_pairs_for_outlet`` (Kahn topological sort +
head-set propagation), but operating on the Phase-1 directed graph
(``upstream_node_id`` → ``downstream_node_id``) instead of a TopoToolbox
StreamObject.

Each emitted pair carries enough geometry to drive both downstream tracks
without rerunning topology:

  Track A — tabular features for the production XGBoost
            (orientation_diff_deg, headhead_dist_norm, apex_angle_deg,
             strahler_order_diff, proximity_profile_norm)
  Track B — 5-class CNN raster patches for ``models/cnn_outlet_final.pt``
            (BACKGROUND=0, BRANCH_A=1, BRANCH_B=2, OTHER_STREAMS=3,
             CONFLUENCE_MARKER=4)

The compatibility report and its addendum document why both tracks must
be supported and why the CNN encoding must remain 5-class (so the
existing Earth-trained CNN can be applied to Mars without retraining).

Output GeoPackage layers:

  - ``mars_pairs``: one row per (network_id, confluence, head_1, head_2).
    Geometry is a MultiLineString containing both branch polylines, for
    QGIS visualization. Fields: identifiers + ``L_1``, ``L_2``,
    ``headhead_dist_m``, plus comma-separated ``path_*_node_ids`` and
    ``path_*_segment_ids`` for reproducibility / joins.
  - ``mars_pair_paths``: two rows per pair (one per branch). Geometry is
    the chained head→confluence LineString (oriented head-to-confluence,
    with constituent segment LineStrings reversed where needed). Fields:
    ``pair_id``, ``branch`` ("A" or "B"), ``head_node_id``,
    ``confluence_node_id``, ``length_m``, ``path_node_ids``,
    ``path_segment_ids``.

Primary interface
-----------------
``notebooks/mars/02_first_meet_pairs.ipynb`` is the primary, documented way to
run and understand this step; it calls the same shared package functions
(:mod:`channel_heads.pairing` — ``build_directed_adjacency``,
``first_meet_pairs_on_dag``, ``trace_downstream_path``,
``chain_segment_geometries``). This script is the headless batch wrapper that
writes the GeoPackage for the rest of the pipeline.

Run:
    python scripts/extract_mars_first_meet_pairs.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString
from tqdm import tqdm

from channel_heads.pairing import (
    build_directed_adjacency,
    chain_segment_geometries,
    first_meet_pairs_on_dag,
    trace_downstream_path,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOPOLOGY_GPKG = (
    PROJECT_ROOT
    / "data/Mars/topology/mars_vn_topology_model_ready.gpkg"
)
OUTPUT_GPKG = (
    PROJECT_ROOT / "data/Mars/topology/mars_vn_pairs.gpkg"
)
VALIDATION_PLOT_PATH = (
    PROJECT_ROOT / "data/Mars/topology/mars_pairs_validation_plot.png"
)
VALIDATION_PLOT_N = 6
VALIDATION_PLOT_SEED = 0

COORD_TOL_M = 1e-3  # used to decide LineString orientation against node coord

log = logging.getLogger("mars_pairs")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def load_topology_layers(
    gpkg_path: Path,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load mars_networks, mars_segments, mars_nodes from the Phase-1 GPKG."""
    if not gpkg_path.exists():
        raise FileNotFoundError(f"Topology GeoPackage not found: {gpkg_path}")
    networks = gpd.read_file(gpkg_path, layer="mars_networks")
    segments = gpd.read_file(gpkg_path, layer="mars_segments")
    nodes = gpd.read_file(gpkg_path, layer="mars_nodes")
    log.info(
        "Loaded topology: %d networks, %d segments, %d nodes",
        len(networks),
        len(segments),
        len(nodes),
    )
    return networks, segments, nodes


def compute_first_meet_pairs(
    parents: list[list[int]],
    children: list[list[int]],
    heads_set: set[int],
    confluences_set: set[int],
) -> dict[int, set[tuple[int, int]]]:
    """Propagate head-sets in topological order and emit cross-branch pairs.

    Delegates to the shared graph-agnostic core
    :func:`channel_heads.pairing.dag.first_meet_pairs_on_dag`. The whole
    network is in scope (all node ids ``0..n-1``). ``children`` is accepted
    for call-site compatibility; the core derives adjacency from ``parents``.
    """
    n = len(parents)
    pairs, _ = first_meet_pairs_on_dag(
        parents, set(range(n)), heads_set, confluences_set
    )
    return pairs


def process_network(
    nid: int,
    segs_in_net: gpd.GeoDataFrame,
    nodes_in_net: gpd.GeoDataFrame,
) -> tuple[list[dict], list[dict]] | None:
    """Compute pairs and chained paths for a single Mars network.

    Returns ``(pair_rows, path_rows)`` for assembly into the two output
    layers, or ``None`` if the network has no usable pairs.
    """
    if segs_in_net.empty:
        return None

    # node_id values from Phase 1 are local to the network (0..n-1).
    n_nodes = int(nodes_in_net["node_id"].max()) + 1

    parents, children, edge_to_seg = build_directed_adjacency(
        segs_in_net, n_nodes
    )

    nt_by_id = dict(zip(nodes_in_net["node_id"], nodes_in_net["node_type"]))
    heads_set = {
        int(nid_) for nid_, nt in nt_by_id.items() if nt == "channel_head"
    }
    confluences_set = {
        int(nid_) for nid_, nt in nt_by_id.items() if nt == "confluence"
    }

    if len(heads_set) < 2:
        return None  # nothing to pair

    pair_dict = compute_first_meet_pairs(
        parents, children, heads_set, confluences_set
    )
    if not pair_dict:
        return None

    # Lookup tables for path chaining.
    segment_geom_by_id: dict[int, LineString] = dict(
        zip(segs_in_net["segment_id"], segs_in_net.geometry)
    )
    start_node_by_seg: dict[int, int] = dict(
        zip(segs_in_net["segment_id"], segs_in_net["start_node_id"])
    )
    end_node_by_seg: dict[int, int] = dict(
        zip(segs_in_net["segment_id"], segs_in_net["end_node_id"])
    )

    # Node coordinates for headhead_dist_m.
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
            # h1 < h2 by construction. Path A is from h1, path B from h2.
            path_a = trace_downstream_path(
                int(h1), int(conf_id), children, max_steps
            )
            path_b = trace_downstream_path(
                int(h2), int(conf_id), children, max_steps
            )
            if path_a is None or path_b is None:
                continue
            geom_a, segs_a = chain_segment_geometries(
                path_a,
                edge_to_seg,
                segment_geom_by_id,
                start_node_by_seg,
                end_node_by_seg,
            )
            geom_b, segs_b = chain_segment_geometries(
                path_b,
                edge_to_seg,
                segment_geom_by_id,
                start_node_by_seg,
                end_node_by_seg,
            )
            if geom_a is None or geom_b is None:
                continue
            L_1 = float(geom_a.length)
            L_2 = float(geom_b.length)
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
                    "path_a_segment_ids": ",".join(
                        str(s) for s in segs_a
                    ),
                    "path_b_segment_ids": ",".join(
                        str(s) for s in segs_b
                    ),
                    "geometry": MultiLineString([geom_a, geom_b]),
                }
            )
            path_rows.append(
                {
                    "pair_id": pair_id,
                    "network_id": int(nid),
                    "branch": "A",
                    "head_node_id": int(h1),
                    "confluence_node_id": int(conf_id),
                    "length_m": L_1,
                    "n_segments": int(len(segs_a)),
                    "path_node_ids": ",".join(str(n) for n in path_a),
                    "path_segment_ids": ",".join(str(s) for s in segs_a),
                    "geometry": geom_a,
                }
            )
            path_rows.append(
                {
                    "pair_id": pair_id,
                    "network_id": int(nid),
                    "branch": "B",
                    "head_node_id": int(h2),
                    "confluence_node_id": int(conf_id),
                    "length_m": L_2,
                    "n_segments": int(len(segs_b)),
                    "path_node_ids": ",".join(str(n) for n in path_b),
                    "path_segment_ids": ",".join(str(s) for s in segs_b),
                    "geometry": geom_b,
                }
            )
            pair_counter += 1

    if pair_counter == 0:
        return None
    return pair_rows, path_rows


def write_output_layers(
    pair_rows: list[dict],
    path_rows: list[dict],
    crs,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    pairs_gdf = gpd.GeoDataFrame(pair_rows, crs=crs)
    paths_gdf = gpd.GeoDataFrame(path_rows, crs=crs)
    pairs_gdf.to_file(output_path, layer="mars_pairs", driver="GPKG")
    paths_gdf.to_file(output_path, layer="mars_pair_paths", driver="GPKG")
    log.info(
        "Wrote layers: mars_pairs (%d rows), mars_pair_paths (%d rows) -> %s",
        len(pairs_gdf),
        len(paths_gdf),
        output_path,
    )


def validation_plot(
    networks: gpd.GeoDataFrame,
    segments: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    pair_paths: gpd.GeoDataFrame,
    output_path: Path,
    n: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    candidate_nids = pair_paths["network_id"].unique()
    if len(candidate_nids) == 0:
        log.warning("No pairs available for validation plot")
        return
    sample_n = min(n, len(candidate_nids))
    sample_nids = rng.choice(candidate_nids, size=sample_n, replace=False)

    cols = min(3, sample_n)
    rows = int(np.ceil(sample_n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, nid in zip(axes, sample_nids):
        nid = int(nid)
        segs = segments[segments["network_id"] == nid]
        nds = nodes[nodes["network_id"] == nid]
        paths = pair_paths[pair_paths["network_id"] == nid]

        segs.plot(ax=ax, color="#bbbbbb", linewidth=0.6)
        a_paths = paths[paths["branch"] == "A"]
        b_paths = paths[paths["branch"] == "B"]
        if not a_paths.empty:
            a_paths.plot(ax=ax, color="#d95f02", linewidth=1.4, label="branch A")
        if not b_paths.empty:
            b_paths.plot(ax=ax, color="#1f78b4", linewidth=1.4, label="branch B")

        heads = nds[nds["node_type"] == "channel_head"]
        confs = nds[nds["node_type"] == "confluence"]
        outlet = nds[nds["node_type"] == "outlet"]
        if not heads.empty:
            heads.plot(ax=ax, color="black", markersize=20, marker="o")
        if not confs.empty:
            confs.plot(ax=ax, color="tab:orange", markersize=30, marker="s")
        if not outlet.empty:
            outlet.plot(ax=ax, color="red", markersize=80, marker="*")

        ax.set_title(
            f"network_id={nid} ({len(paths) // 2} pairs)", fontsize=9
        )
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        if not a_paths.empty or not b_paths.empty:
            ax.legend(loc="best", fontsize=6)

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
    networks, segments, nodes = load_topology_layers(TOPOLOGY_GPKG)

    all_pair_rows: list[dict] = []
    all_path_rows: list[dict] = []

    grouped_segments = dict(tuple(segments.groupby("network_id", sort=True)))
    grouped_nodes = dict(tuple(nodes.groupby("network_id", sort=True)))

    for nid in tqdm(
        sorted(grouped_segments.keys()), desc="Networks"
    ):
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

    log.info("Total pairs:      %d", len(all_pair_rows))
    log.info("Total pair paths: %d", len(all_path_rows))

    # Stats
    if all_pair_rows:
        df = pd.DataFrame(all_pair_rows)
        log.info(
            "Pairs per network: mean=%.2f median=%.0f max=%d",
            df.groupby("network_id").size().mean(),
            df.groupby("network_id").size().median(),
            df.groupby("network_id").size().max(),
        )
        log.info(
            "L_1+L_2 (m): mean=%.0f median=%.0f max=%.0f",
            df["L_sum"].mean(),
            df["L_sum"].median(),
            df["L_sum"].max(),
        )
        log.info(
            "headhead_dist_m: mean=%.0f median=%.0f max=%.0f",
            df["headhead_dist_m"].mean(),
            df["headhead_dist_m"].median(),
            df["headhead_dist_m"].max(),
        )

    write_output_layers(all_pair_rows, all_path_rows, segments.crs, OUTPUT_GPKG)

    pair_paths_gdf = gpd.read_file(OUTPUT_GPKG, layer="mars_pair_paths")
    validation_plot(
        networks,
        segments,
        nodes,
        pair_paths_gdf,
        VALIDATION_PLOT_PATH,
        n=VALIDATION_PLOT_N,
        seed=VALIDATION_PLOT_SEED,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
