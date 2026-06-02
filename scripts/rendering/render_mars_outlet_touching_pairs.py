#!/usr/bin/env python
"""Render per-outlet visualisations of all touching pairs predicted for
the Mars network. One PNG per network, top 10 networks chosen by the
number of pairs the geom_plus_cnn_emb model predicted as touching
(i.e. "relatively complex" outlets).

For each selected network the figure shows:
  - the full Mars valley-network polylines in light grey,
  - every touching pair's two head-to-confluence branch polylines
    drawn in a distinct colour (one colour per pair, both branches
    sharing that colour),
  - small dots for channel heads, orange squares for confluences,
    and a red star for the outlet.

Output:
  data/Mars/model_outputs/figures_combined/per_outlet/
      mars_outlet_touching_pairs_net{network_id}.png   (×10)

Primary interface: ``notebooks/presentation/per_outlet_touching_pairs.ipynb``
renders these inline via ``channel_heads.viz.render_outlet_touching_pairs``;
this script is the headless wrapper that writes one PNG per network.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from channel_heads.viz import render_outlet_touching_pairs

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/rendering/ -> repo root

PRED_PARQUET = (
    PROJECT_ROOT
    / "data/Mars/model_outputs/mars_combined_model_predictions.parquet"
)
PAIRS_GPKG = PROJECT_ROOT / "data/Mars/topology/mars_vn_pairs.gpkg"
TOPOLOGY_GPKG = (
    PROJECT_ROOT
    / "data/Mars/topology/mars_vn_topology_model_ready.gpkg"
)
OUTPUT_DIR = (
    PROJECT_ROOT / "data/Mars/model_outputs/figures_combined/per_outlet"
)

N_NETWORKS = 10
TOUCHING_COL = "pred_touching_emb"      # which model defines "touching"
PROB_COL = "prob_touching_emb"
MODEL_NAME = "geom_plus_cnn_emb"

log = logging.getLogger("phase6c_per_outlet")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading predictions: %s", PRED_PARQUET)
    df = pd.read_parquet(PRED_PARQUET)

    log.info("Loading topology + pair paths")
    nodes_gdf = gpd.read_file(TOPOLOGY_GPKG, layer="mars_nodes")
    segments_gdf = gpd.read_file(TOPOLOGY_GPKG, layer="mars_segments")
    outlets_gdf = gpd.read_file(TOPOLOGY_GPKG, layer="mars_outlets")
    pair_paths = gpd.read_file(PAIRS_GPKG, layer="mars_pair_paths")

    nodes_by_nid = dict(tuple(nodes_gdf.groupby("network_id")))
    segs_by_nid = dict(tuple(segments_gdf.groupby("network_id")))
    outlets_by_nid: dict[int, tuple[float, float]] = {
        int(r["network_id"]): (float(r.geometry.x), float(r.geometry.y))
        for _, r in outlets_gdf.iterrows()
    }
    paths_lookup: dict[tuple[str, str], LineString] = {}
    for _, r in pair_paths.iterrows():
        paths_lookup[(str(r["pair_id"]), str(r["branch"]))] = r.geometry

    # Pick top N networks by number of touching pairs (only consider
    # networks that actually have touching pairs).
    counts = (
        df.groupby("network_id")
        .agg(
            n_pairs=("pair_id", "count"),
            n_touching=(TOUCHING_COL, "sum"),
        )
        .sort_values("n_touching", ascending=False)
    )
    counts = counts[counts["n_touching"] > 0]
    top = counts.head(N_NETWORKS)
    log.info(
        "Selected %d networks (by n_touching desc):\n%s",
        len(top),
        top.to_string(),
    )

    for nid in top.index:
        nid_int = int(nid)
        df_net = df[df["network_id"] == nid_int]
        nodes_n = nodes_by_nid[nid_int]
        segs_n = segs_by_nid[nid_int]
        outlet_xy = outlets_by_nid.get(nid_int)
        output = OUTPUT_DIR / f"mars_outlet_touching_pairs_net{nid_int:03d}.png"
        render_outlet_touching_pairs(
            nid_int,
            df_net,
            segs_n,
            nodes_n,
            outlet_xy,
            paths_lookup,
            output,
        )

    log.info("Done. %d PNGs in %s", len(top), OUTPUT_DIR)


if __name__ == "__main__":
    main()
