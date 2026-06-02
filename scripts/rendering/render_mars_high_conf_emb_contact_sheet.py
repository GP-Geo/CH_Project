#!/usr/bin/env python
"""Render a vector-style contact sheet of 20 Mars pairs with the
highest geom_plus_cnn_emb probabilities (high-confidence emb touching).

Companion to ``render_mars_combined_contact_sheets_vector.py`` — uses
the same prediction parquet and same panel layout, but the selection
criterion is "high confidence on the emb model only" rather than
"both combined models agree".

Output:
  data/Mars/model_outputs/figures_combined/contact_sheet_high_conf_emb.png

Primary interface: ``notebooks/presentation/mars_contact_sheets.ipynb`` renders
these inline via ``channel_heads.viz.render_pair_panel`` /
``render_contact_sheet``; this script is the headless wrapper that writes the PNG.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from channel_heads.viz import render_pair_panel

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
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data/Mars/model_outputs/figures_combined/contact_sheet_high_conf_emb.png"
)

HIGH_CONF_PROB_MIN = 0.80
N_SAMPLES = 20

log = logging.getLogger("phase6c_high_conf_emb")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    setup_logging()

    log.info("Loading combined predictions: %s", PRED_PARQUET)
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

    # Top-20 by prob_touching_emb, restricted to prob >= HIGH_CONF_PROB_MIN
    hc_emb = df[df["prob_touching_emb"] >= HIGH_CONF_PROB_MIN].copy()
    n_pool = len(hc_emb)
    hc_emb = hc_emb.sort_values("prob_touching_emb", ascending=False).head(N_SAMPLES)
    log.info(
        "geom_plus_cnn_emb high-confidence pool: %d (prob >= %.2f). "
        "Rendering top %d.",
        n_pool,
        HIGH_CONF_PROB_MIN,
        len(hc_emb),
    )

    cols = 5
    rows = int(np.ceil(len(hc_emb) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (_, r) in zip(axes_flat, hc_emb.iterrows()):
        nid = int(r["network_id"])
        pair_id = str(r["pair_id"])
        h1 = int(r["head_node_id_1"])
        h2 = int(r["head_node_id_2"])
        conf = int(r["confluence_node_id"])

        nodes_n = nodes_by_nid[nid]
        segs_n = segs_by_nid[nid]
        node_xy = {
            int(nr["node_id"]): (float(nr.geometry.x), float(nr.geometry.y))
            for _, nr in nodes_n.iterrows()
        }
        path_a = paths_lookup.get((pair_id, "A"))
        path_b = paths_lookup.get((pair_id, "B"))
        if path_a is None or path_b is None:
            ax.set_visible(False)
            continue

        render_pair_panel(
            ax,
            segs_n=segs_n,
            path_a=path_a,
            path_b=path_b,
            h1_xy=node_xy[h1],
            h2_xy=node_xy[h2],
            conf_xy=node_xy[conf],
            outlet_xy=outlets_by_nid.get(nid),
            title=(
                f"net={nid}  pair={pair_id}\n"
                f"emb={r['prob_touching_emb']:.3f}  "
                f"(logit={r['prob_touching_logit']:.3f})"
            ),
        )

    for ax in axes_flat[len(hc_emb):]:
        ax.set_visible(False)
    fig.suptitle(
        f"20 highest-probability touching pairs — geom_plus_cnn_emb only "
        f"(prob ≥ {HIGH_CONF_PROB_MIN:.2f})",
        fontsize=12,
        y=1.0,
    )
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote: %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
