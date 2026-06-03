#!/usr/bin/env python
"""Re-render the Phase 6C combined-inference contact sheets in vector
network style (matching the QA filter contact sheets), replacing the
raster-patch versions in ``data/Mars/model_outputs/figures_combined/``.

Picks the same 20 high-confidence-both pairs and 20 strongest
disagreement pairs as the Mars combined-inference package stage. Reads
Phase 1 topology + Phase 2B pair paths for geometry. Does NOT re-run
inference and does not touch any parquet/CSV.

Outputs (overwritten):
  data/Mars/model_outputs/figures_combined/contact_sheet_high_conf_both.png
  data/Mars/model_outputs/figures_combined/contact_sheet_disagreement.png

Primary interface: ``notebooks/presentation/mars_contact_sheets.ipynb`` renders
the vector contact sheets inline via ``channel_heads.viz.render_contact_sheet``;
this script is the headless wrapper that writes the PNG files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from channel_heads.viz import render_contact_sheet

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
FIGURES_DIR = PROJECT_ROOT / "data/Mars/model_outputs/figures_combined"

HIGH_CONF_PROB_MIN = 0.80

log = logging.getLogger("phase6c_vector")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# render_pair_panel + render_contact_sheet live in channel_heads.viz
# (shared with notebooks/presentation/).


def main() -> None:
    setup_logging()

    log.info("Loading combined predictions: %s", PRED_PARQUET)
    df = pd.read_parquet(PRED_PARQUET)

    log.info("Loading Phase 1 topology + Phase 2B paths")
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

    # --- Selection 1: 20 high-confidence "both combined models touching" ---
    hc_both = df[
        (df["prob_touching_emb"] >= HIGH_CONF_PROB_MIN)
        & (df["prob_touching_logit"] >= HIGH_CONF_PROB_MIN)
    ].copy()
    hc_both["min_combined_prob"] = hc_both[
        ["prob_touching_emb", "prob_touching_logit"]
    ].min(axis=1)
    hc_both = hc_both.sort_values("min_combined_prob", ascending=False).head(20)
    log.info(
        "High-confidence both: %d candidates, rendering top %d",
        int(((df["prob_touching_emb"] >= HIGH_CONF_PROB_MIN)
             & (df["prob_touching_logit"] >= HIGH_CONF_PROB_MIN)).sum()),
        len(hc_both),
    )

    render_contact_sheet(
        hc_both,
        nodes_by_nid,
        segs_by_nid,
        outlets_by_nid,
        paths_lookup,
        title="20 pairs — both combined models high-confidence touching (prob ≥ 0.80)",
        subtitle_fn=lambda r: (
            f"net={int(r['network_id'])}  pair={r['pair_id']}\n"
            f"emb={r['prob_touching_emb']:.3f}  "
            f"logit={r['prob_touching_logit']:.3f}"
        ),
        output=FIGURES_DIR / "contact_sheet_high_conf_both.png",
    )

    # --- Selection 2: 20 strongest emb-vs-logit disagreements ---
    disag = df[df["emb_logit_disagreement"] == 1].copy()
    if not disag.empty:
        disag["abs_dprob"] = (
            disag["prob_touching_emb"] - disag["prob_touching_logit"]
        ).abs()
        disag = disag.sort_values("abs_dprob", ascending=False).head(20)
        log.info(
            "Disagreement total: %d, rendering top %d by |Δprob|",
            int((df["emb_logit_disagreement"] == 1).sum()),
            len(disag),
        )
        render_contact_sheet(
            disag,
            nodes_by_nid,
            segs_by_nid,
            outlets_by_nid,
            paths_lookup,
            title=(
                "20 pairs — combined-emb vs combined-logit disagreement "
                "(sorted by |Δprob|)"
            ),
            subtitle_fn=lambda r: (
                f"net={int(r['network_id'])}  pair={r['pair_id']}\n"
                f"emb={r['prob_touching_emb']:.3f} "
                f"(pred={int(r['pred_touching_emb'])})  |  "
                f"logit={r['prob_touching_logit']:.3f} "
                f"(pred={int(r['pred_touching_logit'])})"
            ),
            output=FIGURES_DIR / "contact_sheet_disagreement.png",
        )
    else:
        log.warning("No disagreement rows — disagreement sheet not produced")

    log.info("Done.")


if __name__ == "__main__":
    main()
