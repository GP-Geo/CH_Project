#!/usr/bin/env python
"""Visual QA for Mars pairs removed by the stream-crossing filter.

Phase 3A's stream-crossing filter dropped 11,892 of 15,677 Mars first-meet
pairs (75.9%). Before running XGBoost inference, we render 30 of those
removed pairs so a human can confirm the filter is behaving reasonably.

For each sampled removed pair this script renders one PNG showing:
  - the full local valley-network in light grey
  - branch A (head_1 → confluence) in orange
  - branch B (head_2 → confluence) in blue
  - head_1 / head_2 / confluence / outlet markers
  - the straight head-to-head line used by the filter (dashed red)
  - the segment(s) the straight line actually crosses, highlighted red

It also writes a contact sheet (30 panels) and a GeoPackage containing
the sampled pair geometries for QGIS inspection.

Sampling strategy: stratified by per-network pair count to span small,
medium, large, and huge (dense) networks; one removed pair per network
where possible; fixed random seed.

Run:
    python scripts/qa_mars_stream_crossing_filter.py

Primary interface: ``notebooks/diagnostics/stream_crossing_qa.ipynb`` renders a
few removed pairs inline via ``channel_heads.pairing.detect_crossed_segments`` +
``channel_heads.viz.plot_removed_pair``; this script is the headless wrapper that
writes the full QA contact sheet + GeoPackage.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from tqdm import tqdm

from channel_heads.pairing import detect_crossed_segments
from channel_heads.pairing.filtering import stratified_sample_removed_pairs
from channel_heads.viz import plot_removed_pair, removed_pair_legend_handles

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # scripts/diagnostics/ -> repo root

FEATURES_PARQUET = (
    PROJECT_ROOT
    / "data/Mars/model_inputs/mars_pair_features_5feat_all.parquet"
)
TOPOLOGY_GPKG = (
    PROJECT_ROOT
    / "data/Mars/topology/mars_vn_topology_model_ready.gpkg"
)
PAIRS_GPKG = PROJECT_ROOT / "data/Mars/topology/mars_vn_pairs.gpkg"

OUTPUT_DIR = (
    PROJECT_ROOT / "data/Mars/model_inputs/filtering_qa_removed_pairs"
)

N_SAMPLES = 30
RANDOM_SEED = 42

# Must match Phase 3A constants
STREAM_FILTER_HEAD_BUFFER_M = 5.0
EPSILON = 1e-10

# Bucket allocation across pair-count buckets (must sum to N_SAMPLES)
SIZE_BUCKETS: list[tuple[str, int, int, int]] = [
    # (label, n_pairs_min, n_pairs_max_inclusive, n_samples_target)
    ("small", 3, 10, 8),
    ("medium", 11, 30, 8),
    ("large", 31, 100, 8),
    ("huge", 101, 10_000, 6),
]

log = logging.getLogger("mars_qa_filter")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# stratified_sample_removed_pairs lives in channel_heads.pairing.filtering.


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading features: %s", FEATURES_PARQUET)
    df_all = pd.read_parquet(FEATURES_PARQUET)
    log.info("Loading topology + pairs")
    segments = gpd.read_file(TOPOLOGY_GPKG, layer="mars_segments")
    nodes = gpd.read_file(TOPOLOGY_GPKG, layer="mars_nodes")
    outlets = gpd.read_file(TOPOLOGY_GPKG, layer="mars_outlets")
    pair_paths = gpd.read_file(PAIRS_GPKG, layer="mars_pair_paths")
    pairs_gdf = gpd.read_file(PAIRS_GPKG, layer="mars_pairs")
    crs = segments.crs

    # ----------------------------------------------------------------
    # Sample
    # ----------------------------------------------------------------
    sample_df = stratified_sample_removed_pairs(
        df_all, N_SAMPLES, SIZE_BUCKETS, RANDOM_SEED
    )
    if len(sample_df) < N_SAMPLES:
        log.warning(
            "Only collected %d samples (target %d)", len(sample_df), N_SAMPLES
        )

    log.info(
        "Sampled %d pairs across size buckets: %s",
        len(sample_df),
        sample_df["size_bucket"].value_counts().to_dict(),
    )

    # ----------------------------------------------------------------
    # Lookups
    # ----------------------------------------------------------------
    nodes_by_nid = dict(tuple(nodes.groupby("network_id")))
    segs_by_nid = dict(tuple(segments.groupby("network_id")))
    paths_by_pair_branch: dict[tuple[str, str], LineString] = {}
    for _, r in pair_paths.iterrows():
        paths_by_pair_branch[(str(r["pair_id"]), str(r["branch"]))] = r.geometry
    outlets_by_nid: dict[int, Point] = {
        int(r["network_id"]): r.geometry for _, r in outlets.iterrows()
    }

    # ----------------------------------------------------------------
    # Build per-sample geometry + crossed segments
    # ----------------------------------------------------------------
    rendered_rows: list[dict] = []
    path_rows: list[dict] = []
    head_rows: list[dict] = []
    conf_rows: list[dict] = []
    headhead_rows: list[dict] = []
    crossed_rows: list[dict] = []
    crossed_count: list[int] = []

    panel_specs: list[dict] = []

    for _, srow in sample_df.iterrows():
        nid = int(srow["network_id"])
        pair_id = str(srow["pair_id"])
        h1 = int(srow["head_node_id_1"])
        h2 = int(srow["head_node_id_2"])
        conf = int(srow["confluence_node_id"])

        nodes_n = nodes_by_nid[nid]
        segs_n = segs_by_nid[nid]

        node_xy = {
            int(r["node_id"]): (float(r.geometry.x), float(r.geometry.y))
            for _, r in nodes_n.iterrows()
        }
        h1_xy = node_xy[h1]
        h2_xy = node_xy[h2]
        conf_xy = node_xy[conf]
        outlet_pt = outlets_by_nid.get(nid)
        outlet_xy = (
            (float(outlet_pt.x), float(outlet_pt.y))
            if outlet_pt is not None
            else None
        )

        path_a = paths_by_pair_branch.get((pair_id, "A"))
        path_b = paths_by_pair_branch.get((pair_id, "B"))
        if path_a is None or path_b is None:
            log.warning("Missing path for %s", pair_id)
            continue

        crossed_segs = detect_crossed_segments(h1_xy, h2_xy, segs_n)
        crossed_count.append(len(crossed_segs))

        rendered_rows.append(
            {
                "sample_index": int(srow["sample_index"]),
                "pair_id": pair_id,
                "network_id": nid,
                "head_node_id_1": h1,
                "head_node_id_2": h2,
                "confluence_node_id": conf,
                "outlet_node_id": (
                    int(outlets_by_nid[nid].coords[0][0])
                    if False
                    else int(
                        outlets[outlets["network_id"] == nid][
                            "outlet_node_id"
                        ].iloc[0]
                    )
                ),
                "size_bucket": srow["size_bucket"],
                "network_n_pairs": int(srow["network_n_pairs"]),
                "network_n_dropped": int(srow["network_n_dropped"]),
                "network_drop_rate": float(srow["network_drop_rate"]),
                "headhead_dist_m": float(srow["headhead_dist_m"]),
                "path_length_1_m": float(srow["path_length_1_m"]),
                "path_length_2_m": float(srow["path_length_2_m"]),
                "n_crossed_segments": int(len(crossed_segs)),
                "crossed_segment_ids": ",".join(str(s) for s in crossed_segs),
                "filter_status": str(srow["filter_status"]),
                "filter_reason": str(srow["filter_reason"]),
                "geometry": MultiLineString([path_a, path_b]),
            }
        )
        path_rows.append(
            {
                "sample_index": int(srow["sample_index"]),
                "pair_id": pair_id,
                "network_id": nid,
                "branch": "A",
                "geometry": path_a,
            }
        )
        path_rows.append(
            {
                "sample_index": int(srow["sample_index"]),
                "pair_id": pair_id,
                "network_id": nid,
                "branch": "B",
                "geometry": path_b,
            }
        )
        head_rows.append(
            {
                "sample_index": int(srow["sample_index"]),
                "pair_id": pair_id,
                "network_id": nid,
                "branch": "A",
                "node_id": h1,
                "geometry": Point(*h1_xy),
            }
        )
        head_rows.append(
            {
                "sample_index": int(srow["sample_index"]),
                "pair_id": pair_id,
                "network_id": nid,
                "branch": "B",
                "node_id": h2,
                "geometry": Point(*h2_xy),
            }
        )
        conf_rows.append(
            {
                "sample_index": int(srow["sample_index"]),
                "pair_id": pair_id,
                "network_id": nid,
                "node_id": conf,
                "geometry": Point(*conf_xy),
            }
        )
        headhead_rows.append(
            {
                "sample_index": int(srow["sample_index"]),
                "pair_id": pair_id,
                "network_id": nid,
                "headhead_dist_m": float(srow["headhead_dist_m"]),
                "n_crossed_segments": int(len(crossed_segs)),
                "geometry": LineString([h1_xy, h2_xy]),
            }
        )
        for seg_id in crossed_segs:
            seg_row = segs_n[segs_n["segment_id"] == seg_id].iloc[0]
            crossed_rows.append(
                {
                    "sample_index": int(srow["sample_index"]),
                    "pair_id": pair_id,
                    "network_id": nid,
                    "segment_id": int(seg_id),
                    "length_m": float(seg_row["length_m"]),
                    "geometry": seg_row.geometry,
                }
            )

        panel_specs.append(
            {
                "sample_index": int(srow["sample_index"]),
                "pair_id": pair_id,
                "network_id": nid,
                "segs_n": segs_n,
                "path_a": path_a,
                "path_b": path_b,
                "h1_xy": h1_xy,
                "h2_xy": h2_xy,
                "conf_xy": conf_xy,
                "outlet_xy": outlet_xy,
                "crossed_segs": crossed_segs,
                "size_bucket": srow["size_bucket"],
                "headhead_dist_m": float(srow["headhead_dist_m"]),
                "path_length_1_m": float(srow["path_length_1_m"]),
                "path_length_2_m": float(srow["path_length_2_m"]),
                "filter_reason": str(srow["filter_reason"]),
            }
        )

    n = len(panel_specs)
    log.info(
        "Crossed segments per pair: min=%d median=%.0f max=%d (mean=%.1f)",
        min(crossed_count),
        float(np.median(crossed_count)),
        max(crossed_count),
        float(np.mean(crossed_count)),
    )

    # ----------------------------------------------------------------
    # CSV summary
    # ----------------------------------------------------------------
    csv_df = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "geometry"} for r in rendered_rows]
    )
    csv_path = OUTPUT_DIR / "removed_pair_samples_30.csv"
    csv_df.to_csv(csv_path, index=False)
    log.info("Wrote CSV: %s", csv_path)

    # ----------------------------------------------------------------
    # Per-pair PNGs
    # ----------------------------------------------------------------
    for spec in tqdm(panel_specs, desc="Render PNGs"):
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        title = (
            f"sample #{spec['sample_index']:02d}  "
            f"network_id={spec['network_id']}  pair_id={spec['pair_id']}\n"
            f"bucket={spec['size_bucket']}  "
            f"crossed_segs={len(spec['crossed_segs'])}  "
            f"headhead={spec['headhead_dist_m']:.0f} m  "
            f"L1={spec['path_length_1_m']:.0f} m  "
            f"L2={spec['path_length_2_m']:.0f} m\n"
            f"filter_reason={spec['filter_reason']}"
        )
        plot_removed_pair(
            ax,
            sample_row=None,  # title built externally
            network_segments=spec["segs_n"],
            path_a_geom=spec["path_a"],
            path_b_geom=spec["path_b"],
            h1_xy=spec["h1_xy"],
            h2_xy=spec["h2_xy"],
            conf_xy=spec["conf_xy"],
            outlet_xy=spec["outlet_xy"],
            crossed_segment_ids=spec["crossed_segs"],
            title=title,
            title_fontsize=8,
        )
        ax.legend(
            handles=removed_pair_legend_handles(),
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=7,
            frameon=False,
        )
        fig.tight_layout()
        path = (
            OUTPUT_DIR
            / f"removed_pair_sample_{spec['sample_index']:03d}.png"
        )
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)

    # ----------------------------------------------------------------
    # Contact sheet
    # ----------------------------------------------------------------
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, spec in zip(axes_flat, panel_specs):
        title = (
            f"#{spec['sample_index']:02d} net={spec['network_id']} "
            f"crossed={len(spec['crossed_segs'])} "
            f"hh={spec['headhead_dist_m']:.0f}m"
        )
        plot_removed_pair(
            ax,
            sample_row=None,
            network_segments=spec["segs_n"],
            path_a_geom=spec["path_a"],
            path_b_geom=spec["path_b"],
            h1_xy=spec["h1_xy"],
            h2_xy=spec["h2_xy"],
            conf_xy=spec["conf_xy"],
            outlet_xy=spec["outlet_xy"],
            crossed_segment_ids=spec["crossed_segs"],
            title=title,
            title_fontsize=7,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(
        "Mars stream-crossing filter — 30 removed-pair samples",
        fontsize=14,
        y=1.0,
    )
    fig.tight_layout()
    contact_path = OUTPUT_DIR / "removed_pair_samples_contact_sheet.png"
    fig.savefig(contact_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote contact sheet: %s", contact_path)

    # ----------------------------------------------------------------
    # GeoPackage
    # ----------------------------------------------------------------
    gpkg_path = OUTPUT_DIR / "removed_pair_samples_30.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    gpd.GeoDataFrame(rendered_rows, crs=crs).to_file(
        gpkg_path, layer="sampled_removed_pairs", driver="GPKG"
    )
    gpd.GeoDataFrame(path_rows, crs=crs).to_file(
        gpkg_path, layer="sampled_removed_pair_paths", driver="GPKG"
    )
    gpd.GeoDataFrame(head_rows, crs=crs).to_file(
        gpkg_path, layer="sampled_removed_pair_heads", driver="GPKG"
    )
    gpd.GeoDataFrame(conf_rows, crs=crs).to_file(
        gpkg_path, layer="sampled_removed_pair_confluences", driver="GPKG"
    )
    gpd.GeoDataFrame(headhead_rows, crs=crs).to_file(
        gpkg_path, layer="sampled_removed_headhead_lines", driver="GPKG"
    )
    if crossed_rows:
        gpd.GeoDataFrame(crossed_rows, crs=crs).to_file(
            gpkg_path,
            layer="sampled_removed_crossed_segments",
            driver="GPKG",
        )
    log.info("Wrote GPKG: %s", gpkg_path)

    log.info("Done. Outputs in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
