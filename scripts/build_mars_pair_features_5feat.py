#!/usr/bin/env python
"""Phase 3A — Compute the Mars 5-feature table and apply terrestrial
pre-inference filtering logic.

Mirrors the Earth feature definitions used by the production XGBoost
classifier (5 dimensionless / angular features):
  - orientation_diff_deg
  - headhead_dist_norm
  - apex_angle_deg
  - strahler_order_diff
  - proximity_profile_norm

Earth source files / functions copied or adapted:
  - channel_heads/geometric_analysis.py
      * EPSILON
      * DEFAULT_DIRECTION_SAMPLE_DISTANCE_M    (= 500.0)
      * MIN_EDGES_FOR_DIRECTION                (= 3)
      * _angle_between_vectors                  -> apex_angle_deg
      * _compute_azimuth, _azimuth_difference   -> orientation_diff_deg
      * _compute_direction_vector               -> orientation_diff_deg
      * _sample_path_coords                     -> proximity_profile_norm
      * _compute_proximity_profile              -> proximity_profile_norm
      * GeometricFeaturesAnalyzer.compute_pair_geometry (overall recipe)
  - channel_heads/coupling_analysis.py
      * CouplingAnalyzer._crosses_stream         -> stream-crossing filter
        (Bresenham-based on Earth; here re-implemented in projected metres
         using shapely for the Mars vector network)
  - notebooks/training/00_full_pipeline.ipynb
      * stream filter is always-on, applied during pair evaluation
  - notebooks/training/02_train_classifier.ipynb (cell 19, 20)
      * confirms the 5 active features and that XGBoost handles NaN natively
        (so we do NOT drop NaN feature rows here)

Filters NOT applied to Mars (logged with reason in the audit CSV):
  - filter_hard_negatives (geometric_analysis.py): training-time only;
    requires positives to compute thresholds. Inapplicable to Mars inference.
  - stratified_subsample_negatives (00_full_pipeline.ipynb): training-time
    class balance. Inapplicable to Mars inference.
  - prefilter_distance (coupling_analysis.py): Earth-only metric pixel
    threshold (2*sqrt(stream_threshold) px). Does not remove rows, only
    flags them, and is bound to TopoToolbox raster threshold.

Outputs:
  data/Mars/model_inputs/mars_pair_features_5feat_all.{parquet,csv}
  data/Mars/model_inputs/mars_pair_features_5feat_model_ready.{parquet,csv}
  data/Mars/model_inputs/mars_pair_filtering_audit.csv
  data/Mars/model_inputs/mars_features_validation.gpkg

Phase 3A only. Does NOT run the XGBoost model and does NOT generate CNN
patches.

Primary interface
-----------------
``notebooks/mars/03_pair_features.ipynb`` is the primary, documented way to
understand this step; it calls the same shared package functions
(:mod:`channel_heads.features` — ``line_direction_first_n_meters``,
``sample_path_coords_along_line``, ``angle_between_vectors``,
``compute_azimuth``, ``azimuth_difference``, ``compute_proximity_profile``).
This script is the headless batch wrapper that writes the feature tables for
the rest of the pipeline.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
from tqdm import tqdm

from channel_heads.features.geometry import (
    angle_between_vectors,
    azimuth_difference,
    compute_azimuth,
    compute_proximity_profile,
)
from channel_heads.features.paths import (
    DIRECTION_SAMPLE_DISTANCE_M,
    EPSILON,
    N_PROXIMITY_SAMPLES,
    line_direction_first_n_meters,
    sample_path_coords_along_line,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TOPOLOGY_GPKG = (
    PROJECT_ROOT
    / "data/Mars/topology/mars_vn_topology_model_ready.gpkg"
)
PAIRS_GPKG = PROJECT_ROOT / "data/Mars/topology/mars_vn_pairs.gpkg"
OUTPUT_DIR = PROJECT_ROOT / "data/Mars/model_inputs"
FEATURE_COLUMNS_TXT = PROJECT_ROOT / "models/feature_columns.txt"

# EPSILON, DIRECTION_SAMPLE_DISTANCE_M, MIN_EDGES_FOR_DIRECTION and
# N_PROXIMITY_SAMPLES are imported from channel_heads.features.paths (single
# source of truth, shared with notebooks/mars/).

# Stream-crossing filter parameters
# A small buffer around each head excludes the head endpoint itself from the
# intersection check — direct analogue of Earth's "exclude first/last
# Bresenham pixel" rule (coupling_analysis.py:_crosses_stream). Set to a
# fraction of the MOLA pixel size (~200 m) so the head buffer is much smaller
# than the typical channel-to-channel separation.
STREAM_FILTER_HEAD_BUFFER_M = 5.0

# Required model feature columns and order
MODEL_FEATURES: list[str] = [
    "orientation_diff_deg",
    "headhead_dist_norm",
    "apex_angle_deg",
    "strahler_order_diff",
    "proximity_profile_norm",
]

log = logging.getLogger("mars_features_5feat")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Earth-mirrored numeric helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Strahler order on the oriented Mars graph
# ---------------------------------------------------------------------------
def compute_strahler_orders(
    n_nodes: int,
    parents: list[list[int]],
    children: list[list[int]],
) -> np.ndarray:
    """Standard Strahler order on the directed Mars graph.

    - Channel heads (no parents): order = 1
    - Confluence (multiple parents): if the max parent order appears ≥ 2
      times → order = max + 1, else order = max
    - Connector (one parent): order = parent's order

    Returns
    -------
    np.ndarray of int with shape (n_nodes,). Nodes never visited (e.g.
    disconnected from any head) get order 0.
    """
    order = np.zeros(n_nodes, dtype=int)
    in_deg = [len(parents[v]) for v in range(n_nodes)]
    queue: deque[int] = deque(v for v in range(n_nodes) if in_deg[v] == 0)
    while queue:
        v = queue.popleft()
        if not parents[v]:
            order[v] = 1
        else:
            parent_orders = [order[p] for p in parents[v]]
            max_o = max(parent_orders)
            count_max = sum(1 for po in parent_orders if po == max_o)
            order[v] = max_o + 1 if count_max >= 2 else max_o
        for c in children[v]:
            in_deg[c] -= 1
            if in_deg[c] == 0:
                queue.append(c)
    return order


def build_directed_adjacency(
    segments_df: pd.DataFrame, n_nodes: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Build parent/children adjacency from ``mars_segments``.

    Self-loop segments (``upstream_node_id == downstream_node_id``) are
    skipped — they are degenerate Phase-1 segments.
    """
    parents: list[list[int]] = [[] for _ in range(n_nodes)]
    children: list[list[int]] = [[] for _ in range(n_nodes)]
    for _, row in segments_df.iterrows():
        u = int(row["upstream_node_id"])
        v = int(row["downstream_node_id"])
        if u == v or u < 0 or v < 0 or u >= n_nodes or v >= n_nodes:
            continue
        parents[v].append(u)
        children[u].append(v)
    return parents, children


# ---------------------------------------------------------------------------
# Stream-crossing filter on Mars vector geometry
# ---------------------------------------------------------------------------
def crosses_other_channel(
    head_1_xy: tuple[float, float],
    head_2_xy: tuple[float, float],
    network_segments_tree: STRtree,
    network_segments_list: list[LineString],
    head_buffer_m: float,
) -> bool:
    """Mars analogue of CouplingAnalyzer._crosses_stream.

    True if the straight line between the two heads intersects any segment
    of the same Mars network at a point *other than the head endpoints
    themselves* (we exclude a small disc of radius ``head_buffer_m`` around
    each head). False otherwise — i.e., the pair is kept.
    """
    p1 = Point(*head_1_xy)
    p2 = Point(*head_2_xy)
    straight = LineString([head_1_xy, head_2_xy])
    if straight.length < EPSILON:
        return False
    h1_buf = p1.buffer(head_buffer_m)
    h2_buf = p2.buffer(head_buffer_m)
    straight_interior = straight.difference(h1_buf).difference(h2_buf)
    if straight_interior.is_empty:
        return False
    # STRtree.query returns candidate indices intersecting the interior bbox
    cand_idx = network_segments_tree.query(straight_interior)
    for i in cand_idx:
        seg = network_segments_list[int(i)]
        if seg.intersects(straight_interior):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-network feature computation
# ---------------------------------------------------------------------------
def parse_id_list(s: str) -> list[int]:
    if not isinstance(s, str) or not s:
        return []
    return [int(tok) for tok in s.split(",") if tok != ""]


def coords_of_pair_path(
    pair_id: str, branch: str, pair_paths_by_pair_branch: dict
) -> np.ndarray | None:
    geom = pair_paths_by_pair_branch.get((pair_id, branch))
    if geom is None or geom.is_empty:
        return None
    return np.asarray(geom.coords, dtype=float)


def process_network(
    nid: int,
    nodes_in_net: pd.DataFrame,
    segments_in_net: gpd.GeoDataFrame,
    pairs_in_net: pd.DataFrame,
    pair_paths_in_net: gpd.GeoDataFrame,
) -> list[dict]:
    """Compute features for every pair in a single Mars network.

    Returns one record per pair (with feature_qa_flag/reason).
    """
    if pairs_in_net.empty:
        return []

    # ------------------------------------------------------------------
    # Adjacency + Strahler order
    # ------------------------------------------------------------------
    n_nodes = int(nodes_in_net["node_id"].max()) + 1
    parents, children = build_directed_adjacency(segments_in_net, n_nodes)
    strahler = compute_strahler_orders(n_nodes, parents, children)

    # ------------------------------------------------------------------
    # Node coord lookup
    # ------------------------------------------------------------------
    node_xy: dict[int, tuple[float, float]] = {
        int(r["node_id"]): (float(r.geometry.x), float(r.geometry.y))
        for _, r in nodes_in_net.iterrows()
    }

    # ------------------------------------------------------------------
    # Stream-crossing filter setup: STRtree of all segments in this network
    # ------------------------------------------------------------------
    network_segments_list = list(segments_in_net.geometry)
    network_segments_tree = STRtree(network_segments_list)

    # ------------------------------------------------------------------
    # Pair-path lookup (one polyline per (pair_id, branch))
    # ------------------------------------------------------------------
    pair_paths_by_key: dict[tuple[str, str], LineString] = {}
    for _, r in pair_paths_in_net.iterrows():
        pair_paths_by_key[(str(r["pair_id"]), str(r["branch"]))] = r.geometry

    records: list[dict] = []
    for _, pair in pairs_in_net.iterrows():
        pair_id = str(pair["pair_id"])
        h1 = int(pair["head_1_node_id"])
        h2 = int(pair["head_2_node_id"])
        conf = int(pair["confluence_node_id"])
        L_1 = float(pair["L_1"])
        L_2 = float(pair["L_2"])
        headhead_dist_m = float(pair["headhead_dist_m"])
        n_segs_a = int(pair["n_segments_a"])
        n_segs_b = int(pair["n_segments_b"])

        qc_flags: list[str] = []

        # ---- Coords for both branch polylines -------------------------
        coords_a = coords_of_pair_path(pair_id, "A", pair_paths_by_key)
        coords_b = coords_of_pair_path(pair_id, "B", pair_paths_by_key)
        if coords_a is None or coords_b is None:
            qc_flags.append("missing_pair_path")

        # ---- orientation_diff_deg ------------------------------------
        orientation_diff_deg = float("nan")
        if coords_a is not None and coords_b is not None:
            vec_a, qa_a = line_direction_first_n_meters(
                coords_a, DIRECTION_SAMPLE_DISTANCE_M
            )
            vec_b, qa_b = line_direction_first_n_meters(
                coords_b, DIRECTION_SAMPLE_DISTANCE_M
            )
            if qa_a:
                qc_flags.extend(f"orient1:{f}" for f in qa_a)
            if qa_b:
                qc_flags.extend(f"orient2:{f}" for f in qa_b)
            if vec_a is not None and vec_b is not None:
                az_a = compute_azimuth(vec_a[0], vec_a[1])
                az_b = compute_azimuth(vec_b[0], vec_b[1])
                orientation_diff_deg = azimuth_difference(az_a, az_b)
            else:
                qc_flags.append("orientation_vector_undefined")

        # ---- apex_angle_deg ------------------------------------------
        xc, yc = node_xy[conf]
        x1, y1 = node_xy[h1]
        x2, y2 = node_xy[h2]
        apex_angle_deg = angle_between_vectors(
            (x1 - xc, y1 - yc), (x2 - xc, y2 - yc)
        )
        if math.isnan(apex_angle_deg):
            qc_flags.append("apex_zero_vector")

        # ---- headhead_dist_norm --------------------------------------
        L_sum = L_1 + L_2
        if L_sum > EPSILON:
            headhead_dist_norm = headhead_dist_m / L_sum
        else:
            headhead_dist_norm = float("nan")
            qc_flags.append("zero_path_length")
        if headhead_dist_m < EPSILON:
            qc_flags.append("coincident_nodes")

        # ---- strahler_order_diff -------------------------------------
        path_a_nodes = parse_id_list(pair["path_a_node_ids"])
        path_b_nodes = parse_id_list(pair["path_b_node_ids"])
        if (
            len(path_a_nodes) >= 2
            and len(path_b_nodes) >= 2
            and path_a_nodes[-1] == conf
            and path_b_nodes[-1] == conf
        ):
            bp_a = int(path_a_nodes[-2])
            bp_b = int(path_b_nodes[-2])
            if 0 <= bp_a < n_nodes and 0 <= bp_b < n_nodes:
                strahler_order_diff = float(
                    abs(int(strahler[bp_a]) - int(strahler[bp_b]))
                )
            else:
                strahler_order_diff = float("nan")
                qc_flags.append("strahler_branch_parent_out_of_range")
        else:
            strahler_order_diff = float("nan")
            qc_flags.append("strahler_branch_parent_unavailable")

        # ---- proximity_profile_norm ----------------------------------
        proximity_mean_m: float | None = None
        proximity_max_m: float | None = None
        proximity_profile_norm: float = float("nan")
        if coords_a is not None and coords_b is not None:
            sa = sample_path_coords_along_line(coords_a, N_PROXIMITY_SAMPLES)
            sb = sample_path_coords_along_line(coords_b, N_PROXIMITY_SAMPLES)
            if sa is not None and sb is not None:
                proximity_mean_m, proximity_max_m, proximity_profile_norm = (
                    compute_proximity_profile(sa, sb)
                )
            else:
                qc_flags.append("proximity_path_error")
        else:
            qc_flags.append("proximity_path_error")

        # ---- Stream-crossing filter ----------------------------------
        crosses = crosses_other_channel(
            (x1, y1),
            (x2, y2),
            network_segments_tree,
            network_segments_list,
            STREAM_FILTER_HEAD_BUFFER_M,
        )

        records.append(
            {
                "network_id": int(nid),
                "pair_id": pair_id,
                "head_id_1": h1,
                "head_id_2": h2,
                "head_node_id_1": h1,
                "head_node_id_2": h2,
                "confluence_id": conf,
                "confluence_node_id": conf,
                "path_length_1_m": L_1,
                "path_length_2_m": L_2,
                "L_sum_m": L_sum,
                "headhead_dist_m": headhead_dist_m,
                "n_segments_path_1": n_segs_a,
                "n_segments_path_2": n_segs_b,
                # Diagnostics not in MODEL_FEATURES
                "proximity_mean_m": proximity_mean_m,
                "proximity_max_m": proximity_max_m,
                # Model features (order matters for downstream assembly)
                "orientation_diff_deg": orientation_diff_deg,
                "headhead_dist_norm": headhead_dist_norm,
                "apex_angle_deg": apex_angle_deg,
                "strahler_order_diff": strahler_order_diff,
                "proximity_profile_norm": proximity_profile_norm,
                # QA + filter
                "feature_qa_flag": "ok" if not qc_flags else "issues",
                "feature_qa_reason": (
                    "" if not qc_flags else ",".join(qc_flags)
                ),
                "stream_crossing_drop": bool(crosses),
            }
        )

    return records


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_inputs() -> dict:
    log.info("Loading Phase 1 topology: %s", TOPOLOGY_GPKG)
    nodes = gpd.read_file(TOPOLOGY_GPKG, layer="mars_nodes")
    segments = gpd.read_file(TOPOLOGY_GPKG, layer="mars_segments")
    networks = gpd.read_file(TOPOLOGY_GPKG, layer="mars_networks")
    channel_heads = gpd.read_file(TOPOLOGY_GPKG, layer="mars_channel_heads")
    confluences = gpd.read_file(TOPOLOGY_GPKG, layer="mars_confluences")
    log.info("Loading Phase 2B pairs:    %s", PAIRS_GPKG)
    pairs = gpd.read_file(PAIRS_GPKG, layer="mars_pairs")
    pair_paths = gpd.read_file(PAIRS_GPKG, layer="mars_pair_paths")
    return {
        "nodes": nodes,
        "segments": segments,
        "networks": networks,
        "channel_heads": channel_heads,
        "confluences": confluences,
        "pairs": pairs,
        "pair_paths": pair_paths,
    }


def attach_outlet_node_id(
    full_df: pd.DataFrame, networks_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    outlet_by_net = dict(
        zip(networks_gdf["network_id"].astype(int), networks_gdf["outlet_node_id"].astype(int))
    )
    full_df["outlet_node_id"] = (
        full_df["network_id"].astype(int).map(outlet_by_net).astype("Int64")
    )
    return full_df


# ---------------------------------------------------------------------------
# Filter audit + model-ready assembly
# ---------------------------------------------------------------------------
def build_filtering_audit(
    n_total: int,
    n_with_all_features: int,
    n_stream_crossing_drop: int,
    n_model_ready: int,
) -> pd.DataFrame:
    """One row per filter (applied or skipped) for the audit CSV."""
    rows = [
        {
            "filter_name": "valid_pair_path",
            "source_file_or_function": (
                "scripts/extract_mars_first_meet_pairs.py:process_network "
                "(Phase 2B emitted only pairs with valid chained paths)"
            ),
            "filter_applied": True,
            "n_before": n_total,
            "n_removed": 0,
            "n_after": n_total,
            "filter_reason": (
                "Enforced upstream in Phase 2B: only pairs whose two head→"
                "confluence polylines could be chained from mars_segments "
                "were emitted."
            ),
        },
        {
            "filter_name": "all_5_features_computable",
            "source_file_or_function": (
                "channel_heads/geometric_analysis.py:"
                "GeometricFeaturesAnalyzer.compute_pair_geometry"
            ),
            "filter_applied": False,
            "n_before": n_total,
            "n_removed": 0,
            "n_after": n_total,
            "filter_reason": (
                "Earth pipeline does NOT drop pairs with NaN features; "
                "XGBoost handles NaN natively per notebooks/training/"
                "02_train_classifier.ipynb cell 20. NaNs preserved in "
                "the model-ready table; rows flagged via feature_qa_flag. "
                f"({n_with_all_features}/{n_total} have all 5 features "
                "non-NaN.)"
            ),
        },
        {
            "filter_name": "stream_crossing_filter",
            "source_file_or_function": (
                "channel_heads/coupling_analysis.py:"
                "CouplingAnalyzer._crosses_stream "
                "(use_stream_filter=True is the always-on default in "
                "evaluate_pairs_for_outlet)"
            ),
            "filter_applied": True,
            "n_before": n_total,
            "n_removed": n_stream_crossing_drop,
            "n_after": n_total - n_stream_crossing_drop,
            "filter_reason": (
                "Drop pairs whose straight line between the two heads "
                "intersects any segment of the same Mars network at a "
                "point other than the head endpoints themselves. Direct "
                "analogue of Earth's Bresenham-based interior-pixel check, "
                "re-implemented for vector geometry in projected metres "
                "(head buffer = "
                f"{STREAM_FILTER_HEAD_BUFFER_M:g} m). A visible channel "
                "between two heads makes the pair trivially non-touching."
            ),
        },
        {
            "filter_name": "prefilter_distance",
            "source_file_or_function": (
                "channel_heads/coupling_analysis.py:"
                "CouplingAnalyzer._heads_can_touch"
            ),
            "filter_applied": False,
            "n_before": n_total,
            "n_removed": 0,
            "n_after": n_total,
            "filter_reason": (
                "Earth-only. Tied to TopoToolbox raster stream threshold "
                "(2*sqrt(threshold_px)). Does not remove pairs from "
                "Earth either — only marks skipped_prefilter=True. Not "
                "ported to Mars."
            ),
        },
        {
            "filter_name": "filter_hard_negatives",
            "source_file_or_function": (
                "channel_heads/geometric_analysis.py:filter_hard_negatives"
            ),
            "filter_applied": False,
            "n_before": n_total,
            "n_removed": 0,
            "n_after": n_total,
            "filter_reason": (
                "Training-only. Thresholds (max_L_ratio=3, "
                "max_dist_ratio=5) are computed from POSITIVES in the "
                "labeled dataset; Mars has no labels at inference time, "
                "so the thresholds are undefined. Inapplicable to Mars."
            ),
        },
        {
            "filter_name": "stratified_subsample_negatives",
            "source_file_or_function": (
                "notebooks/training/00_full_pipeline.ipynb:"
                "stratified_subsample_negatives"
            ),
            "filter_applied": False,
            "n_before": n_total,
            "n_removed": 0,
            "n_after": n_total,
            "filter_reason": (
                "Training-only class-balance step (neg:pos = 3:1). "
                "Inapplicable to Mars (no labels)."
            ),
        },
        {
            "filter_name": "drop_missing_label_or_group",
            "source_file_or_function": (
                "notebooks/training/02_train_classifier.ipynb cell 20"
            ),
            "filter_applied": False,
            "n_before": n_total,
            "n_removed": 0,
            "n_after": n_total,
            "filter_reason": (
                "Earth-only. Drops rows missing 'touching' or 'basin' "
                "before model fit. Not relevant at inference time."
            ),
        },
        {
            "filter_name": "model_ready_total",
            "source_file_or_function": "this script (composite)",
            "filter_applied": True,
            "n_before": n_total,
            "n_removed": n_total - n_model_ready,
            "n_after": n_model_ready,
            "filter_reason": (
                "Composite: applied filters above. Equivalent to dropping "
                "pairs flagged stream_crossing_drop=True."
            ),
        },
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs()
    networks_gdf = inputs["networks"]
    nodes_gdf = inputs["nodes"]
    segments_gdf = inputs["segments"]
    pairs_gdf = inputs["pairs"]
    pair_paths_gdf = inputs["pair_paths"]

    log.info(
        "Inputs: %d networks, %d nodes, %d segments, %d pairs, %d pair-paths",
        len(networks_gdf),
        len(nodes_gdf),
        len(segments_gdf),
        len(pairs_gdf),
        len(pair_paths_gdf),
    )

    # Pre-group by network_id for speed
    nodes_by_nid = dict(tuple(nodes_gdf.groupby("network_id")))
    segs_by_nid = dict(tuple(segments_gdf.groupby("network_id")))
    pairs_by_nid = dict(tuple(pairs_gdf.groupby("network_id")))
    paths_by_nid = dict(tuple(pair_paths_gdf.groupby("network_id")))

    all_records: list[dict] = []
    for nid in tqdm(
        sorted(pairs_by_nid.keys()), desc="Networks"
    ):
        nid_int = int(nid)
        if (
            nid_int not in nodes_by_nid
            or nid_int not in segs_by_nid
            or nid_int not in paths_by_nid
        ):
            continue
        recs = process_network(
            nid_int,
            nodes_by_nid[nid_int],
            segs_by_nid[nid_int],
            pairs_by_nid[nid_int],
            paths_by_nid[nid_int],
        )
        all_records.extend(recs)

    df_all = pd.DataFrame(all_records)
    n_total = len(df_all)
    log.info("Computed features for %d pairs", n_total)

    df_all = attach_outlet_node_id(df_all, networks_gdf)

    # ----- Stats: how many have all 5 features non-NaN -----------------
    feature_nan_mask = df_all[MODEL_FEATURES].isna().any(axis=1)
    n_with_all_features = int((~feature_nan_mask).sum())
    log.info(
        "Pairs with all 5 features non-NaN: %d / %d (%.1f%%)",
        n_with_all_features,
        n_total,
        100 * n_with_all_features / n_total if n_total else 0,
    )

    # ----- Stream-crossing drop ---------------------------------------
    n_stream_crossing_drop = int(df_all["stream_crossing_drop"].sum())
    log.info(
        "Stream-crossing filter drops: %d / %d (%.1f%%)",
        n_stream_crossing_drop,
        n_total,
        100 * n_stream_crossing_drop / n_total if n_total else 0,
    )

    # ----- Build filter_status / filter_reason / excluded_by_filter ---
    df_all["filter_status"] = np.where(
        df_all["stream_crossing_drop"], "excluded", "kept"
    )
    df_all["filter_reason"] = np.where(
        df_all["stream_crossing_drop"],
        "stream_crossing_filter",
        "",
    )
    df_all["excluded_by_filter"] = df_all["stream_crossing_drop"].astype(bool)

    # ----- Model-ready subset -----------------------------------------
    df_ready = df_all.loc[~df_all["excluded_by_filter"]].copy()
    n_model_ready = len(df_ready)
    log.info("Model-ready pairs: %d", n_model_ready)

    # ----- Audit CSV --------------------------------------------------
    audit_df = build_filtering_audit(
        n_total, n_with_all_features, n_stream_crossing_drop, n_model_ready
    )
    audit_path = OUTPUT_DIR / "mars_pair_filtering_audit.csv"
    audit_df.to_csv(audit_path, index=False)
    log.info("Wrote audit: %s", audit_path)

    # ----- Final column ordering --------------------------------------
    id_cols = [
        "network_id",
        "pair_id",
        "head_id_1",
        "head_id_2",
        "head_node_id_1",
        "head_node_id_2",
        "confluence_id",
        "confluence_node_id",
        "outlet_node_id",
    ]
    path_cols = [
        "path_length_1_m",
        "path_length_2_m",
        "headhead_dist_m",
        "n_segments_path_1",
        "n_segments_path_2",
    ]
    diag_cols = [
        "proximity_mean_m",
        "proximity_max_m",
        "L_sum_m",
        "stream_crossing_drop",
    ]
    qa_cols = [
        "feature_qa_flag",
        "feature_qa_reason",
        "filter_status",
        "filter_reason",
        "excluded_by_filter",
    ]
    final_cols = id_cols + path_cols + MODEL_FEATURES + diag_cols + qa_cols
    df_all = df_all[final_cols]
    df_ready = df_ready[final_cols]

    # ----- Write parquet + csv ----------------------------------------
    all_pq = OUTPUT_DIR / "mars_pair_features_5feat_all.parquet"
    all_csv = OUTPUT_DIR / "mars_pair_features_5feat_all.csv"
    ready_pq = OUTPUT_DIR / "mars_pair_features_5feat_model_ready.parquet"
    ready_csv = OUTPUT_DIR / "mars_pair_features_5feat_model_ready.csv"
    df_all.to_parquet(all_pq, index=False)
    df_all.to_csv(all_csv, index=False)
    df_ready.to_parquet(ready_pq, index=False)
    df_ready.to_csv(ready_csv, index=False)
    log.info("Wrote: %s", all_pq)
    log.info("Wrote: %s", all_csv)
    log.info("Wrote: %s", ready_pq)
    log.info("Wrote: %s", ready_csv)

    # ----- Compatibility check against feature_columns.txt -----------
    saved_feats = [
        line.strip()
        for line in FEATURE_COLUMNS_TXT.read_text().splitlines()
        if line.strip()
    ]
    if saved_feats != MODEL_FEATURES:
        raise RuntimeError(
            "MODEL_FEATURES does not match models/feature_columns.txt:"
            f"\n  expected: {saved_feats}\n  got:      {MODEL_FEATURES}"
        )
    log.info(
        "Compatibility check OK: 5 features in expected order match "
        "models/feature_columns.txt"
    )

    # Sanity checks on model-ready table
    model_X = df_ready[MODEL_FEATURES]
    n_inf = int(np.isinf(model_X.to_numpy(dtype=float)).sum())
    n_nan = int(model_X.isna().sum().sum())
    log.info(
        "Model-ready feature matrix: shape=%s, NaN cells=%d, inf cells=%d "
        "(NaN is acceptable; XGBoost handles natively)",
        model_X.shape,
        n_nan,
        n_inf,
    )
    if n_inf:
        log.warning("Model-ready table contains %d infinite values", n_inf)

    # ----- Summary statistics before/after filter --------------------
    log.info("Feature summaries (all pairs):")
    for f in MODEL_FEATURES:
        s = df_all[f].dropna()
        log.info(
            "  %s: n=%d mean=%.3f median=%.3f std=%.3f",
            f,
            len(s),
            s.mean() if len(s) else float("nan"),
            s.median() if len(s) else float("nan"),
            s.std() if len(s) else float("nan"),
        )
    log.info("Feature summaries (model-ready):")
    for f in MODEL_FEATURES:
        s = df_ready[f].dropna()
        log.info(
            "  %s: n=%d mean=%.3f median=%.3f std=%.3f",
            f,
            len(s),
            s.mean() if len(s) else float("nan"),
            s.median() if len(s) else float("nan"),
            s.std() if len(s) else float("nan"),
        )

    # ----- QGIS validation GPKG ---------------------------------------
    val_gpkg = OUTPUT_DIR / "mars_features_validation.gpkg"
    if val_gpkg.exists():
        val_gpkg.unlink()

    pairs_gdf = pairs_gdf.merge(
        df_all[
            [
                "pair_id",
                "orientation_diff_deg",
                "headhead_dist_norm",
                "apex_angle_deg",
                "strahler_order_diff",
                "proximity_profile_norm",
                "feature_qa_flag",
                "feature_qa_reason",
                "filter_status",
                "filter_reason",
                "excluded_by_filter",
            ]
        ],
        on="pair_id",
        how="left",
    )
    pairs_gdf.to_file(
        val_gpkg, layer="mars_feature_pairs_all", driver="GPKG"
    )
    pairs_model_ready_gdf = pairs_gdf.loc[
        ~pairs_gdf["excluded_by_filter"].fillna(False)
    ]
    pairs_model_ready_gdf.to_file(
        val_gpkg, layer="mars_feature_pairs_model_ready", driver="GPKG"
    )
    # paths for the model-ready subset
    ready_pair_ids = set(pairs_model_ready_gdf["pair_id"])
    paths_ready = pair_paths_gdf.loc[
        pair_paths_gdf["pair_id"].isin(ready_pair_ids)
    ]
    paths_ready.to_file(
        val_gpkg, layer="mars_feature_pair_paths_model_ready", driver="GPKG"
    )
    inputs["channel_heads"].to_file(
        val_gpkg, layer="mars_feature_heads", driver="GPKG"
    )
    inputs["confluences"].to_file(
        val_gpkg, layer="mars_feature_confluences", driver="GPKG"
    )
    log.info("Wrote validation GPKG: %s", val_gpkg)

    log.info("Done.")


if __name__ == "__main__":
    main()
