"""Mars 5-feature table generation + terrestrial pre-inference filtering (Phase 3A).

Moved from ``scripts/build_mars_pair_features_5feat.py``. Computes, for every
Mars channel-head pair, the 5 dimensionless features the production XGBoost
uses, mirroring the Earth definitions:

    orientation_diff_deg, headhead_dist_norm, apex_angle_deg,
    strahler_order_diff, proximity_profile_norm

and applies the always-on **stream-crossing filter** (the Mars vector analogue
of ``CouplingAnalyzer._crosses_stream``). The dimensionless feature *math* lives
in :mod:`channel_heads.features.geometry` / :mod:`channel_heads.features.paths`;
this module holds the Mars-graph orchestration (Strahler order on the oriented
valley graph, table assembly, filtering audit).

Public entry point: :func:`build_mars_features` (load → compute → write the
``mars_pair_features_5feat_*`` tables + audit CSV + validation GeoPackage).
The pure pieces (:func:`compute_strahler_orders`,
:func:`build_mars_directed_adjacency`, :func:`crosses_other_channel`,
:func:`build_feature_table`) are importable and unit-tested.

Filters intentionally NOT applied to Mars (documented in the audit CSV):
``filter_hard_negatives`` / ``stratified_subsample_negatives`` (training-only,
need labels) and ``prefilter_distance`` (Earth raster-threshold bound).
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

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
from channel_heads.io import paths
from channel_heads.logging_config import get_logger

log = get_logger("features.mars")

# Required model feature columns and order (must match models/feature_columns.txt).
MODEL_FEATURES: list[str] = [
    "orientation_diff_deg",
    "headhead_dist_norm",
    "apex_angle_deg",
    "strahler_order_diff",
    "proximity_profile_norm",
]

# Small head buffer (m) excluding the head endpoints from the crossing check —
# the vector analogue of Earth's "exclude first/last Bresenham pixel" rule.
STREAM_FILTER_HEAD_BUFFER_M = 5.0

DEFAULT_FEATURE_COLUMNS_TXT = paths.MODELS_DIR / "feature_columns.txt"


# --------------------------------------------------------------------------- #
# Pure helpers (no IO)
# --------------------------------------------------------------------------- #
def compute_strahler_orders(
    n_nodes: int,
    parents: list[list[int]],
    children: list[list[int]],
) -> np.ndarray:
    """Standard Strahler order on the directed Mars graph.

    Channel heads (no parents) → 1; a confluence whose max parent order appears
    ≥ 2 times → max + 1, else max; a connector (one parent) → parent's order.
    Nodes never visited get order 0.
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


def build_mars_directed_adjacency(
    segments_df: pd.DataFrame, n_nodes: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Build parent/children adjacency from ``mars_segments``.

    Uses the Phase-1 ``upstream_node_id`` → ``downstream_node_id`` orientation.
    Self-loops and out-of-range edges are skipped. (Distinct from
    :func:`channel_heads.pairing.build_directed_adjacency`, which also returns an
    edge→segment map.)
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


def parse_id_list(s: str) -> list[int]:
    if not isinstance(s, str) or not s:
        return []
    return [int(tok) for tok in s.split(",") if tok != ""]


def coords_of_pair_path(pair_id: str, branch: str, pair_paths_by_key: dict) -> np.ndarray | None:
    geom = pair_paths_by_key.get((pair_id, branch))
    if geom is None or geom.is_empty:
        return None
    return np.asarray(geom.coords, dtype=float)


def crosses_other_channel(
    head_1_xy: tuple[float, float],
    head_2_xy: tuple[float, float],
    network_segments_tree,
    network_segments_list: list,
    head_buffer_m: float = STREAM_FILTER_HEAD_BUFFER_M,
) -> bool:
    """Mars analogue of ``CouplingAnalyzer._crosses_stream``.

    True if the straight line between the two heads intersects any segment of
    the same network away from the head endpoints (a small disc of radius
    ``head_buffer_m`` around each head is excluded). True ⇒ drop the pair.
    """
    from shapely.geometry import LineString, Point

    p1 = Point(*head_1_xy)
    p2 = Point(*head_2_xy)
    straight = LineString([head_1_xy, head_2_xy])
    if straight.length < EPSILON:
        return False
    straight_interior = straight.difference(p1.buffer(head_buffer_m)).difference(
        p2.buffer(head_buffer_m)
    )
    if straight_interior.is_empty:
        return False
    cand_idx = network_segments_tree.query(straight_interior)
    for i in cand_idx:
        if network_segments_list[int(i)].intersects(straight_interior):
            return True
    return False


# --------------------------------------------------------------------------- #
# Per-network feature computation
# --------------------------------------------------------------------------- #
def process_network(
    nid: int,
    nodes_in_net: pd.DataFrame,
    segments_in_net,
    pairs_in_net: pd.DataFrame,
    pair_paths_in_net,
) -> list[dict]:
    """Compute one record per pair in a single Mars network."""
    from shapely.strtree import STRtree

    if pairs_in_net.empty:
        return []

    n_nodes = int(nodes_in_net["node_id"].max()) + 1
    parents, children = build_mars_directed_adjacency(segments_in_net, n_nodes)
    strahler = compute_strahler_orders(n_nodes, parents, children)

    node_xy: dict[int, tuple[float, float]] = {
        int(r["node_id"]): (float(r.geometry.x), float(r.geometry.y))
        for _, r in nodes_in_net.iterrows()
    }

    network_segments_list = list(segments_in_net.geometry)
    network_segments_tree = STRtree(network_segments_list)

    pair_paths_by_key: dict[tuple[str, str], object] = {}
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

        coords_a = coords_of_pair_path(pair_id, "A", pair_paths_by_key)
        coords_b = coords_of_pair_path(pair_id, "B", pair_paths_by_key)
        if coords_a is None or coords_b is None:
            qc_flags.append("missing_pair_path")

        # orientation_diff_deg
        orientation_diff_deg = float("nan")
        if coords_a is not None and coords_b is not None:
            vec_a, qa_a = line_direction_first_n_meters(coords_a, DIRECTION_SAMPLE_DISTANCE_M)
            vec_b, qa_b = line_direction_first_n_meters(coords_b, DIRECTION_SAMPLE_DISTANCE_M)
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

        # apex_angle_deg
        xc, yc = node_xy[conf]
        x1, y1 = node_xy[h1]
        x2, y2 = node_xy[h2]
        apex_angle_deg = angle_between_vectors((x1 - xc, y1 - yc), (x2 - xc, y2 - yc))
        if math.isnan(apex_angle_deg):
            qc_flags.append("apex_zero_vector")

        # headhead_dist_norm
        L_sum = L_1 + L_2
        if L_sum > EPSILON:
            headhead_dist_norm = headhead_dist_m / L_sum
        else:
            headhead_dist_norm = float("nan")
            qc_flags.append("zero_path_length")
        if headhead_dist_m < EPSILON:
            qc_flags.append("coincident_nodes")

        # strahler_order_diff
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
                strahler_order_diff = float(abs(int(strahler[bp_a]) - int(strahler[bp_b])))
            else:
                strahler_order_diff = float("nan")
                qc_flags.append("strahler_branch_parent_out_of_range")
        else:
            strahler_order_diff = float("nan")
            qc_flags.append("strahler_branch_parent_unavailable")

        # proximity_profile_norm
        proximity_mean_m: float | None = None
        proximity_max_m: float | None = None
        proximity_profile_norm = float("nan")
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

        crosses = crosses_other_channel(
            (x1, y1), (x2, y2), network_segments_tree, network_segments_list,
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
                "proximity_mean_m": proximity_mean_m,
                "proximity_max_m": proximity_max_m,
                "orientation_diff_deg": orientation_diff_deg,
                "headhead_dist_norm": headhead_dist_norm,
                "apex_angle_deg": apex_angle_deg,
                "strahler_order_diff": strahler_order_diff,
                "proximity_profile_norm": proximity_profile_norm,
                "feature_qa_flag": "ok" if not qc_flags else "issues",
                "feature_qa_reason": "" if not qc_flags else ",".join(qc_flags),
                "stream_crossing_drop": bool(crosses),
            }
        )
    return records


def attach_outlet_node_id(full_df: pd.DataFrame, networks_gdf) -> pd.DataFrame:
    outlet_by_net = dict(
        zip(networks_gdf["network_id"].astype(int), networks_gdf["outlet_node_id"].astype(int))
    )
    full_df["outlet_node_id"] = (
        full_df["network_id"].astype(int).map(outlet_by_net).astype("Int64")
    )
    return full_df


# Final column order (preserved from the script).
_ID_COLS = [
    "network_id", "pair_id", "head_id_1", "head_id_2",
    "head_node_id_1", "head_node_id_2", "confluence_id",
    "confluence_node_id", "outlet_node_id",
]
_PATH_COLS = [
    "path_length_1_m", "path_length_2_m", "headhead_dist_m",
    "n_segments_path_1", "n_segments_path_2",
]
_DIAG_COLS = ["proximity_mean_m", "proximity_max_m", "L_sum_m", "stream_crossing_drop"]
_QA_COLS = [
    "feature_qa_flag", "feature_qa_reason", "filter_status",
    "filter_reason", "excluded_by_filter",
]
FINAL_COLUMNS = _ID_COLS + _PATH_COLS + MODEL_FEATURES + _DIAG_COLS + _QA_COLS


def build_feature_table(
    networks_gdf,
    nodes_gdf,
    segments_gdf,
    pairs_gdf,
    pair_paths_gdf,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute features for all networks and assemble (all, model_ready, audit).

    No files are written — this is the testable core of :func:`build_mars_features`.
    """
    nodes_by_nid = dict(tuple(nodes_gdf.groupby("network_id")))
    segs_by_nid = dict(tuple(segments_gdf.groupby("network_id")))
    pairs_by_nid = dict(tuple(pairs_gdf.groupby("network_id")))
    paths_by_nid = dict(tuple(pair_paths_gdf.groupby("network_id")))

    all_records: list[dict] = []
    for nid in sorted(pairs_by_nid.keys()):
        nid_int = int(nid)
        if nid_int not in nodes_by_nid or nid_int not in segs_by_nid or nid_int not in paths_by_nid:
            continue
        all_records.extend(
            process_network(
                nid_int,
                nodes_by_nid[nid_int],
                segs_by_nid[nid_int],
                pairs_by_nid[nid_int],
                paths_by_nid[nid_int],
            )
        )

    df_all = pd.DataFrame(all_records)
    n_total = len(df_all)
    if n_total == 0:
        empty = pd.DataFrame(columns=FINAL_COLUMNS)
        return empty, empty.copy(), build_filtering_audit(0, 0, 0, 0)

    df_all = attach_outlet_node_id(df_all, networks_gdf)

    n_with_all_features = int((~df_all[MODEL_FEATURES].isna().any(axis=1)).sum())
    n_stream_crossing_drop = int(df_all["stream_crossing_drop"].sum())

    df_all["filter_status"] = np.where(df_all["stream_crossing_drop"], "excluded", "kept")
    df_all["filter_reason"] = np.where(
        df_all["stream_crossing_drop"], "stream_crossing_filter", ""
    )
    df_all["excluded_by_filter"] = df_all["stream_crossing_drop"].astype(bool)

    df_ready = df_all.loc[~df_all["excluded_by_filter"]].copy()
    n_model_ready = len(df_ready)

    audit_df = build_filtering_audit(
        n_total, n_with_all_features, n_stream_crossing_drop, n_model_ready
    )

    df_all = df_all[FINAL_COLUMNS]
    df_ready = df_ready[FINAL_COLUMNS]
    log.info(
        "Features: %d pairs (%d with all 5; %d stream-crossing drops; %d model-ready)",
        n_total, n_with_all_features, n_stream_crossing_drop, n_model_ready,
    )
    return df_all, df_ready, audit_df


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
                "channel_heads/mars/pairs.py:process_network "
                "(Phase 2B emitted only pairs with valid chained paths)"
            ),
            "filter_applied": True,
            "n_before": n_total, "n_removed": 0, "n_after": n_total,
            "filter_reason": (
                "Enforced upstream in Phase 2B: only pairs whose two head→"
                "confluence polylines could be chained from mars_segments were emitted."
            ),
        },
        {
            "filter_name": "all_5_features_computable",
            "source_file_or_function": (
                "channel_heads/features/mars_features.py:build_feature_table"
            ),
            "filter_applied": False,
            "n_before": n_total, "n_removed": 0, "n_after": n_total,
            "filter_reason": (
                "Earth pipeline does NOT drop pairs with NaN features; XGBoost "
                "handles NaN natively. NaNs preserved; rows flagged via "
                f"feature_qa_flag. ({n_with_all_features}/{n_total} have all 5 "
                "features non-NaN.)"
            ),
        },
        {
            "filter_name": "stream_crossing_filter",
            "source_file_or_function": (
                "channel_heads/features/mars_features.py:crosses_other_channel "
                "(Mars analogue of CouplingAnalyzer._crosses_stream, always-on)"
            ),
            "filter_applied": True,
            "n_before": n_total, "n_removed": n_stream_crossing_drop,
            "n_after": n_total - n_stream_crossing_drop,
            "filter_reason": (
                "Drop pairs whose straight head-to-head line intersects any "
                "segment of the same Mars network away from the head endpoints "
                f"(head buffer = {STREAM_FILTER_HEAD_BUFFER_M:g} m). A visible "
                "channel between two heads makes the pair trivially non-touching."
            ),
        },
        {
            "filter_name": "prefilter_distance",
            "source_file_or_function": (
                "channel_heads/coupling_analysis.py:CouplingAnalyzer._heads_can_touch"
            ),
            "filter_applied": False,
            "n_before": n_total, "n_removed": 0, "n_after": n_total,
            "filter_reason": (
                "Earth-only. Tied to TopoToolbox raster stream threshold; only "
                "flags rows on Earth. Not ported to Mars."
            ),
        },
        {
            "filter_name": "filter_hard_negatives",
            "source_file_or_function": (
                "channel_heads/geometric_analysis.py:filter_hard_negatives"
            ),
            "filter_applied": False,
            "n_before": n_total, "n_removed": 0, "n_after": n_total,
            "filter_reason": (
                "Training-only. Thresholds are computed from POSITIVES; Mars has "
                "no labels at inference time, so they are undefined."
            ),
        },
        {
            "filter_name": "stratified_subsample_negatives",
            "source_file_or_function": (
                "notebooks/training/00_full_pipeline.ipynb:stratified_subsample_negatives"
            ),
            "filter_applied": False,
            "n_before": n_total, "n_removed": 0, "n_after": n_total,
            "filter_reason": "Training-only class-balance step. Inapplicable to Mars (no labels).",
        },
        {
            "filter_name": "drop_missing_label_or_group",
            "source_file_or_function": "notebooks/training/02_train_classifier.ipynb cell 20",
            "filter_applied": False,
            "n_before": n_total, "n_removed": 0, "n_after": n_total,
            "filter_reason": "Earth-only. Drops rows missing 'touching'/'basin' before fit.",
        },
        {
            "filter_name": "model_ready_total",
            "source_file_or_function": "this module (composite)",
            "filter_applied": True,
            "n_before": n_total, "n_removed": n_total - n_model_ready, "n_after": n_model_ready,
            "filter_reason": "Composite: equivalent to dropping stream_crossing_drop=True pairs.",
        },
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# IO + high-level entry point
# --------------------------------------------------------------------------- #
def load_inputs(topology_gpkg, pairs_gpkg) -> dict:
    from channel_heads.io import read_gpkg

    log.info("Loading topology %s + pairs %s", topology_gpkg, pairs_gpkg)
    return {
        "nodes": read_gpkg(topology_gpkg, layer="mars_nodes"),
        "segments": read_gpkg(topology_gpkg, layer="mars_segments"),
        "networks": read_gpkg(topology_gpkg, layer="mars_networks"),
        "channel_heads": read_gpkg(topology_gpkg, layer="mars_channel_heads"),
        "confluences": read_gpkg(topology_gpkg, layer="mars_confluences"),
        "pairs": read_gpkg(pairs_gpkg, layer="mars_pairs"),
        "pair_paths": read_gpkg(pairs_gpkg, layer="mars_pair_paths"),
    }


def _verify_feature_columns(feature_columns_txt: Path) -> None:
    """Compare MODEL_FEATURES against models/feature_columns.txt if it exists."""
    feature_columns_txt = Path(feature_columns_txt)
    if not feature_columns_txt.exists():
        log.warning(
            "feature_columns.txt not found (%s); skipping compatibility check",
            feature_columns_txt,
        )
        return
    saved = [ln.strip() for ln in feature_columns_txt.read_text().splitlines() if ln.strip()]
    if saved != MODEL_FEATURES:
        raise RuntimeError(
            f"MODEL_FEATURES does not match {feature_columns_txt}:"
            f"\n  expected: {saved}\n  got:      {MODEL_FEATURES}"
        )
    log.info("Compatibility check OK: 5 features match %s", feature_columns_txt.name)


def _write_validation_gpkg(df_all, inputs, out_gpkg) -> None:
    from channel_heads.io import write_gpkg

    out_gpkg = Path(out_gpkg)
    if out_gpkg.exists():
        out_gpkg.unlink()
    cols = ["pair_id", *MODEL_FEATURES, "feature_qa_flag", "feature_qa_reason",
            "filter_status", "filter_reason", "excluded_by_filter"]
    pairs_gdf = inputs["pairs"].merge(df_all[cols], on="pair_id", how="left")
    write_gpkg(pairs_gdf, out_gpkg, layer="mars_feature_pairs_all")
    ready = pairs_gdf.loc[~pairs_gdf["excluded_by_filter"].fillna(False)]
    write_gpkg(ready, out_gpkg, layer="mars_feature_pairs_model_ready")
    ready_ids = set(ready["pair_id"])
    paths_ready = inputs["pair_paths"].loc[inputs["pair_paths"]["pair_id"].isin(ready_ids)]
    write_gpkg(paths_ready, out_gpkg, layer="mars_feature_pair_paths_model_ready")
    write_gpkg(inputs["channel_heads"], out_gpkg, layer="mars_feature_heads")
    write_gpkg(inputs["confluences"], out_gpkg, layer="mars_feature_confluences")
    log.info("Wrote validation GPKG: %s", out_gpkg)


def build_mars_features(
    topology_gpkg=paths.MARS_TOPOLOGY_GPKG,
    pairs_gpkg=paths.MARS_PAIRS_GPKG,
    output_dir=paths.MARS_MODEL_INPUTS_DIR,
    feature_columns_txt=DEFAULT_FEATURE_COLUMNS_TXT,
    *,
    write: bool = True,
    write_validation_gpkg: bool = True,
) -> dict:
    """Phase 3A: build the Mars 5-feature tables from the topology + pairs GPKGs.

    Writes (when ``write``): ``mars_pair_features_5feat_{all,model_ready}.{parquet,csv}``,
    ``mars_pair_filtering_audit.csv`` and ``mars_features_validation.gpkg`` under
    ``output_dir``. Returns ``{"all", "model_ready", "audit", "paths"}``.
    """
    from channel_heads.io import write_table

    inputs = load_inputs(topology_gpkg, pairs_gpkg)
    df_all, df_ready, audit_df = build_feature_table(
        inputs["networks"], inputs["nodes"], inputs["segments"],
        inputs["pairs"], inputs["pair_paths"],
    )

    written: dict[str, Path] = {}
    if write:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        audit_path = output_dir / "mars_pair_filtering_audit.csv"
        audit_df.to_csv(audit_path, index=False)
        written["audit"] = audit_path
        written["all"] = write_table(df_all, output_dir / "mars_pair_features_5feat_all.parquet")
        written["model_ready"] = write_table(
            df_ready, output_dir / "mars_pair_features_5feat_model_ready.parquet"
        )
        _verify_feature_columns(feature_columns_txt)
        if write_validation_gpkg:
            try:
                _write_validation_gpkg(df_all, inputs, output_dir / "mars_features_validation.gpkg")
                written["validation_gpkg"] = output_dir / "mars_features_validation.gpkg"
            except Exception as exc:  # best-effort QGIS artifact
                log.warning("Validation GPKG failed: %s", exc)

    return {"all": df_all, "model_ready": df_ready, "audit": audit_df, "paths": written}


__all__ = [
    "build_mars_features",
    "build_feature_table",
    "process_network",
    "compute_strahler_orders",
    "build_mars_directed_adjacency",
    "crosses_other_channel",
    "parse_id_list",
    "build_filtering_audit",
    "attach_outlet_node_id",
    "MODEL_FEATURES",
    "FINAL_COLUMNS",
    "STREAM_FILTER_HEAD_BUFFER_M",
]
