#!/usr/bin/env python
"""Phase 3B — Run the production XGBoost model on the Mars 5-feature table.

Inference only. Does NOT recompute features, change filtering, retrain the
model, or do any CNN work.

Inputs:
  - data/Mars/model_inputs/mars_pair_features_5feat_model_ready.parquet
    (Phase 3A output — filtering already applied)
  - models/xgb_touching_classifier.json  (production XGBoost)
  - models/feature_columns.txt           (authoritative feature order)
  - models/optimal_threshold.txt         (production decision threshold)
  - data/Mars/topology/mars_vn_pairs.gpkg (for pair / path geometries)

Outputs (under data/Mars/model_outputs/):
  - mars_xgb_predictions_5feat.parquet
  - mars_xgb_predictions_5feat.csv
  - mars_xgb_predictions_5feat.gpkg        (6 layers)
  - mars_xgb_predictions_5feat_summary.csv
  - mars_xgb_predictions_by_network.csv
  - figures/
      probability_histogram.png
      predicted_touching_pct_by_network.png
      high_confidence_touching_examples.png
      uncertain_examples.png

Pre-prediction checks:
  - all 5 model feature names present
  - feature order matches models/feature_columns.txt
  - all model features are numeric
  - no infinite values in the model-ready feature matrix
  - NaN handling matches Earth: rows with NaN features are kept, XGBoost
    handles them natively (see notebooks/training/02_train_classifier.ipynb cell 20)

Primary interface
-----------------
``notebooks/mars/04_xgb_inference_5feat.ipynb`` is the primary, documented way
to understand this step; it calls the same shared package functions
(:mod:`channel_heads.inference`). This script is the headless batch wrapper that
writes the prediction tables, GeoPackage layers and summary figures.

Run:
    python scripts/run_mars_xgb_inference_5feat.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, MultiLineString

from channel_heads.inference import (
    load_feature_columns,
    load_threshold,
    load_xgb_model,
    predict_with_threshold,
    verify_feature_matrix,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURES_PARQUET = (
    PROJECT_ROOT
    / "data/Mars/model_inputs/mars_pair_features_5feat_model_ready.parquet"
)
PAIRS_GPKG = PROJECT_ROOT / "data/Mars/topology/mars_vn_pairs.gpkg"
TOPOLOGY_GPKG = (
    PROJECT_ROOT
    / "data/Mars/topology/mars_vn_topology_model_ready.gpkg"
)

MODEL_PATH = PROJECT_ROOT / "models/xgb_touching_classifier.json"
FEATURE_COLUMNS_TXT = PROJECT_ROOT / "models/feature_columns.txt"
OPTIMAL_THRESHOLD_TXT = PROJECT_ROOT / "models/optimal_threshold.txt"

OUTPUT_DIR = PROJECT_ROOT / "data/Mars/model_outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

MODEL_FEATURE_SET = "5feat_tabular"

# Bands used to build the GPKG subsets and example plots
HIGH_CONF_PROB_MIN = 0.80
UNCERTAIN_PROB_MIN = 0.45
UNCERTAIN_PROB_MAX = 0.70

log = logging.getLogger("mars_xgb_5feat")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Loading + verification
#
# Artifact loading, feature-matrix verification, model loading and the
# predict-with-threshold step are shared with the other Mars inference scripts
# via ``channel_heads.inference``. This script keeps the bespoke feature-order
# log and NaN reporting on top of the shared checks.
# ---------------------------------------------------------------------------
def verify_feature_table(
    df: pd.DataFrame, model_features: list[str]
) -> None:
    """Pre-prediction checks. Raises if any invariant is violated."""
    stats = verify_feature_matrix(df, model_features)

    # Feature order check: the columns in the dataframe (when subset to
    # model features) must be in the same order as in feature_columns.txt.
    # We will explicitly reorder when building X, but warn if the source
    # table is not pre-ordered.
    present_in_table_order = [c for c in df.columns if c in model_features]
    if present_in_table_order != model_features:
        log.info(
            "Model features appear in different order in the source table; "
            "will reorder explicitly before predict (source=%s, expected=%s)",
            present_in_table_order,
            model_features,
        )

    log.info(
        "Feature matrix: shape=(%d, %d), NaN cells=%d, rows with any NaN=%d "
        "(NaN is acceptable; XGBoost handles natively per notebooks/training/"
        "02_train_classifier.ipynb cell 20)",
        stats["n_rows"],
        stats["n_cols"],
        stats["n_nan_cells"],
        stats["n_nan_rows"],
    )


# ---------------------------------------------------------------------------
# Geometry assembly
# ---------------------------------------------------------------------------
def build_geometry_layers(
    df_pred: pd.DataFrame, pairs_gdf: gpd.GeoDataFrame, paths_gdf: gpd.GeoDataFrame
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Join predictions onto the Phase 2B pair geometries."""
    pair_geom_keep = pairs_gdf[["pair_id", "geometry"]]
    pred_pairs = df_pred.merge(pair_geom_keep, on="pair_id", how="left")
    pred_pairs = gpd.GeoDataFrame(pred_pairs, geometry="geometry", crs=pairs_gdf.crs)

    # paths: 2 rows per pair_id
    pred_paths = paths_gdf.merge(
        df_pred[
            [
                "pair_id",
                "network_id",
                "xgb_prob_touching",
                "xgb_pred_touching",
                "xgb_decision_threshold",
            ]
        ],
        on="pair_id",
        how="inner",
        suffixes=("", "_pred"),
    )
    # drop duplicate network_id if both present
    if "network_id_pred" in pred_paths.columns:
        pred_paths = pred_paths.drop(columns=["network_id_pred"])
    return pred_pairs, pred_paths


# ---------------------------------------------------------------------------
# Summary CSVs
# ---------------------------------------------------------------------------
def build_summary(
    df_pred: pd.DataFrame, threshold: float, model_path: Path
) -> pd.DataFrame:
    n_total = len(df_pred)
    n_pos = int((df_pred["xgb_pred_touching"] == 1).sum())
    n_neg = int((df_pred["xgb_pred_touching"] == 0).sum())
    n_high = int((df_pred["xgb_prob_touching"] >= HIGH_CONF_PROB_MIN).sum())
    n_unc = int(
        (
            (df_pred["xgb_prob_touching"] >= UNCERTAIN_PROB_MIN)
            & (df_pred["xgb_prob_touching"] <= UNCERTAIN_PROB_MAX)
        ).sum()
    )

    p = df_pred["xgb_prob_touching"]
    rows = [
        ("n_pairs_total", n_total),
        ("n_predicted_touching", n_pos),
        ("n_predicted_non_touching", n_neg),
        ("predicted_touching_fraction", n_pos / n_total if n_total else 0.0),
        ("n_high_confidence_touching_prob_ge_0.80", n_high),
        (
            f"n_uncertain_prob_in_[{UNCERTAIN_PROB_MIN}, {UNCERTAIN_PROB_MAX}]",
            n_unc,
        ),
        ("xgb_decision_threshold", threshold),
        ("xgb_prob_min", float(p.min())),
        ("xgb_prob_p05", float(p.quantile(0.05))),
        ("xgb_prob_p25", float(p.quantile(0.25))),
        ("xgb_prob_median", float(p.median())),
        ("xgb_prob_mean", float(p.mean())),
        ("xgb_prob_p75", float(p.quantile(0.75))),
        ("xgb_prob_p95", float(p.quantile(0.95))),
        ("xgb_prob_max", float(p.max())),
        ("model_path", str(model_path)),
        ("model_feature_set", MODEL_FEATURE_SET),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_by_network(df_pred: pd.DataFrame) -> pd.DataFrame:
    grp = df_pred.groupby("network_id")
    out = grp.agg(
        n_pairs=("pair_id", "count"),
        n_touching=("xgb_pred_touching", "sum"),
        mean_prob=("xgb_prob_touching", "mean"),
        median_prob=("xgb_prob_touching", "median"),
        max_prob=("xgb_prob_touching", "max"),
        min_prob=("xgb_prob_touching", "min"),
        n_high_conf=(
            "xgb_prob_touching",
            lambda s: int((s >= HIGH_CONF_PROB_MIN).sum()),
        ),
        n_uncertain=(
            "xgb_prob_touching",
            lambda s: int(
                (
                    (s >= UNCERTAIN_PROB_MIN) & (s <= UNCERTAIN_PROB_MAX)
                ).sum()
            ),
        ),
    ).reset_index()
    out["n_non_touching"] = out["n_pairs"] - out["n_touching"]
    out["touching_fraction"] = out["n_touching"] / out["n_pairs"]
    return out.sort_values(
        ["touching_fraction", "n_pairs"], ascending=[False, False]
    )


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_probability_histogram(
    df_pred: pd.DataFrame, threshold: float, output: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    p = df_pred["xgb_prob_touching"]
    ax.hist(p, bins=50, color="#4C78A8", edgecolor="white")
    ax.axvline(
        threshold,
        color="red",
        linestyle="--",
        label=f"threshold = {threshold:.4f}",
    )
    ax.axvline(
        HIGH_CONF_PROB_MIN,
        color="#2ca02c",
        linestyle=":",
        label=f"high-conf >= {HIGH_CONF_PROB_MIN:.2f}",
    )
    ax.axvspan(
        UNCERTAIN_PROB_MIN,
        UNCERTAIN_PROB_MAX,
        color="#ff7f0e",
        alpha=0.15,
        label=f"uncertain band [{UNCERTAIN_PROB_MIN:.2f}, {UNCERTAIN_PROB_MAX:.2f}]",
    )
    ax.set_xlabel("xgb_prob_touching")
    ax.set_ylabel("count")
    ax.set_title(
        f"Mars XGBoost prediction probabilities  (n={len(df_pred):,})"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_touching_pct_by_network(by_net: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    # Sort by network_id for x-axis stability
    df = by_net.sort_values("network_id").reset_index(drop=True)
    sizes = np.clip(df["n_pairs"].to_numpy(), 5, 200)
    ax.scatter(
        df["network_id"],
        df["touching_fraction"] * 100.0,
        s=sizes,
        c=df["n_pairs"],
        cmap="viridis",
        edgecolors="white",
        linewidths=0.3,
        alpha=0.8,
    )
    ax.set_xlabel("network_id")
    ax.set_ylabel("predicted touching (%)")
    ax.set_title(
        f"Predicted touching fraction per network  "
        f"(point size & color ~ n_pairs, n_networks={len(df)})"
    )
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_example_grid(
    examples: pd.DataFrame,
    pair_paths_lookup: dict[tuple[str, str], LineString],
    segments_by_nid: dict[int, gpd.GeoDataFrame],
    nodes_by_nid: dict[int, gpd.GeoDataFrame],
    title: str,
    output: Path,
) -> None:
    n = len(examples)
    if n == 0:
        log.warning("No examples to plot for %s — skipping", output.name)
        return
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, (_, srow) in zip(axes_flat, examples.iterrows()):
        pair_id = str(srow["pair_id"])
        nid = int(srow["network_id"])
        h1 = int(srow["head_node_id_1"])
        h2 = int(srow["head_node_id_2"])
        conf = int(srow["confluence_node_id"])
        prob = float(srow["xgb_prob_touching"])

        nodes_n = nodes_by_nid[nid]
        segs_n = segments_by_nid[nid]
        node_xy = {
            int(r["node_id"]): (float(r.geometry.x), float(r.geometry.y))
            for _, r in nodes_n.iterrows()
        }

        path_a = pair_paths_lookup.get((pair_id, "A"))
        path_b = pair_paths_lookup.get((pair_id, "B"))
        segs_n.plot(ax=ax, color="#cccccc", linewidth=0.55, zorder=1)
        if path_a is not None:
            gpd.GeoSeries([path_a]).plot(
                ax=ax, color="#e6550d", linewidth=1.8, zorder=3
            )
        if path_b is not None:
            gpd.GeoSeries([path_b]).plot(
                ax=ax, color="#1f78b4", linewidth=1.8, zorder=3
            )
        ax.scatter(*node_xy[h1], c="black", s=22, marker="o", zorder=5)
        ax.scatter(*node_xy[h2], c="black", s=22, marker="o", zorder=5)
        ax.scatter(
            *node_xy[conf],
            c="#ff7f00",
            s=32,
            marker="s",
            edgecolors="black",
            linewidths=0.4,
            zorder=6,
        )
        ax.set_title(
            f"net={nid}  prob={prob:.3f}\n{pair_id}",
            fontsize=7,
        )
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        if path_a is not None and path_b is not None:
            union = MultiLineString([path_a, path_b])
            minx, miny, maxx, maxy = union.bounds
            pad = max(maxx - minx, maxy - miny) * 0.15 + 1.0
            ax.set_xlim(minx - pad, maxx + pad)
            ax.set_ylim(miny - pad, maxy + pad)
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load inputs ---------------------------------------------------
    log.info("Loading model:       %s", MODEL_PATH)
    log.info("Loading features:    %s", FEATURES_PARQUET)
    log.info("Loading feature cols:%s", FEATURE_COLUMNS_TXT)
    log.info("Loading threshold:   %s", OPTIMAL_THRESHOLD_TXT)

    model_features = load_feature_columns(FEATURE_COLUMNS_TXT)
    threshold = load_threshold(OPTIMAL_THRESHOLD_TXT)
    log.info("Production threshold: %.6f", threshold)
    log.info("Model features (in order): %s", model_features)

    df = pd.read_parquet(FEATURES_PARQUET)
    log.info("Mars model-ready feature table: %d rows × %d cols", *df.shape)

    # --- Pre-prediction checks -----------------------------------------
    verify_feature_table(df, model_features)

    # --- Load model + verify its declared feature order -----------------
    model = load_xgb_model(MODEL_PATH, expected_features=model_features)
    log.info(
        "Compatibility check passed: model JSON feature_names match "
        "feature_columns.txt"
    )

    # --- Predict -------------------------------------------------------
    proba, pred = predict_with_threshold(model, df, model_features, threshold)
    log.info(
        "Predicted: %d touching, %d non-touching (touching fraction %.1f%%)",
        int(pred.sum()),
        int((pred == 0).sum()),
        100.0 * pred.mean(),
    )

    # --- Assemble prediction DataFrame ---------------------------------
    df_pred = df.copy()
    nan_per_row = (
        df_pred[model_features].isna().sum(axis=1).astype(int)
    )
    df_pred["xgb_prob_touching"] = proba
    df_pred["xgb_pred_touching"] = pred
    df_pred["xgb_decision_threshold"] = threshold
    df_pred["model_path"] = str(MODEL_PATH.relative_to(PROJECT_ROOT))
    df_pred["model_feature_set"] = MODEL_FEATURE_SET
    df_pred["inference_status"] = np.where(
        nan_per_row > 0, "ok_with_nan_features", "ok"
    )
    df_pred["n_nan_features"] = nan_per_row

    # --- Reorder columns for output -----------------------------------
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
    feat_cols = model_features
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
    pred_cols = [
        "xgb_prob_touching",
        "xgb_pred_touching",
        "xgb_decision_threshold",
        "model_path",
        "model_feature_set",
        "inference_status",
        "n_nan_features",
    ]
    final_cols = id_cols + path_cols + feat_cols + diag_cols + qa_cols + pred_cols
    df_pred = df_pred[[c for c in final_cols if c in df_pred.columns]]

    # --- Write parquet + csv ------------------------------------------
    out_pq = OUTPUT_DIR / "mars_xgb_predictions_5feat.parquet"
    out_csv = OUTPUT_DIR / "mars_xgb_predictions_5feat.csv"
    df_pred.to_parquet(out_pq, index=False)
    df_pred.to_csv(out_csv, index=False)
    log.info("Wrote: %s", out_pq)
    log.info("Wrote: %s", out_csv)

    # --- GeoPackage ---------------------------------------------------
    pairs_gdf = gpd.read_file(PAIRS_GPKG, layer="mars_pairs")
    paths_gdf = gpd.read_file(PAIRS_GPKG, layer="mars_pair_paths")
    pred_pairs, pred_paths = build_geometry_layers(df_pred, pairs_gdf, paths_gdf)

    gpkg = OUTPUT_DIR / "mars_xgb_predictions_5feat.gpkg"
    if gpkg.exists():
        gpkg.unlink()

    pred_pairs.to_file(gpkg, layer="mars_predicted_pairs", driver="GPKG")
    pred_paths.to_file(gpkg, layer="mars_predicted_pair_paths", driver="GPKG")
    pred_pairs[pred_pairs["xgb_pred_touching"] == 1].to_file(
        gpkg, layer="mars_predicted_touching_pairs", driver="GPKG"
    )
    pred_pairs[pred_pairs["xgb_pred_touching"] == 0].to_file(
        gpkg, layer="mars_predicted_non_touching_pairs", driver="GPKG"
    )
    high_conf = pred_pairs[pred_pairs["xgb_prob_touching"] >= HIGH_CONF_PROB_MIN]
    if not high_conf.empty:
        high_conf.to_file(
            gpkg, layer="mars_high_confidence_touching_pairs", driver="GPKG"
        )
    else:
        log.warning(
            "No pairs at prob >= %.2f — skipping mars_high_confidence_touching_pairs layer",
            HIGH_CONF_PROB_MIN,
        )
    unc = pred_pairs[
        (pred_pairs["xgb_prob_touching"] >= UNCERTAIN_PROB_MIN)
        & (pred_pairs["xgb_prob_touching"] <= UNCERTAIN_PROB_MAX)
    ]
    if not unc.empty:
        unc.to_file(gpkg, layer="mars_uncertain_pairs", driver="GPKG")
    else:
        log.warning(
            "No pairs in uncertain band [%.2f, %.2f] — skipping mars_uncertain_pairs layer",
            UNCERTAIN_PROB_MIN,
            UNCERTAIN_PROB_MAX,
        )
    log.info("Wrote GPKG: %s", gpkg)

    # --- Summary CSVs -------------------------------------------------
    summary = build_summary(df_pred, threshold, MODEL_PATH)
    summary_csv = OUTPUT_DIR / "mars_xgb_predictions_5feat_summary.csv"
    summary.to_csv(summary_csv, index=False)
    log.info("Wrote summary: %s", summary_csv)
    for _, row in summary.iterrows():
        log.info("  %s = %s", row["metric"], row["value"])

    by_net = build_by_network(df_pred)
    by_net_csv = OUTPUT_DIR / "mars_xgb_predictions_by_network.csv"
    by_net.to_csv(by_net_csv, index=False)
    log.info("Wrote per-network summary: %s", by_net_csv)

    # --- Plots --------------------------------------------------------
    plot_probability_histogram(
        df_pred, threshold, FIGURES_DIR / "probability_histogram.png"
    )
    plot_touching_pct_by_network(
        by_net, FIGURES_DIR / "predicted_touching_pct_by_network.png"
    )

    # Example plots: prep lookups
    nodes_gdf = gpd.read_file(TOPOLOGY_GPKG, layer="mars_nodes")
    segments_gdf = gpd.read_file(TOPOLOGY_GPKG, layer="mars_segments")
    nodes_by_nid = dict(tuple(nodes_gdf.groupby("network_id")))
    segs_by_nid = dict(tuple(segments_gdf.groupby("network_id")))
    pair_paths_lookup: dict[tuple[str, str], LineString] = {}
    for _, r in paths_gdf.iterrows():
        pair_paths_lookup[(str(r["pair_id"]), str(r["branch"]))] = r.geometry

    high_examples = (
        df_pred[df_pred["xgb_prob_touching"] >= HIGH_CONF_PROB_MIN]
        .sort_values("xgb_prob_touching", ascending=False)
        .head(20)
    )
    plot_example_grid(
        high_examples,
        pair_paths_lookup,
        segs_by_nid,
        nodes_by_nid,
        title=f"20 highest-probability touching predictions (prob >= {HIGH_CONF_PROB_MIN:.2f})",
        output=FIGURES_DIR / "high_confidence_touching_examples.png",
    )

    uncertain_pool = df_pred[
        (df_pred["xgb_prob_touching"] >= UNCERTAIN_PROB_MIN)
        & (df_pred["xgb_prob_touching"] <= UNCERTAIN_PROB_MAX)
    ].copy()
    if len(uncertain_pool) > 0:
        # Pick 20 evenly across the uncertain band
        if len(uncertain_pool) <= 20:
            unc_examples = uncertain_pool.sort_values("xgb_prob_touching")
        else:
            uncertain_pool = uncertain_pool.sort_values("xgb_prob_touching")
            idxs = np.linspace(
                0, len(uncertain_pool) - 1, 20
            ).round().astype(int)
            unc_examples = uncertain_pool.iloc[idxs]
    else:
        unc_examples = df_pred.iloc[0:0]
    plot_example_grid(
        unc_examples,
        pair_paths_lookup,
        segs_by_nid,
        nodes_by_nid,
        title=(
            f"20 uncertain predictions (prob in "
            f"[{UNCERTAIN_PROB_MIN:.2f}, {UNCERTAIN_PROB_MAX:.2f}])"
        ),
        output=FIGURES_DIR / "uncertain_examples.png",
    )

    log.info("Done. Outputs in: %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
