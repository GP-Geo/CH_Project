"""Mars tabular (5-feature) XGBoost inference (Phase 3B).

Moved from ``scripts/run_mars_xgb_inference_5feat.py``. Inference only — does not
recompute features, change filtering, or retrain. Loads the production tabular
XGBoost, predicts P(touching) on the Phase-3A model-ready table, applies the
production operating threshold, and exports prediction tables + GeoPackage
layers + summary CSVs (+ best-effort figures).

The model/artifact glue is shared via :mod:`channel_heads.models.xgboost`
(→ :mod:`channel_heads.inference.xgb`); this module holds the Mars-specific
prediction assembly, summaries, geometry layers and output export.

Public entry point: :func:`run_mars_tabular_inference`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from channel_heads.io import paths
from channel_heads.logging_config import get_logger
from channel_heads.models.xgboost import (
    load_feature_columns,
    load_threshold,
    load_xgb_model,
    predict_with_threshold,
    verify_feature_matrix,
)

log = get_logger("models.mars_inference")

MODEL_FEATURE_SET = "5feat_tabular"
HIGH_CONF_PROB_MIN = 0.80
UNCERTAIN_PROB_MIN = 0.45
UNCERTAIN_PROB_MAX = 0.70

DEFAULT_FEATURES_PARQUET = paths.MARS_MODEL_INPUTS_DIR / "mars_pair_features_5feat_model_ready.parquet"
DEFAULT_FEATURE_COLUMNS_TXT = paths.MODELS_DIR / "feature_columns.txt"
DEFAULT_THRESHOLD_TXT = paths.MODELS_DIR / "optimal_threshold.txt"

# Output prediction-column block (appended after the Phase-3A columns).
_PRED_COLS = [
    "xgb_prob_touching",
    "xgb_pred_touching",
    "xgb_decision_threshold",
    "model_path",
    "model_feature_set",
    "inference_status",
    "n_nan_features",
]
_ID_COLS = [
    "network_id", "pair_id", "head_id_1", "head_id_2", "head_node_id_1",
    "head_node_id_2", "confluence_id", "confluence_node_id", "outlet_node_id",
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


# --------------------------------------------------------------------------- #
# Validation + thresholding
# --------------------------------------------------------------------------- #
def validate_feature_columns(df: pd.DataFrame, model_features: list[str]) -> dict[str, int]:
    """Pre-prediction checks; returns matrix stats. Raises on any violation."""
    stats = verify_feature_matrix(df, model_features)
    present = [c for c in df.columns if c in model_features]
    if present != model_features:
        log.info(
            "Model features in different order in source table; will reorder "
            "before predict (source=%s, expected=%s)", present, model_features,
        )
    log.info(
        "Feature matrix: shape=(%d, %d), NaN cells=%d, rows with any NaN=%d",
        stats["n_rows"], stats["n_cols"], stats["n_nan_cells"], stats["n_nan_rows"],
    )
    return stats


def apply_operating_threshold(proba: np.ndarray, threshold: float) -> np.ndarray:
    """Decision = ``(proba >= threshold)`` as int (the production rule)."""
    return (np.asarray(proba) >= threshold).astype(int)


# --------------------------------------------------------------------------- #
# Prediction assembly + summaries (pure)
# --------------------------------------------------------------------------- #
def assemble_predictions(
    df: pd.DataFrame,
    proba: np.ndarray,
    pred: np.ndarray,
    threshold: float,
    model_features: list[str],
    model_path_rel: str,
) -> pd.DataFrame:
    """Attach prediction columns and apply the canonical output column order."""
    df_pred = df.copy()
    nan_per_row = df_pred[model_features].isna().sum(axis=1).astype(int)
    df_pred["xgb_prob_touching"] = proba
    df_pred["xgb_pred_touching"] = pred
    df_pred["xgb_decision_threshold"] = threshold
    df_pred["model_path"] = model_path_rel
    df_pred["model_feature_set"] = MODEL_FEATURE_SET
    df_pred["inference_status"] = np.where(nan_per_row > 0, "ok_with_nan_features", "ok")
    df_pred["n_nan_features"] = nan_per_row

    final_cols = _ID_COLS + _PATH_COLS + model_features + _DIAG_COLS + _QA_COLS + _PRED_COLS
    return df_pred[[c for c in final_cols if c in df_pred.columns]]


def summarize_predictions(df_pred: pd.DataFrame, threshold: float, model_path) -> pd.DataFrame:
    n_total = len(df_pred)
    p = df_pred["xgb_prob_touching"]
    n_pos = int((df_pred["xgb_pred_touching"] == 1).sum())
    n_neg = int((df_pred["xgb_pred_touching"] == 0).sum())
    n_high = int((p >= HIGH_CONF_PROB_MIN).sum())
    n_unc = int(((p >= UNCERTAIN_PROB_MIN) & (p <= UNCERTAIN_PROB_MAX)).sum())
    rows = [
        ("n_pairs_total", n_total),
        ("n_predicted_touching", n_pos),
        ("n_predicted_non_touching", n_neg),
        ("predicted_touching_fraction", n_pos / n_total if n_total else 0.0),
        ("n_high_confidence_touching_prob_ge_0.80", n_high),
        (f"n_uncertain_prob_in_[{UNCERTAIN_PROB_MIN}, {UNCERTAIN_PROB_MAX}]", n_unc),
        ("xgb_decision_threshold", threshold),
        ("xgb_prob_min", float(p.min()) if n_total else float("nan")),
        ("xgb_prob_p05", float(p.quantile(0.05)) if n_total else float("nan")),
        ("xgb_prob_p25", float(p.quantile(0.25)) if n_total else float("nan")),
        ("xgb_prob_median", float(p.median()) if n_total else float("nan")),
        ("xgb_prob_mean", float(p.mean()) if n_total else float("nan")),
        ("xgb_prob_p75", float(p.quantile(0.75)) if n_total else float("nan")),
        ("xgb_prob_p95", float(p.quantile(0.95)) if n_total else float("nan")),
        ("xgb_prob_max", float(p.max()) if n_total else float("nan")),
        ("model_path", str(model_path)),
        ("model_feature_set", MODEL_FEATURE_SET),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def summarize_by_network(df_pred: pd.DataFrame) -> pd.DataFrame:
    grp = df_pred.groupby("network_id")
    out = grp.agg(
        n_pairs=("pair_id", "count"),
        n_touching=("xgb_pred_touching", "sum"),
        mean_prob=("xgb_prob_touching", "mean"),
        median_prob=("xgb_prob_touching", "median"),
        max_prob=("xgb_prob_touching", "max"),
        min_prob=("xgb_prob_touching", "min"),
        n_high_conf=("xgb_prob_touching", lambda s: int((s >= HIGH_CONF_PROB_MIN).sum())),
        n_uncertain=(
            "xgb_prob_touching",
            lambda s: int(((s >= UNCERTAIN_PROB_MIN) & (s <= UNCERTAIN_PROB_MAX)).sum()),
        ),
    ).reset_index()
    out["n_non_touching"] = out["n_pairs"] - out["n_touching"]
    out["touching_fraction"] = out["n_touching"] / out["n_pairs"]
    return out.sort_values(["touching_fraction", "n_pairs"], ascending=[False, False])


# --------------------------------------------------------------------------- #
# Geometry layers + export
# --------------------------------------------------------------------------- #
def build_geometry_layers(df_pred, pairs_gdf, paths_gdf):
    """Join predictions onto the Phase-2B pair / path geometries."""
    import geopandas as gpd

    pred_pairs = df_pred.merge(pairs_gdf[["pair_id", "geometry"]], on="pair_id", how="left")
    pred_pairs = gpd.GeoDataFrame(pred_pairs, geometry="geometry", crs=pairs_gdf.crs)
    pred_paths = paths_gdf.merge(
        df_pred[[
            "pair_id", "network_id", "xgb_prob_touching",
            "xgb_pred_touching", "xgb_decision_threshold",
        ]],
        on="pair_id", how="inner", suffixes=("", "_pred"),
    )
    if "network_id_pred" in pred_paths.columns:
        pred_paths = pred_paths.drop(columns=["network_id_pred"])
    return pred_pairs, pred_paths


def export_prediction_outputs(
    df_pred: pd.DataFrame,
    summary: pd.DataFrame,
    by_net: pd.DataFrame,
    output_dir: Path,
    pairs_gpkg=paths.MARS_PAIRS_GPKG,
) -> dict[str, Path]:
    """Write parquet/csv + GeoPackage layers + summary CSVs. Returns paths."""
    from channel_heads.io import read_gpkg, write_table

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    written["predictions"] = write_table(
        df_pred, output_dir / "mars_xgb_predictions_5feat.parquet"
    )
    summary.to_csv(output_dir / "mars_xgb_predictions_5feat_summary.csv", index=False)
    written["summary"] = output_dir / "mars_xgb_predictions_5feat_summary.csv"
    by_net.to_csv(output_dir / "mars_xgb_predictions_by_network.csv", index=False)
    written["by_network"] = output_dir / "mars_xgb_predictions_by_network.csv"

    # GeoPackage (6 conditional layers).
    pairs_gdf = read_gpkg(pairs_gpkg, layer="mars_pairs")
    paths_gdf = read_gpkg(pairs_gpkg, layer="mars_pair_paths")
    pred_pairs, pred_paths = build_geometry_layers(df_pred, pairs_gdf, paths_gdf)
    gpkg = output_dir / "mars_xgb_predictions_5feat.gpkg"
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
        high_conf.to_file(gpkg, layer="mars_high_confidence_touching_pairs", driver="GPKG")
    unc = pred_pairs[
        (pred_pairs["xgb_prob_touching"] >= UNCERTAIN_PROB_MIN)
        & (pred_pairs["xgb_prob_touching"] <= UNCERTAIN_PROB_MAX)
    ]
    if not unc.empty:
        unc.to_file(gpkg, layer="mars_uncertain_pairs", driver="GPKG")
    written["gpkg"] = gpkg
    return written


# --------------------------------------------------------------------------- #
# Figures (best-effort diagnostic PNGs — never fatal)
# --------------------------------------------------------------------------- #
def _plot_probability_histogram(df_pred, threshold, output):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    p = df_pred["xgb_prob_touching"]
    ax.hist(p, bins=50, color="#4C78A8", edgecolor="white")
    ax.axvline(threshold, color="red", linestyle="--", label=f"threshold = {threshold:.4f}")
    ax.axvline(HIGH_CONF_PROB_MIN, color="#2ca02c", linestyle=":",
               label=f"high-conf >= {HIGH_CONF_PROB_MIN:.2f}")
    ax.axvspan(UNCERTAIN_PROB_MIN, UNCERTAIN_PROB_MAX, color="#ff7f0e", alpha=0.15,
               label=f"uncertain [{UNCERTAIN_PROB_MIN:.2f}, {UNCERTAIN_PROB_MAX:.2f}]")
    ax.set_xlabel("xgb_prob_touching")
    ax.set_ylabel("count")
    ax.set_title(f"Mars XGBoost prediction probabilities  (n={len(df_pred):,})")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def _plot_touching_pct_by_network(by_net, output):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    df = by_net.sort_values("network_id").reset_index(drop=True)
    sizes = np.clip(df["n_pairs"].to_numpy(), 5, 200)
    ax.scatter(df["network_id"], df["touching_fraction"] * 100.0, s=sizes,
               c=df["n_pairs"], cmap="viridis", edgecolors="white", linewidths=0.3, alpha=0.8)
    ax.set_xlabel("network_id")
    ax.set_ylabel("predicted touching (%)")
    ax.set_title(f"Predicted touching fraction per network (n_networks={len(df)})")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(output, dpi=130)
    plt.close(fig)


def render_figures(df_pred, by_net, threshold, figures_dir) -> None:
    """Write diagnostic PNGs. Best-effort: logs and returns on any failure."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    try:
        _plot_probability_histogram(df_pred, threshold, figures_dir / "probability_histogram.png")
        _plot_touching_pct_by_network(by_net, figures_dir / "predicted_touching_pct_by_network.png")
    except Exception as exc:  # diagnostics only
        log.warning("Figure generation failed: %s", exc)


# --------------------------------------------------------------------------- #
# High-level entry point
# --------------------------------------------------------------------------- #
def run_mars_tabular_inference(
    features_parquet=DEFAULT_FEATURES_PARQUET,
    model_path=paths.XGB_PRODUCTION,
    feature_columns_txt=DEFAULT_FEATURE_COLUMNS_TXT,
    threshold_txt=DEFAULT_THRESHOLD_TXT,
    output_dir=paths.MARS_MODEL_OUTPUTS_DIR,
    pairs_gpkg=paths.MARS_PAIRS_GPKG,
    *,
    write: bool = True,
    make_figures: bool = True,
) -> dict:
    """Phase 3B: production XGBoost on the Mars model-ready 5-feature table.

    Returns ``{"predictions", "summary", "by_network", "threshold", "paths"}``.
    """
    model_features = load_feature_columns(feature_columns_txt)
    threshold = load_threshold(threshold_txt)
    log.info("Production threshold: %.6f; features: %s", threshold, model_features)

    df = pd.read_parquet(features_parquet)
    log.info("Mars model-ready table: %d rows x %d cols", *df.shape)
    validate_feature_columns(df, model_features)

    model = load_xgb_model(model_path, expected_features=model_features)
    proba, pred = predict_with_threshold(model, df, model_features, threshold)
    log.info("Predicted %d touching / %d non-touching (%.1f%%)",
             int(pred.sum()), int((pred == 0).sum()), 100.0 * pred.mean())

    try:
        model_path_rel = str(Path(model_path).relative_to(paths.PROJECT_ROOT))
    except ValueError:
        model_path_rel = str(model_path)
    df_pred = assemble_predictions(df, proba, pred, threshold, model_features, model_path_rel)
    summary = summarize_predictions(df_pred, threshold, model_path_rel)
    by_net = summarize_by_network(df_pred)

    written: dict[str, Path] = {}
    if write:
        written = export_prediction_outputs(df_pred, summary, by_net, output_dir, pairs_gpkg)
        if make_figures:
            render_figures(df_pred, by_net, threshold, Path(output_dir) / "figures")

    return {
        "predictions": df_pred,
        "summary": summary,
        "by_network": by_net,
        "threshold": threshold,
        "paths": written,
    }


__all__ = [
    "run_mars_tabular_inference",
    "validate_feature_columns",
    "apply_operating_threshold",
    "assemble_predictions",
    "summarize_predictions",
    "summarize_by_network",
    "build_geometry_layers",
    "export_prediction_outputs",
    "MODEL_FEATURE_SET",
    "HIGH_CONF_PROB_MIN",
    "UNCERTAIN_PROB_MIN",
    "UNCERTAIN_PROB_MAX",
]
