"""Mars combined XGBoost inference (Phase 6C).

Runs the Earth-trained geometric+CNN XGBoost variants on the Phase-5 Mars
tabular-plus-CNN table, merges the Phase-3B tabular-only predictions, and
exports the comparison outputs. This is the package-resident replacement for
``scripts/run_mars_combined_xgb_inference.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from channel_heads.io import paths
from channel_heads.io.tables import write_table
from channel_heads.logging_config import get_logger
from channel_heads.models.device import pick_device
from channel_heads.models.xgboost import (
    load_feature_columns,
    load_threshold,
    load_xgb_model,
    predict_with_threshold,
    verify_feature_matrix,
    verify_model_feature_order,
)
from channel_heads.rasterizer import (
    BACKGROUND,
    BRANCH_A,
    BRANCH_B,
    CONFLUENCE_MARKER,
    OTHER_STREAMS,
)

log = get_logger("models.mars_combined")

DEFAULT_INPUT_PARQUET = (
    paths.MARS_MODEL_INPUTS_DIR / "mars_model_input_tabular_plus_cnn.parquet"
)
DEFAULT_TABULAR_PREDS_PARQUET = (
    paths.MARS_MODEL_OUTPUTS_DIR / "mars_xgb_predictions_5feat.parquet"
)
DEFAULT_PATCH_INDEX_PARQUET = paths.MARS_CNN_PATCH_INDEX
DEFAULT_PAIRS_GPKG = paths.MARS_PAIRS_GPKG
DEFAULT_OUTPUT_DIR = paths.MARS_MODEL_OUTPUTS_DIR
DEFAULT_FIGURES_DIR = DEFAULT_OUTPUT_DIR / "figures_combined"
DEFAULT_PRED_PARQUET = DEFAULT_OUTPUT_DIR / "mars_combined_model_predictions.parquet"
DEFAULT_PRED_GPKG = DEFAULT_OUTPUT_DIR / "mars_combined_model_predictions.gpkg"
DEFAULT_SUMMARY_CSV = DEFAULT_OUTPUT_DIR / "mars_model_comparison_summary.csv"
DEFAULT_BY_NETWORK_CSV = DEFAULT_OUTPUT_DIR / "mars_predictions_by_network_combined.csv"
DEFAULT_BATCH_SIZE = 64

HIGH_CONF_PROB_MIN = 0.80

CLASS_COLORS: dict[int, tuple[float, float, float]] = {
    BACKGROUND: (0.95, 0.95, 0.95),
    BRANCH_A: (0.85, 0.33, 0.10),
    BRANCH_B: (0.10, 0.45, 0.82),
    OTHER_STREAMS: (0.70, 0.70, 0.70),
    CONFLUENCE_MARKER: (0.90, 0.80, 0.00),
}


@dataclass(frozen=True)
class MarsModelVariant:
    """Artifact bundle for one combined Mars XGBoost variant."""

    name: str
    model_path: Path
    feature_columns_path: Path
    threshold_path: Path


def default_model_variants() -> tuple[MarsModelVariant, ...]:
    """Return the two Phase-6C combined model variants."""
    return (
        MarsModelVariant(
            name="emb",
            model_path=paths.MODELS_DIR / "xgb_geom_plus_cnn_emb.json",
            feature_columns_path=(
                paths.MODELS_DIR / "feature_columns_geom_plus_cnn_emb.txt"
            ),
            threshold_path=paths.MODELS_DIR / "optimal_threshold_geom_plus_cnn_emb.txt",
        ),
        MarsModelVariant(
            name="logit",
            model_path=paths.MODELS_DIR / "xgb_geom_plus_cnn_logit.json",
            feature_columns_path=(
                paths.MODELS_DIR / "feature_columns_geom_plus_cnn_logit.txt"
            ),
            threshold_path=paths.MODELS_DIR / "optimal_threshold_geom_plus_cnn_logit.txt",
        ),
    )


def _resolve_project_path(path: str | Path, project_root: Path = paths.PROJECT_ROOT) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (project_root / path).resolve()


def _relative_to_project(path: str | Path) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(paths.PROJECT_ROOT))
    except ValueError:
        return str(path)


def extract_logits(
    patch_paths: list[Path],
    *,
    model_path: str | Path = paths.CNN_PRODUCTION,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedding_dim: int | None = None,
) -> np.ndarray:
    """Forward pass through ``cnn_outlet_final.pt`` to get classifier logits."""
    import torch
    from torch.utils.data import DataLoader

    from channel_heads.cnn_model import (
        DEFAULT_EMBEDDING_DIM,
        OutletCNN,
        OutletPairDataset,
    )

    embedding_dim = embedding_dim or DEFAULT_EMBEDDING_DIM
    model = OutletCNN(embedding_dim=embedding_dim)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch: missing={missing} unexpected={unexpected}"
        )
    model.to(device)
    model.eval()

    dummy = np.zeros(len(patch_paths), dtype=np.float32)
    ds = OutletPairDataset(patch_paths, dummy, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    out: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            logit = model(images).squeeze(-1)
            out.append(logit.cpu().numpy())
    return np.concatenate(out)


def attach_cnn_logit(
    df: pd.DataFrame,
    *,
    patch_index_parquet: str | Path = DEFAULT_PATCH_INDEX_PARQUET,
    model_path: str | Path = paths.CNN_PRODUCTION,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    project_root: Path = paths.PROJECT_ROOT,
) -> pd.DataFrame:
    """Attach ``cnn_logit`` without recomputing Phase-5 embeddings."""
    if "cnn_logit" in df.columns:
        log.info("cnn_logit already present in input table; skipping extraction")
        return df

    log.info(
        "cnn_logit missing; running one eval-mode forward pass over existing "
        "Phase-4 patches"
    )
    patch_index = pd.read_parquet(patch_index_parquet)
    patch_index = patch_index[patch_index["patch_status"] == "ok"][
        ["pair_id", "patch_path"]
    ].copy()
    patch_index["patch_path_abs"] = [
        str(_resolve_project_path(p, project_root)) for p in patch_index["patch_path"]
    ]
    out = df.merge(patch_index, on="pair_id", how="left")

    has_patch = out["patch_path_abs"].notna()
    out["cnn_logit"] = np.nan
    if has_patch.any():
        logits = extract_logits(
            [Path(p) for p in out.loc[has_patch, "patch_path_abs"]],
            model_path=model_path,
            device=device,
            batch_size=batch_size,
        )
        if logits.shape[0] != int(has_patch.sum()):
            raise RuntimeError(
                f"Logit count {logits.shape[0]} != patched-row count "
                f"{int(has_patch.sum())}"
            )
        if not np.isfinite(logits).all():
            raise RuntimeError(
                f"Non-finite cnn_logit on patched rows: "
                f"nan={int(np.isnan(logits).sum())} "
                f"inf={int(np.isinf(logits).sum())}"
            )
        out.loc[has_patch, "cnn_logit"] = logits

    ok_logits = out.loc[has_patch, "cnn_logit"]
    log.info(
        "cnn_logit attached: ok=%d (mean=%.3f std=%.3f min=%.3f max=%.3f), "
        "withheld_no_patch=%d (NaN)",
        int(has_patch.sum()),
        float(ok_logits.mean()) if len(ok_logits) else float("nan"),
        float(ok_logits.std()) if len(ok_logits) else float("nan"),
        float(ok_logits.min()) if len(ok_logits) else float("nan"),
        float(ok_logits.max()) if len(ok_logits) else float("nan"),
        int((~has_patch).sum()),
    )
    return out.drop(columns=["patch_path", "patch_path_abs"], errors="ignore")


def verify_model_variant_inputs(
    df: pd.DataFrame,
    model,
    feature_cols: list[str],
    name: str,
) -> dict[str, int]:
    """Raise on model/input compatibility issues for one variant."""
    context = f"Variant {name}"
    stats = verify_feature_matrix(df, feature_cols, context=context)
    if hasattr(model, "get_booster"):
        verify_model_feature_order(model, feature_cols, context=context)
    log.info(
        "Variant %s: feature matrix (%d, %d), NaN=%d, inf=0",
        name,
        stats["n_rows"],
        stats["n_cols"],
        stats["n_nan_cells"],
    )
    return stats


def run_model_variant_inference(
    df: pd.DataFrame,
    *,
    name: str,
    model,
    feature_cols: list[str],
    threshold: float,
    model_path: str | Path = "",
) -> dict:
    """Run one combined model variant and return probabilities/decisions."""
    verify_model_variant_inputs(df, model, feature_cols, name)
    proba, pred = predict_with_threshold(model, df, feature_cols, threshold)
    log.info(
        "Variant %s: n=%d, threshold=%.6f, predicted touching=%d (%.1f%%)",
        name,
        len(df),
        threshold,
        int(pred.sum()),
        100 * pred.mean() if len(pred) else 0.0,
    )
    return {
        "name": name,
        "proba": proba,
        "pred": pred,
        "threshold": threshold,
        "feature_cols": feature_cols,
        "model_path": _relative_to_project(model_path) if model_path else "",
    }


def run_variant_from_artifacts(
    df: pd.DataFrame,
    variant: MarsModelVariant,
) -> dict:
    """Load one model variant's artifacts and run thresholded inference."""
    feature_cols = load_feature_columns(variant.feature_columns_path)
    threshold = load_threshold(variant.threshold_path)
    model = load_xgb_model(variant.model_path, expected_features=feature_cols)
    return run_model_variant_inference(
        df,
        name=variant.name,
        model=model,
        feature_cols=feature_cols,
        threshold=threshold,
        model_path=variant.model_path,
    )


def append_variant_predictions(
    df_out: pd.DataFrame,
    df_features: pd.DataFrame,
    result: dict,
) -> pd.DataFrame:
    """Append one variant's prediction block to the output dataframe."""
    name = result["name"]
    feature_cols = result["feature_cols"]
    out = df_out.copy()
    out[f"prob_touching_{name}"] = result["proba"]
    out[f"pred_touching_{name}"] = result["pred"]
    out[f"threshold_{name}"] = result["threshold"]
    out[f"model_path_{name}"] = result["model_path"]
    nan_per_row = df_features[feature_cols].isna().sum(axis=1).astype(int)
    out[f"inference_status_{name}"] = np.where(
        nan_per_row > 0, "ok_with_nan_features", "ok"
    )
    return out


def load_tabular_only_predictions(
    tabular_predictions_parquet: str | Path = DEFAULT_TABULAR_PREDS_PARQUET,
) -> pd.DataFrame:
    """Load Phase-3B tabular-only predictions in Phase-6C column names."""
    return pd.read_parquet(tabular_predictions_parquet)[
        ["pair_id", "xgb_prob_touching", "xgb_pred_touching", "xgb_decision_threshold"]
    ].rename(
        columns={
            "xgb_prob_touching": "prob_touching_tabular_only",
            "xgb_pred_touching": "pred_touching_tabular_only",
            "xgb_decision_threshold": "threshold_tabular_only",
        }
    )


def add_combined_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add agreement, disagreement and high-confidence columns."""
    out = df.copy()
    out["agreement_touching"] = (
        (out["pred_touching_emb"] == 1) & (out["pred_touching_logit"] == 1)
    ).astype(int)
    out["emb_logit_disagreement"] = (
        out["pred_touching_emb"] != out["pred_touching_logit"]
    ).astype(int)
    out["max_combined_prob"] = out[
        ["prob_touching_emb", "prob_touching_logit"]
    ].max(axis=1)
    out["high_confidence_any_combined"] = (
        out["max_combined_prob"] >= HIGH_CONF_PROB_MIN
    ).astype(int)
    return out


def build_comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Overall summary spanning tabular-only and combined predictions."""
    n_total = len(df)
    rows: list[dict] = []

    def prob_stats(series: pd.Series) -> dict:
        return {
            "prob_min": float(series.min()),
            "prob_p25": float(series.quantile(0.25)),
            "prob_median": float(series.median()),
            "prob_mean": float(series.mean()),
            "prob_p75": float(series.quantile(0.75)),
            "prob_max": float(series.max()),
        }

    for label, prob_col, pred_col in [
        ("tabular_only", "prob_touching_tabular_only", "pred_touching_tabular_only"),
        ("geom_plus_cnn_emb", "prob_touching_emb", "pred_touching_emb"),
        ("geom_plus_cnn_logit", "prob_touching_logit", "pred_touching_logit"),
    ]:
        n_touching = int((df[pred_col] == 1).sum())
        n_high = int((df[prob_col] >= HIGH_CONF_PROB_MIN).sum())
        rows.append(
            {
                "metric_kind": "per_model_counts",
                "model_variant": label,
                "n_pairs": n_total,
                "n_touching": n_touching,
                "pct_touching": n_touching / n_total if n_total else 0.0,
                "n_high_confidence_prob_ge_0.80": n_high,
                **prob_stats(df[prob_col]),
            }
        )

    def pct_eq(a, b) -> float:
        return float((a == b).mean())

    rows.extend(
        [
            {
                "metric_kind": "agreements",
                "model_variant": "agreement_tabular_vs_emb",
                "n_pairs": n_total,
                "agreement_fraction": pct_eq(
                    df["pred_touching_tabular_only"], df["pred_touching_emb"]
                ),
                "n_changes": int(
                    (
                        df["pred_touching_tabular_only"]
                        != df["pred_touching_emb"]
                    ).sum()
                ),
            },
            {
                "metric_kind": "agreements",
                "model_variant": "agreement_tabular_vs_logit",
                "n_pairs": n_total,
                "agreement_fraction": pct_eq(
                    df["pred_touching_tabular_only"], df["pred_touching_logit"]
                ),
                "n_changes": int(
                    (
                        df["pred_touching_tabular_only"]
                        != df["pred_touching_logit"]
                    ).sum()
                ),
            },
            {
                "metric_kind": "agreements",
                "model_variant": "agreement_emb_vs_logit",
                "n_pairs": n_total,
                "agreement_fraction": pct_eq(
                    df["pred_touching_emb"], df["pred_touching_logit"]
                ),
                "n_changes": int(
                    (df["pred_touching_emb"] != df["pred_touching_logit"]).sum()
                ),
            },
        ]
    )
    return pd.DataFrame(rows)


def summarize_combined_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """Public alias for the Phase-6C model comparison summary."""
    return build_comparison_summary(df)


def compare_mars_model_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Public alias for comparing Mars model-variant outputs."""
    return build_comparison_summary(df)


def build_by_network(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise Phase-6C outputs by Mars valley-network id."""
    grouped = df.groupby("network_id")
    out = pd.DataFrame(
        {
            "network_id": list(grouped.groups.keys()),
            "n_pairs_scored": grouped.size().to_numpy(),
            "pct_touching_tabular_only": (
                grouped["pred_touching_tabular_only"].mean().to_numpy()
            ),
            "pct_touching_geom_plus_cnn_emb": (
                grouped["pred_touching_emb"].mean().to_numpy()
            ),
            "pct_touching_geom_plus_cnn_logit": (
                grouped["pred_touching_logit"].mean().to_numpy()
            ),
            "mean_prob_tabular_only": (
                grouped["prob_touching_tabular_only"].mean().to_numpy()
            ),
            "mean_prob_geom_plus_cnn_emb": (
                grouped["prob_touching_emb"].mean().to_numpy()
            ),
            "mean_prob_geom_plus_cnn_logit": (
                grouped["prob_touching_logit"].mean().to_numpy()
            ),
        }
    )
    disagreement_counts = (
        df.assign(
            _combined_disagreement=(
                df["pred_touching_emb"] != df["pred_touching_logit"]
            ).astype(int)
        )
        .groupby("network_id")["_combined_disagreement"]
        .sum()
    )
    out["n_disagreements_between_combined_models"] = out["network_id"].map(
        disagreement_counts
    ).to_numpy()
    return out.sort_values("network_id").reset_index(drop=True)


def export_combined_geopackage(
    df_out: pd.DataFrame,
    *,
    pairs_gpkg: str | Path = DEFAULT_PAIRS_GPKG,
    output_gpkg: str | Path = DEFAULT_PRED_GPKG,
) -> Path:
    """Write the Phase-6C GeoPackage layers."""
    import geopandas as gpd

    output_gpkg = Path(output_gpkg)
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    pairs_gdf = gpd.read_file(pairs_gpkg, layer="mars_pairs")
    paths_gdf = gpd.read_file(pairs_gpkg, layer="mars_pair_paths")

    cols_for_gpkg = [
        "pair_id",
        "network_id",
        "head_node_id_1",
        "head_node_id_2",
        "confluence_node_id",
        "outlet_node_id",
        "headhead_dist_m",
        "path_length_1_m",
        "path_length_2_m",
        "prob_touching_tabular_only",
        "pred_touching_tabular_only",
        "threshold_tabular_only",
        "prob_touching_emb",
        "pred_touching_emb",
        "threshold_emb",
        "model_path_emb",
        "inference_status_emb",
        "prob_touching_logit",
        "pred_touching_logit",
        "threshold_logit",
        "model_path_logit",
        "inference_status_logit",
        "agreement_touching",
        "emb_logit_disagreement",
        "max_combined_prob",
        "high_confidence_any_combined",
    ]
    df_for_join = df_out[cols_for_gpkg].copy()
    pairs_with_pred = pairs_gdf.merge(
        df_for_join, on=["pair_id", "network_id"], how="inner"
    )
    pairs_with_pred = gpd.GeoDataFrame(
        pairs_with_pred, geometry="geometry", crs=pairs_gdf.crs
    )
    paths_with_pred = paths_gdf.merge(
        df_for_join, on=["pair_id", "network_id"], how="inner"
    )
    paths_with_pred = gpd.GeoDataFrame(
        paths_with_pred, geometry="geometry", crs=paths_gdf.crs
    )

    if output_gpkg.exists():
        output_gpkg.unlink()
    pairs_with_pred.to_file(
        output_gpkg, layer="mars_combined_predicted_pairs", driver="GPKG"
    )
    paths_with_pred.to_file(
        output_gpkg, layer="mars_combined_predicted_pair_paths", driver="GPKG"
    )
    pairs_with_pred[pairs_with_pred["pred_touching_emb"] == 1].to_file(
        output_gpkg, layer="mars_combined_emb_touching_pairs", driver="GPKG"
    )
    pairs_with_pred[pairs_with_pred["pred_touching_logit"] == 1].to_file(
        output_gpkg, layer="mars_combined_logit_touching_pairs", driver="GPKG"
    )
    pairs_with_pred[pairs_with_pred["agreement_touching"] == 1].to_file(
        output_gpkg, layer="mars_combined_agreement_touching_pairs", driver="GPKG"
    )

    disagreement = pairs_with_pred[pairs_with_pred["emb_logit_disagreement"] == 1]
    if not disagreement.empty:
        disagreement.to_file(
            output_gpkg, layer="mars_combined_disagreement_pairs", driver="GPKG"
        )
    else:
        log.warning("No disagreement rows; skipping disagreement layer")

    high_conf = pairs_with_pred[
        pairs_with_pred["high_confidence_any_combined"] == 1
    ]
    if not high_conf.empty:
        high_conf.to_file(
            output_gpkg, layer="mars_combined_high_confidence_pairs", driver="GPKG"
        )
    else:
        log.warning("No high-confidence rows; skipping high-confidence layer")
    return output_gpkg


def export_combined_prediction_outputs(
    predictions: pd.DataFrame,
    summary: pd.DataFrame,
    by_network: pd.DataFrame,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pairs_gpkg: str | Path = DEFAULT_PAIRS_GPKG,
    write_gpkg: bool = True,
) -> dict[str, Path]:
    """Write Phase-6C tables and optional GeoPackage; return paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "predictions": write_table(
            predictions, output_dir / "mars_combined_model_predictions.parquet"
        ),
        "summary": output_dir / "mars_model_comparison_summary.csv",
        "by_network": output_dir / "mars_predictions_by_network_combined.csv",
    }
    summary.to_csv(written["summary"], index=False)
    by_network.to_csv(written["by_network"], index=False)
    if write_gpkg:
        written["gpkg"] = export_combined_geopackage(
            predictions,
            pairs_gpkg=pairs_gpkg,
            output_gpkg=output_dir / "mars_combined_model_predictions.gpkg",
        )
    return written


def plot_probability_histograms(
    df: pd.DataFrame, thresholds: dict[str, float], output: Path
) -> None:
    """Write probability histograms for the three Mars model variants."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    panels = [
        ("tabular_only", "prob_touching_tabular_only", thresholds["tabular"]),
        ("geom_plus_cnn_emb", "prob_touching_emb", thresholds["emb"]),
        ("geom_plus_cnn_logit", "prob_touching_logit", thresholds["logit"]),
    ]
    for ax, (label, col, threshold) in zip(axes, panels):
        ax.hist(df[col], bins=50, color="#4C78A8", edgecolor="white")
        ax.axvline(threshold, color="red", linestyle="--", label=f"thr={threshold:.4f}")
        ax.axvline(
            HIGH_CONF_PROB_MIN,
            color="#2ca02c",
            linestyle=":",
            label=f"hc>={HIGH_CONF_PROB_MIN:.2f}",
        )
        n_t = int((df[col] >= threshold).sum())
        ax.set_xlabel(col)
        ax.set_title(f"{label}\nn_touching={n_t}/{len(df)} ({100 * n_t / len(df):.1f}%)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("Mars prediction probability distributions", fontsize=13, y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_prob_scatter(
    df: pd.DataFrame,
    *,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
    thresholds: tuple[float, float] | None = None,
) -> None:
    """Write a paired probability scatter plot."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(df[xcol], df[ycol], s=8, alpha=0.5, edgecolors="none", color="#4C78A8")
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.5, label="y=x")
    if thresholds is not None:
        ax.axvline(
            thresholds[0], color="red", linestyle="--", alpha=0.6,
            label=f"x thr={thresholds[0]:.3f}"
        )
        ax.axhline(
            thresholds[1], color="orange", linestyle="--", alpha=0.6,
            label=f"y thr={thresholds[1]:.3f}"
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, linestyle=":")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def plot_touching_bar(summary_df: pd.DataFrame, output: Path) -> None:
    """Write model-variant touching-fraction bar plot."""
    import matplotlib.pyplot as plt

    counts = summary_df[summary_df["metric_kind"] == "per_model_counts"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    bars = ax.bar(
        counts["model_variant"],
        100 * counts["pct_touching"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    for bar, n_touching in zip(bars, counts["n_touching"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(n_touching)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("predicted touching (%)")
    ax.set_title("Mars predicted touching fraction by model variant")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3, axis="y", linestyle=":")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def patch_to_rgb(raster: np.ndarray) -> np.ndarray:
    """Convert a 5-class patch raster to RGB for contact sheets."""
    h, w = raster.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for cls_val, color in CLASS_COLORS.items():
        mask = raster == cls_val
        for channel in range(3):
            rgb[:, :, channel][mask] = color[channel]
    return rgb


def plot_patch_contact_sheet(
    sub: pd.DataFrame,
    patch_paths: dict[str, str],
    *,
    title: str,
    output: Path,
    subtitle_fn,
    project_root: Path = paths.PROJECT_ROOT,
) -> None:
    """Write a contact sheet of selected Mars CNN patches."""
    import matplotlib.pyplot as plt

    n = len(sub)
    if n == 0:
        log.warning("No patches to render for %s; skipping", output.name)
        return
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    axes_flat = np.atleast_1d(axes).ravel()
    for ax, (_, row) in zip(axes_flat, sub.iterrows()):
        rel = patch_paths.get(str(row["pair_id"]))
        if rel is None:
            ax.set_visible(False)
            continue
        patch = np.load(_resolve_project_path(rel, project_root))
        ax.imshow(patch_to_rgb(patch), interpolation="nearest")
        ax.set_title(subtitle_fn(row), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes_flat[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=12, y=1.0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120, bbox_inches="tight")
    plt.close(fig)


def render_combined_figures(
    df_out: pd.DataFrame,
    summary_df: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    patch_index_parquet: str | Path = DEFAULT_PATCH_INDEX_PARQUET,
) -> dict[str, Path]:
    """Write Phase-6C diagnostic figures, best-effort."""
    figures_dir = Path(figures_dir)
    try:
        outputs = {
            "probability_histograms": figures_dir / "probability_histograms.png",
            "scatter_tabular_vs_emb": figures_dir / "scatter_tabular_vs_emb.png",
            "scatter_emb_vs_logit": figures_dir / "scatter_emb_vs_logit.png",
            "touching_pct_by_variant": figures_dir / "touching_pct_by_variant.png",
        }
        plot_probability_histograms(df_out, thresholds, outputs["probability_histograms"])
        plot_prob_scatter(
            df_out,
            xcol="prob_touching_tabular_only",
            ycol="prob_touching_emb",
            xlabel="tabular-only prob_touching",
            ylabel="geom_plus_cnn_emb prob_touching",
            title="Tabular vs. combined-embedding probability",
            output=outputs["scatter_tabular_vs_emb"],
            thresholds=(thresholds["tabular"], thresholds["emb"]),
        )
        plot_prob_scatter(
            df_out,
            xcol="prob_touching_emb",
            ycol="prob_touching_logit",
            xlabel="geom_plus_cnn_emb prob_touching",
            ylabel="geom_plus_cnn_logit prob_touching",
            title="Combined-embedding vs. combined-logit probability",
            output=outputs["scatter_emb_vs_logit"],
            thresholds=(thresholds["emb"], thresholds["logit"]),
        )
        plot_touching_bar(summary_df, outputs["touching_pct_by_variant"])

        patch_index = pd.read_parquet(patch_index_parquet)
        patch_paths = {
            str(row["pair_id"]): str(row["patch_path"])
            for _, row in patch_index.iterrows()
        }

        hc_both = df_out[
            (df_out["prob_touching_emb"] >= HIGH_CONF_PROB_MIN)
            & (df_out["prob_touching_logit"] >= HIGH_CONF_PROB_MIN)
        ].copy()
        hc_both["min_combined_prob"] = hc_both[
            ["prob_touching_emb", "prob_touching_logit"]
        ].min(axis=1)
        hc_both = hc_both.sort_values("min_combined_prob", ascending=False).head(20)
        outputs["contact_sheet_high_conf_both"] = (
            figures_dir / "contact_sheet_high_conf_both.png"
        )
        plot_patch_contact_sheet(
            hc_both,
            patch_paths,
            title="20 pairs - both combined models high-confidence touching (prob >= 0.80)",
            output=outputs["contact_sheet_high_conf_both"],
            subtitle_fn=lambda row: (
                f"net={row['network_id']}\n"
                f"emb={row['prob_touching_emb']:.3f}  "
                f"logit={row['prob_touching_logit']:.3f}"
            ),
        )

        disagreement = df_out[df_out["emb_logit_disagreement"] == 1].copy()
        if not disagreement.empty:
            disagreement["abs_dprob"] = (
                disagreement["prob_touching_emb"]
                - disagreement["prob_touching_logit"]
            ).abs()
            disagreement = disagreement.sort_values("abs_dprob", ascending=False).head(20)
            outputs["contact_sheet_disagreement"] = (
                figures_dir / "contact_sheet_disagreement.png"
            )
            plot_patch_contact_sheet(
                disagreement,
                patch_paths,
                title="20 pairs - combined-emb vs combined-logit disagreement",
                output=outputs["contact_sheet_disagreement"],
                subtitle_fn=lambda row: (
                    f"net={row['network_id']}\n"
                    f"emb={row['prob_touching_emb']:.3f}  "
                    f"logit={row['prob_touching_logit']:.3f}\n"
                    f"emb_pred={int(row['pred_touching_emb'])}  "
                    f"logit_pred={int(row['pred_touching_logit'])}"
                ),
            )
        return outputs
    except Exception as exc:  # diagnostics only
        log.warning("Combined figure generation failed: %s", exc)
        return {}


def run_mars_combined_inference(
    input_parquet: str | Path = DEFAULT_INPUT_PARQUET,
    tabular_predictions_parquet: str | Path = DEFAULT_TABULAR_PREDS_PARQUET,
    patch_index_parquet: str | Path = DEFAULT_PATCH_INDEX_PARQUET,
    cnn_model_path: str | Path = paths.CNN_PRODUCTION,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    pairs_gpkg: str | Path = DEFAULT_PAIRS_GPKG,
    *,
    variants: tuple[MarsModelVariant, ...] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
    write: bool = True,
    write_gpkg: bool = True,
    make_figures: bool = True,
) -> dict:
    """Phase 6C: run combined Mars model variants and comparison outputs.

    Returns ``{"predictions", "summary", "by_network", "thresholds",
    "model_paths", "paths"}``.
    """
    device = device or pick_device()
    log.info("Device: %s", device)
    variants = variants or default_model_variants()

    df_in = pd.read_parquet(input_parquet)
    log.info("Mars combined input: %d rows x %d cols", *df_in.shape)
    df_in = attach_cnn_logit(
        df_in,
        patch_index_parquet=patch_index_parquet,
        model_path=cnn_model_path,
        device=device,
        batch_size=batch_size,
    )

    df_out = df_in.copy()
    thresholds: dict[str, float] = {}
    model_paths: dict[str, str] = {}
    for variant in variants:
        result = run_variant_from_artifacts(df_in, variant)
        df_out = append_variant_predictions(df_out, df_in, result)
        thresholds[result["name"]] = float(result["threshold"])
        model_paths[result["name"]] = str(result["model_path"])

    tabular = load_tabular_only_predictions(tabular_predictions_parquet)
    df_out = df_out.merge(tabular, on="pair_id", how="left")
    thresholds["tabular"] = float(df_out["threshold_tabular_only"].iloc[0])
    df_out = add_combined_derived_columns(df_out)

    summary = build_comparison_summary(df_out)
    by_network = build_by_network(df_out)

    written: dict[str, Path] = {}
    if write:
        written = export_combined_prediction_outputs(
            df_out,
            summary,
            by_network,
            output_dir=output_dir,
            pairs_gpkg=pairs_gpkg,
            write_gpkg=write_gpkg,
        )
        if make_figures:
            written.update(
                render_combined_figures(
                    df_out,
                    summary,
                    thresholds,
                    figures_dir=Path(output_dir) / "figures_combined",
                    patch_index_parquet=patch_index_parquet,
                )
            )

    return {
        "predictions": df_out,
        "summary": summary,
        "by_network": by_network,
        "thresholds": thresholds,
        "model_paths": model_paths,
        "paths": written,
    }


__all__ = [
    "HIGH_CONF_PROB_MIN",
    "MarsModelVariant",
    "add_combined_derived_columns",
    "append_variant_predictions",
    "attach_cnn_logit",
    "build_by_network",
    "build_comparison_summary",
    "compare_mars_model_variants",
    "default_model_variants",
    "export_combined_geopackage",
    "export_combined_prediction_outputs",
    "extract_logits",
    "load_tabular_only_predictions",
    "plot_patch_contact_sheet",
    "render_combined_figures",
    "run_mars_combined_inference",
    "run_model_variant_inference",
    "run_variant_from_artifacts",
    "summarize_combined_predictions",
    "verify_model_variant_inputs",
]
