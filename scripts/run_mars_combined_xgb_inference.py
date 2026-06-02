#!/usr/bin/env python
"""Phase 6C — Run Mars inference with the two combined XGBoost variants
trained in Phase 6B (Earth-trained, threshold-tuned on the Earth PR curve).

Variants:
  - "emb"   ← models/xgb_geom_plus_cnn_emb.json    (9 features = 5 geom + 4 CNN emb)
  - "logit" ← models/xgb_geom_plus_cnn_logit.json  (6 features = 5 geom + 1 cnn_logit)

Each variant loads:
  - models/feature_columns_geom_plus_cnn_{variant}.txt
  - models/optimal_threshold_geom_plus_cnn_{variant}.txt

The Phase 3B tabular-only predictions are loaded for comparison.

Gap from Phase 5
----------------
Phase 5 wrote ``emb_0..emb_3`` to ``mars_model_input_tabular_plus_cnn.parquet``
but not ``cnn_logit``. The "logit" combined model needs it. To stay
faithful to the user constraint *do not re-extract embeddings*, this
script does NOT touch any existing Phase 5 / Phase 4 artifact: it only
runs a one-time CNN forward pass (eval mode, no augmentation) over the
already-rendered patches to obtain logits, used in memory for inference,
and persists them as a new ``cnn_logit`` column in the *Phase 6C output
parquet only*. Embeddings (emb_0..emb_3) are taken verbatim from the
Phase 5 file — no recomputation. A note is included in the summary
markdown.

Outputs:
  data/Mars/model_outputs/mars_combined_model_predictions.parquet
  data/Mars/model_outputs/mars_combined_model_predictions.csv
  data/Mars/model_outputs/mars_combined_model_predictions.gpkg     (7 layers)
  data/Mars/model_outputs/mars_model_comparison_summary.csv
  data/Mars/model_outputs/mars_predictions_by_network_combined.csv
  data/Mars/model_outputs/figures_combined/*.png
  data/Mars/model_outputs/phase_6c_combined_mars_inference_summary.md  (separate write)

This phase does NOT retrain anything, regenerate any patch, or re-extract
embeddings. It is inference + a single transient logit forward pass.
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from shapely.geometry import LineString, MultiLineString
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from channel_heads.cnn_model import (  # noqa: E402
    DEFAULT_EMBEDDING_DIM,
    OutletCNN,
    OutletPairDataset,
)
from channel_heads.inference import (  # noqa: E402
    load_feature_columns,
    load_threshold,
    load_xgb_model,
    pick_device,
    predict_with_threshold,
    verify_feature_matrix,
    verify_model_feature_order,
)
from channel_heads.rasterizer import (  # noqa: E402
    BACKGROUND,
    BRANCH_A,
    BRANCH_B,
    CONFLUENCE_MARKER,
    OTHER_STREAMS,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PARQUET = (
    PROJECT_ROOT
    / "data/Mars/model_inputs/mars_model_input_tabular_plus_cnn.parquet"
)
TABULAR_PREDS_PARQUET = (
    PROJECT_ROOT
    / "data/Mars/model_outputs/mars_xgb_predictions_5feat.parquet"
)
PATCH_INDEX_PARQUET = (
    PROJECT_ROOT / "data/Mars/model_inputs/mars_cnn_patch_index.parquet"
)
PAIRS_GPKG = PROJECT_ROOT / "data/Mars/topology/mars_vn_pairs.gpkg"

CNN_MODEL_PATH = PROJECT_ROOT / "models/cnn_outlet_final.pt"

OUTPUT_DIR = PROJECT_ROOT / "data/Mars/model_outputs"
FIGURES_DIR = OUTPUT_DIR / "figures_combined"
PRED_PARQUET = OUTPUT_DIR / "mars_combined_model_predictions.parquet"
PRED_CSV = OUTPUT_DIR / "mars_combined_model_predictions.csv"
PRED_GPKG = OUTPUT_DIR / "mars_combined_model_predictions.gpkg"
SUMMARY_CSV = OUTPUT_DIR / "mars_model_comparison_summary.csv"
BY_NETWORK_CSV = OUTPUT_DIR / "mars_predictions_by_network_combined.csv"

BATCH_SIZE = 64

# Model variants — each entry binds together its three artifacts.
VARIANTS: list[dict] = [
    {
        "name": "emb",
        "model_path": PROJECT_ROOT / "models/xgb_geom_plus_cnn_emb.json",
        "feature_columns_path": PROJECT_ROOT
        / "models/feature_columns_geom_plus_cnn_emb.txt",
        "threshold_path": PROJECT_ROOT
        / "models/optimal_threshold_geom_plus_cnn_emb.txt",
    },
    {
        "name": "logit",
        "model_path": PROJECT_ROOT / "models/xgb_geom_plus_cnn_logit.json",
        "feature_columns_path": PROJECT_ROOT
        / "models/feature_columns_geom_plus_cnn_logit.txt",
        "threshold_path": PROJECT_ROOT
        / "models/optimal_threshold_geom_plus_cnn_logit.txt",
    },
]

HIGH_CONF_PROB_MIN = 0.80

# Class colour map for contact sheets (mirrors notebook 04 cell 7)
CLASS_COLORS: dict[int, tuple[float, float, float]] = {
    BACKGROUND: (0.95, 0.95, 0.95),
    BRANCH_A: (0.85, 0.33, 0.10),
    BRANCH_B: (0.10, 0.45, 0.82),
    OTHER_STREAMS: (0.70, 0.70, 0.70),
    CONFLUENCE_MARKER: (0.90, 0.80, 0.00),
}

log = logging.getLogger("phase6c")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# Artifact loading (load_feature_columns, load_threshold), model loading,
# feature-matrix verification, the predict-with-threshold step and device
# selection are shared with the other Mars inference scripts via
# ``channel_heads.inference``.


# ---------------------------------------------------------------------------
# One-time logit extraction (gap-fill from Phase 5)
# ---------------------------------------------------------------------------
def extract_logits(
    patch_paths: list[Path], device: str, batch_size: int
) -> np.ndarray:
    """Forward pass through cnn_outlet_final.pt to get classifier logits.

    Embeddings already exist; this only fills the cnn_logit gap. Same
    eval-mode, no-augment pipeline as Phase 5's embedding extraction.
    """
    model = OutletCNN(embedding_dim=DEFAULT_EMBEDDING_DIM)
    state = torch.load(CNN_MODEL_PATH, map_location="cpu", weights_only=True)
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


def attach_cnn_logit(df: pd.DataFrame, device: str) -> pd.DataFrame:
    """Add a `cnn_logit` column by running the CNN over the Phase-4
    patches. Phase 5's embeddings are NOT recomputed.
    """
    if "cnn_logit" in df.columns:
        log.info("cnn_logit already present in input table — skipping extraction")
        return df

    log.info(
        "cnn_logit missing from input — running ONE-TIME forward pass over "
        "the existing Phase-4 patches (no re-extraction of embeddings, no "
        "patch regeneration)"
    )
    idx = pd.read_parquet(PATCH_INDEX_PARQUET)
    idx = idx[idx["patch_status"] == "ok"][["pair_id", "patch_path"]]
    idx["patch_path_abs"] = idx["patch_path"].apply(
        lambda p: str((PROJECT_ROOT / p).resolve())
    )
    df = df.merge(idx, on="pair_id", how="left")

    # Pairs whose patch failed structural QA (raster_status != "ok") have no
    # usable patch and are withheld by the direct-rasterization pipeline. Their
    # emb_* columns are already NaN in the input table; mirror that for
    # cnn_logit (NaN) rather than aborting. XGBoost handles NaN features, so the
    # geom_plus_cnn_logit variant still predicts for these rows using geometry.
    has_patch = df["patch_path_abs"].notna()
    n_missing = int((~has_patch).sum())

    df["cnn_logit"] = np.nan
    if has_patch.any():
        logits = extract_logits(
            [Path(p) for p in df.loc[has_patch, "patch_path_abs"]],
            device=device,
            batch_size=BATCH_SIZE,
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
        df.loc[has_patch, "cnn_logit"] = logits

    df = df.drop(columns=["patch_path", "patch_path_abs"], errors="ignore")
    ok_logits = df.loc[has_patch, "cnn_logit"]
    log.info(
        "cnn_logit attached: ok=%d (mean=%.3f std=%.3f min=%.3f max=%.3f), "
        "withheld_no_patch=%d (NaN)",
        int(has_patch.sum()),
        float(ok_logits.mean()) if len(ok_logits) else float("nan"),
        float(ok_logits.std()) if len(ok_logits) else float("nan"),
        float(ok_logits.min()) if len(ok_logits) else float("nan"),
        float(ok_logits.max()) if len(ok_logits) else float("nan"),
        n_missing,
    )
    return df


# ---------------------------------------------------------------------------
# Per-variant inference
# ---------------------------------------------------------------------------
def verify_variant_inputs(
    df: pd.DataFrame, model: XGBClassifier, feature_cols: list[str], name: str
) -> None:
    """Raise on any compatibility issue for one variant."""
    context = f"Variant {name}"
    stats = verify_feature_matrix(df, feature_cols, context=context)
    verify_model_feature_order(model, feature_cols, context=context)
    log.info(
        "Variant %s: feature matrix (%d, %d), NaN=%d, inf=0 (XGBoost handles NaN)",
        name,
        stats["n_rows"],
        stats["n_cols"],
        stats["n_nan_cells"],
    )


def run_variant(
    df: pd.DataFrame, variant: dict
) -> tuple[np.ndarray, np.ndarray, float, list[str], str]:
    name = variant["name"]
    feature_cols = load_feature_columns(variant["feature_columns_path"])
    threshold = load_threshold(variant["threshold_path"])

    model = load_xgb_model(variant["model_path"])
    verify_variant_inputs(df, model, feature_cols, name)

    proba, pred = predict_with_threshold(df=df, model=model, feature_cols=feature_cols, threshold=threshold)
    log.info(
        "Variant %s: n=%d, threshold=%.6f, predicted touching = %d (%.1f%%)",
        name,
        len(df),
        threshold,
        int(pred.sum()),
        100 * pred.mean(),
    )
    model_rel = str(variant["model_path"].relative_to(PROJECT_ROOT))
    return proba, pred, threshold, feature_cols, model_rel


# ---------------------------------------------------------------------------
# Comparison + per-network summaries
# ---------------------------------------------------------------------------
def build_comparison_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Overall summary spanning the three predictions."""
    n_total = len(df)
    rows: list[dict] = []

    def prob_stats(s: pd.Series) -> dict:
        return {
            "prob_min": float(s.min()),
            "prob_p25": float(s.quantile(0.25)),
            "prob_median": float(s.median()),
            "prob_mean": float(s.mean()),
            "prob_p75": float(s.quantile(0.75)),
            "prob_max": float(s.max()),
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

    # Agreement metrics
    def pct_eq(a, b) -> float:
        return float((a == b).mean())

    agree_tab_emb = pct_eq(
        df["pred_touching_tabular_only"], df["pred_touching_emb"]
    )
    agree_tab_logit = pct_eq(
        df["pred_touching_tabular_only"], df["pred_touching_logit"]
    )
    agree_emb_logit = pct_eq(
        df["pred_touching_emb"], df["pred_touching_logit"]
    )
    change_tab_to_emb = int(
        (df["pred_touching_tabular_only"] != df["pred_touching_emb"]).sum()
    )
    change_tab_to_logit = int(
        (df["pred_touching_tabular_only"] != df["pred_touching_logit"]).sum()
    )
    rows.append(
        {
            "metric_kind": "agreements",
            "model_variant": "agreement_tabular_vs_emb",
            "n_pairs": n_total,
            "agreement_fraction": agree_tab_emb,
            "n_changes": change_tab_to_emb,
        }
    )
    rows.append(
        {
            "metric_kind": "agreements",
            "model_variant": "agreement_tabular_vs_logit",
            "n_pairs": n_total,
            "agreement_fraction": agree_tab_logit,
            "n_changes": change_tab_to_logit,
        }
    )
    rows.append(
        {
            "metric_kind": "agreements",
            "model_variant": "agreement_emb_vs_logit",
            "n_pairs": n_total,
            "agreement_fraction": agree_emb_logit,
            "n_changes": int(
                (df["pred_touching_emb"] != df["pred_touching_logit"]).sum()
            ),
        }
    )
    return pd.DataFrame(rows)


def build_by_network(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("network_id")
    out = pd.DataFrame(
        {
            "network_id": list(g.groups.keys()),
            "n_pairs_scored": g.size().to_numpy(),
            "pct_touching_tabular_only": g["pred_touching_tabular_only"].mean().to_numpy(),
            "pct_touching_geom_plus_cnn_emb": g["pred_touching_emb"].mean().to_numpy(),
            "pct_touching_geom_plus_cnn_logit": g["pred_touching_logit"].mean().to_numpy(),
            "mean_prob_tabular_only": g["prob_touching_tabular_only"].mean().to_numpy(),
            "mean_prob_geom_plus_cnn_emb": g["prob_touching_emb"].mean().to_numpy(),
            "mean_prob_geom_plus_cnn_logit": g["prob_touching_logit"].mean().to_numpy(),
        }
    )
    out["n_disagreements_between_combined_models"] = g.apply(
        lambda s: int((s["pred_touching_emb"] != s["pred_touching_logit"]).sum())
    ).to_numpy()
    return out.sort_values("network_id").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_probability_histograms(df: pd.DataFrame, thresholds: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    panels = [
        ("tabular_only", "prob_touching_tabular_only", thresholds["tabular"]),
        ("geom_plus_cnn_emb", "prob_touching_emb", thresholds["emb"]),
        ("geom_plus_cnn_logit", "prob_touching_logit", thresholds["logit"]),
    ]
    for ax, (label, col, thr) in zip(axes, panels):
        ax.hist(df[col], bins=50, color="#4C78A8", edgecolor="white")
        ax.axvline(thr, color="red", linestyle="--", label=f"thr={thr:.4f}")
        ax.axvline(HIGH_CONF_PROB_MIN, color="#2ca02c", linestyle=":",
                   label=f"hc≥{HIGH_CONF_PROB_MIN:.2f}")
        n_t = int((df[col] >= thr).sum())
        ax.set_xlabel(col)
        ax.set_title(f"{label}\nn_touching={n_t}/{len(df)} ({100*n_t/len(df):.1f}%)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("count")
    fig.suptitle("Mars prediction probability distributions", fontsize=13, y=1.02)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_prob_scatter(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output: Path,
    thresholds: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(df[xcol], df[ycol], s=8, alpha=0.5, edgecolors="none", color="#4C78A8")
    ax.plot([0, 1], [0, 1], "k:", linewidth=0.8, alpha=0.5, label="y=x")
    if thresholds is not None:
        ax.axvline(thresholds[0], color="red", linestyle="--", alpha=0.6,
                   label=f"x thr={thresholds[0]:.3f}")
        ax.axhline(thresholds[1], color="orange", linestyle="--", alpha=0.6,
                   label=f"y thr={thresholds[1]:.3f}")
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
    for bar, n in zip(bars, counts["n_touching"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(n)}",
            ha="center", va="bottom", fontsize=10,
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
    h, w = raster.shape
    rgb = np.zeros((h, w, 3), dtype=float)
    for cls_val, color in CLASS_COLORS.items():
        mask = raster == cls_val
        for ch in range(3):
            rgb[:, :, ch][mask] = color[ch]
    return rgb


def plot_patch_contact_sheet(
    sub: pd.DataFrame,
    patch_paths: dict[str, str],
    title: str,
    output: Path,
    subtitle_fn,
) -> None:
    n = len(sub)
    if n == 0:
        log.warning("No patches to render for %s — skipping", output.name)
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
        patch = np.load(PROJECT_ROOT / rel)
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    setup_logging()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    log.info("Device: %s", device)

    # --- Load inputs --------------------------------------------------
    log.info("Loading Mars combined input: %s", INPUT_PARQUET)
    df_in = pd.read_parquet(INPUT_PARQUET)
    log.info("Mars combined input: %d rows × %d cols", *df_in.shape)

    # Gap-fill cnn_logit (Phase 5 omitted it)
    df_in = attach_cnn_logit(df_in, device=device)

    # --- Load tabular-only predictions for comparison ------------------
    log.info("Loading tabular-only predictions: %s", TABULAR_PREDS_PARQUET)
    df_tab = pd.read_parquet(TABULAR_PREDS_PARQUET)[
        ["pair_id", "xgb_prob_touching", "xgb_pred_touching", "xgb_decision_threshold"]
    ].rename(
        columns={
            "xgb_prob_touching": "prob_touching_tabular_only",
            "xgb_pred_touching": "pred_touching_tabular_only",
            "xgb_decision_threshold": "threshold_tabular_only",
        }
    )

    # --- Run each combined variant ------------------------------------
    df_out = df_in.copy()
    thresholds: dict[str, float] = {}
    model_paths: dict[str, str] = {}
    for variant in VARIANTS:
        proba, pred, thr, feats, model_rel = run_variant(df_in, variant)
        name = variant["name"]
        df_out[f"prob_touching_{name}"] = proba
        df_out[f"pred_touching_{name}"] = pred
        df_out[f"threshold_{name}"] = thr
        df_out[f"model_path_{name}"] = model_rel
        # Inference status mirrors Phase 3B convention
        nan_per_row = df_in[feats].isna().sum(axis=1).astype(int)
        df_out[f"inference_status_{name}"] = np.where(
            nan_per_row > 0, "ok_with_nan_features", "ok"
        )
        thresholds[name] = thr
        model_paths[name] = model_rel

    # Merge tabular-only predictions
    df_out = df_out.merge(df_tab, on="pair_id", how="left")
    thresholds["tabular"] = float(df_out["threshold_tabular_only"].iloc[0])

    # Derived columns
    df_out["agreement_touching"] = (
        (df_out["pred_touching_emb"] == 1) & (df_out["pred_touching_logit"] == 1)
    ).astype(int)
    df_out["emb_logit_disagreement"] = (
        df_out["pred_touching_emb"] != df_out["pred_touching_logit"]
    ).astype(int)
    df_out["max_combined_prob"] = df_out[
        ["prob_touching_emb", "prob_touching_logit"]
    ].max(axis=1)
    df_out["high_confidence_any_combined"] = (
        df_out["max_combined_prob"] >= HIGH_CONF_PROB_MIN
    ).astype(int)

    # --- Persist parquet + csv ----------------------------------------
    df_out.to_parquet(PRED_PARQUET, index=False)
    df_out.to_csv(PRED_CSV, index=False)
    log.info("Wrote: %s", PRED_PARQUET)
    log.info("Wrote: %s", PRED_CSV)

    # --- GeoPackage ---------------------------------------------------
    log.info("Building GPKG: %s", PRED_GPKG)
    pairs_gdf = gpd.read_file(PAIRS_GPKG, layer="mars_pairs")
    paths_gdf = gpd.read_file(PAIRS_GPKG, layer="mars_pair_paths")

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
    pairs_with_pred = pairs_gdf.merge(df_for_join, on=["pair_id", "network_id"], how="inner")
    pairs_with_pred = gpd.GeoDataFrame(pairs_with_pred, geometry="geometry", crs=pairs_gdf.crs)
    paths_with_pred = paths_gdf.merge(
        df_for_join, on=["pair_id", "network_id"], how="inner"
    )
    paths_with_pred = gpd.GeoDataFrame(paths_with_pred, geometry="geometry", crs=paths_gdf.crs)

    if PRED_GPKG.exists():
        PRED_GPKG.unlink()
    pairs_with_pred.to_file(PRED_GPKG, layer="mars_combined_predicted_pairs", driver="GPKG")
    paths_with_pred.to_file(
        PRED_GPKG, layer="mars_combined_predicted_pair_paths", driver="GPKG"
    )
    pairs_with_pred[pairs_with_pred["pred_touching_emb"] == 1].to_file(
        PRED_GPKG, layer="mars_combined_emb_touching_pairs", driver="GPKG"
    )
    pairs_with_pred[pairs_with_pred["pred_touching_logit"] == 1].to_file(
        PRED_GPKG, layer="mars_combined_logit_touching_pairs", driver="GPKG"
    )
    pairs_with_pred[pairs_with_pred["agreement_touching"] == 1].to_file(
        PRED_GPKG, layer="mars_combined_agreement_touching_pairs", driver="GPKG"
    )
    disagreement = pairs_with_pred[pairs_with_pred["emb_logit_disagreement"] == 1]
    if not disagreement.empty:
        disagreement.to_file(
            PRED_GPKG, layer="mars_combined_disagreement_pairs", driver="GPKG"
        )
    else:
        log.warning("No disagreement rows — skipping that layer")
    high_conf = pairs_with_pred[pairs_with_pred["high_confidence_any_combined"] == 1]
    if not high_conf.empty:
        high_conf.to_file(
            PRED_GPKG, layer="mars_combined_high_confidence_pairs", driver="GPKG"
        )
    else:
        log.warning("No high-confidence rows — skipping that layer")
    log.info("GPKG written: %d layers", 7)

    # --- Summary CSVs --------------------------------------------------
    summary_df = build_comparison_summary(df_out)
    summary_df.to_csv(SUMMARY_CSV, index=False)
    log.info("Wrote: %s", SUMMARY_CSV)

    by_net = build_by_network(df_out)
    by_net.to_csv(BY_NETWORK_CSV, index=False)
    log.info("Wrote: %s", BY_NETWORK_CSV)

    # --- Plots --------------------------------------------------------
    plot_probability_histograms(
        df_out,
        thresholds={"tabular": thresholds["tabular"],
                     "emb": thresholds["emb"], "logit": thresholds["logit"]},
        output=FIGURES_DIR / "probability_histograms.png",
    )
    plot_prob_scatter(
        df_out,
        xcol="prob_touching_tabular_only",
        ycol="prob_touching_emb",
        xlabel="tabular-only prob_touching",
        ylabel="geom_plus_cnn_emb prob_touching",
        title="Tabular vs. combined-embedding probability",
        output=FIGURES_DIR / "scatter_tabular_vs_emb.png",
        thresholds=(thresholds["tabular"], thresholds["emb"]),
    )
    plot_prob_scatter(
        df_out,
        xcol="prob_touching_emb",
        ycol="prob_touching_logit",
        xlabel="geom_plus_cnn_emb prob_touching",
        ylabel="geom_plus_cnn_logit prob_touching",
        title="Combined-embedding vs. combined-logit probability",
        output=FIGURES_DIR / "scatter_emb_vs_logit.png",
        thresholds=(thresholds["emb"], thresholds["logit"]),
    )
    plot_touching_bar(summary_df, FIGURES_DIR / "touching_pct_by_variant.png")

    # Contact sheets — need patch paths
    patch_index = pd.read_parquet(PATCH_INDEX_PARQUET)
    patch_paths = {
        str(r["pair_id"]): str(r["patch_path"]) for _, r in patch_index.iterrows()
    }

    # 20 pairs where both combined models predict high-confidence touching
    hc_both = df_out[
        (df_out["prob_touching_emb"] >= HIGH_CONF_PROB_MIN)
        & (df_out["prob_touching_logit"] >= HIGH_CONF_PROB_MIN)
    ].copy()
    hc_both["min_combined_prob"] = hc_both[
        ["prob_touching_emb", "prob_touching_logit"]
    ].min(axis=1)
    hc_both = hc_both.sort_values("min_combined_prob", ascending=False).head(20)
    plot_patch_contact_sheet(
        hc_both,
        patch_paths,
        title="20 pairs — both combined models high-confidence touching (prob ≥ 0.80)",
        output=FIGURES_DIR / "contact_sheet_high_conf_both.png",
        subtitle_fn=lambda r: (
            f"net={r['network_id']}\n"
            f"emb={r['prob_touching_emb']:.3f}  "
            f"logit={r['prob_touching_logit']:.3f}"
        ),
    )

    # 20 pairs where combined models disagree
    disag = df_out[df_out["emb_logit_disagreement"] == 1].copy()
    if not disag.empty:
        # Take ones with the strongest disagreement (largest |Δprob|)
        disag["abs_dprob"] = (
            disag["prob_touching_emb"] - disag["prob_touching_logit"]
        ).abs()
        disag = disag.sort_values("abs_dprob", ascending=False).head(20)
        plot_patch_contact_sheet(
            disag,
            patch_paths,
            title="20 pairs — combined-emb vs combined-logit disagreement (sorted by |Δprob|)",
            output=FIGURES_DIR / "contact_sheet_disagreement.png",
            subtitle_fn=lambda r: (
                f"net={r['network_id']}\n"
                f"emb={r['prob_touching_emb']:.3f}  "
                f"logit={r['prob_touching_logit']:.3f}\n"
                f"emb_pred={int(r['pred_touching_emb'])}  "
                f"logit_pred={int(r['pred_touching_logit'])}"
            ),
        )

    log.info("Done. Outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
