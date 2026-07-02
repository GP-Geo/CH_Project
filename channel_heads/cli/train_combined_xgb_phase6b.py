#!/usr/bin/env python
"""Phase 6B — Train and persist three Earth XGBoost variants for research
comparison and future Mars combined-feature inference.

Headless replay of the experimental notebook 04 / 05 recipe with three
key improvements:

  1. The CNN embeddings + classifier logit are re-extracted for ALL 17
     Earth basins using ``models/cnn_outlet_final.pt`` so every row has
     a consistent, in-distribution-for-the-CNN representation (the
     existing ``master_dataset_v3_cnn.csv`` covers only 16 basins —
     Taiwan was held out during CNN training and never persisted).
  2. The three XGBoost variants are trained with identical
     hyperparameters and the identical ``GroupShuffleSplit`` so the
     comparison isolates "does adding CNN features help?" rather than
     mixing in hyperparameter-search differences.
  3. The decision threshold for each variant is tuned with the same
     PR-curve protocol as ``notebooks/training/02_train_classifier.ipynb``
     cell 33 — precision maximisation subject to recall ≥ 0.50, fall
     back to 0.5 if no threshold satisfies the constraint.

Variants:
  - A "geom_only"           — 5 geometric features
  - B "geom_plus_cnn_emb"   — 5 geom + 4 CNN embeddings  (9 feats)
  - C "geom_plus_cnn_logit" — 5 geom + 1 CNN classifier logit (6 feats)

Outputs under ``models/``:
  xgb_geom_only.json
  xgb_geom_plus_cnn_emb.json
  xgb_geom_plus_cnn_logit.json
  feature_columns_geom_only.txt
  feature_columns_geom_plus_cnn_emb.txt
  feature_columns_geom_plus_cnn_logit.txt
  optimal_threshold_geom_only.txt
  optimal_threshold_geom_plus_cnn_emb.txt
  optimal_threshold_geom_plus_cnn_logit.txt
  combined_models_comparison.csv

Also writes:
  data/results/master_dataset_v4_cnn_full.csv  — 5922 rows × all 17
  basins with emb_0..emb_3 and cnn_logit attached.

This script does NOT touch:
  - models/xgb_touching_classifier.json (the production tabular model)
  - models/feature_columns.txt
  - models/optimal_threshold.txt
  - any Mars artifact under data/Mars/

Run:
    python -m channel_heads train-combined-xgb-phase6b
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from channel_heads.io.paths import PROJECT_ROOT  # noqa: E402
from channel_heads.models.cnn import DEFAULT_EMBEDDING_DIM  # noqa: E402
from channel_heads.models.device import pick_device  # noqa: E402
from channel_heads.training import xgboost as xgb_training  # noqa: E402
from channel_heads.training.datasets import (  # noqa: E402
    EMB_FEATURES,
    GEOM_FEATURES,
    GEOM_PLUS_EMB,
    GEOM_PLUS_LOGIT,
    filter_valid_raster_rows,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
RASTER_MANIFEST_CSV = PROJECT_ROOT / "data/results/raster_manifest.csv"
CNN_MODEL_PATH = PROJECT_ROOT / "models/cnn_outlet_final.pt"
MASTER_V4_CSV = PROJECT_ROOT / "data/results/master_dataset_v4_cnn_full.csv"

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_CSV = MODELS_DIR / "combined_models_comparison.csv"

# Mirror notebooks/training/04_cnn_embeddings.ipynb cell 13 hyperparameters so the
# three variants are directly comparable.
N_ESTIMATORS = xgb_training.N_ESTIMATORS
MAX_DEPTH = xgb_training.MAX_DEPTH
LEARNING_RATE = xgb_training.LEARNING_RATE
RANDOM_STATE = xgb_training.RANDOM_STATE

# Mirror notebooks/training/02_train_classifier.ipynb cell 24
TEST_SIZE = xgb_training.TEST_SIZE

# Mirror notebooks/training/02_train_classifier.ipynb cell 33
THRESHOLD_MIN_RECALL = xgb_training.THRESHOLD_MIN_RECALL

BATCH_SIZE = xgb_training.BATCH_SIZE

log = logging.getLogger("phase6b")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# CNN embedding + logit extraction (in eval mode, no augmentation)
# ---------------------------------------------------------------------------
def extract_emb_and_logit(
    model_path: Path,
    manifest_df: pd.DataFrame,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run cnn_outlet_final.pt over all rasters listed in manifest_df.

    Returns (embeddings (N, 4), logits (N,)). Embeddings are
    ``model.embed(x)`` (== ReLU(fc_embed(features)) without dropout).
    Logits are ``model(x).squeeze()`` — in eval mode the Dropout layer is
    a no-op, so the forward pass produces ``fc_head(embedding)`` directly.
    """
    paths = [Path(p) for p in manifest_df["raster_path"]]
    return xgb_training.extract_emb_and_logit_strict(
        model_path, paths, device=device, batch_size=batch_size
    )


# ---------------------------------------------------------------------------
# Build the unified Earth dataset
# ---------------------------------------------------------------------------
def build_master_v4(device: str) -> pd.DataFrame:
    log.info("Loading raster manifest: %s", RASTER_MANIFEST_CSV)
    df = pd.read_csv(RASTER_MANIFEST_CSV)
    log.info(
        "Manifest: %d rows, %d basins (incl. Taiwan), %d with valid raster_path",
        len(df),
        df["basin"].nunique(),
        int(df["raster_path"].notna().sum()),
    )
    df = filter_valid_raster_rows(df)

    # Sanity: at least the 5 model features and y label must be present.
    required = GEOM_FEATURES + ["y", "basin", "outlet", "raster_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Manifest missing required columns: {missing}")

    log.info("Extracting CNN embeddings + logits on %s ...", device)
    emb, logit = extract_emb_and_logit(
        CNN_MODEL_PATH, df, batch_size=BATCH_SIZE, device=device
    )
    if emb.shape[0] != len(df):
        raise RuntimeError(
            f"Embedding row count {emb.shape[0]} != manifest rows {len(df)}"
        )
    for i in range(DEFAULT_EMBEDDING_DIM):
        df[f"emb_{i}"] = emb[:, i]
    df["cnn_logit"] = logit

    # Quick finiteness check
    for col in EMB_FEATURES + ["cnn_logit"]:
        arr = df[col].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"Non-finite values in {col}: nan={int(np.isnan(arr).sum())} "
                f"inf={int(np.isinf(arr).sum())}"
            )

    log.info("Writing unified Earth dataset: %s", MASTER_V4_CSV)
    df.to_csv(MASTER_V4_CSV, index=False)
    return df


# ---------------------------------------------------------------------------
# Train + tune a single variant
# ---------------------------------------------------------------------------
def train_variant(
    name: str,
    feature_cols: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[object, dict]:
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = n_neg / max(n_pos, 1)
    log.info(
        "Variant %s | n_features=%d | n_train=%d (pos=%d, spw=%.3f) | n_test=%d",
        name,
        len(feature_cols),
        len(X_train),
        n_pos,
        spw,
        len(X_test),
    )

    model, metrics = xgb_training.train_combined_variant(
        "variant",
        name,
        feature_cols,
        X_train,
        y_train,
        X_test,
        y_test,
        threshold_min_recall=THRESHOLD_MIN_RECALL,
    )
    if metrics["threshold_source"] == "fallback_default_0.5":
        log.warning(
            "Variant %s: no PR-curve point with recall >= %.2f; using 0.5",
            name,
            THRESHOLD_MIN_RECALL,
        )
    return model, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="channel-heads train-combined-xgb-phase6b",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.parse_args(argv)
    setup_logging()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    log.info("Device: %s", device)

    # --- 1. Build the unified Earth dataset ----------------------------
    if MASTER_V4_CSV.exists():
        log.info(
            "Master v4 dataset already exists at %s; rebuilding to ensure "
            "freshness (cheap on MPS, ensures CNN ↔ model consistency).",
            MASTER_V4_CSV,
        )
    df = build_master_v4(device)

    # --- 2. Train/test split (same protocol as nb02 cell 24-25) --------
    df = df.copy()
    df["outlet_group"] = df["basin"].astype(str) + "__" + df["outlet"].astype(str)
    gss = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y = df["y"].astype(int).to_numpy()
    train_idx, test_idx = next(gss.split(df, y, groups=df["outlet_group"]))
    log.info(
        "Split: %d train rows, %d test rows; %d/%d outlets in test",
        len(train_idx),
        len(test_idx),
        df.iloc[test_idx]["outlet_group"].nunique(),
        df["outlet_group"].nunique(),
    )

    # --- 3. Train three variants ---------------------------------------
    variants: list[tuple[str, list[str]]] = [
        ("geom_only", GEOM_FEATURES),
        ("geom_plus_cnn_emb", GEOM_PLUS_EMB),
        ("geom_plus_cnn_logit", GEOM_PLUS_LOGIT),
    ]

    all_metrics: list[dict] = []
    for name, feats in variants:
        X = df[feats].to_numpy()
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model, metrics = train_variant(
            name, feats, X_train, y_train, X_test, y_test
        )

        # --- 4. Persist artifacts -------------------------------------
        model_path = MODELS_DIR / f"xgb_{name}.json"
        feature_path = MODELS_DIR / f"feature_columns_{name}.txt"
        threshold_path = MODELS_DIR / f"optimal_threshold_{name}.txt"

        model.save_model(str(model_path))
        xgb_training.write_feature_columns(feature_path, feats)
        xgb_training.write_threshold(threshold_path, metrics["optimal_threshold"])

        log.info("Saved: %s", model_path)
        log.info("Saved: %s", feature_path)
        log.info("Saved: %s (threshold=%.6f)", threshold_path, metrics["optimal_threshold"])

        # Log key metrics for visibility
        log.info(
            "  ROC AUC=%.3f  PR AUC=%.3f  | tuned: P=%.3f R=%.3f F1=%.3f Acc=%.3f",
            metrics["roc_auc_test"],
            metrics["pr_auc_test"],
            metrics["precision_tuned"],
            metrics["recall_tuned"],
            metrics["f1_tuned"],
            metrics["accuracy_tuned"],
        )
        metrics["model_path"] = str(model_path.relative_to(PROJECT_ROOT))
        metrics["feature_columns_path"] = str(
            feature_path.relative_to(PROJECT_ROOT)
        )
        metrics["threshold_path"] = str(
            threshold_path.relative_to(PROJECT_ROOT)
        )
        all_metrics.append(metrics)

    # --- 5. Comparison metrics CSV -------------------------------------
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(METRICS_CSV, index=False)
    log.info("Wrote comparison metrics: %s", METRICS_CSV)

    # Compact comparison print for the log
    log.info(
        "\n%s",
        metrics_df[
            [
                "variant",
                "n_features",
                "roc_auc_test",
                "pr_auc_test",
                "optimal_threshold",
                "precision_tuned",
                "recall_tuned",
                "f1_tuned",
                "accuracy_tuned",
            ]
        ]
        .round(4)
        .to_string(index=False),
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
