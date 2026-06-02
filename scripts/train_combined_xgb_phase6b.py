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
    python scripts/train_combined_xgb_phase6b.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from channel_heads.models.cnn import (  # noqa: E402
    DEFAULT_EMBEDDING_DIM,
    OutletCNN,
    OutletPairDataset,
)
from channel_heads.models.device import pick_device  # noqa: E402

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RASTER_MANIFEST_CSV = PROJECT_ROOT / "data/results/raster_manifest.csv"
CNN_MODEL_PATH = PROJECT_ROOT / "models/cnn_outlet_final.pt"
MASTER_V4_CSV = PROJECT_ROOT / "data/results/master_dataset_v4_cnn_full.csv"

MODELS_DIR = PROJECT_ROOT / "models"
METRICS_CSV = MODELS_DIR / "combined_models_comparison.csv"

# Mirror notebooks/training/04_cnn_embeddings.ipynb cell 13 hyperparameters so the
# three variants are directly comparable.
N_ESTIMATORS = 200
MAX_DEPTH = 4
LEARNING_RATE = 0.1
RANDOM_STATE = 42

# Mirror notebooks/training/02_train_classifier.ipynb cell 24
TEST_SIZE = 0.20

# Mirror notebooks/training/02_train_classifier.ipynb cell 33
THRESHOLD_MIN_RECALL = 0.50

BATCH_SIZE = 64

GEOM_FEATURES: list[str] = [
    "orientation_diff_deg",
    "headhead_dist_norm",
    "apex_angle_deg",
    "strahler_order_diff",
    "proximity_profile_norm",
]
EMB_FEATURES: list[str] = [f"emb_{i}" for i in range(DEFAULT_EMBEDDING_DIM)]

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
    model = OutletCNN(embedding_dim=DEFAULT_EMBEDDING_DIM)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"State-dict mismatch: missing={missing} unexpected={unexpected}"
        )
    model.to(device)
    model.eval()

    paths = [Path(p) for p in manifest_df["raster_path"]]
    dummy_labels = np.zeros(len(paths), dtype=np.float32)
    ds = OutletPairDataset(paths, dummy_labels, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_emb: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            emb = model.embed(images)
            logit = model(images).squeeze(-1)
            all_emb.append(emb.cpu().numpy())
            all_logits.append(logit.cpu().numpy())
    return np.vstack(all_emb), np.concatenate(all_logits)


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
    valid_mask = df["raster_path"].notna()
    if "raster_status" in df.columns:
        valid_mask &= df["raster_status"].eq("ok")
    df = df[valid_mask].reset_index(drop=True)

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
) -> tuple[XGBClassifier, dict]:
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

    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]

    # Threshold tuning — port of nb02 cell 33
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    valid_mask = recalls[:-1] >= THRESHOLD_MIN_RECALL
    if valid_mask.any():
        local_best = int(np.argmax(precisions[:-1][valid_mask]))
        orig_indices = np.where(valid_mask)[0]
        best_idx = int(orig_indices[local_best])
        opt_threshold = float(thresholds[best_idx])
        tuned_precision = float(precisions[best_idx])
        tuned_recall = float(recalls[best_idx])
        threshold_source = "max_precision_at_recall>=0.50"
    else:
        opt_threshold = 0.5
        tuned_precision = float("nan")
        tuned_recall = float("nan")
        threshold_source = "fallback_default_0.5"
        log.warning(
            "Variant %s: no PR-curve point with recall >= %.2f; using 0.5",
            name,
            THRESHOLD_MIN_RECALL,
        )

    pred = (proba >= opt_threshold).astype(int)
    pred_default = (proba >= 0.5).astype(int)

    metrics = {
        "variant": name,
        "n_features": len(feature_cols),
        "feature_columns": ",".join(feature_cols),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_train_pos": n_pos,
        "n_train_neg": n_neg,
        "scale_pos_weight": float(spw),
        "test_pos_fraction": float(y_test.mean()),
        "optimal_threshold": opt_threshold,
        "threshold_source": threshold_source,
        "roc_auc_test": float(roc_auc_score(y_test, proba)),
        "pr_auc_test": float(average_precision_score(y_test, proba)),
        "precision_tuned": float(precision_score(y_test, pred)),
        "recall_tuned": float(recall_score(y_test, pred)),
        "f1_tuned": float(f1_score(y_test, pred)),
        "accuracy_tuned": float(accuracy_score(y_test, pred)),
        "precision_default_0.5": float(precision_score(y_test, pred_default)),
        "recall_default_0.5": float(recall_score(y_test, pred_default)),
        "f1_default_0.5": float(f1_score(y_test, pred_default)),
        "accuracy_default_0.5": float(accuracy_score(y_test, pred_default)),
        "pr_curve_tuned_precision": tuned_precision,
        "pr_curve_tuned_recall": tuned_recall,
    }
    return model, metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
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
        ("geom_plus_cnn_emb", GEOM_FEATURES + EMB_FEATURES),
        ("geom_plus_cnn_logit", GEOM_FEATURES + ["cnn_logit"]),
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
        feature_path.write_text("\n".join(feats) + "\n")
        threshold_path.write_text(f"{metrics['optimal_threshold']:.6f}\n")

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
