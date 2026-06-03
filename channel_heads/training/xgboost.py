"""Reusable Earth XGBoost training helpers (Phase 6B + regime combined models).

Centralizes the training-recipe logic that was duplicated in
``scripts/train_combined_xgb_phase6b.py`` and
``scripts/train_combined_xgb_regime.py``:

* strict CNN feature extraction (embeddings + classifier logit / embeddings
  only) using ``load_state_dict(..., strict=True)``,
* the shared XGBoost hyperparameter configuration and ``scale_pos_weight``,
* the PR-curve threshold policy (max precision subject to ``recall >= 0.50``,
  fallback ``0.5``) with its exact ``threshold_source`` strings,
* the combined-variant fit-score-and-metrics routine, and
* feature-columns / threshold file writers with the exact on-disk format.

Behaviour is extracted verbatim so the scripts can call these helpers without
any change in model config, feature order, split, threshold, or output files.
The strict extractors are intentionally distinct from the lenient
:func:`channel_heads.models.cnn_features.extract_embeddings`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
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
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from channel_heads.models.cnn import (
    DEFAULT_EMBEDDING_DIM,
    OutletCNN,
    OutletPairDataset,
)
from channel_heads.training.datasets import (  # noqa: F401  (re-exported feature constants)
    CNN_LOGIT_FEATURE,
    EMB_FEATURES,
    GEOM_FEATURES,
    GEOM_PLUS_EMB,
    GEOM_PLUS_LOGIT,
)

# =============================================================================
# Shared hyperparameters (mirror notebooks/training cell 13 / nb02 cell 24, 33)
# =============================================================================

N_ESTIMATORS = 200
MAX_DEPTH = 4
LEARNING_RATE = 0.1
RANDOM_STATE = 42
TEST_SIZE = 0.20
THRESHOLD_MIN_RECALL = 0.50
BATCH_SIZE = 64


# =============================================================================
# Strict CNN feature extraction (eval mode, no augmentation)
# =============================================================================


def _load_strict_cnn(model_path: Path, device: str, embedding_dim: int) -> OutletCNN:
    """Load an OutletCNN with a strict state-dict check, in eval mode on device."""
    model = OutletCNN(embedding_dim=embedding_dim)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch: missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model


def extract_emb_and_logit_strict(
    model_path: Path,
    raster_paths: list[Path],
    device: str,
    batch_size: int = BATCH_SIZE,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the CNN over rasters; return ``(embeddings (N, dim), logits (N,))``.

    Embeddings are ``model.embed(x)``; logits are ``model(x).squeeze(-1)``. Eval
    mode (dropout is a no-op), no augmentation, strict state-dict load.
    """
    model = _load_strict_cnn(model_path, device, embedding_dim)
    dummy_labels = np.zeros(len(raster_paths), dtype=np.float32)
    ds = OutletPairDataset(raster_paths, dummy_labels, augment=False)
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


def extract_emb_strict(
    model_path: Path,
    raster_paths: list[Path],
    device: str,
    batch_size: int = BATCH_SIZE,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> np.ndarray:
    """Run the CNN in eval mode over rasters; return ``(N, dim)`` embeddings.

    Strict state-dict load, no augmentation. Used by the regime combined-XGB
    training step.
    """
    model = _load_strict_cnn(model_path, device, embedding_dim)
    dummy = np.zeros(len(raster_paths), dtype=np.float32)
    ds = OutletPairDataset(raster_paths, dummy, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            out.append(model.embed(images).cpu().numpy())
    return np.vstack(out)


# =============================================================================
# XGBoost config / threshold policy
# =============================================================================


def xgb_scale_pos_weight(y_train: npt.NDArray) -> float:
    """``n_neg / max(n_pos, 1)`` — the class-imbalance weight used by all variants."""
    n_pos = int(np.asarray(y_train).sum())
    n_neg = int(len(y_train) - n_pos)
    return n_neg / max(n_pos, 1)


def build_xgb_classifier(scale_pos_weight: float) -> XGBClassifier:
    """Construct the combined-model XGBoost classifier with the frozen config."""
    return XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
        tree_method="hist",
    )


def tune_threshold_max_precision(
    y_test: npt.NDArray,
    proba: npt.NDArray,
    min_recall: float = THRESHOLD_MIN_RECALL,
) -> tuple[float, str, float, float]:
    """PR-curve threshold: max precision subject to ``recall >= min_recall``.

    Returns ``(optimal_threshold, threshold_source, tuned_precision,
    tuned_recall)``. Falls back to ``0.5`` (with NaN tuned precision/recall and
    ``threshold_source == "fallback_default_0.5"``) when no PR point reaches
    ``min_recall``. This is the exact tie-breaking used by the training scripts.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    valid_mask = recalls[:-1] >= min_recall
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
    return opt_threshold, threshold_source, tuned_precision, tuned_recall


# =============================================================================
# Combined-variant training + metrics
# =============================================================================


def train_combined_variant(
    name_key: str,
    name_value: str,
    feature_cols: list[str],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold_min_recall: float = THRESHOLD_MIN_RECALL,
) -> tuple[XGBClassifier, dict]:
    """Fit one combined XGBoost variant, tune its threshold, and build metrics.

    Reproduces the ``train_variant`` recipe shared by the Phase 6B and regime
    training scripts. ``name_key`` is ``"variant"`` for Phase 6B and ``"regime"``
    for the regime trainer; it is placed first in the returned metrics dict so
    the persisted CSV column order is unchanged.
    """
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = n_neg / max(n_pos, 1)

    model = build_xgb_classifier(spw)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]

    opt_threshold, threshold_source, tuned_precision, tuned_recall = (
        tune_threshold_max_precision(y_test, proba, threshold_min_recall)
    )

    pred = (proba >= opt_threshold).astype(int)
    pred_default = (proba >= 0.5).astype(int)

    metrics = {
        name_key: name_value,
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


# =============================================================================
# Artifact file writers (exact on-disk format)
# =============================================================================


def write_feature_columns(path: str | Path, feature_cols: list[str]) -> None:
    """Write one feature per line, in order, with a trailing newline."""
    Path(path).write_text("\n".join(feature_cols) + "\n")


def write_threshold(path: str | Path, threshold: float) -> None:
    """Write the decision threshold as ``"{threshold:.6f}\\n"``."""
    Path(path).write_text(f"{threshold:.6f}\n")


__all__ = [
    "N_ESTIMATORS",
    "MAX_DEPTH",
    "LEARNING_RATE",
    "RANDOM_STATE",
    "TEST_SIZE",
    "THRESHOLD_MIN_RECALL",
    "BATCH_SIZE",
    "GEOM_FEATURES",
    "EMB_FEATURES",
    "CNN_LOGIT_FEATURE",
    "GEOM_PLUS_EMB",
    "GEOM_PLUS_LOGIT",
    "extract_emb_and_logit_strict",
    "extract_emb_strict",
    "xgb_scale_pos_weight",
    "build_xgb_classifier",
    "tune_threshold_max_precision",
    "train_combined_variant",
    "write_feature_columns",
    "write_threshold",
]
