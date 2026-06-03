"""Diagnostic helpers for model threshold and PR-curve inspection.

Used by ``scripts/diagnostics/diag_regB_threshold.py`` and by
``notebooks/diagnostics/regB_threshold.ipynb``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import precision_recall_curve

from channel_heads.eval.metrics import classification_metrics, f1_optimal_threshold
from channel_heads.eval.splitting import outlet_group_holdout
from channel_heads.models.xgboost import load_feature_columns, load_xgb_model


def holdout_split_predict(
    csv: Path,
    model_path: Path,
    feature_cols_path: Path,
) -> tuple[npt.NDArray, npt.NDArray, str]:
    """Reproduce the outlet-group train/test split and predict on the held-out test set.

    Returns ``(y_test, proba, feature_label)`` where *feature_label* is the
    comma-joined feature column names (useful for plot titles).
    """
    df = pd.read_csv(csv)
    feats = load_feature_columns(feature_cols_path)
    missing = [c for c in feats + ["y", "basin", "outlet"] if c not in df.columns]
    if missing:
        raise RuntimeError(f"{Path(csv).name}: missing columns {missing}")

    _, test_idx = outlet_group_holdout(df)
    y = df["y"].astype(int).to_numpy()
    X = df[feats].to_numpy(dtype=float)
    X_test, y_test = X[test_idx], y[test_idx]

    model = load_xgb_model(model_path)
    proba = model.predict_proba(X_test)[:, 1]
    return y_test, proba, ",".join(feats)


def pr_curve_metrics(
    y_test: npt.NDArray,
    proba: npt.NDArray,
    threshold: float,
) -> dict:
    """Compute scalar and curve PR/ROC metrics for a given operating threshold.

    Returns a dict with keys:
    ``pr_auc``, ``roc_auc``, ``P_at_thr``, ``R_at_thr``, ``F1_at_thr``,
    ``F1_optimal_threshold``, ``F1_optimal_max``,
    ``precisions``, ``recalls``, ``thresholds`` (sklearn PR-curve arrays).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_test, proba)
    m = classification_metrics(y_test, proba, threshold)
    f1_thr, f1_max = f1_optimal_threshold(y_test, proba)
    return {
        "pr_auc": m["pr_auc"],
        "roc_auc": m["roc_auc"],
        "P_at_thr": m["precision"],
        "R_at_thr": m["recall"],
        "F1_at_thr": m["f1"],
        "F1_optimal_threshold": f1_thr,
        "F1_optimal_max": f1_max,
        "precisions": precisions,
        "recalls": recalls,
        "thresholds": thresholds,
    }


__all__ = ["holdout_split_predict", "pr_curve_metrics"]
