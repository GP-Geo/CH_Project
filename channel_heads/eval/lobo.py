"""Leave-one-basin-out (LOBO) XGBoost report for the geom+CNN-embedding model.

Owns the LOBO-CV diagnostic that was inline in ``scripts/eval_lobo_cv.py``:
the per-config dataset map, the per-fold XGBoost fit, the leave-one-basin-out
out-of-fold probabilities, and the pooled-metrics summary. The generic
leave-one-group-out splitter and metric primitives stay in
:mod:`channel_heads.eval.splitting` / :mod:`channel_heads.eval.metrics`.

The fold AUC list includes only held-out basins that contain both classes
(handled by :func:`channel_heads.eval.splitting.leave_one_group_out_oof`).

This module deliberately uses its own XGBoost configuration — note it has **no**
``n_jobs`` or ``tree_method`` argument, unlike the training recipe in
:mod:`channel_heads.training.xgboost` — to preserve the exact LOBO behaviour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from xgboost import XGBClassifier

from channel_heads.eval.metrics import classification_metrics, max_precision_threshold
from channel_heads.eval.splitting import leave_one_group_out_oof

# Five geom features + four CNN embeddings, in the model's feature order.
GEOM = [
    "orientation_diff_deg",
    "headhead_dist_norm",
    "apex_angle_deg",
    "strahler_order_diff",
    "proximity_profile_norm",
]
EMB = [f"emb_{i}" for i in range(4)]
FEATURES = GEOM + EMB

N_ESTIMATORS, MAX_DEPTH, LR, SEED = 200, 4, 0.1, 42


def lobo_dataset_paths(root: str | Path) -> dict[str, Path]:
    """Return the ``config -> dataset CSV path`` map used by the LOBO report."""
    root = Path(root)
    return {
        "baseline": root / "data/results/master_dataset_v4_cnn_full.csv",
        "regA": root / "data/results/master_dataset_regA_with_emb.csv",
        "regB": root / "data/results/master_dataset_regB_with_emb.csv",
        "regC": root / "data/results/master_dataset_regC_with_emb.csv",
    }


def _fit_predict(X_tr: pd.DataFrame, y_tr: npt.NDArray, X_te: pd.DataFrame) -> np.ndarray:
    """Train the geom+emb XGBoost on a fold and predict P(touching)."""
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = XGBClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LR,
        scale_pos_weight=spw,
        eval_metric="logloss",
        random_state=SEED,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def lobo_xgb_report(df: pd.DataFrame) -> dict:
    """Leave-one-basin-out report for one dataset.

    Drops rows with NaN in ``FEATURES + ["y", "basin"]``, computes
    leave-one-basin-out OOF probabilities, picks the max-precision threshold
    (``min_recall=0.5``), and returns the pooled-metrics summary dict.
    """
    df = df.dropna(subset=FEATURES + ["y", "basin"]).reset_index(drop=True)
    oof, y, groups, fold_aucs = leave_one_group_out_oof(df, FEATURES, _fit_predict)
    thr = max_precision_threshold(y, oof)
    m = classification_metrics(y, oof, thr)
    return {
        "n": len(df),
        "n_basins": groups.nunique(),
        "pooled_auc": m["roc_auc"],
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "threshold": thr,
        "precision": m["precision"],
        "recall": m["recall"],
        "f1": m["f1"],
        "accuracy": m["accuracy"],
    }


__all__ = [
    "GEOM",
    "EMB",
    "FEATURES",
    "lobo_dataset_paths",
    "lobo_xgb_report",
]
