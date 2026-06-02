"""Classification-evaluation primitives shared by the threshold-tuning and
diagnostics workflows.

Pure functions over ``y_true`` / predicted probabilities — the F1-optimal
threshold search and the standard metric bundle that were inline in
``scripts/retune_threshold_regime.py`` (and mirrored in the training /
diagnostics scripts). No model or I/O dependency.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def f1_optimal_threshold(y_true: npt.NDArray, proba: npt.NDArray) -> tuple[float, float]:
    """Return ``(threshold, f1)`` maximising F1 along the precision-recall curve.

    Mirrors the regime retune logic exactly: F1 is computed from the PR-curve
    precision/recall arrays, and the final PR point (recall = 0, which has no
    corresponding threshold) is excluded when picking the argmax.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1_arr = 2 * precisions * recalls / np.where(precisions + recalls > 0, precisions + recalls, 1)
    f1_idx = int(np.argmax(f1_arr[:-1]))
    return float(thresholds[f1_idx]), float(f1_arr[f1_idx])


def max_precision_threshold(
    y_true: npt.NDArray, proba: npt.NDArray, min_recall: float = 0.5
) -> float:
    """Threshold maximising precision subject to ``recall >= min_recall``.

    The production / regime *training* protocol. Returns ``0.5`` if no PR point
    reaches ``min_recall``. (The final PR point, which has no threshold, is
    excluded.)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    ok = recalls[:-1] >= min_recall
    if not ok.any():
        return 0.5
    idx = int(np.argmax(np.where(ok, precisions[:-1], -1)))
    return float(thresholds[idx])


def classification_metrics(
    y_true: npt.NDArray, proba: npt.NDArray, threshold: float
) -> dict[str, float]:
    """Standard threshold-decision metrics + probability-ranking metrics.

    ``precision`` / ``recall`` / ``f1`` / ``accuracy`` are computed at
    ``threshold``; ``roc_auc`` / ``pr_auc`` are threshold-independent.
    """
    pred = (proba >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_true, pred)),
        "recall": float(recall_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
    }
