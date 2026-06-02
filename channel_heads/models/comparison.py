"""Model-variant comparison utilities.

Compares the geometric-only / geom+CNN-embedding / geom+CNN-logit variants (and
regime variants) on a shared labelled set. Thin layer over
:func:`channel_heads.eval.metrics.classification_metrics`.
"""

from __future__ import annotations

import pandas as pd

from channel_heads.eval.metrics import classification_metrics


def compare_predictions(
    y_true,
    proba_by_model: dict[str, pd.Series],
    threshold_by_model: dict[str, float] | float = 0.5,
) -> pd.DataFrame:
    """Tabulate metrics for several models on the same ground truth.

    Parameters
    ----------
    y_true : array-like
        Shared binary labels.
    proba_by_model : dict[str, array-like]
        Model name → predicted probabilities.
    threshold_by_model : dict[str, float] | float
        Per-model decision threshold, or a single shared threshold.

    Returns a DataFrame indexed by model name with the standard metric columns.
    """
    import numpy as np

    y_true = np.asarray(y_true)
    rows = {}
    for name, proba in proba_by_model.items():
        thr = (
            threshold_by_model[name]
            if isinstance(threshold_by_model, dict)
            else threshold_by_model
        )
        rows[name] = classification_metrics(y_true, np.asarray(proba), thr)
    return pd.DataFrame(rows).T


__all__ = ["compare_predictions"]
