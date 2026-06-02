"""XGBoost model loading, verification, and thresholded inference.

Canonical implementation of the shared XGBoost inference helpers used by every
Mars prediction entry point (tabular 5-feature, combined emb/logit, and the
regime variant). These scripts each once carried their own near-identical
copies of these small artifact-loading, compatibility-checking, and
predict-with-threshold routines; they are consolidated here so every inference
entry point performs the *same* pre-prediction checks.

Everything in this module is pure model/IO glue — it does not change any
scientific behavior: the same artifacts are loaded, the same invariants raise,
and prediction is the standard ``predict_proba(...)[:, 1] >= threshold``.

The historical import location :mod:`channel_heads.inference.xgb` re-exports
everything here as a compatibility shim.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:  # avoid importing pandas/xgboost at module load
    import pandas as pd
    from xgboost import XGBClassifier


def load_feature_columns(path: Path | str) -> list[str]:
    """Read the authoritative feature-name order, one per line.

    Blank lines are ignored. Raises if no feature names remain (a missing or
    empty feature-columns file is always a hard error — the model's feature
    order is never optional).
    """
    feats = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not feats:
        raise RuntimeError(f"No features listed in {path}")
    return feats


def load_threshold(path: Path | str) -> float:
    """Read the decision threshold from the first non-empty line of ``path``."""
    lines = Path(path).read_text().strip().splitlines()
    if not lines:
        raise RuntimeError(f"Empty threshold file: {path}")
    return float(lines[0].strip())


def load_xgb_model(
    model_path: Path | str, expected_features: list[str] | None = None
) -> XGBClassifier:
    """Load an ``XGBClassifier`` from JSON; optionally verify its feature order.

    When ``expected_features`` is given, the model's stored ``feature_names``
    (if any) must match it exactly, guarding against a model/feature-columns
    mismatch before predicting.
    """
    from xgboost import XGBClassifier

    model = XGBClassifier()
    model.load_model(str(model_path))
    if expected_features is not None:
        verify_model_feature_order(model, expected_features, context=str(model_path))
    return model


def verify_model_feature_order(
    model: XGBClassifier, expected_features: list[str], context: str = ""
) -> None:
    """Raise if the booster's stored feature_names differ from ``expected_features``.

    A booster with no stored feature_names (older artifacts) is accepted —
    only an explicit mismatch is fatal.
    """
    booster_feats = list(getattr(model.get_booster(), "feature_names", []) or [])
    if booster_feats and booster_feats != expected_features:
        prefix = f"{context}: " if context else ""
        raise RuntimeError(
            f"{prefix}model feature_names {booster_feats} differ from expected "
            f"feature order {expected_features}"
        )


def verify_feature_matrix(
    df: pd.DataFrame, feature_cols: list[str], context: str = ""
) -> dict[str, int]:
    """Pre-prediction invariants on the model feature matrix.

    Raises if any model feature column is missing, non-numeric, or contains an
    infinite value. NaN cells are *allowed* (XGBoost handles them natively, as
    on Earth) and reported in the returned stats so the caller can log them.

    Returns a dict with ``n_rows``, ``n_cols``, ``n_nan_cells`` and
    ``n_nan_rows``.
    """
    prefix = f"{context}: " if context else ""

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{prefix}missing model feature columns: {missing}")

    X = df[feature_cols]
    non_numeric = [c for c in feature_cols if not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric:
        raise RuntimeError(
            f"{prefix}non-numeric model feature columns: {non_numeric} "
            f"(dtypes: {X[non_numeric].dtypes.to_dict()})"
        )

    arr = X.to_numpy(dtype=float)
    n_inf = int(np.isinf(arr).sum())
    if n_inf:
        raise RuntimeError(
            f"{prefix}found {n_inf} inf values in model feature matrix — " "refusing to predict."
        )

    return {
        "n_rows": int(arr.shape[0]),
        "n_cols": int(arr.shape[1]),
        "n_nan_cells": int(np.isnan(arr).sum()),
        "n_nan_rows": int(np.isnan(arr).any(axis=1).sum()),
    }


def predict_with_threshold(
    model: XGBClassifier,
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int_]]:
    """Standard touching-probability prediction + threshold decision.

    Returns ``(proba, pred)`` where ``proba`` is P(touching) and ``pred`` is
    ``(proba >= threshold)`` as int.
    """
    proba = model.predict_proba(df[feature_cols])[:, 1]
    pred = (proba >= threshold).astype(int)
    return proba, pred


__all__ = [
    "load_feature_columns",
    "load_threshold",
    "load_xgb_model",
    "verify_model_feature_order",
    "verify_feature_matrix",
    "predict_with_threshold",
]
