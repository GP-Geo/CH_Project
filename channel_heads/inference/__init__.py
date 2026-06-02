"""Shared inference helpers for the Mars XGBoost prediction scripts.

Artifact loading, model/feature compatibility checks, and predict-with-threshold
glue used by the tabular, combined (emb/logit), and regime inference entry
points. The XGBoost helpers now live in :mod:`channel_heads.models.xgboost`
(re-exported here via the :mod:`channel_heads.inference.xgb` shim); device
selection now lives in :mod:`channel_heads.models.device` (re-exported here via
the :mod:`channel_heads.inference.device` shim).
"""

from __future__ import annotations

from .device import pick_device
from .xgb import (
    load_feature_columns,
    load_threshold,
    load_xgb_model,
    predict_with_threshold,
    verify_feature_matrix,
    verify_model_feature_order,
)

__all__ = [
    "load_feature_columns",
    "load_threshold",
    "load_xgb_model",
    "pick_device",
    "predict_with_threshold",
    "verify_feature_matrix",
    "verify_model_feature_order",
]
