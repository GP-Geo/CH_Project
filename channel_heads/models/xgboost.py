"""XGBoost model loading, verification, and thresholded inference.

Curated surface over :mod:`channel_heads.inference.xgb`.
"""

from __future__ import annotations

from channel_heads.inference.xgb import (
    load_feature_columns,
    load_threshold,
    load_xgb_model,
    predict_with_threshold,
    verify_feature_matrix,
    verify_model_feature_order,
)

__all__ = [
    "load_xgb_model",
    "load_feature_columns",
    "load_threshold",
    "verify_model_feature_order",
    "verify_feature_matrix",
    "predict_with_threshold",
]
