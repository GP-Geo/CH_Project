"""Compatibility shim — XGBoost inference helpers moved to ``channel_heads.models.xgboost``.

The canonical implementation now lives in :mod:`channel_heads.models.xgboost`.
This module is preserved so existing imports (``channel_heads.inference.xgb``
and ``from channel_heads.inference import ...``), older notebooks, and scripts
keep working unchanged. New code should import from
:mod:`channel_heads.models.xgboost`.
"""

from __future__ import annotations

from channel_heads.models.xgboost import (
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
    "verify_model_feature_order",
    "verify_feature_matrix",
    "predict_with_threshold",
]
