"""Inference helpers — canonical implementations live in ``channel_heads.models``.

Re-exported here for any remaining external callers. New code should import
directly from :mod:`channel_heads.models.xgboost` and
:mod:`channel_heads.models.device`.
"""

from __future__ import annotations

from channel_heads.models.device import pick_device
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
    "pick_device",
    "predict_with_threshold",
    "verify_feature_matrix",
    "verify_model_feature_order",
]
