"""Evaluation helpers: thresholding, classification metrics, grouped splitting.

Shared by the regime threshold-retune step and the diagnostics notebooks. See
:mod:`channel_heads.eval.metrics` and :mod:`channel_heads.eval.splitting`.
"""

from __future__ import annotations

from .metrics import (
    classification_metrics,
    f1_optimal_threshold,
    max_precision_threshold,
)
from .splitting import (
    RANDOM_STATE,
    TEST_SIZE,
    leave_one_group_out_oof,
    outlet_group_holdout,
)

__all__ = [
    "RANDOM_STATE",
    "TEST_SIZE",
    "classification_metrics",
    "f1_optimal_threshold",
    "leave_one_group_out_oof",
    "max_precision_threshold",
    "outlet_group_holdout",
]
