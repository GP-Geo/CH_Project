"""Operating-threshold selection and classification metrics.

Curated surface over :mod:`channel_heads.eval.metrics`. The Mars *operating*
threshold should be chosen deliberately (precision-oriented / Dd-calibrated),
**not** copied blindly from the Earth F1-optimal threshold — see
``docs/modeling.md``.
"""

from __future__ import annotations

from channel_heads.eval.metrics import (
    classification_metrics,
    f1_optimal_threshold,
    max_precision_threshold,
)


def load_threshold_file(path) -> float:
    """Read a single-float ``optimal_threshold_*.txt`` artifact."""
    from pathlib import Path

    return float(Path(path).read_text().strip())


__all__ = [
    "f1_optimal_threshold",
    "max_precision_threshold",
    "classification_metrics",
    "load_threshold_file",
]
