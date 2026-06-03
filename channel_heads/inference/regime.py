"""Compatibility shim — regime Mars-inference helpers moved to ``channel_heads.models.regime``.

The canonical implementation now lives in :mod:`channel_heads.models.regime`.
This module is preserved so existing imports
(``channel_heads.inference.regime`` and
``from channel_heads.inference.regime import attach_regime_embeddings``),
``scripts/run_mars_combined_regime.py``, and ``notebooks/regime/`` keep working
unchanged. New code should import from :mod:`channel_heads.models.regime`.

The strict state-dict load + finite-value checks (distinct from the lenient
:func:`channel_heads.models.cnn_features.extract_embeddings`) are unchanged.
"""

from __future__ import annotations

from channel_heads.models.regime import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EMBEDDING_DIM,
    attach_regime_embeddings,
    extract_regime_embeddings,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EMBEDDING_DIM",
    "extract_regime_embeddings",
    "attach_regime_embeddings",
]
