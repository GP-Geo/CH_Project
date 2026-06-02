"""Compatibility shim: CNN embedding helpers moved to ``channel_heads.models.cnn_features``.

The canonical home of the generic/Earth CNN embedding helpers
(:func:`extract_embeddings`, :func:`merge_cnn_features`) and the
``CNN_FEATURE_COLS`` constant is now :mod:`channel_heads.models.cnn_features`.
This module re-exports them so historical import paths keep working unchanged::

    from channel_heads.cnn_features import extract_embeddings, CNN_FEATURE_COLS
    from channel_heads.cnn_features import merge_cnn_features

New code should import from :mod:`channel_heads.models.cnn_features`.
"""

from __future__ import annotations

from channel_heads.models.cnn_features import (
    CNN_FEATURE_COLS,
    extract_embeddings,
    merge_cnn_features,
)

__all__ = [
    "CNN_FEATURE_COLS",
    "extract_embeddings",
    "merge_cnn_features",
]
