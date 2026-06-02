"""Compatibility shim: CNN architecture/dataset moved to ``channel_heads.models.cnn``.

The canonical home of the :class:`OutletCNN` architecture, the
:class:`OutletPairDataset` loader, and the :func:`encode_raster_onehot`
preprocessing helper is now :mod:`channel_heads.models.cnn`. This module
re-exports them so historical import paths keep working unchanged::

    from channel_heads.cnn_model import OutletCNN, OutletPairDataset
    from channel_heads.cnn_model import DEFAULT_EMBEDDING_DIM, encode_raster_onehot

``NUM_CLASSES`` is re-exported here as well, matching this module's historical
namespace.
"""

from __future__ import annotations

from channel_heads.models.cnn import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_TARGET_SIZE,
    NUM_CLASSES,
    OutletCNN,
    OutletPairDataset,
    encode_raster_onehot,
)

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_TARGET_SIZE",
    "NUM_CLASSES",
    "OutletCNN",
    "OutletPairDataset",
    "encode_raster_onehot",
]
