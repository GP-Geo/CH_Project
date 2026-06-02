"""Outlet-pair CNN: architecture, dataset, and training loop.

Curated surface over :mod:`channel_heads.cnn_model` (the ``OutletCNN`` 5-class
encoder producing a 4-d embedding) and :mod:`channel_heads.cnn_training`
(shared training loop + defaults). Requires PyTorch.
"""

from __future__ import annotations

from channel_heads.cnn_model import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_TARGET_SIZE,
    OutletCNN,
    OutletPairDataset,
    encode_raster_onehot,
)
from channel_heads.cnn_training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    HOLDOUT_BASIN,
    pick_device,
    train_cnn,
)

__all__ = [
    "OutletCNN",
    "OutletPairDataset",
    "encode_raster_onehot",
    "train_cnn",
    "pick_device",
    "DEFAULT_EMBEDDING_DIM",
    "DEFAULT_TARGET_SIZE",
    "DEFAULT_EPOCHS",
    "DEFAULT_LR",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DROPOUT",
    "DEFAULT_PATIENCE",
    "HOLDOUT_BASIN",
]
