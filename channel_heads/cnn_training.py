"""Compatibility shim: CNN training core moved to ``channel_heads.training.cnn``.

The canonical home of the shared OutletCNN training loop (:func:`train_cnn`),
the hyperparameter defaults (``DEFAULT_*``, ``HOLDOUT_BASIN``, ``RANDOM_STATE``),
and the re-exported ``pick_device`` is now :mod:`channel_heads.training.cnn`.
This module re-exports them so historical import paths keep working unchanged::

    from channel_heads.cnn_training import train_cnn, pick_device
    from channel_heads.cnn_training import DEFAULT_EPOCHS, HOLDOUT_BASIN

New code should import from :mod:`channel_heads.training.cnn`.
"""

from __future__ import annotations

from channel_heads.training.cnn import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    HOLDOUT_BASIN,
    RANDOM_STATE,
    pick_device,
    train_cnn,
)

__all__ = [
    "train_cnn",
    "pick_device",
    "DEFAULT_EPOCHS",
    "DEFAULT_LR",
    "DEFAULT_WEIGHT_DECAY",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DROPOUT",
    "DEFAULT_PATIENCE",
    "HOLDOUT_BASIN",
    "RANDOM_STATE",
]
