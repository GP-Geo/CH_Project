"""Outlet-pair CNN: architecture, dataset, and one-hot encoding.

Canonical home of the :class:`OutletCNN` 5-class encoder (producing a 4-d
embedding), the :class:`OutletPairDataset` patch loader, and the
:func:`encode_raster_onehot` preprocessing helper. The trained artifact
``models/cnn_outlet_final.pt`` is keyed to this exact architecture, so the class
definitions here are behavior-preserving (state-dict key names, dimensions,
constructor defaults, and forward shapes are unchanged from the historical
``channel_heads.cnn_model`` module).

The shared training loop and hyperparameter defaults live in
:mod:`channel_heads.training.cnn` and are re-exported from this module's curated
surface *lazily* (via module ``__getattr__``). The lazy re-export keeps this
module importable without pulling in :mod:`channel_heads.training.cnn`, which
imports back from this module (``training.cnn`` → ``models.cnn``) and would
otherwise create an import cycle.

The historical import location :mod:`channel_heads.cnn_model` re-exports the
architecture/dataset symbols from here as a compatibility shim. Requires
PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import Dataset

from channel_heads.rasterization.schema import NUM_CLASSES

if TYPE_CHECKING:
    # Re-exported lazily at runtime via module ``__getattr__`` (see below); the
    # static import here only exists so type checkers / linters see these names
    # as part of this module's public surface. It is never executed at runtime,
    # so it does not reintroduce the training.cnn import cycle.
    from channel_heads.training.cnn import (
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

# Default dimensions
DEFAULT_EMBEDDING_DIM = 4
DEFAULT_TARGET_SIZE = 128


# =============================================================================
# One-Hot Encoding
# =============================================================================


def encode_raster_onehot(raster: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """Convert categorical raster to one-hot encoded channels.

    Parameters
    ----------
    raster : np.ndarray
        Array of shape (H, W) with integer values in {0, 1, 2, 3, 4}.

    Returns
    -------
    np.ndarray
        Array of shape (NUM_CLASSES, H, W) with float32 values in {0.0, 1.0}.
    """
    h, w = raster.shape
    onehot = np.zeros((NUM_CLASSES, h, w), dtype=np.float32)
    for c in range(NUM_CLASSES):
        onehot[c] = (raster == c).astype(np.float32)
    return onehot


# =============================================================================
# CNN Architecture
# =============================================================================


class OutletCNN(nn.Module):
    """Shallow CNN for outlet stream network spatial feature extraction.

    Architecture:
        Conv(in→16, 3×3) + BN + ReLU + MaxPool(2)
        Conv(16→32, 3×3) + BN + ReLU + MaxPool(2)
        Conv(32→64, 3×3) + BN + ReLU + MaxPool(2)
        AdaptiveAvgPool2d(1)
        Linear(64→embedding_dim) + ReLU + Dropout
        Linear(embedding_dim→1)  [classification head]

    Parameters
    ----------
    in_channels : int
        Number of input channels. Default is NUM_CLASSES (5) for one-hot.
    embedding_dim : int
        Dimensionality of the embedding layer. Default 4.
    dropout : float
        Dropout probability after embedding layer. Default 0.3.
    """

    def __init__(
        self,
        in_channels: int = NUM_CLASSES,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Conv block 1: in_channels → 16
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2)

        # Conv block 2: 16 → 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2)

        # Conv block 3: 32 → 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2)

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Embedding layer
        self.fc_embed = nn.Linear(64, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        # Classification head (for pre-training)
        self.fc_head = nn.Linear(embedding_dim, 1)

    def _conv_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract convolutional features before embedding."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # flatten to (B, 64)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logit.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, 1).
        """
        features = self._conv_features(x)
        embedding = self.dropout(F.relu(self.fc_embed(features)))
        logit = self.fc_head(embedding)
        return logit

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning the embedding vector.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, in_channels, H, W).

        Returns
        -------
        torch.Tensor
            Embedding of shape (B, embedding_dim).
        """
        features = self._conv_features(x)
        embedding = F.relu(self.fc_embed(features))
        return embedding


# =============================================================================
# Dataset
# =============================================================================


class OutletPairDataset(Dataset):
    """PyTorch Dataset for pre-rendered outlet pair raster patches.

    Parameters
    ----------
    raster_paths : list[str | Path]
        Paths to .npy raster files.
    labels : list[int] | np.ndarray
        Binary labels (0 or 1) for each sample.
    augment : bool
        Whether to apply data augmentation (random flips). Default False.
    """

    def __init__(
        self,
        raster_paths: list[str | Path],
        labels: list[int] | npt.NDArray[np.int_],
        augment: bool = False,
    ):
        self.raster_paths = [Path(p) for p in raster_paths]
        self.labels = np.asarray(labels, dtype=np.float32)
        self.augment = augment

        if len(self.raster_paths) != len(self.labels):
            raise ValueError(
                f"Length mismatch: {len(self.raster_paths)} paths vs " f"{len(self.labels)} labels"
            )

    def __len__(self) -> int:
        return len(self.raster_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load and return a single sample.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (image, label) where image has shape (NUM_CLASSES, H, W)
            and label is a scalar tensor.
        """
        raster = np.load(self.raster_paths[idx])
        onehot = encode_raster_onehot(raster)

        if self.augment:
            # Random horizontal flip
            if np.random.random() > 0.5:
                onehot = onehot[:, :, ::-1].copy()
            # Random vertical flip
            if np.random.random() > 0.5:
                onehot = onehot[:, ::-1, :].copy()

        image = torch.from_numpy(onehot)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return image, label


# =============================================================================
# Lazy re-export of the shared training core (curated surface)
# =============================================================================

# These symbols live in ``channel_heads.training.cnn``. They are exposed here
# lazily so importing ``channel_heads.models.cnn`` never eagerly imports the
# training module — which imports back from this module, closing an import cycle
# (see module docstring).
_TRAINING_EXPORTS = frozenset(
    {
        "train_cnn",
        "pick_device",
        "DEFAULT_EPOCHS",
        "DEFAULT_LR",
        "DEFAULT_WEIGHT_DECAY",
        "DEFAULT_BATCH_SIZE",
        "DEFAULT_DROPOUT",
        "DEFAULT_PATIENCE",
        "HOLDOUT_BASIN",
    }
)


def __getattr__(name: str):
    if name in _TRAINING_EXPORTS:
        from channel_heads.training import cnn as training_cnn

        return getattr(training_cnn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
