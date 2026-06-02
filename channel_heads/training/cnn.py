"""Shared OutletCNN training loop, device picker, and hyperparameter defaults.

Canonical home of the CNN training core. Extracted originally from
``scripts/train_cnn_regime.py`` so the regime trainer and the
production/multi-seed trainers (``train_cnn_baseline.py``,
``train_cnn_multiseed.py``) can import the *same* training core from the package
instead of sibling-importing each other via ``sys.path`` hacks.

The defaults mirror ``notebooks/training/04_cnn_embeddings.ipynb`` (cell 1).

The CNN architecture/dataset classes come from :mod:`channel_heads.models.cnn`
and ``pick_device`` from :mod:`channel_heads.models.device` (re-exported here for
backward compatibility). The historical import location
:mod:`channel_heads.cnn_training` re-exports this module's surface as a
compatibility shim. Requires PyTorch.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from channel_heads.models.cnn import OutletCNN, OutletPairDataset

# ``pick_device`` is canonical in :mod:`channel_heads.models.device`; re-exported
# here so ``from channel_heads.cnn_training import pick_device`` (and the
# ``channel_heads.training.cnn`` path) keep working.
from channel_heads.models.device import pick_device  # noqa: F401  (re-export)
from channel_heads.rasterizer import NUM_CLASSES

log = logging.getLogger("channel_heads.cnn_training")

# Mirror nb04 cell 1.
DEFAULT_EPOCHS = 60
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_DROPOUT = 0.3
DEFAULT_PATIENCE = 12
HOLDOUT_BASIN = "taiwan"
RANDOM_STATE = 42


def train_cnn(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    *,
    n_epochs: int,
    lr: float,
    batch_size: int,
    embedding_dim: int,
    dropout: float,
    weight_decay: float,
    patience: int,
    device: str,
) -> tuple[OutletCNN, dict]:
    ds_train = OutletPairDataset(
        df_train["raster_path"].tolist(),
        df_train["y"].values,
        augment=True,
    )
    ds_val = OutletPairDataset(
        df_val["raster_path"].tolist(),
        df_val["y"].values,
        augment=False,
    )
    loader_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = OutletCNN(
        in_channels=NUM_CLASSES,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)

    n_pos = int(df_train["y"].sum())
    n_neg = int(len(df_train) - n_pos)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict = {"train_loss": [], "val_loss": [], "best_epoch": -1}
    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0

    for epoch in range(n_epochs):
        model.train()
        train_losses: list[float] = []
        for images, labels in loader_train:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images).squeeze(1), labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for images, labels in loader_val:
                images, labels = images.to(device), labels.to(device)
                val_losses.append(
                    criterion(model(images).squeeze(1), labels).item()
                )

        avg_train = float(np.mean(train_losses)) if train_losses else float("nan")
        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == n_epochs - 1 or patience_counter >= patience:
            log.info(
                "epoch %3d/%d  train=%.4f  val=%.4f%s",
                epoch + 1,
                n_epochs,
                avg_train,
                avg_val,
                "  <best>" if patience_counter == 0 else "",
            )

        if patience_counter >= patience:
            log.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch + 1,
                patience,
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, history


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
