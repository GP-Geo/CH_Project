"""Per-fold CNN model factory for *leakage-free* leave-one-basin-out (torch).

The model-agnostic LOBO engine in :mod:`channel_heads.eval.lobo` is torch-free.
This module supplies the one heavy ``ModelFactory`` that closes the CNN-embedding
leakage path: instead of reading precomputed ``emb_*`` (produced by a CNN that
saw 16/17 basins), it **retrains a fresh CNN inside each fold on the training
basins only**, then embeds whatever frame it is asked to score. Patches already
exist on disk (``raster_path``), so no re-rasterization is needed.

Cost: one CNN train per outer fold (+ one more for the inner threshold split) —
heavy, but the only honest estimate of the geom+CNN model under LOBO.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from xgboost import XGBClassifier

from channel_heads.eval.lobo import EMB, GEOM, FoldModel, ModelFactory
from channel_heads.models.cnn import DEFAULT_EMBEDDING_DIM, OutletCNN, OutletPairDataset
from channel_heads.models.device import pick_device
from channel_heads.training.cnn import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DROPOUT,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_PATIENCE,
    DEFAULT_WEIGHT_DECAY,
    train_cnn,
)

log = logging.getLogger("channel_heads.eval.lobo_cnn")

GEOM_PLUS_EMB = GEOM + EMB


def _embed(model: OutletCNN, raster_paths, device: str, batch_size: int) -> np.ndarray:
    """Eval-mode embeddings ``(N, embedding_dim)`` for the given raster patches."""
    dummy = np.zeros(len(raster_paths), dtype=np.float32)
    loader = DataLoader(
        OutletPairDataset(list(raster_paths), dummy, augment=False),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            out.append(model.embed(images.to(device)).cpu().numpy())
    return np.vstack(out)


class _CNNFoldModel:
    """Fold model = (per-fold CNN) -> embeddings -> (per-fold XGBoost)."""

    def __init__(self, cnn: OutletCNN, xgb: XGBClassifier, device: str, batch_size: int) -> None:
        self._cnn = cnn
        self._xgb = xgb
        self._device = device
        self._batch_size = batch_size

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        emb = _embed(self._cnn, frame["raster_path"], self._device, self._batch_size)
        X = np.hstack([frame[GEOM].to_numpy(dtype=float), emb])
        return self._xgb.predict_proba(X)[:, 1]


def make_per_fold_cnn_factory(
    *,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    lr: float = DEFAULT_LR,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    val_basin_frac: float = 0.15,
    seed: int = 42,
    n_estimators: int = 200,
    max_depth: int = 4,
    xgb_lr: float = 0.1,
    device: str | None = None,
) -> ModelFactory:
    """Build a :data:`~channel_heads.eval.lobo.ModelFactory` that retrains the CNN.

    The returned factory, given a training frame (already free of the held-out
    basin by the engine), trains a CNN on a *basin-grouped* inner split (so no
    outlet leaks between CNN train/val), embeds the training rows with it, and
    fits the geom+emb XGBoost. The resulting :class:`_CNNFoldModel` re-embeds any
    frame it is asked to score with that same fold CNN.
    """
    device = device or pick_device()

    def factory(df_train: pd.DataFrame) -> FoldModel:
        groups = df_train["basin"].astype(str)
        if groups.nunique() >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_basin_frac, random_state=seed)
            itr, iva = next(gss.split(df_train, df_train["y"], groups=groups))
            d_tr, d_va = df_train.iloc[itr], df_train.iloc[iva]
        else:  # degenerate (single training basin) — random early-stopping split
            d_va = df_train.sample(frac=0.1, random_state=seed)
            d_tr = df_train.drop(index=d_va.index)

        cnn, _ = train_cnn(
            d_tr,
            d_va,
            n_epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
            dropout=dropout,
            weight_decay=weight_decay,
            patience=patience,
            device=device,
        )

        emb = _embed(cnn, df_train["raster_path"], device, batch_size)
        X = np.hstack([df_train[GEOM].to_numpy(dtype=float), emb])
        y = df_train["y"].astype(int).to_numpy()
        spw = (y == 0).sum() / max((y == 1).sum(), 1)
        xgb = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=xgb_lr,
            scale_pos_weight=spw,
            eval_metric="logloss",
            random_state=seed,
        )
        xgb.fit(X, y)
        return _CNNFoldModel(cnn, xgb, device, batch_size)

    return factory


__all__ = ["make_per_fold_cnn_factory", "GEOM_PLUS_EMB"]
