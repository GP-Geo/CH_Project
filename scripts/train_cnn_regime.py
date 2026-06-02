#!/usr/bin/env python
"""Step 4 (Mars calibration) — Train the per-regime OutletCNN.

Mirrors the training loop in ``notebooks/training/04_cnn_embeddings.ipynb``
(cells 5-6) with three minor changes:

  - Reads the regime-specific manifest
    ``data/results/raster_manifest_<regime>.csv`` (from Step 3).
  - Holds out Taiwan from training (LOBO-style), splits the remaining
    16 basins 90/10 for early stopping. Same hyperparameters as nb04.
  - Saves the trained model to ``models/cnn_outlet_<regime>.pt`` without
    touching the production ``models/cnn_outlet_final.pt``.

Run::

    python scripts/train_cnn_regime.py --regime regA
    python scripts/train_cnn_regime.py --regime regB --epochs 40
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from channel_heads.cnn_model import (
    DEFAULT_EMBEDDING_DIM,
    OutletCNN,
    OutletPairDataset,
)
from channel_heads.config import PROJECT_ROOT, RESULTS_DIR
from channel_heads.rasterizer import NUM_CLASSES

# Re-use regime presets.
from channel_heads.regimes import REGIMES

log = logging.getLogger("train_cnn_regime")

# Mirror nb04 cell 1.
DEFAULT_EPOCHS = 60
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_DROPOUT = 0.3
DEFAULT_PATIENCE = 12
HOLDOUT_BASIN = "taiwan"
RANDOM_STATE = 42


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=sorted(REGIMES.keys()))
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument(
        "--val-frac", type=float, default=0.1, help="Fraction of CV pool for early stopping."
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    regime = REGIMES[args.regime]
    manifest_csv = RESULTS_DIR / f"raster_manifest_{regime.name}.csv"
    if not manifest_csv.exists():
        log.error("Missing manifest: %s — run Step 3 first.", manifest_csv)
        return 1

    df = pd.read_csv(manifest_csv)
    valid_mask = df["raster_path"].notna()
    if "raster_status" in df.columns:
        valid_mask &= df["raster_status"].eq("ok")
    df = df[valid_mask].copy().reset_index(drop=True)
    log.info(
        "Manifest: %d rows with rasters, %d basins; y=%d/%d touching",
        len(df),
        df["basin"].nunique(),
        int(df["y"].sum()),
        len(df),
    )

    # LOBO holdout: Taiwan never enters training/val. If Taiwan is absent
    # (e.g., regime dropped it), proceed with all available basins.
    df_cv = df[df["basin"] != HOLDOUT_BASIN].copy().reset_index(drop=True)
    n_holdout = int((df["basin"] == HOLDOUT_BASIN).sum())
    log.info(
        "CV pool: %d rows from %d basins (holdout %s: %d rows)",
        len(df_cv),
        df_cv["basin"].nunique(),
        HOLDOUT_BASIN,
        n_holdout,
    )

    if len(df_cv) < 20:
        log.error("CV pool too small (%d rows) — aborting.", len(df_cv))
        return 1

    val_size = max(int(len(df_cv) * args.val_frac), 10)
    perm = np.random.default_rng(RANDOM_STATE).permutation(len(df_cv))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    df_train = df_cv.iloc[train_idx].reset_index(drop=True)
    df_val = df_cv.iloc[val_idx].reset_index(drop=True)
    log.info(
        "Train: %d (%.1f%% touching)  Val: %d (%.1f%% touching)",
        len(df_train),
        100.0 * df_train["y"].mean(),
        len(df_val),
        100.0 * df_val["y"].mean(),
    )

    device = pick_device()
    log.info("Device: %s", device)

    model, history = train_cnn(
        df_train,
        df_val,
        n_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        embedding_dim=DEFAULT_EMBEDDING_DIM,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        patience=args.patience,
        device=device,
    )

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"cnn_outlet_{regime.name}.pt"
    history_path = models_dir / f"cnn_outlet_{regime.name}_history.csv"

    torch.save(model.state_dict(), model_path)
    log.info("Saved CNN -> %s (best_epoch=%d)", model_path, history["best_epoch"])

    pd.DataFrame(
        {
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
        }
    ).to_csv(history_path, index=False)
    log.info("Saved training history -> %s", history_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
