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

from channel_heads.io.paths import PROJECT_ROOT, RESULTS_DIR
from channel_heads.models.cnn import DEFAULT_EMBEDDING_DIM

# Re-use regime presets.
from channel_heads.regimes import REGIMES
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

log = logging.getLogger("train_cnn_regime")


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
