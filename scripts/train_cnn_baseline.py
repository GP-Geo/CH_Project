#!/usr/bin/env python
"""Train the production (baseline) OutletCNN -> models/cnn_outlet_final.pt.

Scripted, reproducible equivalent of the training loop in
``notebooks/training/04_cnn_embeddings.ipynb`` (cells 5-6) so the full model
rebuild can run unattended without executing a notebook headless.

Differences from ``scripts/train_cnn_regime.py``:

  - Reads the production manifest ``data/results/raster_manifest.csv``.
  - Saves to ``models/cnn_outlet_final.pt`` (the production model consumed by
    the Mars pipeline and ``train_combined_xgb_phase6b.py``).

Everything else (Taiwan LOBO holdout, 90/10 early-stopping split, hyper-
parameters) is identical, and only ``raster_status == "ok"`` patches are used.

Run::

    python scripts/train_cnn_baseline.py
    python scripts/train_cnn_baseline.py --epochs 40
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import torch

from channel_heads.io.paths import PROJECT_ROOT, RESULTS_DIR
from channel_heads.models.cnn import DEFAULT_EMBEDDING_DIM
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

log = logging.getLogger("train_cnn_baseline")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    manifest_csv = RESULTS_DIR / "raster_manifest.csv"
    if not manifest_csv.exists():
        log.error("Missing manifest: %s", manifest_csv)
        return 1

    df = pd.read_csv(manifest_csv)
    valid_mask = df["raster_path"].notna()
    if "raster_status" in df.columns:
        valid_mask &= df["raster_status"].eq("ok")
    df = df[valid_mask].copy().reset_index(drop=True)
    log.info(
        "Manifest: %d ok rasters, %d basins; y=%d/%d touching",
        len(df),
        df["basin"].nunique(),
        int(df["y"].sum()),
        len(df),
    )

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
    df_val = df_cv.iloc[perm[:val_size]].reset_index(drop=True)
    df_train = df_cv.iloc[perm[val_size:]].reset_index(drop=True)
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
    model_path = models_dir / "cnn_outlet_final.pt"
    history_path = models_dir / "cnn_outlet_final_history.csv"
    torch.save(model.state_dict(), model_path)
    pd.DataFrame(
        {
            "epoch": list(range(1, len(history["train_loss"]) + 1)),
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
        }
    ).to_csv(history_path, index=False)
    log.info("Saved CNN -> %s (best_epoch=%d)", model_path, history["best_epoch"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
