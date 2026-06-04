#!/usr/bin/env python
"""Multi-seed CNN training (best-of-N) to reduce single-run variance.

For one config (``baseline`` or ``regA``/``regB``/``regC``) this trains the
OutletCNN ``--seeds`` times with different seeds and keeps the model with the
lowest validation loss, written to the canonical path. The early-stopping
val split and torch RNG are both seeded per run so the seeds are genuinely
different draws.

Run::

    python scripts/train_cnn_multiseed.py --config baseline --seeds 3
    python scripts/train_cnn_multiseed.py --config regA --seeds 3
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
    pick_device,
    train_cnn,
)
from channel_heads.training.datasets import (
    HOLDOUT_BASIN,
    cv_pool,
    deterministic_val_split,
    load_valid_raster_manifest,
)

log = logging.getLogger("train_cnn_multiseed")

# config -> (manifest filename, output model filename)
CONFIGS = {
    "baseline": ("raster_manifest.csv", "cnn_outlet_final.pt"),
    "regA": ("raster_manifest_regA.csv", "cnn_outlet_regA.pt"),
    "regB": ("raster_manifest_regB.csv", "cnn_outlet_regB.pt"),
    "regC": ("raster_manifest_regC.csv", "cnn_outlet_regC.pt"),
}


def load_cv_pool(manifest_csv) -> pd.DataFrame:
    return cv_pool(load_valid_raster_manifest(manifest_csv))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, choices=sorted(CONFIGS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    manifest_name, model_name = CONFIGS[args.config]
    manifest_csv = RESULTS_DIR / manifest_name
    if not manifest_csv.exists():
        log.error("Missing manifest: %s", manifest_csv)
        return 1

    df_cv = load_cv_pool(manifest_csv)
    log.info(
        "[%s] CV pool: %d rows from %d basins (holdout %s)",
        args.config,
        len(df_cv),
        df_cv["basin"].nunique(),
        HOLDOUT_BASIN,
    )
    device = pick_device()
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / model_name

    best_val = float("inf")
    best_state = None
    best_seed = -1
    per_seed = []

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        df_train, df_val = deterministic_val_split(
            df_cv, val_frac=args.val_frac, seed=seed
        )

        model, history = train_cnn(
            df_train,
            df_val,
            n_epochs=args.epochs,
            lr=DEFAULT_LR,
            batch_size=DEFAULT_BATCH_SIZE,
            embedding_dim=DEFAULT_EMBEDDING_DIM,
            dropout=DEFAULT_DROPOUT,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            patience=DEFAULT_PATIENCE,
            device=device,
        )
        seed_best = min(history["val_loss"])
        per_seed.append((seed, seed_best, history["best_epoch"]))
        log.info(
            "[%s] seed=%d best_val=%.4f (epoch %d)",
            args.config,
            seed,
            seed_best,
            history["best_epoch"],
        )
        if seed_best < best_val:
            best_val = seed_best
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_seed = seed

    if best_state is None:
        log.error("No model trained")
        return 1

    torch.save(best_state, model_path)
    log.info(
        "[%s] kept seed=%d best_val=%.4f -> %s  | per-seed: %s",
        args.config,
        best_seed,
        best_val,
        model_path,
        ", ".join(f"s{s}={v:.4f}" for s, v, _ in per_seed),
    )
    pd.DataFrame(per_seed, columns=["seed", "best_val_loss", "best_epoch"]).to_csv(
        models_dir / f"{model_name.replace('.pt', '')}_multiseed.csv", index=False
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
