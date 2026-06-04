#!/usr/bin/env python
"""Step 5 (Mars calibration) — Train the per-regime combined
geom + CNN-embedding XGBoost.

A simplified replay of ``scripts/train_combined_xgb_phase6b.py`` that
trains ONLY the ``geom_plus_cnn_emb`` variant (geom 5 + emb 4 = 9
features), using the regime-specific CNN and the regime-specific
manifest:

  - Reads ``data/results/raster_manifest_<regime>.csv`` (from Step 3).
  - Extracts emb_0..emb_3 from ``models/cnn_outlet_<regime>.pt``
    (from Step 4) over every Earth pair with a valid raster_path.
  - GroupShuffleSplit (test_size=0.2, seed=42, groups=basin__outlet) —
    same protocol as nb02 cell 24 / phase 6B.
  - PR-curve threshold tuning (max precision at recall >= 0.50) — same as
    phase 6B.

Outputs:
  models/xgb_geom_plus_cnn_emb_<regime>.json
  models/feature_columns_geom_plus_cnn_emb_<regime>.txt
  models/optimal_threshold_geom_plus_cnn_emb_<regime>.txt
  models/xgb_geom_plus_cnn_emb_<regime>_metrics.csv
  data/results/master_dataset_<regime>_with_emb.csv

Production ``models/xgb_geom_plus_cnn_emb.json`` and the Phase 6B
``master_dataset_v4_cnn_full.csv`` are not touched.

Run::

    python scripts/train_combined_xgb_regime.py --regime regA
    python scripts/train_combined_xgb_regime.py --regime regB
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from channel_heads.io.paths import PROJECT_ROOT, RESULTS_DIR
from channel_heads.models.cnn import DEFAULT_EMBEDDING_DIM
from channel_heads.models.device import pick_device
from channel_heads.regimes import REGIMES
from channel_heads.training import xgboost as xgb_training
from channel_heads.training.datasets import (
    EMB_FEATURES,
    GEOM_FEATURES,
    GEOM_PLUS_EMB,
    load_valid_raster_manifest,
)

log = logging.getLogger("train_combined_xgb_regime")

N_ESTIMATORS = xgb_training.N_ESTIMATORS
MAX_DEPTH = xgb_training.MAX_DEPTH
LEARNING_RATE = xgb_training.LEARNING_RATE
RANDOM_STATE = xgb_training.RANDOM_STATE
TEST_SIZE = xgb_training.TEST_SIZE
THRESHOLD_MIN_RECALL = xgb_training.THRESHOLD_MIN_RECALL
BATCH_SIZE = xgb_training.BATCH_SIZE


def extract_emb(model_path: Path, paths: list[Path], device: str) -> np.ndarray:
    """Run the regime CNN in eval mode over all rasters; return (N, 4) embeddings."""
    return xgb_training.extract_emb_strict(
        model_path, paths, device=device, batch_size=BATCH_SIZE
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", required=True, choices=sorted(REGIMES.keys()))
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    regime = REGIMES[args.regime]
    manifest_csv = RESULTS_DIR / f"raster_manifest_{regime.name}.csv"
    cnn_path = PROJECT_ROOT / "models" / f"cnn_outlet_{regime.name}.pt"
    if not manifest_csv.exists():
        log.error("Missing manifest: %s — run Step 3.", manifest_csv)
        return 1
    if not cnn_path.exists():
        log.error("Missing CNN model: %s — run Step 4.", cnn_path)
        return 1

    df = load_valid_raster_manifest(manifest_csv)

    required = GEOM_FEATURES + ["y", "basin", "outlet", "raster_path"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error("Manifest missing columns: %s", missing)
        return 1

    device = pick_device()
    log.info("Device: %s | Extracting embeddings on %d rasters ...", device, len(df))
    emb = extract_emb(cnn_path, [Path(p) for p in df["raster_path"]], device=device)
    if emb.shape[0] != len(df):
        log.error("Embedding row count %d != manifest rows %d", emb.shape[0], len(df))
        return 1
    for i in range(DEFAULT_EMBEDDING_DIM):
        df[f"emb_{i}"] = emb[:, i]

    # Sanity: embeddings must be finite (BatchNorm + ReLU should guarantee this).
    for col in EMB_FEATURES:
        arr = df[col].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            log.error(
                "Non-finite values in %s: nan=%d inf=%d",
                col,
                int(np.isnan(arr).sum()),
                int(np.isinf(arr).sum()),
            )
            return 1

    full_csv = RESULTS_DIR / f"master_dataset_{regime.name}_with_emb.csv"
    df.to_csv(full_csv, index=False)
    log.info("Wrote dataset with embeddings -> %s", full_csv)

    # GroupShuffleSplit by basin__outlet (same as Phase 6B / nb02 cell 24).
    df["outlet_group"] = df["basin"].astype(str) + "__" + df["outlet"].astype(str)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y = df["y"].astype(int).to_numpy()
    train_idx, test_idx = next(gss.split(df, y, groups=df["outlet_group"]))
    log.info(
        "Split: %d train rows, %d test rows; %d/%d outlets in test",
        len(train_idx),
        len(test_idx),
        df.iloc[test_idx]["outlet_group"].nunique(),
        df["outlet_group"].nunique(),
    )

    feats = GEOM_PLUS_EMB
    X = df[feats].to_numpy()
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    spw = n_neg / max(n_pos, 1)
    log.info(
        "Train: n=%d (pos=%d, spw=%.3f) | Test: n=%d (pos_frac=%.3f)",
        len(X_train),
        n_pos,
        spw,
        len(X_test),
        float(y_test.mean()),
    )

    model, metrics = xgb_training.train_combined_variant(
        "regime",
        regime.name,
        feats,
        X_train,
        y_train,
        X_test,
        y_test,
        threshold_min_recall=THRESHOLD_MIN_RECALL,
    )
    if metrics["threshold_source"] == "fallback_default_0.5":
        log.warning(
            "No PR-curve point with recall >= %.2f; using 0.5", THRESHOLD_MIN_RECALL
        )
    opt_threshold = metrics["optimal_threshold"]

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    # Match the Phase 6B naming convention: bare `<variant>_<regime>` slug,
    # `xgb_` prefix only on the model JSON itself.
    slug = f"geom_plus_cnn_emb_{regime.name}"
    model_path = models_dir / f"xgb_{slug}.json"
    feature_path = models_dir / f"feature_columns_{slug}.txt"
    threshold_path = models_dir / f"optimal_threshold_{slug}.txt"
    metrics_csv = models_dir / f"xgb_{slug}_metrics.csv"

    model.save_model(str(model_path))
    xgb_training.write_feature_columns(feature_path, feats)
    xgb_training.write_threshold(threshold_path, metrics["optimal_threshold"])
    pd.DataFrame([metrics]).to_csv(metrics_csv, index=False)

    log.info("Saved model      -> %s", model_path)
    log.info("Saved features   -> %s", feature_path)
    log.info("Saved threshold  -> %s  (=%.6f)", threshold_path, opt_threshold)
    log.info("Saved metrics    -> %s", metrics_csv)
    log.info(
        "Test: ROC AUC=%.3f  PR AUC=%.3f  | tuned: P=%.3f R=%.3f F1=%.3f Acc=%.3f",
        metrics["roc_auc_test"],
        metrics["pr_auc_test"],
        metrics["precision_tuned"],
        metrics["recall_tuned"],
        metrics["f1_tuned"],
        metrics["accuracy_tuned"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
