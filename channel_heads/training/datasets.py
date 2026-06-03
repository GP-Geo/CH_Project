"""Reusable training-dataset helpers for Earth/regime CNN + XGBoost recipes.

Centralizes logic that was duplicated across the Earth/regime training scripts:

* raster-manifest loading and valid-row filtering
  (``raster_path`` present and, if a ``raster_status`` column exists,
  ``raster_status == "ok"``),
* the Taiwan leave-one-basin-out CV pool, and
* the deterministic early-stopping validation split
  (``np.random.default_rng(seed)`` with ``val_size = max(int(n * val_frac), 10)``),

plus the canonical Earth model feature constants and their exact order.

Behaviour is extracted verbatim from ``scripts/train_cnn_baseline.py``,
``scripts/train_cnn_regime.py``, ``scripts/train_cnn_multiseed.py``,
``scripts/train_combined_xgb_phase6b.py``, and
``scripts/train_combined_xgb_regime.py`` so those scripts can call these helpers
without any change in selected rows, split indices, or feature order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from channel_heads.models.cnn import DEFAULT_EMBEDDING_DIM
from channel_heads.training.cnn import HOLDOUT_BASIN, RANDOM_STATE

# =============================================================================
# Feature constants and exact feature order (XGBoost model contract)
# =============================================================================

# Five dimensionless geometric features, in the exact order the combined
# XGBoost models were trained on.
GEOM_FEATURES: list[str] = [
    "orientation_diff_deg",
    "headhead_dist_norm",
    "apex_angle_deg",
    "strahler_order_diff",
    "proximity_profile_norm",
]

# CNN embedding features ``emb_0..emb_{N-1}``.
EMB_FEATURES: list[str] = [f"emb_{i}" for i in range(DEFAULT_EMBEDDING_DIM)]

# Single CNN classifier-logit feature.
CNN_LOGIT_FEATURE: str = "cnn_logit"

# Combined feature lists (geom first, then the CNN feature(s)).
GEOM_PLUS_EMB: list[str] = GEOM_FEATURES + EMB_FEATURES
GEOM_PLUS_LOGIT: list[str] = GEOM_FEATURES + [CNN_LOGIT_FEATURE]


# =============================================================================
# Raster-manifest loading / filtering
# =============================================================================


def filter_valid_raster_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Keep rows with a present ``raster_path`` (and ``raster_status == "ok"``).

    Mirrors the filter used by every CNN/XGBoost training script: a row is valid
    when ``raster_path`` is not null and, *only if* the manifest carries a
    ``raster_status`` column, that status equals ``"ok"``. Returns a copy with a
    reset index.
    """
    valid_mask = df["raster_path"].notna()
    if "raster_status" in df.columns:
        valid_mask &= df["raster_status"].eq("ok")
    return df[valid_mask].copy().reset_index(drop=True)


def load_valid_raster_manifest(manifest_csv: str | Path) -> pd.DataFrame:
    """Read a raster manifest CSV and return only the valid rows.

    Equivalent to ``filter_valid_raster_rows(pd.read_csv(manifest_csv))``.
    """
    return filter_valid_raster_rows(pd.read_csv(manifest_csv))


# =============================================================================
# CV pool / validation split
# =============================================================================


def cv_pool(df: pd.DataFrame, holdout_basin: str = HOLDOUT_BASIN) -> pd.DataFrame:
    """Return the cross-validation pool — all rows except the holdout basin.

    Taiwan (``HOLDOUT_BASIN``) is excluded from training/validation so it can
    serve as an untouched leave-one-basin-out test set. Returns a copy with a
    reset index.
    """
    return df[df["basin"] != holdout_basin].copy().reset_index(drop=True)


def deterministic_val_split(
    df_cv: pd.DataFrame,
    val_frac: float = 0.1,
    seed: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Deterministic early-stopping split of the CV pool into (train, val).

    Reproduces the exact split used by the CNN training scripts:

    * ``val_size = max(int(len(df_cv) * val_frac), 10)``,
    * ``perm = np.random.default_rng(seed).permutation(len(df_cv))``,
    * the first ``val_size`` permuted rows are validation, the rest are train.

    Both returned frames have a reset index. ``seed`` is ``RANDOM_STATE`` for the
    baseline/regime trainers and the per-seed value for the multi-seed trainer.
    """
    val_size = max(int(len(df_cv) * val_frac), 10)
    perm = np.random.default_rng(seed).permutation(len(df_cv))
    df_val = df_cv.iloc[perm[:val_size]].reset_index(drop=True)
    df_train = df_cv.iloc[perm[val_size:]].reset_index(drop=True)
    return df_train, df_val


__all__ = [
    "GEOM_FEATURES",
    "EMB_FEATURES",
    "CNN_LOGIT_FEATURE",
    "GEOM_PLUS_EMB",
    "GEOM_PLUS_LOGIT",
    "HOLDOUT_BASIN",
    "RANDOM_STATE",
    "filter_valid_raster_rows",
    "load_valid_raster_manifest",
    "cv_pool",
    "deterministic_val_split",
]
