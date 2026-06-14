"""Regime Mars-inference helpers: attach regime-CNN embeddings to the table.

The per-regime Mars inference step overrides the baseline ``emb_0..emb_N``
columns with embeddings from that regime's CNN, then runs the regime's combined
XGBoost. The embedding-extraction + patch-index merge glue was inline in the
``run-mars-combined-regime`` command; it now lives here so that command and
``notebooks/archive/regime/`` call the same implementation.

This is the canonical home for the regime inference helpers;
:mod:`channel_heads.inference.regime` re-exports them as a compatibility shim.

This is a deliberate, behavior-preserving move: ``extract_regime_embeddings``
keeps the *strict* state-dict load + finite-value checks the regime pipeline
relies on (distinct from the lenient
:func:`channel_heads.models.cnn_features.extract_embeddings`, which returns a
manifest-keyed DataFrame). Module globals in the original script become explicit
parameters (``patch_index_path``, ``project_root``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from channel_heads.models.cnn import (
    DEFAULT_EMBEDDING_DIM,
    OutletCNN,
    OutletPairDataset,
)

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64


def extract_regime_embeddings(
    model_path: Path,
    patch_paths: list[Path],
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> np.ndarray:
    """Run a regime CNN over patches and return the embedding matrix.

    Strict state-dict load (raises on any missing/unexpected key), eval mode,
    no augmentation — identical to the regime pipeline's forward pass.
    """
    model = OutletCNN(embedding_dim=embedding_dim)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch: missing={missing} unexpected={unexpected}")
    model.to(device).eval()
    dummy = np.zeros(len(patch_paths), dtype=np.float32)
    ds = OutletPairDataset(patch_paths, dummy, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            out.append(model.embed(images.to(device)).cpu().numpy())
    return np.vstack(out)


def attach_regime_embeddings(
    df_in: pd.DataFrame,
    cnn_model_path: Path,
    patch_index_path: Path,
    project_root: Path,
    device: str,
    batch_size: int = DEFAULT_BATCH_SIZE,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> pd.DataFrame:
    """Replace ``emb_0..emb_N`` in ``df_in`` with embeddings from the regime CNN.

    Maps each Mars ``pair_id`` to its existing patch path via the CNN patch
    index, drops pairs without a patch on disk, runs the regime CNN, and writes
    the embedding columns back. Raises if any embedding is non-finite.
    """
    idx = pd.read_parquet(patch_index_path)
    idx = idx[idx["patch_status"] == "ok"][["pair_id", "patch_path"]].copy()
    idx["patch_path_abs"] = idx["patch_path"].apply(
        lambda p: str(project_root / p) if not Path(p).is_absolute() else p
    )

    df = df_in.merge(idx[["pair_id", "patch_path_abs"]], on="pair_id", how="left")
    missing = int(df["patch_path_abs"].isna().sum())
    if missing:
        logger.warning("Dropping %d pairs without a patch on disk", missing)
        df = df[df["patch_path_abs"].notna()].copy().reset_index(drop=True)

    logger.info("Extracting regime embeddings on %d patches ...", len(df))
    emb = extract_regime_embeddings(
        cnn_model_path,
        [Path(p) for p in df["patch_path_abs"]],
        device,
        batch_size=batch_size,
        embedding_dim=embedding_dim,
    )

    for i in range(embedding_dim):
        df[f"emb_{i}"] = emb[:, i]

    for col in [f"emb_{i}" for i in range(embedding_dim)]:
        arr = df[col].to_numpy(dtype=float)
        if not np.isfinite(arr).all():
            raise RuntimeError(
                f"Non-finite values in {col}: nan={int(np.isnan(arr).sum())} "
                f"inf={int(np.isinf(arr).sum())}"
            )

    df = df.drop(columns=["patch_path_abs"])
    return df


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_EMBEDDING_DIM",
    "extract_regime_embeddings",
    "attach_regime_embeddings",
]
