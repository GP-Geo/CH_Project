"""CNN embedding extraction and integration with XGBoost features.

This module provides functions to:
1. Extract embeddings from a trained CNN for all pairs in a dataset.
2. Merge CNN embedding columns into existing feature DataFrames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .cnn_model import DEFAULT_EMBEDDING_DIM, OutletCNN, OutletPairDataset
from .logging_config import get_logger

logger = get_logger(__name__)

# Column names for CNN embedding features
CNN_FEATURE_COLS: list[str] = [f"emb_{i}" for i in range(DEFAULT_EMBEDDING_DIM)]

# Merge keys used to join CNN features with other DataFrames
_MERGE_KEYS = ["outlet", "confluence", "head_1", "head_2"]


def extract_embeddings(
    model_path: Path,
    raster_dir: Path | None,
    manifest_df: pd.DataFrame,
    batch_size: int = 64,
    device: str = "cpu",
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> pd.DataFrame:
    """Load a trained CNN and extract embeddings for all pairs.

    Parameters
    ----------
    model_path : Path
        Path to saved CNN state dict (.pt file).
    raster_dir : Path or None
        Base directory for raster files. If None, ``manifest_df`` must have
        an absolute ``raster_path`` column.
    manifest_df : pd.DataFrame
        Must have columns: ``raster_path`` (relative or absolute paths to
        .npy files) and the merge keys (outlet, confluence, head_1, head_2).
    batch_size : int
        Inference batch size.
    device : str
        PyTorch device string (e.g., "cpu", "cuda").
    embedding_dim : int
        Embedding dimension of the trained model.

    Returns
    -------
    pd.DataFrame
        Copy of ``manifest_df`` with added columns ``emb_0`` .. ``emb_{N-1}``.
    """
    # Resolve raster paths
    if raster_dir is not None:
        paths = [raster_dir / p for p in manifest_df["raster_path"]]
    else:
        paths = [Path(p) for p in manifest_df["raster_path"]]

    # Load model
    model = OutletCNN(embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    # Create a dummy-label dataset (labels unused for inference)
    dummy_labels = np.zeros(len(paths), dtype=np.float32)
    ds = OutletPairDataset(paths, dummy_labels, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    # Extract embeddings
    all_embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            emb = model.embed(images)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(all_embeddings)  # (N, embedding_dim)

    # Build output DataFrame
    result = manifest_df.copy()
    emb_cols = [f"emb_{i}" for i in range(embedding_dim)]
    for i, col in enumerate(emb_cols):
        result[col] = embeddings[:, i]

    logger.info(
        "Extracted %d embeddings (dim=%d) from %s",
        len(result),
        embedding_dim,
        model_path,
    )
    return result


def merge_cnn_features(
    base_df: pd.DataFrame,
    embedding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge CNN embedding columns into an existing feature DataFrame.

    Parameters
    ----------
    base_df : pd.DataFrame
        Base DataFrame with geometric/coupling features.
    embedding_df : pd.DataFrame
        DataFrame with CNN embedding columns (emb_0, ..., emb_N).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with CNN features added.
    """
    # Identify embedding columns in the embedding_df
    emb_cols = [c for c in embedding_df.columns if c.startswith("emb_")]
    merge_cols = _MERGE_KEYS + emb_cols

    # Only keep merge keys + embedding columns from embedding_df
    right = embedding_df[merge_cols].copy()

    merged = base_df.merge(right, on=_MERGE_KEYS, how="left")

    n_matched = merged[emb_cols[0]].notna().sum() if emb_cols else 0
    logger.info("Merged CNN features: %d/%d rows matched", n_matched, len(base_df))
    return merged
