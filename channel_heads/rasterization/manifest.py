"""Mars CNN patch-index (manifest) schema + validation.

One row per model-ready pair describing the patch written for it (path, shape,
dtype, 5-class encoding, structural-QA status/flags).
"""

from __future__ import annotations

import pandas as pd

from channel_heads.rasterization.schema import PATCH_FLAG_COLUMNS

# Full manifest schema, in order.
PATCH_INDEX_COLUMNS = [
    "network_id",
    "pair_id",
    "head_id_1",
    "head_id_2",
    "head_node_id_1",
    "head_node_id_2",
    "confluence_id",
    "confluence_node_id",
    "patch_path",
    "patch_shape",
    "patch_dtype",
    "patch_encoding",
    "patch_status",
    "patch_qa_reason",
    *PATCH_FLAG_COLUMNS,
]

PATCH_STATUSES = {"ok", "invalid", "failed", "skipped"}


def empty_flags() -> dict[str, bool]:
    """All structural-QA flags False (used for skipped/failed rows)."""
    return {k: False for k in PATCH_FLAG_COLUMNS}


def build_patch_manifest(rows: list[dict]) -> pd.DataFrame:
    """Assemble index rows into a manifest DataFrame with the canonical columns."""
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=PATCH_INDEX_COLUMNS)
    return df[[c for c in PATCH_INDEX_COLUMNS if c in df.columns]]


def validate_patch_manifest(df: pd.DataFrame, *, target_size: int = 128) -> None:
    """Raise if the manifest is missing columns or has an unexpected schema."""
    missing = [c for c in PATCH_INDEX_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Patch manifest missing columns: {missing}")
    if df.empty:
        return
    bad_status = set(df["patch_status"].unique()) - PATCH_STATUSES
    if bad_status:
        raise ValueError(f"Unexpected patch_status values: {bad_status}")
    bad_shape = set(df["patch_shape"].unique()) - {f"{target_size},{target_size}"}
    if bad_shape:
        raise ValueError(f"Unexpected patch_shape values: {bad_shape}")
    bad_dtype = set(df["patch_dtype"].unique()) - {"uint8"}
    if bad_dtype:
        raise ValueError(f"Unexpected patch_dtype values: {bad_dtype}")


__all__ = [
    "PATCH_INDEX_COLUMNS",
    "PATCH_FLAG_COLUMNS",
    "PATCH_STATUSES",
    "empty_flags",
    "build_patch_manifest",
    "validate_patch_manifest",
]
