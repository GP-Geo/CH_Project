"""Tabular IO helpers (parquet / csv).

The Mars and training scripts all wrote each table twice (``.parquet`` for the
pipeline, ``.csv`` for humans) with copy-pasted boilerplate. These helpers make
that a one-liner and standardise on "parquet is canonical, csv is a sidecar".
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    also_csv: bool = True,
    index: bool = False,
) -> Path:
    """Write ``df`` to ``path`` (``.parquet`` or ``.csv``).

    When the target is parquet and ``also_csv`` is true, a sibling ``.csv`` is
    written alongside it (the long-standing project convention). Parent
    directories are created. Returns the primary path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=index)
        if also_csv:
            df.to_csv(path.with_suffix(".csv"), index=index)
    elif path.suffix == ".csv":
        df.to_csv(path, index=index)
    else:
        raise ValueError(f"Unsupported table suffix: {path.suffix!r} ({path})")
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a ``.parquet`` or ``.csv`` table, dispatching on the suffix."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table suffix: {path.suffix!r} ({path})")


def read_table_preferring_parquet(stem: str | Path) -> pd.DataFrame:
    """Read ``<stem>.parquet`` if present, else ``<stem>.csv``.

    ``stem`` may be given with or without a suffix; only the path stem is used.
    """
    stem = Path(stem)
    base = stem.with_suffix("")
    parquet = base.with_suffix(".parquet")
    csv = base.with_suffix(".csv")
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Neither {parquet} nor {csv} exists")


__all__ = ["write_table", "read_table", "read_table_preferring_parquet"]
