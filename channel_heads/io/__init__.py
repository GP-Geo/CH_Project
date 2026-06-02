"""IO layer: paths, tabular IO, GeoPackage IO, and generated-data cleanup.

This is the single home for "where do files live" and "how do we read/write
them". Pipeline and notebook code should import from here rather than hardcoding
paths or repeating parquet/csv/geopackage boilerplate.
"""

from channel_heads.io import cleanup, geopackage, paths, tables
from channel_heads.io.geopackage import read_gpkg, write_gpkg
from channel_heads.io.tables import (
    read_table,
    read_table_preferring_parquet,
    write_table,
)

__all__ = [
    "paths",
    "tables",
    "geopackage",
    "cleanup",
    "read_table",
    "read_table_preferring_parquet",
    "write_table",
    "read_gpkg",
    "write_gpkg",
]
