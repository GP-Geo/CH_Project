"""GeoPackage IO helpers (thin geopandas wrappers).

Centralises the ``gpd.read_file`` / ``GeoDataFrame.to_file(driver="GPKG")``
boilerplate that appears ~75 times across the Mars scripts, and keeps the
geopandas import lazy so importing :mod:`channel_heads.io` never requires the
geo stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import geopandas as gpd


def read_gpkg(path: str | Path, layer: str | None = None) -> "gpd.GeoDataFrame":
    """Read a GeoPackage (optionally a named layer)."""
    import geopandas as gpd

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GeoPackage not found: {path}")
    return gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)


def write_gpkg(
    gdf: "gpd.GeoDataFrame",
    path: str | Path,
    layer: str | None = None,
) -> Path:
    """Write a GeoDataFrame to a GeoPackage, creating parent dirs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if layer:
        gdf.to_file(path, layer=layer, driver="GPKG")
    else:
        gdf.to_file(path, driver="GPKG")
    return path


__all__ = ["read_gpkg", "write_gpkg"]
