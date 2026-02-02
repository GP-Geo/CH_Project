"""Project configuration and path management.

This module provides centralized path management for the channel-heads project,
supporting both development (local) and installed package usage.

Usage:
    from channel_heads.config import CROPPED_DEMS_DIR, EXAMPLE_DEMS

    # Use predefined paths
    dem_path = EXAMPLE_DEMS["inyo"]

    # Or build paths from directories
    dem_path = CROPPED_DEMS_DIR / "MyCustomDEM.tif"

Environment Variables:
    CHANNEL_HEADS_ROOT: Override the auto-detected project root directory.
    CHANNEL_HEADS_DATA: Override the data directory location.
"""

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Find the project root directory.

    Searches upward from this file for a directory containing pyproject.toml
    or CLAUDE.md, which indicates the project root.

    Returns:
        Path to project root directory.
    """
    # Check environment variable first
    if os.getenv("CHANNEL_HEADS_ROOT"):
        return Path(os.getenv("CHANNEL_HEADS_ROOT"))

    # Start from this file's location
    current = Path(__file__).resolve().parent

    # Search upward for project markers
    markers = ["pyproject.toml", "CLAUDE.md", ".git"]

    for _ in range(5):  # Limit search depth
        for marker in markers:
            if (current / marker).exists():
                return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent

    # Fallback to current working directory
    return Path.cwd()


def _get_data_dir() -> Path:
    """Get the data directory.

    Returns:
        Path to data directory.
    """
    if os.getenv("CHANNEL_HEADS_DATA"):
        return Path(os.getenv("CHANNEL_HEADS_DATA"))
    return PROJECT_ROOT / "data"


# Core paths
PROJECT_ROOT: Path = _find_project_root()
"""Project root directory."""

DATA_DIR: Path = _get_data_dir()
"""Main data directory."""

RAW_DATA_DIR: Path = DATA_DIR / "raw"
"""Raw input data directory (SRTM downloads, etc.)."""

CROPPED_DEMS_DIR: Path = DATA_DIR / "cropped_DEMs"
"""Directory containing cropped study area DEMs."""

OUTPUTS_DIR: Path = DATA_DIR / "outputs"
"""Directory for analysis outputs (CSVs, figures)."""

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
"""Jupyter notebooks directory."""


# Example DEMs with friendly names
EXAMPLE_DEMS: dict[str, Path] = {
    "inyo": CROPPED_DEMS_DIR / "Inyo_strm_crop.tif",
    "humboldt": CROPPED_DEMS_DIR / "Humboldt_strm_crop.tif",
    "calnalpine": CROPPED_DEMS_DIR / "CalnAlpine_strm_crop.tif",
    "daqing": CROPPED_DEMS_DIR / "Daqing_strm_crop.tif",
    "luliang": CROPPED_DEMS_DIR / "Luliang_strm_crop.tif",
    "kammanassie": CROPPED_DEMS_DIR / "Kammanasie_strm_crop.tif",
    "finisterre": CROPPED_DEMS_DIR / "Finisterre_strm_crop.tif",
}
"""Dictionary mapping friendly names to DEM file paths."""


def get_output_dir(study_area: str, create: bool = True) -> Path:
    """Get output directory for a specific study area.

    Parameters
    ----------
    study_area : str
        Name of the study area (e.g., "inyo", "humboldt").
    create : bool
        If True, create the directory if it doesn't exist.

    Returns
    -------
    Path
        Path to the study area's output directory.

    Example
    -------
    >>> output_dir = get_output_dir("inyo")
    >>> results_path = output_dir / "coupling_results.csv"
    """
    output_dir = OUTPUTS_DIR / study_area
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def list_available_dems() -> dict[str, Path]:
    """List all available DEM files.

    Scans the cropped DEMs directory for .tif files.

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping DEM names (without extension) to file paths.

    Example
    -------
    >>> dems = list_available_dems()
    >>> for name, path in dems.items():
    ...     print(f"{name}: {path}")
    """
    dems = {}
    if CROPPED_DEMS_DIR.exists():
        for tif_path in CROPPED_DEMS_DIR.glob("*.tif"):
            # Remove _strm_crop suffix if present
            name = tif_path.stem
            if name.endswith("_strm_crop"):
                name = name[:-10]
            dems[name.lower()] = tif_path
    return dems


def ensure_directories() -> None:
    """Create all standard project directories if they don't exist.

    Creates:
        - data/
        - data/raw/
        - data/cropped_DEMs/
        - data/outputs/
    """
    for directory in [DATA_DIR, RAW_DATA_DIR, CROPPED_DEMS_DIR, OUTPUTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_dem_path(dem_ref: str) -> Path | None:
    """Resolve a DEM reference to an absolute path.

    Accepts multiple input formats:
    - Friendly name: "inyo" -> data/cropped_DEMs/Inyo_strm_crop.tif
    - Relative path: "data/cropped_DEMs/custom.tif"
    - Absolute path: "/full/path/to/dem.tif"

    Parameters
    ----------
    dem_ref : str
        DEM reference (name, relative path, or absolute path).

    Returns
    -------
    Optional[Path]
        Resolved path if found, None otherwise.

    Example
    -------
    >>> path = resolve_dem_path("inyo")
    >>> path = resolve_dem_path("data/cropped_DEMs/MyDEM.tif")
    """
    # Check if it's a friendly name
    if dem_ref.lower() in EXAMPLE_DEMS:
        return EXAMPLE_DEMS[dem_ref.lower()]

    # Check if it's a path
    path = Path(dem_ref)

    # Absolute path
    if path.is_absolute():
        return path if path.exists() else None

    # Relative to project root
    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path

    # Relative to cropped DEMs directory
    dem_path = CROPPED_DEMS_DIR / path.name
    if dem_path.exists():
        return dem_path

    return None
