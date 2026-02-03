"""Project configuration and path management.

This module provides centralized path management for the channel-heads project,
supporting both development (local) and installed package usage.

Usage:
    from channel_heads.config import PROCESSED_DIR, EXAMPLE_DEMS

    # Use predefined paths
    dem_path = EXAMPLE_DEMS["inyo"]

    # Or build paths from directories
    dem_path = PROCESSED_DIR / "MyCustomDEM.tif"

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

RAW_DIR: Path = DATA_DIR / "raw"
"""Raw input data directory (SRTM downloads, etc.)."""

PROCESSED_DIR: Path = DATA_DIR / "processed"
"""Directory containing processed/cropped study area DEMs."""

RESULTS_DIR: Path = DATA_DIR / "results"
"""Directory for analysis results (CSVs)."""

EXPORTS_DIR: Path = DATA_DIR / "exports"
"""Directory for exported figures and PDFs."""

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
"""Jupyter notebooks directory."""

# Backward compatibility aliases
RAW_DATA_DIR: Path = RAW_DIR
CROPPED_DEMS_DIR: Path = PROCESSED_DIR
OUTPUTS_DIR: Path = RESULTS_DIR


# Example DEMs with friendly names
# Maps lowercase basin names to their DEM file paths
EXAMPLE_DEMS: dict[str, Path] = {
    # Original 7 basins
    "inyo": PROCESSED_DIR / "Inyo_strm_crop.tif",
    "humboldt": PROCESSED_DIR / "Humboldt_strm_crop.tif",
    "calnalpine": PROCESSED_DIR / "CalnAlpine_strm_crop.tif",
    "daqing": PROCESSED_DIR / "Daqing_strm_crop.tif",
    "luliang": PROCESSED_DIR / "Luliang_strm_crop.tif",
    "kammanasie": PROCESSED_DIR / "Kammanasie_strm_crop.tif",
    "finisterre": PROCESSED_DIR / "Finisterre_strm_crop.tif",
    # Additional basins from Goren & Shelef (2024)
    "taiwan": PROCESSED_DIR / "Taiwan_strm_crop.tif",
    "panamint": PROCESSED_DIR / "Panamint_strm_crop.tif",
    "sakhalin": PROCESSED_DIR / "Sakhalin_strm_crop.tif",
    "vallefertil": PROCESSED_DIR / "SierradelValleFertil_strm_crop.tif",
    "sierramadre": PROCESSED_DIR / "SierraMadre_strm_crop.tif",
    "sierranevadaspain": PROCESSED_DIR / "SierraNevadaSpain_strm_crop.tif",
    "toano": PROCESSED_DIR / "Toano_strm_crop.tif",
    "troodos": PROCESSED_DIR / "Troodos_strm_crop.tif",
    "tsugaru": PROCESSED_DIR / "Tsugaru_strm_crop.tif",
    "yoro": PROCESSED_DIR / "Yoro_strm_crop.tif",
}
"""Dictionary mapping friendly basin names to DEM file paths."""


def get_output_dir(
    study_area: str,
    experiment: str | None = None,
    threshold: int | None = None,
    create: bool = True,
) -> Path:
    """Get output directory for a specific study area and experiment.

    Parameters
    ----------
    study_area : str
        Name of the study area (e.g., "inyo", "humboldt").
    experiment : str, optional
        Experiment name or identifier. If provided, creates a subdirectory
        for this experiment (e.g., "results/inyo/exp_v2/").
    threshold : int, optional
        Stream threshold value. If provided (and experiment is None),
        creates a subdirectory named "th{threshold}" (e.g., "results/inyo/th300/").
        If both experiment and threshold are provided, threshold is appended
        to the experiment name.
    create : bool
        If True, create the directory if it doesn't exist.

    Returns
    -------
    Path
        Path to the study area's output directory.

    Examples
    --------
    >>> get_output_dir("inyo")
    PosixPath('.../data/results/inyo')

    >>> get_output_dir("inyo", threshold=300)
    PosixPath('.../data/results/inyo/th300')

    >>> get_output_dir("inyo", experiment="v2_calibration")
    PosixPath('.../data/results/inyo/v2_calibration')

    >>> get_output_dir("inyo", experiment="v2", threshold=500)
    PosixPath('.../data/results/inyo/v2_th500')
    """
    output_dir = RESULTS_DIR / study_area

    # Build subdirectory name from experiment and/or threshold
    if experiment and threshold:
        output_dir = output_dir / f"{experiment}_th{threshold}"
    elif experiment:
        output_dir = output_dir / experiment
    elif threshold:
        output_dir = output_dir / f"th{threshold}"

    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_experiment_output_dir(
    experiment_name: str,
    create: bool = True,
) -> Path:
    """Get a top-level experiment output directory.

    Use this for experiments that span multiple basins with the same parameters.

    Parameters
    ----------
    experiment_name : str
        Name of the experiment (e.g., "th500_all_basins", "2026-02-02_calibration").
    create : bool
        If True, create the directory if it doesn't exist.

    Returns
    -------
    Path
        Path to the experiment output directory.

    Example
    -------
    >>> output_dir = get_experiment_output_dir("th500_all_basins")
    PosixPath('.../data/results/experiments/th500_all_basins')
    """
    output_dir = RESULTS_DIR / "experiments" / experiment_name
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def list_available_dems() -> dict[str, Path]:
    """List all available DEM files.

    Scans the processed DEMs directory for .tif files.

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
    if PROCESSED_DIR.exists():
        for tif_path in PROCESSED_DIR.glob("*.tif"):
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
        - data/processed/
        - data/results/
        - data/exports/
    """
    for directory in [DATA_DIR, RAW_DIR, PROCESSED_DIR, RESULTS_DIR, EXPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_dem_path(dem_ref: str) -> Path | None:
    """Resolve a DEM reference to an absolute path.

    Accepts multiple input formats:
    - Friendly name: "inyo" -> data/processed/Inyo_strm_crop.tif
    - Relative path: "data/processed/custom.tif"
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
    >>> path = resolve_dem_path("data/processed/MyDEM.tif")
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

    # Relative to processed DEMs directory
    dem_path = PROCESSED_DIR / path.name
    if dem_path.exists():
        return dem_path

    return None
