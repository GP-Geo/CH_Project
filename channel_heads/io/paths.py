"""Single source of truth for every path the project reads or writes.

This module owns project-root detection, Earth DEM/result paths, Mars data
paths, and trained-model artifact paths. Import paths from here; the historical
``channel_heads.config`` module re-exports this API for compatibility.

    from channel_heads.io import paths
    dem = paths.EXAMPLE_DEMS["inyo"]
    gdf = paths.MARS_VALLEYS          # input vectors
    out = paths.MARS_MODEL_OUTPUTS_DIR / "mars_predictions.gpkg"

The directory accessors (:func:`mars_topology_dir`, ...) create the directory on
demand so pipeline code does not need to ``mkdir`` by hand.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root() -> Path:
    """Find the project root directory.

    ``CHANNEL_HEADS_ROOT`` overrides auto-detection. Otherwise, search upward
    from this file for ``pyproject.toml`` or ``.git``; if no marker is found,
    fall back to the current working directory. This preserves the historical
    :mod:`channel_heads.config` behavior.
    """

    if os.getenv("CHANNEL_HEADS_ROOT"):
        return Path(os.getenv("CHANNEL_HEADS_ROOT"))

    current = Path(__file__).resolve().parent
    markers = ("pyproject.toml", ".git")

    for _ in range(6):
        if any((current / marker).exists() for marker in markers):
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    return Path.cwd()


def _get_data_dir() -> Path:
    """Get the data directory, honoring ``CHANNEL_HEADS_DATA`` when set."""

    if os.getenv("CHANNEL_HEADS_DATA"):
        return Path(os.getenv("CHANNEL_HEADS_DATA"))
    return PROJECT_ROOT / "data"


# --------------------------------------------------------------------------- #
# Earth/core directories
# --------------------------------------------------------------------------- #
PROJECT_ROOT: Path = _find_project_root()
"""Project root directory."""

DATA_DIR: Path = _get_data_dir()
"""Main data directory."""

RAW_DIR: Path = DATA_DIR / "raw"
"""Raw input data directory (SRTM downloads, etc.)."""

CROPPED_DEMS_DIR: Path = DATA_DIR / "cropped_DEMs"
"""Directory containing processed/cropped study area DEMs."""

RESULTS_DIR: Path = DATA_DIR / "results"
"""Directory for analysis results and pipeline outputs."""

EXPORTS_DIR: Path = DATA_DIR / "exports"
"""Directory for exported figures and PDFs."""

NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
"""Jupyter notebooks directory."""

# Historical compatibility aliases from channel_heads.config.
RAW_DATA_DIR: Path = RAW_DIR
PROCESSED_DIR: Path = CROPPED_DEMS_DIR
OUTPUTS_DIR: Path = RESULTS_DIR


EXAMPLE_DEMS: dict[str, Path] = {
    "inyo": CROPPED_DEMS_DIR / "Inyo_strm_crop.tif",
    "humboldt": CROPPED_DEMS_DIR / "Humboldt_strm_crop.tif",
    "calnalpine": CROPPED_DEMS_DIR / "CalnAlpine_strm_crop.tif",
    "daqing": CROPPED_DEMS_DIR / "Daqing_strm_crop.tif",
    "luliang": CROPPED_DEMS_DIR / "Luliang_strm_crop.tif",
    "kammanasie": CROPPED_DEMS_DIR / "Kammanasie_strm_crop.tif",
    "finisterre": CROPPED_DEMS_DIR / "Finisterre_strm_crop.tif",
    "taiwan": CROPPED_DEMS_DIR / "Taiwan_strm_crop.tif",
    "panamint": CROPPED_DEMS_DIR / "Panamint_strm_crop.tif",
    "sakhalin": CROPPED_DEMS_DIR / "Sakhalin_strm_crop.tif",
    "vallefertil": CROPPED_DEMS_DIR / "SierradelValleFertil_strm_crop.tif",
    "sierramadre": CROPPED_DEMS_DIR / "SierraMadre_strm_crop.tif",
    "sierranevadaspain": CROPPED_DEMS_DIR / "SierraNevadaSpain_strm_crop.tif",
    "toano": CROPPED_DEMS_DIR / "Toano_strm_crop.tif",
    "troodos": CROPPED_DEMS_DIR / "Troodos_strm_crop.tif",
    "tsugaru": CROPPED_DEMS_DIR / "Tsugaru_strm_crop.tif",
    "yoro": CROPPED_DEMS_DIR / "Yoro_strm_crop.tif",
}
"""Dictionary mapping friendly basin names to DEM file paths."""


def get_output_dir(
    study_area: str,
    experiment: str | None = None,
    threshold: int | None = None,
    create: bool = True,
) -> Path:
    """Get output directory for a specific study area and experiment."""

    output_dir = RESULTS_DIR / study_area

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
    """Get a top-level experiment output directory."""

    output_dir = RESULTS_DIR / "experiments" / experiment_name
    if create:
        output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def list_available_dems() -> dict[str, Path]:
    """List all available DEM files in :data:`CROPPED_DEMS_DIR`."""

    dems = {}
    if CROPPED_DEMS_DIR.exists():
        for tif_path in CROPPED_DEMS_DIR.glob("*.tif"):
            name = tif_path.stem
            if name.endswith("_strm_crop"):
                name = name[:-10]
            dems[name.lower()] = tif_path
    return dems


def ensure_directories() -> None:
    """Create all standard Earth/core project directories if missing."""

    for directory in [DATA_DIR, RAW_DIR, CROPPED_DEMS_DIR, RESULTS_DIR, EXPORTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def resolve_dem_path(dem_ref: str) -> Path | None:
    """Resolve a DEM reference to an absolute path when it can be found."""

    if dem_ref.lower() in EXAMPLE_DEMS:
        return EXAMPLE_DEMS[dem_ref.lower()]

    path = Path(dem_ref)

    if path.is_absolute():
        return path if path.exists() else None

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return project_path

    dem_path = CROPPED_DEMS_DIR / path.name
    if dem_path.exists():
        return dem_path

    return None

# --------------------------------------------------------------------------- #
# Top-level directories
# --------------------------------------------------------------------------- #
MODELS_DIR: Path = PROJECT_ROOT / "models"
"""Trained-model artifacts (gitignored). Never auto-deleted."""

MARS_DIR: Path = DATA_DIR / "Mars"
"""Root of the Mars cross-planet data tree."""

FINAL_VALLEYS_DIR: Path = DATA_DIR / "final_valleys"
"""Mars valley-network source vectors (RAW_KEEP)."""

# Mars sub-trees
MARS_TOPOLOGY_DIR: Path = MARS_DIR / "topology"
MARS_OUTLET_CANDIDATES_DIR: Path = MARS_DIR / "outlet_candidates"
MARS_MODEL_INPUTS_DIR: Path = MARS_DIR / "model_inputs"
MARS_MODEL_OUTPUTS_DIR: Path = MARS_DIR / "model_outputs"
MARS_CNN_PATCHES_DIR: Path = MARS_MODEL_INPUTS_DIR / "cnn_patches_5class"

# Generated-raster homes (Earth + regime), used by cleanup tooling.
RESULTS_FIGURES_DIR: Path = RESULTS_DIR / "figures_models"
POSTER_FIGURES_DIR: Path = RESULTS_DIR / "final_figures"
"""Canonical home for poster/report-quality figure exports."""

# --------------------------------------------------------------------------- #
# Named source inputs (RAW_KEEP — never delete)
# --------------------------------------------------------------------------- #
MARS_DEM: Path = MARS_DIR / "Mars_DEM_reprojected.tif"
MARS_HILLSHADE: Path = MARS_DIR / "MOLA_Hillshade_Robinson_128ppd.tif"
MARS_VALLEYS: Path = FINAL_VALLEYS_DIR / "final_valleys_fixed.gpkg"

# --------------------------------------------------------------------------- #
# Named generated artifacts (CAN_REGENERATE)
# --------------------------------------------------------------------------- #
MARS_OUTLET_CANDIDATES_GPKG: Path = (
    MARS_OUTLET_CANDIDATES_DIR / "mars_vn_outlet_candidates.gpkg"
)
MARS_TOPOLOGY_GPKG: Path = MARS_TOPOLOGY_DIR / "mars_vn_topology_model_ready.gpkg"
MARS_PAIRS_GPKG: Path = MARS_TOPOLOGY_DIR / "mars_vn_pairs.gpkg"
MARS_CNN_PATCH_INDEX: Path = MARS_MODEL_INPUTS_DIR / "mars_cnn_patch_index.parquet"
MARS_CNN_EMBEDDINGS: Path = MARS_MODEL_INPUTS_DIR / "mars_cnn_embeddings.parquet"

# --------------------------------------------------------------------------- #
# Production model artifacts (preserved as-is — see CLAUDE.md)
# --------------------------------------------------------------------------- #
XGB_PRODUCTION: Path = MODELS_DIR / "xgb_touching_classifier.json"
CNN_PRODUCTION: Path = MODELS_DIR / "cnn_outlet_final.pt"
PRODUCTION_THRESHOLD: float = 0.577406
"""Frozen Earth F1-optimal decision threshold of the production XGBoost model."""


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    """Models directory, created on demand."""
    return _ensure(MODELS_DIR)


def mars_topology_dir() -> Path:
    return _ensure(MARS_TOPOLOGY_DIR)


def mars_outlet_candidates_dir() -> Path:
    return _ensure(MARS_OUTLET_CANDIDATES_DIR)


def mars_model_inputs_dir() -> Path:
    return _ensure(MARS_MODEL_INPUTS_DIR)


def mars_model_outputs_dir() -> Path:
    return _ensure(MARS_MODEL_OUTPUTS_DIR)


def poster_figures_dir() -> Path:
    return _ensure(POSTER_FIGURES_DIR)


def model_path(name: str) -> Path:
    """Path to a model artifact by file name (e.g. ``"xgb_geom_only.json"``)."""
    return MODELS_DIR / name


__all__ = [
    # Earth/core
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "RAW_DATA_DIR",
    "CROPPED_DEMS_DIR",
    "PROCESSED_DIR",
    "RESULTS_DIR",
    "OUTPUTS_DIR",
    "EXPORTS_DIR",
    "NOTEBOOKS_DIR",
    "EXAMPLE_DEMS",
    "get_output_dir",
    "get_experiment_output_dir",
    "list_available_dems",
    "resolve_dem_path",
    "ensure_directories",
    # models + Mars dirs
    "MODELS_DIR",
    "MARS_DIR",
    "FINAL_VALLEYS_DIR",
    "MARS_TOPOLOGY_DIR",
    "MARS_OUTLET_CANDIDATES_DIR",
    "MARS_MODEL_INPUTS_DIR",
    "MARS_MODEL_OUTPUTS_DIR",
    "MARS_CNN_PATCHES_DIR",
    "RESULTS_FIGURES_DIR",
    "POSTER_FIGURES_DIR",
    # named inputs
    "MARS_DEM",
    "MARS_HILLSHADE",
    "MARS_VALLEYS",
    # named generated artifacts
    "MARS_OUTLET_CANDIDATES_GPKG",
    "MARS_TOPOLOGY_GPKG",
    "MARS_PAIRS_GPKG",
    "MARS_CNN_PATCH_INDEX",
    "MARS_CNN_EMBEDDINGS",
    # production models
    "XGB_PRODUCTION",
    "CNN_PRODUCTION",
    "PRODUCTION_THRESHOLD",
    # accessors
    "models_dir",
    "mars_topology_dir",
    "mars_outlet_candidates_dir",
    "mars_model_inputs_dir",
    "mars_model_outputs_dir",
    "poster_figures_dir",
    "model_path",
]
