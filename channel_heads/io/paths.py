"""Single source of truth for every path the project reads or writes.

Earth (training) paths come from the low-level :mod:`channel_heads.config`
root-finder; this module re-exports them and adds the **Mars** and **models**
constants that were previously redeclared (inconsistently) at the top of every
script. Import paths from here — never hardcode ``PROJECT_ROOT / "data/..."``.

    from channel_heads.io import paths
    gdf = paths.MARS_VALLEYS          # input vectors
    out = paths.MARS_MODEL_OUTPUTS_DIR / "mars_predictions.gpkg"

The directory accessors (:func:`mars_topology_dir`, ...) create the directory on
demand so pipeline code does not need to ``mkdir`` by hand.
"""

from __future__ import annotations

from pathlib import Path

# Re-export the Earth/core paths from the low-level config module so callers
# only need to know about channel_heads.io.paths.
from channel_heads.config import (
    CROPPED_DEMS_DIR,
    DATA_DIR,
    EXAMPLE_DEMS,
    EXPORTS_DIR,
    NOTEBOOKS_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    RESULTS_DIR,
    ensure_directories,
    get_experiment_output_dir,
    get_output_dir,
    list_available_dems,
    resolve_dem_path,
)

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
    # core (re-exported from config)
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DIR",
    "CROPPED_DEMS_DIR",
    "RESULTS_DIR",
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
