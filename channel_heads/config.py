"""Compatibility shim for project configuration and path management.

Path ownership has moved to :mod:`channel_heads.io.paths`. This module keeps
the historical import path working for notebooks, scripts, and user code.
"""

from __future__ import annotations

from .io.paths import (
    CROPPED_DEMS_DIR,
    DATA_DIR,
    EXAMPLE_DEMS,
    EXPORTS_DIR,
    NOTEBOOKS_DIR,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RAW_DIR,
    RESULTS_DIR,
    _find_project_root,
    _get_data_dir,
    ensure_directories,
    get_experiment_output_dir,
    get_output_dir,
    list_available_dems,
    resolve_dem_path,
)

__all__ = [
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
    "ensure_directories",
    "resolve_dem_path",
    "_find_project_root",
    "_get_data_dir",
]
