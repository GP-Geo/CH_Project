"""Tests for the channel_heads.io layer (paths, tables, cleanup)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest

from channel_heads.io import cleanup, paths, read_table, write_table


def test_paths_are_under_project_root():
    assert paths.MARS_DIR == paths.DATA_DIR / "Mars"
    assert paths.MARS_PAIRS_GPKG.parent == paths.MARS_TOPOLOGY_DIR
    assert paths.MODELS_DIR.name == "models"
    assert paths.XGB_PRODUCTION.name == "xgb_touching_classifier.json"
    assert paths.PRODUCTION_THRESHOLD == pytest.approx(0.577406)


def test_io_paths_has_canonical_constants():
    for name in (
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
    ):
        assert hasattr(paths, name)


def test_core_path_categories_and_legacy_aliases():
    assert paths.RAW_DIR == paths.DATA_DIR / "raw"
    assert paths.RAW_DATA_DIR == paths.RAW_DIR
    assert paths.CROPPED_DEMS_DIR == paths.DATA_DIR / "cropped_DEMs"
    assert paths.PROCESSED_DIR == paths.CROPPED_DEMS_DIR
    assert paths.RESULTS_DIR == paths.DATA_DIR / "results"
    assert paths.OUTPUTS_DIR == paths.RESULTS_DIR
    assert paths.EXPORTS_DIR == paths.DATA_DIR / "exports"
    assert paths.MARS_DIR == paths.DATA_DIR / "Mars"
    assert paths.FINAL_VALLEYS_DIR == paths.DATA_DIR / "final_valleys"
    assert paths.MARS_VALLEYS == paths.FINAL_VALLEYS_DIR / "final_valleys_fixed.gpkg"
    assert paths.MARS_TOPOLOGY_GPKG.parent == paths.MARS_TOPOLOGY_DIR
    assert paths.MARS_CNN_PATCH_INDEX.parent == paths.MARS_MODEL_INPUTS_DIR


def test_models_artifact_dir_is_not_models_package():
    import channel_heads.models as models_package

    package_dir = Path(models_package.__file__).parent
    assert paths.MODELS_DIR == paths.PROJECT_ROOT / "models"
    assert paths.MODELS_DIR != package_dir
    model_name = "xgb_touching_classifier.json"
    assert paths.model_path(model_name) == paths.MODELS_DIR / model_name


def test_environment_overrides_are_preserved(monkeypatch, tmp_path):
    root_override = tmp_path / "project-root"
    data_override = tmp_path / "external-data"
    monkeypatch.setenv("CHANNEL_HEADS_ROOT", str(root_override))
    monkeypatch.setenv("CHANNEL_HEADS_DATA", str(data_override))

    try:
        importlib.reload(paths)

        assert paths.PROJECT_ROOT == root_override
        assert paths.DATA_DIR == data_override
        assert paths.CROPPED_DEMS_DIR == data_override / "cropped_DEMs"
        assert paths.EXAMPLE_DEMS["inyo"] == data_override / "cropped_DEMs/Inyo_strm_crop.tif"
    finally:
        monkeypatch.delenv("CHANNEL_HEADS_ROOT", raising=False)
        monkeypatch.delenv("CHANNEL_HEADS_DATA", raising=False)
        importlib.reload(paths)


def test_path_accessor_creates_dir(tmp_path, monkeypatch):
    target = tmp_path / "model_outputs"
    monkeypatch.setattr(paths, "MARS_MODEL_OUTPUTS_DIR", target)
    out = paths.mars_model_outputs_dir()
    assert out.exists() and out.is_dir()


@pytest.mark.parametrize("suffix", [".parquet", ".csv"])
def test_write_then_read_roundtrip(tmp_path, suffix):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / f"t{suffix}"
    try:
        write_table(df, path)
    except ImportError:
        pytest.skip("pyarrow not installed")
    back = read_table(path)
    pd.testing.assert_frame_equal(df, back)
    if suffix == ".parquet":
        assert path.with_suffix(".csv").exists()  # sidecar written


def test_write_table_rejects_unknown_suffix(tmp_path):
    with pytest.raises(ValueError):
        write_table(pd.DataFrame({"a": [1]}), tmp_path / "t.feather")


def test_cleanup_scan_never_lists_raw_or_models(tmp_path, monkeypatch):
    # Point DATA_DIR at an empty tree: scan must be empty and never error.
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path)
    items = cleanup.scan()
    assert items == []
    assert "No generated artifacts" in cleanup.format_manifest(items)
    # Dry-run clean is a no-op returning an empty action list.
    assert cleanup.clean(dry_run=True) == []
