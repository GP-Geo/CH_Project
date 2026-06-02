"""Tests for the channel_heads.io layer (paths, tables, cleanup)."""

from __future__ import annotations

import pandas as pd
import pytest

from channel_heads.io import cleanup, paths, read_table, write_table


def test_paths_are_under_project_root():
    assert paths.MARS_DIR == paths.DATA_DIR / "Mars"
    assert paths.MARS_PAIRS_GPKG.parent == paths.MARS_TOPOLOGY_DIR
    assert paths.MODELS_DIR.name == "models"
    assert paths.XGB_PRODUCTION.name == "xgb_touching_classifier.json"
    assert paths.PRODUCTION_THRESHOLD == pytest.approx(0.577406)


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
