"""Tests for channel_heads.training.regime — pure helpers (no DEM/TopoToolbox)."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# regime imports analyzers/pruning which may pull TopoToolbox; skip if absent.
regime = pytest.importorskip("channel_heads.training.regime")


class TestConstants:
    def test_regime_constants(self):
        assert regime.NEGATIVE_RATIO == 3.0
        assert regime.RANDOM_SEED == 42
        assert regime.HARD_NEG_MAX_L_RATIO == 3.0
        assert regime.HARD_NEG_MAX_DIST_RATIO == 5.0
        assert regime.CONNECTIVITY == 8

    def test_dem_to_basin_has_17_basins(self):
        assert len(regime.DEM_TO_BASIN) == 17
        assert regime.DEM_TO_BASIN["Taiwan_strm_crop.tif"] == "taiwan"


class TestStratifiedSubsample:
    def test_empty_returns_unchanged(self):
        df = pd.DataFrame(columns=["y", "basin", "outlet", "confluence"])
        out = regime.stratified_subsample_negatives(df)
        assert out.empty

    def test_negatives_below_target_returns_unchanged(self):
        df = pd.DataFrame(
            {
                "y": [1, 1, 0, 0],  # n_pos=2 -> target 6 negs; only 2 present
                "basin": ["a", "a", "a", "a"],
                "outlet": [1, 1, 1, 1],
                "confluence": [1, 2, 3, 4],
            }
        )
        out = regime.stratified_subsample_negatives(df, target_ratio=3.0)
        # returned object is the original df (no subsampling)
        assert len(out) == 4

    def test_downsamples_to_target_ratio_deterministic(self):
        rng = np.random.default_rng(1)
        n_pos = 10
        pos = pd.DataFrame(
            {
                "y": [1] * n_pos,
                "basin": ["a"] * n_pos,
                "outlet": rng.integers(0, 3, n_pos),
                "confluence": rng.integers(0, 100, n_pos),
            }
        )
        # 200 negatives across two basins
        neg = pd.DataFrame(
            {
                "y": [0] * 200,
                "basin": ["a"] * 120 + ["b"] * 80,
                "outlet": rng.integers(0, 5, 200),
                "confluence": rng.integers(0, 100, 200),
            }
        )
        df = pd.concat([pos, neg], ignore_index=True)

        out1 = regime.stratified_subsample_negatives(df, target_ratio=3.0, random_state=42)
        out2 = regime.stratified_subsample_negatives(df, target_ratio=3.0, random_state=42)

        n_pos_out = int((out1["y"] == 1).sum())
        n_neg_out = int((out1["y"] == 0).sum())
        assert n_pos_out == 10
        assert n_neg_out == 30  # 10 * 3.0
        # deterministic for fixed seed
        pd.testing.assert_frame_equal(out1, out2)
        # sorted by basin, outlet, confluence
        expected = out1.sort_values(["basin", "outlet", "confluence"], ignore_index=True)
        pd.testing.assert_frame_equal(out1, expected)


class TestResolveRegimeBasins:
    def test_filters_to_existing_dems_and_requested(self, tmp_path, monkeypatch):
        inyo = tmp_path / "inyo.tif"
        inyo.write_bytes(b"x")
        yoro = tmp_path / "yoro.tif"
        yoro.write_bytes(b"x")
        # taiwan path points to a missing file
        fake_dems = {
            "inyo": inyo,
            "yoro": yoro,
            "taiwan": tmp_path / "missing_taiwan.tif",
        }
        monkeypatch.setattr(regime, "EXAMPLE_DEMS", fake_dems)

        # No restriction: only existing DEMs, sorted by DEM filename order.
        out = regime.resolve_regime_basins(None)
        names = [n for n, _ in out]
        assert "taiwan" not in names  # missing on disk
        assert set(names) == {"inyo", "yoro"}

        # Restriction to inyo only.
        out2 = regime.resolve_regime_basins(["inyo"])
        assert [n for n, _ in out2] == ["inyo"]


class TestRegimeStreamLoaderFactory:
    def test_loader_returns_none_when_dem_missing(self, monkeypatch):
        from channel_heads.regimes import REGIMES

        loader = regime.make_regime_stream_loader(REGIMES["regA"])
        monkeypatch.setattr(regime, "resolve_dem_path", lambda basin: None)
        # DEM cannot be resolved -> loader returns None (no TopoToolbox needed).
        assert loader("inyo", lat=36.0, z_th=0.0, threshold=0) is None

    def test_loader_uses_regime_threshold_mask_and_pruning(self, tmp_path, monkeypatch):
        from channel_heads.regimes import REGIMES

        dem_path = tmp_path / "inyo.tif"
        dem_path.write_bytes(b"x")

        class FakeDem:
            def __init__(self):
                self.z = np.array([[0.0, 10.0], [20.0, 30.0]])

        fake_dem = FakeDem()
        stream_full = object()
        stream_pruned = object()
        calls = {}

        def fake_stream_object(_fd, threshold):
            calls["stream_threshold"] = threshold
            return stream_full

        fake_tt3 = types.SimpleNamespace(
            read_tif=lambda path: fake_dem,
            FlowObject=lambda dem: ("fd", dem),
            StreamObject=fake_stream_object,
        )
        monkeypatch.setitem(sys.modules, "topotoolbox", fake_tt3)
        monkeypatch.setattr(regime, "resolve_dem_path", lambda basin: dem_path)

        def fake_pixel_size(_dem, lat_deg):
            calls["lat"] = lat_deg
            return 10.0

        def fake_threshold_cells(threshold_km2, pixel_size_m):
            calls["threshold_args"] = (threshold_km2, pixel_size_m)
            return 123

        monkeypatch.setattr(regime, "compute_pixel_size_m_from_dem", fake_pixel_size)
        monkeypatch.setattr(regime, "compute_threshold_cells", fake_threshold_cells)

        def fake_apply_strategy(
            stream,
            *,
            pre_remove_max_order,
            order_gap_to_prune,
        ):
            calls["pruning"] = (
                stream,
                pre_remove_max_order,
                order_gap_to_prune,
            )
            return stream_pruned

        monkeypatch.setattr(regime, "apply_strategy", fake_apply_strategy)

        regime_cfg = REGIMES["regA"]
        loader = regime.make_regime_stream_loader(regime_cfg)
        out = loader("inyo", lat=36.5, z_th=15.0, threshold=999)

        assert out == (stream_pruned, fake_dem)
        assert np.isnan(fake_dem.z[0, 0])
        assert np.isnan(fake_dem.z[0, 1])
        assert fake_dem.z[1, 0] == 20.0
        assert calls["lat"] == 36.5
        assert calls["threshold_args"] == (regime_cfg.threshold_km2, 10.0)
        assert calls["stream_threshold"] == 123
        assert calls["pruning"] == (
            stream_full,
            regime_cfg.pre_remove_max_order,
            regime_cfg.order_gap_to_prune,
        )


class TestRegimePatchBuild:
    def test_patch_paths_use_regime_outputs(self, tmp_path):
        from channel_heads.regimes import REGIMES

        master_csv, output_root, manifest_path = regime.regime_patch_paths(
            REGIMES["regB"],
            results_dir=tmp_path,
        )

        assert master_csv == tmp_path / "master_dataset_regB.csv"
        assert output_root == tmp_path / "_rasters_regB"
        assert manifest_path == tmp_path / "raster_manifest_regB.csv"

    def test_build_regime_patch_dataset_uses_paths_and_defaults(
        self,
        tmp_path,
    ):
        from channel_heads.regimes import REGIMES

        regime_cfg = REGIMES["regA"]
        master_csv = tmp_path / "master_dataset_regA.csv"
        master_csv.write_text("basin,outlet,confluence,head_1,head_2,y\n")
        stream_loader = object()
        calls = {}

        def fake_precompute_func(**kwargs):
            calls.update(kwargs)
            return pd.DataFrame(
                {
                    "basin": ["inyo", "yoro"],
                    "raster_path": ["p", np.nan],
                    "raster_status": ["ok", "skipped"],
                }
            )

        df, manifest_path = regime.build_regime_patch_dataset(
            regime_cfg,
            results_dir=tmp_path,
            stream_loader=stream_loader,
            precompute_func=fake_precompute_func,
        )

        assert calls["master_csv"] == master_csv
        assert calls["output_dir"] == tmp_path / "_rasters_regA"
        assert calls["dem_loader"] is stream_loader
        assert calls["target_size"] == 128
        assert calls["threshold"] == 0
        assert calls["output_dir"].is_dir()
        assert manifest_path == tmp_path / "raster_manifest_regA.csv"
        assert manifest_path.exists()
        pd.testing.assert_frame_equal(pd.read_csv(manifest_path), df)

    def test_script_import_smoke_and_default_target_size(self, tmp_path, monkeypatch):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build_cnn_patches_regime.py"
        )
        spec = importlib.util.spec_from_file_location(
            "build_cnn_patches_regime_smoke",
            script_path,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        (tmp_path / "master_dataset_regA.csv").write_text(
            "basin,outlet,confluence,head_1,head_2,y\n"
        )
        calls = {}

        def fake_build(regime_cfg, *, results_dir, target_size, log_override):
            calls["regime"] = regime_cfg.name
            calls["results_dir"] = results_dir
            calls["target_size"] = target_size
            calls["log_name"] = log_override.name
            return pd.DataFrame(), tmp_path / "raster_manifest_regA.csv"

        monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(module, "build_regime_patch_dataset", fake_build)

        assert module.main(["--regime", "regA"]) == 0
        assert calls == {
            "regime": "regA",
            "results_dir": tmp_path,
            "target_size": 128,
            "log_name": "build_cnn_patches_regime",
        }
