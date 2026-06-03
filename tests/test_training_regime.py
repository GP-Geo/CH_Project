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

    def test_largest_basin_adjusts_rounding_diff(self):
        positives = pd.DataFrame(
            {
                "y": [1, 1],
                "basin": ["p", "p"],
                "outlet": [0, 0],
                "confluence": [0, 1],
            }
        )
        negatives = pd.DataFrame(
            {
                "y": [0] * 6,
                "basin": ["a", "a", "b", "b", "c", "c"],
                "outlet": [2, 1, 1, 2, 1, 2],
                "confluence": [20, 10, 10, 20, 10, 20],
            }
        )
        df = pd.concat([positives, negatives], ignore_index=True)

        out = regime.stratified_subsample_negatives(
            df,
            target_ratio=2.0,
            random_state=7,
        )

        neg_counts = out[out["y"] == 0].groupby("basin").size().to_dict()
        assert neg_counts == {"a": 2, "b": 1, "c": 1}
        expected = out.sort_values(["basin", "outlet", "confluence"], ignore_index=True)
        pd.testing.assert_frame_equal(out, expected)


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


class TestRegimeFeatureBuild:
    def test_feature_paths_use_regime_outputs(self, tmp_path):
        from channel_heads.regimes import REGIMES

        stats_path, master_path = regime.regime_feature_paths(
            REGIMES["regC"],
            results_dir=tmp_path,
        )
        cache_path = regime.regime_basin_feature_cache_path(
            "inyo",
            REGIMES["regC"],
            results_dir=tmp_path,
        )

        assert stats_path == tmp_path / "build_earth_features_regC_stats.csv"
        assert master_path == tmp_path / "master_dataset_regC.csv"
        assert cache_path == tmp_path / "inyo" / "full_features_regC.csv"

    def test_prefilter_distance_formula(self):
        from channel_heads.regimes import REGIMES

        regime_cfg = REGIMES["regA"]
        assert regime.regime_prefilter_distance(regime_cfg, 1) == 30.0
        assert regime.regime_prefilter_distance(regime_cfg, 400) == 40.0

    def test_assemble_master_forwards_filter_and_subsample_params(self):
        calls = {}
        df = pd.DataFrame(
            {
                "basin": ["a", "a"],
                "outlet": [1, 1],
                "confluence": [2, 3],
                "y": [1, 0],
            }
        )

        def fake_filter(df_in, *, max_L_ratio, max_dist_ratio):
            calls["filter"] = (df_in.copy(), max_L_ratio, max_dist_ratio)
            return df_in

        def fake_subsample(df_in, *, target_ratio, random_state):
            calls["subsample"] = (df_in.copy(), target_ratio, random_state)
            return df_in

        out = regime.assemble_regime_master_dataset(
            df,
            filter_func=fake_filter,
            subsample_func=fake_subsample,
        )

        pd.testing.assert_frame_equal(out, df)
        assert calls["filter"][1:] == (3.0, 5.0)
        assert calls["subsample"][1:] == (3.0, 42)

    def test_build_feature_dataset_loads_cache_unless_force(self, tmp_path):
        from channel_heads.regimes import REGIMES

        regime_cfg = REGIMES["regB"]
        cache_path = tmp_path / "inyo" / "full_features_regB.csv"
        cache_path.parent.mkdir()
        pd.DataFrame(
            {
                "basin": ["inyo", "inyo"],
                "outlet": [1, 1],
                "confluence": [2, 3],
                "y": [1, 0],
            }
        ).to_csv(cache_path, index=False)
        calls = {"processed": 0, "gc": 0}

        def fake_resolve(requested):
            calls["requested"] = requested
            return [("inyo", tmp_path / "inyo.tif")]

        def fake_process(*args, **kwargs):
            calls["processed"] += 1
            raise AssertionError("cache should be used")

        def fake_gc():
            calls["gc"] += 1

        rc = regime.build_regime_feature_dataset(
            regime_cfg,
            results_dir=tmp_path,
            requested_basins=["inyo"],
            force=False,
            no_master=True,
            resolve_basins_func=fake_resolve,
            process_basin_func=fake_process,
            gc_collect=fake_gc,
        )

        assert rc == 0
        assert calls == {"processed": 0, "gc": 0, "requested": ["inyo"]}
        stats_path = tmp_path / "build_earth_features_regB_stats.csv"
        stats = pd.read_csv(stats_path)
        assert stats.to_dict("records") == [
            {
                "basin": "inyo",
                "n_pairs": 2,
                "n_touching": 1,
                "n_not_touching": 1,
                "from_cache": True,
                "time_s": 0.0,
            }
        ]
        assert not (tmp_path / "master_dataset_regB.csv").exists()

    def test_build_feature_dataset_force_processes_and_writes_master(self, tmp_path):
        from channel_heads.regimes import REGIMES

        regime_cfg = REGIMES["regB"]
        cache_path = tmp_path / "inyo" / "full_features_regB.csv"
        cache_path.parent.mkdir()
        pd.DataFrame(
            {
                "basin": ["old"],
                "outlet": [9],
                "confluence": [9],
                "y": [0],
            }
        ).to_csv(cache_path, index=False)
        calls = {}

        def fake_resolve(_requested):
            return [("inyo", tmp_path / "inyo.tif")]

        def fake_process(
            basin_name,
            dem_path,
            regime_arg,
            *,
            min_basin_px,
            max_outlets,
            log_override,
        ):
            calls["process"] = {
                "basin": basin_name,
                "dem_path": dem_path,
                "regime": regime_arg.name,
                "min_basin_px": min_basin_px,
                "max_outlets": max_outlets,
                "log_name": log_override.name,
            }
            return (
                pd.DataFrame(
                    {
                        "basin": ["inyo", "inyo"],
                        "outlet": [1, 1],
                        "confluence": [2, 3],
                        "y": [1, 0],
                    }
                ),
                {
                    "basin": "inyo",
                    "n_pairs": 2,
                    "n_touching": 1,
                    "n_not_touching": 1,
                    "time_s": 2.5,
                    "error": None,
                    "max_outlets": None,
                },
            )

        def fake_filter(df_in, **kwargs):
            calls["filter_kwargs"] = kwargs
            return df_in

        def fake_subsample(df_in, **kwargs):
            calls["subsample_kwargs"] = kwargs
            return df_in

        rc = regime.build_regime_feature_dataset(
            regime_cfg,
            results_dir=tmp_path,
            force=True,
            max_outlets=0,
            resolve_basins_func=fake_resolve,
            process_basin_func=fake_process,
            filter_func=fake_filter,
            subsample_func=fake_subsample,
            gc_collect=lambda: calls.setdefault("gc", 0) or calls.__setitem__("gc", 1),
        )

        assert rc == 0
        assert calls["process"] == {
            "basin": "inyo",
            "dem_path": tmp_path / "inyo.tif",
            "regime": "regB",
            "min_basin_px": 500,
            "max_outlets": None,
            "log_name": "channel_heads.training.regime",
        }
        assert calls["filter_kwargs"] == {
            "max_L_ratio": 3.0,
            "max_dist_ratio": 5.0,
        }
        assert calls["subsample_kwargs"] == {
            "target_ratio": 3.0,
            "random_state": 42,
        }
        assert pd.read_csv(cache_path)["basin"].tolist() == ["inyo", "inyo"]
        assert (tmp_path / "master_dataset_regB.csv").exists()

    def test_build_feature_dataset_empty_basin_resolution_returns_error(self, tmp_path):
        from channel_heads.regimes import REGIMES

        rc = regime.build_regime_feature_dataset(
            REGIMES["regA"],
            results_dir=tmp_path,
            resolve_basins_func=lambda requested: [],
        )

        assert rc == 1
        assert not (tmp_path / "build_earth_features_regA_stats.csv").exists()

    def test_earth_feature_script_import_smoke_and_default_args(
        self,
        tmp_path,
        monkeypatch,
    ):
        script_path = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "build_earth_features_regime.py"
        )
        spec = importlib.util.spec_from_file_location(
            "build_earth_features_regime_smoke",
            script_path,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        calls = {}

        def fake_build(regime_cfg, **kwargs):
            calls["regime"] = regime_cfg.name
            calls.update(kwargs)
            return 0

        monkeypatch.setattr(module, "RESULTS_DIR", tmp_path)
        monkeypatch.setattr(module, "build_regime_feature_dataset", fake_build)

        assert module.main(["--regime", "regA"]) == 0
        assert calls["regime"] == "regA"
        assert calls["results_dir"] == tmp_path
        assert calls["requested_basins"] is None
        assert calls["force"] is False
        assert calls["no_master"] is False
        assert calls["min_basin_px"] == 500
        assert calls["max_outlets"] == 40
        assert calls["log_override"].name == "build_earth_features_regime"
