"""Tests for channel_heads.training.datasets — manifest filtering + CV split."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")  # datasets imports DEFAULT_EMBEDDING_DIM / training.cnn

from channel_heads.training import datasets as ds


class TestFeatureConstants:
    def test_geom_feature_order(self):
        assert ds.GEOM_FEATURES == [
            "orientation_diff_deg",
            "headhead_dist_norm",
            "apex_angle_deg",
            "strahler_order_diff",
            "proximity_profile_norm",
        ]

    def test_emb_features(self):
        assert ds.EMB_FEATURES == ["emb_0", "emb_1", "emb_2", "emb_3"]

    def test_combined_feature_orders(self):
        assert ds.GEOM_PLUS_EMB == ds.GEOM_FEATURES + ds.EMB_FEATURES
        assert ds.GEOM_PLUS_LOGIT == ds.GEOM_FEATURES + ["cnn_logit"]
        assert ds.CNN_LOGIT_FEATURE == "cnn_logit"

    def test_holdout_and_seed_match_training_cnn(self):
        from channel_heads.training import cnn

        assert ds.HOLDOUT_BASIN == cnn.HOLDOUT_BASIN == "taiwan"
        assert ds.RANDOM_STATE == cnn.RANDOM_STATE == 42


class TestManifestFiltering:
    def test_filter_drops_missing_raster_path(self):
        df = pd.DataFrame(
            {"raster_path": ["a.npy", None, "c.npy"], "basin": ["x", "x", "y"]}
        )
        out = ds.filter_valid_raster_rows(df)
        assert list(out["raster_path"]) == ["a.npy", "c.npy"]
        assert list(out.index) == [0, 1]  # reset index

    def test_filter_applies_raster_status_when_present(self):
        df = pd.DataFrame(
            {
                "raster_path": ["a.npy", "b.npy", "c.npy"],
                "raster_status": ["ok", "fail", "ok"],
                "basin": ["x", "x", "y"],
            }
        )
        out = ds.filter_valid_raster_rows(df)
        assert list(out["raster_path"]) == ["a.npy", "c.npy"]

    def test_filter_without_status_column_keeps_all_present_paths(self):
        df = pd.DataFrame({"raster_path": ["a.npy", "b.npy"], "basin": ["x", "y"]})
        out = ds.filter_valid_raster_rows(df)
        assert len(out) == 2

    def test_load_valid_raster_manifest(self, tmp_path):
        df = pd.DataFrame(
            {
                "raster_path": ["a.npy", None, "c.npy"],
                "raster_status": ["ok", "ok", "fail"],
                "basin": ["x", "x", "y"],
            }
        )
        p = tmp_path / "manifest.csv"
        df.to_csv(p, index=False)
        out = ds.load_valid_raster_manifest(p)
        # row 0 only: row1 has no path, row2 status != ok
        assert list(out["raster_path"]) == ["a.npy"]


class TestCvPoolAndSplit:
    def _manifest(self, n_per_basin=20):
        basins = ["taiwan", "inyo", "yoro"]
        rows = []
        for b in basins:
            for i in range(n_per_basin):
                rows.append({"raster_path": f"{b}_{i}.npy", "basin": b, "y": i % 2})
        return pd.DataFrame(rows)

    def test_cv_pool_excludes_taiwan(self):
        df = self._manifest()
        pool = ds.cv_pool(df)
        assert "taiwan" not in set(pool["basin"])
        assert set(pool["basin"]) == {"inyo", "yoro"}
        assert list(pool.index) == list(range(len(pool)))

    def test_val_split_matches_rng_permutation(self):
        df = self._manifest(n_per_basin=40)
        pool = ds.cv_pool(df)
        train, val = ds.deterministic_val_split(pool, val_frac=0.1, seed=42)

        n = len(pool)
        val_size = max(int(n * 0.1), 10)
        perm = np.random.default_rng(42).permutation(n)
        expected_val = pool.iloc[perm[:val_size]].reset_index(drop=True)
        expected_train = pool.iloc[perm[val_size:]].reset_index(drop=True)

        assert len(val) == val_size
        assert len(train) == n - val_size
        pd.testing.assert_frame_equal(val, expected_val)
        pd.testing.assert_frame_equal(train, expected_train)

    def test_val_size_floor_is_ten(self):
        # 30 rows, val_frac 0.1 -> int(3.0)=3 -> floored to 10
        df = self._manifest(n_per_basin=15)  # taiwan excluded -> 30 cv rows
        pool = ds.cv_pool(df)
        assert len(pool) == 30
        train, val = ds.deterministic_val_split(pool, val_frac=0.1, seed=42)
        assert len(val) == 10

    def test_split_is_deterministic_for_seed(self):
        df = self._manifest(n_per_basin=40)
        pool = ds.cv_pool(df)
        t1, v1 = ds.deterministic_val_split(pool, seed=7)
        t2, v2 = ds.deterministic_val_split(pool, seed=7)
        pd.testing.assert_frame_equal(v1, v2)
        pd.testing.assert_frame_equal(t1, t2)
