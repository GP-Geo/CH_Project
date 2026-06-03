"""Tests for channel_heads.training.regime — pure helpers (no DEM/TopoToolbox)."""

from __future__ import annotations

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
