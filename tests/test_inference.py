"""Tests for channel_heads.inference — shared XGBoost inference helpers.

These helpers were extracted from the three Mars inference scripts
(``run_mars_xgb_inference_5feat``, ``run_mars_combined_xgb_inference`` and
``run_mars_combined_regime``), which each carried near-identical copies of the
artifact loaders, feature-matrix verification, and predict-with-threshold step.
The tests pin the shared behavior: blank/empty-file handling, the
missing/non-numeric/inf invariants, NaN being allowed (XGBoost handles it
natively), model feature-order verification, and the standard
``predict_proba >= threshold`` decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from channel_heads import inference
from channel_heads.inference import (
    load_feature_columns,
    load_threshold,
    load_xgb_model,
    predict_with_threshold,
    verify_feature_matrix,
    verify_model_feature_order,
)


# --------------------------------------------------------------------------
# Artifact loaders
# --------------------------------------------------------------------------
class TestArtifactLoaders:
    def test_load_feature_columns_strips_and_ignores_blanks(self, tmp_path):
        p = tmp_path / "feature_columns.txt"
        p.write_text("  a \n\n b\nc\n\n")
        assert load_feature_columns(p) == ["a", "b", "c"]

    def test_load_feature_columns_empty_raises(self, tmp_path):
        p = tmp_path / "empty.txt"
        p.write_text("\n  \n")
        with pytest.raises(RuntimeError):
            load_feature_columns(p)

    def test_load_threshold_first_line(self, tmp_path):
        p = tmp_path / "thr.txt"
        p.write_text("0.577406\nsome note\n")
        assert load_threshold(p) == pytest.approx(0.577406)

    def test_load_threshold_empty_raises(self, tmp_path):
        p = tmp_path / "thr.txt"
        p.write_text("   \n")
        with pytest.raises(RuntimeError):
            load_threshold(p)


# --------------------------------------------------------------------------
# Feature-matrix verification
# --------------------------------------------------------------------------
class TestVerifyFeatureMatrix:
    def _df(self):
        return pd.DataFrame({"f1": [1.0, 2.0, np.nan], "f2": [3.0, 4.0, 5.0]})

    def test_ok_with_nan_stats(self):
        stats = verify_feature_matrix(self._df(), ["f1", "f2"])
        assert stats == {
            "n_rows": 3,
            "n_cols": 2,
            "n_nan_cells": 1,
            "n_nan_rows": 1,
        }

    def test_missing_column_raises(self):
        with pytest.raises(RuntimeError, match="missing"):
            verify_feature_matrix(self._df(), ["f1", "nope"])

    def test_non_numeric_raises(self):
        df = self._df()
        df["f2"] = ["a", "b", "c"]
        with pytest.raises(RuntimeError, match="non-numeric"):
            verify_feature_matrix(df, ["f1", "f2"])

    def test_inf_raises(self):
        df = self._df()
        df.loc[0, "f2"] = np.inf
        with pytest.raises(RuntimeError, match="inf"):
            verify_feature_matrix(df, ["f1", "f2"])

    def test_context_prefix_in_message(self):
        with pytest.raises(RuntimeError, match="Variant emb"):
            verify_feature_matrix(self._df(), ["x"], context="Variant emb")


# --------------------------------------------------------------------------
# Model loading / feature-order verification / prediction
# --------------------------------------------------------------------------
def _toy_model(features):
    xgboost = pytest.importorskip("xgboost")

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, len(features))), columns=features)
    y = (X[features[0]] > 0).astype(int)
    model = xgboost.XGBClassifier(n_estimators=5, max_depth=2, use_label_encoder=False)
    model.fit(X, y)
    return model, X


class TestModelHelpers:
    def test_verify_model_feature_order_match(self):
        model, _ = _toy_model(["a", "b", "c"])
        verify_model_feature_order(model, ["a", "b", "c"])  # no raise

    def test_verify_model_feature_order_mismatch(self):
        model, _ = _toy_model(["a", "b", "c"])
        with pytest.raises(RuntimeError, match="differ"):
            verify_model_feature_order(model, ["a", "c", "b"])

    def test_load_xgb_model_roundtrip_and_verify(self, tmp_path):
        model, X = _toy_model(["a", "b", "c"])
        path = tmp_path / "m.json"
        model.save_model(str(path))
        loaded = load_xgb_model(path, expected_features=["a", "b", "c"])
        # predictions identical to the original model
        np.testing.assert_allclose(loaded.predict_proba(X)[:, 1], model.predict_proba(X)[:, 1])

    def test_load_xgb_model_bad_features_raises(self, tmp_path):
        model, _ = _toy_model(["a", "b", "c"])
        path = tmp_path / "m.json"
        model.save_model(str(path))
        with pytest.raises(RuntimeError):
            load_xgb_model(path, expected_features=["a", "b"])

    def test_predict_with_threshold(self):
        model, X = _toy_model(["a", "b", "c"])
        proba, pred = predict_with_threshold(model, X, ["a", "b", "c"], 0.5)
        assert proba.shape == (len(X),)
        np.testing.assert_array_equal(pred, (proba >= 0.5).astype(int))


# --------------------------------------------------------------------------
# Package surface
# --------------------------------------------------------------------------
class TestPackageSurface:
    def test_pick_device_returns_known(self):
        assert inference.pick_device() in {"mps", "cuda", "cpu"}

    def test_reexports(self):
        from channel_heads.inference import xgb as xgb_mod

        assert inference.load_feature_columns is xgb_mod.load_feature_columns
        assert inference.predict_with_threshold is xgb_mod.predict_with_threshold

    def test_canonical_location_is_models_xgboost(self):
        """The implementation now lives in models.xgboost; inference.xgb is a shim."""
        from channel_heads.models import xgboost as xgb_new

        assert load_feature_columns is xgb_new.load_feature_columns
        assert predict_with_threshold is xgb_new.predict_with_threshold

    def test_old_and_new_paths_resolve_to_same_impl(self):
        from channel_heads.inference import xgb as xgb_old
        from channel_heads.models import xgboost as xgb_new

        for name in (
            "load_feature_columns",
            "load_threshold",
            "load_xgb_model",
            "verify_model_feature_order",
            "verify_feature_matrix",
            "predict_with_threshold",
        ):
            assert getattr(xgb_old, name) is getattr(xgb_new, name)
