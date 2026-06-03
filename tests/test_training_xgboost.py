"""Tests for channel_heads.training.xgboost — config, threshold policy, files."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("xgboost")

from channel_heads.training import xgboost as txgb


class TestScalePosWeight:
    def test_ratio(self):
        y = np.array([0, 0, 0, 1])  # 3 neg, 1 pos
        assert txgb.xgb_scale_pos_weight(y) == 3.0

    def test_zero_positive_uses_max_one(self):
        y = np.array([0, 0, 0])
        assert txgb.xgb_scale_pos_weight(y) == 3.0  # 3 / max(0, 1)


class TestBuildClassifier:
    def test_frozen_config(self):
        clf = txgb.build_xgb_classifier(2.5)
        params = clf.get_params()
        assert params["n_estimators"] == 200
        assert params["max_depth"] == 4
        assert params["learning_rate"] == 0.1
        assert params["random_state"] == 42
        assert params["n_jobs"] == -1
        assert params["eval_metric"] == "logloss"
        assert params["tree_method"] == "hist"
        assert params["scale_pos_weight"] == 2.5


class TestThresholdPolicy:
    def test_max_precision_at_recall(self):
        # Well-separated scores so a high-precision, high-recall point exists.
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        proba = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        thr, source, p, r = txgb.tune_threshold_max_precision(y, proba, min_recall=0.5)
        assert source == "max_precision_at_recall>=0.50"
        assert 0.0 < thr < 1.0
        assert not np.isnan(p) and not np.isnan(r)
        assert r >= 0.5

    def test_threshold_matches_eval_helper_value(self):
        from channel_heads.eval.metrics import max_precision_threshold

        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, size=200)
        proba = rng.random(200)
        thr, source, _, _ = txgb.tune_threshold_max_precision(y, proba, min_recall=0.5)
        if source == "max_precision_at_recall>=0.50":
            assert thr == pytest.approx(max_precision_threshold(y, proba, 0.5))

    def test_fallback_when_no_recall_point(self):
        # Require recall >= 0.99 so the constraint cannot be met meaningfully;
        # craft probas where no PR point (excluding the last) reaches it.
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.9, 0.8, 0.2, 0.1])  # model ranks negatives high
        thr, source, p, r = txgb.tune_threshold_max_precision(y, proba, min_recall=1.01)
        assert thr == 0.5
        assert source == "fallback_default_0.5"
        assert np.isnan(p) and np.isnan(r)


class TestArtifactWriters:
    def test_write_feature_columns(self, tmp_path):
        p = tmp_path / "feats.txt"
        feats = ["a", "b", "c"]
        txgb.write_feature_columns(p, feats)
        assert p.read_text() == "a\nb\nc\n"

    def test_write_threshold(self, tmp_path):
        p = tmp_path / "thr.txt"
        txgb.write_threshold(p, 0.577406)
        assert p.read_text() == "0.577406\n"

    def test_write_threshold_six_decimals(self, tmp_path):
        p = tmp_path / "thr.txt"
        txgb.write_threshold(p, 0.5)
        assert p.read_text() == "0.500000\n"


class TestTrainCombinedVariant:
    def _xy(self, seed=0, n=120):
        rng = np.random.default_rng(seed)
        feats = txgb.GEOM_PLUS_EMB
        X = rng.random((n, len(feats)))
        # make label loosely depend on first feature so XGB can learn something
        y = (X[:, 0] + rng.normal(0, 0.1, n) > 0.5).astype(int)
        return feats, X, y

    def test_metrics_schema_and_order(self):
        feats, X, y = self._xy()
        ntr = 90
        model, metrics = txgb.train_combined_variant(
            "variant", "geom_plus_cnn_emb", feats,
            X[:ntr], y[:ntr], X[ntr:], y[ntr:],
        )
        keys = list(metrics.keys())
        assert keys[0] == "variant"  # name key first (CSV column order)
        assert keys == [
            "variant", "n_features", "feature_columns", "n_train", "n_test",
            "n_train_pos", "n_train_neg", "scale_pos_weight", "test_pos_fraction",
            "optimal_threshold", "threshold_source", "roc_auc_test", "pr_auc_test",
            "precision_tuned", "recall_tuned", "f1_tuned", "accuracy_tuned",
            "precision_default_0.5", "recall_default_0.5", "f1_default_0.5",
            "accuracy_default_0.5", "pr_curve_tuned_precision", "pr_curve_tuned_recall",
        ]
        assert metrics["n_features"] == len(feats)
        assert metrics["feature_columns"] == ",".join(feats)
        assert metrics["n_train"] == ntr
        assert hasattr(model, "predict_proba")

    def test_regime_name_key(self):
        feats, X, y = self._xy(seed=1)
        ntr = 90
        _, metrics = txgb.train_combined_variant(
            "regime", "regA", feats, X[:ntr], y[:ntr], X[ntr:], y[ntr:],
        )
        assert list(metrics.keys())[0] == "regime"
        assert metrics["regime"] == "regA"


class TestStrictExtraction:
    def _rasters(self, tmp_path, n=3):
        rng = np.random.default_rng(0)
        paths = []
        for i in range(n):
            raster = rng.integers(0, 5, size=(64, 64), dtype=np.uint8)
            p = tmp_path / f"r_{i}.npy"
            np.save(p, raster)
            paths.append(p)
        return paths

    def test_emb_and_logit_shapes_finite(self, tmp_path):
        import torch

        from channel_heads.models.cnn import OutletCNN

        model = OutletCNN(embedding_dim=4)
        mp = tmp_path / "cnn.pt"
        torch.save(model.state_dict(), mp)

        paths = self._rasters(tmp_path, n=3)
        emb, logit = txgb.extract_emb_and_logit_strict(mp, paths, "cpu", embedding_dim=4)
        assert emb.shape == (3, 4)
        assert logit.shape == (3,)
        assert np.isfinite(emb).all() and np.isfinite(logit).all()

    def test_emb_only_shape(self, tmp_path):
        import torch

        from channel_heads.models.cnn import OutletCNN

        model = OutletCNN(embedding_dim=4)
        mp = tmp_path / "cnn.pt"
        torch.save(model.state_dict(), mp)
        paths = self._rasters(tmp_path, n=2)
        emb = txgb.extract_emb_strict(mp, paths, "cpu", embedding_dim=4)
        assert emb.shape == (2, 4)

    def test_strict_load_rejects_mismatch(self, tmp_path):
        import torch

        from channel_heads.models.cnn import OutletCNN

        mp = tmp_path / "cnn.pt"
        torch.save(OutletCNN(embedding_dim=8).state_dict(), mp)
        paths = self._rasters(tmp_path, n=1)
        with pytest.raises(RuntimeError):
            txgb.extract_emb_strict(mp, paths, "cpu", embedding_dim=4)
