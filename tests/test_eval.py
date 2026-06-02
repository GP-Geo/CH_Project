"""Tests for channel_heads.eval — thresholding, metrics, grouped splitting.

These primitives were extracted from ``scripts/retune_threshold_regime.py``
(and the regime training split). The tests pin the F1-argmax-excluding-last-PR-
point behaviour, the metric bundle, and the whole-outlet holdout contract.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from channel_heads.eval import (
    classification_metrics,
    f1_optimal_threshold,
    max_precision_threshold,
    outlet_group_holdout,
)


class TestThresholdAndMetrics:
    def test_f1_optimal_threshold_separable(self):
        # perfectly separable: threshold between the classes, F1 == 1
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        thr, f1 = f1_optimal_threshold(y, proba)
        assert f1 == pytest.approx(1.0)
        assert 0.2 < thr <= 0.8

    def test_classification_metrics_perfect(self):
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        m = classification_metrics(y, proba, threshold=0.5)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["roc_auc"] == pytest.approx(1.0)
        assert m["pr_auc"] == pytest.approx(1.0)

    def test_metrics_keys(self):
        y = np.array([0, 1, 0, 1])
        proba = np.array([0.3, 0.6, 0.4, 0.7])
        m = classification_metrics(y, proba, 0.5)
        assert set(m) == {"precision", "recall", "f1", "accuracy", "roc_auc", "pr_auc"}

    def test_max_precision_threshold_separable(self):
        y = np.array([0, 0, 1, 1])
        proba = np.array([0.1, 0.2, 0.8, 0.9])
        thr = max_precision_threshold(y, proba, min_recall=0.5)
        # any threshold in (0.2, 0.8] gives precision 1.0 at recall >= 0.5
        assert 0.2 < thr <= 0.8

    def test_max_precision_threshold_unreachable_recall_defaults_half(self):
        # min_recall above 1.0 can never be met -> fallback 0.5
        y = np.array([0, 1, 0, 1])
        proba = np.array([0.4, 0.6, 0.45, 0.7])
        assert max_precision_threshold(y, proba, min_recall=1.5) == 0.5


class TestOutletGroupHoldout:
    def _df(self):
        # 4 outlets x 5 pairs each; label varies
        rng = np.random.default_rng(0)
        rows = []
        for basin in ("b1", "b2"):
            for outlet in range(2):
                for _ in range(5):
                    rows.append({"basin": basin, "outlet": outlet, "y": int(rng.integers(0, 2))})
        return pd.DataFrame(rows)

    def test_disjoint_and_grouped(self):
        df = self._df()
        train_idx, test_idx = outlet_group_holdout(df)
        # no index overlap
        assert set(train_idx).isdisjoint(set(test_idx))
        # whole-outlet holdout: a group is never split across train/test
        grp = df["basin"].astype(str) + "__" + df["outlet"].astype(str)
        assert set(grp.iloc[train_idx]).isdisjoint(set(grp.iloc[test_idx]))

    def test_deterministic(self):
        df = self._df()
        a = outlet_group_holdout(df)
        b = outlet_group_holdout(df)
        np.testing.assert_array_equal(a[1], b[1])
